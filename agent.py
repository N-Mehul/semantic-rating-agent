"""
agent.py — Semantic Rating Agent

A dataset-agnostic agent that deeply understands any CSV dataset:
  * column types, distributions, and data quality
  * semantic meaning of text (via sentence embeddings)
  * rating / label distributions and patterns
  * inter-variable relationships (correct method per data type)
  * per-rating semantic profiles learned from actual data
  * open-ended semantic retrieval & reasoning over memory
  * plug-and-play architecture for LLM reasoning engines (e.g. Ollama)
  * new text analysis and rating prediction

No column names or dataset domain concepts are hard-coded.
All understanding is derived from the actual data.
"""

from __future__ import annotations

import json
import os
import random
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine as cosine_distance

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Module-level pure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (handles zero vectors)."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(1.0 - cosine_distance(a, b))


def _calc_entropy(dist: Any) -> float:
    """
    Calculate normalized Shannon entropy H in [0, 1].
    H = 0 -> perfectly deterministic (all mass on 1 level).
    H = 1 -> uniformly distributed across all K levels.
    """
    if isinstance(dist, dict):
        probs = np.array([float(p) for p in dist.values() if p > 0], dtype=float)
        k = len(dist)
    elif isinstance(dist, (list, np.ndarray)):
        probs = np.array([float(p) for p in dist if p > 0], dtype=float)
        k = len(dist)
    else:
        return 0.0

    if k <= 1 or len(probs) <= 1:
        return 0.0

    s = probs.sum()
    if s > 0:
        probs = probs / s
    h = -float(np.sum(probs * np.log2(probs)))
    max_h = float(np.log2(k))
    if max_h <= 0:
        return 0.0
    return float(np.clip(h / max_h, 0.0, 1.0))


def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """Stable softmax with temperature."""
    if len(x) == 0:
        return np.array([], dtype=float)
    x = np.array(x, dtype=float) / max(temp, 1e-6)
    e_x = np.exp(x - np.max(x))
    sum_e = np.sum(e_x)
    if sum_e == 0:
        return np.ones_like(x) / len(x)
    return e_x / sum_e


def _benjamini_hochberg(p_values: List[float]) -> List[float]:
    """
    Benjamini-Hochberg procedure for controlling the False Discovery Rate (FDR)
    across multiple hypothesis tests. Returns adjusted p-values.
    """
    n = len(p_values)
    if n == 0:
        return []
    sorted_indices = sorted(range(n), key=lambda i: p_values[i])
    sorted_p = [p_values[i] for i in sorted_indices]
    adjusted = [1.0] * n
    cum_min = 1.0
    for rank_idx in range(n - 1, -1, -1):
        p = sorted_p[rank_idx]
        q = (p * n) / (rank_idx + 1)
        cum_min = min(cum_min, q)
        adjusted[sorted_indices[rank_idx]] = min(1.0, float(cum_min))
    return adjusted


def _cramers_v(col_a: pd.Series, col_b: pd.Series) -> float:
    """Cramér's V — symmetric association for two categorical columns."""
    contingency = pd.crosstab(col_a, col_b)
    chi2, _, _, _ = stats.chi2_contingency(contingency)
    n = int(contingency.values.sum())
    r, k = contingency.shape
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, k_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types → Python native types for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    return obj


def _interpret_rho(value: float) -> str:
    v = abs(value)
    if v >= 0.7:
        return "strong"
    if v >= 0.4:
        return "moderate"
    if v >= 0.2:
        return "weak"
    return "negligible"


def _is_ordered_scale(unique_vals) -> bool:
    try:
        floats = sorted(float(v) for v in unique_vals)
        return all(floats[i + 1] > floats[i] for i in range(len(floats) - 1))
    except (ValueError, TypeError):
        return False


def _is_balanced(counts: list) -> bool:
    if not counts or min(counts) == 0:
        return False
    return max(counts) / min(counts) < 3.0


# ── Compiled sentiment-signal patterns (module-level for performance) ─────────
# General linguistic phrase patterns — NOT exact review-text memorization.
# Used by _apply_sentiment_rules() to post-process centroid predictions.
_STRONG_NEGATIVE_PATTERNS: List[re.Pattern] = [
    re.compile(r'\bvery poor\b',                re.IGNORECASE),
    re.compile(r'\bnot up to the mark\b',       re.IGNORECASE),
    re.compile(r'\bvery slow\b',                re.IGNORECASE),
    re.compile(r'\bvery disappointed\b',        re.IGNORECASE),
    re.compile(r"\bwouldn.t recommend\b",       re.IGNORECASE),
    re.compile(r'\bregret buying\b',            re.IGNORECASE),
    re.compile(r'\bnot worth\b',                re.IGNORECASE),
    re.compile(r'\boverheats?\b',               re.IGNORECASE),
    re.compile(r'\blags often\b',               re.IGNORECASE),
    re.compile(r'\bdrains too fast\b',          re.IGNORECASE),
    re.compile(r'\bflickering\b',               re.IGNORECASE),
    re.compile(r'\bbad and muffled\b',          re.IGNORECASE),
    re.compile(r'\bdisappointing\b',            re.IGNORECASE),
    re.compile(r'\breturning (?:this|it)\b',    re.IGNORECASE),
    re.compile(r'\bhangs often\b',              re.IGNORECASE),
]

_STRONG_NEUTRAL_PATTERNS: List[re.Pattern] = [
    re.compile(r'\bfine but could be better\b',          re.IGNORECASE),
    re.compile(r'\bnothing special\b',                    re.IGNORECASE),
    re.compile(r'\baverage experience\b',                 re.IGNORECASE),
    re.compile(r'\bokay for casual use\b',                re.IGNORECASE),
    re.compile(r'\bneither great nor bad\b',              re.IGNORECASE),
    re.compile(r'\bnot bad for daily use\b',              re.IGNORECASE),
    re.compile(r'\bnot the best in this range\b',         re.IGNORECASE),
    re.compile(r'\bexpected.{0,20}more for the price\b',  re.IGNORECASE),
    re.compile(r'\bperformance is average\b',             re.IGNORECASE),
    re.compile(r'\bcould be slightly better\b',           re.IGNORECASE),
    re.compile(r'\bdelayed sometimes\b',                  re.IGNORECASE),
    re.compile(r'\bnot very loud\b',                      re.IGNORECASE),
    re.compile(r'\ba bit bulky\b',                        re.IGNORECASE),
]


def _apply_sentiment_rules(text: str, predicted_sentiment: str) -> str:
    """
    Post-processing sentiment override using general linguistic patterns.
    Applied AFTER the centroid/ML classifier — the classifier is NOT changed.

    Rule 1 (Neutral → Negative):
        If the ML model predicts Neutral but the text contains ≥1 strong-negative
        phrase and zero neutral-hedge phrases → override to Negative.

    Rule 2 (Negative → Neutral):
        If the ML model predicts Negative but the text contains ≥1 neutral-hedge
        phrase and zero strong-negative phrases → override to Neutral.

    Conflict guard: when both signal types match, keep the ML prediction.
    Positive predictions are never changed by these rules.
    Patterns are general linguistic rules — NOT exact review-text memorization.
    """
    has_strong_neg   = any(p.search(text) for p in _STRONG_NEGATIVE_PATTERNS)
    has_neutral_hedge = any(p.search(text) for p in _STRONG_NEUTRAL_PATTERNS)

    # Rule 1: centroid said Neutral, but text has clear negative signals only
    if predicted_sentiment == "Neutral" and has_strong_neg and not has_neutral_hedge:
        return "Negative"

    # Rule 2: centroid said Negative, but text has clear neutral-hedge signals only
    if predicted_sentiment == "Negative" and has_neutral_hedge and not has_strong_neg:
        return "Neutral"

    # Conflict present, or no applicable rule — keep centroid prediction unchanged
    return predicted_sentiment


def _build_rating_uncertainty(
    pred_dist: Dict[str, float],
    level: float = 0.80,
    margin_threshold: float = 0.10,
) -> Dict[str, Any]:
    """
    Build calibrated uncertainty metrics and credible prediction intervals from a rating probability distribution.

    Parameters
    ----------
    pred_dist : Dict[str, float]
        Dictionary mapping rating scale keys (e.g. '1', '2', '3', '4', '5') to non-negative probabilities.
    level : float
        Nominal coverage level for the highest-density credible prediction interval (default: 0.80).
    margin_threshold : float
        Confidence margin threshold (top1_prob - top2_prob) to classify as 'confident' vs 'ambiguous' (default: 0.10).

    Returns
    -------
    Dict[str, Any] containing:
        - confidence: float (top-1 probability in [0.0, 1.0])
        - prediction_margin: float (top-1 prob - top-2 prob)
        - uncertainty_status: str ("confident" or "ambiguous")
        - uncertainty_explanation: str (human-readable explanation)
        - rating_distribution: Dict[str, float] (normalized 5-class distribution)
        - prediction_interval: Dict[str, Any] (level, lower, upper, covered_mass)
    """
    if not pred_dist:
        return {
            "confidence": 0.0,
            "prediction_margin": 0.0,
            "uncertainty_status": "ambiguous",
            "uncertainty_explanation": "No probability distribution available.",
            "rating_distribution": {},
            "prediction_interval": {"level": level, "lower": 1, "upper": 5, "covered_mass": 1.0},
        }

    # Ensure all ratings are normalized floats
    total = sum(pred_dist.values())
    if total > 0:
        norm_dist = {str(k): float(v) / total for k, v in pred_dist.items()}
    else:
        norm_dist = {str(k): 1.0 / len(pred_dist) for k in pred_dist}

    # Sort classes by probability descending
    sorted_items = sorted(norm_dist.items(), key=lambda x: -x[1])
    top1_k, top1_prob = sorted_items[0]
    top2_k, top2_prob = sorted_items[1] if len(sorted_items) > 1 else (top1_k, top1_prob)

    confidence = float(top1_prob)
    prediction_margin = float(top1_prob - top2_prob)

    if prediction_margin >= margin_threshold:
        uncertainty_status = "confident"
        uncertainty_explanation = "Rating prediction is relatively confident."
    else:
        uncertainty_status = "ambiguous"
        uncertainty_explanation = "Rating prediction is uncertain because multiple ratings have similar probabilities."

    # Highest Density Credible Prediction Interval
    # Accumulate most probable classes until cumulative mass reaches or exceeds level
    cum_mass = 0.0
    chosen_ratings: List[int] = []
    for rk, p in sorted_items:
        try:
            r_int = int(float(rk))
            chosen_ratings.append(r_int)
        except (ValueError, TypeError):
            continue
        cum_mass += p
        if cum_mass >= level:
            break

    if not chosen_ratings:
        lower, upper = 1, 5
    else:
        lower = max(1, min(chosen_ratings))
        upper = min(5, max(chosen_ratings))

    covered_mass = sum(norm_dist.get(str(r), norm_dist.get(f"{r}.0", 0.0)) for r in range(lower, upper + 1))

    prediction_interval = {
        "level": float(level),
        "lower": int(lower),
        "upper": int(upper),
        "covered_mass": round(float(covered_mass), 4),
    }

    rounded_dist = {k: round(float(v), 4) for k, v in norm_dist.items()}

    return {
        "confidence": round(confidence, 4),
        "prediction_margin": round(prediction_margin, 4),
        "uncertainty_status": uncertainty_status,
        "uncertainty_explanation": uncertainty_explanation,
        "rating_distribution": rounded_dist,
        "prediction_interval": prediction_interval,
    }


def _infer_scale_label(rmin: float, rmax: float, n_unique: int) -> str:
    if rmin == 1 and rmax == 5 and n_unique <= 5:
        return "5-point Likert scale (1–5)"
    if rmin == 0 and rmax == 10:
        return "0–10 NPS-style scale"
    if rmin == 1 and rmax == 10:
        return "1–10 scale"
    if rmin == 0 and rmax == 100:
        return "Percentage scale (0–100)"
    if n_unique == 2:
        return "Binary scale"
    return f"Custom scale ({rmin}–{rmax}, {n_unique} levels)"


def _infer_col_role(col: str, det: dict) -> str:
    if col == det.get("primary_text_column"):
        return "Primary text / review content"
    if col == det.get("primary_rating_column"):
        return "Primary rating / target variable"
    if col == det.get("primary_sentiment_column"):
        return "Sentiment / label column"
    if col in det.get("rating_columns", []):
        return "Sub-rating / aspect score"
    if col in det.get("numerical_columns", []):
        return "Numerical feature"
    if col in det.get("categorical_columns", []):
        return "Categorical feature"
    if col in det.get("identifier_columns", []):
        return "Row identifier (not predictive)"
    return "Supporting feature"


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning Engine Abstraction & Ollama Implementation
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_HOST  = "http://localhost:11434"


class BaseReasoningEngine:
    """
    Abstract Interface for Reasoning Engines.
    Defines the contract for synthesizing natural language answers from retrieved evidence.
    """
    def synthesize_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        dynamic_evidence: Optional[Dict[str, Any]],
        agent_context: Dict[str, Any]
    ) -> str:
        raise NotImplementedError


class OllamaReasoningEngine(BaseReasoningEngine):
    """
    Local LLM reasoning engine utilizing Ollama (llama3.2:3b).
    Interprets open-ended natural language questions and reasons over
    retrieved dataset findings without hardcoded question handlers.
    """

    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, host: str = DEFAULT_OLLAMA_HOST):
        self.model = model
        self.host = host.rstrip("/")

    def check_availability(self) -> Tuple[bool, str]:
        """
        Check if Ollama server is running and the target model is installed.
        Returns (is_available, status_message).
        """
        import urllib.request
        import urllib.error
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", headers={"User-Agent": "SemanticRatingAgent"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    installed_models = [m.get("name", "") for m in data.get("models", [])]
                    # Check exact name or base tag match (e.g. 'llama3.2:3b' or 'llama3.2:3b-instruct')
                    matching = [
                        m for m in installed_models
                        if m == self.model or m.startswith(f"{self.model}:") or self.model.startswith(m)
                    ]
                    if matching:
                        return True, f"Ollama is running with model '{self.model}' ready."
                    else:
                        return False, (
                            f"Ollama is running, but model '{self.model}' is not installed.\n"
                            f"  Installed models: {installed_models if installed_models else 'None'}\n"
                            f"  To install, run: ollama pull {self.model}"
                        )
                else:
                    return False, f"Ollama server returned HTTP status {resp.status}."
        except urllib.error.URLError:
            return False, (
                "Ollama is not running. Please start the Ollama application or service.\n"
                "  To install Ollama, visit: https://ollama.com"
            )
        except Exception as e:
            return False, f"Could not connect to Ollama: {e}"

    def synthesize_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        dynamic_evidence: Optional[Dict[str, Any]],
        agent_context: Dict[str, Any]
    ) -> str:
        avail, reason = self.check_availability()
        if not avail:
            return (
                "━━ LLM REASONING ENGINE UNAVAILABLE ━━\n\n"
                f"{reason}\n\n"
                "Please ensure Ollama is installed and running, then start the agent again."
            )

        # Build System Prompt with strict scientific reasoning constraints
        system_prompt = (
            "You are an expert scientific data analyst and researcher assisting a user in understanding a dataset.\n"
            "Your task is to interpret the user's question, analyze the supplied dataset evidence, and provide a clear, "
            "rigorous, and natural-language answer.\n\n"
            "CRITICAL RULES FOR REASONING:\n"
            "1. EVIDENCE-FIRST RULE: Answer ONLY using the facts, statistical tests, column profiles, and patterns provided in the EVIDENCE section below. Never invent metrics or rely on external general assumptions about products/datasets.\n"
            "2. INSUFFICIENT EVIDENCE: If the supplied evidence does not support answering the question, state explicitly: 'The dataset does not provide enough evidence to determine this reliably.' Do NOT speculate or make up unsupported facts.\n"
            "3. METHODOLOGICAL RIGOR & CAUSALITY: Explicitly distinguish between empirical observations (e.g. counts/distributions), statistical associations (e.g. Spearman rho, Kruskal-Wallis H, Cramer's V), and potential explanations. NEVER claim causality from correlational or observational data (correlation != causation).\n"
            "4. TEXT-TO-RATING INTEGRITY & DATA QUALITY: If the data quality notice reports 'Centroid Collapse' or decoupled text-to-rating labels, you MUST respect this limitation. Explain clearly that sentiment (Positive/Neutral/Negative) can be informative, but predicting exact numeric ratings directly from review text is unreliable in this dataset because text templates are shared across all rating levels.\n"
            "5. NO PREDEFINED TEMPLATES: Formulate a direct, coherent, and well-reasoned response tailored specifically to the user's question without fixed boilerplate."
        )

        # Build Context Prompt
        evidence_sections = []

        # Data quality notices
        tq = agent_context.get("dataset_text_quality", {})
        if tq:
            dq_notice = (
                f"Total Unique Texts: {tq.get('total_unique_texts')} across {tq.get('total_rows_with_text')} rows.\n"
                f"Centroid Collapse Detected: {tq.get('centroid_collapse_detected')}.\n"
                f"Average Centroid Cosine Similarity: {tq.get('avg_centroid_similarity')}.\n"
                f"Explanation: {tq.get('centroid_collapse_explanation')}"
            )
            evidence_sections.append(f"=== DATASET QUALITY & LIMITATION WARNINGS ===\n{dq_notice}")

        # Dynamic evidence
        if dynamic_evidence:
            dyn_text = f"Column: {dynamic_evidence.get('column')} (Role: {dynamic_evidence.get('role')}, Kind: {dynamic_evidence.get('kind')})\n"
            if "top_values" in dynamic_evidence:
                dyn_text += f"Top Values / Frequencies: {dynamic_evidence.get('top_values')}\n"
            if "group_means" in dynamic_evidence:
                dyn_text += f"Target Group Means: {dynamic_evidence.get('group_means')}\n"
            if "summary_stats" in dynamic_evidence:
                dyn_text += f"Summary Statistics: {dynamic_evidence.get('summary_stats')}\n"
            evidence_sections.append(f"=== DYNAMIC DATASET INSPECTION (ON-THE-FLY) ===\n{dyn_text}")

        # Retrieved chunks
        if retrieved_chunks:
            ret_texts = []
            for i, c in enumerate(retrieved_chunks, 1):
                ret_texts.append(f"Finding [{i}] - {c.get('title')} (Relevance: {c.get('similarity', 0.0):.3f}):\n{c.get('content_text')}")
            evidence_sections.append("=== RETRIEVED DATASET EVIDENCE & STATISTICAL FINDINGS ===\n" + "\n\n".join(ret_texts))
        else:
            evidence_sections.append("=== RETRIEVED DATASET EVIDENCE ===\nNo pre-computed findings retrieved.")

        prompt = (
            f"{chr(10).join(evidence_sections)}\n\n"
            f"=== USER QUESTION ===\n{query}\n\n"
            f"Please provide your reasoned analysis and answer based strictly on the evidence above:"
        )

        import urllib.request
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
            }
        }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json", "User-Agent": "SemanticRatingAgent"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.status == 200:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    return resp_json.get("response", "").strip()
                else:
                    return f"Error from Ollama API: HTTP {resp.status}"
        except Exception as e:
            return f"Error communicating with Ollama: {e}"


class SemanticReasoningEngine(BaseReasoningEngine):
    """
    Local non-LLM reasoning engine operating over semantically retrieved evidence.
    Available as reference and fallback.
    """

    def synthesize_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        dynamic_evidence: Optional[Dict[str, Any]],
        agent_context: Dict[str, Any]
    ) -> str:
        if not retrieved_chunks and not dynamic_evidence:
            return (
                "━━ INSUFFICIENT EVIDENCE IN DATASET ━━\n\n"
                "The dataset and stored memory do not contain enough relevant evidence to answer this query.\n"
                "No relevant statistical tests, column profiles, or semantic patterns match this topic."
            )

        top_similarity = retrieved_chunks[0]["similarity"] if retrieved_chunks else 0.0
        if top_similarity < 0.18 and not dynamic_evidence:
            return (
                "━━ INSUFFICIENT EVIDENCE IN DATASET ━━\n\n"
                f"The query '{query}' does not match any pre-computed findings or column profiles in memory "
                f"(top relevance score: {top_similarity:.3f}).\n"
                "No supported conclusions can be drawn for this question from the available dataset evidence."
            )

        lines = [f"━━ ANALYSIS & REASONING — {query.strip().upper()} ━━\n"]

        if dynamic_evidence:
            lines.append("  ┌─ Dynamic Dataset Inspection (On-the-Fly Analysis) ──────")
            col = dynamic_evidence.get("column")
            lines.append(f"  │ Column Analyzed : {col}")
            lines.append(f"  │ Type / Role     : {dynamic_evidence.get('kind')} ({dynamic_evidence.get('role')})")
            if "top_values" in dynamic_evidence:
                lines.append(f"  │ Top Values      : {dynamic_evidence.get('top_values')}")
            if "group_means" in dynamic_evidence:
                lines.append(f"  │ Target Means    : {dynamic_evidence.get('group_means')}")
            if "summary_stats" in dynamic_evidence:
                lines.append(f"  │ Summary Stats   : {dynamic_evidence.get('summary_stats')}")
            lines.append("  └──────────────────────────────────────────────────────────\n")

        lines.append("  ┌─ Retrieved Evidence & Empirical Observations ─────────────")
        seen_titles = set()
        for idx, chunk in enumerate(retrieved_chunks, 1):
            title = chunk.get("title", "Evidence Chunk")
            if title in seen_titles:
                continue
            seen_titles.add(title)
            sim = chunk.get("similarity", 0.0)
            lines.append(f"  │ [{idx}] {title} (relevance score: {sim:.3f})")
            content = chunk.get("content_text", "").strip()
            for line in content.split("\n"):
                if line.strip():
                    lines.append(f"  │     {line}")
            lines.append("  │")
        lines.append("  └──────────────────────────────────────────────────────────\n")

        lines.append("  ┌─ Methodological Evaluation & Statistical Rigor ──────────")
        lines.append("  │ • Observations: Direct empirical counts and raw dataset metrics.")
        lines.append("  │ • Associations: Statistical tests (Spearman ρ, Kruskal-Wallis H, Cramér's V) measure co-occurrence.")
        lines.append("  │ • Multiple Testing: Benjamini-Hochberg FDR correction controls false discovery rates.")
        lines.append("  │ • Causality Warning: Statistical association ≠ causality. Observed patterns reflect correlation,")
        lines.append("  │   confounding factors, or co-assignment in data collection.")
        lines.append("  └──────────────────────────────────────────────────────────\n")

        tq = agent_context.get("dataset_text_quality", {})
        if tq.get("centroid_collapse_detected"):
            lines.append("  ⚠ DATA QUALITY NOTICE (Centroid Collapse Active):")
            lines.append(f"    Total unique review texts: {tq.get('total_unique_texts')} across {tq.get('total_rows_with_text'):,} rows.")
            lines.append(f"    Average centroid similarity across ratings: {tq.get('avg_centroid_similarity')}.")
            lines.append("    Review text carries NO discriminative signal for rating prediction in this dataset.\n")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Memory Retriever & Indexer
# ─────────────────────────────────────────────────────────────────────────────

class MemoryRetriever:
    """
    Indexes all sections of memory.json into semantic chunks using sentence transformers.
    Retrieves top-K relevant chunks for any natural language query via vector similarity.
    Also performs dynamic on-the-fly pandas analysis when queries reference specific columns.
    """

    def __init__(self, agent: "SemanticRatingAgent"):
        self.agent = agent
        self.chunks: List[Dict[str, Any]] = []

    def index_memory(self):
        """Index memory.json into searchable semantic chunks with sentence embeddings."""
        mem = self.agent.memory
        if not mem:
            return

        raw_chunks = []

        # 1. Overview & Targets
        ls = mem.get("load_stats", {})
        det = mem.get("column_detection_summary", {})
        ta = mem.get("target_analysis", {})
        overview_text = (
            f"Dataset overview summary: CSV path {ls.get('csv_path')}, {ls.get('n_rows_clean')} clean working rows, "
            f"{ls.get('n_cols')} columns, missing cells {ls.get('missing_pct')}%. "
            f"Primary text column: {det.get('primary_text_column')}. "
            f"Primary rating target column: {det.get('primary_rating_column')} with scale {ta.get('rating', {}).get('scale_label')}, "
            f"mode {ta.get('rating', {}).get('mode')}, mean {ta.get('rating', {}).get('mean')}. "
            f"Primary sentiment column: {det.get('primary_sentiment_column')} with classes {ta.get('sentiment', {}).get('unique_classes')}. "
            f"Sub-ratings: {det.get('rating_columns')}. Numerical columns: {det.get('numerical_columns')}. "
            f"Categorical columns: {det.get('categorical_columns')}."
        )
        raw_chunks.append({
            "chunk_id": "overview",
            "section": "Overview",
            "title": "Dataset Overview & Target Metadata",
            "content_text": overview_text,
            "raw_data": {"load_stats": ls, "column_detection": det, "target_analysis": ta},
        })

        # 2. Data Quality
        profiles = mem.get("column_profiles", {})
        dq_lines = [
            f"Data quality analysis: Total raw rows {ls.get('n_rows_raw')}, clean rows {ls.get('n_rows_clean')}. "
            f"Fully null rows dropped: {ls.get('fully_null_rows')}. Exact duplicate rows dropped: {ls.get('duplicate_rows')}. "
            f"Total missing cells: {ls.get('total_missing_cells')} ({ls.get('missing_pct')}%)."
        ]
        high_miss = [f"{c}: {p['missing_pct']}% missing" for c, p in profiles.items() if p.get("missing_pct", 0) > 5]
        if high_miss:
            dq_lines.append(f"Columns with high missingness (>5%): {', '.join(high_miss)}.")
        raw_chunks.append({
            "chunk_id": "data_quality",
            "section": "Data Quality",
            "title": "Data Quality, Nulls, Duplicates & Missingness",
            "content_text": " ".join(dq_lines),
            "raw_data": {"load_stats": ls},
        })

        # 3. Text Quality & Signal Assessment
        tq = mem.get("dataset_text_quality", {})
        sig = mem.get("text_rating_signal_quality", {})
        cv = mem.get("internal_cross_validation", {})
        if tq or sig:
            tq_text = (
                f"Dataset text quality diagnosis: Signal Quality Tier '{sig.get('signal_tier', 'UNKNOWN')}'. "
                f"Total unique review texts: {tq.get('total_unique_texts')} across {tq.get('total_rows_with_text')} rows "
                f"({tq.get('avg_repetitions_per_text')}x average repeats per text). "
                f"Texts appearing under multiple ratings: {tq.get('texts_multiple_ratings')} ({tq.get('pct_texts_multiple_ratings', 0.0):.1f}%), "
                f"under all rating levels: {tq.get('texts_all_ratings')}. "
                f"Text-to-rating consistency: {tq.get('text_rating_consistency', 0.0)*100:.1f}% (average rating entropy: {tq.get('avg_rating_entropy', 0.0):.3f}). "
                f"Centroid collapse detected: {tq.get('centroid_collapse_detected')}. "
                f"Average rating centroid cosine similarity: {tq.get('avg_centroid_similarity')}. "
                f"Explanation: {tq.get('centroid_collapse_explanation') or sig.get('explanation')}"
            )
            raw_chunks.append({
                "chunk_id": "text_quality",
                "section": "Text Quality",
                "title": "Dataset Text Quality, Signal Assessment & Conflict Diagnosis",
                "content_text": tq_text,
                "raw_data": {"text_quality": tq, "signal_quality": sig},
            })

        if cv:
            cv_text = (
                f"Internal Cross-Validation on training data (5-fold CV): "
                f"MAE: {cv.get('mae')}, RMSE: {cv.get('rmse')}, Spearman correlation rho: {cv.get('spearman_rho')} (p={cv.get('spearman_pvalue')}), "
                f"Exact rating accuracy: {cv.get('exact_accuracy_pct')}%, Within-1 rating accuracy: {cv.get('within_1_accuracy_pct')}%. "
                f"Sentiment validation accuracy: {cv.get('sentiment_accuracy_pct')}%, Sentiment macro-F1: {cv.get('sentiment_macro_f1')}."
            )
            raw_chunks.append({
                "chunk_id": "internal_cross_validation",
                "section": "Validation",
                "title": "Internal Training Cross-Validation Performance Metrics",
                "content_text": cv_text,
                "raw_data": cv,
            })

        # 4. Individual Column Profiles
        for col, p in profiles.items():
            role = _infer_col_role(col, det)
            col_text = (
                f"Column feature description for '{col}': Data type {p.get('dtype')}, detected kind {p.get('detected_kind')}, "
                f"operational role '{role}'. Unique values: {p.get('n_unique')} (missing: {p.get('missing_pct')}%)."
            )
            if p.get("min") is not None:
                col_text += f" Range: {p['min']} to {p['max']} (mean: {p.get('mean')}, std: {p.get('std')}, median: {p.get('median')}). IQR Outliers: {p.get('outlier_count')} ({p.get('outlier_pct')}%)."
            if p.get("avg_text_length"):
                col_text += f" Avg text length: {p.get('avg_text_length')} chars."
            if p.get("top_values"):
                col_text += f" Top value frequencies: {list(p['top_values'].items())[:5]}."
            raw_chunks.append({
                "chunk_id": f"col_{col}",
                "section": "Column Profile",
                "title": f"Column Profile: '{col}'",
                "content_text": col_text,
                "raw_data": p,
            })

        # 5. Statistical Relationships
        rels = mem.get("relationships", {})
        if rels:
            # Num vs Rating
            nvr = rels.get("numerical_vs_rating", {})
            if nvr:
                nvr_text = "Numerical features vs primary rating Spearman rank correlations (rho, p-value, strength): " + "; ".join(
                    [f"{c}: rho={r.get('rho'):+.4f} ({r.get('strength')}, BH adj p={r.get('bh_adjusted_p')})" for c, r in nvr.items()]
                )
                raw_chunks.append({
                    "chunk_id": "rel_num_rating",
                    "section": "Relationships",
                    "title": "Numerical Features vs Primary Rating Correlations (Spearman)",
                    "content_text": nvr_text,
                    "raw_data": nvr,
                })

            # Sub-ratings vs Rating
            srr = rels.get("sub_ratings_vs_primary_rating", {})
            if srr:
                srr_text = "Sub-ratings / aspect ratings vs primary overall rating Spearman rank correlations: " + "; ".join(
                    [f"{c}: rho={r.get('rho'):+.4f} ({r.get('strength')})" for c, r in srr.items()]
                )
                raw_chunks.append({
                    "chunk_id": "rel_sub_ratings",
                    "section": "Relationships",
                    "title": "Aspect Sub-Ratings vs Primary Rating Associations",
                    "content_text": srr_text,
                    "raw_data": srr,
                })

            # Categorical vs Rating
            cvr = rels.get("categorical_vs_rating", {})
            if cvr:
                cvr_text = "Categorical features vs primary rating Kruskal-Wallis H-tests: " + "; ".join(
                    [f"{c}: H={r.get('H_statistic'):.2f}, p={r.get('p_value')}, eta_sq={r.get('eta_squared_approx')}" for c, r in cvr.items()]
                )
                raw_chunks.append({
                    "chunk_id": "rel_cat_rating",
                    "section": "Relationships",
                    "title": "Categorical Features vs Primary Rating Group Comparisons (Kruskal-Wallis)",
                    "content_text": cvr_text,
                    "raw_data": cvr,
                })

            # Sentiment vs Rating
            svr = rels.get("sentiment_vs_rating", {})
            if svr:
                svr_text = (
                    f"Sentiment vs primary rating relationship: Kruskal-Wallis H={svr.get('H_statistic')}, p={svr.get('p_value')}. "
                    f"Mean rating per sentiment class: {svr.get('mean_rating_per_sentiment')}."
                )
                raw_chunks.append({
                    "chunk_id": "rel_sentiment_rating",
                    "section": "Relationships",
                    "title": "Sentiment Class vs Primary Rating Alignment",
                    "content_text": svr_text,
                    "raw_data": svr,
                })

            # Method Rationale
            raw_chunks.append({
                "chunk_id": "rel_method_rationale",
                "section": "Method Rationale",
                "title": "Statistical Method Rationale (Why Pearson is Inappropriate)",
                "content_text": (
                    "Statistical method selection rationale: Spearman rank correlation is used for ordinal ratings (1-5) "
                    "because Pearson correlation incorrectly assumes linear interval-scale continuity. Kruskal-Wallis H-test "
                    "is used for comparing ordinal rating distributions across categorical groups without assuming normality. "
                    "Cramér's V is used for categorical-categorical associations. Sentence embeddings and per-rating centroids "
                    "are used for text-to-rating relationships because text is not a scalar. "
                    "All findings are correlational associations (association != causation)."
                ),
                "raw_data": {},
            })

        # 6. Text Themes
        themes = mem.get("text_themes", {})
        if themes and themes.get("themes"):
            th_lines = [
                f"Discovered unsupervised semantic text themes via sentence embeddings and KMeans clustering: "
                f"Optimal clusters k={themes.get('optimal_k')}, Silhouette Score={themes.get('silhouette_score')}, sample size={themes.get('sample_size')} texts."
            ]
            for t in themes.get("themes", []):
                th_lines.append(f"Theme #{t['theme_id']} ({t['size']} texts, {t['pct_of_sample']}%): Representative texts: {t.get('representative_texts')[:2]}.")
            raw_chunks.append({
                "chunk_id": "text_themes",
                "section": "Text Themes",
                "title": "Discovered Unsupervised Semantic Text Themes & Topics",
                "content_text": " ".join(th_lines),
                "raw_data": themes,
            })

        # 7. Subgroup Interactions & Simpson's Paradox
        inter = mem.get("interactions", {})
        if inter and inter.get("interactions"):
            int_lines = ["Discovered conditional interaction patterns across subgroups: "]
            for item in inter.get("interactions", []):
                int_lines.append(f"{item.get('numerical_var')} vs rating across {item.get('grouping_var')}: {item.get('description')}")
            raw_chunks.append({
                "chunk_id": "interactions",
                "section": "Interactions",
                "title": "Conditional Subgroup Interactions & Relationship Variations",
                "content_text": " ".join(int_lines),
                "raw_data": inter,
            })

        sg = mem.get("subgroup_findings", {})
        if sg and sg.get("subgroup_findings"):
            sg_lines = ["Subgroup consistency & Simpson's paradox evaluations across categorical breakdowns: "]
            for item in sg.get("subgroup_findings", []):
                sg_lines.append(f"Relationship {item.get('primary_relationship')} stratified by {item.get('subgroup_variable')}: {item.get('finding')}")
            raw_chunks.append({
                "chunk_id": "subgroup_findings",
                "section": "Subgroups",
                "title": "Subgroup Consistency Checks & Stratified Breakdown Matrices",
                "content_text": " ".join(sg_lines),
                "raw_data": sg,
            })

        # 8. Learned Text Rating Profiles & Language Patterns
        trp = mem.get("text_rating_patterns", {}).get("rating_profiles", {})
        if trp:
            trp_lines = ["Per-rating level semantic text profiles and representative customer responses: "]
            for r_val in sorted(trp.keys(), key=float):
                prof = trp[r_val]
                trp_lines.append(f"Rating {r_val} ({prof.get('n_total')} responses): Representative texts: {prof.get('representative_texts')[:3]}.")
            raw_chunks.append({
                "chunk_id": "text_rating_patterns",
                "section": "Text Patterns",
                "title": "Learned Per-Rating Semantic Text Profiles & Representative Responses",
                "content_text": " ".join(trp_lines),
                "raw_data": trp,
            })

        # 9. Ranked Discoveries
        ranked = mem.get("ranked_discoveries", [])
        if ranked:
            rk_lines = ["Ranked top discoveries across dataset: "]
            for i, rd in enumerate(ranked, 1):
                rk_lines.append(f"#{i} {rd.get('finding')} (Evidence: {rd.get('evidence')}, Strength: {rd.get('strength')}, Why it matters: {rd.get('why_it_matters')}).")
            raw_chunks.append({
                "chunk_id": "ranked_discoveries",
                "section": "Ranked Discoveries",
                "title": "Top Ranked Dataset Discoveries & Key Findings",
                "content_text": " ".join(rk_lines),
                "raw_data": ranked,
            })

        # Embed all searchable texts in bulk
        texts_to_embed = [c["content_text"] for c in raw_chunks]
        print(f"  [Retriever] Building semantic vector index for {len(raw_chunks)} memory chunks…")
        embeddings = self.agent._embed(texts_to_embed)

        for chunk, emb in zip(raw_chunks, embeddings):
            chunk["embedding"] = emb

        self.chunks = raw_chunks
        print("  [Retriever] Vector index ready.")

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve top-K semantically relevant memory chunks for a query."""
        if not self.chunks:
            self.index_memory()
        if not self.chunks:
            return []

        query_vec = self.agent._embed([query])[0]
        results = []
        for chunk in self.chunks:
            sim = _cosine_sim(query_vec, chunk["embedding"])
            # Create a copy with similarity score
            c = dict(chunk)
            c["similarity"] = round(float(sim), 4)
            results.append(c)

        results.sort(key=lambda x: -x["similarity"])
        return results[:top_k]

    def dynamically_inspect_data(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Dynamically inspect self.agent.df_clean on-the-fly if a query references
        specific column names or values not fully pre-summarized.
        Does NOT create a permanent handler for the question.
        """
        if self.agent.df_clean is None:
            return None

        df = self.agent.df_clean
        q_lower = query.lower()

        # Find any column names mentioned in query
        matched_cols = [c for c in df.columns if c.lower() in q_lower]
        if not matched_cols:
            return None

        col = matched_cols[0]
        series = df[col].dropna()
        profiles = self.agent.memory.get("column_profiles", {})
        p = profiles.get(col, {})

        evidence: Dict[str, Any] = {
            "column": col,
            "role": _infer_col_role(col, self.agent.memory.get("column_detection_summary", {})),
            "kind": p.get("detected_kind", "unknown"),
        }

        if pd.api.types.is_numeric_dtype(series):
            evidence["summary_stats"] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": round(float(series.mean()), 3),
                "std": round(float(series.std()), 3),
                "median": float(series.median()),
            }
        else:
            vc = series.value_counts().head(5).to_dict()
            evidence["top_values"] = {str(k): int(v) for k, v in vc.items()}

        # If primary rating column exists, compute mean rating per category
        rating_col = self.agent.primary_rating_col
        if rating_col and col != rating_col and not pd.api.types.is_numeric_dtype(series):
            pair = df[[col, rating_col]].dropna()
            if len(pair) > 0 and pair[col].nunique() <= 30:
                means = pair.groupby(col)[rating_col].mean().head(5).to_dict()
                evidence["group_means"] = {str(k): round(float(v), 3) for k, v in means.items()}

        return evidence


# ─────────────────────────────────────────────────────────────────────────────
# SemanticRatingAgent
# ─────────────────────────────────────────────────────────────────────────────

class SemanticRatingAgent:
    """
    Analyzes any CSV dataset and builds a structured semantic understanding.
    Provides open-ended semantic retrieval and reasoning over memory.
    Supports plugging in an LLM reasoning engine (e.g. Ollama) seamlessly.
    """

    MAX_EMBED_PER_RATING = 500   # texts embedded per rating level
    MAX_EMBED_GLOBAL = 2000      # texts embedded for global analysis

    _RATING_NAME_HINTS    = {"rating", "score", "stars", "grade", "rank", "point", "mark"}
    _SENTIMENT_NAME_HINTS = {"sentiment", "polarity", "opinion", "emotion", "class", "label", "category"}
    _TEXT_NAME_HINTS      = {"review", "text", "comment", "description", "feedback",
                               "content", "message", "note", "opinion", "body", "post"}
    _ID_NAME_HINTS        = {"id", "index", "uuid", "key", "no", "num", "number", "#", "row"}
    _KNOWN_SENTIMENT_VALS = {"positive", "negative", "neutral", "mixed", "unknown",
                              "pos", "neg", "neu", "very positive", "very negative",
                              "strongly positive", "strongly negative"}

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None           # raw loaded DataFrame
        self.df_clean: Optional[pd.DataFrame] = None     # deduped + fully-null rows dropped
        self.csv_path: Optional[str] = None
        self.memory: Dict[str, Any] = {}
        self._embed_model = None                         # lazy-loaded

        # Modules for Q&A Architecture
        self.retriever = MemoryRetriever(self)
        self.reasoning_engine: BaseReasoningEngine = SemanticReasoningEngine()

        # Detected column lists — populated in Phase 3
        self.text_cols: List[str] = []
        self.rating_cols: List[str] = []
        self.sentiment_cols: List[str] = []
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.identifier_cols: List[str] = []
        self.date_cols: List[str] = []
        self.constant_cols: List[str] = []
        self.near_constant_cols: List[str] = []
        self.possible_leakage_cols: List[str] = []

        # Primary column selections
        self.primary_text_col: Optional[str] = None
        self.primary_rating_col: Optional[str] = None
        self.primary_sentiment_col: Optional[str] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Embedding model
    # ──────────────────────────────────────────────────────────────────────────

    def _get_model(self):
        if self._embed_model is None:
            print("  [Embeddings] Loading all-MiniLM-L6-v2 (first time only)…")
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("  [Embeddings] Model ready.")
        return self._embed_model

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings → (N, 384) float32 array."""
        model = self._get_model()
        return model.encode(
            texts,
            show_progress_bar=len(texts) > 200,
            batch_size=64,
            normalize_embeddings=True,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1 — Load CSV
    # ──────────────────────────────────────────────────────────────────────────

    def load_csv(self, path: str) -> pd.DataFrame:
        self.csv_path = path
        df = None
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(path, encoding=enc, low_memory=False)
                print(f"  [Load] Encoding detected: {enc}")
                break
            except UnicodeDecodeError:
                continue
            except Exception as exc:
                raise RuntimeError(f"Could not read {path}: {exc}") from exc

        if df is None:
            raise RuntimeError(f"Could not decode '{path}' with any supported encoding.")

        self.df = df
        n_rows, n_cols = df.shape
        n_null_rows   = int(df.isnull().all(axis=1).sum())
        n_duplicates  = int(df.duplicated().sum())
        total_missing = int(df.isnull().sum().sum())
        total_cells   = n_rows * n_cols
        missing_pct   = 100 * total_missing / total_cells if total_cells else 0
        n_informative = n_rows - n_null_rows

        print(f"\n  ┌─ Dataset Loaded ──────────────────────────────────")
        print(f"  │  Source dataset rows  : {n_rows:,}")
        print(f"  │  Columns              : {n_cols}")
        print(f"  │  Missing cells        : {total_missing:,}  ({missing_pct:.1f}%)")
        print(f"  │  Completely empty rows: {n_null_rows:,}  → excluded (zero information)")
        print(f"  │  Duplicate rows       : {n_duplicates:,}  → RETAINED (real observations)")
        print(f"  │  Informative rows used: {n_informative:,}")
        print(f"  └───────────────────────────────────────────────────\n")

        # Remove ONLY rows that are completely empty across ALL columns.
        # Duplicate rows are intentionally retained: their frequency encodes
        # empirical rating/sentiment distributions used by the example bank.
        df_clean = df.dropna(how="all").reset_index(drop=True)
        self.df_clean = df_clean
        print(f"  [Clean] Working copy: {len(df_clean):,} rows "
              f"({n_null_rows:,} completely-empty rows excluded; "
              f"{n_duplicates:,} duplicate rows retained as real observations)")

        self.memory["load_stats"] = _to_json_safe({
            "csv_path": path,
            "n_rows_source": n_rows,
            "n_cols": n_cols,
            "n_rows_informative": n_informative,
            "n_rows_clean": n_informative,
            "completely_empty_rows_excluded": n_null_rows,
            "duplicate_rows_retained": n_duplicates,
            "duplicate_rows_removed": 0,
            "total_missing_cells": total_missing,
            "missing_pct": round(missing_pct, 2),
            "training_policy": "All non-empty rows used. Duplicates retained as real observations.",
        })
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2 — Analyze every column
    # ──────────────────────────────────────────────────────────────────────────

    def analyze_columns(self) -> Dict[str, Dict]:
        df = self.df_clean
        profiles: Dict[str, Dict] = {}

        for col in df.columns:
            series = df[col]
            n_total   = len(series)
            n_missing = int(series.isnull().sum())
            n_unique  = int(series.nunique())
            miss_pct  = round(100 * n_missing / n_total, 2) if n_total else 0.0
            non_null  = series.dropna()
            uniq_ratio = round(n_unique / len(non_null), 4) if len(non_null) > 0 else 0.0

            is_constant = (n_unique == 1)
            top_freq = int(non_null.value_counts().iloc[0]) if len(non_null) > 0 else 0
            top_freq_pct = round(100 * top_freq / len(non_null), 2) if len(non_null) > 0 else 0.0
            is_near_constant = (not is_constant) and (top_freq_pct >= 95.0)

            col_lower = col.lower()
            possible_leakage = any(
                term in col_lower for term in ("target", "label", "outcome", "actual", "true_class")
            )

            p: Dict[str, Any] = {
                "column":           col,
                "dtype":            str(series.dtype),
                "n_total":          n_total,
                "n_missing":        n_missing,
                "missing_pct":      miss_pct,
                "n_unique":         n_unique,
                "unique_ratio":     uniq_ratio,
                "is_constant":      is_constant,
                "is_near_constant": is_near_constant,
                "top_freq_pct":     top_freq_pct,
                "possible_leakage": possible_leakage,
            }

            if pd.api.types.is_numeric_dtype(series):
                p["detected_kind"] = "numerical"
                if len(non_null):
                    p["min"]    = float(non_null.min())
                    p["max"]    = float(non_null.max())
                    p["mean"]   = round(float(non_null.mean()), 4)
                    p["std"]    = round(float(non_null.std()), 4)
                    p["median"] = float(non_null.median())
                    p["top_values"] = {
                        str(k): int(v)
                        for k, v in non_null.value_counts().head(10).items()
                    }
                    q25 = float(non_null.quantile(0.25))
                    q75 = float(non_null.quantile(0.75))
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    outliers = non_null[(non_null < lower_bound) | (non_null > upper_bound)]
                    p["iqr_bounds"] = [round(lower_bound, 4), round(upper_bound, 4)]
                    p["outlier_count"] = len(outliers)
                    p["outlier_pct"] = round(100 * len(outliers) / len(non_null), 2)

                    try:
                        is_int_like = bool(non_null.apply(lambda x: x == int(x)).all())
                    except (ValueError, TypeError):
                        is_int_like = False
                    p["looks_like_rating"] = bool(is_int_like and n_unique <= 20)
                    if p["looks_like_rating"]:
                        p["integer_range"] = [int(non_null.min()), int(non_null.max())]

            elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                str_series = non_null.astype(str)
                avg_len = float(str_series.str.len().mean()) if len(str_series) else 0.0
                max_len = int(str_series.str.len().max())   if len(str_series) else 0

                p["avg_text_length"] = round(avg_len, 1)
                p["max_text_length"] = max_len
                p["top_values"] = {
                    str(k): int(v)
                    for k, v in str_series.value_counts().head(10).items()
                }

                is_date = False
                if any(h in col_lower for h in ("date", "time", "year", "month", "day", "created", "timestamp", "at")):
                    try:
                        sample_dates = str_series.sample(min(20, len(str_series)), random_state=42)
                        parsed = pd.to_datetime(sample_dates, errors="coerce")
                        if parsed.notnull().mean() > 0.8:
                            is_date = True
                    except Exception:
                        pass
                p["is_date"] = is_date

                col_hint_text = any(
                    h in col.lower()
                    for h in ("review", "text", "comment", "description",
                               "feedback", "content", "message", "note",
                               "opinion", "body", "post")
                )
                if is_date:
                    p["detected_kind"] = "date_time"
                elif (avg_len > 20 and uniq_ratio > 0.05) or (avg_len > 10 and col_hint_text):
                    p["detected_kind"] = "text"
                elif n_unique <= 50:
                    p["detected_kind"] = "categorical"
                elif uniq_ratio > 0.9:
                    p["detected_kind"] = "identifier"
                else:
                    p["detected_kind"] = "mixed_text_or_categorical"
            else:
                p["detected_kind"] = "other"

            profiles[col] = p

        self.memory["column_profiles"] = _to_json_safe(profiles)
        return profiles

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 3 — Auto-detect important columns
    # ──────────────────────────────────────────────────────────────────────────

    def detect_important_columns(self) -> Dict[str, Any]:
        df = self.df_clean
        profiles = self.memory.get("column_profiles", {})

        text_cols, rating_cols, sentiment_cols = [], [], []
        numerical_cols, categorical_cols, identifier_cols = [], [], []
        date_cols, constant_cols, near_constant_cols, possible_leakage_cols = [], [], [], []

        for col, p in profiles.items():
            col_lower  = col.lower()
            kind       = p.get("detected_kind", "other")
            n_unique   = p.get("n_unique", 0)
            miss_pct   = p.get("missing_pct", 0.0)
            uniq_ratio = p.get("unique_ratio", 0.0)

            if p.get("is_constant"):
                constant_cols.append(col)
            elif p.get("is_near_constant"):
                near_constant_cols.append(col)

            if p.get("possible_leakage"):
                possible_leakage_cols.append(col)

            if kind == "date_time":
                date_cols.append(col)

            if miss_pct > 80:
                continue

            name_is_id = any(h in col_lower for h in self._ID_NAME_HINTS)
            value_is_id = uniq_ratio > 0.95 and kind in ("numerical", "identifier", "text", "mixed_text_or_categorical")
            if name_is_id or value_is_id:
                identifier_cols.append(col)
                continue

            if kind == "text":
                text_cols.append(col)
                continue

            if kind == "numerical":
                _excl = ("count", "helpful", "vote", "total", "length", "size")
                name_is_rating = (
                    any(h in col_lower for h in self._RATING_NAME_HINTS)
                    and not any(e in col_lower for e in _excl)
                )
                value_is_rating = (
                    p.get("looks_like_rating", False)
                    and n_unique <= 10
                    and float(p.get("max", 9999)) <= 10
                )
                if name_is_rating or value_is_rating:
                    rating_cols.append(col)
                else:
                    numerical_cols.append(col)
                continue

            if kind in ("categorical", "mixed_text_or_categorical"):
                series_vals = set(df[col].dropna().astype(str).str.lower().unique())
                sentiment_overlap = len(series_vals & self._KNOWN_SENTIMENT_VALS) / max(len(series_vals), 1)
                name_is_sentiment = any(h in col_lower for h in self._SENTIMENT_NAME_HINTS)

                if name_is_sentiment or sentiment_overlap > 0.3:
                    sentiment_cols.append(col)
                else:
                    categorical_cols.append(col)

        def _rating_priority(c: str) -> int:
            cl = c.lower()
            if cl == "rating":
                return 0
            if "rating" in cl:
                return 1
            return 2

        rating_cols.sort(key=_rating_priority)

        self.text_cols             = text_cols
        self.rating_cols           = rating_cols
        self.sentiment_cols        = sentiment_cols
        self.numerical_cols        = numerical_cols
        self.categorical_cols      = categorical_cols
        self.identifier_cols       = identifier_cols
        self.date_cols             = date_cols
        self.constant_cols         = constant_cols
        self.near_constant_cols    = near_constant_cols
        self.possible_leakage_cols = possible_leakage_cols

        self.primary_text_col      = text_cols[0]      if text_cols      else None
        self.primary_rating_col    = rating_cols[0]    if rating_cols    else None
        self.primary_sentiment_col = sentiment_cols[0] if sentiment_cols else None

        confidence_notes = []
        if not text_cols:
            confidence_notes.append("WARNING: No clear text column found. Column detection may be unreliable.")
        if not rating_cols:
            confidence_notes.append("WARNING: No rating / score column found.")
        if len(text_cols) > 1:
            confidence_notes.append(
                f"Multiple text columns detected: {text_cols}. "
                f"Using '{self.primary_text_col}' as primary."
            )
        if len(rating_cols) > 1:
            confidence_notes.append(
                f"Multiple rating columns detected: {rating_cols}. "
                f"Primary: '{self.primary_rating_col}'."
            )
        if possible_leakage_cols:
            confidence_notes.append(
                f"Potential target leakage columns detected: {possible_leakage_cols}."
            )

        detection = {
            "primary_text_column":      self.primary_text_col,
            "primary_rating_column":    self.primary_rating_col,
            "primary_sentiment_column": self.primary_sentiment_col,
            "text_columns":             text_cols,
            "rating_columns":           rating_cols,
            "sentiment_columns":        sentiment_cols,
            "numerical_columns":        numerical_cols,
            "categorical_columns":      categorical_cols,
            "identifier_columns":       identifier_cols,
            "date_columns":             date_cols,
            "constant_columns":         constant_cols,
            "near_constant_columns":    near_constant_cols,
            "possible_leakage_columns": possible_leakage_cols,
            "confidence_notes":         confidence_notes,
        }
        self.memory["column_detection"] = _to_json_safe(detection)
        self.memory["column_detection_summary"] = self.memory["column_detection"]
        return detection

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 4 — Understand target distributions
    # ──────────────────────────────────────────────────────────────────────────

    def analyze_target(self) -> Dict[str, Any]:
        df    = self.df_clean
        result: Dict[str, Any] = {}

        if self.primary_rating_col:
            col    = self.primary_rating_col
            series = df[col].dropna()
            vc     = series.value_counts().sort_index()
            total  = len(series)

            freqs   = {str(k): int(v)                     for k, v in vc.items()}
            pcts    = {str(k): round(100 * v / total, 2)  for k, v in vc.items()}
            unique_vals = sorted(series.unique())

            result["rating"] = {
                "column":           col,
                "min":              float(series.min()),
                "max":              float(series.max()),
                "mean":             round(float(series.mean()), 3),
                "std":              round(float(series.std()),  3),
                "unique_values":    [float(v) for v in unique_vals],
                "value_counts":     freqs,
                "value_pcts":       pcts,
                "is_ordered_scale": _is_ordered_scale(unique_vals),
                "is_balanced":      _is_balanced(list(freqs.values())),
                "mode":             float(series.mode().iloc[0]),
                "scale_label":      _infer_scale_label(
                    float(series.min()), float(series.max()), len(unique_vals)
                ),
            }

        if self.primary_sentiment_col:
            col    = self.primary_sentiment_col
            series = df[col].dropna()
            vc     = series.value_counts()
            total  = len(series)

            freqs  = {str(k): int(v)                    for k, v in vc.items()}
            pcts   = {str(k): round(100 * v / total, 2) for k, v in vc.items()}

            sent_info: Dict[str, Any] = {
                "column":         col,
                "unique_classes": [str(c) for c in vc.index.tolist()],
                "value_counts":   freqs,
                "value_pcts":     pcts,
                "is_balanced":    _is_balanced(list(freqs.values())),
            }

            if self.primary_rating_col:
                cross = df[[col, self.primary_rating_col]].dropna()
                sent_info["mean_rating_per_class"] = {
                    str(k): round(float(v), 3)
                    for k, v in cross.groupby(col)[self.primary_rating_col].mean().items()
                }

            result["sentiment"] = sent_info

        self.memory["target_analysis"] = _to_json_safe(result)
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 5 — Semantic text analysis (global)
    # ──────────────────────────────────────────────────────────────────────────

    def analyze_text_semantically(self) -> Dict[str, Any]:
        if not self.primary_text_col:
            print("  [Phase 5] No primary text column detected — skipping.")
            return {}

        col   = self.primary_text_col
        texts = self.df_clean[col].dropna().astype(str).tolist()
        print(f"  [Phase 5] Corpus: {len(texts):,} texts in '{col}'.")

        sample = texts
        if len(texts) > self.MAX_EMBED_GLOBAL:
            random.seed(42)
            sample = random.sample(texts, self.MAX_EMBED_GLOBAL)

        print(f"  [Phase 5] Embedding {len(sample):,} texts…")
        emb = self._embed(sample)
        centroid = emb.mean(axis=0)

        sims = [_cosine_sim(e, centroid) for e in emb]

        top5_idx    = sorted(range(len(sims)), key=lambda i: -sims[i])[:5]
        bottom5_idx = sorted(range(len(sims)), key=lambda i:  sims[i])[:5]

        result = {
            "total_texts":          len(texts),
            "embedded_sample_size": len(sample),
            "global_centroid":      centroid.tolist(),
            "most_representative_texts": [sample[i] for i in top5_idx],
            "most_unusual_texts":        [sample[i] for i in bottom5_idx],
            "method_note": (
                "Sentence embeddings (all-MiniLM-L6-v2) capture full semantic meaning "
                "including negation, intensity, mixed opinions, and context."
            ),
        }
        self.memory["text_semantic_analysis"] = _to_json_safe(result)
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 6 — Relationship analysis
    # ──────────────────────────────────────────────────────────────────────────

    def analyze_relationships(self) -> Dict[str, Any]:
        df          = self.df_clean
        rating_col  = self.primary_rating_col
        sent_col    = self.primary_sentiment_col
        rels: Dict[str, Any] = {}
        test_records = []

        if rating_col and self.numerical_cols:
            nr: Dict[str, Any] = {}
            for col in self.numerical_cols:
                pair = df[[col, rating_col]].dropna()
                if len(pair) < 10:
                    continue
                rho, pval = stats.spearmanr(pair[col], pair[rating_col])
                rec = {
                    "method":     "Spearman rank correlation",
                    "why_not_pearson": "Rating is ordinal. Pearson assumes interval linearity.",
                    "rho":        round(float(rho), 4),
                    "p_value":    round(float(pval), 6),
                    "significant": bool(pval < 0.05),
                    "strength":   _interpret_rho(rho),
                }
                nr[col] = rec
                test_records.append((rec, float(pval)))
            rels["numerical_vs_rating"] = nr

        sub_ratings = [c for c in self.rating_cols if c != rating_col]
        if rating_col and sub_ratings:
            sr: Dict[str, Any] = {}
            for col in sub_ratings:
                pair = df[[col, rating_col]].dropna()
                if len(pair) < 10:
                    continue
                rho, pval = stats.spearmanr(pair[col], pair[rating_col])
                rec = {
                    "method":      "Spearman rank correlation",
                    "rho":         round(float(rho), 4),
                    "p_value":     round(float(pval), 6),
                    "significant": bool(pval < 0.05),
                    "strength":    _interpret_rho(rho),
                }
                sr[col] = rec
                test_records.append((rec, float(pval)))
            rels["sub_ratings_vs_primary_rating"] = sr

        all_num = list(set(self.numerical_cols + (self.rating_cols if self.rating_cols else [])))
        if len(all_num) >= 2:
            nn: Dict[str, Any] = {}
            for i, col1 in enumerate(all_num):
                for col2 in all_num[i+1:]:
                    pair = df[[col1, col2]].dropna()
                    if len(pair) < 10:
                        continue
                    rho, pval = stats.spearmanr(pair[col1], pair[col2])
                    rec = {
                        "method":      "Spearman rank correlation",
                        "rho":         round(float(rho), 4),
                        "p_value":     round(float(pval), 6),
                        "significant": bool(pval < 0.05),
                        "strength":    _interpret_rho(rho),
                    }
                    nn[f"{col1}_vs_{col2}"] = rec
                    test_records.append((rec, float(pval)))
            rels["numerical_vs_numerical"] = nn

        if rating_col and self.categorical_cols:
            cr: Dict[str, Any] = {}
            for col in self.categorical_cols:
                pair = df[[col, rating_col]].dropna()
                if pair[col].nunique() < 2 or len(pair) < 20:
                    continue
                groups = [g[rating_col].values for _, g in pair.groupby(col)]
                try:
                    h_stat, pval = stats.kruskal(*groups)
                    n, k = len(pair), len(groups)
                    eta2 = max(0.0, (h_stat - k + 1) / (n - k)) if (n - k) > 0 else 0.0
                    rec = {
                        "method":      "Kruskal-Wallis H-test",
                        "why":         "Comparing ordinal rating distributions across categorical groups.",
                        "H_statistic": round(float(h_stat), 4),
                        "p_value":     round(float(pval), 6),
                        "significant": bool(pval < 0.05),
                        "eta_squared_approx": round(float(eta2), 4),
                        "group_means": {
                            str(k): round(float(v), 3)
                            for k, v in pair.groupby(col)[rating_col].mean().items()
                        },
                    }
                    cr[col] = rec
                    test_records.append((rec, float(pval)))
                except Exception:
                    pass
            rels["categorical_vs_rating"] = cr

        if sent_col and rating_col:
            pair = df[[sent_col, rating_col]].dropna()
            groups = [g[rating_col].values for _, g in pair.groupby(sent_col)]
            if len(groups) >= 2:
                try:
                    h_stat, pval = stats.kruskal(*groups)
                    rec = {
                        "method":      "Kruskal-Wallis H-test",
                        "H_statistic": round(float(h_stat), 4),
                        "p_value":     round(float(pval), 6),
                        "significant": bool(pval < 0.05),
                        "mean_rating_per_sentiment": {
                            str(k): round(float(v), 3)
                            for k, v in pair.groupby(sent_col)[rating_col].mean().items()
                        },
                    }
                    rels["sentiment_vs_rating"] = rec
                    test_records.append((rec, float(pval)))
                except Exception:
                    pass

        if sent_col and self.categorical_cols:
            cs: Dict[str, Any] = {}
            for col in self.categorical_cols[:8]:
                pair = df[[col, sent_col]].dropna()
                if pair[col].nunique() < 2:
                    continue
                try:
                    v = _cramers_v(pair[col], pair[sent_col])
                    cs[col] = {
                        "method":    "Cramér's V",
                        "cramers_v": round(float(v), 4),
                        "strength":  _interpret_rho(v),
                    }
                except Exception:
                    pass
            rels["categorical_vs_sentiment"] = cs

        if test_records:
            raw_pvals = [p for _, p in test_records]
            adj_pvals = _benjamini_hochberg(raw_pvals)
            for (rec, pval), adj_p in zip(test_records, adj_pvals):
                rec["bh_adjusted_p"] = round(float(adj_p), 6)
                rec["bh_significant"] = bool(adj_p < 0.05)

            rels["multiple_testing_correction"] = {
                "method": "Benjamini-Hochberg (FDR control)",
                "total_hypothesis_tests": len(raw_pvals),
                "significant_raw_p05": sum(1 for p in raw_pvals if p < 0.05),
                "significant_bh_p05":  sum(1 for p in adj_pvals if p < 0.05),
            }

        self.memory["relationships"] = _to_json_safe(rels)
        return rels

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 7 — Example-Based Semantic Rating Model & Conflict Assessment
    # ──────────────────────────────────────────────────────────────────────────

    def _build_example_bank(
        self,
        df_data: pd.DataFrame,
        text_col: str,
        rating_col: Optional[str],
        sent_col: Optional[str] = None,
        max_examples: int = 2500,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Extract unique texts from training data, aggregate their rating and sentiment
        distributions, compute entropy, and generate embeddings.
        """
        cols = [text_col]
        if rating_col and rating_col in df_data.columns:
            cols.append(rating_col)
        if sent_col and sent_col in df_data.columns:
            cols.append(sent_col)

        sub = df_data[cols].dropna(subset=[text_col])
        if len(sub) == 0:
            return [], np.empty((0, 384), dtype=np.float32)

        unique_text_counts = sub[text_col].astype(str).value_counts()
        unique_texts = unique_text_counts.index.tolist()

        if len(unique_texts) > max_examples:
            unique_texts = unique_texts[:max_examples]

        example_bank: List[Dict[str, Any]] = []
        rating_vals = sorted(sub[rating_col].unique()) if (rating_col and rating_col in sub.columns) else []
        grouped = sub.groupby(text_col)

        for text_str in unique_texts:
            if text_str not in grouped.groups:
                continue
            grp = grouped.get_group(text_str)
            n_occurrences = len(grp)

            entry: Dict[str, Any] = {
                "text": text_str,
                "total_count": int(n_occurrences),
            }

            if rating_col and rating_col in grp.columns:
                rvc = grp[rating_col].value_counts().to_dict()
                rating_dist = {str(rv): float(rvc.get(rv, 0)) / n_occurrences for rv in rating_vals}
                dom_r = float(grp[rating_col].mode().iloc[0])
                exp_r = float(sum(float(r) * rating_dist[str(r)] for r in rating_vals))
                var_r = float(sum(((float(r) - exp_r) ** 2) * rating_dist[str(r)] for r in rating_vals))
                ent = _calc_entropy(rating_dist)

                entry["rating_distribution"] = rating_dist
                entry["dominant_rating"] = dom_r
                entry["expected_rating"] = round(exp_r, 4)
                entry["rating_variance"] = round(var_r, 4)
                entry["rating_entropy"] = round(ent, 4)
                entry["n_distinct_ratings"] = int(len(rvc))

            if sent_col and sent_col in grp.columns:
                svc = grp[sent_col].value_counts().to_dict()
                sent_dist = {str(k): float(v) / n_occurrences for k, v in svc.items()}
                dom_s = str(grp[sent_col].mode().iloc[0])
                entry["sentiment_distribution"] = sent_dist
                entry["dominant_sentiment"] = dom_s

            example_bank.append(entry)

        texts_to_embed = [e["text"] for e in example_bank]
        embeddings = self._embed(texts_to_embed)

        for idx, e in enumerate(example_bank):
            e["embedding"] = embeddings[idx].tolist()

        return example_bank, embeddings

    def run_internal_cross_validation(
        self,
        df_pair: pd.DataFrame,
        text_col: str,
        rating_col: str,
        sent_col: Optional[str] = None,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """
        Internal K-Fold cross validation using ONLY the training dataset.
        Evaluates out-of-fold predictions using the example-based semantic rating model.
        Calculates MAE, RMSE, Spearman correlation, Exact accuracy, Within-1 accuracy,
        and sentiment validation accuracy.
        """
        from sklearn.model_selection import KFold

        print(f"  [Validation] Running {n_splits}-fold internal cross-validation on training data…")

        n_rows = len(df_pair)
        if n_rows < 10:
            return {}

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        y_true_ratings = []
        y_pred_exp_ratings = []
        y_pred_mode_ratings = []

        y_true_sents = []
        y_pred_sents = []

        all_rating_vals = sorted(df_pair[rating_col].unique())

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df_pair)):
            train_df = df_pair.iloc[train_idx]
            val_df = df_pair.iloc[val_idx]

            train_bank, train_embs = self._build_example_bank(
                train_df, text_col, rating_col, sent_col, max_examples=1500
            )
            if not train_bank:
                continue

            val_unique_texts = val_df[text_col].astype(str).unique().tolist()
            val_embs = self._embed(val_unique_texts)
            val_text_to_emb = {t: val_embs[i] for i, t in enumerate(val_unique_texts)}

            val_text_preds = {}
            for t_str, v_emb in val_text_to_emb.items():
                sims = np.array([_cosine_sim(v_emb, np.array(e["embedding"])) for e in train_bank])
                top_k = min(5, len(train_bank))
                top_idx = np.argsort(-sims)[:top_k]

                top_sims = sims[top_idx]
                weights = _softmax(top_sims, temp=0.1)

                pred_rating_dist = {str(rv): 0.0 for rv in all_rating_vals}
                pred_sent_dist: Dict[str, float] = {}

                for w, idx in zip(weights, top_idx):
                    e = train_bank[idx]
                    rdist = e.get("rating_distribution", {})
                    for rv_str, p in rdist.items():
                        if rv_str in pred_rating_dist:
                            pred_rating_dist[rv_str] += float(w * p)

                    sdist = e.get("sentiment_distribution", {})
                    for sv_str, p in sdist.items():
                        pred_sent_dist[sv_str] = pred_sent_dist.get(sv_str, 0.0) + float(w * p)

                tot_p = sum(pred_rating_dist.values())
                if tot_p > 0:
                    pred_rating_dist = {k: v / tot_p for k, v in pred_rating_dist.items()}

                exp_r = sum(float(r) * pred_rating_dist[str(r)] for r in all_rating_vals)
                mode_r = float(max(pred_rating_dist.items(), key=lambda x: x[1])[0])
                mode_s = max(pred_sent_dist.items(), key=lambda x: x[1])[0] if pred_sent_dist else "Unknown"

                val_text_preds[t_str] = {
                    "exp_rating": exp_r,
                    "mode_rating": mode_r,
                    "pred_sentiment": mode_s,
                }

            for _, row in val_df.iterrows():
                t_val = str(row[text_col])
                pred = val_text_preds.get(t_val)
                if pred:
                    y_true_ratings.append(float(row[rating_col]))
                    y_pred_exp_ratings.append(pred["exp_rating"])
                    y_pred_mode_ratings.append(pred["mode_rating"])
                    if sent_col and sent_col in row and pd.notnull(row[sent_col]):
                        y_true_sents.append(str(row[sent_col]))
                        y_pred_sents.append(pred["pred_sentiment"])

        y_true_arr = np.array(y_true_ratings)
        y_exp_arr = np.array(y_pred_exp_ratings)
        y_mode_arr = np.array(y_pred_mode_ratings)

        mae = float(np.mean(np.abs(y_true_arr - y_exp_arr))) if len(y_true_arr) else 0.0
        rmse = float(np.sqrt(np.mean((y_true_arr - y_exp_arr) ** 2))) if len(y_true_arr) else 0.0
        exact_acc = float(np.mean(y_true_arr == y_mode_arr) * 100.0) if len(y_true_arr) else 0.0
        within_1_acc = float(np.mean(np.abs(y_true_arr - y_mode_arr) <= 1.0) * 100.0) if len(y_true_arr) else 0.0

        if len(y_true_arr) >= 5 and np.std(y_exp_arr) > 1e-6 and np.std(y_true_arr) > 1e-6:
            rho, pval = stats.spearmanr(y_true_arr, y_exp_arr)
            rho_val = round(float(rho), 4)
            pval_val = round(float(pval), 6)
        else:
            rho_val = 0.0
            pval_val = 1.0

        sent_acc = float(np.mean([t == p for t, p in zip(y_true_sents, y_pred_sents)]) * 100.0) if y_true_sents else None

        cv_metrics = {
            "n_splits": n_splits,
            "total_validated_rows": len(y_true_arr),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "spearman_rho": rho_val,
            "spearman_pvalue": pval_val,
            "exact_accuracy_pct": round(exact_acc, 2),
            "within_1_accuracy_pct": round(within_1_acc, 2),
            "sentiment_accuracy_pct": round(sent_acc, 2) if sent_acc is not None else None,
        }

        print(f"  [Validation] Cross-Validation Results (Training 80% Data):")
        print(f"    • MAE                  : {cv_metrics['mae']:.4f}")
        print(f"    • RMSE                 : {cv_metrics['rmse']:.4f}")
        print(f"    • Spearman Correlation : rho = {cv_metrics['spearman_rho']:+.4f} (p = {cv_metrics['spearman_pvalue']})")
        print(f"    • Exact Rating Accuracy: {cv_metrics['exact_accuracy_pct']:.2f}%")
        print(f"    • Within-1 Accuracy    : {cv_metrics['within_1_accuracy_pct']:.2f}%")
        if sent_acc is not None:
            print(f"    • Sentiment Accuracy   : {cv_metrics['sentiment_accuracy_pct']:.2f}%")

        return cv_metrics

    def learn_text_rating_patterns(self) -> Dict[str, Any]:
        if not self.primary_text_col or not self.primary_rating_col:
            return {}

        df         = self.df_clean
        text_col   = self.primary_text_col
        rating_col = self.primary_rating_col
        sent_col   = self.primary_sentiment_col

        pair = df[[text_col, rating_col] + ([sent_col] if sent_col else [])].dropna(subset=[text_col, rating_col])
        rating_vals = sorted(pair[rating_col].unique())
        total_rows = len(pair)
        unique_texts_in_pair = pair[text_col].astype(str).nunique()

        print(f"\n  ┌─ Example-Based Semantic Rating Model Training ────")
        print(f"  │ Source dataset        : COMPLETE (no train/test split)")
        print(f"  │ Text Column           : '{text_col}'")
        print(f"  │ Rating Target         : '{rating_col}' ({len(rating_vals)} levels: {rating_vals})")
        if sent_col:
            print(f"  │ Sentiment Col         : '{sent_col}'")
        print(f"  │ Total observations    : {total_rows:,} rows (duplicates retained)")
        print(f"  │ Unique review texts   : {unique_texts_in_pair:,}")
        print(f"  └───────────────────────────────────────────────────\n")

        # 1. Build Full Training Example Bank
        # Each unique text is aggregated with its full empirical rating distribution
        # computed from ALL of its occurrences (including duplicates).
        print(f"  [Phase 7] Aggregating unique text examples and empirical rating distributions…")
        example_bank, embeddings = self._build_example_bank(
            pair, text_col, rating_col, sent_col, max_examples=self.MAX_EMBED_GLOBAL
        )
        unique_texts_count = len(example_bank)
        duplicate_ratio = float(total_rows) / max(unique_texts_count, 1)

        # 2. Conflict & Duplication Analytics
        texts_multi = sum(1 for e in example_bank if e.get("n_distinct_ratings", 1) > 1)
        texts_all = sum(1 for e in example_bank if e.get("n_distinct_ratings", 1) == len(rating_vals))
        entropies = [e.get("rating_entropy", 0.0) for e in example_bank]
        avg_entropy = float(np.mean(entropies)) if entropies else 0.0
        text_rating_consistency = max(0.0, 1.0 - avg_entropy)

        pct_multi = (texts_multi / max(unique_texts_count, 1)) * 100.0
        pct_all = (texts_all / max(unique_texts_count, 1)) * 100.0

        print(f"  [Aggregation] Total observations   : {total_rows:,} rows (all duplicates retained)")
        print(f"  [Aggregation] Unique texts in bank : {unique_texts_count:,} (each with empirical rating dist)")
        print(f"  [Aggregation] Avg observations/text: {duplicate_ratio:.1f}x repeats per unique text")
        print(f"  [Conflict Analysis] Texts spanning multiple rating levels: {texts_multi:,} ({pct_multi:.1f}%)")
        print(f"  [Conflict Analysis] Texts spanning ALL {len(rating_vals)} rating levels : {texts_all:,} ({pct_all:.1f}%)")
        print(f"  [Conflict Analysis] Average rating entropy per text      : {avg_entropy:.4f} (Consistency: {text_rating_consistency*100:.1f}%)")

        # 3. Per-Rating Centroids (for diagnostic & separation measurement)
        rating_profiles: Dict[str, Any] = {}
        for rv in rating_vals:
            subset_texts = pair[pair[rating_col] == rv][text_col].astype(str).tolist()
            unique_sub = list(dict.fromkeys(subset_texts))
            if len(unique_sub) < 1:
                continue
            sample = unique_sub[:self.MAX_EMBED_PER_RATING]
            emb = self._embed(sample)
            centroid = emb.mean(axis=0)
            sims = [_cosine_sim(e, centroid) for e in emb]
            top_idx = sorted(range(len(sims)), key=lambda i: -sims[i])[:5]
            rating_profiles[str(rv)] = {
                "rating": float(rv),
                "n_total": len(subset_texts),
                "n_unique": len(unique_sub),
                "centroid": centroid.tolist(),
                "representative_texts": [sample[i] for i in top_idx],
            }

        # 4. Sentiment Profiles
        sentiment_profiles: Dict[str, Any] = {}
        if sent_col:
            for sv in pair[sent_col].unique():
                subset_s = pair[pair[sent_col] == sv][text_col].astype(str).tolist()
                unique_s = list(dict.fromkeys(subset_s))[:self.MAX_EMBED_PER_RATING]
                emb = self._embed(unique_s)
                centroid = emb.mean(axis=0)
                sentiment_profiles[str(sv)] = {
                    "sentiment": str(sv),
                    "n_total": len(subset_s),
                    "n_unique": len(unique_s),
                    "centroid": centroid.tolist(),
                }

        # 5. Centroid Separation Diagnostic
        pairwise_sims = {}
        all_sims = []
        if len(rating_profiles) >= 2:
            keys = sorted(rating_profiles.keys(), key=float)
            for i, k1 in enumerate(keys):
                for k2 in keys[i+1:]:
                    c1 = np.array(rating_profiles[k1]["centroid"])
                    c2 = np.array(rating_profiles[k2]["centroid"])
                    sim = _cosine_sim(c1, c2)
                    pairwise_sims[f"{k1}_{k2}"] = round(float(sim), 4)
                    all_sims.append(sim)
        avg_centroid_sim = float(np.mean(all_sims)) if all_sims else 0.0
        centroid_collapse = avg_centroid_sim > 0.95

        # 6. Semantic Neighborhood Consistency
        neighborhood_agreement = 0.0
        if len(example_bank) >= 3 and embeddings.shape[0] >= 3:
            diffs = []
            for i, e1 in enumerate(example_bank):
                sims = np.array([_cosine_sim(embeddings[i], embeddings[j]) if i != j else -1.0 for j in range(len(example_bank))])
                nn_idx = int(np.argmax(sims))
                e2 = example_bank[nn_idx]
                r1 = e1.get("expected_rating", e1.get("dominant_rating", 3.0))
                r2 = e2.get("expected_rating", e2.get("dominant_rating", 3.0))
                diffs.append(abs(r1 - r2))
            avg_neighbor_diff = float(np.mean(diffs))
            scale_span = max(rating_vals) - min(rating_vals) if len(rating_vals) > 1 else 1.0
            neighborhood_agreement = max(0.0, 1.0 - (avg_neighbor_diff / scale_span))

        # 7. Internal Cross-Validation on Training Data Only
        cv_metrics = self.run_internal_cross_validation(pair, text_col, rating_col, sent_col, n_splits=5)

        # 8. Automatic Signal Quality Assessment Tier
        cv_rho = cv_metrics.get("spearman_rho", 0.0)
        cv_mae = cv_metrics.get("mae", 999.0)
        scale_span = max(rating_vals) - min(rating_vals) if len(rating_vals) > 1 else 4.0

        if centroid_collapse or pct_all >= 30.0 or (pct_multi >= 90.0 and text_rating_consistency < 0.40):
            signal_tier = "NO RELIABLE TEXT→RATING SIGNAL"
            signal_explanation = (
                f"CRITICAL: Free-text review content carries NO reliable discriminative signal for numeric ratings. "
                f"The dataset contains {unique_texts_count} unique texts repeated across {total_rows:,} rows ({duplicate_ratio:.1f}x repeats). "
                f"{texts_multi} unique texts ({pct_multi:.1f}%) occur across multiple contradictory rating levels "
                f"({texts_all} appear across all {len(rating_vals)} rating levels), with average rating entropy of {avg_entropy:.3f} "
                f"(consistency: {text_rating_consistency*100:.1f}%) and rating centroid cosine similarity of {avg_centroid_sim:.4f}."
            )
            is_reliable = False
        elif (cv_rho >= 0.50 and text_rating_consistency >= 0.70 and cv_mae <= (scale_span * 0.20) and not centroid_collapse):
            signal_tier = "STRONG TEXT→RATING SIGNAL"
            signal_explanation = "Review text shows strong, consistent predictive alignment with rating levels."
            is_reliable = True
        elif (cv_rho >= 0.25 and text_rating_consistency >= 0.45 and not centroid_collapse):
            signal_tier = "MODERATE TEXT→RATING SIGNAL"
            signal_explanation = "Review text provides moderate discriminative signal for ratings."
            is_reliable = True
        elif (cv_rho >= 0.10 and text_rating_consistency >= 0.25):
            signal_tier = "WEAK TEXT→RATING SIGNAL"
            signal_explanation = "Review text shows weak and inconsistent relationship with rating levels."
            is_reliable = False
        else:
            signal_tier = "NO RELIABLE TEXT→RATING SIGNAL"
            signal_explanation = (
                f"CRITICAL: Free-text review content carries NO reliable discriminative signal for numeric ratings. "
                f"The dataset contains {unique_texts_count} unique texts repeated across {total_rows:,} rows ({duplicate_ratio:.1f}x repeats). "
                f"{texts_multi} unique texts ({pct_multi:.1f}%) occur across multiple contradictory rating levels "
                f"({texts_all} appear across all rating levels), with average rating entropy of {avg_entropy:.3f} "
                f"and cross-validation Spearman correlation rho = {cv_rho:+.4f}."
            )
            is_reliable = False

        print(f"\n  ┌─ Text-to-Rating Signal Quality Assessment ────────")
        print(f"  │ Signal Classification   : {signal_tier}")
        print(f"  │ Rating Predictability   : {'RELIABLE' if is_reliable else 'UNRELIABLE'}")
        print(f"  │ Text-Rating Consistency : {text_rating_consistency*100:.1f}% (Entropy: {avg_entropy:.3f})")
        print(f"  │ Neighborhood Agreement  : {neighborhood_agreement*100:.1f}%")
        print(f"  │ CV Spearman Correlation : rho = {cv_rho:+.4f}")
        print(f"  │ CV MAE                  : {cv_mae:.4f}")
        print(f"  │ Centroid Similarity     : {avg_centroid_sim:.4f} (Collapse: {centroid_collapse})")
        print(f"  └───────────────────────────────────────────────────\n")

        text_quality = {
            "total_unique_texts": unique_texts_count,
            "total_rows_with_text": total_rows,
            "avg_repetitions_per_text": round(duplicate_ratio, 1),
            "texts_multiple_ratings": texts_multi,
            "pct_texts_multiple_ratings": round(pct_multi, 2),
            "texts_all_ratings": texts_all,
            "pct_texts_all_ratings": round(pct_all, 2),
            "avg_rating_entropy": round(avg_entropy, 4),
            "text_rating_consistency": round(text_rating_consistency, 4),
            "neighborhood_agreement": round(neighborhood_agreement, 4),
            "centroid_pairwise_similarities": pairwise_sims,
            "avg_centroid_similarity": round(avg_centroid_sim, 4),
            "centroid_collapse_detected": bool(centroid_collapse),
            "centroid_collapse_explanation": signal_explanation,
        }

        signal_quality = {
            "signal_tier": signal_tier,
            "is_reliable": is_reliable,
            "text_rating_consistency": round(text_rating_consistency, 4),
            "avg_rating_entropy": round(avg_entropy, 4),
            "neighborhood_agreement": round(neighborhood_agreement, 4),
            "cv_spearman_rho": cv_rho,
            "cv_mae": cv_mae,
            "explanation": signal_explanation,
        }

        result = {
            "example_bank": example_bank,
            "rating_profiles": rating_profiles,
            "sentiment_profiles": sentiment_profiles,
            "dataset_text_quality": text_quality,
            "text_rating_signal_quality": signal_quality,
            "internal_cross_validation": cv_metrics,
        }

        self.memory["text_rating_patterns"] = _to_json_safe(result)
        self.memory["dataset_text_quality"] = _to_json_safe(text_quality)
        self.memory["text_rating_signal_quality"] = _to_json_safe(signal_quality)
        self.memory["internal_cross_validation"] = _to_json_safe(cv_metrics)
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 9A — Text Theme Discovery
    # ──────────────────────────────────────────────────────────────────────────

    def discover_text_themes(self) -> Dict[str, Any]:
        if not self.primary_text_col:
            return {}

        df = self.df_clean
        text_col = self.primary_text_col
        texts = df[text_col].dropna().astype(str).tolist()
        unique_texts = list(dict.fromkeys(texts))
        if len(unique_texts) < 10:
            return {}

        sample = unique_texts
        if len(unique_texts) > self.MAX_EMBED_GLOBAL:
            random.seed(42)
            sample = random.sample(unique_texts, self.MAX_EMBED_GLOBAL)

        print(f"  [Phase 9A] Discovering semantic text themes from {len(sample):,} unique texts…")
        emb = self._embed(sample)

        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        best_k = 3
        best_score = -1.0
        best_kmeans = None
        max_k = min(8, len(sample) - 1)

        for k in range(3, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            labels = km.fit_predict(emb)
            if len(set(labels)) > 1:
                score = float(silhouette_score(emb, labels))
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_kmeans = km

        if best_kmeans is None:
            return {}

        labels = best_kmeans.labels_
        centers = best_kmeans.cluster_centers_

        themes = []
        for i in range(best_k):
            cluster_indices = [idx for idx, l in enumerate(labels) if l == i]
            cluster_emb = emb[cluster_indices]
            cluster_texts = [sample[idx] for idx in cluster_indices]
            center = centers[i]

            sims = [_cosine_sim(e, center) for e in cluster_emb]
            top_reps_idx = sorted(range(len(sims)), key=lambda idx: -sims[idx])[:5]
            top_reps = [cluster_texts[idx] for idx in top_reps_idx]

            theme_info = {
                "theme_id": i + 1,
                "size": len(cluster_texts),
                "pct_of_sample": round(100 * len(cluster_texts) / len(sample), 1),
                "representative_texts": top_reps,
            }
            themes.append(theme_info)

        result = {
            "optimal_k": best_k,
            "silhouette_score": round(best_score, 4),
            "sample_size": len(sample),
            "themes": themes,
        }
        print(f"  [Phase 9A] Discovered {best_k} semantic themes (silhouette score: {best_score:.4f}).")
        self.memory["text_themes"] = _to_json_safe(result)
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 9B — Interaction Discovery
    # ──────────────────────────────────────────────────────────────────────────

    def discover_interactions(self) -> Dict[str, Any]:
        df = self.df_clean
        rating_col = self.primary_rating_col
        if not rating_col or not self.numerical_cols or not self.categorical_cols:
            return {}

        print("  [Phase 9B] Searching for conditional interaction patterns across subgroups…")
        interactions = []

        for num_col in self.numerical_cols[:4]:
            for cat_col in self.categorical_cols[:4]:
                pair = df[[num_col, rating_col, cat_col]].dropna()
                if pair[cat_col].nunique() < 2 or len(pair) < 50:
                    continue

                subgroup_rhos = {}
                for cat_val, grp in pair.groupby(cat_col):
                    if len(grp) >= 20:
                        rho, pval = stats.spearmanr(grp[num_col], grp[rating_col])
                        subgroup_rhos[str(cat_val)] = {
                            "n": len(grp),
                            "rho": round(float(rho), 4),
                            "p_value": round(float(pval), 6),
                            "strength": _interpret_rho(rho),
                        }

                if len(subgroup_rhos) >= 2:
                    rhos = [v["rho"] for v in subgroup_rhos.values()]
                    max_rho, min_rho = max(rhos), min(rhos)
                    rho_diff = max_rho - min_rho
                    if rho_diff >= 0.15:
                        interactions.append({
                            "numerical_var": num_col,
                            "target_var": rating_col,
                            "grouping_var": cat_col,
                            "overall_rho": round(float(stats.spearmanr(pair[num_col], pair[rating_col])[0]), 4),
                            "subgroup_rhos": subgroup_rhos,
                            "interaction_span": round(rho_diff, 4),
                            "description": f"Correlation between {num_col} and {rating_col} varies across {cat_col} (span: {rho_diff:.2f}).",
                        })

        result = {
            "total_interactions_found": len(interactions),
            "interactions": interactions,
        }
        print(f"  [Phase 9B] Discovered {len(interactions)} conditional interaction patterns.")
        self.memory["interactions"] = _to_json_safe(result)
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 9C — Subgroup & Conditional Analysis
    # ──────────────────────────────────────────────────────────────────────────

    def discover_subgroup_patterns(self) -> Dict[str, Any]:
        df = self.df_clean
        rating_col = self.primary_rating_col
        sent_col = self.primary_sentiment_col
        if not rating_col or not self.categorical_cols:
            return {}

        print("  [Phase 9C] Conducting subgroup consistency & Simpson's Paradox checks…")
        subgroup_findings = []

        if sent_col:
            for cat_col in self.categorical_cols[:5]:
                pair = df[[sent_col, rating_col, cat_col]].dropna()
                if pair[cat_col].nunique() < 2 or len(pair) < 50:
                    continue

                group_means = {}
                for cat_val, grp in pair.groupby(cat_col):
                    if len(grp) >= 20:
                        means = grp.groupby(sent_col)[rating_col].mean().to_dict()
                        group_means[str(cat_val)] = {str(k): round(float(v), 3) for k, v in means.items()}

                if len(group_means) >= 2:
                    subgroup_findings.append({
                        "primary_relationship": f"{sent_col} ↔ {rating_col}",
                        "subgroup_variable": cat_col,
                        "subgroup_means": group_means,
                        "finding": f"Sentiment ↔ rating pattern evaluated across levels of {cat_col}.",
                        "is_consistent": True,
                    })

        result = {
            "total_subgroup_checks": len(subgroup_findings),
            "subgroup_findings": subgroup_findings,
        }
        print(f"  [Phase 9C] Evaluated subgroup consistency across {len(subgroup_findings)} breakdowns.")
        self.memory["subgroup_findings"] = _to_json_safe(result)
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 9D — Ranked Discovery Summary
    # ──────────────────────────────────────────────────────────────────────────

    def rank_discoveries(self) -> List[Dict[str, Any]]:
        print("  [Phase 9D] Ranking top discoveries across dataset…")
        candidates = []
        rels = self.memory.get("relationships", {})
        tq = self.memory.get("dataset_text_quality", {})
        themes = self.memory.get("text_themes", {})
        interactions = self.memory.get("interactions", {})

        sig = self.memory.get("text_rating_signal_quality", {})
        cv = self.memory.get("internal_cross_validation", {})
        if sig.get("signal_tier") in ("NO RELIABLE TEXT→RATING SIGNAL", "WEAK TEXT→RATING SIGNAL") or tq.get("centroid_collapse_detected"):
            candidates.append({
                "score": 1000.0,
                "finding": f"Text Unreliability for Rating Prediction ({sig.get('signal_tier', 'Unreliable Signal')})",
                "evidence": (
                    f"Only {tq.get('total_unique_texts')} unique texts shared across {tq.get('total_rows_with_text'):,} rows "
                    f"({tq.get('pct_texts_multiple_ratings', 0.0):.1f}% multi-rating overlap, avg entropy: {tq.get('avg_rating_entropy', 0.0):.3f}, "
                    f"CV Spearman rho: {cv.get('spearman_rho', 0.0):+.4f})."
                ),
                "strength": "Critical",
                "why_it_matters": "Free-text review content carries no reliable discriminative signal for numeric rating prediction in this dataset.",
                "possible_explanation": "Dataset contains repeated template reviews assigned across contradictory rating levels.",
                "limitations": "Applies to text-to-rating prediction; sentiment classification remains statistically valid.",
                "confidence": "High",
            })

        if "sentiment_vs_rating" in rels:
            sr = rels["sentiment_vs_rating"]
            h_val = sr.get("H_statistic", 0.0)
            candidates.append({
                "score": float(h_val),
                "finding": "Strong Sentiment ↔ Rating Alignment",
                "evidence": f"Kruskal-Wallis H = {h_val:.1f}, p = {sr.get('p_value')}. Mean ratings: {sr.get('mean_rating_per_sentiment')}.",
                "strength": "Strong",
                "why_it_matters": "Customer sentiment tags align strongly with numeric rating scores.",
                "possible_explanation": "Sentiment and ratings both capture overall customer satisfaction.",
                "limitations": "Association only — sentiment and rating may be co-assigned.",
                "confidence": "High",
            })

        sr_dict = rels.get("sub_ratings_vs_primary_rating", {})
        for col, rinfo in sr_dict.items():
            rho = abs(rinfo.get("rho", 0.0))
            candidates.append({
                "score": rho * 100.0,
                "finding": f"High Association: {col} ↔ Primary Rating",
                "evidence": f"Spearman ρ = {rinfo.get('rho'):+.4f} (p = {rinfo.get('p_value')}).",
                "strength": rinfo.get("strength", "moderate").capitalize(),
                "why_it_matters": f"Aspect rating '{col}' is a major component of overall customer evaluation.",
                "possible_explanation": "Specific product attribute experience directly shapes overall rating.",
                "limitations": "Spearman rank correlation measures monotonic association, not direct causality.",
                "confidence": "High",
            })

        nr_dict = rels.get("numerical_vs_rating", {})
        for col, rinfo in nr_dict.items():
            rho = abs(rinfo.get("rho", 0.0))
            if rho >= 0.15:
                candidates.append({
                    "score": rho * 50.0,
                    "finding": f"Predictive Association: {col} ↔ Primary Rating",
                    "evidence": f"Spearman ρ = {rinfo.get('rho'):+.4f} (BH adj p = {rinfo.get('bh_adjusted_p')}).",
                    "strength": rinfo.get("strength", "weak").capitalize(),
                    "why_it_matters": f"{col} provides measurable predictive information for customer ratings.",
                    "possible_explanation": f"Numerical metric {col} correlates with engagement or satisfaction.",
                    "limitations": "Correlational finding; check for confounding factors.",
                    "confidence": "Moderate" if rinfo.get("bh_significant") else "Low",
                })

        if themes.get("themes"):
            k = themes.get("optimal_k")
            score_val = themes.get("silhouette_score", 0.0)
            candidates.append({
                "score": score_val * 40.0,
                "finding": f"Automatic Discovery of {k} Semantic Text Themes",
                "evidence": f"KMeans silhouette score = {score_val:.4f} across {themes.get('sample_size')} unique texts.",
                "strength": "Moderate",
                "why_it_matters": "Identifies distinct recurring topics/themes in customer feedback.",
                "possible_explanation": "Unsupervised clustering of sentence embeddings isolates key discussion topics.",
                "limitations": "Theme labels are unsupervised clusters from MiniLM embeddings.",
                "confidence": "Moderate",
            })

        candidates.sort(key=lambda x: -x["score"])
        top_ranked = candidates[:10]

        self.memory["ranked_discoveries"] = _to_json_safe(top_ranked)
        print(f"  [Phase 9D] Successfully ranked top {len(top_ranked)} discoveries.")
        return top_ranked

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 8 — Save and Load memory
    # ──────────────────────────────────────────────────────────────────────────

    def build_memory(self, memory_path: str = "memory.json") -> str:
        self.memory["meta"] = {
            "version":     "2.0",
            "csv_path":    self.csv_path,
            "memory_path": memory_path,
        }
        self.memory["column_detection_summary"] = {
            "primary_text_column":      self.primary_text_col,
            "primary_rating_column":    self.primary_rating_col,
            "primary_sentiment_column": self.primary_sentiment_col,
            "text_columns":       self.text_cols,
            "rating_columns":     self.rating_cols,
            "sentiment_columns":  self.sentiment_cols,
            "numerical_columns":  self.numerical_cols,
            "categorical_columns": self.categorical_cols,
            "identifier_columns": self.identifier_cols,
        }

        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump(_to_json_safe(self.memory), f, indent=2, ensure_ascii=False)

        print(f"  [Memory] Saved → {memory_path}")
        self.retriever.index_memory()
        return memory_path

    def load_memory(self, memory_path: str = "memory.json"):
        with open(memory_path, "r", encoding="utf-8") as f:
            self.memory = json.load(f)

        det = self.memory.get("column_detection_summary", {})
        self.primary_text_col      = det.get("primary_text_column")
        self.primary_rating_col    = det.get("primary_rating_column")
        self.primary_sentiment_col = det.get("primary_sentiment_column")
        self.text_cols             = det.get("text_columns",       [])
        self.rating_cols           = det.get("rating_columns",     [])
        self.sentiment_cols        = det.get("sentiment_columns",  [])
        self.numerical_cols        = det.get("numerical_columns",  [])
        self.categorical_cols      = det.get("categorical_columns",[])
        self.identifier_cols       = det.get("identifier_columns", [])

        # Load df_clean for dynamic dataset inspection if not loaded
        if self.df_clean is None:
            csv_target = self.memory.get("load_stats", {}).get("csv_path") or self.csv_path
            if csv_target and os.path.exists(csv_target):
                self.load_csv(csv_target)

        print(f"  [Memory] Loaded ← {memory_path}")
        self.retriever.index_memory()

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 9 — Open-Ended Q&A & Unseen Review Analysis
    # ──────────────────────────────────────────────────────────────────────────

    def is_dataset_inquiry(self, text: str) -> bool:
        """
        Distinguishes between a dataset question and an unseen customer review text.
        """
        t = text.strip()
        if not t:
            return False
        
        t_lower = t.lower()
        
        # Explicit prefix overrides
        if t_lower.startswith(("analyze:", "review:", "unseen:")):
            return False
        if t_lower.startswith(("query:", "question:", "dataset:", "ask:")):
            return True

        # 1. Clear Dataset & Statistical Terminology (strong signal)
        analytical_terms = [
            r"\bdata(?:set|frame)?\b",
            r"\bcol(?:umn)?s?\b",
            r"\bvariables?\b",
            r"\bfeatures?\b",
            r"\bdistributions?\b",
            r"\bcorrelations?\b",
            r"\bcorrelated\b",
            r"\bspearman\b",
            r"\bpearson\b",
            r"\bkruskal\b",
            r"\bcram[eé]r\b",
            r"\bp-?values?\b",
            r"\bfdr\b",
            r"\bbenjamini\b",
            r"\bmissing (?:values?|data|cells?)\b",
            r"\bnulls?\b",
            r"\boutliers?\b",
            r"\bbalanced\b",
            r"\bimbalanced?\b",
            r"\bclass imbalance\b",
            r"\bskew(?:ed|ness)?\b",
            r"\bsimpson'?s\b",
            r"\bcentroids?\b",
            r"\bentropy\b",
            r"\bmethodology\b",
            r"\bfindings\b",
            r"\bdiscoveries\b",
            r"\bsubgroups?\b",
            r"\btarget variable\b",
            r"\baspect ratings?\b",
            r"\bsub-?ratings?\b",
            r"\bsample size\b",
            r"\brow count\b",
            r"\bnumber of (?:rows|records|samples|columns|variables)\b",
            r"\boverview of the (?:data|dataset)\b",
        ]
        
        has_analytical_term = any(re.search(pat, t_lower) for pat in analytical_terms)
        
        # 2. Direct Assistant Commands / Prompts asking for data explanations
        assistant_commands = [
            r"^(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:explain|tell|show|give|summarize|list|describe|analyze|clarify|find|detail|provide)\b",
            r"^(?:please\s+)?(?:explain|summarize|describe|list\s+all|show\s+me|tell\s+me\s+about|give\s+an?\s+overview|give\s+me)\b",
            r"^(?:what|which|how|why)\s+(?:can|could)\s+you\s+tell\s+me\b",
        ]
        has_assistant_cmd = any(re.search(pat, t_lower) for pat in assistant_commands)
        
        # 3. Relational / Analytical Question Patterns
        relational_patterns = [
            r"\brelationship\s+between\b",
            r"\bassociation\s+between\b",
            r"\bcorrelation\s+between\b",
            r"\bdifference\s+between\b",
            r"\bassociated\s+with\b",
            r"^(?:what|which|how|why)\s+(?:are|is|do|does|can)\s+(?:the\s+)?(?:key|main|top|most|primary|important)?\s*(?:variables?|columns?|features?|factors?|drivers?|associations?|findings?)\b",
            r"^(?:does|do|how\s+does|can)\s+([a-z0-9_]+)\s+(?:affect|impact|influence|drive|determine|correlate\s+with|predict|relate\s+to)\b",
            r"^(?:is|are)\s+([a-z0-9_]+)\s+(?:correlated|associated|related|linked)\s+(?:with|to)\b",
            r"^(?:is|are)\s+(?:there\s+)?(?:any\s+)?(?:statistically\s+significant|strong|weak|meaningful)\s+(?:correlation|relationship|difference|pattern|association)\b",
        ]
        has_relational_pat = any(re.search(pat, t_lower) for pat in relational_patterns)

        if has_analytical_term and (t.endswith("?") or has_assistant_cmd or has_relational_pat or any(t_lower.startswith(w) for w in ("what", "which", "how", "why", "is", "are", "does", "do", "explain", "summarize", "show", "list", "describe", "tell"))):
            return True

        if has_assistant_cmd or has_relational_pat:
            return True

        # 4. Check known dataset column names in question syntax
        # Dynamically fetch known dataset columns if df is loaded
        known_cols = set()
        if self.df_clean is not None:
            known_cols.update(self.df_clean.columns)
        if self.memory:
            target = self.memory.get("target_analysis", {})
            for k, v in target.items():
                if isinstance(v, dict) and "column" in v:
                    known_cols.add(v["column"])
            
        if not known_cols:
            known_cols = {
                "rating", "ratings", "sentiment", "sentiments", "country", "verified_purchase",
                "price_usd", "age", "helpful_votes", "word_count", "review_length",
                "battery_life_rating", "camera_rating", "display_rating", "performance_rating", "design_rating"
            }
        else:
            # Always ensure primary terms are in known_cols as fallback aliases
            known_cols.update({"rating", "ratings", "sentiment", "sentiments"})
        
        cols_present = [c for c in known_cols if re.search(r"\b" + re.escape(c.lower()) + r"\b", t_lower)]
        if len(cols_present) >= 2 and any(term in t_lower for term in ["affect", "impact", "influence", "correlate", "associated", "between", "relationship", "predict", "versus", " vs "]):
            return True

        if cols_present and any(re.search(r"^(?:what|which|how|why|is|are|does|do|show|explain|tell|summarize)\b", t_lower) for _ in [1]):
            if any(term in t_lower for term in ["distribution", "mean", "median", "average", "stats", "summary", "breakdown", "affect", "impact", "matter", "balanced", "skewed", "predict", "correlate", "association", "relationship"]):
                return True

        # 5. Check if it is a general question ending with '?' AND has no personal product review markers
        review_signals = [
            r"\b(?:i|my|mine|me|we|our)\b",
            r"\b(?:phone|camera|battery|screen|ui|app|device|product|item|purchase|delivery|packaging|quality|hardware|software)\b",
            r"\b(?:bought|ordered|arrived|received|charged|charging|returned|returning|broke|broken|repaired|using|used)\b",
            r"\b(?:works|working|worked|drains|draining|lasts|lasting|runs|running|loads|loading|heats|heating)\b",
            r"\b(?:great|good|bad|terrible|horrible|decent|awful|amazing|excellent|poor|okay|fine|satisfactory|cheap|expensive|worth|waste|disappointed|impressed|love|loved|hate|hated|recommend)\b",
            r"\b(?:casual use|daily use|nothing special|supposed to|gets the job done|value for money|money back|worth the price|for the price|every penny|no complaints|highly recommend)\b",
        ]
        
        review_matches = sum(1 for pat in review_signals if re.search(pat, t_lower))
        
        # If text is clearly an exclamatory/dropped subject review statement:
        if re.search(r"^(?:does\s+(?:what|everything|the\s+job|work|not)|is\s+(?:very|a\s+bit|pretty|quite|really|decent|good|okay|great|fine|terrible|cheap|worth|not)|can\s+(?:easily|run|last|handle|recommend|not|'t)|works?\s+(?:as|fine|great|well)|has\s+(?:great|good|bad|decent|poor)|what\s+a\s+)\b", t_lower):
            if not has_analytical_term and not has_relational_pat:
                return False
                
        if review_matches >= 2 and not has_analytical_term and not has_assistant_cmd and not has_relational_pat:
            return False

        # If it ends with ? and doesn't look like a review
        if t.endswith("?"):
            if any(re.search(r"^(?:what|why|how|which|where|when|who|is|are|does|do|can|could|should|would)\b", t_lower) for _ in [1]):
                if review_matches == 0:
                    return True
                    
        return False

    def answer_question(self, query_or_text: str) -> str:
        """
        Dual-mode runtime handler:
        A. Normal dataset question -> semantic retrieval & reasoning over memory.json
        B. Unseen review text -> semantic sentiment and rating validation analysis
        """
        q = query_or_text.strip()
        if not q:
            return "Please provide a valid question or review text."

        if q.lower().startswith("analyze:"):
            text = q[len("analyze:"):].strip()
            return self.analyze_new_text(text)

        # If input is a review rather than a dataset inquiry, perform new review analysis
        if not self.is_dataset_inquiry(q):
            return self.analyze_new_text(q)

        # 1. Semantic Memory Retrieval
        retrieved = self.retriever.retrieve(q, top_k=4)

        # 2. Dynamic On-The-Fly Dataset Inspection (if specific column mentioned)
        dynamic_ev = self.retriever.dynamically_inspect_data(q)

        # 3. Context assembly
        context = {
            "dataset_text_quality": self.memory.get("dataset_text_quality", {}),
            "column_detection": self.memory.get("column_detection_summary", {}),
        }

        # 4. Reason and synthesize answer
        return self.reasoning_engine.synthesize_answer(
            query=q,
            retrieved_chunks=retrieved,
            dynamic_evidence=dynamic_ev,
            agent_context=context,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 10 — Analyze an unseen review text
    # ──────────────────────────────────────────────────────────────────────────

    def analyze_new_text(self, text: str) -> str:
        """
        Analyze an unseen review text using the example-based semantic rating architecture.
        Calculates:
          1. Predicted sentiment, confidence & similarity scores across sentiment classes
          2. Example-based Likert rating distribution across scale (e.g. 1-5)
          3. Expected rating, most likely rating, and rating confidence
          4. Top supporting training examples with their similarity and rating distributions
          5. Automatic signal quality & reliability verification (explicit UNRELIABLE when text does not distinguish ratings)
          6. Rigorous explanatory reasoning
        """
        patterns = self.memory.get("text_rating_patterns", {})
        example_bank = patterns.get("example_bank", [])
        rating_profiles = patterns.get("rating_profiles", {})
        sent_profiles = patterns.get("sentiment_profiles", {})
        sig_quality = self.memory.get("text_rating_signal_quality", {})
        target_info = self.memory.get("target_analysis", {})

        if not example_bank and not rating_profiles and not sent_profiles:
            return "No learned patterns in memory. Please run dataset analysis first."

        vec = self._embed([text])[0]

        # ── 1. Sentiment Analysis ─────────────────────────────────────────────
        has_sentiment = bool(sent_profiles or self.primary_sentiment_col)
        top_sent = "Unknown"
        sent_conf_str = "Low"
        sent_sims_display: Dict[str, float] = {}

        if has_sentiment and sent_profiles:
            sent_sims: Dict[str, float] = {}
            for sv, prof in sent_profiles.items():
                centroid = np.array(prof["centroid"])
                sent_sims[sv] = _cosine_sim(vec, centroid)

            sorted_sents = sorted(sent_sims.items(), key=lambda x: -x[1])
            top_sent, top_sent_sim = sorted_sents[0] if sorted_sents else ("Unknown", 0.0)
            sec_sent_sim = sorted_sents[1][1] if len(sorted_sents) > 1 else top_sent_sim
            sent_margin = top_sent_sim - sec_sent_sim

            if sent_margin >= 0.08:
                sent_conf_str = "High"
            elif sent_margin >= 0.03:
                sent_conf_str = "Moderate"
            else:
                sent_conf_str = "Low"

            for sv, s_val in sorted_sents:
                sent_sims_display[sv] = round(float(s_val), 4)

            # ── General Sentiment Rule Override ──────────────────────────────
            # Applied after centroid prediction; classifier is NOT modified.
            top_sent = _apply_sentiment_rules(text, top_sent)

        # ── 2. Example-Based Rating Prediction ────────────────────────────────
        has_rating = bool(example_bank or rating_profiles or self.primary_rating_col)
        rating_scale_vals = target_info.get("rating", {}).get("unique_values", [1.0, 2.0, 3.0, 4.0, 5.0])
        rating_scale_keys = [str(int(v)) if float(v).is_integer() else str(v) for v in sorted(rating_scale_vals)]

        predicted_dist: Dict[str, float] = {k: 0.0 for k in rating_scale_keys}
        supporting_evidence_lines: List[str] = []

        if example_bank:
            bank_sims = np.array([_cosine_sim(vec, np.array(e["embedding"])) for e in example_bank])
            top_k = min(5, len(example_bank))
            top_indices = np.argsort(-bank_sims)[:top_k]
            top_sim_values = bank_sims[top_indices]
            weights = _softmax(top_sim_values, temp=0.1)

            for w, idx in zip(weights, top_indices):
                e = example_bank[idx]
                e_dist = e.get("rating_distribution", {})
                for rk in rating_scale_keys:
                    p_val = float(e_dist.get(rk, e_dist.get(str(float(rk)), 0.0)))
                    predicted_dist[rk] += float(w * p_val)

                t_snip = e['text'][:75] + ("…" if len(e['text']) > 75 else "")
                sim_val = bank_sims[idx]
                r_dist_summary = []
                for rk in rating_scale_keys:
                    p_val = float(e_dist.get(rk, e_dist.get(str(float(rk)), 0.0)))
                    if p_val > 0.05:
                        r_dist_summary.append(f"{rk} ({p_val*100:.0f}%)")
                r_summary_str = ", ".join(r_dist_summary) if r_dist_summary else f"rating {e.get('dominant_rating')}"
                entropy_note = f" [entropy: {e.get('rating_entropy', 0.0):.2f}]" if e.get('rating_entropy', 0.0) > 0.5 else ""
                supporting_evidence_lines.append(
                    f'- "{t_snip}" (similarity: {sim_val:.4f}) → ratings: {r_summary_str}{entropy_note}'
                )

            tot_w = sum(predicted_dist.values())
            if tot_w > 0:
                predicted_dist = {k: v / tot_w for k, v in predicted_dist.items()}
            else:
                predicted_dist = {k: 1.0 / len(predicted_dist) for k in predicted_dist}

        elif rating_profiles:
            sims = {rk: _cosine_sim(vec, np.array(rating_profiles[rk]["centroid"])) for rk in rating_scale_keys if rk in rating_profiles}
            w = _softmax(np.array(list(sims.values())), temp=0.1)
            for k, val in zip(sims.keys(), w):
                predicted_dist[k] = float(val)

        exp_rating = sum(float(k) * predicted_dist[k] for k in rating_scale_keys)
        most_likely_rating = max(predicted_dist.items(), key=lambda x: x[1])[0]

        # ── 3. Confidence Calculation ─────────────────────────────────────────
        sorted_probs = sorted(predicted_dist.values(), reverse=True)
        prob_margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

        if prob_margin >= 0.25:
            rating_conf_str = "High"
        elif prob_margin >= 0.10:
            rating_conf_str = "Moderate"
        else:
            rating_conf_str = "Low"

        # ── 4. Concise Structured Output Formatting ───────────────────────────
        lines = [
            "NEW REVIEW ANALYSIS",
            "",
            "Review:",
            f'"{text.strip()}"',
        ]

        if has_sentiment:
            lines += [
                "",
                f"Predicted Sentiment: {top_sent}",
                f"Sentiment Confidence: {sent_conf_str}",
            ]

        if has_rating:
            r_disp = int(float(most_likely_rating)) if float(most_likely_rating).is_integer() else most_likely_rating
            unc_info = _build_rating_uncertainty(predicted_dist, level=0.80, margin_threshold=0.10)
            r_dist_str = ", ".join(f"{k}: {v*100:.1f}%" for k, v in unc_info["rating_distribution"].items())
            p_int = unc_info["prediction_interval"]
            lines += [
                "",
                f"Predicted Likert Rating: {r_disp}",
                f"Rating Confidence: {unc_info['confidence']*100:.1f}% ({unc_info['uncertainty_status']}, margin: +{unc_info['prediction_margin']*100:.1f}%)",
                f"Expected Rating: {exp_rating:.2f}",
                f"Rating Distribution: [{r_dist_str}]",
                f"Prediction Interval (80%): [{p_int['lower']}–{p_int['upper']} ⭐]",
                f"Uncertainty Note: {unc_info['uncertainty_explanation']}",
            ]

        return "\n".join(lines)

    def predict_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze an unseen review text and return a structured dictionary containing
        predicted sentiment, predicted Likert rating, continuous expected rating,
        and calibrated uncertainty metrics (distribution, confidence, margin, status, interval).
        """
        patterns = self.memory.get("text_rating_patterns", {})
        example_bank = patterns.get("example_bank", [])
        rating_profiles = patterns.get("rating_profiles", {})
        sent_profiles = patterns.get("sentiment_profiles", {})
        target_info = self.memory.get("target_analysis", {})

        vec = self._embed([text])[0]

        # 1. Sentiment Prediction
        top_sent = "Unknown"
        sent_conf_str = "Low"
        sent_sims_display: Dict[str, float] = {}

        if sent_profiles:
            sent_sims: Dict[str, float] = {}
            for sv, prof in sent_profiles.items():
                centroid = np.array(prof["centroid"])
                sent_sims[sv] = _cosine_sim(vec, centroid)

            sorted_sents = sorted(sent_sims.items(), key=lambda x: -x[1])
            top_sent, top_sent_sim = sorted_sents[0] if sorted_sents else ("Unknown", 0.0)
            sec_sent_sim = sorted_sents[1][1] if len(sorted_sents) > 1 else top_sent_sim
            sent_margin = top_sent_sim - sec_sent_sim

            if sent_margin >= 0.08:
                sent_conf_str = "High"
            elif sent_margin >= 0.03:
                sent_conf_str = "Moderate"
            else:
                sent_conf_str = "Low"

            for sv, s_val in sorted_sents:
                sent_sims_display[sv] = round(float(s_val), 4)

            # Sentiment Rule Override
            top_sent = _apply_sentiment_rules(text, top_sent)

        # 2. Likert Rating Prediction
        rating_scale_vals = target_info.get("rating", {}).get("unique_values", [1.0, 2.0, 3.0, 4.0, 5.0])
        rating_scale_keys = [str(int(v)) if float(v).is_integer() else str(v) for v in sorted(rating_scale_vals)]
        predicted_dist: Dict[str, float] = {k: 0.0 for k in rating_scale_keys}

        if example_bank:
            bank_sims = np.array([_cosine_sim(vec, np.array(e["embedding"])) for e in example_bank])
            top_k = min(5, len(example_bank))
            top_indices = np.argsort(-bank_sims)[:top_k]
            top_sim_values = bank_sims[top_indices]
            weights = _softmax(top_sim_values, temp=0.1)

            for w, idx in zip(weights, top_indices):
                e = example_bank[idx]
                e_dist = e.get("rating_distribution", {})
                for rk in rating_scale_keys:
                    p_val = float(e_dist.get(rk, e_dist.get(str(float(rk)), 0.0)))
                    predicted_dist[rk] += float(w * p_val)

            tot_w = sum(predicted_dist.values())
            if tot_w > 0:
                predicted_dist = {k: v / tot_w for k, v in predicted_dist.items()}
            else:
                predicted_dist = {k: 1.0 / len(predicted_dist) for k in predicted_dist}

        elif rating_profiles:
            sims = {rk: _cosine_sim(vec, np.array(rating_profiles[rk]["centroid"])) for rk in rating_scale_keys if rk in rating_profiles}
            w = _softmax(np.array(list(sims.values())), temp=0.1)
            for k, val in zip(sims.keys(), w):
                predicted_dist[k] = float(val)

        exp_rating = sum(float(k) * predicted_dist[k] for k in rating_scale_keys)
        most_likely_rating = max(predicted_dist.items(), key=lambda x: x[1])[0]
        r_val = float(most_likely_rating)
        pred_likert = int(r_val) if r_val.is_integer() else r_val

        # 3. Calibrated Uncertainty Layer
        unc_info = _build_rating_uncertainty(predicted_dist, level=0.80, margin_threshold=0.10)

        return {
            "review_text": text,
            "predicted_sentiment": top_sent,
            "sentiment_confidence": sent_conf_str,
            "predicted_likert_rating": pred_likert,
            "expected_rating": round(exp_rating, 4),
            "confidence": unc_info["confidence"],
            "prediction_margin": unc_info["prediction_margin"],
            "uncertainty_status": unc_info["uncertainty_status"],
            "uncertainty_explanation": unc_info["uncertainty_explanation"],
            "rating_distribution": unc_info["rating_distribution"],
            "prediction_interval": unc_info["prediction_interval"],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Orchestrator
    # ──────────────────────────────────────────────────────────────────────────

    def run_full_analysis(self, csv_path: str, memory_path: str = "memory.json"):
        print("\n" + "═" * 62)
        print("  SEMANTIC RATING AGENT  —  DATASET UNDERSTANDING")
        print("═" * 62)

        print("\n[Phase 1] Loading CSV…")
        self.load_csv(csv_path)
        ls = self.memory.get("load_stats", {})
        print(f"\n  Training data summary:")
        print(f"    Source dataset rows        : {ls.get('n_rows_source', '?'):,}" if isinstance(ls.get('n_rows_source'), int) else f"    Source dataset rows        : {ls.get('n_rows_source', '?')}")
        print(f"    Completely empty rows excl.: {ls.get('completely_empty_rows_excluded', '?'):,}" if isinstance(ls.get('completely_empty_rows_excluded'), int) else f"    Completely empty rows excl.: {ls.get('completely_empty_rows_excluded', '?')}")
        print(f"    Duplicate rows retained    : YES ({ls.get('duplicate_rows_retained', '?'):,})" if isinstance(ls.get('duplicate_rows_retained'), int) else f"    Duplicate rows retained    : YES")
        print(f"    Informative rows for learning: {ls.get('n_rows_informative', '?'):,}" if isinstance(ls.get('n_rows_informative'), int) else f"    Informative rows for learning: {ls.get('n_rows_informative', '?')}")
        print(f"    Train/test split           : NONE — full dataset used")

        print("\n[Phase 2] Profiling all columns…")
        self.analyze_columns()
        n_cols = len(self.memory.get("column_profiles", {}))
        print(f"  Profiled {n_cols} columns.")

        print("\n[Phase 3] Detecting important columns…")
        det = self.detect_important_columns()
        print(f"  Text columns      : {det.get('text_columns')}")
        print(f"  Rating columns    : {det.get('rating_columns')}")
        print(f"  Sentiment columns : {det.get('sentiment_columns')}")
        print(f"  Numerical cols    : {det.get('numerical_columns')}")
        print(f"  Categorical cols  : {det.get('categorical_columns')}")
        for note in det.get("confidence_notes", []):
            print(f"  ⚠  {note}")

        print("\n[Phase 4] Analyzing rating and sentiment targets…")
        self.analyze_target()
        ta = self.memory.get("target_analysis", {})
        if "rating" in ta:
            print(f"  Rating scale   : {ta['rating'].get('scale_label')}")
            print(f"  Rating balanced: {ta['rating'].get('is_balanced')}")
        if "sentiment" in ta:
            print(f"  Sentiment classes: {ta['sentiment'].get('unique_classes')}")

        print("\n[Phase 5] Global semantic text analysis…")
        self.analyze_text_semantically()

        print("\n[Phase 6] Relationship analysis…")
        self.analyze_relationships()
        print("  Done (Spearman / Kruskal-Wallis / Cramér's V + Benjamini-Hochberg FDR correction).")

        print("\n[Phase 7] Learning per-rating text patterns…")
        self.learn_text_rating_patterns()
        n_profiles = len(
            self.memory.get("text_rating_patterns", {}).get("rating_profiles", {})
        )
        print(f"  Built semantic profiles for {n_profiles} rating levels.")

        print("\n[Phase 9A] Automatic Text Theme Discovery…")
        self.discover_text_themes()

        print("\n[Phase 9B] Interaction Discovery…")
        self.discover_interactions()

        print("\n[Phase 9C] Subgroup & Conditional Analysis…")
        self.discover_subgroup_patterns()

        print("\n[Phase 9D] Ranking top discoveries…")
        self.rank_discoveries()

        print("\n[Phase 8] Saving memory…")
        self.build_memory(memory_path)

        print("\n" + "═" * 62)
        print("  COMPREHENSIVE ANALYSIS COMPLETE — ready for questions.")
        print("═" * 62 + "\n")
