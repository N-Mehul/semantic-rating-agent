"""API-independent helpers for SSR evaluation and API-generation safety checks (unit-tested)."""
from __future__ import annotations

import re

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

# Words/patterns that would leak an outcome or ask for a rating inside a generation prompt.
FORBIDDEN_PROMPT_PATTERNS = [r"\bQ21\b", r"\blikert\b", r"out of 5", r"\b[1-5]\s*/\s*5\b",
                             r"likely to buy the same", r"rebuy_brand", r"expected rating"]
# A generated answer must not itself contain a numeric rating.
RATING_IN_TEXT = re.compile(r"(\b\d+(\.\d+)?\s*/\s*\d+\b)|(\b\d+(\.\d+)?\s*(out of|stars?)\b)|(\b(rate|rating|score)\w*\s*[:=]?\s*\d)",
                            re.IGNORECASE)


def prompt_leak_findings(prompt: str) -> list[str]:
    return [p for p in FORBIDDEN_PROMPT_PATTERNS if re.search(p, prompt, re.IGNORECASE)]


def response_contains_rating(text: str) -> bool:
    return bool(RATING_IN_TEXT.search(text))


TERMINAL_PUNCT = re.compile(r"[.!?\"')\]…”’]$")
OK_FINISH = {"STOP", "FINISH_REASON_STOP"}


def response_quality(text, finish_reason=None) -> dict:
    """Quality flags for one generated answer. Nothing is dropped here: callers keep every record and decide.

    finish_reason: the API's reason string (e.g. 'STOP', 'MAX_TOKENS') or None when unknown (older pilots).
    is_complete is False for empty text, a non-STOP finish reason, or text not ending in terminal punctuation.
    """
    t = (text or "").strip()
    fr = None if finish_reason is None else str(finish_reason).upper()
    flags = {
        "empty": not t,
        "finish_reason_not_stop": fr is not None and fr not in OK_FINISH,
        "no_terminal_punctuation": bool(t) and not TERMINAL_PUNCT.search(t),
        "too_short_lt3_words": bool(t) and len(t.split()) < 3,
        "contains_rating": bool(t) and response_contains_rating(t),
    }
    flags["is_complete"] = not (flags["empty"] or flags["finish_reason_not_stop"] or flags["no_terminal_punctuation"])
    flags["safe_for_ssr"] = flags["is_complete"] and not flags["contains_rating"]
    return flags


def split_by_quality(records: list[dict]):
    """Split saved generation records into (passed, excluded). Quality is recomputed from text + finish_reason
    (so older records without a stored `quality` field are handled the same way); duplicates keep the first record.
    Excluded records are returned with their reasons, never silently dropped."""
    passed, excluded, seen = [], [], set()
    for r in records:
        sid = r.get("synthetic_id")
        if sid in seen:
            excluded.append({**r, "exclusion_reasons": ["duplicate_id"]})
            continue
        seen.add(sid)
        q = response_quality(r.get("response_text"), r.get("finish_reason"))
        reasons = [k for k in ("empty", "finish_reason_not_stop", "no_terminal_punctuation", "contains_rating") if q[k]]
        (excluded if reasons else passed).append({**r, "exclusion_reasons": reasons} if reasons else r)
    return passed, excluded


def validate_responses(rows: list[dict]) -> list[str]:
    """Return a list of problems (empty = valid)."""
    problems, seen = [], set()
    for r in rows:
        sid = r.get("synthetic_id")
        if not sid or sid in seen:
            problems.append(f"missing/duplicate id: {sid}")
        seen.add(sid)
        t = r.get("response_text")
        if not isinstance(t, str) or not t.strip():
            problems.append(f"{sid}: empty response")
        elif response_contains_rating(t):
            problems.append(f"{sid}: contains numeric rating")
    return problems


def boot_ci(fn, arrays, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = fn(*[a[idx] for a in arrays])
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    if not vals:  # e.g. constant predictor: correlation undefined, report NaN rather than invent a value
        return [float('nan'), float('nan')]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def metrics_vs_q21(expected: np.ndarray, argmax: np.ndarray, q21: np.ndarray, n_boot=2000, seed=0) -> dict:
    """Spearman (+CI), MAE, exact-match accuracy and the mean / majority-class baselines."""
    expected, argmax, q21 = (np.asarray(a, dtype=float) for a in (expected, argmax, q21))
    rho = spearmanr(expected, q21).statistic
    majority = pd_mode(q21)
    return {
        "n": int(len(q21)),
        "spearman": float(rho),
        "spearman_ci95": boot_ci(lambda a, b: spearmanr(a, b).statistic, [expected, q21], n_boot, seed),
        "mae": float(np.abs(expected - q21).mean()),
        "mae_baseline_mean": float(np.abs(q21.mean() - q21).mean()),
        "accuracy_exact": float((argmax == q21).mean()),
        "accuracy_baseline_majority": float((q21 == majority).mean()),
        "q21_distribution": {int(k): int((q21 == k).sum()) for k in range(1, 6)},
    }


def pd_mode(x: np.ndarray) -> float:
    vals, counts = np.unique(x, return_counts=True)
    return float(vals[np.argmax(counts)])


def dev_test_split(q21: np.ndarray, test_size=0.5, seed=0):
    """Respondent-level, stratified-by-Q21 split. Returns (dev_idx, test_idx), disjoint."""
    idx = np.arange(len(q21))
    dev, test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=q21)
    return np.sort(dev), np.sort(test)


def check_pmf(p: np.ndarray, tol=1e-9) -> bool:
    p = np.asarray(p, dtype=float)
    return bool(p.shape[-1] == 5 and np.all(p >= -tol) and np.allclose(p.sum(axis=-1), 1.0, atol=tol))
