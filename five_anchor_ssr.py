"""
five_anchor_ssr.py — Isolated Five-Likert-Anchor Semantic Similarity Rating (SSR) Module.

This module provides reusable canonical Likert anchors and isolated SSR scoring
for customer review satisfaction constructs without modifying existing production paths.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from agent import SemanticRatingAgent, _cosine_sim, _softmax


def get_canonical_mobile_anchors() -> Dict[int, str]:
    """
    Returns the five canonical Likert anchor statements representing the
    construct of mobile phone review customer satisfaction.
    
    1 = Strongly Disagree (extreme dissatisfaction, poor experience)
    2 = Disagree (mild dissatisfaction, multiple flaws)
    3 = Neutral (average experience, balanced positive and negative trade-offs)
    4 = Agree (solid satisfaction, performs well despite minor limitations)
    5 = Strongly Agree (extreme satisfaction, outstanding, top-tier excellence)
    """
    return {
        1: "I am very dissatisfied with this phone overall and feel that its performance and experience are poor.",
        2: "I am somewhat dissatisfied with this phone overall and feel that several aspects could be better.",
        3: "My experience with this phone is average; it has both positive and negative aspects and I feel neither satisfied nor dissatisfied overall.",
        4: "I am satisfied with this phone overall and think it performs well, despite some minor limitations.",
        5: "I am extremely satisfied with this phone overall and consider it an excellent phone with very few meaningful problems."
    }


def get_canonical_anchors(construct: str = "mobile_phone_satisfaction") -> Dict[int, str]:
    """
    Centralized factory for canonical anchors across different rating constructs.
    Supports future extension to other domains while preserving a single source of truth.
    """
    construct_lower = construct.strip().lower()
    if construct_lower in ("mobile_phone_satisfaction", "mobile", "phone", "reviews"):
        return get_canonical_mobile_anchors()
    else:
        # Fallback to general satisfaction construct
        return {
            1: "I strongly disagree and am very dissatisfied with this product overall.",
            2: "I disagree and am somewhat dissatisfied with this product overall.",
            3: "I am neutral; my experience is average with both positive and negative aspects.",
            4: "I agree and am satisfied with this product overall.",
            5: "I strongly agree and am extremely satisfied with this product overall."
        }


def embed_canonical_anchors(agent: SemanticRatingAgent, anchors: Optional[Dict[int, str]] = None) -> Dict[int, np.ndarray]:
    """
    Embeds the 5 canonical anchors using the agent's validated embedding model.
    Returns a dictionary mapping rating integer (1..5) to normalized 1-D embedding vector.
    """
    if anchors is None:
        anchors = get_canonical_mobile_anchors()
    
    ordered_keys = sorted(anchors.keys())
    texts = [anchors[k] for k in ordered_keys]
    embeddings = agent._embed(texts)
    
    anchor_embs = {}
    for k, emb in zip(ordered_keys, embeddings):
        vec = np.array(emb, dtype=float)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        anchor_embs[k] = vec
    return anchor_embs


def score_text_against_anchors(
    agent: SemanticRatingAgent,
    text: str,
    anchor_embeddings: Dict[int, np.ndarray],
    temperature: float = 0.1
) -> Dict[str, Any]:
    """
    Evaluates one review text against the 5 canonical anchor embeddings:
      1. response_embedding = embed(text)
      2. similarity_i = cosine_similarity(response_embedding, anchor_embedding_i)
      3. P_i = softmax(similarity_i / temperature)
      4. E[R] = sum(P_i * i)
      5. predicted_rating = argmax(P_i)
    """
    text_emb = agent._embed([text])[0]
    vec = np.array(text_emb, dtype=float)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
        
    ordered_ratings = sorted(anchor_embeddings.keys())
    similarities = [float(_cosine_sim(vec, anchor_embeddings[r])) for r in ordered_ratings]
    
    sim_array = np.array(similarities, dtype=float)
    probs = _softmax(sim_array, temp=temperature)
    
    expected_rating = float(sum(p * r for p, r in zip(probs, ordered_ratings)))
    predicted_rating = int(ordered_ratings[int(np.argmax(probs))])
    
    return {
        "text": text,
        "similarities": {r: round(similarities[i], 4) for i, r in enumerate(ordered_ratings)},
        "probabilities": {r: round(float(probs[i]), 4) for i, r in enumerate(ordered_ratings)},
        "expected_rating": round(expected_rating, 4),
        "predicted_rating": predicted_rating
    }
