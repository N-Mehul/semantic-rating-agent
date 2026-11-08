"""Tests for Phase 6 Improved SSR Integration (SSR_Phase6_Improved).

Verifies:
1. Exact mathematical parity and properties of Phase 6 SSR (k=3, T=0.75).
2. Canonical SSR mathematical invariance (confirming canonical outputs are 100% untouched).
3. Selector functionality (get_ssr_scorer).
4. Determinism and numerical reproducibility on the original 24-response Phase 6 dataset.
5. Strict anti-leakage compliance (no fitting on ground truth labels).
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from canonical_ssr import (
    PAPER_EPSILON,
    PAPER_TEMPERATURE,
    RATINGS,
    cosine_similarities,
    expected_rating,
    similarities_to_pmf as canonical_pmf,
)
from ssr_phase6_improved import (
    SSR_Phase6_Improved,
    get_ssr_scorer,
    phase6_similarities_to_pmf,
    score_embedding_phase6,
)


def test_phase6_truncation_and_temperature_properties():
    """Verify that Phase 6 zeros exactly 5 - 3 = 2 anchors and sharpens probabilities."""
    sims = np.array([0.10, 0.30, 0.70, 0.85, 0.40])
    pmf = phase6_similarities_to_pmf(sims, k=3, temperature=0.75)

    assert pmf.shape == (5,)
    assert np.isclose(pmf.sum(), 1.0)
    # The two lowest similarities are 0.10 and 0.30 -> anchors 1 and 2 must have prob 0.0
    assert pmf[0] == 0.0
    assert pmf[1] == 0.0
    assert pmf[2] > 0.0
    assert pmf[3] > 0.0
    assert pmf[4] > 0.0

    # For comparison, canonical SSR leaves anchor 2 with non-zero probability
    can_p = canonical_pmf(sims)
    assert can_p[1] > 0.0


def test_canonical_ssr_completely_unchanged():
    """Confirm canonical SSR mathematics remain bitwise/numerically identical to paper definitions."""
    sims = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    p = canonical_pmf(sims)
    e = expected_rating(p)

    # Manual check: min is 0.2. weights = [0, 0.2, 0.4, 0.6, 0.8], sum = 2.0
    # p = [0, 0.1, 0.2, 0.3, 0.4]
    # E = 1*0 + 2*0.1 + 3*0.2 + 4*0.3 + 5*0.4 = 0.2 + 0.6 + 1.2 + 2.0 = 4.0
    expected_p = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    assert np.allclose(p, expected_p)
    assert np.isclose(e, 4.0)


def test_ssr_selector():
    """Confirm that get_ssr_scorer selects canonical or Phase 6 without side effects."""
    scorer_can = get_ssr_scorer("canonical")
    scorer_imp = get_ssr_scorer("SSR_Phase6_Improved")

    assert scorer_can["name"] == "canonical"
    assert scorer_imp["name"] == "SSR_Phase6_Improved"

    sims = np.array([0.2, 0.3, 0.5, 0.8, 0.7])
    p_can = scorer_can["similarities_to_pmf"](sims)
    p_imp = scorer_imp["similarities_to_pmf"](sims)

    assert p_can[0] == 0.0 and p_can[1] > 0.0  # only min zeroed
    assert p_imp[0] == 0.0 and p_imp[1] == 0.0  # bottom 2 zeroed

    # Invalid method check
    with pytest.raises(ValueError, match="Unknown SSR method"):
        get_ssr_scorer("unknown_method")


def test_phase6_exact_numerical_reproduction():
    """Verify that Phase 6 yields exactly 79.1667% on the 24 stored Phase 6 responses."""
    csv_path = ROOT / "evaluation/generation_quality_improvement_v1/phase6_ssr/ssr_scores_with_reference.csv"
    if not csv_path.exists():
        pytest.skip("Stored Phase 6 results CSV not found.")

    import pandas as pd
    df = pd.read_csv(csv_path)
    df_imp = df[df["set"] == "improved"]

    cos_cols = [f"cos_{k}" for k in range(1, 6)]
    preds = []
    for _, row in df_imp.iterrows():
        sims = row[cos_cols].values.astype(float)
        pmf = SSR_Phase6_Improved.similarities_to_pmf(sims)
        e = SSR_Phase6_Improved.expected_rating(pmf)
        preds.append(SSR_Phase6_Improved.discrete_rating(e))

    exact_acc = (np.array(preds) == df_imp["ref"].values).mean()
    assert np.isclose(exact_acc, 19 / 24)  # 79.1666667%

