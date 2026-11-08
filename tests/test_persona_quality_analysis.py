"""
Automated validation tests for the 400 synthetic response quality analysis.
"""

import csv
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
QUALITY_DIR = ROOT / "data" / "experiments" / "persona_validation_400" / "quality_analysis"
TABLES_DIR = QUALITY_DIR / "tables"
FIGURES_DIR = QUALITY_DIR / "figures"
REPORT_PATH = QUALITY_DIR / "PERSONA_CONSISTENCY_AND_QUALITY_REPORT.md"


def test_quality_analysis_files_exist():
    assert QUALITY_DIR.exists(), "Quality analysis directory missing"
    assert TABLES_DIR.exists(), "Tables directory missing"
    assert FIGURES_DIR.exists(), "Figures directory missing"
    assert REPORT_PATH.exists(), "Research report markdown missing"


def test_table_files_exist_and_valid():
    expected_tables = [
        "dataset_overview.csv",
        "persona_construct_matrix.csv",
        "text_length_statistics.csv",
        "lexical_diversity_metrics.csv",
        "persona_distinctive_terms.csv",
        "persona_centroid_cosine_similarity.csv",
        "construct_distinctive_terms.csv",
        "pairwise_similarity_summary.csv",
        "top_similar_pairs.csv",
    ]
    for table_name in expected_tables:
        p = TABLES_DIR / table_name
        assert p.exists(), f"Missing table: {table_name}"
        assert p.stat().st_size > 50, f"Table {table_name} is too small / empty"


def test_figures_exist_and_valid_png():
    expected_figures = [
        "fig1_response_counts_by_persona.png",
        "fig2_persona_construct_heatmap.png",
        "fig3_response_length_distribution.png",
        "fig4_persona_distinctive_terms.png",
        "fig5_semantic_similarity_matrix.png",
    ]
    for fig_name in expected_figures:
        p = FIGURES_DIR / fig_name
        assert p.exists(), f"Missing figure: {fig_name}"
        assert p.stat().st_size > 10000, f"Figure {fig_name} is suspiciously small (<10KB)"
        # Verify PNG signature
        header = p.read_bytes()[:8]
        assert header == b"\x89PNG\r\n\x1a\n", f"Figure {fig_name} is not a valid PNG file"


def test_dataset_overview_metrics():
    overview_path = TABLES_DIR / "dataset_overview.csv"
    df = pd.read_csv(overview_path)
    assert len(df) == 4, f"Expected 4 personas, found {len(df)}"
    for pid in ["P1", "P2", "P3", "P4"]:
        row = df[df["Persona_ID"] == pid].iloc[0]
        assert row["Total_Responses"] == 100, f"Persona {pid} count is {row['Total_Responses']}, expected 100"


def test_pairwise_similarity_bounds():
    sim_summary_path = TABLES_DIR / "pairwise_similarity_summary.csv"
    df = pd.read_csv(sim_summary_path)
    lookup = dict(zip(df["Metric"], df["Value"]))

    total_pairs = float(lookup["Total_Pairwise_Comparisons"])
    assert total_pairs == 79800.0, f"Expected 79800 pairs, found {total_pairs}"

    mean_sim = float(lookup["Mean_Pairwise_Cosine_Similarity"])
    assert 0.005 <= mean_sim <= 0.25, f"Mean similarity {mean_sim} out of expected bounds [0.005, 0.25]"

    max_sim = float(lookup["Maximum_Pairwise_Cosine_Similarity"])
    assert max_sim <= 1.0, f"Maximum similarity {max_sim} exceeds 1.0"
    p95_sim = float(lookup["95th_Percentile_Cosine_Similarity"])
    assert p95_sim < 0.20, f"95th percentile similarity {p95_sim} indicates high corpus-wide redundancy"

    # Verify that near-duplicate detection table correctly records top pairs
    top_pairs_path = TABLES_DIR / "top_similar_pairs.csv"
    top_df = pd.read_csv(top_pairs_path)
    assert len(top_df) == 20, "Expected top 20 similar pairs logged"
    assert top_df["Cosine_Similarity"].max() == max_sim


def test_text_length_and_diversity_bounds():
    stats_path = TABLES_DIR / "text_length_statistics.csv"
    stats_df = pd.read_csv(stats_path)
    all_row = stats_df[stats_df["Segment"] == "ALL_RESPONSES"].iloc[0]

    assert 40.0 <= float(all_row["Word_Mean"]) <= 70.0, f"Word mean {all_row['Word_Mean']} out of expected range"
    assert 2.5 <= float(all_row["Sentence_Mean"]) <= 3.5, f"Sentence mean {all_row['Sentence_Mean']} out of range"

    div_path = TABLES_DIR / "lexical_diversity_metrics.csv"
    div_df = pd.read_csv(div_path)
    all_div = div_df[div_df["Segment"] == "ALL_400_RESPONSES"].iloc[0]

    assert float(all_div["Type_Token_Ratio"]) >= 0.10, "TTR is suspiciously low"
    assert int(all_div["Unique_Tokens"]) >= 1500, "Unique vocabulary count is suspiciously low"
