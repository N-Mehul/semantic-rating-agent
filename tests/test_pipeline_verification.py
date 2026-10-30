import re
from pathlib import Path

from scripts import verify_pipeline_outputs as vp

ROOT = Path(__file__).resolve().parents[1]


def readme_section():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    start = text.index("## Digital Persona Pipeline")
    return text[start: text.index("## Default Showcase Result")]


def test_verifier_passes_on_real_outputs():
    results = vp.run_checks()
    failed = [n for n, ok, d in results if not ok]
    assert not failed and len(results) >= 35


def test_verifier_reports_missing_outputs_without_raising(tmp_path):
    results = vp.run_checks(tmp_path)
    assert len(results) == 1 and results[0][1] is False and "missing" in results[0][2]


def test_readme_documented_scripts_exist():
    sec = readme_section()
    paths = set(re.findall(r"scripts/[A-Za-z0-9_]+\.py", sec))
    assert {"scripts/run_pipeline.py", "scripts/dp_persona_generator.py", "scripts/dp_synthetic_dataset.py",
            "scripts/dp_validation_agent.py", "scripts/pilot_generate_gemini.py", "scripts/pilot_generate_openai.py",
            "scripts/test_gemini_api.py"} <= paths
    assert all((ROOT / p).exists() for p in paths)


def test_readme_pipeline_commands_are_valid_stages():
    sec = readme_section()
    stages = set(re.findall(r"run_pipeline\.py (\w+)", sec))
    from scripts import run_pipeline as rp
    valid = {"status", "all", "dataset", "personas", "prompts", "validate"}
    assert stages <= valid and {"status", "all", "prompts"} <= stages
    assert all(f"run_pipeline.py {s}" in rp.__doc__ for s in stages)          # every documented stage is described by the runner itself
    assert "python -m pytest --confcutdir=tests tests/" in sec


def test_readme_setup_lists_packages_missing_from_requirements():
    sec = readme_section()
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for pkg in ("matplotlib", "openpyxl"):
        if pkg not in req:                       # requirements.txt is not guaranteed to list them
            assert f" {pkg}" in sec.split("### Commands")[0], f"README setup must install {pkg}"


def test_readme_states_key_limitations():
    sec = readme_section()
    for phrase in ("P1 and P2", "P3 and P4 have none", "does not", "never"):
        assert phrase in sec
