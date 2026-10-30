"""Digital Persona pipeline runner (human survey -> personas -> saved LLM responses -> SSR -> dataset -> validation).

Windows CMD, from the repository root:
    python scripts/run_pipeline.py status      # what exists / what is missing / commands for missing LLM generation
    python scripts/run_pipeline.py personas    # persona cards + respondent specs      -> data/persona_generator/
    python scripts/run_pipeline.py prompts     # construct-specific prompts (text only)  -> data/construct_prompts/
    python scripts/run_pipeline.py dataset     # ingest saved responses, SSR scoring   -> data/synthetic_dataset/
    python scripts/run_pipeline.py validate    # Validation Agent                      -> data/validation/, docs/project_status/
    python scripts/run_pipeline.py all         # personas -> prompts -> dataset -> validate -> status

The pipeline NEVER calls an LLM or any paid API. LLM generation is a separate, explicit step (see `status` for the
commands); `dataset` only reads responses that are already saved. Prerequisites that this runner does not rebuild:
data/human_survey_cleaned.csv (scripts/clean_human_survey.py), data/personas/ (scripts/build_personas.py) and the human
score files in data/pilot_embedding_comparison/ (scripts/ssr_embedding_comparison.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PREREQ = {
    "cleaned human survey": ("data/human_survey_cleaned.csv", "python scripts/clean_human_survey.py"),
    "persona assignments": ("data/personas/persona_assignments.csv", "python scripts/build_personas.py"),
    "persona profiles": ("data/personas/persona_profiles.json", "python scripts/build_personas.py"),
    "human SSR scores (3 encoders)": ("data/pilot_embedding_comparison/human_scores_local_bge_base.csv", "python scripts/ssr_embedding_comparison.py"),
    "approved anchors": ("docs/survey_instrument/smartphone_construct_anchor_statements.csv", "(frozen input; do not edit)"),
}
STAGE_OUTPUTS = {
    "persona generator": "data/persona_generator/respondent_specs.jsonl",
    "construct prompts": "data/construct_prompts/construct_registry.json",
    "synthetic dataset": "data/synthetic_dataset/synthetic_survey_dataset.csv",
    "validation report": "docs/project_status/validation_agent_report.md",
}
GENERATE_COMMANDS = """\
Missing LLM responses are generated separately (the pipeline never does this for you), in the SAME CMD window:
  Gemini (resumes, skips finished prompts):
      set GEMINI_API_KEY=your-key
      python scripts/test_gemini_api.py
      python scripts/pilot_generate_gemini.py
  OpenAI (choose a model your account can use; the script checks it first):
      set OPENAI_API_KEY=your-key
      python scripts/pilot_generate_openai.py --model YOUR-MODEL-ID
  Construct-specific questions (separate folders; then register/score with --sources gemini_FT_PE01 etc.):
      python scripts/pilot_generate_gemini.py --prompts-file data/construct_prompts/FT_PE01_prompts.jsonl --out-dir data/pilot_gemini_FT_PE01
      python scripts/pilot_generate_gemini.py --prompts-file data/construct_prompts/FT_BI01_prompts.jsonl --out-dir data/pilot_gemini_FT_BI01
Then:  python scripts/run_pipeline.py dataset   and   python scripts/run_pipeline.py validate"""


def check_prereqs() -> list[str]:
    return [f"missing {name}: {path}  ->  {cmd}" for name, (path, cmd) in PREREQ.items() if not (ROOT / path).exists()]


def status(print_fn=print) -> dict:
    from dp_synthetic_dataset import SOURCES, read_jsonl
    out = {"prereq_missing": check_prereqs(), "stages": {}, "sources": {}}
    print_fn("== Pipeline status ==")
    for name, rel in STAGE_OUTPUTS.items():
        ok = (ROOT / rel).exists()
        out["stages"][name] = ok
        print_fn(f"[{'ok' if ok else '--'}] {name}: {rel}")
    for m in out["prereq_missing"]:
        print_fn("[!!] " + m)
    print_fn("== Saved LLM responses (separate experiments) ==")
    for key, s in SOURCES.items():
        d = ROOT / s["dir"]
        n = len(read_jsonl(d / s["responses"]))
        planned = len(read_jsonl(d / "prompts.jsonl"))
        out["sources"][key] = {"responses": n, "planned": planned, "enabled": s["enabled"]}
        print_fn(f"{key:10s} {s['provider']:13s} responses={n:3d}/{planned or '?'}  enabled={s['enabled']}  {s['note']}")
    print_fn(GENERATE_COMMANDS)
    return out


def run_stage(name: str) -> int:
    missing = check_prereqs()
    if missing and name != "status":
        print("Cannot run: prerequisites are missing.\n  " + "\n  ".join(missing))
        return 2
    if name == "status":
        status()
        return 0
    if name == "personas":
        import dp_persona_generator as m
        return m.main()
    if name == "prompts":
        import dp_constructs as m
        return m.main()
    if name == "dataset":
        import dp_synthetic_dataset as m
        sys.argv = [sys.argv[0]]
        return m.main()
    if name == "validate":
        import dp_validation_agent as m
        return m.main()
    raise ValueError(name)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "status"
    stages = {"all": ["personas", "prompts", "dataset", "validate", "status"]}.get(cmd, [cmd])
    if cmd not in ("status", "personas", "prompts", "dataset", "validate", "all"):
        print(__doc__)
        return 2
    for st in stages:
        print(f"\n=== {st} ===")
        rc = run_stage(st)
        if rc != 0:
            print(f"Stage '{st}' stopped with exit code {rc}; later stages were not run.")
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
