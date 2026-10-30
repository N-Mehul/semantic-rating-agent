"""Regression: running the dashboard workflow through its default constructor must not touch the real run log."""
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL_LOG = ROOT / "data" / "dashboard_demo_runs" / "runs.jsonl"
sys.path.insert(0, str(ROOT))

from dashboard import workflow as wf  # noqa: E402


def _digest():
    return hashlib.sha256(REAL_LOG.read_bytes()).hexdigest() if REAL_LOG.exists() else None


def test_default_log_dir_is_redirected_during_tests():
    w = wf.Workflow(ROOT)                                        # no log_dir argument, as the server creates it
    assert w.log_dir != ROOT / "data" / "dashboard_demo_runs"
    assert Path(os.environ["DP_DASHBOARD_RUN_LOG_DIR"]) == w.log_dir


def test_logging_through_default_workflow_leaves_real_log_unchanged():
    before = _digest()
    w = wf.Workflow(ROOT)
    w._log("isolation-test", {"text": "synthetic test entry"})
    assert (w.log_dir / "runs.jsonl").exists()                   # it was written, but to the temporary directory
    assert _digest() == before


def test_env_override_is_optional(monkeypatch):
    monkeypatch.delenv("DP_DASHBOARD_RUN_LOG_DIR")
    assert wf.Workflow(ROOT).log_dir == ROOT / "data" / "dashboard_demo_runs"
    assert wf.Workflow(ROOT, log_dir=ROOT / "x").log_dir == ROOT / "x"
