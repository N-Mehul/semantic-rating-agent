"""Test isolation: the suite must never append to the real dashboard run log.

Server-level tests build ``Workflow(<repo root>)`` with its default log directory, which is
data/dashboard_demo_runs/. ``DP_DASHBOARD_RUN_LOG_DIR`` redirects that default for every test, and a
session-level guard fails the run if the real log changed anyway.
"""
import hashlib
from pathlib import Path

import pytest

REAL_LOG = Path(__file__).resolve().parents[1] / "data" / "dashboard_demo_runs" / "runs.jsonl"


def _digest():
    return hashlib.sha256(REAL_LOG.read_bytes()).hexdigest() if REAL_LOG.exists() else None


@pytest.fixture(autouse=True)
def _redirect_dashboard_run_log(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("DP_DASHBOARD_RUN_LOG_DIR", str(tmp_path_factory.mktemp("dashboard_run_log")))


@pytest.fixture(scope="session", autouse=True)
def _real_run_log_untouched():
    before = _digest()
    yield
    assert _digest() == before, "the test session modified data/dashboard_demo_runs/runs.jsonl"
