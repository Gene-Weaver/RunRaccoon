import os

import pytest


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Each test gets its own working dir and registry; no dashboard, no live re-renders."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUNRACCOON_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("RUNRACCOON_DASHBOARD", "0")
    monkeypatch.setenv("RUNRACCOON_LIVE_PLOTS", "0")
    monkeypatch.setenv("RUNRACCOON_FINAL_PLOTS", "0")
    monkeypatch.setenv("RUNRACCOON_QUIET", "1")
    monkeypatch.setenv("RUNRACCOON_CONSOLE", "off")
    for k in [k for k in os.environ if k.startswith("WANDB_") and k != "WANDB_MODE"]:
        monkeypatch.delenv(k, raising=False)
    yield tmp_path
    import runraccoon
    while runraccoon.run is not None:
        runraccoon.run.finish()
