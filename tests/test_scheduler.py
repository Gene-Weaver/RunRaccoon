import multiprocessing as mp
import time

from runraccoon.plotting import scheduler as sch


def _sleep_forever(seconds):
    time.sleep(seconds)


def test_spawns_and_finish_survive_forked_workers(tmp_path, monkeypatch):
    """Regression: renders started from the scheduler thread must not hang when the main thread
    forks long-lived workers (PyTorch DataLoader). With Popen, an inherited pipe could block the
    thread - and finish() behind it - forever."""
    monkeypatch.setattr(sch, "_render_cmd", lambda *a, **k: ["/bin/sh", "-c", "sleep 0.2"])
    s = sch.PlotScheduler(tmp_path, tmp_path / "plots.log", every_s=0.0, enabled=True)
    workers = [mp.get_context("fork").Process(target=_sleep_forever, args=(30,), daemon=True) for _ in range(4)]
    try:
        for i in range(6):                                   # keep forking while renders are spawned
            s.notify({"loss": float(i)})
            workers[i % 4].start() if i < 4 else None
            time.sleep(0.6)
        t0 = time.time()
        assert s.finish(final=True, formats=("png",), status="finished", timeout=10)
        assert time.time() - t0 < 5
        assert not s._thread.is_alive()
    finally:
        for w in workers:
            if w.is_alive():
                w.kill()
