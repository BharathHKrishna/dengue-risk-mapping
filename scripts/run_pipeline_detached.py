"""Run the full pipeline unattended (survives closing the editor when started detached).

Runs the two data pulls in parallel (both resume from their caches), then every
later stage in order. Stops at the first failing stage. Progress goes to
``logs/pipeline.log``; each stage also has its own log in ``logs/``.

Start detached on Windows (PowerShell):
    Start-Process -WindowStyle Hidden .venv\\Scripts\\python.exe scripts\\run_pipeline_detached.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
PY = sys.executable
PARALLEL = ["cases", "gee_brazil"]
SEQUENCE = ["features", "temporal", "spatial", "explain", "final", "transfer", "export"]


def note(msg: str) -> None:
    """Append a timestamped line to the pipeline log."""
    with open(LOGS / "pipeline.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def start(stage: str) -> subprocess.Popen:
    """Launch one CLI stage with its own log file."""
    log = open(LOGS / f"stage_{stage}.log", "a", encoding="utf-8")  # noqa: SIM115
    return subprocess.Popen(
        [PY, "-m", "dengue.cli", stage], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
    )


def main() -> int:
    """Run parallel pulls, then the sequential stages."""
    LOGS.mkdir(exist_ok=True)
    note(f"pipeline started (pid of runner: {__import__('os').getpid()})")
    procs = {s: start(s) for s in PARALLEL}
    while any(p.poll() is None for p in procs.values()):
        time.sleep(30)
    for s, p in procs.items():
        note(f"stage {s} exit code {p.returncode}")
        if p.returncode != 0:
            note("STOPPED: a data pull failed, see its stage log")
            return 1
    for s in SEQUENCE:
        note(f"stage {s} started")
        rc = start(s).wait()
        note(f"stage {s} exit code {rc}")
        if rc != 0:
            note(f"STOPPED at stage {s}, see logs/stage_{s}.log")
            return 1
    note("ALL STAGES DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
