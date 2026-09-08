"""Nsight Compute CLI wrapper — offline calibration only.  Not imported by service_main."""
from __future__ import annotations

import shutil
from typing import Dict, List, Optional


def ncu_available() -> bool:
    return shutil.which("ncu") is not None


def ncu_help() -> Dict:
    import subprocess
    exe = shutil.which("ncu")
    if not exe:
        return {"available": False, "reason": "ncu not on PATH"}
    p = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=15)
    return {
        "available": p.returncode == 0,
        "path": exe,
        "version": (p.stdout or p.stderr).strip().splitlines()[:3],
        "returncode": p.returncode,
    }


def ncu_metrics_offline(cmd: List[str], metrics: Optional[List[str]] = None) -> Dict:
    """Run ncu on a user command.  Never called from the online swap path."""
    import subprocess
    exe = shutil.which("ncu")
    if not exe:
        return {"available": False, "reason": "ncu not on PATH"}
    mets = metrics or [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active",
    ]
    args = [exe, "--metrics", ",".join(mets)] + cmd
    p = subprocess.run(args, capture_output=True, text=True, timeout=180)
    return {
        "available": True,
        "returncode": p.returncode,
        "stdout_tail": (p.stdout or "")[-2000:],
        "stderr_tail": (p.stderr or "")[-1000:],
    }
