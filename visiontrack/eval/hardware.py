"""Records the hardware/software context of an experiment."""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone

import psutil


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _gpu() -> dict | None:
    try:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        name, mem_total, mem_used, util, driver = (s.strip() for s in out.split(","))
        return {
            "name": name,
            "memory_total_mb": int(mem_total),
            "memory_used_mb_at_start": int(mem_used),
            "utilization_pct_at_start": int(util),
            "driver": driver,
        }
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def hardware_info() -> dict:
    info: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_model(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "ram_available_gb_at_start": round(psutil.virtual_memory().available / 2**30, 1),
        "load_avg_1m_at_start": round(os.getloadavg()[0], 2),
        "gpu": _gpu(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["ort_providers"] = ort.get_available_providers()
    except ImportError:
        pass
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        pass
    return info
