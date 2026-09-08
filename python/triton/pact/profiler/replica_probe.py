"""L2 collection: CUDA-event median of 1–3 replica launches.  Always numeric."""
from __future__ import annotations

import statistics
from typing import Callable, Dict


def replica_median_us(launch_fn: Callable, iters: int = 3) -> Dict:
    import torch
    times = []
    # warmup outside the timed set
    launch_fn()
    torch.cuda.synchronize()
    for _ in range(max(1, iters)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch_fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return {
        "replica_median_us": statistics.median(times),
        "replica_iters": len(times),
        "source": "cuda_event",
    }
