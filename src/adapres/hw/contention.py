"""Controlled GPU contention (PLAN RQ6, prereg_qroute hardware_contention). KAGGLE-ONLY when used with CUDA.

A separate PROCESS (not a thread: no GIL sharing with the inference loop) runs square matrix
multiplications on the same GPU with a duty cycle d in [0, 1]: within each period it computes for d*period
seconds and sleeps for the rest. The level can be changed while it runs (shared Value).
"""

from __future__ import annotations

import multiprocessing as mp
import time


def _worker(level, stop, ready, period_s: float, size: int, device: str):
    import torch

    cuda = device.startswith("cuda")
    a = torch.randn(size, size, device=device)
    (a @ a).sum().item()                       # CUDA context + cuBLAS initialised before 'ready'
    ready.set()
    while not stop.is_set():
        d = float(level.value)
        t0 = time.perf_counter()
        if d > 0:
            while time.perf_counter() - t0 < d * period_s:
                a = (a @ a).clamp_(-1, 1)
                if cuda:
                    torch.cuda.synchronize()   # one kernel at a time: busy time tracks the duty cycle, no queue build-up
        rest = period_s - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)


def _sleep_worker(level, stop, ready, period_s: float, size: int, device: str):
    """CPU-only stand-in used by unit tests (no torch, no GPU)."""
    ready.set()
    while not stop.is_set():
        time.sleep(period_s)


class ContentionGenerator:
    """with ContentionGenerator() as g: g.set_level(0.5) ..."""

    def __init__(self, period_s: float = 0.02, size: int = 2048, device: str = "cuda", worker=None,
                 ready_timeout_s: float = 180.0):
        ctx = mp.get_context("spawn")
        self.level = ctx.Value("d", 0.0)
        self.stop = ctx.Event()
        self.ready = ctx.Event()
        self.ready_timeout_s = ready_timeout_s
        self.proc = ctx.Process(target=worker or _worker, args=(self.level, self.stop, self.ready, period_s, size, device),
                                daemon=True)

    def __enter__(self):
        self.proc.start()
        t0 = time.perf_counter()
        while not self.ready.wait(0.5):
            if not self.proc.is_alive():
                raise RuntimeError(f"contention process exited during start-up (exit code {self.proc.exitcode})")
            if time.perf_counter() - t0 > self.ready_timeout_s:
                self.__exit__()
                raise RuntimeError("contention process not ready within the timeout")
        return self

    def set_level(self, d: float):
        if not 0.0 <= d <= 1.0:
            raise ValueError("duty cycle must be in [0, 1]")
        self.level.value = float(d)

    def __exit__(self, *exc):
        self.stop.set()
        self.proc.join(timeout=10)
        if self.proc.is_alive():
            self.proc.terminate()
        return False
