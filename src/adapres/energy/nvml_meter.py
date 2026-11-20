"""NVML meter (PLAN §7.2): GPU board energy counter (mJ) as the primary method, plus a
background thread sampling power/util/SM clock/memory/temperature. GPU board energy only.

`nvml` can be injected (tests use a fake); default is the pynvml module (nvidia-ml-py)."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager


class NvmlMeter:
    def __init__(self, device_index: int = 0, sample_hz: float = 50.0, nvml=None):
        if nvml is None:
            import pynvml as nvml  # noqa: N813
        self.nvml = nvml
        nvml.nvmlInit()
        self.h = nvml.nvmlDeviceGetHandleByIndex(device_index)
        self.period = 1.0 / sample_hz
        try:
            nvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
            self.energy_counter = True
        except Exception:  # noqa: BLE001 - fall back to power sampling, recorded in method
            self.energy_counter = False
        self.method = "nvml_energy_counter" if self.energy_counter else "nvml_power_sampling"

    def energy_mj(self) -> float | None:
        return float(self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.h)) if self.energy_counter else None

    def sample(self) -> dict:
        n, h = self.nvml, self.h
        s = {"t": time.perf_counter(), "power_mw": float(n.nvmlDeviceGetPowerUsage(h))}
        for key, fn in (("util_gpu", lambda: n.nvmlDeviceGetUtilizationRates(h).gpu),
                        ("sm_clock_mhz", lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)),
                        ("mem_used_mib", lambda: n.nvmlDeviceGetMemoryInfo(h).used / 2**20),
                        ("temp_c", lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU))):
            try:
                s[key] = float(fn())
            except Exception:  # noqa: BLE001
                s[key] = None
        return s

    def memory_used_mib(self) -> float:
        return self.nvml.nvmlDeviceGetMemoryInfo(self.h).used / 2**20

    def temperature(self) -> float:
        return float(self.nvml.nvmlDeviceGetTemperature(self.h, self.nvml.NVML_TEMPERATURE_GPU))

    @contextmanager
    def window(self):
        """Measure one window: yields a dict filled on exit with energy (counter and trapezoidal
        integral of sampled power), duration and samples."""
        out: dict = {"samples": []}
        stop = threading.Event()

        def loop():
            while not stop.is_set():
                out["samples"].append(self.sample())
                stop.wait(self.period)

        th = threading.Thread(target=loop, daemon=True)
        e0 = self.energy_mj()
        t0 = time.perf_counter()
        th.start()
        try:
            yield out
        finally:
            stop.set()
            th.join()
            t1 = time.perf_counter()
            e1 = self.energy_mj()
            out["T_s"] = t1 - t0
            out["energy_counter_mJ"] = (e1 - e0) if (e0 is not None and e1 is not None) else None
            sm = out["samples"]
            if len(sm) >= 2:
                integ = sum((b["t"] - a["t"]) * (a["power_mw"] + b["power_mw"]) / 2 for a, b in zip(sm, sm[1:], strict=False))
                out["energy_sampled_mJ"] = integ  # trapezoid of mW over s = mJ
            else:
                out["energy_sampled_mJ"] = None
            out["energy_mJ"] = out["energy_counter_mJ"] if out["energy_counter_mJ"] is not None else out["energy_sampled_mJ"]
            out["method"] = self.method
