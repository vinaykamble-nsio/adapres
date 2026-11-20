"""NVML meter + protocol with a fake NVML; live controller (sequential and pipelined) must make
exactly the same decisions as offline replay on the same detections."""

import time
import types

import numpy as np
import pytest

from adapres.controller import ControllerParams
from adapres.energy import protocol as P
from adapres.energy.nvml_meter import NvmlMeter
from adapres.features import FrameFeaturizer
from adapres.live import AdaptiveStepper, FixedStepper, run_live_sequences
from adapres.replay import replay


class FakeNvml(types.SimpleNamespace):
    """Energy counter = 50 W * elapsed (+ 10 W extra while 'busy' is set), mJ."""

    def __init__(self, counter=True):
        super().__init__(NVML_CLOCK_SM=0, NVML_TEMPERATURE_GPU=0)
        self.t0, self.counter, self.busy_mj, self.mem = time.perf_counter(), counter, 0.0, 300.0

    def nvmlInit(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetTotalEnergyConsumption(self, h):
        if not self.counter:
            raise RuntimeError("not supported")
        return (time.perf_counter() - self.t0) * 50_000 + self.busy_mj

    def nvmlDeviceGetPowerUsage(self, h):
        return 50_000

    def nvmlDeviceGetUtilizationRates(self, h):
        return types.SimpleNamespace(gpu=90)

    def nvmlDeviceGetClockInfo(self, h, k):
        return 1590

    def nvmlDeviceGetMemoryInfo(self, h):
        return types.SimpleNamespace(used=self.mem * 2**20)

    def nvmlDeviceGetTemperature(self, h, k):
        return 60


class FakeBackend:
    def __init__(self, name, ms, dets_for, nvml=None, mj=0.0):
        self.name, self.ms, self.dets_for, self.nvml, self.mj = name, ms, dets_for, nvml, mj

    def predict(self, img):
        time.sleep(self.ms / 1000)
        if self.nvml is not None:
            self.nvml.busy_mj += self.mj
        return self.dets_for(img), {"infer_ms": self.ms}


def test_meter_window_counter_and_sampling():
    nv = FakeNvml()
    m = NvmlMeter(nvml=nv, sample_hz=200)
    with m.window() as w:
        time.sleep(0.2)
    assert m.method == "nvml_energy_counter"
    assert w["energy_counter_mJ"] == pytest.approx(50 * w["T_s"] * 1000, rel=0.05)
    assert w["energy_sampled_mJ"] == pytest.approx(w["energy_counter_mJ"], rel=0.15)
    m2 = NvmlMeter(nvml=FakeNvml(counter=False), sample_hz=200)
    with m2.window() as w2:
        time.sleep(0.2)
    assert m2.method == "nvml_power_sampling" and w2["energy_counter_mJ"] is None and w2["energy_mJ"] > 0


def _frames(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        img = np.full((18, 32, 3), int(rng.integers(0, 255)), np.uint8)
        img[:, : int(rng.integers(2, 30))] = 255
        img[0, 0, 0] = k % 256  # frame tag for the fake detectors
        out.append(img)
    return out


def _dets_by_tag(level):
    def f(img):
        k = int(img[0, 0, 0])
        n = (k * 7 + (3 if level == "fp16" else 0)) % 9
        return np.column_stack([np.zeros((n, 4)), np.full(n, 0.6), np.full(n, 2.0)]) if n else np.zeros((0, 6))
    return f


NORM = {"m": (0.0, 0.3), "c": (0.0, 0.5), "n": (0.0, 8.0), "u": (0.0, 0.6)}


def _params():
    return ControllerParams(weights={"m": 0.25, "c": 0.25, "n": 0.25, "u": 0.25, "s": 0.0}, theta=0.35, norm=NORM,
                            K_up=2, K_down=2, H=1, theta_sc=0.5)


def test_live_sequential_pipelined_and_replay_agree():
    frames = _frames(40)
    starts = [0, 25]
    be = {lv: FakeBackend(lv, 0.5, _dets_by_tag(lv)) for lv in ("int8", "fp16")}
    ff = lambda: FrameFeaturizer(width=16)  # noqa: E731
    seq = run_live_sequences(be, _params(), ff, frames, starts, 0.25, 0.0, pipelined=False)
    pip = run_live_sequences(be, _params(), ff, frames, starts, 0.25, 0.0, pipelined=True)
    assert [r["level"] for r in seq] == [r["level"] for r in pip]
    assert len({r["level"] for r in seq}) == 2  # the test exercises both levels
    # offline replay on the same (precomputed) cues and detections
    ids = list(range(40))
    cues = {}
    for s0, s1 in ((0, 25), (25, 40)):
        fz = ff()
        for i in range(s0, s1):
            cues[i] = fz(frames[i])
    dets = {lv: {i: _dets_by_tag(lv)(frames[i]) for i in ids} for lv in ("int8", "fp16")}
    rep = replay(_params(), {"a": ids[:25], "b": ids[25:]}, cues, dets, 0.25, 0.0)
    assert [r["level"] for r in rep] == [r["level"] for r in seq]


def test_protocol_switch_penalty_and_memory():
    nv = FakeNvml()
    meter = NvmlMeter(nvml=nv, sample_hz=100)
    frames = _frames(10)
    be = {"int8": FakeBackend("int8", 1.0, _dets_by_tag("int8"), nv, mj=5.0),
          "fp16": FakeBackend("fp16", 2.0, _dets_by_tag("fp16"), nv, mj=12.0)}
    steppers = {"fixed_int8": FixedStepper(be["int8"], frames), "fixed_fp16": FixedStepper(be["fp16"], frames),
                "adaptive_sequential": AdaptiveStepper(be, _params(), lambda: FrameFeaturizer(width=16), frames, [0],
                                                       0.25, 0.0, pipelined=False)}
    out = P.run_protocol(steppers, meter, warmup_s=0.02, settle_s=0.0, idle_s=0.05, measure_s=0.15, repeats=2,
                         seed=1, log=lambda *a: None)
    s = out["summary"]
    assert set(s) == set(steppers) and len(out["runs"]) == 6
    # dynamic energy per frame recovers the injected per-frame energy (idle-subtracted)
    assert s["fixed_int8"]["E_dyn_mJ_per_frame"]["median"] == pytest.approx(5.0, rel=0.25)
    assert s["fixed_fp16"]["E_dyn_mJ_per_frame"]["median"] == pytest.approx(12.0, rel=0.25)
    assert s["fixed_int8"]["throughput_fps"]["median"] > s["fixed_fp16"]["throughput_fps"]["median"]
    assert 0.0 <= s["adaptive_sequential"]["frac_int8"] <= 1.0
    sp = P.switch_penalty(be, frames, n_frames=10, repeats=1)
    assert sp["penalty_ms"] == pytest.approx(0.0, abs=1.5)
    loaded = []

    def load(lv):
        nv.mem += {"fp32": 100, "fp16": 50, "int8": 30}[lv]
        loaded.append(lv)
        return types.SimpleNamespace(level=lv)

    def release():
        nv.mem = 300.0 + 0.0

    mem = P.memory_profile(meter, load, release_fn=release)
    assert mem["alone_delta_mib"] == {"fp32": 100, "fp16": 50, "int8": 30}
    assert mem["adaptive_set_delta_mib"] == 80
