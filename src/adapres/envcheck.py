"""GPU / TensorRT / ONNX Runtime / NVML checks. KAGGLE-ONLY (needs a GPU).

Every probe returns a dict with either results or the real error string; nothing
is assumed to work.
"""

from __future__ import annotations

import tempfile
import time
import traceback
from pathlib import Path


def _err(e: BaseException) -> dict:
    return {"ok": False, "error": repr(e), "traceback": traceback.format_exc()}


def _tiny_onnx(path: Path) -> None:
    """y = relu(x @ W) with x:[1,64], W:[64,64]."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w = numpy_helper.from_array(np.random.default_rng(0).standard_normal((64, 64)).astype("float32"), "W")
    g = helper.make_graph(
        [helper.make_node("MatMul", ["x", "W"], ["h"]), helper.make_node("Relu", ["h"], ["y"])],
        "tiny", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64])], [w])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def onnxruntime_probe() -> dict:
    try:
        import numpy as np
        import onnxruntime as ort

        out = {"ok": True, "version": ort.__version__}
        # ORT >= 1.21 can preload the CUDA/cuDNN libs installed by the [cuda,cudnn] extras.
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
                out["preload_dlls"] = "ok"
            except Exception as e:  # noqa: BLE001 - recorded
                out["preload_dlls"] = f"error: {e!r}"
        else:
            out["preload_dlls"] = "not available in this ORT version"
        provs = ort.get_available_providers()
        out["available_providers"] = provs
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tiny.onnx"
            _tiny_onnx(p)
            sess = ort.InferenceSession(str(p), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            out["session_providers"] = sess.get_providers()
            y = sess.run(None, {"x": np.ones((1, 64), "float32")})[0]
            out["output_shape"] = list(y.shape)
        out["cuda_ep_active"] = "CUDAExecutionProvider" in out["session_providers"]
        if not out["cuda_ep_active"]:
            out["ok"] = False
            out["error"] = "CUDAExecutionProvider not active (ORT fell back to CPU)"
        return out
    except Exception as e:  # noqa: BLE001
        return _err(e)


def tensorrt_probe() -> dict:
    """Parse the tiny ONNX and build FP32 and FP16 engines. INT8 needs calibration
    data, so it is exercised in notebook 02 on real calib frames, not here."""
    try:
        import tensorrt as trt

        out = {"ok": True, "version": trt.__version__, "builds": {}}
        logger = trt.Logger(trt.Logger.WARNING)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tiny.onnx"
            _tiny_onnx(p)
            for prec in ("fp32", "fp16"):
                builder = trt.Builder(logger)
                for attr in ("platform_has_fast_fp16", "platform_has_fast_int8"):
                    try:
                        out[attr] = bool(getattr(builder, attr))
                    except Exception as e:  # noqa: BLE001 - deprecated in some TRT versions
                        out[attr] = f"unavailable: {e!r}"
                try:
                    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                except AttributeError:  # TRT 10 removed the flag (explicit batch is default)
                    flags = 0
                net = builder.create_network(flags)
                parser = trt.OnnxParser(net, logger)
                if not parser.parse(p.read_bytes()):
                    errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                    raise RuntimeError(f"ONNX parse failed: {errs}")
                cfg = builder.create_builder_config()
                if prec == "fp16":
                    cfg.set_flag(trt.BuilderFlag.FP16)
                t0 = time.perf_counter()
                ser = builder.build_serialized_network(net, cfg)
                if ser is None:
                    raise RuntimeError(f"TensorRT {prec} build returned None")
                out["builds"][prec] = {"bytes": int(ser.nbytes), "build_s": round(time.perf_counter() - t0, 3)}
        return out
    except Exception as e:  # noqa: BLE001
        return _err(e)


def nvml_probe(seconds: float = 5.0, device_index: int = 0) -> dict:
    """Check the NVML energy counter and power sampling under a small torch load."""
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        name = pynvml.nvmlDeviceGetName(h)
        out: dict = {"ok": True, "gpu": name if isinstance(name, str) else name.decode()}
        try:
            out["driver"] = pynvml.nvmlSystemGetDriverVersion()
        except Exception as e:  # noqa: BLE001
            out["driver"] = f"error: {e!r}"
        try:
            e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
            out["energy_counter_supported"] = True
        except Exception as e:  # noqa: BLE001
            e0 = None
            out["energy_counter_supported"] = False
            out["energy_counter_error"] = repr(e)
        samples = []
        try:
            import torch

            a = torch.randn(2048, 2048, device="cuda")
            t_end = time.perf_counter() + seconds
            t0 = time.perf_counter()
            while time.perf_counter() < t_end:
                for _ in range(20):
                    a = (a @ a).clamp_(-1, 1)
                torch.cuda.synchronize()
                samples.append(pynvml.nvmlDeviceGetPowerUsage(h))  # mW
            elapsed = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            out["load_error"] = repr(e)
            elapsed = None
        if e0 is not None:
            e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
            out["energy_delta_mJ"] = int(e1 - e0)
            if elapsed:
                out["energy_counter_mean_W"] = round((e1 - e0) / 1000 / elapsed, 2)
        if samples:
            out["power_samples_n"] = len(samples)
            out["power_sample_mean_W"] = round(sum(samples) / len(samples) / 1000, 2)
        out["elapsed_s"] = elapsed
        try:
            out["power_limit_W"] = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000
        except Exception as e:  # noqa: BLE001
            out["power_limit_W"] = f"error: {e!r}"
        pynvml.nvmlShutdown()
        return out
    except Exception as e:  # noqa: BLE001
        return _err(e)


def run_all(energy_seconds: float = 5.0) -> dict:
    return {"onnxruntime": onnxruntime_probe(), "tensorrt": tensorrt_probe(), "nvml": nvml_probe(energy_seconds)}
