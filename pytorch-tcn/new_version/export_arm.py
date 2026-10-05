#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test whether the old Brevitas/PyTorch quantized network and checkpoint
can still run in the current environment.

Run directly:

    python test_brevitas_env_compat.py
"""

import argparse
import importlib.util
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ============================================================
# User settings
# ============================================================

MODEL_PY = "./pytorch-tcn/new_version/newnet.py"
MODEL_CLASS = "ECG5000FullQuantTCN"
CKPT = "./pytorch-tcn/new_version/best_qtcn_model_W4A4_finetune.pth"

MODEL_KWARGS = {
    "num_inputs": 1,
    "num_channels": [16, 16, 16],
    "num_classes": 5,
    "kernel_size": 3,
    "dropout": 0.1,
    "seq_len": 140,
}

# Dummy input shape for first forward test.
# For NCL TCN input: [batch, channels, length]
INPUT_SHAPE = (1, 1, 140, 1)

# Optional ECG5000 test.
RUN_ECG_TEST = True
DATA_TXT = "./pytorch-tcn/new_version/ECG5000_TEST.txt"
MAX_SAMPLES = 4500

# Dataset layout passed into the model.
# NCL  -> [N, 1, 140]
# NLC  -> [N, 140, 1]
# NCHW -> [N, 1, 140, 1]
# NHWC -> [N, 140, 1, 1]
DATA_LAYOUT = "NCHW"

# If your model file imports other local files, add paths here.
# Example:
# EXTRA_IMPORT_PATHS = ["./", "./models"]
EXTRA_IMPORT_PATHS = ["./"]


# ============================================================
# Helper functions
# ============================================================

def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def add_import_paths():
    for p in EXTRA_IMPORT_PATHS:
        p = str(Path(p).resolve())
        if p not in sys.path:
            sys.path.insert(0, p)

    model_dir = str(Path(MODEL_PY).resolve().parent)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)


def try_import_brevitas():
    try:
        import brevitas
        print("brevitas:", brevitas.__version__)
    except Exception as e:
        print("[ERROR] Failed to import brevitas:")
        print(repr(e))
        raise


def load_python_module(py_path):
    py_path = Path(py_path).resolve()
    if not py_path.exists():
        raise FileNotFoundError("Model python file not found: {}".format(py_path))

    module_name = py_path.stem
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(module, class_name, model_kwargs):
    if not hasattr(module, class_name):
        print("[ERROR] Model class not found:", class_name)
        print("Available classes in module:")
        for name in sorted(dir(module)):
            obj = getattr(module, name)
            if inspect.isclass(obj):
                print("  class:", name)
        raise AttributeError(class_name)

    cls = getattr(module, class_name)
    print("Model class:", cls)

    print("\nModel kwargs:")
    print(json.dumps(model_kwargs, indent=2))

    model = cls(**model_kwargs)
    return model


def extract_state_dict(ckpt):
    """
    Support common checkpoint formats:
      1. torch.save(model.state_dict())
      2. torch.save({"model_state_dict": model.state_dict(), ...})
      3. torch.save({"state_dict": model.state_dict(), ...})
      4. torch.save(model)
    """
    if isinstance(ckpt, torch.nn.Module):
        print("[info] Checkpoint is a full torch.nn.Module object.")
        return ckpt.state_dict()

    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "net", "model"]:
            if key in ckpt:
                value = ckpt[key]
                if isinstance(value, dict):
                    print("[info] Using checkpoint key:", key)
                    return value
                if isinstance(value, torch.nn.Module):
                    print("[info] Using checkpoint module key:", key)
                    return value.state_dict()

        tensor_like_count = 0
        for _, v in ckpt.items():
            if torch.is_tensor(v):
                tensor_like_count += 1

        if tensor_like_count > 0:
            print("[info] Checkpoint looks like a raw state_dict.")
            return ckpt

    raise RuntimeError("Could not extract state_dict from checkpoint.")


def strip_prefix_from_state_dict(state_dict, prefix):
    new_sd = {}
    changed = False

    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_sd[k[len(prefix):]] = v
            changed = True
        else:
            new_sd[k] = v

    return new_sd, changed


def print_state_dict_summary(name, state_dict, max_keys=30):
    print("\n{}: {} tensors".format(name, len(state_dict)))

    keys = list(state_dict.keys())
    for k in keys[:max_keys]:
        v = state_dict[k]
        if torch.is_tensor(v):
            print("  {}  shape={} dtype={}".format(k, tuple(v.shape), v.dtype))
        else:
            print("  {}  type={}".format(k, type(v)))

    if len(keys) > max_keys:
        print("  ... {} more keys".format(len(keys) - max_keys))

def print_quant_layers(model):
    print("=" * 60)
    print("Quant layers")
    print("=" * 60)

    for name, m in model.named_modules():
        cls_name = m.__class__.__name__

        if "Quant" in cls_name:
            print(name, "->", cls_name)

            if hasattr(m, "weight_quant"):
                print("   weight_quant:", m.weight_quant)

            if hasattr(m, "act_quant"):
                print("   act_quant:", m.act_quant)

            if hasattr(m, "bias_quant"):
                print("   bias_quant:", m.bias_quant)

def load_weights_strictly(model, state_dict):
    print_header("Loading checkpoint")

    candidates = [("original", state_dict)]

    sd_module, changed_module = strip_prefix_from_state_dict(state_dict, "module.")
    if changed_module:
        candidates.append(("strip module.", sd_module))

    sd_model, changed_model = strip_prefix_from_state_dict(state_dict, "model.")
    if changed_model:
        candidates.append(("strip model.", sd_model))

    last_error = None

    for name, sd in candidates:
        print("\nTrying strict=True:", name)
        try:
            model.load_state_dict(sd, strict=True)
            print("[OK] strict=True load succeeded with:", name)
            return sd
        except Exception as e:
            print("[FAIL] strict=True load failed with:", name)
            print(repr(e))
            last_error = e

        print("\nTrying controlled strict=False:", name)
        ret = model.load_state_dict(sd, strict=False)

        missing = list(ret.missing_keys)
        unexpected = list(ret.unexpected_keys)

        allowed_missing = [
            k for k in missing
            if ".pad_quant." in k
        ]

        bad_missing = [
            k for k in missing
            if ".pad_quant." not in k
        ]

        if len(bad_missing) == 0 and len(unexpected) == 0:
            print("[OK] controlled strict=False load succeeded.")
            print("Ignored missing pad_quant keys:")
            for k in allowed_missing:
                print("  ", k)
            return sd

        print("[FAIL] controlled strict=False load failed.")
        print("bad missing keys:")
        for k in bad_missing:
            print("  ", k)

        print("unexpected keys:")
        for k in unexpected:
            print("  ", k)

    raise RuntimeError("Checkpoint is not compatible with this model definition.") from last_error


def unwrap_output(y):
    """
    Brevitas may return QuantTensor.
    If output has .value, use it.
    """
    if hasattr(y, "value"):
        return y.value

    if isinstance(y, (tuple, list)):
        # Use first tensor-like output if model returns tuple/list.
        for item in y:
            if hasattr(item, "value"):
                return item.value
            if torch.is_tensor(item):
                return item
        return y[0]

    return y


def run_dummy_forward(model, input_shape):
    print_header("Dummy forward test")

    x = torch.randn(*input_shape, dtype=torch.float32)

    print("input shape:", tuple(x.shape))
    print("input dtype :", x.dtype)

    model.eval()

    with torch.no_grad():
        t0 = time.perf_counter()
        y = model(x)
        t1 = time.perf_counter()

    y = unwrap_output(y)

    print("output type :", type(y))

    if torch.is_tensor(y):
        print("output shape:", tuple(y.shape))
        print("output dtype :", y.dtype)
        print("output min  :", float(y.min()))
        print("output max  :", float(y.max()))
        print("output mean :", float(y.float().mean()))

        if y.ndim >= 2:
            pred = torch.argmax(y, dim=1)
            print("pred:", pred.cpu().numpy())
        else:
            print("argmax:", int(torch.argmax(y).item()))
    else:
        print("output:", y)

    print("forward time: {:.6f} s".format(t1 - t0))
    print("[OK] Dummy forward succeeded.")


def load_ecg5000_txt(path):
    arr = np.loadtxt(path, dtype=np.float32)

    labels = arr[:, 0].astype(np.int64)
    x = arr[:, 1:].astype(np.float32)

    # ECG5000 labels are usually 1..5. Convert to 0..4.
    if labels.min() == 1 and labels.max() <= 5:
        labels = labels - 1

    return x, labels


def make_input_tensor_from_ecg(x_np, layout):
    """
    x_np shape: [N, 140]
    """
    if layout == "NCL":
        x = x_np[:, None, :]
    elif layout == "NLC":
        x = x_np[:, :, None]
    elif layout == "NCHW":
        x = x_np[:, None, :, None]
    elif layout == "NHWC":
        x = x_np[:, :, None, None]
    else:
        raise ValueError("Unknown layout: {}".format(layout))

    return torch.tensor(x, dtype=torch.float32)


def run_ecg_test(model, data_txt, max_samples, layout):
    print_header("ECG5000 small test")

    data_path = Path(data_txt).resolve()
    if not data_path.exists():
        raise FileNotFoundError("DATA_TXT not found: {}".format(data_path))

    x_np, labels = load_ecg5000_txt(str(data_path))

    if max_samples is not None:
        x_np = x_np[:max_samples]
        labels = labels[:max_samples]

    x = make_input_tensor_from_ecg(x_np, layout)

    print("data path :", data_path)
    print("samples   :", x.shape[0])
    print("x shape   :", tuple(x.shape))
    print("labels min/max:", int(labels.min()), int(labels.max()))

    model.eval()

    correct = 0
    times = []
    preds = []

    with torch.no_grad():
        for i in range(x.shape[0]):
            xi = x[i:i + 1]

            t0 = time.perf_counter()
            y = model(xi)
            t1 = time.perf_counter()

            y = unwrap_output(y)

            if not torch.is_tensor(y):
                raise RuntimeError("Model output is not a tensor after unwrap_output.")

            if y.ndim == 1:
                pred = int(torch.argmax(y).item())
            else:
                pred = int(torch.argmax(y, dim=1).item())

            preds.append(pred)
            correct += int(pred == int(labels[i]))
            times.append(t1 - t0)

    acc = correct / float(len(labels))
    avg_time = sum(times) / len(times)

    print("accuracy on tested samples:", acc)
    print("avg forward time: {:.6f} s/sample".format(avg_time))
    print("avg forward time: {:.3f} ms/sample".format(avg_time * 1000.0))
    print("FPS: {:.3f}".format(1.0 / avg_time))
    print("first preds :", preds[:20])
    print("first labels:", labels[:20].tolist())
    print("[OK] ECG small test finished.")


def main():
    print_header("Environment")
    print("python :", sys.version)
    print("torch  :", torch.__version__)
    print("torch file:", torch.__file__)
    try_import_brevitas()

    print("cwd:", os.getcwd())

    add_import_paths()

    print_header("User settings")
    print("MODEL_PY    :", MODEL_PY)
    print("MODEL_CLASS :", MODEL_CLASS)
    print("CKPT        :", CKPT)
    print("INPUT_SHAPE :", INPUT_SHAPE)
    print("RUN_ECG_TEST:", RUN_ECG_TEST)
    print("DATA_TXT    :", DATA_TXT)
    print("MAX_SAMPLES :", MAX_SAMPLES)
    print("DATA_LAYOUT :", DATA_LAYOUT)

    print_header("Import model definition")
    module = load_python_module(MODEL_PY)
    model = build_model(module, MODEL_CLASS, MODEL_KWARGS)
    #print_quant_layers(model)
    #print("\nModel:")
    #print(model)

    print_header("Load checkpoint")
    ckpt_path = Path(CKPT).resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(ckpt_path))

    print("checkpoint:", ckpt_path)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = extract_state_dict(ckpt)

    print_state_dict_summary("checkpoint state_dict", state_dict)
    print_state_dict_summary("model state_dict", model.state_dict())

    load_weights_strictly(model, state_dict)

    run_dummy_forward(model, INPUT_SHAPE)

    if RUN_ECG_TEST:
        run_ecg_test(
            model=model,
            data_txt=DATA_TXT,
            max_samples=MAX_SAMPLES,
            layout=DATA_LAYOUT,
        )

    print_header("Final result")
    print("[SUCCESS] This environment can import the model, load the checkpoint, and run forward.")


if __name__ == "__main__":
    main()