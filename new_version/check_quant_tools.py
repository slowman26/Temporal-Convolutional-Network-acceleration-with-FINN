import inspect
from pathlib import Path

import torch
import brevitas.nn as qnn

from newnet import ECG5000FullQuantTCN


# ============================================================
# User settings
# ============================================================

NUM_CLASSES = 5
SEQ_LEN = 140
NUM_INPUTS = 1
NUM_CHANNELS = (16, 16, 16)
KERNEL_SIZE = 3
DROPOUT = 0.1
CAUSAL = True

# Change these to check different quantization settings.
TARGET_WEIGHT_BITS = 4
TARGET_ACT_BITS = 4
TARGET_INPUT_BITS = 4

BASE_DIR = Path(__file__).resolve().parent

# Change this to your checkpoint path.
CKPT_PATH = BASE_DIR / "best_qtcn_model_W4A4_finetune.pth"

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

# ============================================================
# Helper functions
# ============================================================

def qt_value(x):
    return x.value if hasattr(x, "value") else x


def scalar(x):
    if x is None:
        return None

    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.numel() == 1:
            return x.item()
        return x.flatten()[:10].tolist()

    return x


def strip_module_prefix(state_dict):
    new_state = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v

    return new_state


def extract_model_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            print("[info] Using checkpoint key: model_state_dict")
            return ckpt["model_state_dict"]

        if "state_dict" in ckpt:
            print("[info] Using checkpoint key: state_dict")
            return ckpt["state_dict"]

    print("[info] Checkpoint seems to be a raw state_dict")
    return ckpt


def print_state_dict_summary(title, state_dict, max_items=80):
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(f"number of tensors: {len(state_dict)}")

    for i, (k, v) in enumerate(state_dict.items()):
        if i >= max_items:
            print(f"... skipped remaining {len(state_dict) - max_items} tensors")
            break

        if torch.is_tensor(v):
            print(f"{k:90s} shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"{k:90s} type={type(v)}")

    print()


def build_model():
    kwargs = dict(
        num_classes=NUM_CLASSES,
        seq_len=SEQ_LEN,
        num_inputs=NUM_INPUTS,
        num_channels=NUM_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        causal=CAUSAL,
    )

    print("=" * 80)
    print("Model constructor check")
    print("=" * 80)

    sig = inspect.signature(ECG5000FullQuantTCN)
    print("ECG5000FullQuantTCN signature:")
    print(sig)

    if "export_all_time_logits" in sig.parameters:
        kwargs["export_all_time_logits"] = False
        print("[info] Pass export_all_time_logits=False")

    if "weight_bits" in sig.parameters:
        kwargs["weight_bits"] = TARGET_WEIGHT_BITS
        print(f"[info] Pass weight_bits={TARGET_WEIGHT_BITS}")
    else:
        print("[warn] Model constructor does NOT support weight_bits")

    if "act_bits" in sig.parameters:
        kwargs["act_bits"] = TARGET_ACT_BITS
        print(f"[info] Pass act_bits={TARGET_ACT_BITS}")
    else:
        print("[warn] Model constructor does NOT support act_bits")

    if "input_bits" in sig.parameters:
        kwargs["input_bits"] = TARGET_INPUT_BITS
        print(f"[info] Pass input_bits={TARGET_INPUT_BITS}")
    else:
        print("[warn] Model constructor does NOT support input_bits")

    print("=" * 80)
    print()

    model = ECG5000FullQuantTCN(**kwargs)
    return model


def load_checkpoint(model, ckpt_path):
    print("=" * 80)
    print("Load checkpoint")
    print("=" * 80)

    print("checkpoint:", ckpt_path)

    if not ckpt_path.exists():
        print("[warn] Checkpoint not found. Skip loading.")
        return model

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_state = extract_model_state_dict(ckpt)
    ckpt_state = strip_module_prefix(ckpt_state)

    model_state = model.state_dict()

    print_state_dict_summary("Checkpoint state_dict", ckpt_state)
    print_state_dict_summary("Current model state_dict before loading", model_state)

    loadable_state = {}
    skipped = []

    for k, v in ckpt_state.items():
        if k in model_state and torch.is_tensor(v) and model_state[k].shape == v.shape:
            loadable_state[k] = v
        else:
            if k in model_state and torch.is_tensor(v):
                skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            elif torch.is_tensor(v):
                skipped.append((k, tuple(v.shape), None))
            else:
                skipped.append((k, type(v), None))

    missing, unexpected = model.load_state_dict(loadable_state, strict=False)

    print("=" * 80)
    print("Checkpoint loading result")
    print("=" * 80)
    print(f"loaded tensors : {len(loadable_state)}")
    print(f"missing keys   : {len(missing)}")
    print(f"unexpected keys: {len(unexpected)}")
    print(f"skipped keys   : {len(skipped)}")

    if missing:
        print("\nFirst missing keys:")
        for k in missing[:40]:
            print("  ", k)

    if unexpected:
        print("\nFirst unexpected keys:")
        for k in unexpected[:40]:
            print("  ", k)

    if skipped:
        print("\nFirst skipped keys:")
        for item in skipped[:40]:
            print("  ", item)

    print()

    return model


# ============================================================
# Bit-width inspection
# ============================================================

def get_tensor_quant(proxy):
    if proxy is None:
        return None

    if hasattr(proxy, "tensor_quant"):
        return proxy.tensor_quant

    if hasattr(proxy, "fused_activation_quant_proxy"):
        fused = proxy.fused_activation_quant_proxy
        if hasattr(fused, "tensor_quant"):
            return fused.tensor_quant

    return None


def get_bit_width_from_proxy(proxy):
    tq = get_tensor_quant(proxy)

    if tq is None:
        return None

    bw_impl = getattr(tq, "msb_clamp_bit_width_impl", None)

    if bw_impl is None:
        return None

    # Try common Brevitas APIs.
    candidate_attrs = [
        "bit_width",
        "_bit_width",
    ]

    for attr in candidate_attrs:
        if hasattr(bw_impl, attr):
            try:
                bw = getattr(bw_impl, attr)
                if callable(bw):
                    bw = bw()
                return scalar(bw)
            except Exception:
                pass

    # Try calling the module directly.
    try:
        bw = bw_impl()
        return scalar(bw)
    except Exception:
        pass

    # Last fallback: scan buffers.
    try:
        for _, buf in bw_impl.named_buffers():
            if torch.is_tensor(buf) and buf.numel() == 1:
                return scalar(buf)
    except Exception:
        pass

    return None


def print_brevitas_bit_widths(model):
    print("=" * 80)
    print("Brevitas bit-width check")
    print("=" * 80)

    found = False

    for name, m in model.named_modules():
        if isinstance(m, qnn.QuantConv2d):
            found = True

            w_bw = get_bit_width_from_proxy(m.weight_quant)
            b_bw = get_bit_width_from_proxy(m.bias_quant)
            in_bw = get_bit_width_from_proxy(m.input_quant)
            out_bw = get_bit_width_from_proxy(m.output_quant)

            print(f"{name}: QuantConv2d")
            print(f"  weight bit_width : {w_bw}")
            print(f"  bias bit_width   : {b_bw}")
            print(f"  input bit_width  : {in_bw}")
            print(f"  output bit_width : {out_bw}")

        elif isinstance(m, qnn.QuantReLU):
            found = True

            a_bw = get_bit_width_from_proxy(m.act_quant)

            print(f"{name}: QuantReLU")
            print(f"  act bit_width    : {a_bw}")

        elif isinstance(m, qnn.QuantIdentity):
            found = True

            a_bw = get_bit_width_from_proxy(m.act_quant)

            print(f"{name}: QuantIdentity")
            print(f"  act bit_width    : {a_bw}")

    if not found:
        print("[warn] No Brevitas QuantConv2d / QuantReLU / QuantIdentity found.")

    print()


# ============================================================
# Forward QuantTensor inspection
# ============================================================

def print_qtensor_meta(name, x):
    print("-" * 80)
    print(name)
    print("-" * 80)

    if not hasattr(x, "value"):
        print("type       :", type(x))
        if torch.is_tensor(x):
            print("shape      :", tuple(x.shape))
            print("dtype      :", x.dtype)
            print("min/max    :", float(x.min()), float(x.max()))
        print("QuantTensor: False")
        return

    print("type       :", type(x))
    print("QuantTensor: True")
    print("value shape:", tuple(x.value.shape))
    print("value dtype:", x.value.dtype)

    if hasattr(x, "scale"):
        print("scale      :", scalar(x.scale))

    if hasattr(x, "zero_point"):
        print("zero_point :", scalar(x.zero_point))

    if hasattr(x, "bit_width"):
        print("bit_width  :", scalar(x.bit_width))

    if hasattr(x, "signed"):
        print("signed     :", x.signed)

    val = x.value.detach()
    print("value min  :", float(val.min()))
    print("value max  :", float(val.max()))
    print("value mean :", float(val.mean()))


def debug_forward(model):
    print("=" * 80)
    print("Dummy forward QuantTensor check")
    print("=" * 80)

    model.eval()
    model.to(DEVICE)

    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            # Print only important quantized activation points.
            if (
                name == "input_quant"
                or name.endswith("act1")
                or name.endswith("pre_add_quant")
                or name.endswith("post_add_act")
            ):
                print_qtensor_meta(name, output)
        return hook

    for name, m in model.named_modules():
        if isinstance(m, (qnn.QuantIdentity, qnn.QuantReLU)):
            hooks.append(m.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        dummy = torch.zeros(1, NUM_INPUTS, SEQ_LEN, 1, dtype=torch.float32).to(DEVICE)
        out = model(dummy)

    for h in hooks:
        h.remove()

    print_qtensor_meta("model output", out)

    out_val = qt_value(out)
    print("=" * 80)
    print("Output shape check")
    print("=" * 80)
    print("output shape:", tuple(out_val.shape))

    if tuple(out_val.shape) == (1, NUM_CLASSES, 1, 1):
        print("[OK] Output shape is [1, 5, 1, 1]")
    elif tuple(out_val.shape) == (1, NUM_CLASSES):
        print("[OK] Output shape is [1, 5]")
    else:
        print("[warn] Output shape is not last-step-only [1, 5, 1, 1] or [1, 5]")

    print()


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Quant model checker")
    print("=" * 80)
    print("device             :", DEVICE)
    print("checkpoint         :", CKPT_PATH)
    print("target weight_bits :", TARGET_WEIGHT_BITS)
    print("target act_bits    :", TARGET_ACT_BITS)
    print("target input_bits  :", TARGET_INPUT_BITS)
    print("=" * 80)
    print()

    model = build_model()

    # Run one dummy forward before loading to initialize runtime stats if needed.
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        dummy = torch.zeros(1, NUM_INPUTS, SEQ_LEN, 1, dtype=torch.float32).to(DEVICE)
        _ = model(dummy)

    print_state_dict_summary(
        "Current model state_dict after dummy forward before loading",
        model.state_dict(),
    )

    model = load_checkpoint(model, CKPT_PATH)

    # Run another dummy forward after loading.
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        dummy = torch.zeros(1, NUM_INPUTS, SEQ_LEN, 1, dtype=torch.float32).to(DEVICE)
        _ = model(dummy)

    print_state_dict_summary(
        "Current model state_dict after loading and dummy forward",
        model.state_dict(),
    )

    print_brevitas_bit_widths(model)
    debug_forward(model)

    print("=" * 80)
    print("Check finished")
    print("=" * 80)


if __name__ == "__main__":
    main()