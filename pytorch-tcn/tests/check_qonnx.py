from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_tcn.ECG5000_QTCN import ECG5000QTCNClassifier
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx


def load_ecg5000_txt(file_path):
    data = np.loadtxt(file_path, dtype=np.float32)

    y = data[:, 0].astype(np.int64)
    x = data[:, 1:].astype(np.float32)

    # ECG5000 常见标签是 1~5，转成 0~4
    if y.min() == 1:
        y = y - 1

    # [N, 140] -> [N, 1, 140]
    x = x[:, np.newaxis, :]

    return x, y


def build_model_from_checkpoint(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    # 如果你保存了 config，就自动用 config 初始化
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        cfg = checkpoint["config"]
        model = ECG5000QTCNClassifier(
            num_inputs=cfg.get("input_channels", 1),
            num_channels=cfg.get("num_channels", [16, 16, 32]),
            kernel_size=cfg.get("kernel_size", 3),
            dropout=cfg.get("dropout", 0.0),
            num_classes=cfg.get("num_classes", 5),
        )
    else:
        # 如果没保存 config，这里就按你训练时的参数手动写
        model = ECG5000QTCNClassifier(
            num_inputs=1,
            num_channels=[16, 16, 16],
            kernel_size=3,
            dropout=0.0,
            num_classes=5,
        )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def get_first_input_name(model_wrapper):
    return model_wrapper.graph.input[0].name


def get_first_output_name(model_wrapper):
    return model_wrapper.graph.output[0].name


def main():
    device = torch.device("cpu")

    ckpt_path = ROOT / "best_qtcn_model.pth"
    qonnx_path = ROOT / "ecg5000_qtcn_qonnx.onnx"
    test_file = ROOT / "datasets" / "ECG5000" / "ECG5000_TEST.txt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not qonnx_path.exists():
        raise FileNotFoundError(f"QONNX file not found: {qonnx_path}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    # 1) 加载 PyTorch 模型
    pt_model = build_model_from_checkpoint(ckpt_path, device)

    # 2) 加载 QONNX
    qonnx_model = ModelWrapper(str(qonnx_path))
    input_name = get_first_input_name(qonnx_model)
    output_name = get_first_output_name(qonnx_model)

    print(f"QONNX input name : {input_name}")
    print(f"QONNX output name: {output_name}")

    # 3) 读取测试集
    x_test, y_test = load_ecg5000_txt(test_file)

    # 只测前 N 个样本，避免太慢
    num_samples = 20
    x_test = x_test[:num_samples]
    y_test = y_test[:num_samples]

    same_pred_count = 0
    max_abs_diff_all = []
    mean_abs_diff_all = []

    print("\nStart checking QONNX vs PyTorch...\n")

    for i in range(num_samples):
        x_np = x_test[i:i+1].astype(np.float32)   # shape [1,1,140]
        y_true = int(y_test[i])

        # PyTorch 输出
        with torch.no_grad():
            x_torch = torch.from_numpy(x_np).to(device)
            pt_out = pt_model(x_torch).cpu().numpy()

        pt_pred = int(np.argmax(pt_out, axis=1)[0])

        # QONNX 输出
        qonnx_out_dict = execute_onnx(qonnx_model, {input_name: x_np})
        qonnx_out = qonnx_out_dict[output_name]
        qonnx_pred = int(np.argmax(qonnx_out, axis=1)[0])

        # 差异
        abs_diff = np.abs(pt_out - qonnx_out)
        max_abs_diff = float(abs_diff.max())
        mean_abs_diff = float(abs_diff.mean())

        max_abs_diff_all.append(max_abs_diff)
        mean_abs_diff_all.append(mean_abs_diff)

        pred_same = (pt_pred == qonnx_pred)
        if pred_same:
            same_pred_count += 1

        print(f"Sample {i:02d} | true={y_true} | "
              f"pt_pred={pt_pred} | qonnx_pred={qonnx_pred} | "
              f"same={pred_same} | "
              f"max_abs_diff={max_abs_diff:.8f} | "
              f"mean_abs_diff={mean_abs_diff:.8f}")

    print("\n========== Summary ==========")
    print(f"Checked samples        : {num_samples}")
    print(f"Prediction match count : {same_pred_count}/{num_samples}")
    print(f"Prediction match ratio : {same_pred_count / num_samples:.4f}")
    print(f"Global max abs diff    : {max(max_abs_diff_all):.8f}")
    print(f"Avg of mean abs diff   : {np.mean(mean_abs_diff_all):.8f}")

    # 一个简单判断
    if same_pred_count == num_samples:
        print("\nQONNX functional check looks GOOD: all predictions match.")
    else:
        print("\nQONNX functional check is NOT perfect: some predictions differ.")
        print("You should inspect the exported graph or compare intermediate tensors.")


if __name__ == "__main__":
    main()