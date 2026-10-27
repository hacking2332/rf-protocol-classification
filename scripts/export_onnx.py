"""
scripts/export_onnx.py
----------------------
Export a trained PyTorch model to ONNX format for edge deployment.
Also runs a quick validation to confirm PyTorch and ONNX outputs match.

Usage:
    python scripts/export_onnx.py --model cnn1d --checkpoint models/cnn1d_best.pt
    python scripts/export_onnx.py --model transformer --checkpoint models/transformer_best.pt
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import onnx
import onnxruntime as ort

from src.models import build_model
from src.utils  import get_device


# Input shapes per model
INPUT_SHAPES = {
    "cnn1d":       (1, 2, 128),
    "cnn2d":       (1, 1, 17, 9),
    "transformer": (1, 2, 128),
}


def export(model_name: str, checkpoint: str, output_path: str, device: torch.device):
    print(f"\n[Export] Model    : {model_name.upper()}")
    print(f"[Export] Checkpoint: {checkpoint}")
    print(f"[Export] Output   : {output_path}")

    # Load model
    model = build_model(model_name)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    model.to(device)

    dummy = torch.randn(*INPUT_SHAPES[model_name]).to(device)

    # Export
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names    = ["iq_signal"],
        output_names   = ["class_logits"],
        dynamic_axes   = {"iq_signal": {0: "batch_size"},
                          "class_logits": {0: "batch_size"}},
        opset_version  = 17,
        export_params  = True,
    )
    print(f"[Export] Exported successfully.")

    # Validate ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[Export] ONNX model check passed.")

    # Compare PyTorch vs ONNX outputs
    with torch.no_grad():
        torch_out = model(dummy).cpu().numpy()

    ort_session = ort.InferenceSession(output_path)
    ort_out = ort_session.run(
        None, {"iq_signal": dummy.cpu().numpy()}
    )[0]

    max_diff = np.abs(torch_out - ort_out).max()
    print(f"[Export] Max output diff (PyTorch vs ONNX): {max_diff:.2e}")

    if max_diff < 1e-4:
        print(f"[Export] Validation PASSED — outputs match within tolerance.")
    else:
        print(f"[Export] WARNING — outputs differ by {max_diff:.2e}, check export settings.")

    # Print model size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Export] Model size: {size_mb:.2f} MB")

    return output_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      type=str, required=True,
                   choices=["cnn1d", "cnn2d", "transformer"])
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output",     type=str, default=None)
    args = p.parse_args()

    device = get_device()

    checkpoint = args.checkpoint or f"models/{args.model}_best.pt"
    output     = args.output     or f"models/{args.model}.onnx"

    if not os.path.exists(checkpoint):
        print(f"[Error] Checkpoint not found: {checkpoint}")
        print(f"        Train the model first with: python scripts/run_training.py --model {args.model}")
        sys.exit(1)

    export(args.model, checkpoint, output, device)
    print(f"\n[Done] ONNX model ready for edge deployment → {output}")


if __name__ == "__main__":
    main()
