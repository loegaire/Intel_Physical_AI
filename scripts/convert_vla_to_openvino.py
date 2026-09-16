#!/usr/bin/env python3
"""Convert VLA model (SmolVLA/Pi0.5) to OpenVINO IR."""

import argparse
import os
import sys
from pathlib import Path

try:  # optional: loads HF_TOKEN from a local .env when python-dotenv is present
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch
from huggingface_hub import hf_hub_download, snapshot_download


def download_model(repo_id: str, local_dir: Path) -> Path:
    """Download model from HuggingFace."""
    print(f"Downloading {repo_id} to {local_dir}...")
    token = os.getenv("HF_TOKEN")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False, token=token)
    model_path = local_dir / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {local_dir}")
    return model_path


def load_smolvla(model_dir: Path):
    """Load SmolVLA model from local directory."""
    from transformers import AutoModel, AutoProcessor
    
    print("Loading SmolVLA model...")
    token = os.getenv("HF_TOKEN")
    model = AutoModel.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        token=token,
    )
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, token=token)
    return model, processor


def load_pi05(model_dir: Path):
    """Load Pi0.5 model from local directory."""
    from transformers import AutoModel, AutoProcessor
    
    print("Loading Pi0.5 model...")
    token = os.getenv("HF_TOKEN")
    model = AutoModel.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        token=token,
    )
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, token=token)
    return model, processor


def export_to_onnx(model, model_type: str, onnx_path: Path, input_shape: tuple = (1, 3, 224, 224)):
    """Export model to ONNX."""
    print(f"Exporting to ONNX: {onnx_path}")
    
    model.eval()
    
    if model_type == "smolvla":
        # SmolVLA inputs: input_ids, attention_mask, pixel_values
        dummy_input_ids = torch.randint(0, 32000, (1, 512), dtype=torch.long)
        dummy_attention_mask = torch.ones(1, 512, dtype=torch.long)
        dummy_pixel_values = torch.randn(*input_shape, dtype=torch.float32)
        
        dummy_inputs = {
            "input_ids": dummy_input_ids,
            "attention_mask": dummy_attention_mask,
            "pixel_values": dummy_pixel_values,
        }
        input_names = ["input_ids", "attention_mask", "pixel_values"]
        output_names = ["logits"]
        
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask, dummy_pixel_values),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "pixel_values": {0: "batch"},
                "logits": {0: "batch", 1: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    
    elif model_type == "pi05":
        # Pi0.5 inputs: similar structure
        dummy_input_ids = torch.randint(0, 32000, (1, 512), dtype=torch.long)
        dummy_attention_mask = torch.ones(1, 512, dtype=torch.long)
        dummy_pixel_values = torch.randn(*input_shape, dtype=torch.float32)
        
        dummy_inputs = {
            "input_ids": dummy_input_ids,
            "attention_mask": dummy_attention_mask,
            "pixel_values": dummy_pixel_values,
        }
        input_names = ["input_ids", "attention_mask", "pixel_values"]
        output_names = ["actions"]
        
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask, dummy_pixel_values),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "pixel_values": {0: "batch"},
                "actions": {0: "batch", 1: "action_dim"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    
    print(f"ONNX exported to {onnx_path}")


def convert_onnx_to_openvino(onnx_path: Path, output_dir: Path, precision: str = "FP32"):
    """Convert ONNX to OpenVINO IR using Model Optimizer."""
    import subprocess
    
    print(f"Converting ONNX to OpenVINO IR ({precision})...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "mo",
        "--input_model", str(onnx_path),
        "--output_dir", str(output_dir),
        "--compress_to_fp16" if precision == "FP16" else "",
    ]
    cmd = [c for c in cmd if c]  # Remove empty strings
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Model Optimizer failed:")
        print(result.stderr)
        raise RuntimeError("OpenVINO conversion failed")
    
    print(f"OpenVINO IR saved to {output_dir}")
    xml_files = list(output_dir.glob("*.xml"))
    bin_files = list(output_dir.glob("*.bin"))
    print(f"Generated: {xml_files}, {bin_files}")
    return xml_files[0] if xml_files else None


def verify_openvino_model(xml_path: Path):
    """Verify the converted model loads correctly."""
    from openvino.runtime import Core
    
    print(f"Verifying OpenVINO model: {xml_path}")
    core = Core()
    model = core.read_model(str(xml_path))
    compiled = core.compile_model(model, "CPU")
    print(f"  Input: {compiled.input(0).shape}")
    print(f"  Output: {compiled.output(0).shape}")
    print("Verification OK!")


def main():
    parser = argparse.ArgumentParser(description="Convert VLA model to OpenVINO IR")
    parser.add_argument("--model", type=str, default="smolvla", choices=["smolvla", "pi05"],
                        help="Model type to convert")
    parser.add_argument("--repo", type=str, default="lerobot/smolvla_base",
                        help="HuggingFace repo ID")
    parser.add_argument("--output-dir", type=str, default="models/vla_openvino",
                        help="Output directory for OpenVINO IR")
    parser.add_argument("--precision", type=str, default="FP32", choices=["FP32", "FP16"],
                        help="Precision for OpenVINO IR")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use existing model in models/<model>/")
    parser.add_argument("--verify", action="store_true", help="Verify converted model")
    
    args = parser.parse_args()
    
    # Setup paths
    ROOT = Path(__file__).parent.parent
    model_dir = ROOT / "models" / args.model
    onnx_path = ROOT / "models" / f"{args.model}.onnx"
    output_dir = ROOT / args.output_dir
    
    # Step 1: Download model
    if not args.skip_download:
        download_model(args.repo, model_dir)
    else:
        print(f"Using existing model in {model_dir}")
    
    # Step 2: Load model
    if args.model == "smolvla":
        model, processor = load_smolvla(model_dir)
    elif args.model == "pi05":
        model, processor = load_pi05(model_dir)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Step 3: Export to ONNX
    export_to_onnx(model, args.model, onnx_path)
    
    # Step 4: Convert to OpenVINO IR
    xml_path = convert_onnx_to_openvino(onnx_path, output_dir, args.precision)
    
    # Step 5: Verify
    if args.verify and xml_path:
        verify_openvino_model(xml_path)
    
    print("\nDone! Run benchmark with:")
    print(f"  python scripts/benchmark_openvino.py --model-path {xml_path} --device CPU")


if __name__ == "__main__":
    main()