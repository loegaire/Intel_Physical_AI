#!/usr/bin/env python3
"""Export SmolVLA policy to ONNX and convert to OpenVINO IR."""

import os
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def export_smolvla_to_onnx():
    """Export SmolVLA policy to ONNX format."""
    print("Loading SmolVLA policy...")
    checkpoint = os.getenv("SMOLVLA_CHECKPOINT", "models/smolvla")
    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.eval()
    
    # Create dummy inputs matching the expected format
    batch_size = 1
    image_size = (256, 256)
    state_dim = 6
    
    # Observation images: separate keys for each camera
    dummy_camera1 = torch.randn(batch_size, 3, *image_size)
    dummy_camera2 = torch.randn(batch_size, 3, *image_size)
    dummy_camera3 = torch.randn(batch_size, 3, *image_size)
    
    # Observation state: (batch, state_dim)
    dummy_state = torch.randn(batch_size, state_dim)
    
    # Task/language instruction - tokenize with processor
    from transformers import AutoProcessor
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM2-500M-Video-Instruct')
    
    dummy_task = ["Set the table for dinner"]
    lang_inputs = processor(text=dummy_task, return_tensors="pt", padding=True, truncation=True)
    dummy_lang_tokens = lang_inputs["input_ids"]
    dummy_lang_mask = lang_inputs["attention_mask"]
    
    # The policy's forward expects a dict with specific keys
    dummy_input = {
        "observation.images.camera1": dummy_camera1,
        "observation.images.camera2": dummy_camera2,
        "observation.images.camera3": dummy_camera3,
        "observation.state": dummy_state,
        OBS_LANGUAGE_TOKENS: dummy_lang_tokens,
        OBS_LANGUAGE_ATTENTION_MASK: dummy_lang_mask,
    }
    
    # Check what the policy actually expects
    print("Policy input features:", policy.config.input_features)
    print("Policy output features:", policy.config.output_features)
    
    # Export to ONNX
    onnx_path = Path("models/smolvla.onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting to ONNX: {onnx_path}")
    
    # We need to wrap the policy to handle the dict input
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
        
        def forward(self, camera1, camera2, camera3, observation_state, lang_tokens, lang_mask):
            # Combine into dict format expected by policy
            batch = {
                "observation.images.camera1": camera1,
                "observation.images.camera2": camera2,
                "observation.images.camera3": camera3,
                "observation.state": observation_state,
                OBS_LANGUAGE_TOKENS: lang_tokens,
                OBS_LANGUAGE_ATTENTION_MASK: lang_mask,
            }
            # Get action from policy
            with torch.no_grad():
                action = self.policy.select_action(batch)
            return action
    
    wrapper = PolicyWrapper(policy)
    
    # Export
    torch.onnx.export(
        wrapper,
        (dummy_camera1, dummy_camera2, dummy_camera3, dummy_state, dummy_lang_tokens, dummy_lang_mask),
        onnx_path,
        input_names=["camera1", "camera2", "camera3", "observation_state", "lang_tokens", "lang_mask"],
        output_names=["action"],
        dynamic_axes={
            "camera1": {0: "batch"},
            "camera2": {0: "batch"},
            "camera3": {0: "batch"},
            "observation_state": {0: "batch"},
            "lang_tokens": {0: "batch", 1: "seq_len"},
            "lang_mask": {0: "batch", 1: "seq_len"},
            "action": {0: "batch", 1: "chunk_size"},
        },
        opset_version=18,
        do_constant_folding=True,
    )
    
    print(f"ONNX exported to {onnx_path}")
    return onnx_path


def convert_onnx_to_openvino(onnx_path: Path, output_dir: Path):
    """Convert ONNX to OpenVINO IR."""
    from openvino.tools.ovc import convert_model
    from openvino import save_model
    
    print(f"Converting ONNX to OpenVINO IR...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = convert_model(str(onnx_path))
    xml_path = output_dir / "smolvla.xml"
    save_model(model, str(xml_path))
    
    print(f"OpenVINO IR saved to {xml_path}")
    return xml_path


def verify_openvino_model(xml_path: Path):
    """Verify the converted model."""
    from openvino import Core
    from transformers import AutoProcessor
    
    print(f"Verifying OpenVINO model: {xml_path}")
    core = Core()
    model = core.read_model(str(xml_path))
    compiled = core.compile_model(model, "CPU")
    print(f"  Inputs: {[(inp.get_any_name(), inp.shape) for inp in compiled.inputs]}")
    print(f"  Outputs: {[(out.get_any_name(), out.shape) for out in compiled.outputs]}")
    
    # Test inference - need to tokenize task
    processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM2-500M-Video-Instruct')
    dummy_task = ["Set the table for dinner"]
    lang_inputs = processor(text=dummy_task, return_tensors="pt", padding=True, truncation=True)
    dummy_lang_tokens = lang_inputs["input_ids"].numpy().astype(np.int64)
    dummy_lang_mask = lang_inputs["attention_mask"].numpy().astype(np.int64)
    
    dummy_camera1 = np.random.randn(1, 3, 256, 256).astype(np.float32)
    dummy_camera2 = np.random.randn(1, 3, 256, 256).astype(np.float32)
    dummy_camera3 = np.random.randn(1, 3, 256, 256).astype(np.float32)
    dummy_state = np.random.randn(1, 6).astype(np.float32)
    
    results = compiled({
        "camera1": dummy_camera1,
        "camera2": dummy_camera2,
        "camera3": dummy_camera3,
        "observation_state": dummy_state,
        "lang_tokens": dummy_lang_tokens,
        "lang_mask": dummy_lang_mask,
    })
    print(f"  Test inference output shape: {results[compiled.outputs[0]].shape}")
    print("Verification OK!")


if __name__ == "__main__":
    onnx_path = export_smolvla_to_onnx()
    xml_path = convert_onnx_to_openvino(onnx_path, Path("models/smolvla_openvino"))
    verify_openvino_model(xml_path)
    
    print("\nDone! Run benchmark with:")
    print(f"  python scripts/benchmark_openvino.py --model-path {xml_path} --device CPU")
