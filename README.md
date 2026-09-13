# Intel Physical AI Online Challenge
## Bimanual VLA Manipulation with Multi-Modal Reasoning
### Challenge Option: Setting Up a Dinner Table

---

## Overview

This repository implements a complete Physical AI solution for the Intel Physical AI Online Challenge. The task requires two simulated SO-101 arms in MuJoCo to interpret natural-language instructions, reason over camera observations, coordinate both manipulators, and complete a multi-step table-setting task.

**Core Goal**: Demonstrate a robust perception-to-action pipeline where a multi-modal policy understands task instructions, reasons over the simulated scene, coordinates two robot arms, and completes the requested manipulation sequence on Intel Core Ultra Series 2/3 hardware.

---

## Repository Structure

```
Intel_Physical_AI/
├── assets/
│   └── so101/                 # SO-101 robot meshes and XML
├── envs/
│   ├── dinner_table_env.py    # Main MuJoCo environment
│   └── world.xml              # World scene (table, drawer, cameras)
├── perception/
│   ├── camera_model.py        # Pinhole camera model for MuJoCo
│   ├── detector.py            # RGB-D object detector (classical CV)
│   └── perception.py          # Multi-camera fusion module
├── policy/
│   ├── ik.py                  # Inverse kinematics solvers
│   ├── skills.py              # Manipulation skills (Grasp, Place, etc.)
│   └── orchestrator.py        # Task planner and skill sequencer
├── scripts/
│   ├── run_demo.py            # Demo/evaluation runner
│   ├── benchmark_openvino.py  # OpenVINO inference benchmark
│   ├── study_workspace.py     # Workspace feasibility study
│   └── check_perception.py    # Perception accuracy check
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/loegaire/Intel_Physical_AI
cd Intel_Physical_AI
pip install -e .[all]
```

### Run a Demo

```bash
# Single seed demo with video
python scripts/run_demo.py --seed 0 --video demo.mp4

# Multi-seed evaluation (10 seeds)
python scripts/run_demo.py --seeds 10 --output-dir results

# Evaluation only (no video)
python scripts/run_demo.py --eval-only --seeds 10
```

### Benchmark Inference

```bash
# OpenVINO benchmark (requires model)
python scripts/benchmark_openvino.py --model-path models/vla.xml --device CPU

# Perception pipeline benchmark
python scripts/benchmark_openvino.py --perception-only

# Skill execution benchmark
python scripts/benchmark_openvino.py --skills-only
```

### Workspace Study

```bash
# Check reachability of key workspace targets
python scripts/study_workspace.py
```

### Perception Check

```bash
# Evaluate perception accuracy across seeds
python scripts/check_perception.py 5
```

---

## System Architecture

### 1. Simulation Environment (`envs/dinner_table_env.py`)

- **Physics**: MuJoCo 3.x, 200 Hz simulation, 50 Hz control
- **Robots**: Two Menagerie SO-101 arms (6 DoF each + gripper)
- **Scene**: Counter with sliding drawer (left), placemat with slots (right)
- **Objects**: Plate, mug, bottle, spoon, fork (randomized per seed)
- **Domain Randomization**: Position, yaw, friction, mass, size, color, lighting
- **Cameras**: Overview, overhead, drawer_cam, placemat_cam, wrist_top

### 2. Perception Pipeline (`perception/`)

- **Camera Model**: Pinhole projection with MuJoCo camera poses
- **Detector**: Classical RGB-D detection using HSV color thresholds + depth masks
- **Fusion**: Exponential moving average with outlier gating
- **Outputs**: World-frame object positions, yaw estimates, drawer slide estimate

**Perception-to-Skill Interface**: The environment provides a `set_pose_provider()` method that routes object pose queries through the perception module, enabling seamless swap from privileged state to camera-based estimates.

### 3. Manipulation Skills (`policy/skills.py`)

Each skill is a finite state machine producing 13-DoF actions (12 arm joints + drawer force) at 50 Hz:

| Skill | Description |
|-------|-------------|
| `MoveTo` | Cartesian transit with lift/translate/descent phases |
| `Grasp` | Side/top-down grasp with jaw alignment via roll sweep |
| `Place` | Carry to goal, descend, release, retreat |
| `OpenDrawer` | Side-grasp handle post, pull drawer open |
| `Pour` | Tilt bottle over mug until pour detected |

**IK Solvers** (`policy/ik.py`):
- Damped least-squares position IK (5-DoF)
- Oriented IK (position + approach axis, 5-DoF)
- Fixed-roll IK (4-DoF position with pinned wrist_roll)
- Randomized restarts for global convergence

### 4. Orchestrator (`policy/orchestrator.py`)

- **Planner**: Rule-based natural language → skill sequence (VLA-ready stub)
- **Executor**: Runs skills to completion with perception updates
- **Perception Integration**: Skills consume poses via env pose provider
- **Evaluation**: Multi-seed success rate reporting

### 5. VLA Integration (Planned)

The architecture is designed for VLA policy swap-in:

```python
# Current: Classical skills
orchestrator = Orchestrator(env, perception)
orchestrator.execute_instruction("Set the table for dinner")

# Future: VLA policy (SmolVLA, Pi0.5, ACT)
vla_policy = load_vla_model("openvino_model.xml")
actions = vla_policy(obs, instruction)
```

The perception module (`PerceptionModule.update()`) is the single swap point for a learned perception backbone.

---

## Challenge Requirements Mapping

| Requirement | Implementation |
|-------------|----------------|
| Bimanual manipulation | Two SO-101 arms with coordinated skills |
| Multi-modal reasoning | Orchestrator parses NL → skill plans; perception fuses RGB-D |
| Robustness under perturbation | Domain randomization in env; evaluation across 10 seeds |
| Simulation training | LeRobot-compatible env; policy/skills as imitation baseline |
| Intel Edge Optimization | OpenVINO benchmark script; model compilation for CPU/GPU/NPU |
| Reproducible repo | Setup.py, requirements, entry points, deterministic seeds |
| MuJoCo simulation | `envs/dinner_table_env.py` with full randomization |
| Benchmark script | `scripts/benchmark_openvino.py` |
| Demo video | `scripts/run_demo.py --video` |
| Technical README | This file |

---

## Key Features

### Domain Randomization
Per-seed randomization of:
- Object positions (±3 cm), yaw (±0.4 rad)
- Friction (0.7–1.3×), mass (0.75–1.3×), size (0.92–1.08×)
- Colors (±0.05 RGB), background hue, lighting

### Honest Physics
- Drawer only moves when gripper physically holds handle (`gripper_handles_drawer()`)
- Grasp contact detected via MuJoCo contact pairs
- No privileged state in skill execution (except for debugging)

### Intel Optimization Ready
- OpenVINO benchmark measures latency, throughput, device utilization
- Async inference queue for sustained throughput
- FP16/INT8 precision support via model conversion
- Performance hints for Intel CPU/GPU/NPU

---

## Evaluation

Run the standard evaluation across 10 randomized seeds:

```bash
python scripts/run_demo.py --eval-only --seeds 10
```

Expected outputs:
- Per-seed success/failure
- Overall success rate
- Per-skill timing and status
- Task state (object placement, mug filled)

---

## Extending for VLA Policies

To integrate a VLA policy (SmolVLA, Pi0.5, ACT):

1. **Convert model to OpenVINO IR**:
   ```bash
   mo --input_model model.onnx --output_dir models/vla_openvino
   ```

2. **Replace orchestrator execution**:
   ```python
   from policy.orchestrator import Orchestrator
   
   class VLAOrchestrator(Orchestrator):
       def __init__(self, env, perception, vla_model_path):
           super().__init__(env, perception)
           self.vla = self.core.compile_model(vla_model_path, "CPU")
       
       def execute_instruction(self, instruction):
           obs = self._get_obs()
           action = self.vla({"obs": obs, "instruction": instruction})
           return self._execute_actions(action)
   ```

3. **Benchmark**:
   ```bash
   python scripts/benchmark_openvino.py --model-path models/vla_openvino/vla.xml --device CPU
   ```

---

## Citation

If you use this codebase, please cite the Intel Physical AI Online Challenge.

---

## License

MIT License - See LICENSE file for details.