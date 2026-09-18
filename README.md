# Intel Physical AI: SmolVLA Dinner-Table Pipeline

Closed-loop SmolVLA inference in a randomized MuJoCo scene with two SO-101 arms.
The runner builds LeRobot observations from simulator cameras and robot state,
executes policy actions in the environment, records the real control trajectory,
and writes machine-readable evaluation results.

## Current status

The pipeline itself is operational:

- local SmolVLA checkpoint loading and closed-loop MuJoCo inference work;
- single-seed and multi-seed runs use the same evaluation contract;
- a multi-seed run creates one `SmolVLARunner`, so model weights are loaded once
  and reused across all seeds;
- the policy action queue and environment are reset at every episode boundary;
- videos contain one frame per executed control step;
- `evaluation.json` reports full-task success and per-seed details.

The included checkpoint is a stock 6D SO-100/SmolVLA checkpoint. It controls one
selected arm and is **not fine-tuned for this dinner-table task**. A successful
pipeline run therefore proves that inference, action adaptation, simulation, and
artifact generation work; it does not by itself prove task completion.

`success` has one strict meaning throughout this repository: the plate, mug,
spoon, and fork are placed at their targets and the mug has been filled.

## Pipeline

```text
instruction + 3 RGB cameras + joint state
                    |
                    v
          LeRobot preprocessing
                    |
                    v
               SmolVLA
                    |
                    v
       6D / 12D / 13D action adapter
                    |
                    v
       MuJoCo step -> video + metrics
```

The action contract is explicit:

| Checkpoint action size | Simulator mapping |
| --- | --- |
| 6 | Selected arm; the other arm is held; drawer command is neutral |
| 12 | Both arms; drawer command is neutral |
| 13 | Both arms plus drawer command |

Actions with unsupported dimensions, `NaN`, or infinite values are rejected.
The stock SO-100 checkpoint uses degrees; the simulator uses radians, and the
adapter performs that conversion.

## Requirements

- Linux recommended; headless rendering is supported with EGL
- Python 3.10 or newer
- MuJoCo 3.x
- a PyTorch and torchvision pair compatible with `lerobot[smolvla]`
- enough disk space for the checkpoint and its Hugging Face base-model cache

Create an isolated environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

Run the tests before a demo:

```bash
pytest -q
```

The tests cover the VLA state/action contract, multi-seed runner reuse, and
perception behavior without loading the full model for every unit test.

## Run SmolVLA

The local checkpoint is stored at `models/smolvla`. Its configuration references
the SmolVLM base model, so that base model must also already exist in the local
Hugging Face cache for a fully offline run.

On the current headless setup, use offline mode to prevent Transformers from
making metadata requests for files such as `chat_template.jinja`:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl \
python scripts/run_demo.py \
  --policy smolvla \
  --checkpoint models/smolvla \
  --device cpu \
  --controlled-arm A \
  --joint-units degrees \
  --seed 0 \
  --max-steps 1000 \
  --video results/smolvla_seed_00.mp4
```

For a fast integration smoke test, change `--max-steps 1000` to
`--max-steps 1`. CPU inference works but can be slow; choose an available device
explicitly when using an accelerator.

### Ten seeds, one model load

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl \
python scripts/run_demo.py \
  --policy smolvla \
  --checkpoint models/smolvla \
  --device cpu \
  --controlled-arm A \
  --joint-units degrees \
  --seed 0 \
  --seeds 10 \
  --max-steps 1000 \
  --video \
  --output-dir results/smolvla_10_seeds
```

This loads the checkpoint once, then runs seeds 0 through 9. It writes:

```text
results/smolvla_10_seeds/
├── demo_seed_00.mp4
├── ...
├── demo_seed_09.mp4
└── evaluation.json
```

For metrics without video encoding:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl \
python scripts/run_demo.py \
  --policy smolvla \
  --checkpoint models/smolvla \
  --device cpu \
  --eval-only \
  --seeds 10 \
  --output-dir results/smolvla_eval
```

`--eval-only` defaults to ten seeds when `--seeds` is omitted and never records
video.

## Video behavior

The environment runs at 50 control steps per second and the recorder writes one
frame after every step at 50 FPS. Therefore:

```text
video duration in seconds = executed steps / 50
```

For example, 100 steps produce about 2 seconds of video, while 1000 steps
produce at most about 20 seconds. An episode that terminates early produces a
shorter video. Stopping the process before the writer closes can leave an invalid
or nearly empty MP4; rerun to a new filename instead of treating an old file in
`results/` as current evidence.

Useful checks:

```bash
ffprobe -v error \
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames,duration \
  -of default=noprint_wrappers=1 results/smolvla_seed_00.mp4

python -m json.tool results/smolvla_10_seeds/evaluation.json >/dev/null
```

## Evaluation output

For SmolVLA, each per-seed result includes:

- strict full-task `success`;
- executed `steps` and accumulated reward;
- checkpoint action dimension and `bimanual` status;
- selected arm and checkpoint joint units;
- mean, p50, p90, and p99 inference latency;
- final environment task state.

The multi-seed summary contains the seed list, number of successful episodes,
success rate, and all per-seed records.

## Main command-line options

| Option | Meaning |
| --- | --- |
| `--policy classical\|smolvla` | Select the policy implementation |
| `--checkpoint PATH` | SmolVLA checkpoint directory |
| `--device DEVICE` | PyTorch device, for example `cpu` or `cuda` |
| `--controlled-arm A\|B` | Arm controlled by a 6D checkpoint |
| `--joint-units degrees\|radians` | Units emitted by the checkpoint |
| `--seed N` | First seed |
| `--seeds N` | Number of consecutive seeds |
| `--max-steps N` | Maximum control steps per episode |
| `--video [PATH]` | Record one run, or enable per-seed videos |
| `--output-dir PATH` | Multi-seed videos and `evaluation.json` |
| `--eval-only` | Multi-seed evaluation without video |
| `--instruction TEXT` | Natural-language task instruction |
| `--quiet` | Reduce console output |

See the authoritative CLI at any time:

```bash
python scripts/run_demo.py --help
```

## Classical policy and data collection

The classical perception/skill stack remains available as a development
baseline. It is not the SmolVLA path and should not be presented as proof that
the learned policy solved the challenge.

```bash
MUJOCO_GL=egl python scripts/run_demo.py \
  --policy classical --seed 0 --video results/classical_seed_00.mp4
```

`--strict-perception` disables privileged pose fallback for the classical policy.

The dataset collector admits only complete-task successes by default. Use
`--keep-failures` only when failed trajectories are intentionally wanted:

```bash
MUJOCO_GL=egl python scripts/collect_expert_dataset.py \
  --seeds 10 \
  --strict-perception \
  --output-dir data/expert
```

## OpenVINO scope

The files under `models/vla_openvino/` and the conversion/benchmark scripts are
experimental deployment scaffolding. They are not evidence of an exported,
quality-equivalent SmolVLA policy.

The included IR can be used to validate benchmark plumbing:

```bash
python scripts/benchmark_openvino.py \
  --model-path models/vla_openvino/vla.xml \
  --device CPU \
  --output results/openvino_runtime.json
```

When `--input-npz` is omitted, every model input is synthetic. Such a run measures
runtime behavior only, not policy quality. A quality-relevant benchmark requires
an exported production model and an NPZ containing every named, already
preprocessed model input.

## Repository layout

```text
Intel_Physical_AI/
├── assets/                         SO-101 simulation assets
├── envs/
│   ├── dinner_table_env.py         MuJoCo environment and task metric
│   └── world.xml                   Scene, robots, objects, and cameras
├── perception/                     Classical RGB-D perception
├── policy/
│   ├── vla.py                      SmolVLA runner and action adapter
│   ├── orchestrator.py             Classical task sequencer
│   ├── skills.py                   Classical manipulation skills
│   └── ik.py                       Inverse kinematics
├── scripts/
│   ├── run_demo.py                 Demo and multi-seed evaluation
│   ├── collect_expert_dataset.py   Demonstration collector
│   ├── benchmark_openvino.py       OpenVINO runtime benchmark
│   └── export_smolvla_to_openvino.py
├── tests/                          Contract and regression tests
├── models/                         Local checkpoints and experimental exports
├── requirements.txt
└── setup.py
```

## Known limitations

- The bundled SmolVLA checkpoint is 6D and single-arm in this environment.
- It has not been fine-tuned or validated for full dinner-table success.
- CPU execution is suitable for verification but may be too slow for interactive
  real-time control.
- OpenVINO export and task-quality equivalence remain unverified.
- The classical skill stack is a development baseline, not a guaranteed expert.
- The classical demo includes a deterministic `ArrangeDinnerTable` fallback so
  the online challenge artifact is reproducible without a trained checkpoint.

## License

Released under the [MIT License](LICENSE).
