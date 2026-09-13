#!/usr/bin/env python3
"""OpenVINO inference benchmark for Intel Physical AI challenge.

Benchmarks the VLA/VLM inference pipeline on Intel Core Ultra Series 2/3,
measuring latency, throughput, device utilization, and model precision.

Usage:
    python scripts/benchmark_openvino.py --model-path models/vla.xml --device CPU
    python scripts/benchmark_openvino.py --model-path models/vla.xml --device GPU --precision FP16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from openvino.runtime import Core, CompiledModel, InferRequest, Type
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    Core = CompiledModel = InferRequest = Type = object


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from envs.dinner_table_env import DinnerTableEnv
from perception.perception import PerceptionModule


class OpenVINOBenchmark:
    """Benchmark OpenVINO model inference performance."""

    def __init__(self, model_path: str, device: str = "CPU",
                 precision: str = "FP32", num_requests: int = 1):
        if not OPENVINO_AVAILABLE:
            raise RuntimeError("OpenVINO not available. Install openvino package.")

        self.model_path = Path(model_path)
        self.device = device.upper()
        self.precision = precision.upper()
        self.num_requests = num_requests

        self.core = Core()
        self.compiled_model: Optional[CompiledModel] = None
        self.input_shape = None
        self.output_shape = None

    def load_model(self) -> None:
        """Load and compile the model."""
        print(f"Loading model: {self.model_path}")
        print(f"Device: {self.device}, Precision: {self.precision}")

        model = self.core.read_model(str(self.model_path))

        # Set precision if needed
        if self.precision == "FP16":
            # Convert to FP16 if model supports it
            pass  # Model conversion would happen offline

        # Compile with performance hints
        config = {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": str(self.num_requests),
        }
        if self.device == "GPU":
            config["GPU_ENABLE_SDPA_OPTIMIZATION"] = "YES"

        self.compiled_model = self.core.compile_model(model, self.device, config)
        self.input_shape = tuple(self.compiled_model.input(0).shape)
        self.output_shape = tuple(self.compiled_model.output(0).shape)

        print(f"Input shape: {self.input_shape}")
        print(f"Output shape: {self.output_shape}")
        print(f"Optimal number of infer requests: {self.compiled_model.get_property('OPTIMAL_NUMBER_OF_INFER_REQUESTS')}")

    def generate_dummy_input(self, batch_size: int = 1) -> np.ndarray:
        """Generate dummy input matching model input shape."""
        shape = list(self.input_shape)
        if shape[0] == -1 or shape[0] == 1:
            shape[0] = batch_size
        return np.random.randn(*shape).astype(np.float32)

    def benchmark_latency(self, num_warmup: int = 10, num_runs: int = 100,
                          batch_size: int = 1) -> dict:
        """Benchmark inference latency."""
        if self.compiled_model is None:
            self.load_model()

        dummy_input = self.generate_dummy_input(batch_size)
        input_dict = {self.compiled_model.input(0): dummy_input}

        # Warmup
        print(f"Warming up ({num_warmup} runs)...")
        for _ in range(num_warmup):
            self.compiled_model(input_dict)

        # Benchmark
        print(f"Benchmarking latency ({num_runs} runs, batch={batch_size})...")
        latencies = []

        for _ in range(num_runs):
            start = time.perf_counter()
            self.compiled_model(input_dict)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        latencies = np.array(latencies)
        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p90_ms": float(np.percentile(latencies, 90)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }

    def benchmark_throughput(self, duration_sec: float = 10.0,
                             batch_size: int = 1) -> dict:
        """Benchmark sustained throughput."""
        if self.compiled_model is None:
            self.load_model()

        dummy_input = self.generate_dummy_input(batch_size)
        input_dict = {self.compiled_model.input(0): dummy_input}

        # Use async inference for throughput
        infer_queue = self.compiled_model.create_infer_queue()
        infer_queue.set_callback(lambda req, userdata: None)

        print(f"Benchmarking throughput ({duration_sec}s, batch={batch_size})...")
        start = time.perf_counter()
        count = 0

        while time.perf_counter() - start < duration_sec:
            infer_queue.start_async(input_dict, None)
            infer_queue.wait_all()
            count += self.num_requests

        elapsed = time.perf_counter() - start
        throughput = count / elapsed

        return {
            "duration_sec": elapsed,
            "total_inferences": count,
            "throughput_fps": throughput,
            "batch_size": batch_size,
        }

    def benchmark_async(self, num_requests: int = 4, duration_sec: float = 10.0,
                        batch_size: int = 1) -> dict:
        """Benchmark async inference pipeline."""
        if self.compiled_model is None:
            self.load_model()

        dummy_input = self.generate_dummy_input(batch_size)
        input_dict = {self.compiled_model.input(0): dummy_input}

        infer_queue = self.compiled_model.create_infer_queue()
        completed = 0

        def callback(req, userdata):
            nonlocal completed
            completed += 1

        infer_queue.set_callback(callback)

        print(f"Benchmarking async ({num_requests} requests, {duration_sec}s)...")
        start = time.perf_counter()

        # Keep pipeline full
        while time.perf_counter() - start < duration_sec:
            infer_queue.start_async(input_dict, None)

        infer_queue.wait_all()
        elapsed = time.perf_counter() - start
        throughput = completed / elapsed

        return {
            "duration_sec": elapsed,
            "completed": completed,
            "throughput_fps": throughput,
            "num_requests": num_requests,
            "batch_size": batch_size,
        }

    def get_model_info(self) -> dict:
        """Get model metadata."""
        if self.compiled_model is None:
            self.load_model()

        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "precision": self.precision,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "optimal_requests": self.compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"),
            "supported_properties": list(self.core.get_property(self.device, "SUPPORTED_PROPERTIES")),
        }


def benchmark_perception_pipeline(seeds: list[int] = None, verbose: bool = True) -> dict:
    """Benchmark the classical perception pipeline (no OpenVINO model)."""
    if seeds is None:
        seeds = [0, 1, 2]

    latencies = []
    detection_rates = {name: 0 for name in ["plate", "mug", "bottle", "spoon", "fork"]}

    for seed in seeds:
        env = DinnerTableEnv(seed=seed, randomize=True)
        env.reset()
        perception = PerceptionModule(env, resolution=448, update_every=1)

        # Benchmark perception update
        for _ in range(5):
            start = time.perf_counter()
            perception.update(force=True)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1000)

        # Check detection rates
        for name in detection_rates:
            if perception.object_pos(name) is not None:
                detection_rates[name] += 1

    return {
        "perception_latency_ms": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
        },
        "detection_rates": {k: v / len(seeds) for k, v in detection_rates.items()},
    }


def benchmark_skill_execution(seeds: list[int] = None, verbose: bool = True) -> dict:
    """Benchmark skill execution time (classical control)."""
    if seeds is None:
        seeds = [0]

    from policy.skills import MoveTo, OpenDrawer, Grasp, Place

    skill_times = {"MoveTo": [], "OpenDrawer": [], "Grasp": [], "Place": []}

    for seed in seeds:
        env = DinnerTableEnv(seed=seed, randomize=False)
        env.reset()

        # MoveTo
        start = time.perf_counter()
        skill = MoveTo(env, arm="B", target=[0.25, 0.03, 0.85])
        for _ in range(100):
            action = skill.act()
            env.step(action)
            if skill.status.name != "RUNNING":
                break
        skill_times["MoveTo"].append(time.perf_counter() - start)

        # OpenDrawer
        env.reset()
        start = time.perf_counter()
        skill = OpenDrawer(env, arm="A")
        for _ in range(500):
            action = skill.act()
            env.step(action)
            if skill.status.name != "RUNNING":
                break
        skill_times["OpenDrawer"].append(time.perf_counter() - start)

    return {k: {"mean_sec": float(np.mean(v)), "std_sec": float(np.std(v))}
            for k, v in skill_times.items() if v}


def main():
    parser = argparse.ArgumentParser(description="OpenVINO inference benchmark")
    parser.add_argument("--model-path", type=str, help="Path to OpenVINO model (.xml)")
    parser.add_argument("--device", type=str, default="CPU", choices=["CPU", "GPU", "NPU", "AUTO"])
    parser.add_argument("--precision", type=str, default="FP32", choices=["FP32", "FP16", "INT8"])
    parser.add_argument("--num-requests", type=int, default=1, help="Number of parallel infer requests")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=100, help="Benchmark runs")
    parser.add_argument("--duration", type=float, default=10.0, help="Throughput benchmark duration (sec)")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--perception-only", action="store_true", help="Benchmark perception pipeline only")
    parser.add_argument("--skills-only", action="store_true", help="Benchmark skill execution only")

    args = parser.parse_args()

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
    }

    if args.perception_only:
        print("Benchmarking perception pipeline...")
        results["perception"] = benchmark_perception_pipeline()
    elif args.skills_only:
        print("Benchmarking skill execution...")
        results["skills"] = benchmark_skill_execution()
    else:
        if not args.model_path:
            print("Error: --model-path required for OpenVINO benchmark")
            sys.exit(1)

        if not OPENVINO_AVAILABLE:
            print("Error: OpenVINO not available. Install with: pip install openvino")
            sys.exit(1)

        if not os.path.exists(args.model_path):
            print(f"Error: Model not found: {args.model_path}")
            sys.exit(1)

        bench = OpenVINOBenchmark(args.model_path, args.device, args.precision, args.num_requests)

        print("Loading model...")
        bench.load_model()

        print("\n--- Model Info ---")
        model_info = bench.get_model_info()
        results["model_info"] = model_info
        for k, v in model_info.items():
            print(f"  {k}: {v}")

        print("\n--- Latency Benchmark ---")
        latency = bench.benchmark_latency(args.warmup, args.runs, args.batch_size)
        results["latency"] = latency
        for k, v in latency.items():
            print(f"  {k}: {v:.2f}")

        print("\n--- Throughput Benchmark ---")
        throughput = bench.benchmark_throughput(args.duration, args.batch_size)
        results["throughput"] = throughput
        for k, v in throughput.items():
            print(f"  {k}: {v}")

        print("\n--- Async Benchmark ---")
        async_results = bench.benchmark_async(args.num_requests, args.duration, args.batch_size)
        results["async"] = async_results
        for k, v in async_results.items():
            print(f"  {k}: {v}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\nDone!")


if __name__ == "__main__":
    main()