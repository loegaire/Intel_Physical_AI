#!/usr/bin/env python3
"""Setup script for Intel Physical AI Challenge package."""

from setuptools import setup, find_packages
from pathlib import Path

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

setup(
    name="intel-physical-ai",
    version="0.1.0",
    description="Intel Physical AI Online Challenge - Bimanual VLA Manipulation",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Challenge Participant",
    author_email="",
    url="https://github.com/loegaire/Intel_Physical_AI",
    packages=find_packages(include=["envs", "perception", "policy", "scripts"]),
    python_requires=">=3.10",
    install_requires=[
        "mujoco>=3.2.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
    ],
    extras_require={
        "openvino": ["openvino>=2024.0.0"],
        "ml": [
            "transformers>=4.36.0",
            "torch>=2.1.0",
            "accelerate>=0.25.0",
        ],
        "viz": [
            "matplotlib>=3.8.0",
            "imageio>=2.34.0",
            "imageio-ffmpeg>=0.4.9",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pyright>=1.1.380",
            "ruff>=0.1.0",
        ],
        "all": [
            "openvino>=2024.0.0",
            "transformers>=4.36.0",
            "torch>=2.1.0",
            "accelerate>=0.25.0",
            "matplotlib>=3.8.0",
            "imageio>=2.34.0",
            "imageio-ffmpeg>=0.4.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "physical-ai-demo=scripts.run_demo:main",
            "physical-ai-benchmark=scripts.benchmark_openvino:main",
            "physical-ai-study=scripts.study_workspace:main",
            "physical-ai-perception-check=scripts.check_perception:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    include_package_data=True,
    package_data={
        "envs": ["*.xml"],
        "assets": ["so101/*.stl", "so101/*.xml"],
    },
)