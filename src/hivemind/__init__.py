"""Project HiveMind: Decentralized MoE Training & Inference via Steam, BOINC, and DiLoCo."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

__version__ = "1.9.1"

__codename__ = "Project HiveMind"
