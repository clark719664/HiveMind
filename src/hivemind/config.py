"""Hardware & Algorithmic Configuration for Project HiveMind.

Automatically configures local disk caches on D:\\xdev\\DecentralizedAI\\.cache to protect C: drive,
enforces deterministic CUDA execution (CUBLAS_WORKSPACE_CONFIG=:4096:8), and manages dual hardware
profiles for the local NVIDIA GeForce RTX 2070 (8 GB VRAM, Turing SM 7.5) and 12 GB reference nodes.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / ".cache"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"


def ensure_local_drive_caches() -> dict[str, str]:
    """Redirect all heavy caches (HuggingFace, PyTorch, TMP, uv) to D:\\xdev\\DecentralizedAI\\.cache."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
        load_dotenv(Path.home() / ".env", override=False)
    except Exception:
        pass

    cache_dirs = {
        "HF_HOME": str(CACHE_ROOT / "huggingface"),
        "HF_HUB_CACHE": str(CACHE_ROOT / "huggingface" / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(CACHE_ROOT / "huggingface" / "hub"),
        "TORCH_HOME": str(CACHE_ROOT / "torch"),
        "UV_CACHE_DIR": str(CACHE_ROOT / "uv"),
        "TMP": str(CACHE_ROOT / "tmp"),
        "TEMP": str(CACHE_ROOT / "tmp"),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    }
    for env_key, dir_path in cache_dirs.items():
        if env_key not in ("CUBLAS_WORKSPACE_CONFIG", "HF_HUB_DISABLE_SYMLINKS_WARNING"):
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            os.environ[env_key] = dir_path
        else:
            os.environ.setdefault(env_key, dir_path)
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    return cache_dirs


def get_hf_token() -> str | None:
    """Resolve Hugging Face token from environment variables (.env, process env, or Windows HKCU)."""
    env_keys = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_TOKEN")
    for key in env_keys:
        val = os.environ.get(key)
        if val and val.strip():
            token = val.strip()
            os.environ["HF_TOKEN"] = token
            return token

    # Also check Windows User/Machine Registry in case `setx HF_TOKEN ...` was run in another shell
    if os.name == "nt":
        import winreg

        for hive, subkey in (
            (winreg.HKEY_CURRENT_USER, r"Environment"),
            (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
        ):
            try:
                with winreg.OpenKey(hive, subkey) as reg_key:
                    for k in env_keys:
                        try:
                            val, _ = winreg.QueryValueEx(reg_key, k)
                            if isinstance(val, str) and val.strip():
                                token = val.strip()
                                os.environ["HF_TOKEN"] = token
                                return token
                        except FileNotFoundError:
                            continue
            except OSError:
                continue

    try:
        from huggingface_hub import get_token

        tok = get_token()
        if tok:
            os.environ["HF_TOKEN"] = tok
            return tok
    except Exception:
        pass
    return None


# Apply cache redirection and token resolution immediately on import
ensure_local_drive_caches()
get_hf_token()


@dataclass
class VRAMBudget:
    """VRAM Allocation Budget for a DiLoCo-QLoRA Worker Node."""

    frozen_base_nf4_gb: float
    lora_trainable_r64_gb: float
    paged_adamw_8bit_gb: float
    activations_kv_cache_gb: float
    cuda_context_overhead_gb: float

    @property
    def total_peak_vram_gb(self) -> float:
        return round(
            self.frozen_base_nf4_gb
            + self.lora_trainable_r64_gb
            + self.paged_adamw_8bit_gb
            + self.activations_kv_cache_gb
            + self.cuda_context_overhead_gb,
            2,
        )


@dataclass
class HardwareProfile:
    """Hardware and Model Profile for Project HiveMind."""

    name: str
    description: str
    target_vram_gb: float
    available_vram_budget_gb: float
    hf_model_id: str
    compute_dtype: str = "float16"
    quantization_4bit: str = "nf4"
    double_quantization: bool = True
    lora_rank_r: int = 64
    lora_alpha: int = 128
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    micro_batch_size: int = 1
    seq_len: int = 512
    gradient_checkpointing: bool = True
    sim_inner_steps_H: int = 100
    sim_spot_check_interval_s: int = 25
    sim_workers_per_branch: int = 5
    sim_byzantine_workers: int = 1
    vram_budget: VRAMBudget = field(
        default_factory=lambda: VRAMBudget(
            frozen_base_nf4_gb=0.35,
            lora_trainable_r64_gb=0.08,
            paged_adamw_8bit_gb=0.08,
            activations_kv_cache_gb=0.35,
            cuda_context_overhead_gb=0.60,
        )
    )


@dataclass
class HiveMindConfig:
    """Master Configuration for Project HiveMind."""

    codename: str = "Project HiveMind"
    deterministic_cuda: bool = True
    cublas_workspace_config: str = ":4096:8"
    spot_check_interval_steps: int = 50
    spot_check_tolerance_eps: float = 1e-4
    idle_threshold_seconds: float = 30.0
    emergency_checkpoint_timeout_sec: float = 1.5
    expert_domains: list[str] = field(
        default_factory=lambda: [
            "Code",
            "Mathematics",
            "Creative_Writing",
            "STEM",
            "Legal_Medical",
            "Multilingual_FR",
            "Reasoning",
            "General",
        ]
    )
    inner_steps_H: int = 500
    inner_lr: float = 2e-4
    inner_weight_decay: float = 0.01
    outer_lr_eta: float = 0.7
    outer_momentum_mu: float = 0.9
    compression_topk_ratio: float = 0.20
    compression_dtype: str = "fp8_e4m3fn"
    multikrum_max_byzantine_fraction: float = 0.25
    cosine_outlier_sigma: float = 2.0
    num_experts_E: int = 8
    router_top_k: int = 2
    router_aux_loss_coef: float = 0.01
    profiles: dict[str, HardwareProfile] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: Path | str | None = None) -> HiveMindConfig:
        if yaml_path is None:
            yaml_path = PROJECT_ROOT / "configs" / "rtx2070_dual_profile.yaml"
        path = Path(yaml_path)
        if not path.exists():
            return cls()

        with path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        sys_cfg = raw.get("system", {})
        diloco_cfg = raw.get("diloco", {})
        btm_cfg = raw.get("btm_moe", {})
        raw_profiles = raw.get("profiles", {})

        profiles: dict[str, HardwareProfile] = {}
        for pname, pdata in raw_profiles.items():
            vb_raw = pdata.get("vram_breakdown_estimate_gb")
            if vb_raw:
                vram_budget = VRAMBudget(
                    frozen_base_nf4_gb=float(vb_raw["frozen_base_nf4_gb"]),
                    lora_trainable_r64_gb=float(vb_raw["lora_trainable_r64_gb"]),
                    paged_adamw_8bit_gb=float(vb_raw["paged_adamw_8bit_gb"]),
                    activations_kv_cache_gb=float(vb_raw["activations_kv_cache_gb"]),
                    cuda_context_overhead_gb=float(vb_raw["cuda_context_overhead_gb"]),
                )
            else:
                vram_budget = VRAMBudget(
                    frozen_base_nf4_gb=0.35,
                    lora_trainable_r64_gb=0.08,
                    paged_adamw_8bit_gb=0.08,
                    activations_kv_cache_gb=0.35,
                    cuda_context_overhead_gb=0.60,
                )

            profiles[pname] = HardwareProfile(
                name=pname,
                description=pdata.get("description", pname),
                target_vram_gb=float(pdata.get("target_vram_gb", 8.0)),
                available_vram_budget_gb=float(pdata.get("available_vram_budget_gb", 6.4)),
                hf_model_id=pdata.get("hf_model_id", "Qwen/Qwen2.5-0.5B"),
                compute_dtype=pdata.get("compute_dtype", "float16"),
                quantization_4bit=pdata.get("quantization_4bit", "nf4"),
                double_quantization=bool(pdata.get("double_quantization", True)),
                lora_rank_r=int(pdata.get("lora_rank_r", 64)),
                lora_alpha=int(pdata.get("lora_alpha", 128)),
                lora_target_modules=pdata.get(
                    "lora_target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                ),
                micro_batch_size=int(pdata.get("micro_batch_size", 1)),
                seq_len=int(pdata.get("seq_len", 512)),
                gradient_checkpointing=bool(pdata.get("gradient_checkpointing", True)),
                sim_inner_steps_H=int(pdata.get("sim_inner_steps_H", 100)),
                sim_spot_check_interval_s=int(pdata.get("sim_spot_check_interval_s", 25)),
                sim_workers_per_branch=int(pdata.get("sim_workers_per_branch", 5)),
                sim_byzantine_workers=int(pdata.get("sim_byzantine_workers", 1)),
                vram_budget=vram_budget,
            )

        return cls(
            codename=sys_cfg.get("codename", "Project HiveMind"),
            deterministic_cuda=bool(sys_cfg.get("deterministic_cuda", True)),
            cublas_workspace_config=sys_cfg.get("cublas_workspace_config", ":4096:8"),
            spot_check_interval_steps=int(sys_cfg.get("spot_check_interval_steps", 50)),
            spot_check_tolerance_eps=float(sys_cfg.get("spot_check_tolerance_eps", 1e-4)),
            idle_threshold_seconds=float(sys_cfg.get("idle_threshold_seconds", 30.0)),
            emergency_checkpoint_timeout_sec=float(
                sys_cfg.get("emergency_checkpoint_timeout_sec", 1.5)
            ),
            expert_domains=sys_cfg.get(
                "expert_domains",
                [
                    "Code",
                    "Mathematics",
                    "Creative_Writing",
                    "STEM",
                    "Legal_Medical",
                    "Multilingual_FR",
                    "Reasoning",
                    "General",
                ],
            ),
            inner_steps_H=int(diloco_cfg.get("inner_steps_H", 500)),
            inner_lr=float(diloco_cfg.get("inner_lr", 2e-4)),
            inner_weight_decay=float(diloco_cfg.get("inner_weight_decay", 0.01)),
            outer_lr_eta=float(diloco_cfg.get("outer_lr_eta", 0.7)),
            outer_momentum_mu=float(diloco_cfg.get("outer_momentum_mu", 0.9)),
            compression_topk_ratio=float(diloco_cfg.get("compression_topk_ratio", 0.20)),
            compression_dtype=diloco_cfg.get("compression_dtype", "fp8_e4m3fn"),
            multikrum_max_byzantine_fraction=float(
                diloco_cfg.get("multikrum_max_byzantine_fraction", 0.25)
            ),
            cosine_outlier_sigma=float(diloco_cfg.get("cosine_outlier_sigma", 2.0)),
            num_experts_E=int(btm_cfg.get("num_experts_E", 8)),
            router_top_k=int(btm_cfg.get("router_top_k", 2)),
            router_aux_loss_coef=float(btm_cfg.get("router_aux_loss_coef", 0.01)),
            profiles=profiles,
        )


def configure_deterministic_runtime(seed: int = 42, warn_only: bool = True) -> None:
    """Enforce deterministic CUDA/CPU kernels for BOINC Spot-Check Verification."""
    import torch

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)


def detect_local_hardware() -> dict[str, Any]:
    """Inspect local GPU (CUDA / NVIDIA-SMI) and system RAM for node assignment."""
    import subprocess
    import psutil
    import torch

    vm = psutil.virtual_memory()
    info: dict[str, Any] = {
        "system_ram_total_gb": round(vm.total / (1024**3), 2),
        "system_ram_available_gb": round(vm.available / (1024**3), 2),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": "CPU (CUDA not linked in active Python)" if not torch.cuda.is_available() else "",
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "compute_capability": None,
        "supports_native_bf16": False,
        "recommended_dtype": "float16",
        "assigned_backbone_tier": "7B/9B (8GB Consumer Node)",
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free_b, total_b = torch.cuda.mem_get_info(0)
        cc = f"{props.major}.{props.minor}"
        supports_bf16 = props.major >= 8
        info.update(
            {
                "gpu_name": props.name,
                "vram_total_gb": round(total_b / (1024**3), 2),
                "vram_free_gb": round(free_b / (1024**3), 2),
                "compute_capability": cc,
                "supports_native_bf16": supports_bf16,
                "recommended_dtype": "bfloat16" if supports_bf16 else "float16",
                "assigned_backbone_tier": (
                    "14B (12GB+ Target Node)"
                    if (total_b / (1024**3)) >= 11.5
                    else "7B/9B or 8x0.5B/1.5B MoE (8GB Consumer Node)"
                ),
            }
        )
    else:
        # Query nvidia-smi if torch is currently CPU-only
        try:
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                if len(parts) >= 4:
                    total_gb = round(float(parts[1]) / 1024.0, 2)
                    free_gb = round(float(parts[2]) / 1024.0, 2)
                    cc = parts[3]
                    major = int(cc.split(".")[0]) if "." in cc else 7
                    info.update(
                        {
                            "gpu_name": f"{parts[0]} (Detected via NVML/nvidia-smi)",
                            "vram_total_gb": total_gb,
                            "vram_free_gb": free_gb,
                            "compute_capability": cc,
                            "supports_native_bf16": major >= 8,
                            "recommended_dtype": "bfloat16" if major >= 8 else "float16",
                        }
                    )
        except Exception:
            pass

    return info
