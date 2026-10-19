"""Phase 2.0: Distributed Disk Vault (Kademlia DHT + Content-Addressed Storage) & Proof-of-Storage.

Stores sharded Hugging Face datasets and converged LoRA domain adapters on local NVMe/SSD storage
(`D:\\xdev\\DecentralizedAI\\.cache\\swarm_store`), tracks R=3 peer replication pins, and answers
block-hash-seeded Merkle Byte-Range Proof-of-Storage challenges.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from hivemind.swarm.ledger import StorageProofReceipt


class ContentAddressedSwarmStore:
    """Local Agent Disk Vault & Kademlia-style Content-Addressed Storage (CAS).

    Stores sharded Hugging Face datasets and converged LoRA domain adapters on disk,
    and generates deterministic Merkle Proof-of-Storage receipts.
    """

    SHARD_BLOCK_ALIGNMENT_BYTES = 16_384

    def __init__(self, root_dir: Path, replication_factor: int = 3) -> None:
        self.root_dir = Path(root_dir)
        self.replication_factor = replication_factor
        self.shards_dir = self.root_dir / "dataset_shards"
        self.experts_dir = self.root_dir / "lora_experts"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.experts_dir.mkdir(parents=True, exist_ok=True)
        self.node_pins: Dict[str, List[str]] = {}
        self.cid_to_path: Dict[str, Path] = {}

    def store_hf_dataset_shard(
        self,
        node_id: str,
        domain_name: str,
        shard_id: int,
        samples: List[str],
    ) -> Tuple[str, Path, int]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", domain_name):
            raise ValueError("invalid domain name")
        payload = {
            "domain": domain_name,
            "shard_id": shard_id,
            "num_samples": len(samples),
            "samples": samples,
        }
        raw_json = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        # Align shard container to a fixed 16 KB CAS block boundary for uniform quota accounting
        rem = len(raw_json) % self.SHARD_BLOCK_ALIGNMENT_BYTES
        if rem != 0:
            pad_len = self.SHARD_BLOCK_ALIGNMENT_BYTES - rem
            raw_bytes = raw_json + (b" " * pad_len)
        else:
            raw_bytes = raw_json

        digest = hashlib.sha256(raw_bytes).hexdigest()
        cid = f"cid://sha256/{digest[:32]}"

        shard_path = self.shards_dir / f"{domain_name}_shard_{shard_id}_{digest[:12]}.json"
        shard_path.write_bytes(raw_bytes)

        self.node_pins.setdefault(node_id, []).append(cid)
        self.cid_to_path[cid] = shard_path
        return cid, shard_path, len(raw_bytes)

    def verify_proof_of_storage(
        self,
        node_id: str,
        shard_cid: str,
        shard_path: Path,
        prev_block_hash: str,
        chunk_size: int = 256,
    ) -> StorageProofReceipt:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        raw_bytes = Path(shard_path).read_bytes()
        actual_cid = f"cid://sha256/{hashlib.sha256(raw_bytes).hexdigest()[:32]}"
        chunks = [
            raw_bytes[i : i + chunk_size]
            for i in range(0, max(1, len(raw_bytes)), chunk_size)
        ]
        challenge_seed = hashlib.sha256(
            f"{prev_block_hash}:{node_id}:{shard_cid}".encode("utf-8")
        ).hexdigest()
        challenged_idx = int(challenge_seed, 16) % len(chunks)

        challenged_chunk = chunks[challenged_idx]
        merkle_proof = hashlib.sha256(
            challenged_chunk + prev_block_hash.encode("utf-8")
        ).hexdigest()

        return StorageProofReceipt(
            node_id=node_id,
            shard_cid=shard_cid,
            challenged_chunk_index=challenged_idx,
            merkle_proof_hash=merkle_proof,
            bytes_stored=len(raw_bytes),
            verified=(shard_cid == actual_cid and shard_cid in self.node_pins.get(node_id, [])),
        )

    def publish_lora_expert_checkpoint(
        self,
        domain_name: str,
        lora_state_dict: Dict[str, torch.Tensor],
        metadata: Dict[str, object],
    ) -> Tuple[str, Path]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", domain_name):
            raise ValueError("invalid domain name")
        cpu_state = {key: tensor.detach().cpu().to(torch.float16).contiguous()
                     for key, tensor in lora_state_dict.items()}
        digest = self._expert_digest(metadata, cpu_state)
        expert_cid = f"cid://sha256/expert-{domain_name.lower()}-{digest[:24]}"
        out_path = self.experts_dir / f"expert_{domain_name.lower()}_{digest[:12]}.pt"
        torch.save({"cid": expert_cid, "domain": domain_name, "metadata": metadata, "lora_state": cpu_state}, out_path)
        self.cid_to_path[expert_cid] = out_path
        return expert_cid, out_path

    def load_lora_expert_checkpoint(
        self,
        expert_cid_or_path: str | Path,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[str, Dict[str, object], Dict[str, torch.Tensor]]:
        """Retrieve a published LoRA domain expert from the local CAS vault by CID or file path."""
        if isinstance(expert_cid_or_path, Path) or not str(expert_cid_or_path).startswith("cid://"):
            ckpt_path = Path(expert_cid_or_path)
        else:
            cid_str = str(expert_cid_or_path)
            if cid_str in self.cid_to_path:
                ckpt_path = self.cid_to_path[cid_str]
            else:
                raise FileNotFoundError(f"Expert CID {cid_str} not found in local CAS index.")

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("invalid checkpoint payload")
        cid = payload.get("cid", "")
        meta = payload.get("metadata", {})
        raw_state = payload.get("lora_state", {})
        domain = payload.get("domain", "")
        if not isinstance(domain, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", domain):
            raise ValueError("checkpoint lacks a valid bound domain")
        digest = self._expert_digest(meta, raw_state)
        expected = f"cid://sha256/expert-{domain.lower()}-{digest[:24]}"
        if cid != expected or (str(expert_cid_or_path).startswith("cid://") and str(expert_cid_or_path) != cid):
            raise ValueError("checkpoint content does not match requested CID")
        lora_state = {k: v.to(dtype=dtype) for k, v in raw_state.items()}
        return cid, meta, lora_state

    @staticmethod
    def _expert_digest(metadata: Dict[str, object], state: Dict[str, torch.Tensor]) -> str:
        if not isinstance(metadata, dict) or not isinstance(state, dict) or not state:
            raise ValueError("checkpoint metadata and nonempty tensor state required")
        hasher = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode("utf-8"))
        for name in sorted(state):
            tensor = state[name]
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
                raise ValueError("checkpoint requires named finite tensors")
            tensor = tensor.detach().cpu().contiguous()
            descriptor = json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode("utf-8")
            hasher.update(len(descriptor).to_bytes(8, "big"))
            hasher.update(descriptor)
            raw = tensor.view(torch.uint8).numpy().tobytes()
            hasher.update(len(raw).to_bytes(8, "big"))
            hasher.update(raw)
        return hasher.hexdigest()
