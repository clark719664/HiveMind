"""Phase 2.5: Bitcoin-Inspired Proof-of-Useful-Work (PoUW) Cryptographic Ledger (`$HIVE`).

Implements:
- 21,000,000 `$HIVE` maximum supply cap with 50.0 `$HIVE` initial block reward and epoch halving
  every 10,000 outer DiLoCo steps.
- Multi-pool block reward distribution across:
    1. GPU Training Miners (60% training + 15% inference reserve pool = 75%)
    2. Dataset & Expert Storage Seeders (15% Proof-of-Storage pool)
    3. Spot-Check Validators (10% Proof-of-Verification pool)
- Automatic Byzantine slashing (50% balance slash, reputation penalty, and `slashed_registry` ban)
  when a worker fails Deterministic Spot-Check replay or Multi-Krum filtering.
- Circular-economy P2P inference micro-fee settlement (`settle_swarm_inference`).
- Phase S-5 Ledger Hardening (`LEDGER-01` .. `LEDGER-04`):
    * Blocks negative/zero token counts or negative fee rates in `settle_swarm_inference`.
    * Blocks negative/non-finite `credit_reward` amounts and bars slashed Byzantine nodes from
      receiving rewards or provider fees.
    * Deduplicates per-block `WorkReceipt`, `StorageProofReceipt`, and validator IDs to prevent
      receipt-flooding inflation, and enforces the 21,000,000 `$HIVE` supply ceiling.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from hivemind.security.swarm_runtime_hardening import SwarmSecurityViolationError


@dataclass
class WorkReceipt:
    """Cryptographic receipt of a worker node's DiLoCo-QLoRA inner-loop contribution."""

    node_id: str
    steam_id64: str
    branch_name: str
    flops_contributed: float
    cosine_alignment: float
    spot_check_passed: bool
    multikrum_accepted: bool
    delta_sha256: str


@dataclass
class StorageProofReceipt:
    """Merkle-tree byte-range Proof-of-Storage receipt for a pinned Hugging Face shard."""

    node_id: str
    shard_cid: str
    challenged_chunk_index: int
    merkle_proof_hash: str
    bytes_stored: int
    verified: bool


@dataclass
class InferenceSettlementReceipt:
    """Micro-fee settlement receipt for a P2P speculative MoE inference session."""

    consumer_node_id: str
    expert_provider_nodes: List[str]
    domain_expert: str
    verified_tokens: int
    rtt_speedup: float
    fee_paid_hive: float


@dataclass
class HivePoUWBlock:
    """Single block in the `$HIVE` Proof-of-Useful-Work hash-chain."""

    block_height: int
    prev_block_hash: str
    timestamp: float
    outer_step: int
    expert_cid: str
    merkle_root: str
    total_block_reward: float
    reward_allocations: Dict[str, float]
    slashed_nodes: List[str]
    block_hash: str = ""

    def compute_hash(self) -> str:
        payload = {
            "block_height": self.block_height,
            "prev_block_hash": self.prev_block_hash,
            "timestamp": round(self.timestamp, 4),
            "outer_step": self.outer_step,
            "expert_cid": self.expert_cid,
            "merkle_root": self.merkle_root,
            "total_block_reward": round(self.total_block_reward, 8),
            "reward_allocations": {
                k: round(v, 8) for k, v in sorted(self.reward_allocations.items())
            },
            "slashed_nodes": sorted(self.slashed_nodes),
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


class HiveSwarmLedger:
    """Bitcoin-inspired Proof-of-Useful-Work (PoUW) Ledger for Project HiveMind.

    Rewards verified DiLoCo-QLoRA training, Merkle dataset storage, and MoE inference.
    """

    MAX_SUPPLY_HIVE = 21_000_000.0
    INITIAL_BLOCK_REWARD = 50.0
    HALVING_INTERVAL_BLOCKS = 10_000

    def __init__(self, ledger_path: Optional[Path] = None) -> None:
        self.ledger_path = ledger_path
        self.chain: List[HivePoUWBlock] = []
        self.balances: Dict[str, float] = {}
        self.reputation_scores: Dict[str, int] = {}
        self.slashed_registry: List[str] = []
        self.settlement_history: List[InferenceSettlementReceipt] = []
        self._init_genesis_block()

    def _init_genesis_block(self) -> None:
        genesis = HivePoUWBlock(
            block_height=0,
            prev_block_hash="0" * 64,
            timestamp=1700000000.0,
            outer_step=0,
            expert_cid="cid://sha256/genesis-qwen25-nf4-backbone",
            merkle_root=hashlib.sha256(b"HiveMind-SETI-Swarm-Genesis").hexdigest(),
            total_block_reward=0.0,
            reward_allocations={},
            slashed_nodes=[],
        )
        genesis.block_hash = genesis.compute_hash()
        self.chain.append(genesis)

    def total_circulating_supply(self) -> float:
        """Return total `$HIVE` currently held across all node balances."""
        return round(sum(self.balances.values()), 6)

    def block_reward_at_height(self, height: int) -> float:
        if height < 0:
            return 0.0
        halvings = height // self.HALVING_INTERVAL_BLOCKS
        if halvings >= 64:
            return 0.0
        raw_reward = self.INITIAL_BLOCK_REWARD * (0.5**halvings)
        remaining_cap = max(0.0, self.MAX_SUPPLY_HIVE - self.total_circulating_supply())
        return round(min(raw_reward, remaining_cap), 6)

    @staticmethod
    def _compute_merkle_root(items: List[str]) -> str:
        if not items:
            return hashlib.sha256(b"empty").hexdigest()
        layer = [hashlib.sha256(x.encode("utf-8")).hexdigest() for x in items]
        while len(layer) > 1:
            if len(layer) % 2 == 1:
                layer.append(layer[-1])
            next_layer = []
            for i in range(0, len(layer), 2):
                combined = (layer[i] + layer[i + 1]).encode("utf-8")
                next_layer.append(hashlib.sha256(combined).hexdigest())
            layer = next_layer
        return layer[0]

    def mine_pouw_block(
        self,
        outer_step: int,
        expert_cid: str,
        work_receipts: List[WorkReceipt],
        storage_receipts: List[StorageProofReceipt],
        validator_node_ids: List[str],
    ) -> HivePoUWBlock:
        height = len(self.chain)
        reward = self.block_reward_at_height(height)
        allocations: Dict[str, float] = {}
        slashed: List[str] = []

        # Rule LEDGER-03: Deduplicate work receipts per node_id & reject negative/NaN FLOPs
        dedup_work: List[WorkReceipt] = []
        seen_workers: set[str] = set()
        for w in work_receipts:
            if not math.isfinite(float(w.flops_contributed)) or float(w.flops_contributed) < 0.0:
                raise SwarmSecurityViolationError(
                    "LEDGER-03",
                    "INVALID_NEGATIVE_OR_NAN_FLOPS_RECEIPT",
                    f"Worker '{w.node_id}' submitted invalid flops_contributed={w.flops_contributed}.",
                )
            if w.node_id in seen_workers:
                raise SwarmSecurityViolationError(
                    "LEDGER-03",
                    "DUPLICATE_WORK_RECEIPT_IN_BLOCK",
                    f"Duplicate WorkReceipt for worker '{w.node_id}' in block height {height}.",
                )
            seen_workers.add(w.node_id)
            dedup_work.append(w)

        # 1. Process GPU Training Miners (60% of Block Reward + 15% Inference Reserve Pool)
        gpu_pool = 0.75 * reward
        valid_workers = [
            w
            for w in dedup_work
            if w.spot_check_passed
            and w.multikrum_accepted
            and w.node_id not in self.slashed_registry
            and float(w.flops_contributed) > 0.0
        ]
        for w in dedup_work:
            if not w.spot_check_passed or not w.multikrum_accepted:
                slashed.append(w.node_id)
                if w.node_id not in self.slashed_registry:
                    self.slashed_registry.append(w.node_id)
                self.reputation_scores[w.node_id] = self.reputation_scores.get(w.node_id, 100) - 50
                # Slash 50% of any existing balance for Byzantine behavior
                if w.node_id in self.balances:
                    self.balances[w.node_id] = round(self.balances[w.node_id] * 0.5, 6)
                else:
                    self.balances[w.node_id] = 0.0

        total_weighted_flops = sum(
            w.flops_contributed * max(0.05, w.cosine_alignment) for w in valid_workers
        )
        if total_weighted_flops > 0:
            for w in valid_workers:
                weight = (
                    w.flops_contributed * max(0.05, w.cosine_alignment)
                ) / total_weighted_flops
                earned = round(gpu_pool * weight, 6)
                allocations[w.node_id] = round(allocations.get(w.node_id, 0.0) + earned, 6)
                self.reputation_scores[w.node_id] = self.reputation_scores.get(w.node_id, 100) + 10

        # 2. Process Proof-of-Storage Seeders (15% of Block Reward)
        storage_pool = 0.15 * reward
        seen_seeders: set[str] = set()
        valid_seeders: List[StorageProofReceipt] = []
        for s in storage_receipts:
            if int(s.bytes_stored) < 0:
                raise SwarmSecurityViolationError(
                    "LEDGER-03",
                    "NEGATIVE_STORAGE_BYTES_RECEIPT",
                    f"Seeder '{s.node_id}' submitted negative bytes_stored={s.bytes_stored}.",
                )
            if s.verified and s.node_id not in self.slashed_registry and s.node_id not in seen_seeders and int(s.bytes_stored) > 0:
                seen_seeders.add(s.node_id)
                valid_seeders.append(s)

        total_bytes = sum(s.bytes_stored for s in valid_seeders)
        if total_bytes > 0:
            for s in valid_seeders:
                earned = round(storage_pool * (s.bytes_stored / total_bytes), 6)
                allocations[s.node_id] = round(allocations.get(s.node_id, 0.0) + earned, 6)

        # 3. Process Spot-Check Validators (10% of Block Reward)
        validator_pool = 0.10 * reward
        unique_validators = [
            vid for vid in dict.fromkeys(validator_node_ids) if vid not in self.slashed_registry
        ]
        if unique_validators:
            per_val = round(validator_pool / len(unique_validators), 6)
            for vid in unique_validators:
                allocations[vid] = round(allocations.get(vid, 0.0) + per_val, 6)

        # Commit balances
        for node_id, amount in allocations.items():
            self.balances[node_id] = round(self.balances.get(node_id, 0.0) + amount, 6)

        merkle_leaves = (
            [w.delta_sha256 for w in dedup_work]
            + [s.merkle_proof_hash for s in storage_receipts]
            + [expert_cid]
        )
        merkle_root = self._compute_merkle_root(merkle_leaves)

        block = HivePoUWBlock(
            block_height=height,
            prev_block_hash=self.chain[-1].block_hash,
            timestamp=time.time(),
            outer_step=outer_step,
            expert_cid=expert_cid,
            merkle_root=merkle_root,
            total_block_reward=reward,
            reward_allocations=allocations,
            slashed_nodes=slashed,
        )
        block.block_hash = block.compute_hash()
        self.chain.append(block)
        self._persist()
        return block

    def settle_swarm_inference(
        self,
        consumer_node_id: str,
        provider_node_ids: List[str],
        domain_expert: str,
        verified_tokens: int,
        rtt_speedup: float,
        rate_per_token: float = 0.005,
    ) -> InferenceSettlementReceipt:
        # Rule LEDGER-01: Prevent negative token / negative rate balance minting exploit
        if int(verified_tokens) <= 0:
            raise SwarmSecurityViolationError(
                "LEDGER-01",
                "NEGATIVE_OR_ZERO_INFERENCE_TOKENS",
                f"verified_tokens ({verified_tokens}) must be a positive integer >= 1.",
            )
        if not math.isfinite(float(rate_per_token)) or float(rate_per_token) <= 0.0 or float(rate_per_token) > 10.0:
            raise SwarmSecurityViolationError(
                "LEDGER-01",
                "INVALID_INFERENCE_FEE_RATE",
                f"rate_per_token ({rate_per_token}) must be finite and within (0.0, 10.0].",
            )
        if not provider_node_ids:
            raise SwarmSecurityViolationError(
                "LEDGER-01",
                "EMPTY_INFERENCE_PROVIDERS",
                "provider_node_ids must contain at least one active expert provider node.",
            )
        for pid in provider_node_ids:
            if pid in self.slashed_registry:
                raise SwarmSecurityViolationError(
                    "LEDGER-01",
                    "SLASHED_PROVIDER_SETTLEMENT_BLOCKED",
                    f"Provider node '{pid}' is in the Byzantine slashed_registry and cannot collect fees.",
                )

        total_fee = round(int(verified_tokens) * float(rate_per_token), 6)
        consumer_bal = self.balances.get(consumer_node_id, 0.0)
        if consumer_bal < total_fee:
            raise ValueError(
                f"Node {consumer_node_id} has insufficient $HIVE ({consumer_bal} < {total_fee})"
            )

        self.balances[consumer_node_id] = round(consumer_bal - total_fee, 6)
        share = round(total_fee / len(provider_node_ids), 6)
        for pid in provider_node_ids:
            self.balances[pid] = round(self.balances.get(pid, 0.0) + share, 6)

        receipt = InferenceSettlementReceipt(
            consumer_node_id=consumer_node_id,
            expert_provider_nodes=provider_node_ids,
            domain_expert=domain_expert,
            verified_tokens=int(verified_tokens),
            rtt_speedup=rtt_speedup,
            fee_paid_hive=total_fee,
        )
        self.settlement_history.append(receipt)
        self._persist()
        return receipt

    def credit_reward(
        self,
        node_id: str,
        amount_hive: float,
        reason: str = "",
    ) -> float:
        """Credit a peer reuse royalty or community bundle reward to `node_id`."""
        del reason
        if not node_id or not str(node_id).strip():
            raise SwarmSecurityViolationError(
                "LEDGER-02",
                "EMPTY_REWARD_NODE_ID",
                "Cannot credit $HIVE reward to an empty node_id.",
            )
        if node_id in self.slashed_registry:
            raise SwarmSecurityViolationError(
                "LEDGER-02",
                "SLASHED_NODE_REWARD_BLOCKED",
                f"Node '{node_id}' is in the Byzantine slashed_registry and cannot receive rewards.",
            )
        if not math.isfinite(float(amount_hive)) or float(amount_hive) <= 0.0:
            raise SwarmSecurityViolationError(
                "LEDGER-02",
                "NON_POSITIVE_OR_NON_FINITE_REWARD",
                f"amount_hive ({amount_hive}) must be finite and strictly positive (> 0.0).",
            )
        if self.total_circulating_supply() + float(amount_hive) > self.MAX_SUPPLY_HIVE:
            raise SwarmSecurityViolationError(
                "LEDGER-04",
                "MAX_SUPPLY_CAP_EXCEEDED",
                f"Crediting {amount_hive} $HIVE would exceed the 21,000,000 $HIVE supply cap.",
            )
        self.balances[node_id] = round(self.balances.get(node_id, 0.0) + float(amount_hive), 6)
        self.reputation_scores[node_id] = self.reputation_scores.get(node_id, 100) + 5
        self._persist()
        return self.balances[node_id]

    def get_balance(self, node_id: str) -> float:
        """Return the current `$HIVE` balance for `node_id`."""
        return round(self.balances.get(node_id, 0.0), 6)

    def verify_chain_integrity(self) -> bool:
        """Cryptographically verify every block hash and prev_block_hash link from genesis."""
        if not self.chain:
            return False
        for i, block in enumerate(self.chain):
            if block.compute_hash() != block.block_hash:
                return False
            if i > 0 and block.prev_block_hash != self.chain[i - 1].block_hash:
                return False
        return True

    def _persist(self) -> None:
        if self.ledger_path is not None:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "chain": [asdict(b) for b in self.chain],
                "balances": self.balances,
                "reputation_scores": self.reputation_scores,
                "slashed_registry": self.slashed_registry,
                "settlement_history": [asdict(s) for s in self.settlement_history],
            }
            self.ledger_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
