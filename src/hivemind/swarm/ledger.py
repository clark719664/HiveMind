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
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


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
    A configured ledger file is loaded and validated on restart; corrupt files
    are rejected, never silently replaced with an empty ledger. This local
    snapshot is not an authenticated distributed consensus database.
    """

    MAX_SUPPLY_HIVE = 21_000_000.0
    INITIAL_BLOCK_REWARD = 50.0
    HALVING_INTERVAL_BLOCKS = 10_000

    def __init__(self, ledger_path: Optional[Path] = None) -> None:
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        self.chain: List[HivePoUWBlock] = []
        self.balances: Dict[str, float] = {}
        self.reputation_scores: Dict[str, int] = {}
        self.slashed_registry: List[str] = []
        self.settlement_history: List[InferenceSettlementReceipt] = []
        self._init_genesis_block()
        if self.ledger_path is not None and (self.ledger_path.exists() or self.ledger_path.is_symlink()):
            self._load_persisted_state()

    @staticmethod
    def _make_genesis_block() -> HivePoUWBlock:
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
        return genesis

    def _init_genesis_block(self) -> None:
        self.chain.append(self._make_genesis_block())

    @staticmethod
    def _snapshot_digest(data: dict) -> str:
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_persisted_state(self) -> None:
        """Read current snapshots and the original unversioned schema without rewriting either."""
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate JSON key: {key}")
                result[key] = value
            return result

        def reject_constant(value):
            raise ValueError(f"Non-finite JSON constant: {value}")

        def text(value, label):
            if not isinstance(value, str) or not value:
                raise ValueError(f"Invalid {label}.")

        def number(value, label):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {label}.")

        def names(values, label, *, unique=True):
            if not isinstance(values, list):
                raise ValueError(f"Invalid {label}.")
            for value in values:
                text(value, label)
            if unique and len(set(values)) != len(values):
                raise ValueError(f"Duplicate {label}.")

        try:
            raw = json.loads(self.ledger_path.read_text(encoding="utf-8"),
                             object_pairs_hook=unique_object, parse_constant=reject_constant)
            if not isinstance(raw, dict):
                raise ValueError("Ledger snapshot must be an object.")
            required = {"chain", "balances", "reputation_scores", "slashed_registry", "settlement_history"}
            version = raw.get("schema_version", 0)
            if type(version) is not int or version not in (0, 1):
                raise ValueError("Unsupported ledger schema version.")
            if version == 0:
                # Legacy files contained precisely these five fields and no checksum.
                if set(raw) != required:
                    raise ValueError("Malformed legacy ledger snapshot.")
            else:
                if set(raw) != required | {"schema_version", "snapshot_sha256"}:
                    raise ValueError("Malformed versioned ledger snapshot.")
                unsigned = {key: value for key, value in raw.items() if key != "snapshot_sha256"}
                if raw["snapshot_sha256"] != self._snapshot_digest(unsigned):
                    raise ValueError("Ledger snapshot checksum mismatch.")
            for field in ("balances", "reputation_scores"):
                if not isinstance(raw[field], dict):
                    raise ValueError(f"Invalid {field} map.")
                for node_id, value in raw[field].items():
                    text(node_id, "node ID")
                    if field == "balances":
                        number(value, "balance")
                    elif type(value) is not int:
                        raise ValueError("Reputation scores must be integers.")
            names(raw["slashed_registry"], "slashed node ID")
            if not isinstance(raw["chain"], list) or not raw["chain"]:
                raise ValueError("Ledger chain must be nonempty.")
            chain = [HivePoUWBlock(**entry) for entry in raw["chain"]]
            for index, block in enumerate(chain):
                if type(block.block_height) is not int or block.block_height != index:
                    raise ValueError("Non-contiguous block heights.")
                if type(block.outer_step) is not int or block.outer_step < 0:
                    raise ValueError("Invalid block outer step.")
                number(block.timestamp, "block timestamp")
                number(block.total_block_reward, "block reward")
                text(block.expert_cid, "expert CID")
                for label in ("prev_block_hash", "merkle_root", "block_hash"):
                    digest = getattr(block, label)
                    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                        raise ValueError(f"Invalid {label}.")
                if not isinstance(block.reward_allocations, dict):
                    raise ValueError("Invalid reward allocations.")
                for node_id, amount in block.reward_allocations.items():
                    text(node_id, "reward node ID")
                    number(amount, "reward allocation")
                names(block.slashed_nodes, "block slashed node ID", unique=False)
                if block.compute_hash() != block.block_hash:
                    raise ValueError("Invalid block hash.")
                if index and block.prev_block_hash != chain[index - 1].block_hash:
                    raise ValueError("Broken block hash link.")
            if chain[0] != self._make_genesis_block():
                raise ValueError("Unexpected genesis block.")
            if not isinstance(raw["settlement_history"], list):
                raise ValueError("Invalid settlement history.")
            settlements = [InferenceSettlementReceipt(**entry) for entry in raw["settlement_history"]]
            for receipt in settlements:
                text(receipt.consumer_node_id, "settlement consumer")
                text(receipt.domain_expert, "settlement domain")
                names(receipt.expert_provider_nodes, "settlement provider")
                if not receipt.expert_provider_nodes or type(receipt.verified_tokens) is not int or receipt.verified_tokens <= 0:
                    raise ValueError("Invalid settlement provider/token count.")
                number(receipt.rtt_speedup, "settlement RTT speedup")
                number(receipt.fee_paid_hive, "settlement fee")
        except (ValueError, TypeError, KeyError, OverflowError, UnicodeError) as exc:
            raise ValueError(f"Invalid persisted ledger: {exc}") from exc
        # Publish only after every field has passed validation.
        self.chain = chain
        self.balances = raw["balances"]
        self.reputation_scores = raw["reputation_scores"]
        self.slashed_registry = raw["slashed_registry"]
        self.settlement_history = settlements

    def block_reward_at_height(self, height: int) -> float:
        halvings = height // self.HALVING_INTERVAL_BLOCKS
        if halvings >= 64:
            return 0.0
        return self.INITIAL_BLOCK_REWARD * (0.5**halvings)

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

        # 1. Process GPU Training Miners (60% of Block Reward + 15% Inference Reserve Pool)
        gpu_pool = 0.75 * reward
        valid_workers = [
            w for w in work_receipts if w.spot_check_passed and w.multikrum_accepted
        ]
        for w in work_receipts:
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
        valid_seeders = [s for s in storage_receipts if s.verified]
        total_bytes = sum(s.bytes_stored for s in valid_seeders)
        if total_bytes > 0:
            for s in valid_seeders:
                earned = round(storage_pool * (s.bytes_stored / total_bytes), 6)
                allocations[s.node_id] = round(allocations.get(s.node_id, 0.0) + earned, 6)

        # 3. Process Spot-Check Validators (10% of Block Reward)
        validator_pool = 0.10 * reward
        if validator_node_ids:
            per_val = round(validator_pool / len(validator_node_ids), 6)
            for vid in validator_node_ids:
                allocations[vid] = round(allocations.get(vid, 0.0) + per_val, 6)

        # Commit balances
        for node_id, amount in allocations.items():
            self.balances[node_id] = round(self.balances.get(node_id, 0.0) + amount, 6)

        merkle_leaves = (
            [w.delta_sha256 for w in work_receipts]
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
        if type(verified_tokens) is not int or verified_tokens <= 0:
            raise ValueError("Verified token count must be a positive integer.")
        if not math.isfinite(rate_per_token) or rate_per_token < 0:
            raise ValueError("Token rate must be finite and nonnegative.")
        if not math.isfinite(rtt_speedup) or rtt_speedup < 0:
            raise ValueError("RTT speedup must be finite and nonnegative.")
        if not isinstance(consumer_node_id, str) or not consumer_node_id:
            raise ValueError("A consumer node ID is required.")
        providers = list(provider_node_ids)
        if (not providers or any(not isinstance(pid, str) or not pid for pid in providers)
                or len(set(providers)) != len(providers)):
            raise ValueError("Settlement requires distinct, nonempty provider IDs.")
        try:
            total_fee = round(verified_tokens * rate_per_token, 6)
        except OverflowError as exc:
            raise ValueError("Settlement fee is not finite.") from exc
        if not math.isfinite(total_fee):
            raise ValueError("Settlement fee is not finite.")
        for node_id in set(providers) | {consumer_node_id}:
            balance = self.balances.get(node_id, 0.0)
            if not math.isfinite(balance) or balance < 0:
                raise ValueError("Settlement balances must be finite and nonnegative.")
        consumer_bal = self.balances.get(consumer_node_id, 0.0)
        if consumer_bal < total_fee:
            raise ValueError(
                f"Node {consumer_node_id} has insufficient $HIVE ({consumer_bal} < {total_fee})"
            )

        old_balances = self.balances.copy()
        old_history_length = len(self.settlement_history)
        receipt = InferenceSettlementReceipt(
            consumer_node_id=consumer_node_id,
            expert_provider_nodes=providers,
            domain_expert=domain_expert,
            verified_tokens=verified_tokens,
            rtt_speedup=rtt_speedup,
            fee_paid_hive=total_fee,
        )
        try:
            self.balances[consumer_node_id] = round(consumer_bal - total_fee, 6)
            # Distribute integer micro-HIVE, preserving total value despite rounding.
            share_micro, remainder = divmod(round(total_fee * 1_000_000), len(providers))
            for index, pid in enumerate(providers):
                share = (share_micro + (index < remainder)) / 1_000_000
                self.balances[pid] = round(self.balances.get(pid, 0.0) + share, 6)
            self.settlement_history.append(receipt)
            self._persist()
        except Exception:
            self.balances.clear()
            self.balances.update(old_balances)
            del self.settlement_history[old_history_length:]
            raise
        return receipt

    def credit_reward(
        self,
        node_id: str,
        amount_hive: float,
        reason: str = "",
    ) -> float:
        """Credit a peer reuse royalty or community bundle reward to `node_id`."""
        del reason
        self.balances[node_id] = round(self.balances.get(node_id, 0.0) + float(amount_hive), 6)
        self.reputation_scores[node_id] = self.reputation_scores.get(node_id, 100) + 5
        self._persist()
        return self.balances[node_id]

    def get_balance(self, node_id: str) -> float:
        """Return the current `$HIVE` balance for `node_id`."""
        return round(self.balances.get(node_id, 0.0), 6)

    def verify_chain_integrity(self) -> bool:
        """Cryptographically verify every block hash and prev_block_hash link from genesis."""
        if not self.chain or self.chain[0] != self._make_genesis_block():
            return False
        for i, block in enumerate(self.chain):
            if block.block_height != i or block.compute_hash() != block.block_hash:
                return False
            if i > 0 and block.prev_block_hash != self.chain[i - 1].block_hash:
                return False
        return True

    def _persist(self) -> None:
        if self.ledger_path is not None:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "schema_version": 1,
                "chain": [asdict(b) for b in self.chain],
                "balances": self.balances,
                "reputation_scores": self.reputation_scores,
                "slashed_registry": self.slashed_registry,
                "settlement_history": [asdict(s) for s in self.settlement_history],
            }
            data["snapshot_sha256"] = self._snapshot_digest(data)
            # Commit atomically so a failed write cannot truncate the previous ledger.
            fd, temporary_path = tempfile.mkstemp(prefix=f".{self.ledger_path.name}.", dir=self.ledger_path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(data, handle, indent=2, allow_nan=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, self.ledger_path)
            finally:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
