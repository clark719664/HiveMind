"""Bounded deterministic acceptance frames, not authentication or WAN proof.

The header binds the original length and the operator-selected run seed. Receivers
also check an independent test plan, so a shortened or differently sized frame
cannot become valid by changing its header. Use a fresh seed for each run.
"""
from __future__ import annotations

import hashlib
import math
import struct

MAGIC = b"HMWAN2"
HEADER = struct.Struct("!6sIIIQ")
MIN_FRAME_BYTES = 64
MAX_FRAME_BYTES = 1024 * 1024
MAX_TOTAL_FRAMES = 100_000


def validate_plan(seed: int, batches: int, frames: int, size: int) -> None:
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if not 1 <= batches <= 10_000 or not 1 <= frames <= 4096:
        raise ValueError("batches must be 1..10000 and frames must be 1..4096")
    if batches * frames > MAX_TOTAL_FRAMES:
        raise ValueError("at most 100000 frames per run")
    if not MIN_FRAME_BYTES <= size <= MAX_FRAME_BYTES:
        raise ValueError("frame size must be 64..1048576 bytes")
    if frames * size > 16 * 1024 * 1024 or batches * frames * size > 512 * 1024 * 1024:
        raise ValueError("at most 16 MiB per batch and 512 MiB per run")


def validate_seconds(value: float, name: str, maximum: float, *, allow_zero: bool = False) -> None:
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero) or value > maximum:
        raise ValueError(f"{name} must be finite, {'nonnegative' if allow_zero else 'positive'}, and <= {maximum}")


def make_frame(seed: int, batch: int, index: int, size: int) -> bytes:
    if not MIN_FRAME_BYTES <= size <= MAX_FRAME_BYTES:
        raise ValueError(f"frame size must be {MIN_FRAME_BYTES}..{MAX_FRAME_BYTES}")
    body = bytearray()
    counter = 0
    while len(body) < size - HEADER.size:
        body += hashlib.sha256(f"{seed}:{batch}:{index}:{counter}".encode()).digest()
        counter += 1
    return HEADER.pack(MAGIC, batch, index, size, seed) + bytes(body[: size - HEADER.size])


def verify_frame(frame: bytes, expected_seed: int, expected_size: int | None = None) -> tuple[bool, int, int]:
    """Check the complete declared frame; malformed/short frames never raise."""
    if not MIN_FRAME_BYTES <= len(frame) <= MAX_FRAME_BYTES:
        return False, -1, -1
    magic, batch, index, size, seed = HEADER.unpack(frame[: HEADER.size])
    if magic != MAGIC or seed != expected_seed or size != len(frame):
        return False, batch, index
    if expected_size is not None and size != expected_size:
        return False, batch, index
    return frame == make_frame(seed, batch, index, size), batch, index


class FrameTracker:
    """Exactly-once acceptance with bounded identity tracking; arrival order is informational."""

    def __init__(self, seed: int, batches: int, frames: int, size: int):
        validate_plan(seed, batches, frames, size)
        self.seed, self.batches, self.frames, self.size = seed, batches, frames, size
        self.seen: set[tuple[int, int]] = set()
        self.received = self.invalid = self.duplicates = self.out_of_order = 0
        self.last_ordinal = -1

    def observe(self, frame: bytes) -> dict:
        intact, batch, index = verify_frame(frame, self.seed, self.size)
        expected = 0 <= batch < self.batches and 0 <= index < self.frames
        valid = intact and expected
        duplicate = valid and (batch, index) in self.seen
        self.received += 1
        if not valid:
            self.invalid += 1
        elif duplicate:
            self.duplicates += 1
        else:
            ordinal = batch * self.frames + index
            self.out_of_order += ordinal < self.last_ordinal
            self.last_ordinal = max(self.last_ordinal, ordinal)
            self.seen.add((batch, index))
        return {"batch": batch, "index": index, "bytes": len(frame), "intact": intact,
                "expected": expected, "duplicate": duplicate, "valid": valid and not duplicate,
                "sha256": hashlib.sha256(frame).hexdigest()}

    def summary(self) -> dict:
        expected = self.batches * self.frames
        missing = expected - len(self.seen)
        return {"seed": self.seed, "batches": self.batches, "frames_per_batch": self.frames,
                "frame_bytes": self.size, "expected_frames": expected, "received_frames": self.received,
                "unique_valid_frames": len(self.seen), "missing_frames": missing,
                "invalid_frames": self.invalid, "duplicate_frames": self.duplicates,
                "out_of_order_frames": self.out_of_order,
                "accepted": missing == 0 and self.invalid == 0 and self.duplicates == 0}
