"""Deterministic, self-verifying test frames shared by the WAN acceptance sender and receiver.

The body is a function of (seed, batch, index) only, so the receiver can prove a frame arrived
intact without sharing any secret. This is an integrity check for the test, NOT authentication:
peer authentication in the link comes from the TLS certificate chain pinned via ca.pem.
"""
from __future__ import annotations

import hashlib
import struct

MAGIC = b"HMWAN1"
HEADER = struct.Struct("!6sIIQ")
MIN_FRAME_BYTES = 64


def make_frame(seed: int, batch: int, index: int, size: int) -> bytes:
    if size < MIN_FRAME_BYTES:
        raise ValueError(f"frame size must be at least {MIN_FRAME_BYTES}")
    body = bytearray()
    counter = 0
    while len(body) < size - HEADER.size:
        body += hashlib.sha256(f"{seed}:{batch}:{index}:{counter}".encode()).digest()
        counter += 1
    return HEADER.pack(MAGIC, batch, index, seed) + bytes(body[: size - HEADER.size])


def verify_frame(frame: bytes, expected_seed: int) -> tuple[bool, int, int]:
    """Return (intact, batch, index). Unparseable frames are (False, -1, -1)."""
    if len(frame) < HEADER.size:
        return False, -1, -1
    magic, batch, index, seed = HEADER.unpack(frame[: HEADER.size])
    if magic != MAGIC or seed != expected_seed:
        return False, -1, -1
    return frame == make_frame(seed, batch, index, len(frame)), batch, index
