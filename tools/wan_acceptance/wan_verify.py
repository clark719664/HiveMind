#!/usr/bin/env python
"""Reconcile sender and receiver artifacts after one stopped acceptance run.

A sender count ACK alone is never acceptance. This checks every expected frame's
full SHA-256 against independently verified receiver evidence, regardless of
arrival order. Use private, trusted artifacts from the two authorized operators.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wan_frames import MAX_TOTAL_FRAMES, validate_plan


def read_records(path: Path) -> list[dict]:
    records = []
    total_bytes = 0
    with path.open(encoding="utf-8") as source:
        while line := source.readline(1024 * 1024 + 1):
            total_bytes += len(line)
            if total_bytes > 128 * 1024 * 1024:
                raise ValueError("evidence exceeds 128 MiB")
            if len(records) >= MAX_TOTAL_FRAMES or len(line) > 1024 * 1024:
                raise ValueError("evidence exceeds bounded run limits")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("evidence records must be objects")
            records.append(record)
    return records


def reconcile(sender: Path, receiver: Path) -> dict:
    sent = read_records(sender)
    received = read_records(receiver / "frames.jsonl")
    with (receiver / "summary.json").open(encoding="utf-8") as source:
        raw_summary = source.read(1024 * 1024 + 1)
    if len(raw_summary) > 1024 * 1024:
        raise ValueError("summary exceeds 1 MiB")
    summary = json.loads(raw_summary)
    if not isinstance(summary, dict):
        raise ValueError("summary must be an object")
    if not sent:
        raise ValueError("no sender batches")
    seed, batches, size = (sent[0][key] for key in ("seed", "batches", "frame_bytes"))
    count = sent[0]["frames_sent"]
    validate_plan(seed, batches, count, size)
    expected = {}
    for record in sent:
        batch = record["batch"]
        if (not isinstance(batch, int) or not 0 <= batch < batches
                or record.get("seed") != seed or record.get("batches") != batches
                or record.get("frame_bytes") != size or record.get("frames_sent") != count
                or record.get("bytes") != count * size or record.get("ok") is not True
                or record.get("frames_delivered") != count or len(record["frame_sha256"]) != count):
            raise ValueError("sender plan mismatch or unacknowledged batch")
        for index, digest in enumerate(record["frame_sha256"]):
            if (batch, index) in expected:
                raise ValueError("duplicate sender batch")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("sender digest must be a full SHA-256")
            expected[batch, index] = digest
    if len(expected) != batches * count:
        raise ValueError("missing sender batches")
    actual = {}
    for record in received:
        key = record.get("batch"), record.get("index")
        if (record.get("valid") is not True or record.get("intact") is not True
                or record.get("duplicate") is not False or record.get("expected") is not True
                or record.get("bytes") != size or key in actual):
            raise ValueError("invalid or duplicate receiver frame")
        actual[key] = record.get("sha256")
    if actual != expected:
        raise ValueError("receiver identities/digests do not match all sender frames")
    for key, value in {"seed": seed, "batches": batches, "frames_per_batch": count,
                       "frame_bytes": size, "expected_frames": len(expected),
                       "received_frames": len(expected), "unique_valid_frames": len(expected),
                       "missing_frames": 0, "invalid_frames": 0, "duplicate_frames": 0}.items():
        if summary.get(key) != value:
            raise ValueError(f"receiver summary mismatch: {key}")
    if summary.get("accepted") is not True or summary.get("counts", {}).get("handler_errors", 0):
        raise ValueError("receiver did not accept the run")
    return {"accepted": True, "frames_verified": len(expected), "seed": seed,
            "note": "Artifact integrity reconciled; topology and WAN claims require separate operator evidence."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sender", type=Path, required=True)
    parser.add_argument("--receiver", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = reconcile(args.sender, args.receiver)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"accepted": False, "error": str(exc)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
