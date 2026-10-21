"""Modified BOINC Orchestration, Signed Parquet Shards, Spot-Check Validation & Local GPU Astrophysics Campaign."""

from hivemind.boinc.local_astrophysics_boinc import (
    create_signed_astrophysics_parquet_shard,
    run_local_boinc_astrophysics_campaign,
)

__all__ = [
    "create_signed_astrophysics_parquet_shard",
    "run_local_boinc_astrophysics_campaign",
]

