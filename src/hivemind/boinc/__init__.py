"""Modified BOINC Orchestration, Signed Parquet Shards, Spot-Check Validation & Local GPU Astrophysics Campaign."""

from hivemind.boinc.astrophysics_24h_campaign import (
    render_astrophysics_24h_dashboard_png,
    render_cosmology_deep_field_discovery_png,
    run_24h_astrophysics_gpu_campaign,
)
from hivemind.boinc.local_astrophysics_boinc import (
    create_signed_astrophysics_parquet_shard,
    run_local_boinc_astrophysics_campaign,
)

__all__ = [
    "create_signed_astrophysics_parquet_shard",
    "render_astrophysics_24h_dashboard_png",
    "render_cosmology_deep_field_discovery_png",
    "run_24h_astrophysics_gpu_campaign",
    "run_local_boinc_astrophysics_campaign",
]


