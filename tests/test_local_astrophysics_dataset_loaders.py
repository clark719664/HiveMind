"""Cached Hugging Face dataset parsing in the local astrophysics BOINC campaign, on tiny fixtures."""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import hivemind.boinc as boinc_pkg
from hivemind.boinc import local_astrophysics_boinc as lab


def _cache_file(root: Path, repo_id: str, filename: str) -> Path:
    path = root / "huggingface" / "hub" / ("datasets--" + repo_id.replace("/", "--")) / "snapshots" / "rev0" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture()
def cache_root(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(lab, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(lab.Path, "home", staticmethod(lambda: tmp_path / "no-home"))
    return tmp_path


def test_cached_file_lookup_finds_snapshot_files_and_misses_cleanly(cache_root: Path) -> None:
    target = _cache_file(cache_root, "org/data", "train.parquet")
    target.write_bytes(b"x")
    assert lab._find_cached_hf_dataset_file("org/data", "nested/train.parquet") == target
    assert lab._find_cached_hf_dataset_file("org/data", "missing.parquet") is None
    assert lab._find_cached_hf_dataset_file("org/absent", "train.parquet") is None


def test_kepler_rows_are_classified_by_radius_and_incomplete_rows_skipped(cache_root: Path) -> None:
    repo = "astro/exoplanets-unified"
    pq.write_table(
        pa.table(
            {
                "period_days": [3.5, 10.0, None, 40.0, 7.0],
                "planet_radius_rearth": [1.0, 2.0, 1.1, 6.0, 1.0],
                "transit_depth_ppm": [100.0, None, 1.0, 900.0, 1.0],
                "model_snr": [20.0, 12.0, 1.0, None, 1.0],
                "stellar_teff_k": [5700.0, 5100.0, 5000.0, 6200.0, 5000.0],
                "target": ["confirmed", None, "x", "candidate", "unused"],
            }
        ),
        _cache_file(cache_root, repo, "train-00000-of-00001.parquet"),
    )
    samples = lab._load_astrophysics_examples("Kepler_Exoplanet_Transit_Classifier", repo, num_samples=3)
    assert len(samples) == 3
    assert "Rocky Terrestrial / Earth-Size World" in samples[0] and "Disposition: CONFIRMED" in samples[0]
    assert "Super-Earth Exoplanet" in samples[1] and "Transit Depth: 250.0 ppm" in samples[1]
    assert "Sub-Neptune / Gas Giant" in samples[2] and "Disposition: CANDIDATE" in samples[2]


def test_nasa_archive_rows_fill_defaults_and_skip_empty_rows(cache_root: Path) -> None:
    repo = "juliensimon/nasa-exoplanets"
    pq.write_table(
        pa.table(
            {
                "pl_name": ["Kep-1 b", "Gone", None, "Unused"],
                "hostname": ["Kep-1", "Star-3", "Star-2", "Star-4"],
                "pl_orbper": [5.0, None, 20.0, 1.0],
                "pl_rade": [2.0, None, None, 1.0],
                "pl_bmasse": [None, 1.0, 300.0, 1.0],
                "st_teff": [5600.0, 1.0, None, 1.0],
            }
        ),
        _cache_file(cache_root, repo, "exoplanets.parquet"),
    )
    samples = lab._load_astrophysics_examples("Kepler_Exoplanet_Transit_Classifier", repo, num_samples=2)
    assert "Kep-1 b around Kep-1" in samples[0] and "Super-Earth / Sub-Neptune" in samples[0]
    assert "Mass: 15.00 M_earth" in samples[0]
    assert "EXO-2 around Star-2" in samples[1] and "Radius: 2.10 R_earth" in samples[1]
    assert "Host Star Teff: 5700 K" in samples[1]


def test_sdss_rows_get_spectral_types_by_temperature(cache_root: Path) -> None:
    repo = "sdss/stars_regression"
    pq.write_table(
        pa.table(
            {
                "spec_id": [11, 12, 13, 14],
                "plate": [1001, 1002, None, 1004],
                "fiber": [1, 2, 3, 4],
                "ra": [0.0, 1.0, 2.0, 3.0],
                "dec": [0.0, 1.0, 2.0, 3.0],
                "snr": [30.0, 30.0, 30.0, 30.0],
                "rv_adop": [0.0, 0.0, 0.0, 0.0],
                "catalog_teff": [6500.0, 5500.0, 4500.0, 4000.0],
                "catalog_feh": [0.1, -0.2, None, 0.0],
                "catalog_logg": [4.0, 4.4, None, 4.0],
            }
        ),
        _cache_file(cache_root, repo, "train.parquet"),
    )
    samples = lab._load_astrophysics_examples("SDSS_Stellar_Spectra", repo, num_samples=3)
    assert ["F-Type Star" in samples[0], "G-Type Solar Star" in samples[1], "K-Type Star" in samples[2]] == [True] * 3
    assert "Plate 1000" in samples[2] and "[Fe/H]: -0.30 dex" in samples[2]


def test_spectroscopy_csv_rows_are_parsed_with_defaults(cache_root: Path) -> None:
    repo = "shoriful025/spectroscopy"
    path = _cache_file(cache_root, repo, "main.csv")
    fields = ["exoplanet_id", "host_star_type", "transit_depth_ppm", "h2o_abundance_log",
              "ch4_abundance_log", "co2_abundance_log", "equilibrium_temp_k", "habitability_score"]
    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(zip(fields, ["EXO-7", "M-Dwarf", "800", "-3.1", "-5.5", "-4.0", "300", "0.9"])))
        writer.writerow(dict(zip(fields, ["EXO-8", "K-Type", "", "", "", "", "", ""])))
        writer.writerow(dict(zip(fields, ["EXO-9", "G-Type", "1", "1", "1", "1", "1", "1"])))
    samples = lab._load_astrophysics_examples("JWST_Spectroscopy", repo, num_samples=2)
    assert "EXO-7 around M-Dwarf" in samples[0] and "log(H2O): -3.10" in samples[0]
    assert "EXO-8 around K-Type" in samples[1] and "Transit Depth: 1200 ppm" in samples[1]
    assert len(samples) == 2


@pytest.mark.parametrize(
    "repo, filename",
    [
        ("astro/exoplanets-unified", "train-00000-of-00001.parquet"),
        ("juliensimon/nasa-exoplanets", "exoplanets.parquet"),
        ("sdss/stars_regression", "train.parquet"),
        ("shoriful025/spectroscopy", "main.csv"),
    ],
)
def test_unreadable_cache_files_fall_back_to_the_curated_corpus(cache_root: Path, repo: str, filename: str) -> None:
    path = _cache_file(cache_root, repo, filename)
    path.write_bytes(b"\xff\xfe not a parquet or utf-8 csv \xff")
    samples = lab._load_astrophysics_examples("Kepler_Exoplanet_Transit_Classifier", repo, num_samples=3)
    assert len(samples) == 3 and all("[ObsSeq #" in s for s in samples)


def test_boinc_package_resolves_lazy_exports_in_process() -> None:
    assert "run_local_boinc_astrophysics_campaign" in boinc_pkg.__all__
    func = boinc_pkg.create_signed_astrophysics_parquet_shard
    assert func is lab.create_signed_astrophysics_parquet_shard
    assert boinc_pkg.__dict__["create_signed_astrophysics_parquet_shard"] is func
    with pytest.raises(AttributeError, match="no attribute 'not_exported'"):
        boinc_pkg.not_exported


def test_campaign_calendar_cli_writes_its_report_and_calendar(tmp_path: Path, capsys) -> None:
    import json

    from hivemind.cli import campaign_calendar_quorum as cal

    ics, report = tmp_path / "cal.ics", tmp_path / "quorum.json"
    assert cal.main(["--ics-output", str(ics), "--output", str(report)]) == 0
    assert ics.read_text(encoding="utf-8").startswith("BEGIN:VCALENDAR")
    assert json.loads(report.read_text(encoding="utf-8")) == json.loads(capsys.readouterr().out)
