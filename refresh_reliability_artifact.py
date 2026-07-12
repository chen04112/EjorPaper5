#!/usr/bin/env python3
"""Refresh derived reliability records after replacing the four raw sweeps."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RAW = RESULTS / "raw_records"
CODES = ("00", "03", "07", "09")
SOURCE_ROOT = (
    "DROBaseline/artifacts/results/"
    "WashRuns_Beta_sigma0.15_etaU1.5_R4.5_scale150_tol1e-4"
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_values(rows: list[dict[str, str]], field: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row.get(field, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def mean(rows: list[dict[str, str]], field: str) -> str:
    values = finite_values(rows, field)
    return str(statistics.fmean(values)) if values else ""


def ci95(rows: list[dict[str, str]], field: str) -> str:
    values = finite_values(rows, field)
    if len(values) < 2:
        return "0.0" if values else ""
    return str(1.96 * statistics.stdev(values) / math.sqrt(len(values)))


def refresh_consolidated() -> None:
    path = RESULTS / "computational_results_all_records.csv"
    fields, records = read_csv(path)
    for code in CODES:
        raw_path = RAW / f"ExpD_Reliability_raw_corr{code}.csv"
        raw_fields, raw_rows = read_csv(raw_path)
        dataset = f"ExpD_reliability_corr{code}"
        indexes = [i for i, row in enumerate(records) if row["source_dataset"] == dataset]
        if len(indexes) != len(raw_rows):
            raise RuntimeError(f"{dataset}: {len(indexes)} consolidated rows != {len(raw_rows)} raw rows")
        source = (
            f"{SOURCE_ROOT}/beta_corr{code}_s2000/"
            f"ExpD_Evidence_raw_beta_corr{code}_s2000.csv"
        )
        for index, raw_row in zip(indexes, raw_rows):
            target = records[index]
            for field in raw_fields:
                if field in target:
                    target[field] = raw_row.get(field, "")
            target["source_original_relative_path"] = source
    write_csv(path, fields, records)


def refresh_aggregates() -> None:
    path = RESULTS / "aggregate_reproduction_checks.csv"
    fields, aggregate_rows = read_csv(path)
    for code in CODES:
        _, raw_rows = read_csv(RAW / f"ExpD_Reliability_raw_corr{code}.csv")
        for config in ("mc_dro", "mc_nominal"):
            selected = [row for row in raw_rows if row.get("config") == config and row.get("status") == "OK"]
            target = next(
                row
                for row in aggregate_rows
                if row.get("source_dataset") == "ExpD_reliability_sweep"
                and row.get("corr_label") == f"corr{code}"
                and row.get("config") == config
            )
            target.update(
                {
                    "count": str(float(len(selected))),
                    "mean_final_obj": mean(selected, "final_obj"),
                    "ci95_final_obj": ci95(selected, "final_obj"),
                    "mean_time_total_s": mean(selected, "time_total"),
                    "ci95_time_total_s": ci95(selected, "time_total"),
                    "mean_ship_time": mean(selected, "ship_time"),
                    "mean_drone_time": mean(selected, "drone_time"),
                    "best_final_obj": str(min(finite_values(selected, "final_obj"))),
                    "mean_mc_infeasible_rate": mean(selected, "mc_infeasible_rate"),
                    "mean_mc_obj_mean": mean(selected, "mc_obj_mean"),
                }
            )
    write_csv(path, fields, aggregate_rows)


def refresh_source_index() -> None:
    path = RESULTS / "source_file_index.csv"
    fields, rows = read_csv(path)
    for code in CODES:
        tag = f"beta_corr{code}_s2000"
        sources = {
            f"raw_records/ExpD_Reliability_raw_corr{code}.csv": (
                f"{SOURCE_ROOT}/{tag}/ExpD_Evidence_raw_{tag}.csv"
            ),
            f"raw_records/ExpD_Reliability_summary_corr{code}.csv": (
                f"{SOURCE_ROOT}/{tag}/ExpD_Evidence_summary_{tag}.csv"
            ),
            f"raw_records/ExpD_Reliability_manifest_corr{code}.json": (
                f"{SOURCE_ROOT}/{tag}/manifest_{tag}.json"
            ),
        }
        for target_name, source_name in sources.items():
            row = next(item for item in rows if item["target_file"] == target_name)
            target = RESULTS / target_name
            row["original_relative_path"] = source_name
            row["sha256"] = sha256(target)
            if target.suffix.lower() == ".csv":
                csv_fields, csv_rows = read_csv(target)
                row["rows_excluding_header"] = str(len(csv_rows))
                row["column_count"] = str(len(csv_fields))
                row["columns"] = ";".join(csv_fields)
    write_csv(path, fields, rows)


def refresh_checksums() -> None:
    path = RESULTS / "ADDED_ARTIFACT_SHA256SUMS.txt"
    refreshed = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _, relative = line.split(maxsplit=1)
        if relative == "results/manifest.json":
            continue
        target = ROOT / relative
        if target.exists():
            refreshed.append(f"{sha256(target)}  {relative}")
    script_entry = "refresh_reliability_artifact.py"
    if not any(line.endswith(f"  {script_entry}") for line in refreshed):
        refreshed.append(f"{sha256(ROOT / script_entry)}  {script_entry}")
    path.write_text("\n".join(refreshed) + "\n", encoding="utf-8")


def refresh_manifest() -> None:
    path = RESULTS / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["generated_at_local"] = datetime.now().isoformat(timespec="seconds")
    manifest["primary_files"] = [
        item for item in manifest["primary_files"] if not item.endswith(".xlsx")
    ]
    if "../refresh_reliability_artifact.py" not in manifest["primary_files"]:
        manifest["primary_files"].append("../refresh_reliability_artifact.py")
    files = []
    for entry in manifest["files"]:
        relative = entry["file"]
        target = RESULTS / relative
        if target.exists():
            files.append({"file": relative, "bytes": target.stat().st_size, "sha256": sha256(target)})
    script_relative = "../refresh_reliability_artifact.py"
    if not any(entry["file"] == script_relative for entry in files):
        target = RESULTS / script_relative
        files.append({"file": script_relative, "bytes": target.stat().st_size, "sha256": sha256(target)})
    manifest["files"] = files
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    refresh_consolidated()
    refresh_aggregates()
    refresh_source_index()
    refresh_checksums()
    refresh_manifest()
    print("Refreshed reliability-derived records and manifests.")


if __name__ == "__main__":
    main()
