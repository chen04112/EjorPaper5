# Supplementary Computational Results

This folder is the reproducibility artifact referenced in the manuscript "Data and Result Availability" statement. It has three parts: the benchmark **instances**, the seeded **generator** used to synthesize the generated families, and the per-instance **records** plus **reproduction checks**. **Start with `REPRODUCE.md`**, which maps every instance file to the table/experiment that uses it and documents the generator seed convention.

This supplementary artifact accompanies the submission and is also available at `https://github.com/chen04112/EjorPaper5`. A version-archived repository release with a citable DOI will be added upon acceptance. The shipped instance files are the authoritative benchmark.

## Contents

- `REPRODUCE.md`: reproducibility guide (instance-to-table map, generator seeds, reproduction checks).
- `instances/`: the exact benchmark instances as coordinate/JSON files for every reported size up to `n=60`, plus the scaled Horns Rev 1 public-layout subset and its raw public coordinates.
- `generator/bundle_instances.py`: the deterministic, seeded instance generator (global `SEED=2025`; per-instance seed `SEED + 10000*n + 97*r`; default fleet mapping `(20,2)`, `(40,3)`, `(60,4)`, `(80,5)`).
- `reproduction_checks/pair_perimeter_lb_reproduction.csv`: independent recomputation of the Table 5 pair-perimeter lower bounds (reproduce to rounding).
- `reproduction_checks/table5_lower_bound_intervals.csv`: current Table 5 lower-bound intervals, including the incumbent values used in the manuscript and the lower bounds behind the reported intervals.
- `ADDED_ARTIFACT_SHA256SUMS.txt`: SHA-256 hashes for the instance/generator/reproduction files added in round 2.
- `computational_results_all_records.csv`: consolidated per-instance/per-seed records.
- `aggregate_reproduction_checks.csv`: grouped checks used to trace the manuscript's aggregate tables and figures.
- `data_dictionary.csv`: field definitions.
- `source_file_index.csv`: original source paths, submitted file names, row counts, and SHA-256 hashes.
- `../refresh_reliability_artifact.py`: deterministic refresh of the reliability-derived consolidated records and manifests.
- `small_template_validation.csv`: instance-level small-template validation values from the manuscript table.
- `raw_records/`: copied raw CSV, JSON, and run manifest files used to build the consolidated records.
- `manifest.json`: file-level manifest and hashes for this folder.

## Scope

The records support the aggregate computational comparisons reported in the manuscript:

- Mini public-layout probe on the Horns Rev 1 subset.
- Small-scale validation against exhaustive enumeration within the template-induced family.
- Scalability and interleaving evidence.
- Medium-scale ablation at `n=40`, `L=3`.
- Planned-cost premium relative to nominal planning.
- Out-of-sample reliability simulation sweep under wait-and-sync execution.
- Auxiliary fixed-buffer and staged nominal replanning probe.
- Appendix tiny monolithic validation note.

The generated layouts are stylized, inspection-inspired computational instances. They are included for traceability of algorithmic comparisons and should not be interpreted as field-calibrated operator data. The representative large-instance spatial and timing figures in the manuscript are diagnostic visualizations rather than aggregate comparison tables; no separate seed-level aggregate table is claimed for those figures in this supplement. Their fixed seeds, generated instances, plan records, summary CSV, and regeneration script are provided in `../large_diagnostics/` and `../reproduce_large_diagnostics.py`.

## How To Trace Manuscript Items

Use `aggregate_reproduction_checks.csv` first. It lists the manuscript item, grouping variables, row count, mean objective, runtime, confidence intervals where applicable, simulation metrics, and notes. To inspect the underlying records, follow `source_dataset` into `computational_results_all_records.csv`, or use `source_file_index.csv` to open the exact copied raw file in `raw_records/`.

Key mappings:

- Public-layout probe: `PublicLayoutProbe_raw.csv`, `PublicLayoutProbe_summary.csv`, `PublicLayoutProbe_hornsrev1_public_n24_L3_scaled.json`, and `PublicLayout_HornsRev1_source_coordinates.csv`.
- Scalability evidence: `ExpA_Scalability_raw.csv` and `ExpA_Scalability_summary.csv`.
- Medium-scale ablation and planned premium: `ExpB_Ablation_raw.csv` and `ExpB_Ablation_summary.csv`.
- Reliability simulation sweep: `ExpD_Reliability_raw_corr00.csv`, `ExpD_Reliability_raw_corr03.csv`, `ExpD_Reliability_raw_corr07.csv`, `ExpD_Reliability_raw_corr09.csv`, with matching summary and manifest files. Their manifests record eight workers, 80 iterations per worker, 2,000 execution scenarios, and `mc_R_tol=1e-4`.
- Auxiliary fixed-buffer/replanning probe: `ExpD_AuxiliaryProbe_raw.csv`, `ExpD_AuxiliaryProbe_summary.csv`, and `ExpD_AuxiliaryProbe_manifest.json`.
- Tiny monolithic validation note: `TinyMonolithicCert_combined.csv` and the individual 900-second certificate files.
- Small-template validation table: `small_template_validation.csv`.

## Core Fields

The main run-level fields are:

- `instance_id`, `instance_index`, `layout_id`: instance identifiers.
- `config`, `policy_type`, `mode`: planning or simulation configuration.
- `run_seed`: seed for the planning run.
- `time_total`: planning runtime in seconds.
- `palns_obj`, `final_obj`: legacy column names for the objective before and after optional OPS-level local refinement. The `palns_*` prefix is retained for traceability to the archived raw files; in manuscript terminology it corresponds to the pre-refinement PLNS-SA/LNS incumbent, not to a claim of operator-weight adaptation.
- `ship_time`, `drone_time`: objective decomposition returned by the exact oracle.
- `mc_samples`, `mc_seed`, `mc_dist`, `mc_corr`: execution-simulation design fields.
- `mc_infeasible_rate`, `mc_obj_mean`, `mc_obj_p90`, `mc_obj_p95`, `mc_makespan_mean`, `mc_drone_time_mean`: wait-and-sync execution summaries.
- `replan_count`, `replan_time_total`, `realized_cost_inflation`: auxiliary execution-comparison fields.
- `cert_lb`, `cert_ub`, `cert_gap`, `cert_time`: certificate/monolithic validation fields.

See `data_dictionary.csv` for the complete list.
