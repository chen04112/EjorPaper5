# Reproducibility Guide

This package makes the computational study of "Routing and Scheduling for Vessel-Based
Drone Inspection: Interleaved Sorties and Endogenous Rendezvous" reproducible. It has
three parts: the benchmark **instances**, the seeded **generator** used to synthesize the
generated families, and the per-instance **records** plus **reproduction checks**.

The shipped instance files are the **authoritative benchmark** (bit-for-bit the instances
used in the paper). The generator is provided so the construction procedure is fully
specified and auditable. This supplementary artifact accompanies the submission and is
also available at `https://github.com/chen04112/EjorPaper5`; a version-archived
repository release with a citable DOI will be added upon acceptance.

## 1. Instances (`instances/`)

| File | Instance in paper | n | L | R | Used in |
|---|---|---|---|---|---|
| `Example3_15_1_bundle_separate_depot.json` | Ex6-15-1 (and its n=2..10 prefixes) | 14 | 3 | 5.0 | Table 5 lower-bound interval (n=14); certified calibration prefixes n<=6 (Supplementary Appendix S.9); n=7/8/10 monolithic-diagnostic probe (Section 6.2) |
| `Example3_30_1_bundle_separate_depot.json` | Ex6-30-1 | 30 | 3 | 5.0 | Table 5 lower-bound interval (n=30) |
| `Example3_40_1_bundle_separate_depot.json` | Ex3-40-1 | 40 | 3 | 5.0 | Table 5 lower-bound interval (n=40) |
| `Example3_60_1_bundle_separate_depot.json` | Ex3-60-1 | 60 | 3 | 5.0 | Pair-perimeter lower-bound timing check at n=60 (Supplementary Appendix S.6) |
| `Example3_5_1_bundle_separate_depot.json` | Ex3-5-1 | 4 | 3 | 1.0 (nominal) | Table 6 (template-family validation) |
| `Example3_6_1_bundle_separate_depot.json` | Ex3-6-1 | 5 | 3 | 3.0 (nominal) | Table 6 (template-family validation) |
| `hornsrev1_public_n24_L3_scaled.json` | Horns Rev 1 subset | 24 | 3 | -- | Table 3 (public-layout probe) |
| `WPPcoords_HornsRev1.csv` | Horns Rev 1 raw public coordinates | -- | -- | -- | Source for the scaled subset above |

Notes:
- The certified-calibration prefixes (Supplementary Appendix S.9, n=2..6) and the n=7/8/10
  monolithic-dominance probe use **prefixes** of `Example3_15_1` (the first k tasks),
  all under the reliability-aware effective speeds with `R=5`. The prefix construction is
  the first `k` task coordinates of the file; see `prefix_instance(k)` in the exact-runner
  script referenced in the records.
- Each JSON stores `tasks` (task coordinates), `port_start`/`port_end` (vessel depot), and
  `params` (`L`, `vv`, `cv`, `vd`, `cd`, `R`, `seed`), directly loadable by the solver's
  `Instance.from_dict`.
- The Horns Rev 1 subset is an affine scaling of the first 24 turbine rows after the
  substation from the public dataset of Abritta et al. (2024) to the same 100-unit span
  used by the generated layouts.

## 2. Generator (`generator/bundle_instances.py`)

The deterministic, seeded generator synthesizes generated instances along two distinct
geometric dimensions: task field geometry (uniform, clustered, or approximately linear
coordinates) and depot configuration (co-located or separate starting and terminal points).
It is fully specified by:

- A global `SEED` (default `2025`) and a per-instance seed convention
  `inst_seed = SEED + 10000*n + 97*r` for size `n` and repetition index `r`, so every
  instance is reconstructible from its `(n, r)` label.
- The conservative-feasibility guarantee `2*dist(port, task) <= vd*R` (a plan is feasible
  even if the vessel does not move), so the inner SOCP is not vacuously infeasible.
- Controllable size list `SIZES`, fleet mapping `L_BY_SIZE`, fallback fleet choices
  `L_CHOICES`, and layout parameters, all set at the top of the script. The default
  size/fleet pairs are `(20,2)`, `(40,3)`, `(60,4)`, and `(80,5)`, matching the
  main scalability configurations in the manuscript.

To regenerate: `python bundle_instances.py` (requires `numpy`). The shipped instance files
remain the authoritative benchmark; the generator documents how such families are built.

## 3. Records and reproduction checks

- `computational_results_all_records.csv` -- consolidated run-level records (objectives,
  runtimes, execution-simulation summaries).
- `aggregate_reproduction_checks.csv` -- grouped checks behind each manuscript table/figure.
- `small_template_validation.csv` -- values behind Table 6.
- `reproduction_checks/table5_lower_bound_intervals.csv` -- current Table 5
  lower-bound intervals, including the incumbent values used in the manuscript
  and the lower bounds used to compute the reported intervals.
- `reproduction_checks/pair_perimeter_lb_reproduction.csv` -- an **independent recomputation
  of the pair-perimeter lower bound** used in Table 5. The lower-bound values reproduce to
  rounding: n=14 -> 16392.24, n=30 -> 25047.76, n=40 -> 28224.07 (matching the bounds implied
  by Table 5), confirming the certified bound is exactly reproducible from the committed
  code (`lb_relax_socp.solve_lb_relax_socp`, pair-cuts enabled). The n=60 bound (30838.61)
  is computed in about 5 seconds, confirming the Supplementary Appendix S.6 scalability statement.
  The lower-bound recomputation script is a portable certificate check; it is not a rerun of
  the full multi-seed incumbent search.
- `data_dictionary.csv`, `source_file_index.csv`, `manifest.json` -- field definitions,
  file-to-experiment mapping, and file-level SHA-256 manifests.

### Reliability sweep

The four correlation sweeps behind Figure 4 are stored in
`results/raw_records/ExpD_Reliability_*_corr{00,03,07,09}.*`. Each sweep uses
five generated 40-task instances, three planning seeds per instance, eight
parallel workers, 80 PLNS-SA iterations per worker, and 2,000 beta-distributed
execution scenarios. The endurance test uses the numerical tolerance
`mc_R_tol=1e-4`. The manifest beside each raw and summary CSV records these
settings. After replacing a sweep, run `python refresh_reliability_artifact.py`
to rebuild the consolidated records, aggregate checks, source index, and hashes.

## 4. Large diagnostic figures in Supplementary Appendix S.7

The large-instance visual diagnostics are reproducible from fixed seeds. They
are representative stress runs and are not used as aggregate scalability tables.
The submitted artifact includes:

- `reproduce_large_diagnostics.py` -- regeneration script;
- `large_diagnostics/instances/` -- generated n=200, n=300, and n=400 instance
  JSON files;
- `large_diagnostics/plans/` -- realized event orders, event locations, event
  times, objectives, effective speeds, and solver settings;
- `large_diagnostics/results/large_diagnostic_summary.csv` -- one row per case;
- `large_diagnostics/figures/` -- individual figures and triptychs;
- `large_diagnostics/manifest.json` -- SHA-256 hashes for the generated files.

The fixed cases are:

| n | L | instance/run seed |
|---:|---:|---:|
| 200 | 3 | 22260128 |
| 300 | 4 | 23260128 |
| 400 | 5 | 24260128 |

To regenerate the records and figures from the repository root:

```bash
python reproduce_large_diagnostics.py
```

A MOSEK installation with a valid license is required.
