# Reproducibility Guide

This package makes the computational study of "Routing and Scheduling for Vessel-Based
Drone Inspection: Interleaved Sorties and Endogenous Rendezvous" reproducible. It has
three parts: the benchmark **instances**, the seeded **generator** used to synthesize the
generated families, and the per-instance **records** plus **reproduction checks**.

The shipped instance files are the **authoritative benchmark** (bit-for-bit the instances
used in the paper). The generator is provided so the construction procedure is fully
specified and auditable. This artifact accompanies the submission and will be deposited in
a public, version-archived repository with a citable DOI upon acceptance.

## 1. Instances (`instances/`)

| File | Instance in paper | n | L | R | Used in |
|---|---|---|---|---|---|
| `Example3_15_1_bundle_separate_depot.json` | Ex6-15-1 (and its n=2..10 prefixes) | 14 | 3 | 5.0 | Table 4 (n=14); certified calibration prefixes n<=6 (Supplementary Appendix S.9); n=7/8/10 monolithic-dominance probe (Sec 6.2) |
| `Example3_30_1_bundle_separate_depot.json` | Ex6-30-1 | 30 | 3 | 5.0 | Table 4 (n=30) |
| `Example3_40_1_bundle_separate_depot.json` | Ex3-40-1 | 40 | 3 | 5.0 | Table 4 (n=40) |
| `Example3_60_1_bundle_separate_depot.json` | Ex3-60-1 | 60 | 3 | 5.0 | Pair-perimeter lower-bound timing check at n=60 (Supplementary Appendix S.6) |
| `Example3_5_1_bundle_separate_depot.json` | Ex3-5-1 | 4 | 3 | 1.0 (nominal) | Table 5 (template-family validation) |
| `Example3_6_1_bundle_separate_depot.json` | Ex3-6-1 | 5 | 3 | 3.0 (nominal) | Table 5 (template-family validation) |
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

The deterministic, seeded generator that synthesizes the generated instance families
(irregular-field / line-depot / co-located motifs). It is fully specified by:

- A global `SEED` (default `2025`) and a per-instance seed convention
  `inst_seed = SEED + 10000*n + 97*r` for size `n` and repetition index `r`, so every
  instance is reconstructible from its `(n, r)` label.
- The conservative-feasibility guarantee `2*dist(port, task) <= vd*R` (a plan is feasible
  even if the vessel does not move), so the inner SOCP is not vacuously infeasible.
- Controllable size list `SIZES`, fleet choices `L_CHOICES`, and layout parameters, all set
  at the top of the script.

To regenerate: `python bundle_instances.py` (requires `numpy`). The shipped instance files
remain the authoritative benchmark; the generator documents how such families are built.

## 3. Records and reproduction checks

- `computational_results_all_records.csv` -- consolidated run-level records (objectives,
  runtimes, execution-simulation summaries).
- `aggregate_reproduction_checks.csv` -- grouped checks behind each manuscript table/figure.
- `small_template_validation.csv` -- values behind Table 5.
- `reproduction_checks/pair_perimeter_lb_reproduction.csv` -- an **independent recomputation
  of the pair-perimeter lower bound** used in Table 4. The lower-bound values reproduce to
  rounding: n=14 -> 16392.24, n=30 -> 25047.76, n=40 -> 28224.07 (matching the bounds implied
  by Table 4), confirming the certified bound is exactly reproducible from the committed
  code (`lb_relax_socp.solve_lb_relax_socp`, pair-cuts enabled). The n=60 bound (30838.61)
  is computed in about 5 seconds, confirming the Supplementary Appendix S.6 scalability statement.
  The strong incumbents (upper bounds) in Table 4 are the best found across search
  configurations and require the full multi-seed search; the lower bounds here are the
  seconds-level certified component.
- `data_dictionary.csv`, `source_file_index.csv`, `manifest.json` -- field definitions,
  file-to-experiment mapping, and file-level SHA-256 manifests.
