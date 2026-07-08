# Large Diagnostic Reproduction

This folder contains the reproducible record for the large-instance diagnostic
visuals in Supplementary Appendix S.7. The diagnostics are representative stress
runs, not aggregate scalability tables.

## Fixed Cases

| case | n | L | instance seed | run seed |
|---|---:|---:|---:|---:|
| ExampleLarge_n200_L3_seed22260128 | 200 | 3 | 22260128 | 22260128 |
| ExampleLarge_n300_L4_seed23260128 | 300 | 4 | 23260128 | 23260128 |
| ExampleLarge_n400_L5_seed24260128 | 400 | 5 | 24260128 | 24260128 |

## How to Regenerate

From the repository root, run:

```bash
python reproduce_large_diagnostics.py
```

A MOSEK installation with a valid license is required because the diagnostic
plans are evaluated by the submitted fixed-OPS SOCP oracle.

The default command uses the diagnostic budget submitted with the artifact:
`max_iter=100`, `repair_k=7`, `refine_policy=best`, `refine_steps=30`,
`refine_evals=1`, `cache_size=5000`, and `mosek_threads=1`. For a quick
installation check, use:

```bash
python reproduce_large_diagnostics.py --smoke --outdir large_diagnostics_smoke
```

## Contents

- `instances/`: generated instance JSON files for the three fixed cases.
- `plans/`: realized event orders, event locations, event times, objectives,
  effective speeds, and solver settings. Wall-clock times are printed by the
  script but not stored because they are environment dependent.
- `results/large_diagnostic_summary.csv`: one row per case.
- `figures/`: individual spatial/timeline figures and triptychs.
- `manifest.json`: file sizes and SHA-256 hashes for the generated files.
