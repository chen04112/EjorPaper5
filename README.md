# Reproducibility Package

This repository contains the code, benchmark instances, and computational records
for the manuscript:

> Routing and Scheduling for Vessel-Based Drone Inspection: Interleaved Sorties
> and Endogenous Rendezvous.

## Contents

- `src/`: solver and model code used for the computational study.
- `instances/`: benchmark instances used in the reported experiments.
- `results/`: raw and summary computational records, data dictionary, and file
  manifests.
- `generator/`: seeded instance generator used to document the construction of
  the generated instance families.
- `reproduce_pair_perimeter_lb.py`: portable check that recomputes the
  pair-perimeter lower bounds reported in the manuscript.
- `reproduce_large_diagnostics.py`: regenerates the Supplementary Appendix S.7
  large-instance diagnostic cases from fixed seeds, solves them with the
  submitted PLNS-SOCP code, and writes instance/plan/result records.
- `large_diagnostics/`: reproducible records for the n=200, n=300, and n=400
  diagnostic figures, including generated instances, realized plans, summary
  CSV, figures, and SHA-256 manifest.
- `diagnostic_figures/`: large-instance diagnostic figures referenced in the
  Supplementary Appendix. These are convenience copies of the figures generated
  by `reproduce_large_diagnostics.py`.

## Quick Check

Install the minimal Python requirements and run:

```bash
pip install -r requirements.txt
python reproduce_pair_perimeter_lb.py
python reproduce_large_diagnostics.py
```

The first script recomputes the lower-bound values for the benchmark instances
and prints `MATCH` for the reported values. The second script regenerates the
large diagnostic records and figures from fixed seeds using the submitted
diagnostic budget. A MOSEK installation with a valid license is required for the
conic solves. For a quick installation check, run
`python reproduce_large_diagnostics.py --smoke --outdir large_diagnostics_smoke`.

For the instance-to-table map and additional reproduction notes, see
`REPRODUCE.md`.

## Citation Status

This public repository is provided as the supplementary reproducibility artifact
for the revised manuscript. A version-archived repository release with a citable
DOI will be added upon acceptance.
