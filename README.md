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
- `diagnostic_figures/`: large-instance diagnostic figures referenced in the
  Supplementary Appendix. These figures are illustrative diagnostics, not an
  aggregate scalability table.

## Quick Check

Install the minimal Python requirements and run:

```bash
pip install -r requirements.txt
python reproduce_pair_perimeter_lb.py
```

The script recomputes the lower-bound values for the benchmark instances and
prints `MATCH` for the reported values. A MOSEK installation with a valid license
is required for the conic solves.

For the instance-to-table map and additional reproduction notes, see
`REPRODUCE.md`.

## Citation Status

This public repository is provided as the review artifact for the revised
manuscript. A version-archived release with a citable DOI will be added upon
acceptance.
