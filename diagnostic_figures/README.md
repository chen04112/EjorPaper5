# Diagnostic Figures

This folder contains convenience copies of the large-instance spatial and
timeline diagnostic figures referenced from Supplementary Appendix S.7. The file
names record the instance size, fleet size, and seed where applicable.

The reproducible source records are in `../large_diagnostics/`: fixed seeds,
generated instance JSON files, realized plan JSON files, summary CSV, generated
figures, and a SHA-256 manifest. Regenerate them with:

```bash
python reproduce_large_diagnostics.py --max-iter 12 --repair-k 5
```

These figures are illustrative diagnostics, not aggregate scalability tables;
the aggregate computational records are in `../results/`.
