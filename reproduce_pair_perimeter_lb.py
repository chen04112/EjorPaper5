#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Portable reproduction of the valid pair-perimeter lower bounds behind Table 4
("Valid SOCP lower bounds and certified optimality gaps") of the manuscript.

This recomputes the certified lower bound for each reference instance directly from
the committed solver code (src/lb_relax_socp.solve_lb_relax_socp with pair-cuts enabled)
and checks it against the values reported in the paper. The lower bound is the
seconds-level certified component of the reported optimality-gap intervals; the paper's
upper bounds (incumbents) come from the multi-seed search and are not recomputed here.

Requirements: Python 3.9+, numpy, scipy, and a working MOSEK installation with a license.
Run from anywhere:  python reproduce_pair_perimeter_lb.py

Paths are resolved relative to this file, so no machine-specific paths are needed.
"""
import os
import sys
import json
import csv

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
INST = os.path.join(ROOT, "instances")

import hybrid_vmurp_lbbd_v3 as vm   # noqa: E402
import lb_relax_socp as lb          # noqa: E402

# (instance file, R override, label, expected pair-perimeter LB from the paper)
CASES = [
    ("Example3_15_1_bundle_separate_depot.json", None, "n14", 16392.24),
    ("Example3_30_1_bundle_separate_depot.json", None, "n30", 25047.76),
    ("Example3_40_1_bundle_separate_depot.json", 5.0,  "n40", 28224.07),
    ("Example3_60_1_bundle_separate_depot.json", None, "n60", None),  # scalability check (Supplementary Appendix S.6)
]


def main():
    dro = vm.DROConfig(enabled=True)
    rows = []
    print("Reproducing pair-perimeter lower bounds (reliability-aware effective speeds):\n")
    for fname, rov, label, expected in CASES:
        path = os.path.join(INST, fname)
        d = json.load(open(path, encoding="utf-8"))
        idict = d.get("instances", [d])[0]
        if rov is not None:
            idict = dict(idict)
            idict["params"] = dict(idict["params"])
            idict["params"]["R"] = float(rov)
        inst = vm.Instance.from_dict(idict)
        n = len(inst._task_ids)
        vd_out, vd_in, _, _ = dro.speeds(inst.vd)
        seq = list(inst._task_ids)
        lbp, tsec, _ = lb.solve_lb_relax_socp(
            inst, vd_out, vd_in, seq, prefix_k=0, mosek_threads=4,
            add_prefix_chain=False, add_pair_cuts=True,
        )
        if expected is None:
            tag = "(scalability check)"
        elif abs(lbp - expected) < 0.5:
            tag = "MATCH"
        else:
            tag = "DIFF (paper %.2f)" % expected
        print("  %-4s n=%-3d  pair_perimeter_LB = %10.2f   (%.2f s)   %s"
              % (label, n, lbp, tsec, tag))
        rows.append([label, n, round(lbp, 2), round(tsec, 3),
                     expected if expected is not None else ""])

    out = os.path.join(ROOT, "reproduced_pair_perimeter_lb.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "n", "pair_perimeter_LB", "time_s", "expected_LB_paper"])
        w.writerows(rows)
    print("\nWrote %s" % out)


if __name__ == "__main__":
    main()
