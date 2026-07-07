#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
lb_relax_socp.py
Compute:
  - UB: from the existing solver pipeline (nn / PLNS-SA legacy mode / hybrid-lite)
  - LB: convex SOCP relaxation with optional prefix chain constraints

Outputs CSV lines:
  instance_id,json,idx,n,L,R,ub_method,ub_obj,ub_time,lb_k,lb_obj,lb_time,gap_pct,lb_status
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import hybrid_vmurp_lbbd_v3 as vm  # your file

try:
    from mosek.fusion import Model, Domain, Expr, ObjectiveSense, SolutionStatus
    HAS_MOSEK = True
except Exception:
    HAS_MOSEK = False


def _parse_k_list(k_list: str, n: int) -> List[int]:
    """
    Accept formats like:
      "0,2,5,10"
      "0,10%,20%,50%,100%"
    """
    out: List[int] = []
    for tok in (k_list or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.endswith("%"):
            p = float(tok[:-1])
            k = int(round(p * n / 100.0))
        else:
            k = int(tok)
        k = max(0, min(n, k))
        out.append(k)
    # unique, keep order
    seen = set()
    res = []
    for k in out:
        if k not in seen:
            seen.add(k)
            res.append(k)
    return res


def _maybe_write_header(csv_path: str) -> None:
    if not csv_path:
        return
    need = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    if not need:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "instance_id", "json", "instance_index", "n", "L", "R",
            "ub_method", "ub_obj", "ub_time",
            "lb_k", "lb_obj", "lb_time", "gap_pct", "lb_status",
        ])


def solve_lb_relax_socp(
    inst: vm.Instance,
    vd_out: float,
    vd_in: float,
    seq: List[int],
    prefix_k: int,
    mosek_threads: int = 1,
    use_bounds: bool = True,
    add_utilization_cut: bool = True,
    add_prefix_chain: bool = True,
    add_pair_cuts: bool = False,
) -> Tuple[float, float, str]:
    """
    LB SOCP relaxation:

    Variables per task i:
      - ship takeoff pos p_to[i], ship landing pos p_ld[i]
      - times: t_to[i] <= T_serv[i] <= t_ld[i], and t_ld[i]-t_to[i] <= R
      - mission end time t_end >= max_i t_ld[i]

    Constraints:
      - drone out/in cones (same spirit as your inner SOCP)
      - ship reachability cones:
          from start to p_to by t_to
          from start to p_ld by t_ld
          from p_to to p_ld by (t_ld - t_to)
          from p_to/p_ld to end by (t_end - t_to/t_ld)
      - optional utilization: sum_i (t_ld - t_to) <= L * t_end
      - optional prefix chain: enforce takeoff time order for first k tasks in seq,
        plus ship-speed cone between consecutive prefix takeoff positions.

    Objective:
      cv*vv*t_end + cd*vd*sum_i (t_ld - t_to)
    """
    if not HAS_MOSEK:
        return float("inf"), 0.0, "NO_MOSEK"

    tids = list(inst._task_ids)
    n = len(tids)
    if n == 0:
        return 0.0, 0.0, "EMPTY"

    tid_to_j = {tid: j for j, tid in enumerate(tids)}

    # Safe bounds (optional, but helps speed & stability)
    if use_bounds:
        xb, yb = vm.global_xy_bounds(inst, float(vd_out), float(vd_in))
        T_max = vm.conservative_time_ub(inst, n_events=2 * n + 2, xb=xb, yb=yb)
        xL, xU = float(xb[0]), float(xb[1])
        yL, yU = float(yb[0]), float(yb[1])
        tL, tU = 0.0, float(T_max)
    else:
        xL = yL = -1.0e9
        xU = yU =  1.0e9
        tL, tU = 0.0, 1.0e9

    vv = float(inst.vv)
    cv = float(inst.cv)
    vd = float(inst.vd)
    cd = float(inst.cd)
    R = float(inst.R)

    sx, sy = float(inst.port_start[0]), float(inst.port_start[1])
    ex, ey = float(inst.port_end[0]), float(inst.port_end[1])

    t0 = time.perf_counter()
    try:
        with Model("LBRelax") as M:
            if mosek_threads and mosek_threads > 0:
                M.setSolverParam("numThreads", int(mosek_threads))

            # Variables
            px_to = M.variable("px_to", n, Domain.inRange(xL, xU))
            py_to = M.variable("py_to", n, Domain.inRange(yL, yU))
            px_ld = M.variable("px_ld", n, Domain.inRange(xL, xU))
            py_ld = M.variable("py_ld", n, Domain.inRange(yL, yU))

            t_to = M.variable("t_to", n, Domain.inRange(tL, tU))
            t_ld = M.variable("t_ld", n, Domain.inRange(tL, tU))
            Tsv  = M.variable("Tsv",  n, Domain.inRange(tL, tU))
            t_end = M.variable("t_end", 1, Domain.inRange(tL, tU))
            t_end0 = t_end.index(0)

            # Per-task constraints
            for j, tid in enumerate(tids):
                tx, ty = inst.tasks[int(tid)]

                # time order within task
                M.constraint(Expr.sub(Tsv.index(j), t_to.index(j)), Domain.greaterThan(0.0))
                M.constraint(Expr.sub(t_ld.index(j), Tsv.index(j)), Domain.greaterThan(0.0))

                # endurance
                M.constraint(Expr.sub(t_ld.index(j), t_to.index(j)), Domain.lessThan(R))

                # must finish before mission end
                M.constraint(Expr.sub(t_end0, t_ld.index(j)), Domain.greaterThan(0.0))
                M.constraint(Expr.sub(t_end0, t_to.index(j)), Domain.greaterThan(0.0))

                # Drone cones: out and in
                M.constraint(
                    Expr.vstack(
                        Expr.mul(float(vd_out), Expr.sub(Tsv.index(j), t_to.index(j))),
                        Expr.sub(px_to.index(j), float(tx)),
                        Expr.sub(py_to.index(j), float(ty)),
                    ),
                    Domain.inQCone(),
                )
                M.constraint(
                    Expr.vstack(
                        Expr.mul(float(vd_in), Expr.sub(t_ld.index(j), Tsv.index(j))),
                        Expr.sub(px_ld.index(j), float(tx)),
                        Expr.sub(py_ld.index(j), float(ty)),
                    ),
                    Domain.inQCone(),
                )

                # Ship reach cones (necessary conditions)
                # start -> takeoff
                M.constraint(
                    Expr.vstack(
                        Expr.mul(vv, t_to.index(j)),
                        Expr.sub(px_to.index(j), sx),
                        Expr.sub(py_to.index(j), sy),
                    ),
                    Domain.inQCone(),
                )
                # start -> landing
                M.constraint(
                    Expr.vstack(
                        Expr.mul(vv, t_ld.index(j)),
                        Expr.sub(px_ld.index(j), sx),
                        Expr.sub(py_ld.index(j), sy),
                    ),
                    Domain.inQCone(),
                )
                # takeoff -> landing within (t_ld - t_to)
                M.constraint(
                    Expr.vstack(
                        Expr.mul(vv, Expr.sub(t_ld.index(j), t_to.index(j))),
                        Expr.sub(px_ld.index(j), px_to.index(j)),
                        Expr.sub(py_ld.index(j), py_to.index(j)),
                    ),
                    Domain.inQCone(),
                )
                # takeoff -> end within (t_end - t_to)
                M.constraint(
                    Expr.vstack(
                        Expr.mul(vv, Expr.sub(t_end0, t_to.index(j))),
                        Expr.sub(px_to.index(j), ex),
                        Expr.sub(py_to.index(j), ey),
                    ),
                    Domain.inQCone(),
                )
                # landing -> end within (t_end - t_ld)
                M.constraint(
                    Expr.vstack(
                        Expr.mul(vv, Expr.sub(t_end0, t_ld.index(j))),
                        Expr.sub(px_ld.index(j), ex),
                        Expr.sub(py_ld.index(j), ey),
                    ),
                    Domain.inQCone(),
                )

            # Utilization cut: sum(durations) <= L * t_end
            if add_utilization_cut and inst.L > 0:
                sum_dur = Expr.sum(Expr.sub(t_ld, t_to))
                M.constraint(Expr.sub(Expr.mul(float(inst.L), t_end0), sum_dur), Domain.greaterThan(0.0))

            # Pair-perimeter cuts (order-free, valid). The vessel travels a single
            # trajectory of length <= vv*t_end that must visit every one of the 2n
            # event points. For any two event points p,q:
            #   - closed-tour case (port_start == port_end == s): the sub-tour s->p->q->s
            #     has length >= |s-p| + |p-q| + |q-s|, so  vv*t_end >= a_p + b_pq + a_q.
            #   - open-tour case (start != end): the path visits both p and q, so the
            #     segment between them is >= |p-q|, giving the weaker chord cut
            #     vv*t_end >= b_pq (always valid).
            # a_p = ||point_p - s||, b_pq = ||point_p - point_q||, both epigraph cones.
            if add_pair_cuts and n >= 2:
                closed_tour = (abs(sx - ex) < 1e-9 and abs(sy - ey) < 1e-9)
                P = 2 * n

                def _PX(p):
                    return px_to.index(p) if p < n else px_ld.index(p - n)

                def _PY(p):
                    return py_to.index(p) if p < n else py_ld.index(p - n)

                if closed_tour:
                    a_star = M.variable("a_star", P, Domain.greaterThan(0.0))
                    for p in range(P):
                        M.constraint(
                            Expr.vstack(a_star.index(p), Expr.sub(_PX(p), sx), Expr.sub(_PY(p), sy)),
                            Domain.inQCone(),
                        )
                pairs = list(itertools.combinations(range(P), 2))
                b_pair = M.variable("b_pair", len(pairs), Domain.greaterThan(0.0))
                for e, (p, q) in enumerate(pairs):
                    M.constraint(
                        Expr.vstack(b_pair.index(e), Expr.sub(_PX(p), _PX(q)), Expr.sub(_PY(p), _PY(q))),
                        Domain.inQCone(),
                    )
                    if closed_tour:
                        M.constraint(
                            Expr.sub(
                                Expr.mul(vv, t_end0),
                                Expr.add(Expr.add(a_star.index(p), a_star.index(q)), b_pair.index(e)),
                            ),
                            Domain.greaterThan(0.0),
                        )
                    else:
                        M.constraint(
                            Expr.sub(Expr.mul(vv, t_end0), b_pair.index(e)),
                            Domain.greaterThan(0.0),
                        )

            # Prefix chain constraints (simulate B&B node fixing first k order).
            # WARNING: for k>=2 this FIXES the incumbent's launch order onto specific
            # physical task locations, deleting genuinely feasible reverse-order
            # solutions. The result is therefore an ORDER-RESTRICTED (single B&B node)
            # bound, NOT a valid global lower bound: it can strictly EXCEED the true
            # optimum, so gaps computed from it are unreliable (and can go negative).
            # Only k in {0,1} (chain off) and the order-free pair cuts are valid global
            # lower bounds. This path is retained for diagnostics only; do not report
            # its value as a certified bound.
            if add_prefix_chain and prefix_k >= 2:
                k = min(int(prefix_k), n)
                for r in range(k - 1):
                    a = int(seq[r])
                    b = int(seq[r + 1])
                    if a not in tid_to_j or b not in tid_to_j:
                        continue
                    ja = tid_to_j[a]
                    jb = tid_to_j[b]
                    # time order
                    M.constraint(Expr.sub(t_to.index(jb), t_to.index(ja)), Domain.greaterThan(0.0))
                    # ship displacement between the two takeoff positions
                    M.constraint(
                        Expr.vstack(
                            Expr.mul(vv, Expr.sub(t_to.index(jb), t_to.index(ja))),
                            Expr.sub(px_to.index(jb), px_to.index(ja)),
                            Expr.sub(py_to.index(jb), py_to.index(ja)),
                        ),
                        Domain.inQCone(),
                    )

            # Objective: same structure as your code/paper
            sum_dur = Expr.sum(Expr.sub(t_ld, t_to))
            obj = Expr.add(
                Expr.mul(cv * vv, t_end0),
                Expr.mul(cd * vd, sum_dur),
            )
            M.objective(ObjectiveSense.Minimize, obj)

            M.setLogHandler(None)
            M.solve()

            status = M.getPrimalSolutionStatus()
            t1 = time.perf_counter()

            if status == SolutionStatus.Optimal:
                # S2 soundness: report the DUAL objective as the lower bound. By weak
                # duality it satisfies dual <= opt(relaxation) <= opt(full problem), so it
                # is a certified valid bound; the primal objective of an interior-point
                # solution can sit slightly above the relaxation optimum and is not
                # guaranteed to bound the true optimum.
                lb_primal = float(M.primalObjValue())
                try:
                    lb_dual = float(M.dualObjValue())
                except Exception:
                    lb_dual = float("nan")
                lb_valid = lb_dual if math.isfinite(lb_dual) else lb_primal
                return lb_valid, (t1 - t0), "OPTIMAL"
            return float("inf"), (t1 - t0), f"NOT_OPTIMAL:{status}"

    except Exception as e:
        t1 = time.perf_counter()
        return float("inf"), (t1 - t0), f"ERROR:{type(e).__name__}"


def compute_ub(
    inst_dict: dict,
    ub_method: str,
    seed: int,
    mosek_threads: int,
    dro_cfg: vm.DROConfig,
    ub_max_iter: int,
    intensify_time: float,
) -> Tuple[vm.Instance, float, float, List[int], float, float]:
    """
    Returns: (inst, ub_obj, ub_time, seq, vd_out, vd_in)
    """
    inst = vm.Instance.from_dict(inst_dict)
    vd_out, vd_in, _, _ = dro_cfg.speeds(inst.vd)

    t0 = time.perf_counter()

    if ub_method == "nn":
        rng = random.Random(int(seed))
        seq = vm._nn_init(inst, rng)
        evaluator = vm.InnerEvaluator(
            inst,
            vd_out=float(vd_out),
            vd_in=float(vd_in),
            base_op_mode="interleaved",
            refine_steps=30,
            refine_evals=1,
            mosek_threads=int(mosek_threads),
            seed=int(seed) + 7,
            cache_size=2000,
            use_mosek_bounds=True,
        )
        ops = evaluator.build_ops(seq)
        res = evaluator.solve_socp_mosek(ops, need_details=False, assume_feasible=True, use_cache=True)
        ub_obj = float(res.get("obj", float("inf")))
        # a light refine to make UB more fair
        if math.isfinite(ub_obj):
            ub_obj, _ = evaluator.refine_ops(ops, ub_obj)

        t1 = time.perf_counter()
        return inst, ub_obj, (t1 - t0), seq, float(vd_out), float(vd_in)

    # PLNS-SA legacy mode / hybrid-lite: call palns_worker directly (single-process)
    params = {
        "dro_cfg": dro_cfg.__dict__,
        "base_op_mode": "interleaved",
        "repair_mode": "exact",
        "repair_k": 7,
        "destroy_frac": 0.30,
        "T_start": 5000.0,
        "alpha": 0.99,
        "max_iter": int(ub_max_iter),
        "mosek_threads": int(mosek_threads),
        "refine_policy": "best",
        "refine_steps": 30,
        "refine_evals": 1,
        "refine_trigger_rel": 0.10,
        "search_style": "best_only",
        "p_best": 1.0,
        "R_override": None,
        "cache_size": 5000,
        "use_mosek_bounds": True,
    }

    res = vm.palns_worker(int(seed), inst_dict, params)
    ub_obj = float(res.get("best_obj", float("inf")))
    seq = list(res.get("best_platform_seq", []))

    # hybrid-lite: do a short ops_intensify (optional)
    if ub_method == "hybrid" and math.isfinite(ub_obj) and seq:
        evaluator_main = vm.InnerEvaluator(
            inst,
            vd_out=float(vd_out),
            vd_in=float(vd_in),
            base_op_mode="interleaved",
            refine_steps=20,
            refine_evals=1,
            mosek_threads=int(mosek_threads),
            seed=int(seed) + 202,
            cache_size=5000,
            use_mosek_bounds=True,
        )
        base_ops = list(res.get("best_ops", []))
        obj2, _ops2 = vm.ops_intensify(
            inst=inst,
            evaluator=evaluator_main,
            platform_seq=seq,
            base_ops=base_ops,
            base_obj=ub_obj,
            time_limit=float(intensify_time),
            assign_trials=20,
            move_steps=60,
            move_candidates=10,
            seed=int(seed),
        )
        if math.isfinite(obj2) and obj2 + 1e-9 < ub_obj:
            ub_obj = float(obj2)

    t1 = time.perf_counter()
    return inst, ub_obj, (t1 - t0), seq, float(vd_out), float(vd_in)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--instance_index", type=int, default=0)

    ap.add_argument("--ub_method", type=str, default="palns", choices=["nn", "palns", "hybrid"])
    ap.add_argument("--ub_max_iter", type=int, default=60)
    ap.add_argument("--intensify_time", type=float, default=20.0)

    ap.add_argument("--k_list", type=str, default="0,5,10,20,50,100%")

    ap.add_argument("--mosek_threads", type=int, default=1)

    # DRO switches (match your main script)
    ap.add_argument("--no_dro", action="store_true")
    ap.add_argument("--eta_lower", type=float, default=1.0)
    ap.add_argument("--eta_upper", type=float, default=1.2)
    ap.add_argument("--mu_plus", type=float, default=1.1)
    ap.add_argument("--sigma_plus", type=float, default=0.05)
    ap.add_argument("--eps_out", type=float, default=0.05)
    ap.add_argument("--eps_in", type=float, default=0.05)
    ap.add_argument("--eps_sortie", type=float, default=None)

    # LB options
    ap.add_argument("--no_util_cut", action="store_true", help="disable sum(dur) <= L*t_end cut")
    ap.add_argument("--no_prefix_chain", action="store_true", help="disable prefix chain constraints")
    ap.add_argument("--no_bounds", action="store_true", help="disable variable bounds (debug)")
    ap.add_argument("--pair_cuts", action="store_true", help="add order-free pair-perimeter cuts (valid tightening of the no-order bound)")

    # CSV
    ap.add_argument("--csv", type=str, default="", help="append results to CSV file")

    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()

    if not HAS_MOSEK:
        raise SystemExit("[Fatal] MOSEK Fusion not available. Please install mosek + license.")

    # load instance
    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)
    inst_dict = data.get("instances", [data])[int(args.instance_index)]

    dro_cfg = vm.DROConfig(
        enabled=not bool(args.no_dro),
        eta_lower=float(args.eta_lower),
        eta_upper=float(args.eta_upper),
        mu_plus=float(args.mu_plus),
        sigma_plus=float(args.sigma_plus),
        eps_out=float(args.eps_out),
        eps_in=float(args.eps_in),
        eps_sortie=None if args.eps_sortie is None else float(args.eps_sortie),
    )

    inst, ub_obj, ub_time, seq, vd_out, vd_in = compute_ub(
        inst_dict=inst_dict,
        ub_method=str(args.ub_method),
        seed=int(args.seed),
        mosek_threads=int(args.mosek_threads),
        dro_cfg=dro_cfg,
        ub_max_iter=int(args.ub_max_iter),
        intensify_time=float(args.intensify_time),
    )

    n = len(inst._task_ids)
    k_list = _parse_k_list(str(args.k_list), n=n)
    if not k_list:
        k_list = [0]

    # CSV header
    if args.csv:
        _maybe_write_header(args.csv)

    rows = []
    for k in k_list:
        lb_obj, lb_time, lb_status = solve_lb_relax_socp(
            inst=inst,
            vd_out=float(vd_out),
            vd_in=float(vd_in),
            seq=seq if seq else list(inst._task_ids),
            prefix_k=int(k),
            mosek_threads=int(args.mosek_threads),
            use_bounds=not bool(args.no_bounds),
            add_utilization_cut=not bool(args.no_util_cut),
            add_prefix_chain=not bool(args.no_prefix_chain),
            add_pair_cuts=bool(args.pair_cuts),
        )

        gap = float("nan")
        if math.isfinite(ub_obj) and ub_obj > 1e-12 and math.isfinite(lb_obj):
            gap = (ub_obj - lb_obj) / ub_obj * 100.0

        rows.append([
            inst.instance_id, os.path.basename(args.json), int(args.instance_index),
            int(inst.n_tasks), int(inst.L), float(inst.R),
            str(args.ub_method), float(ub_obj), float(ub_time),
            int(k), float(lb_obj), float(lb_time), float(gap), str(lb_status),
        ])

    # pretty print
    print("\n=== UB ===")
    print(f"instance={inst.instance_id} n={inst.n_tasks} L={inst.L} R={inst.R}")
    print(f"ub_method={args.ub_method} ub_obj={ub_obj:.6f} ub_time={ub_time:.3f}s")
    print(f"seq(head)={seq[:20]}")

    print("\n=== LB (SOCP relax) ===")
    print("k\tlb_obj\t\tlb_time(s)\tgap(%)\tstatus")
    for r in rows:
        k, lb_obj, lb_time, gap, st = r[9], r[10], r[11], r[12], r[13]
        print(f"{k}\t{lb_obj:.6f}\t{lb_time:.3f}\t\t{gap:.3f}\t{st}")

    # append CSV
    if args.csv:
        with open(args.csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow(r)


if __name__ == "__main__":
    main()
