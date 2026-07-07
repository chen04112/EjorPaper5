#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce the large-instance diagnostic figures in Supplementary Appendix S.7.

The figures are illustrative diagnostics, not aggregate scalability tables.  This
script makes their provenance explicit: it regenerates the three large instances
from fixed seeds, solves each with the submitted PLNS-SOCP code under a short
deterministic diagnostic budget, writes the instance and plan records, and
recreates the spatial and timeline figures.

Default cases:
    n=200, L=3, instance/run seed 22260128
    n=300, L=4, instance/run seed 23260128
    n=400, L=5, instance/run seed 24260128
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "generator"))

import bundle_instances as gen  # type: ignore
import hybrid_vmurp_lbbd_v3 as hv3  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required to reproduce diagnostic figures") from exc


DEFAULT_CASES = [
    {"n": 200, "L": 3, "seed": 22260128},
    {"n": 300, "L": 4, "seed": 23260128},
    {"n": 400, "L": 5, "seed": 24260128},
]

PDF_METADATA = {
    "Creator": "reproduce_large_diagnostics.py",
    "Producer": "matplotlib",
    "CreationDate": None,
    "ModDate": None,
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_instance(case: Dict[str, int]) -> Dict[str, Any]:
    n = int(case["n"])
    L = int(case["L"])
    seed = int(case["seed"])
    gen.L_BY_SIZE.update({200: 3, 300: 4, 400: 5})
    spec = gen.step1_generate_one(seed, n)
    return {
        "instance_id": f"ExampleLarge_n{n}_L{L}_seed{seed}",
        "n_tasks": n,
        "L": L,
        "tasks": {str(k): [float(v[0]), float(v[1])] for k, v in spec.tasks.items()},
        "port_start": [float(spec.port_start[0]), float(spec.port_start[1])],
        "port_end": [float(spec.port_end[0]), float(spec.port_end[1])],
        "params": {
            "vv": float(spec.vv),
            "cv": float(spec.cv),
            "vd": float(spec.vd),
            "cd": float(spec.cd),
            "R": float(spec.R),
            "seed": seed,
        },
        "diagnostic_provenance": {
            "generator": "generator/bundle_instances.py",
            "layout_mode": gen.LAYOUT_MODE,
            "instance_seed": seed,
            "note": "Large diagnostic instance used for Supplementary Appendix S.7.",
        },
    }


def _solver_params(args: argparse.Namespace) -> Dict[str, Any]:
    dro_cfg = hv3.DROConfig(
        enabled=True,
        eta_lower=1.0,
        eta_upper=1.5,
        mu_plus=1.2,
        sigma_plus=0.1,
        eps_out=0.1,
        eps_in=0.1,
        eps_sortie=None,
    )
    return {
        "dro_cfg": dro_cfg.__dict__,
        "base_op_mode": "interleaved",
        "repair_mode": "sampled",
        "repair_k": int(args.repair_k),
        "destroy_frac": 0.30,
        "T_start": 5000.0,
        "alpha": 0.99,
        "max_iter": int(args.max_iter),
        "mosek_threads": 1,
        "refine_policy": "never",
        "refine_steps": 0,
        "refine_evals": 0,
        "refine_trigger_rel": 0.10,
        "search_style": "mix",
        "p_best": 0.75,
        "R_override": None,
        "cache_size": int(args.cache_size),
        "use_mosek_bounds": True,
        "debug_mosek": False,
        "fallback_ops": "one_uav",
        "assign_search": True,
        "p_assign_move": 0.50,
        "assign_move_frac": 0.34,
        "p_loiter_move": 0.25,
    }


def _op_to_dict(op: Any) -> Dict[str, Any]:
    if int(op.op_type) == hv3.Operation.TYPE_DEPOT:
        label = "DEP"
    elif int(op.op_type) == hv3.Operation.TYPE_TAKEOFF:
        label = f"TK({int(op.task_id)})"
    else:
        label = f"LD({int(op.task_id)})"
    return {
        "op_type": int(op.op_type),
        "event": label,
        "task_id": None if op.task_id is None else int(op.task_id),
        "uav_id": None if op.uav_id is None else int(op.uav_id),
    }


def _active_counts(ops: List[Any]) -> List[int]:
    active = 0
    out: List[int] = []
    for op in ops:
        if int(op.op_type) == hv3.Operation.TYPE_TAKEOFF:
            active += 1
        elif int(op.op_type) == hv3.Operation.TYPE_LAND:
            active -= 1
        out.append(active)
    return out


def _ship_drone_times(inst: Any, ops: List[Any], details: Dict[str, Any]) -> Tuple[float, float, int]:
    times = list(details["times"])
    ship_time = float(times[-1])
    tk: Dict[int, int] = {}
    ld: Dict[int, int] = {}
    for idx, op in enumerate(ops):
        if int(op.op_type) == hv3.Operation.TYPE_TAKEOFF:
            tk[int(op.task_id)] = idx
        elif int(op.op_type) == hv3.Operation.TYPE_LAND:
            ld[int(op.task_id)] = idx
    drone_time = 0.0
    for tid in sorted(tk):
        drone_time += float(times[ld[tid]] - times[tk[tid]])
    return ship_time, drone_time, max(_active_counts(ops))


def _plot_plan(inst: Any, ops: List[Any], details: Dict[str, Any], title: str, out: Path) -> None:
    px, py = list(details["pos_x"]), list(details["pos_y"])
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    task_ids = sorted(inst.tasks)
    task_ids = [i for i in task_ids if int(i) != 0]
    ax.scatter([inst.tasks[i][0] for i in task_ids], [inst.tasks[i][1] for i in task_ids],
               s=7, c="#4C78A8", alpha=0.55, label="tasks")
    ax.plot(px, py, color="black", lw=1.5, alpha=0.85, label="vessel")
    ax.scatter([inst.port_start[0]], [inst.port_start[1]], marker="D", s=38, c="#111111", label="depot")

    tk: Dict[int, int] = {}
    ld: Dict[int, int] = {}
    for idx, op in enumerate(ops):
        if int(op.op_type) == hv3.Operation.TYPE_TAKEOFF:
            tk[int(op.task_id)] = idx
        elif int(op.op_type) == hv3.Operation.TYPE_LAND:
            ld[int(op.task_id)] = idx
    # Draw a deterministic subset of sorties to keep dense figures readable.
    rng = random.Random(19)
    draw_ids = list(tk)
    rng.shuffle(draw_ids)
    draw_ids = sorted(draw_ids[: min(80, len(draw_ids))])
    for tid in draw_ids:
        i, j = tk[tid], ld[tid]
        tx, ty = inst.tasks[tid]
        ax.plot([px[i], tx, px[j]], [py[i], ty, py[j]], color="#F58518", ls="--", lw=0.45, alpha=0.22)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=7, frameon=False)
    fig.tight_layout()
    if out.suffix.lower() == ".pdf":
        fig.savefig(out, metadata=PDF_METADATA)
    else:
        fig.savefig(out)
    plt.close(fig)


def _plot_timeline(inst: Any, ops: List[Any], details: Dict[str, Any], title: str, out: Path) -> None:
    times = list(details["times"])
    counts = _active_counts(ops)
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.step(times, counts, where="post", color="#D62728", lw=1.5)
    ax.axhline(inst.L, color="0.35", lw=0.8, ls="--")
    ax.set_ylim(-0.2, inst.L + 0.8)
    ax.set_xlabel("event time")
    ax.set_ylabel("active sorties")
    ax.set_title(title)
    fig.tight_layout()
    if out.suffix.lower() == ".pdf":
        fig.savefig(out, metadata=PDF_METADATA)
    else:
        fig.savefig(out)
    plt.close(fig)


def _make_triptych(paths: List[Path], out: Path, kind: str) -> None:
    # The submitted supplement includes these as PDF figures.  We rebuild a
    # compact triptych from the individual vector PDFs.
    from matplotlib.backends.backend_pdf import PdfPages
    # Matplotlib cannot embed PDF pages directly without extra dependencies, so
    # we recreate the triptych by reading the saved PNG previews generated below.
    pngs = [p.with_suffix(".png") for p in paths]
    fig, axes = plt.subplots(1, len(pngs), figsize=(14, 4.4 if kind == "plan" else 3.0))
    if len(pngs) == 1:
        axes = [axes]
    for ax, png in zip(axes, pngs):
        img = plt.imread(str(png))
        ax.imshow(img)
        ax.axis("off")
    fig.tight_layout(pad=0.05)
    with PdfPages(out, metadata=PDF_METADATA) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _also_png(pdf_path: Path) -> Path:
    # Render by saving a PNG version from the current figure is simpler, so this
    # function is only a path helper.  Each plotter saves PDF and PNG together.
    return pdf_path.with_suffix(".png")


def run_case(case: Dict[str, int], args: argparse.Namespace, params: Dict[str, Any], outdir: Path) -> Dict[str, Any]:
    raw = _make_instance(case)
    inst = hv3.Instance.from_dict(raw)
    run_seed = int(case["seed"])

    t0 = time.perf_counter()
    res = hv3.palns_worker(run_seed, raw, params)
    elapsed = time.perf_counter() - t0

    dro = hv3.DROConfig(**params["dro_cfg"])
    vd_out, vd_in, hat_out, hat_in = dro.speeds(inst.vd)
    evaluator = hv3.InnerEvaluator(
        inst,
        vd_out=vd_out,
        vd_in=vd_in,
        base_op_mode=params["base_op_mode"],
        refine_steps=0,
        refine_evals=0,
        mosek_threads=1,
        seed=run_seed + 202,
        cache_size=0,
        use_mosek_bounds=True,
    )
    details = evaluator.solve_socp_mosek(res["best_ops"], need_details=True, assume_feasible=True, use_cache=False)
    if not math.isfinite(float(details.get("obj", float("inf")))):
        raise RuntimeError(f"Detailed SOCP solve failed for {raw['instance_id']}: {details}")

    inst_dir = outdir / "instances"
    plan_dir = outdir / "plans"
    fig_dir = outdir / "figures"
    for d in (inst_dir, plan_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    case_id = raw["instance_id"]
    inst_file = inst_dir / f"{case_id}.json"
    inst_file.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    ship_time, drone_time, max_active = _ship_drone_times(inst, res["best_ops"], details)
    plan = {
        "case_id": case_id,
        "instance_file": str(inst_file.relative_to(outdir).as_posix()),
        "solver": {
            "script": "src/hybrid_vmurp_lbbd_v3.py",
            "runner": "reproduce_large_diagnostics.py",
            "run_seed": run_seed,
            "params": params,
        },
        "objective": float(details["obj"]),
        "runtime_note": "Wall-clock time is printed by the script but not stored because it is environment dependent.",
        "ship_time": float(ship_time),
        "drone_time": float(drone_time),
        "max_active": int(max_active),
        "hat_out": float(hat_out),
        "hat_in": float(hat_in),
        "vd_out": float(vd_out),
        "vd_in": float(vd_in),
        "ops": [_op_to_dict(o) for o in res["best_ops"]],
        "event_locations": {
            "pos_x": [float(x) for x in details["pos_x"]],
            "pos_y": [float(y) for y in details["pos_y"]],
            "times": [float(t) for t in details["times"]],
        },
    }
    plan_file = plan_dir / f"{case_id}_plan.json"
    plan_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    plan_pdf = fig_dir / f"Fig_LargeInstance_Plan_{case_id}.pdf"
    time_pdf = fig_dir / f"Fig_LargeInstance_Timeline_{case_id}.pdf"
    title = f"n={case['n']}, L={case['L']}, seed={case['seed']}"
    _plot_plan(inst, res["best_ops"], details, title, plan_pdf)
    _plot_plan(inst, res["best_ops"], details, title, _also_png(plan_pdf))
    _plot_timeline(inst, res["best_ops"], details, title, time_pdf)
    _plot_timeline(inst, res["best_ops"], details, title, _also_png(time_pdf))

    return {
        "case_id": case_id,
        "n": int(case["n"]),
        "L": int(case["L"]),
        "instance_seed": int(case["seed"]),
        "run_seed": int(run_seed),
        "max_iter": int(args.max_iter),
        "repair_k": int(args.repair_k),
        "objective": float(details["obj"]),
        "runtime_note": "Wall-clock time is printed by the script but not stored because it is environment dependent.",
        "ship_time": float(ship_time),
        "drone_time": float(drone_time),
        "max_active": int(max_active),
        "ops_len": len(res["best_ops"]),
        "hat_out": float(hat_out),
        "hat_in": float(hat_in),
        "instance_file": str(inst_file.relative_to(outdir).as_posix()),
        "plan_file": str(plan_file.relative_to(outdir).as_posix()),
        "plan_figure": str(plan_pdf.relative_to(outdir).as_posix()),
        "timeline_figure": str(time_pdf.relative_to(outdir).as_posix()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce Supplementary Appendix S.7 large diagnostics.")
    ap.add_argument("--outdir", default="large_diagnostics", help="Output directory inside the repository.")
    ap.add_argument("--max-iter", type=int, default=12, help="PLNS iterations per diagnostic case.")
    ap.add_argument("--repair-k", type=int, default=5, help="Sampled insertion candidates.")
    ap.add_argument("--cache-size", type=int, default=2000, help="SOCP oracle cache size used by the PLNS worker.")
    args = ap.parse_args()

    if not hv3.HAS_MOSEK:
        raise RuntimeError("MOSEK Fusion is required for the SOCP oracle.")

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    params = _solver_params(args)

    rows = [run_case(c, args, params, outdir) for c in DEFAULT_CASES]

    summary = outdir / "results" / "large_diagnostic_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig_paths = [outdir / r["plan_figure"] for r in rows]
    time_paths = [outdir / r["timeline_figure"] for r in rows]
    _make_triptych(fig_paths, outdir / "figures" / "Fig_LargeInstance_Plan_Triptych.pdf", "plan")
    _make_triptych(time_paths, outdir / "figures" / "Fig_LargeInstance_Timeline_Triptych.pdf", "timeline")

    manifest_files = [summary]
    manifest_files += list((outdir / "instances").glob("*.json"))
    manifest_files += list((outdir / "plans").glob("*.json"))
    manifest_files += list((outdir / "figures").glob("*"))
    manifest = {
        "scope": "Large diagnostic reproduction for Supplementary Appendix S.7",
        "cases": DEFAULT_CASES,
        "solver_budget": {"max_iter": int(args.max_iter), "repair_k": int(args.repair_k), "workers": 1},
        "files": [
            {
                "file": str(p.relative_to(outdir).as_posix()),
                "bytes": p.stat().st_size,
                "sha256": _sha256(p),
            }
            for p in sorted(set(manifest_files), key=lambda q: q.as_posix())
            if p.is_file()
        ],
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {summary.relative_to(ROOT)}")
    for r in rows:
        print(f"{r['case_id']}: obj={r['objective']:.3f}")


if __name__ == "__main__":
    main()
