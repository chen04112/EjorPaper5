#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_vmurp_v3_paper_tests_v3_fixed.py

Paper-style benchmark & evidence runner for Hybrid VMURP v3 (PLNS-SA + SOCP oracle + OPS-level local refinement),
with **MC evaluation** and **small-scale exact certification** integrated.

REVISION NOTE (Scheme B Fix):
-----------------------------
Fixed the Beta distribution sampling logic in `_sample_eta_one`.
Previously, if sigma was too large for the given range/mu, it silently fell back
to a Uniform distribution. This caused excessive sampling of extreme boundary values
(high infeasibility) in DRO validation.
Now, it clamps the variance to the theoretical maximum and issues a one-time warning,
preserving the bell-shaped distribution.

Key fixes vs previous runner versions
-------------------------------------
1) Unit-test tolerance: MOSEK bounds can change objective at ~1e-5 due to numeric tolerances.
   The "bounds vs no-bounds" check now uses a reasonable tolerance.

2) Data generation-first: if you have no inputs, `--suite paper` will auto-generate synthetic
   instances that match ExpA/ExpD needs (and label them as source_file="GENERATED").

3) MC & cert are *actually executed*, filling the NaN columns you observed:
   - mc_* metrics are computed via a wait-and-sync execution simulation (ship may wait; UAV may loiter),
     and infeasibility is counted when UAV airborne time exceeds endurance R.
   - cert_* metrics come from an event-level MISOCP solved by MOSEK (primal objective + best bound).

Usage (typical)
---------------
python run_vmurp_v3_paper_tests_v3_fixed.py ^
  --inputs ./instances/ ^
  --outdir ./vmurp_v3_results ^
  --suite paper ^
  --runs 3 ^
  --palns_workers 10 ^
  --max_iter 100 ^
  --intensify_time_limit 40 ^
  --mosek_threads 1 ^
  --plots

Notes
-----
- Keep --mosek_threads 1 if palns_workers > 1 (legacy flag name; avoids oversubscription).
- This script imports and calls your solver implementation in `hybrid_vmurp_lbbd_v3.py`.

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Optional analysis stack
try:
    import pandas as pd  # type: ignore

    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt  # type: ignore

    HAS_PLOT = True
except Exception:
    HAS_PLOT = False


# -----------------------------------------------------------------------------
# Import the solver module (hybrid_vmurp_lbbd_v3.py)
# -----------------------------------------------------------------------------

def _import_solver_module(solver_path: Optional[str] = None):
    """
    Import `hybrid_vmurp_lbbd_v3.py` as an importable module.

    Multiprocessing on Windows (spawn) requires that the module containing
    `palns_worker` / `Operation` is importable by *name*. Therefore:
      - If `solver_path` is provided, we add its directory to sys.path and
        import by its filename stem.
      - Otherwise we try `import hybrid_vmurp_lbbd_v3` normally.
      - If that fails, we fall back to loading from the same directory as this runner,
        again via sys.path + import-by-stem.
    """
    import importlib

    if solver_path:
        sp = Path(solver_path).resolve()
        if not sp.exists():
            raise FileNotFoundError(f"--solver_script not found: {sp}")
        if str(sp.parent) not in sys.path:
            sys.path.insert(0, str(sp.parent))
        return importlib.import_module(sp.stem)

    try:
        import hybrid_vmurp_lbbd_v3 as hv3  # type: ignore
        return hv3
    except Exception:
        here = Path(__file__).resolve().parent
        sp = here / "hybrid_vmurp_lbbd_v3.py"
        if not sp.exists():
            raise
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))
        return importlib.import_module(sp.stem)


# -----------------------------------------------------------------------------
# Progress / logging utilities
# -----------------------------------------------------------------------------

def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    """Print with timestamp + flush, unless --quiet."""
    try:
        if args_global is not None and bool(getattr(args_global, "quiet", False)):
            return
    except Exception:
        pass
    print(f"[{_ts()}] {msg}", flush=True)


def _run_with_heartbeat(label: str, fn, heartbeat_sec: float) -> Any:
    """
    Run fn() while emitting a heartbeat line every heartbeat_sec seconds.
    This prevents the impression of a "hang" during long MOSEK / PLNS-SA stages.
    """
    hb = float(heartbeat_sec)
    if hb <= 0:
        return fn()

    done = threading.Event()
    t0 = time.perf_counter()

    def _hb_loop() -> None:
        while not done.wait(hb):
            el = time.perf_counter() - t0
            _log(f"{label} ... running (elapsed={el:.1f}s)")

    th = threading.Thread(target=_hb_loop, daemon=True)
    th.start()
    try:
        return fn()
    finally:
        done.set()
        try:
            th.join(timeout=0.2)
        except Exception:
            pass


def _fmt(v: Any, *, nd: int = 3) -> str:
    """Safe float formatting (handles nan/inf/None)."""
    try:
        x = float(v)
        if not math.isfinite(x):
            return str(v)
        return f"{x:.{nd}f}"
    except Exception:
        return str(v)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _nanmean(xs: Sequence[float]) -> float:
    ys = [x for x in xs if x is not None and math.isfinite(float(x))]
    return float(statistics.fmean(ys)) if ys else float("nan")


def _nanstd(xs: Sequence[float]) -> float:
    ys = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if len(ys) <= 1:
        return float("nan")
    return float(statistics.pstdev(ys))


def _percentile(xs: Sequence[float], q: float) -> float:
    ys = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not ys:
        return float("nan")
    if q <= 0:
        return float(ys[0])
    if q >= 100:
        return float(ys[-1])
    # linear interpolation
    p = (q / 100.0) * (len(ys) - 1)
    lo = int(math.floor(p))
    hi = int(math.ceil(p))
    if lo == hi:
        return float(ys[lo])
    frac = p - lo
    return float(ys[lo] * (1.0 - frac) + ys[hi] * frac)


def _euclid(ax: float, ay: float, bx: float, by: float) -> float:
    return float(math.hypot(ax - bx, ay - by))


def _stable_mod_hash(mod: int, *parts: Any) -> int:
    """Deterministic small integer hash (unlike Python built-in hash)."""
    import hashlib
    h = hashlib.blake2b(("|".join(str(p) for p in parts)).encode("utf-8"), digest_size=8).digest()
    v = int.from_bytes(h, "big", signed=False)
    return int(v % int(mod))


# -----------------------------------------------------------------------------
# Synthetic instance generation (schema compatible with hv3.Instance.from_dict)
# -----------------------------------------------------------------------------

def _gen_tasks_xy(rng: random.Random, n_tasks: int, *, pattern: str, scale: float) -> Dict[str, List[float]]:
    """
    Generate task coordinates (including key "0" as dummy).
    Patterns are intentionally simple and "paper friendly".
    """
    pattern = str(pattern).lower()
    tasks: Dict[str, List[float]] = {"0": [0.0, 0.0]}

    if pattern == "uniform":
        for i in range(1, n_tasks + 1):
            x = rng.uniform(-scale, scale)
            y = rng.uniform(-scale, scale)
            tasks[str(i)] = [float(x), float(y)]

    elif pattern == "cluster":
        # 3 Gaussian-ish clusters
        centers = [(0.6 * scale, 0.3 * scale), (-0.5 * scale, -0.4 * scale), (0.1 * scale, -0.6 * scale)]
        for i in range(1, n_tasks + 1):
            cx, cy = centers[rng.randrange(len(centers))]
            x = rng.gauss(cx, 0.18 * scale)
            y = rng.gauss(cy, 0.18 * scale)
            tasks[str(i)] = [float(x), float(y)]

    elif pattern == "corridor":
        # tasks stretched along x-axis corridor
        for i in range(1, n_tasks + 1):
            x = rng.uniform(-scale, scale)
            y = rng.gauss(0.0, 0.12 * scale)
            tasks[str(i)] = [float(x), float(y)]

    elif pattern == "ring":
        for i in range(1, n_tasks + 1):
            ang = rng.uniform(0, 2 * math.pi)
            r = rng.gauss(0.75 * scale, 0.08 * scale)
            tasks[str(i)] = [float(r * math.cos(ang)), float(r * math.sin(ang))]

    else:
        raise ValueError(f"Unknown pattern: {pattern}. Use uniform/cluster/corridor/ring")

    return tasks


def generate_synthetic_instance_dict(
        *,
        seed: int,
        n_tasks: int,
        L: int,
        pattern: str,
        depot_mode: str,
        scale: float,
        vv: float,
        cv: float,
        vd: float,
        cd: float,
        R: float,
        instance_id: str,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))

    tasks = _gen_tasks_xy(rng, int(n_tasks), pattern=pattern, scale=float(scale))

    # depots
    if str(depot_mode).lower() == "separate":
        port_start = [0.0, 0.0]
        port_end = [0.0, 0.0]
    elif str(depot_mode).lower() == "line":
        port_start = [-0.9 * scale, 0.0]
        port_end = [0.9 * scale, 0.0]
    else:
        raise ValueError("depot_mode must be separate|line")

    return {
        "instance_id": str(instance_id),
        "name": str(instance_id),
        "n_tasks": int(n_tasks),
        "n": int(n_tasks),
        "L": int(L),
        "layout_variant": str(pattern),
        "layout_source": "generated",
        "port_start": port_start,
        "port_end": port_end,
        "tasks": tasks,
        "params": {"vv": float(vv), "cv": float(cv), "vd": float(vd), "cd": float(cd), "R": float(R)},
    }


def build_generated_bundles_for_suite(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Build a list of instance bundles (source_file, instance_index, instance dict)
    to run for the selected suite.

    For --suite paper:
      - ExpA: n in {20,40,60,80} with L = {2,3,4,5} (aligned with your existing CSVs)
      - ExpD fair/mc: n=40, L=3
      - ExpD cert: n=12, L=2
    """
    suite = str(args.suite).lower()
    bundles: List[Dict[str, Any]] = []

    # Global parameter defaults for synthetic instances (paper-friendly)
    vv = float(args.vv)
    cv = float(args.cv)
    vd = float(args.vd)
    cd = float(args.cd)
    R = float(args.R_inst)

    def add_block(n_tasks: int, L: int, n_instances: int, seed0: int, tag: str):
        for j in range(int(n_instances)):
            seed = int(seed0) + j
            iid = f"syn_n{n_tasks}_L{L}_{j}"
            inst = generate_synthetic_instance_dict(
                seed=seed,
                n_tasks=n_tasks,
                L=L,
                pattern=str(args.pattern),
                depot_mode=str(args.depot_mode),
                scale=float(args.scale),
                vv=vv,
                cv=cv,
                vd=vd,
                cd=cd,
                R=R,
                instance_id=iid,
            )
            bundles.append(
                {
                    "source_file": "GENERATED",
                    "instance_index": j,
                    "instance_id": iid,
                    "instance": inst,
                    "_tag": tag,
                    "layout_id": iid,
                    "layout_variant": str(args.pattern),
                    "layout_source": "generated",
                    "subset_rule": str(tag),
                    "perturbation_rule": "",
                }
            )

    if suite in ("paper", "quick", "evidence"):
        # ExpA scale set
        ns = [20, 40, 60, 80] if suite == "paper" else ([20, 40] if suite == "quick" else [])
        for n_tasks in ns:
            L = max(1, n_tasks // 20)  # 20->1, 40->2, ... but your CSV used 20->2
            # align with your earlier results: 20->2,40->3,60->4,80->5
            if n_tasks == 20:
                L = 2
            elif n_tasks == 40:
                L = 3
            elif n_tasks == 60:
                L = 4
            elif n_tasks == 80:
                L = 5
            add_block(n_tasks=n_tasks, L=L, n_instances=int(args.n_instances_A), seed0=1000 + n_tasks, tag="ExpA")

        # ExpD
        add_block(n_tasks=40, L=3, n_instances=int(args.n_instances_D40), seed0=2000, tag="ExpD40")
        add_block(n_tasks=12, L=2, n_instances=int(args.n_instances_D12), seed0=3000, tag="ExpD12")

    else:
        # For other suites, only generate if requested explicitly
        if bool(args.gen_if_empty):
            add_block(n_tasks=int(args.n_tasks), L=int(args.L), n_instances=int(args.n_instances),
                      seed0=int(args.gen_seed), tag="CUSTOM")

    return bundles


def load_instance_bundles_from_inputs(inputs: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Load JSON instances from a folder/list of files.
    Each bundle is:
      {source_file, instance_index, instance_id, instance}
    """
    files: List[Path] = []
    for p in inputs:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(pp.glob("*.json")))
        elif pp.is_file() and pp.suffix.lower() == ".json":
            files.append(pp)

    bundles: List[Dict[str, Any]] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        inst_list = data.get("instances", None)
        if isinstance(inst_list, list):
            for idx, inst in enumerate(inst_list):
                iid = str(inst.get("instance_id", f"{f.stem}_{idx}"))
                bundles.append(
                    {"source_file": str(f), "instance_index": int(idx), "instance_id": iid, "instance": inst})
        elif isinstance(data, dict) and "n_tasks" in data:
            iid = str(data.get("instance_id", f.stem))
            bundles.append({"source_file": str(f), "instance_index": 0, "instance_id": iid, "instance": data})
    return bundles


# -----------------------------------------------------------------------------
# Experiment config objects (ExpA + ExpD)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SolverCfg:
    # meta
    name: str
    subexp: str  # ExpA/fair/mc/cert
    mode: str  # palns|hybrid
    base_op_mode: str  # interleaved|adjacent

    # budgets (paper bookkeeping)
    total_time_budget: float
    palns_share: float
    palns_time_limit: float
    intensify_time_limit: float

    # PLNS-SA core (legacy palns_* field names are retained for traceability)
    palns_workers: int
    max_iter: int
    mosek_threads: int
    policy_type: str = ""
    buffer_rule: str = ""
    buffer_eta: Optional[float] = None
    mc_eta_lower: Optional[float] = None
    mc_eta_upper: Optional[float] = None
    mc_mu_plus: Optional[float] = None
    mc_sigma_plus: Optional[float] = None
    block_cut_tasks: int = 1
    seed_offset: int = 0

    # LNS repair/destroy knobs (keep aligned with solver defaults)
    repair_mode: str = "exact"
    repair_k: int = 7
    destroy_frac: float = 0.30
    T_start: float = 5000.0
    alpha: float = 0.99
    search_style: str = "mix"
    p_best: float = 0.75

    # refine
    refine_policy: str = "best"
    refine_steps: int = 30
    refine_evals: int = 1
    refine_trigger_rel: float = 0.10

    # intensify knobs
    assign_trials: int = 40
    move_steps: int = 120
    move_candidates: int = 10

    # speed knobs
    cache_size: int = 5000
    use_mosek_bounds: bool = True

    # DRO params
    dro_enabled: bool = True
    eta_lower: float = 1.0
    eta_upper: float = 1.2
    mu_plus: float = 1.1
    sigma_plus: float = 0.05
    eps_out: float = 0.05
    eps_in: float = 0.05
    eps_sortie: Optional[float] = None


def build_expA_cfgs(args: argparse.Namespace) -> List[SolverCfg]:
    """
    ExpA aligns with your existing config names: hybrid_main + ablations.
    """
    # "paper budget": not hard-enforced; used for logging only
    total_budget = float(args.total_budget_A)

    base = SolverCfg(
        name="hybrid_main",
        subexp="ExpA",
        mode="hybrid",
        base_op_mode="interleaved",
        policy_type="proposed_reliability_aware",
        total_time_budget=total_budget,
        palns_share=float(args.palns_share_A),
        palns_time_limit=total_budget * float(args.palns_share_A),
        intensify_time_limit=float(args.intensify_time_limit),
        palns_workers=int(args.palns_workers),
        max_iter=int(args.max_iter),
        mosek_threads=int(args.mosek_threads),
        cache_size=int(args.cache_size),
        use_mosek_bounds=bool(not args.no_mosek_bounds),
        dro_enabled=bool(not args.no_dro),
        eta_lower=float(args.eta_lower),
        eta_upper=float(args.eta_upper),
        mu_plus=float(args.mu_plus),
        sigma_plus=float(args.sigma_plus),
        eps_out=float(args.eps_out),
        eps_in=float(args.eps_in),
        eps_sortie=None if args.eps_sortie is None else float(args.eps_sortie),
        refine_policy=str(args.refine_policy),
        refine_steps=int(args.refine_steps),
        refine_evals=int(args.refine_evals),
        assign_trials=int(args.assign_trials),
        move_steps=int(args.move_steps),
        move_candidates=int(args.move_candidates),
    )

    suite: List[SolverCfg] = []

    # baseline palns-only
    suite.append(
        SolverCfg(
            **{
                **asdict(base),
                "name": "palns_only",
                "mode": "palns",
                "policy_type": "palns_only",
                "intensify_time_limit": 0.0,
            }
        )
    )

    # main
    suite.append(base)

    # ablations (keep names matching your ExpA csv)
    suite.append(
        SolverCfg(
            **{
                **asdict(base),
                "name": "abl_no_dro",
                "policy_type": "static_nominal",
                "dro_enabled": False,
            }
        )
    )
    suite.append(
        SolverCfg(
            **{
                **asdict(base),
                "name": "abl_adjacent",
                "policy_type": "adjacent_only",
                "base_op_mode": "adjacent",
            }
        )
    )
    suite.append(SolverCfg(**{**asdict(base), "name": "abl_no_cache", "cache_size": 0}))
    suite.append(SolverCfg(**{**asdict(base), "name": "abl_no_bounds", "use_mosek_bounds": False}))
    suite.append(SolverCfg(**{**asdict(base), "name": "abl_no_refine", "refine_policy": "never"}))

    # A tiny quick mode
    if str(args.suite).lower() == "quick":
        keep = {"palns_only", "hybrid_main", "abl_no_dro", "abl_adjacent"}
        suite = [c for c in suite if c.name in keep]

    # deterministic order
    order = {name: i for i, name in enumerate([c.name for c in suite])}
    suite.sort(key=lambda c: order.get(c.name, 9999))
    return suite


def build_expD_cfgs(args: argparse.Namespace) -> List[SolverCfg]:
    """
    ExpD uses config names matching your existing ExpD_Evidence_*.csv:

      fair: fair_budget_palns vs fair_budget_hybrid
      mc  : mc_nominal vs mc_dro
      cert: cert_heuristic
    """
    total_budget = float(args.total_budget_D)
    palns_share = float(args.palns_share_D)

    # fair
    fair_palns = SolverCfg(
        name="fair_budget_palns",
        subexp="fair",
        mode="palns",
        base_op_mode="interleaved",
        policy_type="palns_only",
        total_time_budget=total_budget,
        palns_share=1.0,
        palns_time_limit=total_budget,
        intensify_time_limit=0.0,
        palns_workers=int(args.palns_workers),
        max_iter=int(args.max_iter),
        mosek_threads=int(args.mosek_threads),
        cache_size=int(args.cache_size),
        use_mosek_bounds=bool(not args.no_mosek_bounds),
        dro_enabled=True,  # your existing ExpD fair used DRO=1
        eta_lower=float(args.eta_lower),
        eta_upper=float(args.eta_upper),
        mu_plus=float(args.mu_plus),
        sigma_plus=float(args.sigma_plus),
        eps_out=float(args.eps_out),
        eps_in=float(args.eps_in),
        eps_sortie=None,
        refine_policy=str(args.refine_policy),
        refine_steps=int(args.refine_steps),
        refine_evals=int(args.refine_evals),
        assign_trials=int(args.assign_trials),
        move_steps=int(args.move_steps),
        move_candidates=int(args.move_candidates),
    )
    fair_hybrid = SolverCfg(**{**asdict(fair_palns), "name": "fair_budget_hybrid", "mode": "hybrid",
                               "policy_type": "proposed_reliability_aware",
                               "palns_share": palns_share,
                               "palns_time_limit": total_budget * palns_share,
                               "intensify_time_limit": total_budget * (1.0 - palns_share)})

    # mc
    mc_nom = SolverCfg(**{**asdict(fair_hybrid), "name": "mc_nominal", "subexp": "mc",
                          "policy_type": "static_nominal", "dro_enabled": False})
    mc_dro = SolverCfg(**{**asdict(fair_hybrid), "name": "mc_dro", "subexp": "mc",
                          "policy_type": "proposed_reliability_aware", "dro_enabled": True})
    mc_buffer = None
    if args.buffer_eta is not None:
        eta_buf = float(args.buffer_eta)
        mc_buffer = SolverCfg(
            **{
                **asdict(fair_hybrid),
                "name": "mc_buffered",
                "subexp": "mc",
                "policy_type": "static_buffered",
                "buffer_rule": "global_speed_shrink",
                "buffer_eta": eta_buf,
                "dro_enabled": True,
                "eta_lower": eta_buf,
                "eta_upper": eta_buf,
                "mu_plus": eta_buf,
                "sigma_plus": 0.0,
                # Preserve the global uncertainty environment in execution MC.
                "mc_eta_lower": float(args.eta_lower),
                "mc_eta_upper": float(args.eta_upper),
                "mc_mu_plus": float(args.mu_plus),
                "mc_sigma_plus": float(args.sigma_plus),
            }
        )
    mc_block = None
    if bool(getattr(args, "enable_block_replan", False)):
        mc_block = SolverCfg(
            **{
                **asdict(fair_hybrid),
                "name": "mc_block_nominal",
                "subexp": "mc",
                "policy_type": "rh_block_nominal",
                "dro_enabled": False,
                "block_cut_tasks": int(getattr(args, "block_cut_tasks", 1)),
            }
        )

    # cert
    cert_cfg = SolverCfg(**{**asdict(fair_hybrid), "name": "cert_heuristic", "subexp": "cert",
                            "dro_enabled": True,
                            "total_time_budget": float(args.total_budget_cert),
                            "palns_time_limit": float(args.total_budget_cert) * palns_share,
                            "intensify_time_limit": float(args.total_budget_cert) * (1.0 - palns_share)})

    suite = [fair_palns, fair_hybrid, mc_nom]
    if mc_buffer is not None:
        suite.append(mc_buffer)
    if mc_block is not None:
        suite.append(mc_block)
    suite.extend([mc_dro, cert_cfg])
    return suite


# -----------------------------------------------------------------------------
# MC evaluation (wait-and-sync simulation)
# -----------------------------------------------------------------------------

def _sample_eta_one(
        rng: random.Random,
        *,
        dist: str,
        eta_lower: float,
        eta_upper: float,
        mu: float,
        sigma: float,
) -> float:
    dist = str(dist).lower()
    lo = float(eta_lower)
    hi = float(eta_upper)
    if hi < lo:
        lo, hi = hi, lo

    if dist == "uniform":
        return float(rng.uniform(lo, hi))

    if dist == "truncnorm":
        # rejection sampling (OK for small sigma)
        for _ in range(2000):
            x = rng.gauss(float(mu), float(sigma))
            if lo <= x <= hi:
                return float(x)
        return float(min(hi, max(lo, mu)))

    if dist == "beta":
        # beta on [lo,hi] using python's betavariate
        # infer alpha/beta from target mean/std; if impossible, CLAMP and WARN instead of fallback to uniform
        width = hi - lo
        if width <= 1e-12:
            return float(lo)

        m = (float(mu) - lo) / width
        v = (float(sigma) / width) ** 2

        # Clip mean to (0,1) exclusive for beta calculation
        m = min(1.0 - 1e-6, max(1e-6, m))

        # Theoretical max variance for this mean
        max_v = m * (1.0 - m)

        # SCHEME B FIX: Clamping logic
        # If variance is too high for this mean, clamp it to just below max_v
        # instead of falling back to Uniform (which changes the distribution shape completely).
        if v >= max_v:
            # One-time warning per process to avoid log spam
            if not getattr(_sample_eta_one, "_has_warned_variance", False):
                print(f"[WARNING] Sigma {sigma:.3f} is too high for Mu {mu:.3f} in range [{lo},{hi}]. "
                      f"Clamping variance to maintain Beta distribution (preventing fallback to Uniform).")
                _sample_eta_one._has_warned_variance = True

            # Clamp to 99% of max possible variance to ensure valid alpha/beta
            v = max_v * 0.99

        if v <= 1e-12:
            # Zero variance -> Point mass at the (clipped) mean within [lo, hi]
            x = lo + m * width
            return float(min(hi, max(lo, x)))
        else:
            k = m * (1.0 - m) / v - 1.0
            a = max(1e-3, m * k)
            b = max(1e-3, (1.0 - m) * k)
            y = rng.betavariate(a, b)

        return float(lo + y * width)

    if dist == "two_point":
        # two-point on {lo,hi} matching mean (variance is whatever it becomes)
        if abs(hi - lo) <= 1e-12:
            return float(lo)
        p_hi = (float(mu) - lo) / (hi - lo)
        p_hi = min(1.0, max(0.0, p_hi))
        return float(hi if rng.random() < p_hi else lo)

    # default
    return float(rng.uniform(lo, hi))


def mc_evaluate_solution(
        hv3,
        *,
        inst,  # hv3.Instance
        final_ops,  # list[hv3.Operation]
        details: Dict[str, Any],
        mc_samples: int,
        mc_seed: int,
        mc_dist: str,
        mc_corr: float,
        eta_lower: float,
        eta_upper: float,
        mu: float,
        sigma: float,
        mc_R_tol: float = 1e-6,
) -> Dict[str, Any]:
    """
    Evaluate a plan under stochastic time inflation using a *wait-and-sync* model:

    - Ship positions follow the planned event positions (from the SOCP solution).
    - Ship travels at vv, but may WAIT at LAND events if the UAV hasn't arrived.
    - UAV may LOITER if it arrives before the ship; loiter time counts toward airborne time.
    - A sample is marked infeasible if any UAV airborne time exceeds endurance R.

    Returns mc_* metrics to be attached to the raw CSV row.
    """
    t0 = time.perf_counter()
    rng = random.Random(int(mc_seed))

    px = details.get("pos_x", None)
    py = details.get("pos_y", None)
    if px is None or py is None:
        return {
            "mc_time": float("nan"),
            "mc_infeasible_rate": float("nan"),
            "mc_obj_mean": float("nan"),
            "mc_obj_p90": float("nan"),
            "mc_obj_p95": float("nan"),
            "mc_makespan_mean": float("nan"),
            "mc_makespan_p90": float("nan"),
            "mc_makespan_p95": float("nan"),
            "mc_drone_time_mean": float("nan"),
            "mc_drone_time_p90": float("nan"),
            "mc_drone_time_p95": float("nan"),
        }

    px = [float(x) for x in px]
    py = [float(y) for y in py]
    m = len(final_ops)
    if len(px) != m or len(py) != m:
        return {
            "mc_time": float("nan"),
            "mc_infeasible_rate": float("nan"),
            "mc_obj_mean": float("nan"),
            "mc_obj_p90": float("nan"),
            "mc_obj_p95": float("nan"),
            "mc_makespan_mean": float("nan"),
            "mc_makespan_p90": float("nan"),
            "mc_makespan_p95": float("nan"),
            "mc_drone_time_mean": float("nan"),
            "mc_drone_time_p90": float("nan"),
            "mc_drone_time_p95": float("nan"),
        }

    # Precompute task takeoff/landing indices and distances
    tk_idx: Dict[int, int] = {}
    ld_idx: Dict[int, int] = {}
    for i, op in enumerate(final_ops):
        if op.op_type == hv3.Operation.TYPE_TAKEOFF and op.task_id is not None:
            tk_idx[int(op.task_id)] = int(i)
        elif op.op_type == hv3.Operation.TYPE_LAND and op.task_id is not None:
            ld_idx[int(op.task_id)] = int(i)

    tids = sorted(tk_idx.keys())
    vv = float(inst.vv)
    vd = float(inst.vd)
    R = float(inst.R)

    # Endurance feasibility tolerance (absorbs tiny numeric artifacts ~1e-6)
    tol_R = float(mc_R_tol)
    if tol_R < 0.0:
        tol_R = 0.0
    cv = float(inst.cv)
    cd = float(inst.cd)

    # Distances from takeoff/landing positions to tasks
    d_out: Dict[int, float] = {}
    d_in: Dict[int, float] = {}
    for tid in tids:
        itk = tk_idx[tid]
        ild = ld_idx[tid]
        tx, ty = inst.tasks[int(tid)]
        d_out[tid] = _euclid(px[itk], py[itk], float(tx), float(ty))
        d_in[tid] = _euclid(px[ild], py[ild], float(tx), float(ty))

    corr = float(mc_corr)
    corr = min(1.0, max(0.0, corr))

    infeas_cnt = 0
    obj_vals: List[float] = []
    makespans: List[float] = []
    drone_times: List[float] = []

    # main sampling loop
    for s in range(int(mc_samples)):
        # correlated sampling via convex mixing (approx)
        eta_common: Dict[int, float] = {}
        eta_out: Dict[int, float] = {}
        eta_in: Dict[int, float] = {}
        for tid in tids:
            eta_common[tid] = _sample_eta_one(
                rng, dist=mc_dist, eta_lower=eta_lower, eta_upper=eta_upper, mu=mu, sigma=sigma
            )
            eo = _sample_eta_one(rng, dist=mc_dist, eta_lower=eta_lower, eta_upper=eta_upper, mu=mu, sigma=sigma)
            ei = _sample_eta_one(rng, dist=mc_dist, eta_lower=eta_lower, eta_upper=eta_upper, mu=mu, sigma=sigma)
            eta_out[tid] = float(corr * eta_common[tid] + (1.0 - corr) * eo)
            eta_in[tid] = float(corr * eta_common[tid] + (1.0 - corr) * ei)
            # clip
            eta_out[tid] = float(min(eta_upper, max(eta_lower, eta_out[tid])))
            eta_in[tid] = float(min(eta_upper, max(eta_lower, eta_in[tid])))

        # simulate event times forward
        ship_t = 0.0
        takeoff_t: Dict[int, float] = {}
        return_t: Dict[int, float] = {}
        land_t: Dict[int, float] = {}

        infeasible = False

        for k in range(1, m):
            travel = _euclid(px[k - 1], py[k - 1], px[k], py[k]) / max(1e-9, vv)
            arr = ship_t + travel

            op = final_ops[k]
            if op.op_type == hv3.Operation.TYPE_TAKEOFF and op.task_id is not None:
                tid = int(op.task_id)
                ship_t = arr
                takeoff_t[tid] = ship_t
                # travel time (inflated) under nominal vd
                t_flight = (eta_out[tid] * d_out[tid] + eta_in[tid] * d_in[tid]) / max(1e-9, vd)
                return_t[tid] = ship_t + t_flight

            elif op.op_type == hv3.Operation.TYPE_LAND and op.task_id is not None:
                tid = int(op.task_id)
                req = return_t.get(tid, float("inf"))
                ship_t = max(arr, req)
                land_t[tid] = ship_t

                # airborne time includes potential loiter if ship arrives late
                t_air = ship_t - takeoff_t.get(tid, ship_t)
                if t_air > R + tol_R:
                    infeasible = True
                    # Keep sim running? not needed; but keep to compute makespan anyway
            else:
                ship_t = arr

        # compute components
        if infeasible:
            infeas_cnt += 1
        else:
            # total airborne time = sum(landing - takeoff)
            t_drone = 0.0
            for tid in tids:
                t_drone += land_t[tid] - takeoff_t[tid]
            makespan = ship_t
            obj = cv * vv * makespan + cd * vd * t_drone

            obj_vals.append(float(obj))
            makespans.append(float(makespan))
            drone_times.append(float(t_drone))

    t1 = time.perf_counter()

    infeas_rate = float(infeas_cnt) / float(max(1, int(mc_samples)))

    return {
        "mc_time": float(t1 - t0),
        "mc_infeasible_rate": float(infeas_rate),
        "mc_obj_mean": _nanmean(obj_vals),
        "mc_obj_p90": _percentile(obj_vals, 90),
        "mc_obj_p95": _percentile(obj_vals, 95),
        "mc_makespan_mean": _nanmean(makespans),
        "mc_makespan_p90": _percentile(makespans, 90),
        "mc_makespan_p95": _percentile(makespans, 95),
        "mc_drone_time_mean": _nanmean(drone_times),
        "mc_drone_time_p90": _percentile(drone_times, 90),
        "mc_drone_time_p95": _percentile(drone_times, 95),
    }


# -----------------------------------------------------------------------------
# Exact certification: event-level MISOCP (for small n)
# -----------------------------------------------------------------------------

def solve_event_level_misocp(
        *,
        hv3,
        inst_raw: Dict[str, Any],
        dro_cfg: Dict[str, Any],
        time_limit: float,
        rel_gap: float,
        threads: int,
        use_bounds: bool,
) -> Dict[str, Any]:
    """
    Solve the event-level MISOCP (exact SOC, integer event assignment):
      - positions k=1..K (K=2n) each assigned to TK(i) or LD(i)
      - precedence + fleet capacity via prefix constraints
      - exact SOC ship motion and (DRO-speed) UAV flight constraints
      - exact endurance R

    Returns:
      - obj  : best primal objective (UB)
      - bound: best objective bound (LB) from MOSEK MIO
      - gap_rel: (obj - bound)/max(1,|obj|)

    Notes:
    - Uses safe global bounds (same idea as hv3.global_xy_bounds + conservative_time_ub).
    """
    # Guard: require MOSEK
    if not getattr(hv3, "HAS_MOSEK", False):
        return {"status": "NO_MOSEK", "obj": float("inf"), "bound": float("nan"), "gap_rel": float("nan"),
                "time_sec": 0.0}

    # Big-M linking constraints in this event-level MISOCP require finite variable bounds.
    # Even if the continuous oracle is run without bounds (for ablation), certification must stay bounded.
    if not bool(use_bounds):
        if not getattr(solve_event_level_misocp, "_warned_force_bounds", False):
            print("[WARNING] Certification MISOCP requires bounds; forcing use_bounds=True (ignoring --no_mosek_bounds).")
            solve_event_level_misocp._warned_force_bounds = True
        use_bounds = True

    # Import Fusion symbols
    from mosek.fusion import Model, Domain, Expr, ObjectiveSense  # type: ignore

    inst = hv3.Instance.from_dict(inst_raw)
    dro = hv3.DROConfig(**dro_cfg)
    vd_out, vd_in, hat_out, hat_in = dro.speeds(inst.vd)

    n = int(inst.n_tasks)
    K = 2 * n  # event positions 1..K, depots 0 and K+1
    tasks = [int(tid) for tid in sorted(inst.tasks.keys()) if int(tid) != 0]

    # bounds
    xb, yb = hv3.global_xy_bounds(inst, float(vd_out), float(vd_in))
    xL, xU = float(xb[0]), float(xb[1])
    yL, yU = float(yb[0]), float(yb[1])
    T_max = float(hv3.conservative_time_ub(inst, n_events=K + 2, xb=xb, yb=yb))

    t_start = time.perf_counter()

    try:
        with Model("VMURP_Event_MISOCP") as M:
            if int(threads) > 0:
                M.setSolverParam("numThreads", int(threads))
            if time_limit is not None and time_limit > 0:
                # seconds
                M.setSolverParam("mioMaxTime", float(time_limit))
            if rel_gap is not None and rel_gap >= 0:
                M.setSolverParam("mioTolRelGap", float(rel_gap))

            # --- Variables ---
            if use_bounds:
                px = M.variable("px", K + 2, Domain.inRange(xL, xU))
                py = M.variable("py", K + 2, Domain.inRange(yL, yU))
                t = M.variable("t", K + 2, Domain.inRange(0.0, T_max))
            else:
                px = M.variable("px", K + 2)
                py = M.variable("py", K + 2)
                t = M.variable("t", K + 2, Domain.greaterThan(0.0))

            # event assignment: tk[i,k], ld[i,k] for i in tasks, k=1..K
            tk = M.variable("tk", [n, K], Domain.binary())
            ld = M.variable("ld", [n, K], Domain.binary())

            # per-task states
            if use_bounds:
                to = M.variable("to", n, Domain.inRange(0.0, T_max))
                tf = M.variable("tf", n, Domain.inRange(0.0, T_max))
            else:
                to = M.variable("to", n, Domain.greaterThan(0.0))
                tf = M.variable("tf", n, Domain.greaterThan(0.0))

            # service time (not needed for simulation, but needed for SOC leg split)
            if use_bounds:
                Ts = M.variable("Ts", n, Domain.inRange(0.0, T_max))
            else:
                Ts = M.variable("Ts", n, Domain.greaterThan(0.0))

            # task takeoff/landing positions
            if use_bounds:
                pox = M.variable("pox", n, Domain.inRange(xL, xU))
                poy = M.variable("poy", n, Domain.inRange(yL, yU))
                pfx = M.variable("pfx", n, Domain.inRange(xL, xU))
                pfy = M.variable("pfy", n, Domain.inRange(yL, yU))
            else:
                pox = M.variable("pox", n)
                poy = M.variable("poy", n)
                pfx = M.variable("pfx", n)
                pfy = M.variable("pfy", n)

            # --- Depot constraints ---
            M.constraint(px.index(0), Domain.equalsTo(float(inst.port_start[0])))
            M.constraint(py.index(0), Domain.equalsTo(float(inst.port_start[1])))
            M.constraint(t.index(0), Domain.equalsTo(0.0))
            M.constraint(px.index(K + 1), Domain.equalsTo(float(inst.port_end[0])))
            M.constraint(py.index(K + 1), Domain.equalsTo(float(inst.port_end[1])))

            # --- (M1) exactly one event per position k=1..K ---
            for k in range(1, K + 1):
                # sum_i (tk[i,k] + ld[i,k]) == 1
                M.constraint(
                    Expr.add(Expr.sum(tk.slice([0, k - 1], [n, k])), Expr.sum(ld.slice([0, k - 1], [n, k]))),
                    Domain.equalsTo(1.0),
                )

            # --- (M2) each task exactly one tk and one ld ---
            for i in range(n):
                M.constraint(Expr.sum(tk.slice([i, 0], [i + 1, K])), Domain.equalsTo(1.0))
                M.constraint(Expr.sum(ld.slice([i, 0], [i + 1, K])), Domain.equalsTo(1.0))

            # --- (M3) precedence prefix: sum_{h<=k} tk >= sum_{h<=k} ld ---
            for i in range(n):
                for k in range(1, K + 1):
                    M.constraint(
                        Expr.sub(
                            Expr.sum(tk.slice([i, 0], [i + 1, k])),
                            Expr.sum(ld.slice([i, 0], [i + 1, k])),
                        ),
                        Domain.greaterThan(0.0),
                    )

            # --- (M4) fleet capacity: active flights <= L ---
            Lcap = int(inst.L)
            for k in range(1, K + 1):
                M.constraint(
                    Expr.sub(Expr.sum(tk.slice([0, 0], [n, k])), Expr.sum(ld.slice([0, 0], [n, k]))),
                    Domain.lessThan(float(Lcap)),
                )

            # --- time monotonicity ---
            for k in range(0, K + 1):
                M.constraint(Expr.sub(t.index(k + 1), t.index(k)), Domain.greaterThan(0.0))

            # --- ship motion SOC between consecutive event positions ---
            vv = float(inst.vv)
            for k in range(0, K + 1):
                M.constraint(
                    Expr.vstack(
                        Expr.mul(vv, Expr.sub(t.index(k + 1), t.index(k))),
                        Expr.sub(px.index(k + 1), px.index(k)),
                        Expr.sub(py.index(k + 1), py.index(k)),
                    ),
                    Domain.inQCone(),
                )

            # --- linking big-M between tk/ld and per-task states ---
            # With explicit bounds, M_t and M_xy can be chosen tight.
            M_t = float(T_max) + 1.0
            M_x = float(xU - xL) + 1.0
            M_y = float(yU - yL) + 1.0

            # Helper: task index map
            # tasks list is [1..n] if generated; but in general tasks may not be contiguous.
            # We store coordinates in arrays aligned with i=0..n-1 in `tasks`.
            txs = [float(inst.tasks[int(tid)][0]) for tid in tasks]
            tys = [float(inst.tasks[int(tid)][1]) for tid in tasks]

            # For each i,k: if tk[i,k]=1 then (to[i], pox[i],poy[i]) = (t[k], px[k],py[k]) with k in 1..K
            # same for ld -> (tf[i], pfx,pfy)
            for i in range(n):
                for k in range(1, K + 1):
                    b_tk = tk.index(i, k - 1)
                    b_ld = ld.index(i, k - 1)

                    # to == t[k]
                    M.constraint(Expr.sub(Expr.sub(to.index(i), t.index(k)), Expr.mul(M_t, Expr.sub(1.0, b_tk))),
                                 Domain.lessThan(0.0))
                    M.constraint(Expr.sub(Expr.sub(t.index(k), to.index(i)), Expr.mul(M_t, Expr.sub(1.0, b_tk))),
                                 Domain.lessThan(0.0))
                    # pox == px[k], poy == py[k]
                    M.constraint(Expr.sub(Expr.sub(pox.index(i), px.index(k)), Expr.mul(M_x, Expr.sub(1.0, b_tk))),
                                 Domain.lessThan(0.0))
                    M.constraint(Expr.sub(Expr.sub(px.index(k), pox.index(i)), Expr.mul(M_x, Expr.sub(1.0, b_tk))),
                                 Domain.lessThan(0.0))
                    M.constraint(Expr.sub(Expr.sub(poy.index(i), py.index(k)), Expr.mul(M_y, Expr.sub(1.0, b_tk))),
                                 Domain.lessThan(0.0))
                    M.constraint(Expr.sub(Expr.sub(py.index(k), poy.index(i)), Expr.mul(M_y, Expr.sub(1.0, b_tk))),
                                 Domain.lessThan(0.0))

                    # tf == t[k]
                    M.constraint(Expr.sub(Expr.sub(tf.index(i), t.index(k)), Expr.mul(M_t, Expr.sub(1.0, b_ld))),
                                 Domain.lessThan(0.0))
                    M.constraint(Expr.sub(Expr.sub(t.index(k), tf.index(i)), Expr.mul(M_t, Expr.sub(1.0, b_ld))),
                                 Domain.lessThan(0.0))
                    # pfx == px[k], pfy == py[k]
                    M.constraint(Expr.sub(Expr.sub(pfx.index(i), px.index(k)), Expr.mul(M_x, Expr.sub(1.0, b_ld))),
                                 Domain.lessThan(0.0))
                    M.constraint(Expr.sub(Expr.sub(px.index(k), pfx.index(i)), Expr.mul(M_x, Expr.sub(1.0, b_ld))),
                                 Domain.lessThan(0.0))
                    M.constraint(Expr.sub(Expr.sub(pfy.index(i), py.index(k)), Expr.mul(M_y, Expr.sub(1.0, b_ld))),
                                 Domain.lessThan(0.0))
                    M.constraint(Expr.sub(Expr.sub(py.index(k), pfy.index(i)), Expr.mul(M_y, Expr.sub(1.0, b_ld))),
                                 Domain.lessThan(0.0))

            # --- per-task temporal structure and endurance ---
            Rmax = float(inst.R)
            for i in range(n):
                M.constraint(Expr.sub(Ts.index(i), to.index(i)), Domain.greaterThan(0.0))
                M.constraint(Expr.sub(tf.index(i), Ts.index(i)), Domain.greaterThan(0.0))
                M.constraint(Expr.sub(tf.index(i), to.index(i)), Domain.lessThan(Rmax))

            # --- flight SOC constraints (DRO speeds if enabled) ---
            vout = float(vd_out)
            vin = float(vd_in)
            for i in range(n):
                M.constraint(
                    Expr.vstack(
                        Expr.mul(vout, Expr.sub(Ts.index(i), to.index(i))),
                        Expr.sub(pox.index(i), txs[i]),
                        Expr.sub(poy.index(i), tys[i]),
                    ),
                    Domain.inQCone(),
                )
                M.constraint(
                    Expr.vstack(
                        Expr.mul(vin, Expr.sub(tf.index(i), Ts.index(i))),
                        Expr.sub(pfx.index(i), txs[i]),
                        Expr.sub(pfy.index(i), tys[i]),
                    ),
                    Domain.inQCone(),
                )

            # --- objective (same nominal cost form) ---
            t_end = t.index(K + 1)
            drone_sum = Expr.sum(Expr.sub(tf, to))
            obj = Expr.add(
                Expr.mul(float(inst.cv) * float(inst.vv), t_end),
                Expr.mul(float(inst.cd) * float(inst.vd), drone_sum),
            )
            M.objective(ObjectiveSense.Minimize, obj)

            # silence log for paper runner
            M.setLogHandler(None)

            M.solve()

            t_end2 = time.perf_counter()

            # Objective + bound (if available)
            try:
                obj_val = float(M.primalObjValue())
            except Exception:
                obj_val = float("inf")

            try:
                bound = float(M.getSolverDoubleInfo("mioObjBound"))
            except Exception:
                bound = float("nan")

            gap_rel = float("nan")
            if math.isfinite(obj_val) and math.isfinite(bound):
                gap_rel = float((obj_val - bound) / max(1.0, abs(obj_val)))

            # --- extract solution for INDEPENDENT verification (additive; no behaviour change) ---
            sol = {}
            try:
                tk_lv = tk.level(); ld_lv = ld.level()
                ev_order = []
                for k in range(1, K + 1):
                    lab = None
                    for i in range(n):
                        if tk_lv[i * K + (k - 1)] > 0.5:
                            lab = ("TK", int(tasks[i])); break
                        if ld_lv[i * K + (k - 1)] > 0.5:
                            lab = ("LD", int(tasks[i])); break
                    ev_order.append(lab)
                sol = {
                    "event_order": ev_order,
                    "tasks": [int(x) for x in tasks],
                    "to": [float(v) for v in to.level()],
                    "tf": [float(v) for v in tf.level()],
                    "Ts": [float(v) for v in Ts.level()],
                    "px": [float(v) for v in px.level()],
                    "py": [float(v) for v in py.level()],
                    "t": [float(v) for v in t.level()],
                }
            except Exception as _e:
                sol = {"extract_error": f"{type(_e).__name__}: {_e}"}

            return {
                "status": "OK",
                "obj": float(obj_val),
                "bound": float(bound),
                "gap_rel": float(gap_rel),
                "time_sec": float(t_end2 - t_start),
                "hat_out": float(hat_out),
                "hat_in": float(hat_in),
                "vd_out": float(vd_out),
                "vd_in": float(vd_in),
                "solution": sol,
            }

    except Exception as e:
        t_end2 = time.perf_counter()
        return {
            "status": f"EXCEPTION:{type(e).__name__}",
            "obj": float("inf"),
            "bound": float("nan"),
            "gap_rel": float("nan"),
            "time_sec": float(t_end2 - t_start),
            "hat_out": float(hat_out),
            "hat_in": float(hat_in),
            "vd_out": float(vd_out),
            "vd_in": float(vd_in),
        }


# -----------------------------------------------------------------------------
# One-run execution (solve + optional MC + optional cert)
# -----------------------------------------------------------------------------

def _metrics_from_details(hv3, ops: List[Any], details: Dict[str, Any]) -> Tuple[str, float, float]:
    st = str(details.get("status", "")).upper()
    ok = ("OPTIMAL" in st) or ("CACHED" in st) or (st == "OK")
    if not ok:
        return str(details.get("status", "")), float("nan"), float("nan")

    ts = details.get("times", None)
    if ts is None:
        ts = details.get("ts", None)
    if ts is None:
        return "MISSING_TIMES", float("nan"), float("nan")
    if not isinstance(ts, list):
        ts = list(ts)
    if not ts:
        return "MISSING_TIMES", float("nan"), float("nan")

    ship_time = float(ts[-1])
    tk_idx: Dict[int, int] = {}
    ld_idx: Dict[int, int] = {}
    for i, op in enumerate(ops):
        if op.op_type == hv3.Operation.TYPE_TAKEOFF and op.task_id is not None:
            tk_idx[int(op.task_id)] = int(i)
        elif op.op_type == hv3.Operation.TYPE_LAND and op.task_id is not None:
            ld_idx[int(op.task_id)] = int(i)

    drone_time = 0.0
    for tid, itk in tk_idx.items():
        ild = ld_idx.get(tid, None)
        if ild is None:
            continue
        drone_time += float(ts[ild] - ts[itk])
    return "OK", float(ship_time), float(drone_time)


def _solve_fixed_plan_core(
        hv3,
        *,
        inst_raw: Dict[str, Any],
        cfg: SolverCfg,
        run_seed: int,
) -> Dict[str, Any]:
    inst = hv3.Instance.from_dict(inst_raw)

    dro_cfg = hv3.DROConfig(
        enabled=bool(cfg.dro_enabled),
        eta_lower=float(cfg.eta_lower),
        eta_upper=float(cfg.eta_upper),
        mu_plus=float(cfg.mu_plus),
        sigma_plus=float(cfg.sigma_plus),
        eps_out=float(cfg.eps_out),
        eps_in=float(cfg.eps_in),
        eps_sortie=None if cfg.eps_sortie is None else float(cfg.eps_sortie),
    )
    vd_out, vd_in, hat_out, hat_in = dro_cfg.speeds(inst.vd)
    dro_clipped_out = int(abs(hat_out - float(cfg.eta_upper)) <= 1e-9) if cfg.dro_enabled else 0
    dro_clipped_in = int(abs(hat_in - float(cfg.eta_upper)) <= 1e-9) if cfg.dro_enabled else 0

    params = {
        "dro_cfg": dro_cfg.__dict__,
        "base_op_mode": str(cfg.base_op_mode),
        "repair_mode": str(cfg.repair_mode),
        "repair_k": int(cfg.repair_k),
        "destroy_frac": float(cfg.destroy_frac),
        "T_start": float(cfg.T_start),
        "alpha": float(cfg.alpha),
        "max_iter": int(cfg.max_iter),
        "mosek_threads": int(cfg.mosek_threads),
        "refine_policy": str(cfg.refine_policy),
        "refine_steps": int(cfg.refine_steps),
        "refine_evals": int(cfg.refine_evals),
        "refine_trigger_rel": float(cfg.refine_trigger_rel),
        "search_style": str(cfg.search_style),
        "p_best": float(cfg.p_best),
        "R_override": None,
        "cache_size": int(cfg.cache_size),
        "use_mosek_bounds": bool(cfg.use_mosek_bounds),
    }

    t0 = time.perf_counter()
    if int(cfg.palns_workers) <= 1:
        worker_results = [hv3.palns_worker(int(run_seed), inst_raw, params)]
    else:
        ctx = __import__("multiprocessing").get_context("spawn")
        with ctx.Pool(int(cfg.palns_workers)) as pool:
            futs = [
                pool.apply_async(hv3.palns_worker, args=(int(run_seed) + i + int(cfg.seed_offset), inst_raw, params))
                for i in range(int(cfg.palns_workers))
            ]
            worker_results = [f.get() for f in futs]

    best = min(worker_results, key=lambda r: float(r.get("best_obj", float("inf"))))
    palns_obj = float(best.get("best_obj", float("inf")))
    platform_seq = list(best.get("best_platform_seq", []))
    palns_ops = list(best.get("best_ops", []))

    final_obj = palns_obj
    final_ops = palns_ops
    intensify_ran = False

    if str(cfg.mode).lower() == "hybrid" and math.isfinite(palns_obj):
        rep0 = hv3.ops_feasibility_report(palns_ops, inst.L)
        if rep0.get("ok", False) and float(cfg.intensify_time_limit) > 1e-9:
            evaluator = hv3.InnerEvaluator(
                inst,
                vd_out=float(vd_out),
                vd_in=float(vd_in),
                base_op_mode=str(cfg.base_op_mode),
                refine_steps=int(cfg.refine_steps),
                refine_evals=int(cfg.refine_evals),
                mosek_threads=int(cfg.mosek_threads),
                seed=int(run_seed) + 202,
                cache_size=int(cfg.cache_size),
                use_mosek_bounds=bool(cfg.use_mosek_bounds),
            )
            obj2, ops2 = hv3.ops_intensify(
                inst=inst,
                evaluator=evaluator,
                platform_seq=platform_seq,
                base_ops=palns_ops,
                base_obj=palns_obj,
                time_limit=float(cfg.intensify_time_limit),
                assign_trials=int(cfg.assign_trials),
                move_steps=int(cfg.move_steps),
                move_candidates=int(cfg.move_candidates),
                seed=int(run_seed),
            )
            intensify_ran = True
            if math.isfinite(obj2) and obj2 + 1e-9 < final_obj:
                final_obj = float(obj2)
                final_ops = list(ops2)

    time_total = float(time.perf_counter() - t0)

    details: Dict[str, Any] = {}
    ship_time = float("nan")
    drone_time = float("nan")
    status = "OK"
    rep_final = hv3.ops_feasibility_report(final_ops, inst.L)
    if not rep_final.get("ok", False) or (not math.isfinite(final_obj)):
        status = f"INFEASIBLE_OPS:{rep_final.get('reason', '')}"
    else:
        evaluator_det = hv3.InnerEvaluator(
            inst,
            vd_out=float(vd_out),
            vd_in=float(vd_in),
            base_op_mode=str(cfg.base_op_mode),
            refine_steps=0,
            refine_evals=0,
            mosek_threads=int(cfg.mosek_threads),
            seed=int(run_seed) + 999,
            cache_size=0,
            use_mosek_bounds=bool(cfg.use_mosek_bounds),
        )
        details = evaluator_det.solve_socp_mosek(final_ops, need_details=True, assume_feasible=True, use_cache=False)
        status, ship_time, drone_time = _metrics_from_details(hv3, final_ops, details)

    return {
        "inst": inst,
        "dro_cfg": dro_cfg,
        "vd_out": float(vd_out),
        "vd_in": float(vd_in),
        "hat_out": float(hat_out),
        "hat_in": float(hat_in),
        "dro_clipped_out": int(dro_clipped_out),
        "dro_clipped_in": int(dro_clipped_in),
        "platform_seq": platform_seq,
        "palns_obj": float(palns_obj),
        "final_obj": float(final_obj),
        "final_ops": final_ops,
        "intensify_ran": bool(intensify_ran),
        "time_total": float(time_total),
        "ship_time": float(ship_time),
        "drone_time": float(drone_time),
        "details": details,
        "status": str(status),
    }


def _find_internal_zero_airborne_cut(hv3, ops: List[Any], min_completed: int) -> Optional[Tuple[int, List[int]]]:
    active = 0
    landed: List[int] = []
    total_tasks = sum(1 for op in ops if op.op_type == hv3.Operation.TYPE_TAKEOFF and op.task_id is not None)
    for idx, op in enumerate(ops):
        if op.op_type == hv3.Operation.TYPE_TAKEOFF and op.task_id is not None:
            active += 1
        elif op.op_type == hv3.Operation.TYPE_LAND and op.task_id is not None:
            active -= 1
            landed.append(int(op.task_id))
            if active == 0 and len(landed) >= int(min_completed) and len(landed) < total_tasks:
                return int(idx), list(landed)
    return None


def _subset_instance_from_cut(
        inst_raw: Dict[str, Any],
        *,
        remaining_tasks: List[int],
        start_xy: Tuple[float, float],
        suffix: str,
) -> Dict[str, Any]:
    tasks_new = {"0": [0.0, 0.0]}
    for tid in remaining_tasks:
        tasks_new[str(int(tid))] = list(inst_raw["tasks"][str(int(tid))])
    return {
        **inst_raw,
        "instance_id": f"{inst_raw.get('instance_id', 'inst')}_{suffix}",
        "n_tasks": int(len(remaining_tasks)),
        "n": int(len(remaining_tasks)),
        "port_start": [float(start_xy[0]), float(start_xy[1])],
        "tasks": tasks_new,
    }


def _stitch_block_nominal_plan(
        hv3,
        *,
        inst_raw: Dict[str, Any],
        cfg: SolverCfg,
        run_seed: int,
) -> Dict[str, Any]:
    remaining = sorted(int(k) for k in inst_raw["tasks"].keys() if int(k) != 0)
    current_raw = dict(inst_raw)
    cumulative_ops: List[Any] = []
    pos_x_all: List[float] = []
    pos_y_all: List[float] = []
    times_all: List[float] = []
    total_plan_time = 0.0
    replan_time_total = 0.0
    replan_count = 0
    intensify_any = False
    palns_first = float("nan")
    hat_out = float("nan")
    hat_in = float("nan")
    dro_clipped_out = 0
    dro_clipped_in = 0

    stage = 0
    while remaining:
        stage_seed = int(run_seed) + 100003 * stage
        res = _solve_fixed_plan_core(hv3, inst_raw=current_raw, cfg=cfg, run_seed=stage_seed)
        total_plan_time += float(res["time_total"])
        if stage == 0:
            palns_first = float(res["palns_obj"])
            hat_out = float(res["hat_out"])
            hat_in = float(res["hat_in"])
            dro_clipped_out = int(res["dro_clipped_out"])
            dro_clipped_in = int(res["dro_clipped_in"])
        else:
            replan_count += 1
            replan_time_total += float(res["time_total"])
        intensify_any = bool(intensify_any or bool(res["intensify_ran"]))

        if str(res["status"]) != "OK":
            return {
                "status": str(res["status"]),
                "palns_obj": float(palns_first),
                "final_obj": float("inf"),
                "final_ops": [],
                "details": {},
                "ship_time": float("nan"),
                "drone_time": float("nan"),
                "time_total": float(total_plan_time),
                "intensify_ran": bool(intensify_any),
                "hat_out": float(hat_out),
                "hat_in": float(hat_in),
                "dro_clipped_out": int(dro_clipped_out),
                "dro_clipped_in": int(dro_clipped_in),
                "replan_count": float(replan_count),
                "replan_time_total": float(replan_time_total),
            }

        ops = list(res["final_ops"])
        det = res["details"]
        px = [float(x) for x in det["pos_x"]]
        py = [float(y) for y in det["pos_y"]]
        ts = [float(t) for t in det["times"]]

        cut = _find_internal_zero_airborne_cut(hv3, ops, min_completed=int(cfg.block_cut_tasks))
        if cut is None:
            keep_upto = len(ops) - 1
            completed = list(remaining)
            done = True
        else:
            keep_upto, completed = cut
            done = False

        prefix_ops = ops[: keep_upto + 1]
        prefix_px = px[: keep_upto + 1]
        prefix_py = py[: keep_upto + 1]
        prefix_ts = ts[: keep_upto + 1]

        if not cumulative_ops:
            cumulative_ops.extend(prefix_ops)
            pos_x_all.extend(prefix_px)
            pos_y_all.extend(prefix_py)
            times_all.extend(prefix_ts)
        else:
            shift = float(times_all[-1])
            cumulative_ops.extend(prefix_ops[1:])
            pos_x_all.extend(prefix_px[1:])
            pos_y_all.extend(prefix_py[1:])
            times_all.extend([shift + t for t in prefix_ts[1:]])

        if done:
            remaining = []
        else:
            completed_set = set(int(t) for t in completed)
            remaining = [tid for tid in remaining if tid not in completed_set]
            current_raw = _subset_instance_from_cut(
                inst_raw,
                remaining_tasks=remaining,
                start_xy=(float(prefix_px[-1]), float(prefix_py[-1])),
                suffix=f"stage{stage+1}",
            )
        stage += 1

    final_details = {"status": "OK", "pos_x": pos_x_all, "pos_y": pos_y_all, "times": times_all}
    status, ship_time, drone_time = _metrics_from_details(hv3, cumulative_ops, final_details)
    inst0 = hv3.Instance.from_dict(inst_raw)
    final_obj = float("inf")
    if status == "OK":
        final_obj = float(inst0.cv * inst0.vv * ship_time + inst0.cd * inst0.vd * drone_time)
    return {
        "status": str(status),
        "palns_obj": float(palns_first),
        "final_obj": float(final_obj),
        "final_ops": cumulative_ops,
        "details": final_details,
        "ship_time": float(ship_time),
        "drone_time": float(drone_time),
        "time_total": float(total_plan_time),
        "intensify_ran": bool(intensify_any),
        "hat_out": float(hat_out),
        "hat_in": float(hat_in),
        "dro_clipped_out": int(dro_clipped_out),
        "dro_clipped_in": int(dro_clipped_in),
        "replan_count": float(replan_count),
        "replan_time_total": float(replan_time_total),
    }

def solve_one(
        hv3,
        *,
        bundle: Dict[str, Any],
        cfg: SolverCfg,
        run_seed: int,
        mc_enabled: bool,
        mc_samples: int,
        mc_dist: str,
        mc_corr: float,
        cert_enabled: bool,
        cert_time_limit: float,
        cert_rel_gap: float,
        cert_threads: int,
) -> Dict[str, Any]:
    """
    Run one instance under one config + one seed.
    Returns a flat dict row compatible with ExpA/ExpD raw CSV formats.
    """
    inst_raw = bundle["instance"]
    inst = hv3.Instance.from_dict(inst_raw)
    if str(cfg.policy_type).lower() == "rh_block_nominal":
        plan_res = _stitch_block_nominal_plan(hv3, inst_raw=inst_raw, cfg=cfg, run_seed=run_seed)
        dro_cfg = hv3.DROConfig(
            enabled=bool(cfg.dro_enabled),
            eta_lower=float(cfg.eta_lower),
            eta_upper=float(cfg.eta_upper),
            mu_plus=float(cfg.mu_plus),
            sigma_plus=float(cfg.sigma_plus),
            eps_out=float(cfg.eps_out),
            eps_in=float(cfg.eps_in),
            eps_sortie=None if cfg.eps_sortie is None else float(cfg.eps_sortie),
        )
    else:
        plan_res = _solve_fixed_plan_core(hv3, inst_raw=inst_raw, cfg=cfg, run_seed=run_seed)
        dro_cfg = plan_res["dro_cfg"]

    palns_obj = float(plan_res["palns_obj"])
    final_obj = float(plan_res["final_obj"])
    final_ops = list(plan_res["final_ops"])
    intensify_ran = bool(plan_res["intensify_ran"])
    time_total = float(plan_res["time_total"])
    ship_time = float(plan_res["ship_time"])
    drone_time = float(plan_res["drone_time"])
    details: Dict[str, Any] = dict(plan_res["details"])
    status = str(plan_res["status"])
    hat_out = float(plan_res["hat_out"])
    hat_in = float(plan_res["hat_in"])
    dro_clipped_out = int(plan_res["dro_clipped_out"])
    dro_clipped_in = int(plan_res["dro_clipped_in"])
    replan_count = float(plan_res.get("replan_count", float("nan")))
    replan_time_total = float(plan_res.get("replan_time_total", float("nan")))

    # MC evaluation only for subexp "mc" unless user explicitly enables for all
    mc_fields: Dict[str, Any] = {
        "mc_samples": float("nan"),
        "mc_seed": float("nan"),
        "mc_dist": float("nan"),
        "mc_corr": float("nan"),
        "mc_R_tol": float(getattr(args_global, "mc_R_tol", 1e-4)),
        "mc_time": float("nan"),
        "mc_infeasible_rate": float("nan"),
        "mc_obj_mean": float("nan"),
        "mc_obj_p90": float("nan"),
        "mc_obj_p95": float("nan"),
        "mc_makespan_mean": float("nan"),
        "mc_makespan_p90": float("nan"),
        "mc_makespan_p95": float("nan"),
        "mc_drone_time_mean": float("nan"),
        "mc_drone_time_p90": float("nan"),
        "mc_drone_time_p95": float("nan"),
    }

    if mc_enabled and str(cfg.subexp).lower() == "mc" and status == "OK":
        mc_seed_eff = int(run_seed) + 12345
        mc_fields.update(
            {
                "mc_samples": int(mc_samples),
                "mc_seed": int(mc_seed_eff),
                "mc_dist": str(mc_dist),
                "mc_corr": float(mc_corr),
                "mc_R_tol": float(getattr(args_global, "mc_R_tol", 1e-4)),
            }
        )
        mc_fields.update(
            mc_evaluate_solution(
                hv3,
                inst=inst,
                final_ops=final_ops,
                details=details,
                mc_samples=int(mc_samples),
                mc_seed=int(mc_seed_eff),
                mc_dist=str(mc_dist),
                mc_corr=float(mc_corr),
                eta_lower=float(cfg.mc_eta_lower if cfg.mc_eta_lower is not None else cfg.eta_lower),
                eta_upper=float(cfg.mc_eta_upper if cfg.mc_eta_upper is not None else cfg.eta_upper),
                mu=float(cfg.mc_mu_plus if cfg.mc_mu_plus is not None else cfg.mu_plus),
                sigma=float(cfg.mc_sigma_plus if cfg.mc_sigma_plus is not None else cfg.sigma_plus),
            mc_R_tol=float(getattr(args_global, "mc_R_tol", 1e-4)),
            )
        )

    # Certification (small n only, subexp "cert")
    cert_fields: Dict[str, Any] = {
        "cert_time": float("nan"),
        "cert_lb": float("nan"),
        "cert_ub": float("nan"),
        "cert_gap": float("nan"),
        "cert_cuts": float("nan"),
        "heur_gap_to_lb": float("nan"),
    }

    if cert_enabled and str(cfg.subexp).lower() == "cert":
        # only attempt on small enough instances
        if int(inst.n_tasks) <= int(args_global.cert_n_max) and status == "OK":
            # Certification MISOCP uses Big-M link constraints; it requires finite variable bounds.
            # We therefore force bounds ON for certification, independent of continuous-Oracle ablations.
            use_bounds_cert = True
            cert_key = _cert_cache_key(
                inst_raw,
                dro_cfg.__dict__,
                use_bounds=use_bounds_cert,
                time_limit=float(cert_time_limit),
                rel_gap=float(cert_rel_gap),
                threads=int(cert_threads),
            )
            if cert_key in _CERT_CACHE:
                cert_res = _CERT_CACHE[cert_key]
            else:
                cert_res = solve_event_level_misocp(
                    hv3=hv3,
                    inst_raw=inst_raw,
                    dro_cfg=dro_cfg.__dict__,
                    time_limit=float(cert_time_limit),
                    rel_gap=float(cert_rel_gap),
                    threads=int(cert_threads),
                    use_bounds=use_bounds_cert,
                )
                _CERT_CACHE[cert_key] = cert_res
            cert_ub = float(cert_res.get("obj", float("inf")))
            cert_lb = float(cert_res.get("bound", float("nan")))
            cert_gap = float(cert_res.get("gap_rel", float("nan")))
            cert_time = float(cert_res.get("time_sec", float("nan")))

            cert_fields.update(
                {
                    "cert_time": cert_time,
                    "cert_lb": cert_lb,
                    "cert_ub": cert_ub,
                    "cert_gap": cert_gap,
                    "cert_cuts": float("nan"),
                }
            )
            if math.isfinite(cert_lb) and cert_lb > 1e-12 and math.isfinite(final_obj):
                cert_fields["heur_gap_to_lb"] = float((final_obj - cert_lb) / max(1.0, abs(cert_lb)))
        else:
            cert_fields["cert_time"] = 0.0
            cert_fields["cert_lb"] = float("nan")
            cert_fields["cert_ub"] = float("nan")
            cert_fields["cert_gap"] = float("nan")
            cert_fields["cert_cuts"] = float("nan")
            cert_fields["heur_gap_to_lb"] = float("nan")

    layout_meta = _extract_layout_meta(bundle, inst_raw)
    realized_failure_rate = float(mc_fields.get("mc_infeasible_rate", float("nan")))
    realized_cost_inflation = float("nan")
    mc_obj_mean = float(mc_fields.get("mc_obj_mean", float("nan")))
    if math.isfinite(mc_obj_mean) and math.isfinite(final_obj) and abs(final_obj) > 1e-12:
        realized_cost_inflation = float((mc_obj_mean - final_obj) / abs(final_obj))

    row: Dict[str, Any] = {
        "source_file": str(bundle.get("source_file", "")),
        "instance_index": int(bundle.get("instance_index", 0)),
        "instance_id": str(bundle.get("instance_id", inst_raw.get("instance_id", "inst"))),
        "layout_id": layout_meta["layout_id"],
        "layout_variant": layout_meta["layout_variant"],
        "layout_source": layout_meta["layout_source"],
        "subset_rule": layout_meta["subset_rule"],
        "perturbation_rule": layout_meta["perturbation_rule"],
        "config": str(cfg.name),
        "policy_type": str(cfg.policy_type or cfg.name),
        "buffer_rule": str(cfg.buffer_rule),
        "buffer_eta": float("nan") if cfg.buffer_eta is None else float(cfg.buffer_eta),
        "run_seed": int(run_seed),
        "status": str(status),
        "n_tasks": int(inst.n_tasks),
        "L": int(inst.L),
        "mode": str(cfg.mode),
        "time_total": float(time_total),
        "palns_obj": float(palns_obj),
        "final_obj": float(final_obj),
        "ship_time": float(ship_time),
        "drone_time": float(drone_time),
        "intensify_ran": bool(intensify_ran),
        "replan_count": float(replan_count),
        "replan_time_total": float(replan_time_total),
        "realized_failure_rate": realized_failure_rate,
        "realized_cost_inflation": realized_cost_inflation,
        # ExpD bookkeeping fields (harmless for ExpA)
        "subexp": str(cfg.subexp),
        "total_time_budget": float(cfg.total_time_budget),
        "palns_share": float(cfg.palns_share),
        "palns_time_limit": float(cfg.palns_time_limit),
        "intensify_time_limit": float(cfg.intensify_time_limit),
        # DRO meta
        "dro_enabled": int(bool(cfg.dro_enabled)),
        "eps_sortie": float("nan") if cfg.eps_sortie is None else float(cfg.eps_sortie),
        "hat_out": float(hat_out),
        "hat_in": float(hat_in),
        "dro_clipped_out": int(dro_clipped_out),
        "dro_clipped_in": int(dro_clipped_in),
    }
    row.update(mc_fields)
    row.update(cert_fields)
    return row


# -----------------------------------------------------------------------------
# Certification cache (avoid repeating exact MISOCP across run seeds)
# -----------------------------------------------------------------------------
# Note: cert (event-level MISOCP) depends only on the instance + DRO parameters + solver settings.
# In ExpD/cert we typically run multiple heuristic seeds for the *same* instance; without caching,
# we'd redundantly solve the same MISOCP many times (very slow).
_CERT_CACHE: Dict[str, Dict[str, Any]] = {}


def _cert_cache_key(
        inst_raw: Dict[str, Any],
        dro_cfg: Dict[str, Any],
        *,
        use_bounds: bool,
        time_limit: float,
        rel_gap: float,
        threads: int,
) -> str:
    import hashlib
    payload = {
        "instance": inst_raw,
        "dro_cfg": dro_cfg,
        "use_bounds": bool(use_bounds),
        "time_limit": float(time_limit),
        "rel_gap": float(rel_gap),
        "threads": int(threads),
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()


# Global holder for args in solve_one (cert_n_max check)
args_global = None  # set in main()

# -----------------------------------------------------------------------------
# Aggregation / Outputs
# -----------------------------------------------------------------------------


def _extract_layout_meta(bundle: Dict[str, Any], inst_raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "layout_id": str(
            bundle.get("layout_id")
            or inst_raw.get("layout_id")
            or inst_raw.get("instance_id")
            or bundle.get("instance_id", "")
        ),
        "layout_variant": str(
            bundle.get("layout_variant")
            or inst_raw.get("layout_variant")
            or inst_raw.get("layout")
            or inst_raw.get("pattern")
            or ""
        ),
        "layout_source": str(
            bundle.get("layout_source")
            or inst_raw.get("layout_source")
            or bundle.get("source_file", "")
        ),
        "subset_rule": str(bundle.get("subset_rule") or inst_raw.get("subset_rule") or ""),
        "perturbation_rule": str(bundle.get("perturbation_rule") or inst_raw.get("perturbation_rule") or ""),
    }

EXP_A_RAW_FIELDS = [
    "source_file", "instance_index", "instance_id", "layout_id", "layout_variant", "layout_source",
    "subset_rule", "perturbation_rule", "config", "policy_type", "buffer_rule", "buffer_eta",
    "run_seed", "status", "n_tasks", "L", "mode",
    "time_total", "palns_obj", "final_obj", "ship_time", "drone_time", "intensify_ran",
    "replan_count", "replan_time_total", "realized_failure_rate", "realized_cost_inflation",
]

EXP_D_RAW_FIELDS = [
    "source_file", "instance_index", "instance_id", "layout_id", "layout_variant", "layout_source",
    "subset_rule", "perturbation_rule", "config", "policy_type", "buffer_rule", "buffer_eta",
    "run_seed", "status", "n_tasks", "L", "mode",
    "time_total", "palns_obj", "final_obj", "ship_time", "drone_time", "intensify_ran",
    "replan_count", "replan_time_total", "realized_failure_rate", "realized_cost_inflation",
    "subexp", "total_time_budget", "palns_share", "palns_time_limit", "intensify_time_limit",
    "dro_enabled", "eps_sortie", "hat_out", "hat_in", "dro_clipped_out", "dro_clipped_in",
    "mc_samples", "mc_seed", "mc_dist", "mc_corr", "mc_R_tol", "mc_time", "mc_infeasible_rate",
    "mc_obj_mean", "mc_obj_p90", "mc_obj_p95",
    "mc_makespan_mean", "mc_makespan_p90", "mc_makespan_p95",
    "mc_drone_time_mean", "mc_drone_time_p90", "mc_drone_time_p95",
    "cert_time", "cert_lb", "cert_ub", "cert_gap", "cert_cuts", "heur_gap_to_lb",
]


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            w.writerow(out)


def summarize_expA(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # group by (config, n_tasks)
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in raw_rows:
        key = (str(r.get("config", "")), int(r.get("n_tasks", 0)))
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for (cfg, n), rows in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        objs = [float(rr.get("final_obj", float("nan"))) for rr in rows]
        times = [float(rr.get("time_total", float("nan"))) for rr in rows]
        out.append(
            {
                "config": cfg,
                "n_tasks": int(n),
                "count": int(len(rows)),
                "mean_obj": _nanmean(objs),
                "mean_time": _nanmean(times),
            }
        )
    return out


def summarize_expD(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # group by (subexp, config, n_tasks)
    groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for r in raw_rows:
        key = (str(r.get("subexp", "")), str(r.get("config", "")), int(r.get("n_tasks", 0)))
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for (subexp, cfg, n), rows in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        objs = [float(rr.get("final_obj", float("nan"))) for rr in rows]
        times = [float(rr.get("time_total", float("nan"))) for rr in rows]

        mc_ir = [float(rr.get("mc_infeasible_rate", float("nan"))) for rr in rows]
        mc_obj = [float(rr.get("mc_obj_mean", float("nan"))) for rr in rows]
        replan_cnt = [float(rr.get("replan_count", float("nan"))) for rr in rows]
        replan_t = [float(rr.get("replan_time_total", float("nan"))) for rr in rows]
        infl = [float(rr.get("realized_cost_inflation", float("nan"))) for rr in rows]
        cert_gap = [float(rr.get("cert_gap", float("nan"))) for rr in rows]
        heur_gap = [float(rr.get("heur_gap_to_lb", float("nan"))) for rr in rows]

        out.append(
            {
                "subexp": subexp,
                "config": cfg,
                "n_tasks": int(n),
                "count": int(len(rows)),
                "mean_obj": _nanmean(objs),
                "mean_time": _nanmean(times),
                "std_obj": _nanstd(objs),
                "std_time": _nanstd(times),
                "mean_mc_infeasible_rate": _nanmean(mc_ir),
                "std_mc_infeasible_rate": _nanstd(mc_ir),
                "mean_mc_obj_mean": _nanmean(mc_obj),
                "std_mc_obj_mean": _nanstd(mc_obj),
                "mean_replan_count": _nanmean(replan_cnt),
                "mean_replan_time_total": _nanmean(replan_t),
                "mean_realized_cost_inflation": _nanmean(infl),
                "mean_cert_gap": _nanmean(cert_gap),
                "std_cert_gap": _nanstd(cert_gap),
                "mean_heur_gap_to_lb": _nanmean(heur_gap),
                "std_heur_gap_to_lb": _nanstd(heur_gap),
            }
        )
    return out


def derive_expA_metrics(expA_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Produce an ExpA derived_metrics table similar to what you've been using:
      - mean/std of objective and runtime
      - mean ship/drone time
      - utilization proxy: avg_concurrent_uav = mean_drone_time / mean_ship_time
      - improvement vs palns_only
    """
    # group by (config, n_tasks)
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in expA_raw:
        key = (str(r.get("config", "")), int(r.get("n_tasks", 0)))
        groups.setdefault(key, []).append(r)

    # baseline objs per n
    baseline: Dict[int, float] = {}
    for (cfg, n), rows in groups.items():
        if cfg == "palns_only":
            baseline[n] = _nanmean([float(rr.get("final_obj", float("nan"))) for rr in rows])

    out: List[Dict[str, Any]] = []
    for (cfg, n), rows in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        objs = [float(rr.get("final_obj", float("nan"))) for rr in rows]
        times = [float(rr.get("time_total", float("nan"))) for rr in rows]
        ship_ts = [float(rr.get("ship_time", float("nan"))) for rr in rows]
        drone_ts = [float(rr.get("drone_time", float("nan"))) for rr in rows]
        L = int(_safe_float(rows[0].get("L", float("nan")), 0))

        mean_obj = _nanmean(objs)
        mean_time = _nanmean(times)
        mean_ship = _nanmean(ship_ts)
        mean_drone = _nanmean(drone_ts)

        avg_conc = float(mean_drone / mean_ship) if (
                    math.isfinite(mean_drone) and math.isfinite(mean_ship) and mean_ship > 1e-9) else float("nan")
        util_frac = float(avg_conc / max(1, L)) if math.isfinite(avg_conc) else float("nan")

        base_obj = baseline.get(int(n), float("nan"))
        improv_abs = float(base_obj - mean_obj) if (math.isfinite(base_obj) and math.isfinite(mean_obj)) else float(
            "nan")
        improv_pct = float(improv_abs / base_obj * 100.0) if (
                    math.isfinite(base_obj) and base_obj > 1e-12 and math.isfinite(improv_abs)) else float("nan")
        improve_rate = float(sum(
            1 for rr in rows if math.isfinite(base_obj) and float(rr.get("final_obj", float("inf"))) < base_obj) / max(
            1, len(rows))) if math.isfinite(base_obj) else float("nan")

        out.append(
            {
                "config": cfg,
                "n_tasks": int(n),
                "L": int(L),
                "runs": int(len(rows)),
                "mean_obj": mean_obj,
                "std_obj": _nanstd(objs),
                "mean_time": mean_time,
                "std_time": _nanstd(times),
                "mean_ship_time": mean_ship,
                "mean_drone_time": mean_drone,
                # weights are instance-dependent; we store blank placeholders here
                "ship_cost_weight": float("nan"),
                "drone_cost_weight": float("nan"),
                "mean_ship_cost": float("nan"),
                "mean_drone_cost": float("nan"),
                "ship_cost_share": float("nan"),
                "avg_concurrent_uav": avg_conc,
                "util_frac_L": util_frac,
                "mean_obj_per_task": float(mean_obj / n) if (math.isfinite(mean_obj) and n > 0) else float("nan"),
                "mean_ship_time_per_task": float(mean_ship / n) if (math.isfinite(mean_ship) and n > 0) else float(
                    "nan"),
                "mean_drone_time_per_task": float(mean_drone / n) if (math.isfinite(mean_drone) and n > 0) else float(
                    "nan"),
                "improve_rate": improve_rate,
                "mean_improv_pct": improv_pct,
                "mean_improv_abs": improv_abs,
            }
        )
    return out


def maybe_make_plots(outdir: Path, expA_summary: List[Dict[str, Any]]) -> None:
    if not HAS_PLOT:
        return
    if not expA_summary:
        return

    # mean objective vs n
    try:
        df = pd.DataFrame(expA_summary) if HAS_PANDAS else None
    except Exception:
        df = None
    if df is None or df.empty:
        return

    # One plot per metric to keep it simple
    for metric, fname in [("mean_obj", "ExpA_mean_obj.png"), ("mean_time", "ExpA_mean_runtime.png")]:
        plt.figure()
        for cfg in sorted(df["config"].unique()):
            sub = df[df["config"] == cfg].sort_values("n_tasks")
            plt.plot(sub["n_tasks"], sub[metric], marker="o", label=str(cfg))
        plt.xlabel("n_tasks")
        plt.ylabel(metric)
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=160)
        plt.close()


# -----------------------------------------------------------------------------
# Unit tests (light sanity checks)
# -----------------------------------------------------------------------------

def run_unit_tests(hv3, verbose: bool = True) -> None:
    """
    Sanity checks that catch common regressions:
      - ops feasibility report works
      - MOSEK bounds don't change objective "materially" (tolerance-based)
    """
    if not getattr(hv3, "HAS_MOSEK", False):
        if verbose:
            print("[UnitTests] Skipped (MOSEK not available).")
        return

    # Tiny deterministic instance
    inst = generate_synthetic_instance_dict(
        seed=777,
        n_tasks=6,
        L=2,
        pattern="uniform",
        depot_mode="separate",
        scale=50.0,
        vv=20.0,
        cv=85.0,
        vd=90.0,
        cd=5.0,
        R=10.0,
        instance_id="unit_n6_L2",
    )
    I = hv3.Instance.from_dict(inst)

    # baseline DRO speeds (enabled)
    dro_cfg = hv3.DROConfig(enabled=True, eta_lower=1.0, eta_upper=1.2, mu_plus=1.1, sigma_plus=0.05, eps_out=0.05,
                            eps_in=0.05)
    vd_out, vd_in, *_ = dro_cfg.speeds(I.vd)

    eval_b = hv3.InnerEvaluator(I, vd_out=vd_out, vd_in=vd_in, base_op_mode="interleaved", refine_steps=0,
                                refine_evals=0, mosek_threads=1, seed=1, cache_size=0, use_mosek_bounds=True)
    eval_nb = hv3.InnerEvaluator(I, vd_out=vd_out, vd_in=vd_in, base_op_mode="interleaved", refine_steps=0,
                                 refine_evals=0, mosek_threads=1, seed=1, cache_size=0, use_mosek_bounds=False)

    # simple platform sequence
    seq = list(range(1, 7))
    ops = eval_b.build_ops(seq)

    res1 = eval_b.solve_socp_mosek(ops, need_details=False, assume_feasible=True, use_cache=False)
    res2 = eval_nb.solve_socp_mosek(ops, need_details=False, assume_feasible=True, use_cache=False)
    obj1 = float(res1.get("obj", float("inf")))
    obj2 = float(res2.get("obj", float("inf")))

    if not (math.isfinite(obj1) and math.isfinite(obj2)):
        raise AssertionError(f"Unit test failed: infeasible objective obj1={obj1}, obj2={obj2}")

    # Tolerance: 1e-3 abs is safe for objectives ~1e4; relative tol is also used.
    abs_tol = 1e-3
    rel_tol = 1e-7
    if abs(obj1 - obj2) > max(abs_tol, rel_tol * max(1.0, abs(obj2))):
        raise AssertionError(f"Bounds changed objective unexpectedly: bounds={obj1}, no_bounds={obj2}")

    if verbose:
        print("[UnitTests] OK (bounds tolerance check passed).")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    global args_global

    ap = argparse.ArgumentParser(description="Hybrid VMURP v3 paper runner (v3: MC + cert integrated)")

    # IO
    ap.add_argument("--inputs", nargs="*", default=["./instances/"],
                    help="Input JSON folder(s) or files (optional for paper suite)")
    ap.add_argument("--outdir", type=str, default="./vmurp_v3_results", help="Output directory")
    ap.add_argument("--suite", type=str, default="paper", choices=["paper", "quick", "evidence", "custom"],
                    help="Experiment suite")

    # Progress / logging
    ap.add_argument("--quiet", action="store_true", help="Reduce console output")
    ap.add_argument("--heartbeat", type=float, default=30.0,
                    help="Heartbeat interval (seconds) during long solves; 0 disables")

    # Repeats
    ap.add_argument("--runs", type=int, default=3, help="Number of random seeds per instance")

    # PLNS-SA controls (legacy flag names retained for archived scripts)
    ap.add_argument("--palns_workers", "--palns_worker", "--plans_workers", type=int, default=10)
    ap.add_argument("--max_iter", type=int, default=100)
    ap.add_argument("--mosek_threads", type=int, default=1)

    # intensify controls (for hybrid modes)
    ap.add_argument("--intensify_time_limit", type=float, default=40.0)
    ap.add_argument("--assign_trials", type=int, default=40)
    ap.add_argument("--move_steps", type=int, default=120)
    ap.add_argument("--move_candidates", type=int, default=10)

    # refine
    ap.add_argument("--refine_policy", type=str, default="best", choices=["never", "best", "always"])
    ap.add_argument("--refine_steps", type=int, default=30)
    ap.add_argument("--refine_evals", type=int, default=1)

    # speed
    ap.add_argument("--cache_size", type=int, default=5000)
    ap.add_argument("--no_mosek_bounds", action="store_true")

    # DRO params (match hv3)
    ap.add_argument("--no_dro", action="store_true")
    ap.add_argument("--eta_lower", type=float, default=1.0)
    ap.add_argument("--eta_upper", type=float, default=1.2)
    ap.add_argument("--mu_plus", type=float, default=1.1)
    ap.add_argument("--sigma_plus", type=float, default=0.2)
    ap.add_argument("--eps_out", type=float, default=0.05)
    ap.add_argument("--eps_in", type=float, default=0.05)
    ap.add_argument("--eps_sortie", type=float, default=None)
    ap.add_argument("--buffer_eta", type=float, default=None,
                    help="Optional global multiplicative speed buffer for a fixed-buffer planning baseline.")
    ap.add_argument("--enable_block_replan", action="store_true",
                    help="Add an auxiliary staged nominal block-replanning baseline in ExpD/mc.")
    ap.add_argument("--block_cut_tasks", type=int, default=1,
                    help="Replan after the first zero-airborne synchronization cut with at least this many completed tasks.")

    # MC controls (ExpD/mc)
    ap.add_argument("--no_mc", action="store_true", help="Disable MC evaluation (ExpD/mc); mc_* columns stay NaN")
    ap.add_argument("--mc_samples", type=int, default=300)
    ap.add_argument("--mc_dist", type=str, default="beta", choices=["beta", "truncnorm", "uniform", "two_point"])
    ap.add_argument("--mc_corr", type=float, default=0.0)
    ap.add_argument("--mc_R_tol", type=float, default=1e-4,
                    help="Absolute tolerance for endurance infeasibility in MC: mark infeasible only if t_air > R + mc_R_tol. Recommended: 1e-4 ~ 1e-6; set 0 for strict.")

    # Certification controls (ExpD/cert)
    ap.add_argument("--no_cert", action="store_true", help="Disable exact certification even in paper/evidence suites")
    ap.add_argument("--cert_enable", action="store_true",
                    help="Enable cert MISOCP solve for ExpD/cert (default ON in paper suite)")
    ap.add_argument("--cert_n_max", type=int, default=12, help="Only run cert when n_tasks <= this threshold")
    ap.add_argument("--cert_time_limit", type=float, default=120.0)
    ap.add_argument("--cert_rel_gap", type=float, default=0.0)
    ap.add_argument("--cert_threads", type=int, default=1)

    # Plotting
    ap.add_argument("--plots", "--plot", action="store_true", help="Write summary plots (ExpA)")
    ap.add_argument("--mc_only", action="store_true",
                    help="For ExpD-style suites, keep only the MC evidence configurations.")

    # Synthetic generation knobs (paper defaults)
    ap.add_argument("--gen_if_empty", action="store_true",
                    help="Generate synthetic instances if inputs are empty (default ON for paper)")
    ap.add_argument("--pattern", type=str, default="uniform", choices=["uniform", "cluster", "corridor", "ring"])
    ap.add_argument("--depot_mode", type=str, default="separate", choices=["separate", "line"])
    ap.add_argument("--scale", type=float, default=100.0)

    # Instance parameter defaults (used for synthetic generation)
    ap.add_argument("--vv", type=float, default=20.0)
    ap.add_argument("--cv", type=float, default=85.0)
    ap.add_argument("--vd", type=float, default=90.0)
    ap.add_argument("--cd", type=float, default=5.0)
    ap.add_argument("--R_inst", type=float, default=10.0)

    # How many instances per block (paper)
    ap.add_argument("--n_instances_A", type=int, default=10, help="Instances per n in ExpA")
    ap.add_argument("--n_instances_D40", type=int, default=5, help="Instances in ExpD 40-task blocks")
    ap.add_argument("--n_instances_D12", type=int, default=5, help="Instances in ExpD 12-task cert block")

    # Budgets bookkeeping
    ap.add_argument("--total_budget_A", type=float, default=100.0)
    ap.add_argument("--palns_share_A", type=float, default=0.6)
    ap.add_argument("--total_budget_D", type=float, default=60.0)
    ap.add_argument("--palns_share_D", type=float, default=0.6)
    ap.add_argument("--total_budget_cert", type=float, default=120.0)

    # Custom suite generation (if needed)
    ap.add_argument("--n_tasks", type=int, default=40)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--n_instances", type=int, default=5)
    ap.add_argument("--gen_seed", type=int, default=1234)

    # Solver script override
    ap.add_argument("--solver_script", type=str, default=None, help="Path to hybrid_vmurp_lbbd_v3.py")

    args = ap.parse_args()
    args_global = args

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Import solver
    hv3 = _import_solver_module(args.solver_script)
    if not getattr(hv3, "HAS_MOSEK", False):
        print("[Fatal] MOSEK not available in solver environment. Install mosek and set license.")
        sys.exit(2)

    # Decide which bundles to run
    bundles_in = load_instance_bundles_from_inputs(args.inputs)
    use_generated = (str(args.suite).lower() in ("paper", "quick", "evidence")) or (
                bool(args.gen_if_empty) and not bundles_in)
    if use_generated:
        bundles = build_generated_bundles_for_suite(args)
    else:
        bundles = bundles_in

    # Unit tests
    run_unit_tests(hv3, verbose=True)

    # Build configs
    expA_cfgs = build_expA_cfgs(args) if str(args.suite).lower() in ("paper", "quick") else []
    expD_cfgs = build_expD_cfgs(args) if str(args.suite).lower() in ("paper", "quick", "evidence") else []
    if bool(args.mc_only):
        expD_cfgs = [c for c in expD_cfgs if str(c.subexp).lower() == "mc"]
    if str(args.suite).lower() == "custom":
        # In custom suite, just run ExpA configs on the provided inputs
        expA_cfgs = build_expA_cfgs(args)
        expD_cfgs = []

    # Split bundles per experiment type
    expA_bundles = [b for b in bundles if str(b.get("_tag", "")).lower() == "expa"] if use_generated else bundles
    expD40_bundles = [b for b in bundles if str(b.get("_tag", "")).lower() == "expd40"]
    expD12_bundles = [b for b in bundles if str(b.get("_tag", "")).lower() == "expd12"]

    # Seeds per run
    base_seed = 10000

    # ----------------------------
    # Run ExpA
    # ----------------------------
    expA_raw: List[Dict[str, Any]] = []
    if expA_cfgs:
        print(f"\n[Runner] ExpA: {len(expA_bundles)} instances x {len(expA_cfgs)} cfgs x {int(args.runs)} runs")
        for b in expA_bundles:
            for cfg in expA_cfgs:
                for r in range(int(args.runs)):
                    run_seed = base_seed + 1000 * r + _stable_mod_hash(1000, b.get("instance_id", ""), cfg.name)
                    row = solve_one(
                        hv3,
                        bundle=b,
                        cfg=cfg,
                        run_seed=run_seed,
                        mc_enabled=False,
                        mc_samples=0,
                        mc_dist=str(args.mc_dist),
                        mc_corr=float(args.mc_corr),
                        cert_enabled=False,
                        cert_time_limit=float(args.cert_time_limit),
                        cert_rel_gap=float(args.cert_rel_gap),
                        cert_threads=int(args.cert_threads),
                    )
                    # keep ExpA raw format (first 15 cols) but we store full dict internally
                    expA_raw.append(row)

        expA_raw_path = outdir / "ExpA_Scalability_raw.csv"
        write_csv(expA_raw_path, expA_raw, EXP_A_RAW_FIELDS)

        expA_summary = summarize_expA(expA_raw)
        expA_sum_path = outdir / "ExpA_Scalability_summary.csv"
        write_csv(expA_sum_path, expA_summary, ["config", "n_tasks", "count", "mean_obj", "mean_time"])

        expA_der = derive_expA_metrics(expA_raw)
        expA_der_path = outdir / "ExpA_Scalability_derived_metrics.csv"
        write_csv(
            expA_der_path,
            expA_der,
            [
                "config", "n_tasks", "L", "runs", "mean_obj", "std_obj", "mean_time", "std_time",
                "mean_ship_time", "mean_drone_time", "ship_cost_weight", "drone_cost_weight",
                "mean_ship_cost", "mean_drone_cost", "ship_cost_share", "avg_concurrent_uav", "util_frac_L",
                "mean_obj_per_task", "mean_ship_time_per_task", "mean_drone_time_per_task",
                "improve_rate", "mean_improv_pct", "mean_improv_abs",
            ],
        )

        if bool(args.plots):
            maybe_make_plots(outdir, expA_summary)

    # ----------------------------
    # Run ExpD (Evidence chain)
    # ----------------------------
    expD_raw: List[Dict[str, Any]] = []
    if expD_cfgs:
        # Determine which bundles are used in each subexp
        d40 = expD40_bundles if expD40_bundles else [b for b in bundles if
                                                     int(b.get("instance", {}).get("n_tasks", 0)) == 40]
        d12 = expD12_bundles if expD12_bundles else [b for b in bundles if
                                                     int(b.get("instance", {}).get("n_tasks", 0)) <= int(
                                                         args.cert_n_max)]

        # default: enable cert in paper/quick unless user explicitly disables by not passing --cert_enable in custom
        cert_on = False if bool(args.no_cert) else (
            True if str(args.suite).lower() in ("paper", "quick", "evidence") else bool(args.cert_enable))

        print(
            f"\n[Runner] ExpD: bundles40={len(d40)} bundles12={len(d12)} | cfgs={len(expD_cfgs)} | runs={int(args.runs)}")

        for cfg in expD_cfgs:
            # choose appropriate instance block
            block = d40 if cfg.subexp in ("fair", "mc") else d12

            # per-subexp runs: match your historical counts if desired
            runs_sub = int(args.runs)
            if cfg.subexp == "fair":
                runs_sub = int(args.runs)  # typically 5 in your CSV
            elif cfg.subexp == "mc":
                runs_sub = int(args.runs)  # typically 3 in your CSV
            elif cfg.subexp == "cert":
                runs_sub = int(args.runs)  # typically 2 in your CSV

            total_cases = int(len(block)) * int(runs_sub)
            done_cases = 0
            _log(
                f"[ExpD][{cfg.subexp}/{cfg.name}] start: instances={len(block)} runs={runs_sub} total={total_cases} mode={cfg.mode} dro={int(cfg.dro_enabled)} mc={(not bool(args.no_mc) and str(cfg.subexp).lower() == 'mc')} cert={(cert_on and str(cfg.subexp).lower() == 'cert')}")
            for b in block:
                inst_id = str(b.get("instance_id") or b.get("instance", {}).get("instance_id", "inst"))
                n_tasks = int(b.get("instance", {}).get("n_tasks", 0) or b.get("instance", {}).get("n", 0) or 0)
                for r in range(runs_sub):
                    done_cases += 1
                    run_seed = base_seed + 2000 * r + _stable_mod_hash(2000, b.get("instance_id", ""), cfg.name,
                                                                       cfg.subexp)
                    label = f"[ExpD][{cfg.subexp}/{cfg.name}] ({done_cases}/{total_cases}) inst={inst_id} n={n_tasks} seed={run_seed}"
                    _log(label + " -> solve_one()")
                    t_wall0 = time.perf_counter()
                    row = _run_with_heartbeat(
                        label,
                        lambda: solve_one(
                            hv3,
                            bundle=b,
                            cfg=cfg,
                            run_seed=run_seed,
                            mc_enabled=(not bool(args.no_mc)),
                            mc_samples=int(args.mc_samples),
                            mc_dist=str(args.mc_dist),
                            mc_corr=float(args.mc_corr),
                            cert_enabled=cert_on,
                            cert_time_limit=float(args.cert_time_limit),
                            cert_rel_gap=float(args.cert_rel_gap),
                            cert_threads=int(args.cert_threads),
                        ),
                        float(args.heartbeat),
                    )
                    t_wall = time.perf_counter() - t_wall0
                    expD_raw.append(row)
                    status = str(row.get("status", ""))
                    msg = f"{label} <- {status} obj={_fmt(row.get('final_obj'), nd=3)} time_total={_fmt(row.get('time_total'), nd=2)}s wall={t_wall:.2f}s"
                    if str(cfg.subexp).lower() == "mc" and status == "OK":
                        msg += f" | mc_infeas={_fmt(row.get('mc_infeasible_rate'), nd=3)} mc_obj_mean={_fmt(row.get('mc_obj_mean'), nd=1)}"
                    if str(cfg.subexp).lower() == "cert" and status == "OK":
                        msg += f" | cert_gap={_fmt(row.get('cert_gap'), nd=4)} cert_lb={_fmt(row.get('cert_lb'), nd=1)}"
                    _log(msg)

        expD_raw_path = outdir / "ExpD_Evidence_raw.csv"
        write_csv(expD_raw_path, expD_raw, EXP_D_RAW_FIELDS)

        expD_summary = summarize_expD(expD_raw)
        expD_sum_path = outdir / "ExpD_Evidence_summary.csv"
        write_csv(
            expD_sum_path,
            expD_summary,
            [
                "subexp", "config", "n_tasks", "count", "mean_obj", "mean_time", "std_obj", "std_time",
                "mean_mc_infeasible_rate", "std_mc_infeasible_rate",
                "mean_mc_obj_mean", "std_mc_obj_mean",
                "mean_replan_count", "mean_replan_time_total", "mean_realized_cost_inflation",
                "mean_cert_gap", "std_cert_gap",
                "mean_heur_gap_to_lb", "std_heur_gap_to_lb",
            ],
        )

    # Manifest
    manifest = {
        "runner": Path(__file__).name,
        "suite": str(args.suite),
        "runs": int(args.runs),
        "palns_workers": int(args.palns_workers),
        "max_iter": int(args.max_iter),
        "mosek_threads": int(args.mosek_threads),
        "mc_samples": int(args.mc_samples),
        "mc_dist": str(args.mc_dist),
        "mc_corr": float(args.mc_corr),
        "mc_R_tol": float(args.mc_R_tol),
        "enable_block_replan": bool(args.enable_block_replan),
        "block_cut_tasks": int(args.block_cut_tasks),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generated": bool(use_generated),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n[Done] Outputs written to:", str(outdir.resolve()))


if __name__ == "__main__":
    main()
