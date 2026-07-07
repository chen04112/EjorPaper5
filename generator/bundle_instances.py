#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic ship-and-multi-UAV benchmark instance generator.

Output: runs/Model1_Benchmark/bundle_instances.json
Each instance is compatible with the solver's Instance data structure:
    Instance(tasks={i:(x,y)}, port_start=(x,y), port_end=(x,y),
             L=..., vv=..., cv=..., vd=..., cd=..., R=..., seed=...)

Design goals:
  1) Generate instances at several sizes and repetitions (n_tasks / L / layout controllable).
  2) Guarantee "feasible even if the vessel does not move" (conservative feasibility):
     2*dist(port, task) <= vd*R, so the inner SOCP is not vacuously infeasible.
  3) Use a fixed global seed and a per-instance seed convention, so every instance
     is reproducible.

Note: the inner model allows zero turnaround time (a drone may land and take off at the
same point and time); this generator does not model deck-resource conflicts.
"""

from __future__ import annotations

import os
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple, Optional

import numpy as np


# =========================
# 0) JSON encoding helper
# =========================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):  # type: ignore[override]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


# =========================
# 1) Global configuration (edit here first)
# =========================
SEED = 2025

# Output
OUT_FILE = os.path.join("runs", "Model1_Benchmark", "bundle_instances.json")

# How many instances (sizes x repeats).  The default size/fleet pairs mirror the
# scalable configurations reported in the manuscript.
SIZES = [20, 40, 60, 80]          # number of tasks n_tasks
REPEATS_PER_SIZE = 5              # random instances per size
L_BY_SIZE = {20: 2, 40: 3, 60: 4, 80: 5}
L_CHOICES = [2, 3, 4, 5]          # fallback candidates for user-added sizes

# Spatial layout: uniform / clustered / line
#   - uniform:   points spread uniformly over a square area
#   - clustered: K Gaussian clusters
#   - line:      tasks approximately along a noisy line (river/coastline style)
LAYOUT_MODE = "clustered"  # "uniform" | "clustered" | "line"

# Endpoints (depot)
#   - loop=True:  port_end = port_start (return to origin)
#   - loop=False: a different endpoint (closer to a one-way route)
LOOP_TRIP = True
PORT_START = (0.0, 0.0)
PORT_END = (0.0, 0.0) if LOOP_TRIP else (120.0, 0.0)

# Coordinate scale (task scatter range). Distances are Euclidean, so the speeds
# vv/vd must be in units consistent with these coordinates.
AREA_BOX = (-80.0, 80.0, -80.0, 80.0)  # (xmin, xmax, ymin, ymax)

# clustered-layout parameters
N_CLUSTERS_RANGE = (2, 5)
CLUSTER_STD_RANGE = (6.0, 18.0)

# line-layout parameter
LINE_NOISE_STD = 8.0

# Motion/cost parameters (objective uses cv*vv*t_end + cd*vd*flight_time).
VV_RANGE = (6.0, 14.0)      # ship speed
VD_RANGE = (16.0, 28.0)     # UAV speed
CV_RANGE = (18.0, 24.0)     # ship unit-distance cost
CD_RANGE = (0.35, 0.75)     # UAV unit-distance cost

# UAV endurance (maximum airborne duration, in time units)
R_RANGE = (8.0, 14.0)

# Feasibility safety factor: require 2*dist <= vd*R*FEAS_MARGIN.
# A larger margin is more conservatively feasible but keeps tasks closer to the depot.
FEAS_MARGIN = 0.95

# Maximum resample attempts (avoid an infinite loop)
MAX_RESAMPLE = 5000


# =========================
# 2) Data structures
# =========================
@dataclass
class Model1InstanceSpec:
    instance_id: str
    seed: int
    n_tasks: int
    L: int
    layout: str
    layout_source: str
    subset_rule: str
    perturbation_rule: str
    port_start: Tuple[float, float]
    port_end: Tuple[float, float]
    vv: float
    cv: float
    vd: float
    cd: float
    R: float
    tasks: Dict[int, Tuple[float, float]]  # i -> (x,y)


# =========================
# 3) Utility functions
# =========================
def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.hypot(dx, dy)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def sample_params(rng: random.Random) -> Tuple[float, float, float, float, float]:
    vv = rng.uniform(*VV_RANGE)
    vd = rng.uniform(*VD_RANGE)
    cv = rng.uniform(*CV_RANGE)
    cd = rng.uniform(*CD_RANGE)
    R = rng.uniform(*R_RANGE)
    return vv, cv, vd, cd, R


def feasible_task(p: Tuple[float, float], port: Tuple[float, float], vd: float, R: float) -> bool:
    # Conservative feasibility: even if the vessel stays put, the task can be reached
    # and returned from: 2*dist <= vd*R*margin.
    return 2.0 * euclid(p, port) <= float(vd) * float(R) * float(FEAS_MARGIN)


def gen_uniform_points(rng: random.Random, n: int, box: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
    xmin, xmax, ymin, ymax = box
    pts = []
    for _ in range(n):
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        pts.append((x, y))
    return pts


def gen_clustered_points(rng: random.Random, n: int, box: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
    xmin, xmax, ymin, ymax = box
    k = rng.randint(N_CLUSTERS_RANGE[0], N_CLUSTERS_RANGE[1])
    # cluster centers
    centers = [(rng.uniform(xmin * 0.6, xmax * 0.6), rng.uniform(ymin * 0.6, ymax * 0.6)) for _ in range(k)]
    stds = [rng.uniform(CLUSTER_STD_RANGE[0], CLUSTER_STD_RANGE[1]) for _ in range(k)]

    pts: List[Tuple[float, float]] = []
    for i in range(n):
        c = rng.randrange(k)
        cx, cy = centers[c]
        s = stds[c]
        x = rng.gauss(cx, s)
        y = rng.gauss(cy, s)
        # clip into box
        x = clamp(x, xmin, xmax)
        y = clamp(y, ymin, ymax)
        pts.append((x, y))
    return pts


def gen_line_points(rng: random.Random, n: int, box: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
    xmin, xmax, ymin, ymax = box
    # base line from left to right with random slope
    x0, x1 = xmin * 0.8, xmax * 0.8
    y0 = rng.uniform(ymin * 0.4, ymax * 0.4)
    y1 = rng.uniform(ymin * 0.4, ymax * 0.4)

    pts: List[Tuple[float, float]] = []
    for i in range(n):
        t = (i + rng.random()) / max(1, n)  # roughly ordered along line
        x = x0 + (x1 - x0) * t + rng.gauss(0.0, LINE_NOISE_STD)
        y = y0 + (y1 - y0) * t + rng.gauss(0.0, LINE_NOISE_STD)
        x = clamp(x, xmin, xmax)
        y = clamp(y, ymin, ymax)
        pts.append((x, y))
    rng.shuffle(pts)
    return pts


def gen_task_points(rng: random.Random, n: int) -> List[Tuple[float, float]]:
    if LAYOUT_MODE == "uniform":
        return gen_uniform_points(rng, n, AREA_BOX)
    if LAYOUT_MODE == "line":
        return gen_line_points(rng, n, AREA_BOX)
    return gen_clustered_points(rng, n, AREA_BOX)


# =========================
# 4) Step 1: generate a single instance (with feasibility filtering)
# =========================
def step1_generate_one(instance_seed: int, n_tasks: int) -> Model1InstanceSpec:
    rng = random.Random(instance_seed)

    # parameters
    L = L_BY_SIZE.get(n_tasks, rng.choice(L_CHOICES))
    vv, cv, vd, cd, R = sample_params(rng)

    # depot
    port_start = PORT_START
    port_end = PORT_END

    # task points: generate, then feasibility-filter / resample
    tasks: Dict[int, Tuple[float, float]] = {}
    attempts = 0
    i = 1
    while i <= n_tasks:
        attempts += 1
        if attempts > MAX_RESAMPLE:
            raise RuntimeError(
                f"Failed to generate feasible tasks after {MAX_RESAMPLE} attempts. "
                f"Try smaller AREA_BOX or larger vd/R."
            )

        cand = gen_task_points(rng, 1)[0]
        # use port_start as the conservative-feasibility reference (vessel-static feasible)
        if feasible_task(cand, port_start, vd, R):
            tasks[i] = cand
            i += 1

    instance_id = f"Model1_{LAYOUT_MODE}_n{n_tasks}_L{L}_seed{instance_seed}"

    return Model1InstanceSpec(
        instance_id=instance_id,
        seed=instance_seed,
        n_tasks=n_tasks,
        L=L,
        layout=LAYOUT_MODE,
        layout_source="generated",
        subset_rule=f"size_n{n_tasks}",
        perturbation_rule="none",
        port_start=port_start,
        port_end=port_end,
        vv=vv,
        cv=cv,
        vd=vd,
        cd=cd,
        R=R,
        tasks=tasks,
    )


# =========================
# 5) Step 2: generate the full bundle
# =========================
def step2_generate_bundle() -> Dict[str, Any]:
    print("[Step 2] Generating Model1 benchmark instances...")

    instances: List[Dict[str, Any]] = []
    idx = 0

    for n in SIZES:
        for r in range(REPEATS_PER_SIZE):
            idx += 1
            inst_seed = SEED + 10_000 * n + 97 * r
            spec = step1_generate_one(inst_seed, n)

            # JSON-friendly: use str task keys
            tasks_json = {str(k): [float(v[0]), float(v[1])] for k, v in spec.tasks.items()}

            instances.append(
                {
                    "instance_id": spec.instance_id,
                    "seed": spec.seed,
                    "n_tasks": spec.n_tasks,
                    "L": spec.L,
                    "layout": spec.layout,
                    "layout_id": spec.instance_id,
                    "layout_variant": spec.layout,
                    "layout_source": spec.layout_source,
                    "subset_rule": spec.subset_rule,
                    "perturbation_rule": spec.perturbation_rule,
                    "port_start": [float(spec.port_start[0]), float(spec.port_start[1])],
                    "port_end": [float(spec.port_end[0]), float(spec.port_end[1])],
                    "params": {
                        "vv": float(spec.vv),
                        "cv": float(spec.cv),
                        "vd": float(spec.vd),
                        "cd": float(spec.cd),
                        "R": float(spec.R),
                    },
                    "tasks": tasks_json,
                }
            )

            if idx % 5 == 0:
                print(f"  - generated {idx} instances (latest: {spec.instance_id})")

    bundle = {
        "meta": {
            "generator": "bundle_instances.py",
            "seed": SEED,
            "layout_mode": LAYOUT_MODE,
            "loop_trip": LOOP_TRIP,
            "sizes": SIZES,
            "fleet_by_size": L_BY_SIZE,
            "repeats_per_size": REPEATS_PER_SIZE,
            "L_choices": L_CHOICES,
            "area_box": list(AREA_BOX),
            "feas_margin": FEAS_MARGIN,
        },
        "instances": instances,
    }
    print(f"  - total instances: {len(instances)}")
    return bundle


# =========================
# 6) Step 3: save
# =========================
def step3_save(bundle: Dict[str, Any]) -> None:
    ensure_dir(OUT_FILE)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"[Success] Saved: {OUT_FILE}")


# =========================
# 7) Main
# =========================
def main() -> None:
    print("======================================================")
    print(" Model1 Deterministic Benchmark Generator")
    print(f" Seed: {SEED}")
    print(f" Layout: {LAYOUT_MODE}")
    print(f" Sizes: {SIZES}, repeats: {REPEATS_PER_SIZE}")
    print(f" UAV L choices: {L_CHOICES}")
    print(f" Port start: {PORT_START}, port end: {PORT_END}")
    print(f" Area box: {AREA_BOX}")
    print(f" Feasibility: 2*dist(port,task) <= vd*R*{FEAS_MARGIN}")
    print(f" Output: {OUT_FILE}")
    print("======================================================\n")

    bundle = step2_generate_bundle()
    step3_save(bundle)


if __name__ == "__main__":
    main()
