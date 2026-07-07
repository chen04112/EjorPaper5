#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hybrid VMURP v3 (speed-focused) -- FIXED3

Fixes vs your original v3:
1) The pre-refinement PLNS-SA stage won't get stuck at obj=inf:
   - If a candidate ops is SOCP-infeasible under the chosen base_op_mode (often happens with small R),
     we automatically try a fallback ops construction that is much more "serial":
       fallback_ops = "one_uav" (default): assign all tasks to UAV=1, which yields T1 L1 T2 L2 ...
       fallback_ops = "adjacent": force adjacent ops order
   This ensures we can almost always get a first finite solution, then optimize from there.

2) Better MOSEK status reporting:
   - Return both primal solution status AND problem status when available, so "PRIMAL_INFEASIBLE"
     is visible in code (not only in solver log).

3) Keep --debug_mosek to print MOSEK log and traceback.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import struct
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

# MOSEK Fusion
try:
    import mosek  # noqa: F401
    from mosek.fusion import Model, Domain, Expr, ObjectiveSense, SolutionStatus
    HAS_MOSEK = True
except Exception:
    HAS_MOSEK = False


# =============================================================================
# 0) Data structures
# =============================================================================

@dataclass(slots=True)
class Operation:
    op_type: int
    task_id: Optional[int] = None
    uav_id: Optional[int] = None

    TYPE_DEPOT = 0
    TYPE_TAKEOFF = 1
    TYPE_LAND = 2

    def __repr__(self) -> str:
        if self.op_type == self.TYPE_DEPOT:
            return "Depot"
        t = "T" if self.op_type == self.TYPE_TAKEOFF else "L"
        return f"{t}(task={self.task_id}, uav={self.uav_id})"


@dataclass
class DROConfig:
    enabled: bool = True
    eta_lower: float = 1.0
    eta_upper: float = 1.2
    mu_plus: float = 1.1
    sigma_plus: float = 0.05
    eps_out: float = 0.05
    eps_in: float = 0.05
    eps_sortie: Optional[float] = None

    def speeds(self, v_d: float) -> Tuple[float, float, float, float]:
        """Return (vd_out, vd_in, hat_out, hat_in)."""
        if not self.enabled:
            return float(v_d), float(v_d), 1.0, 1.0

        def hat_eta(eps: float) -> float:
            k = math.sqrt(max(1e-12, (1.0 - eps) / max(eps, 1e-12)))
            # clip to the support [eta_lower, eta_upper] exactly as in the paper's
            # eq:eta-hat = min(eta_max, max(eta_min, mu+ + sigma+*sqrt((1-eps)/eps))).
            # The eta_lower floor guards against optimistic (too-fast) effective speeds
            # when mu_plus < eta_lower under non-default risk inputs.
            return min(self.eta_upper, max(self.eta_lower, self.mu_plus + k * self.sigma_plus))

        e_o = self.eps_out if self.eps_sortie is None else float(self.eps_sortie) / 2.0
        e_i = self.eps_in if self.eps_sortie is None else float(self.eps_sortie) / 2.0
        hat_out = hat_eta(e_o)
        hat_in = hat_eta(e_i)
        return float(v_d) / hat_out, float(v_d) / hat_in, float(hat_out), float(hat_in)


@dataclass
class Instance:
    instance_id: str
    n_tasks: int
    L: int
    port_start: Tuple[float, float]
    port_end: Tuple[float, float]
    tasks: Dict[int, Tuple[float, float]]
    vv: float
    cv: float
    vd: float
    cd: float
    R: float

    @classmethod
    def from_dict(cls, data: dict) -> "Instance":
        p = data["params"]
        return cls(
            instance_id=str(data.get("instance_id", "inst")),
            n_tasks=int(data["n_tasks"]),
            L=int(data["L"]),
            port_start=tuple(data["port_start"]),
            port_end=tuple(data["port_end"]),
            tasks={int(k): (float(v[0]), float(v[1])) for k, v in data["tasks"].items()},
            vv=float(p["vv"]),
            cv=float(p["cv"]),
            vd=float(p["vd"]),
            cd=float(p["cd"]),
            R=float(p["R"]),
        )

    def __post_init__(self) -> None:
        self._task_ids = sorted(int(k) for k in self.tasks.keys() if int(k) != 0)
        nodes = self._task_ids + [-1, -2]
        self._id_to_idx = {nid: i for i, nid in enumerate(nodes)}
        coords = [
            self.tasks[nid] if nid >= 0 else (self.port_start if nid == -1 else self.port_end)
            for nid in nodes
        ]
        n = len(nodes)
        self._dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            xi, yi = coords[i]
            for j in range(n):
                xj, yj = coords[j]
                self._dist[i][j] = math.hypot(xi - xj, yi - yj)

    def dist(self, a: int, b: int) -> float:
        return float(self._dist[self._id_to_idx[a]][self._id_to_idx[b]])


# =============================================================================
# 1) Safe global bounds (for MOSEK variables)
# =============================================================================

def global_xy_bounds(inst: Instance, vd_out: float, vd_in: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    R = float(inst.R)
    xs: List[float] = [float(inst.port_start[0]), float(inst.port_end[0])]
    ys: List[float] = [float(inst.port_start[1]), float(inst.port_end[1])]
    for tid in inst._task_ids:
        tx, ty = inst.tasks[int(tid)]
        xs.extend([tx - vd_out * R, tx + vd_out * R, tx - vd_in * R, tx + vd_in * R])
        ys.extend([ty - vd_out * R, ty + vd_out * R, ty - vd_in * R, ty + vd_in * R])
    return (min(xs), max(xs)), (min(ys), max(ys))


def conservative_time_ub(inst: Instance, n_events: int, xb: Tuple[float, float], yb: Tuple[float, float]) -> float:
    vv = float(inst.vv)
    vv = vv if (math.isfinite(vv) and vv > 1e-9) else 1.0
    dx = float(xb[1] - xb[0])
    dy = float(yb[1] - yb[0])
    diag = math.hypot(dx, dy)
    ship_ub = float(max(1, n_events - 1)) * diag / vv
    slack = float(inst.n_tasks) * float(max(inst.R, 1.0))  # make it safer
    return float(ship_ub + slack + 50.0)


# =============================================================================
# 2) Feasibility checks for ops (discrete)
# =============================================================================

def ops_feasibility_report(ops: List[Operation], L: int) -> Dict[str, Any]:
    rep: Dict[str, Any] = {
        "ok": True,
        "reason": "",
        "max_active": 0,
        "active_end": 0,
        "n_takeoff": 0,
        "n_land": 0,
    }
    if not ops or ops[0].op_type != Operation.TYPE_DEPOT or ops[-1].op_type != Operation.TYPE_DEPOT:
        rep["ok"] = False
        rep["reason"] = "ops must start and end with DEPOT"
        return rep

    tk_pos: Dict[int, int] = {}
    ld_pos: Dict[int, int] = {}
    active = 0
    max_active = 0
    uav_state: Dict[int, Optional[int]] = {}

    for idx, op in enumerate(ops):
        if op.op_type == Operation.TYPE_DEPOT:
            continue

        if op.op_type == Operation.TYPE_TAKEOFF:
            rep["n_takeoff"] += 1
            if op.task_id is None or op.uav_id is None:
                rep["ok"] = False
                rep["reason"] = "TAKEOFF missing task_id/uav_id"
                return rep
            if op.task_id in tk_pos:
                rep["ok"] = False
                rep["reason"] = f"duplicate TAKEOFF for task {op.task_id}"
                return rep
            tk_pos[op.task_id] = idx

            if op.uav_id not in uav_state:
                uav_state[op.uav_id] = None
            if uav_state[op.uav_id] is not None:
                rep["ok"] = False
                rep["reason"] = f"UAV {op.uav_id} takes off while still active (task {uav_state[op.uav_id]})"
                return rep
            uav_state[op.uav_id] = op.task_id

            active += 1
            max_active = max(max_active, active)
            if active > L:
                rep["ok"] = False
                rep["reason"] = f"fleet capacity exceeded: active={active} > L={L}"
                rep["max_active"] = max_active
                rep["active_end"] = active
                return rep

        elif op.op_type == Operation.TYPE_LAND:
            rep["n_land"] += 1
            if op.task_id is None or op.uav_id is None:
                rep["ok"] = False
                rep["reason"] = "LAND missing task_id/uav_id"
                return rep
            if op.task_id in ld_pos:
                rep["ok"] = False
                rep["reason"] = f"duplicate LAND for task {op.task_id}"
                return rep
            ld_pos[op.task_id] = idx

            if op.uav_id not in uav_state:
                uav_state[op.uav_id] = None
            if uav_state[op.uav_id] != op.task_id:
                rep["ok"] = False
                rep["reason"] = f"UAV {op.uav_id} lands task {op.task_id}, but active is {uav_state[op.uav_id]}"
                return rep
            uav_state[op.uav_id] = None

            active -= 1
            if active < 0:
                rep["ok"] = False
                rep["reason"] = "active flights became negative (LAND before TAKEOFF)"
                rep["max_active"] = max_active
                rep["active_end"] = active
                return rep
        else:
            rep["ok"] = False
            rep["reason"] = f"unknown op_type={op.op_type}"
            return rep

    for tid, tk_i in tk_pos.items():
        if tid not in ld_pos:
            rep["ok"] = False
            rep["reason"] = f"task {tid} has TAKEOFF but no LAND"
            return rep
        if tk_i >= ld_pos[tid]:
            rep["ok"] = False
            rep["reason"] = f"task {tid} violates precedence (TAKEOFF at {tk_i} >= LAND at {ld_pos[tid]})"
            return rep
    for tid in ld_pos.keys():
        if tid not in tk_pos:
            rep["ok"] = False
            rep["reason"] = f"task {tid} has LAND but no TAKEOFF"
            return rep

    if active != 0:
        rep["ok"] = False
        rep["reason"] = f"nonzero active flights at end: {active}"
        rep["max_active"] = max_active
        rep["active_end"] = active
        return rep

    rep["max_active"] = max_active
    rep["active_end"] = active
    return rep


def is_ops_feasible(ops: List[Operation], L: int) -> bool:
    return bool(ops_feasibility_report(ops, L)["ok"])


# =============================================================================
# 3) LRU cache
# =============================================================================

class LRUObjCache:
    def __init__(self, max_size: int = 5000):
        self.max_size = int(max(0, max_size))
        self._od: "OrderedDict[bytes, float]" = OrderedDict()

    @staticmethod
    def ops_key(ops: List[Operation]) -> bytes:
        buf = bytearray()
        for o in ops:
            t = int(o.op_type) & 0xFF
            u = int(o.uav_id or 0) & 0xFF
            tid = int(o.task_id or 0)
            if tid < 0:
                tid = 0
            if tid > 65535:
                tid = tid % 65536
            buf.extend(struct.pack("<BBH", t, u, tid))
        return bytes(buf)

    def get(self, key: bytes) -> Optional[float]:
        if self.max_size <= 0:
            return None
        v = self._od.get(key, None)
        if v is not None:
            self._od.move_to_end(key)
        return v

    def put(self, key: bytes, obj: float) -> None:
        if self.max_size <= 0:
            return
        self._od[key] = float(obj)
        self._od.move_to_end(key)
        if len(self._od) > self.max_size:
            self._od.popitem(last=False)


# =============================================================================
# 4) Evaluator
# =============================================================================

class InnerEvaluator:
    def __init__(
        self,
        inst: Instance,
        vd_out: float,
        vd_in: float,
        base_op_mode: str = "interleaved",
        refine_steps: int = 30,
        refine_evals: int = 1,
        mosek_threads: int = 1,
        seed: int = 42,
        cache_size: int = 5000,
        use_mosek_bounds: bool = True,
        debug_mosek: bool = False,
    ):
        self.inst = inst
        self.vd_out = float(vd_out)
        self.vd_in = float(vd_in)
        self.base_op_mode = str(base_op_mode).lower()
        self.refine_steps = int(refine_steps)
        self.refine_evals = int(refine_evals)
        self.mosek_threads = int(mosek_threads)
        self._rng = random.Random(int(seed))

        self.cache = LRUObjCache(max_size=int(cache_size))
        self.use_mosek_bounds = bool(use_mosek_bounds)
        self._xb, self._yb = global_xy_bounds(inst, self.vd_out, self.vd_in)
        self.debug_mosek = bool(debug_mosek)

    def build_ops(
        self,
        platform_seq: List[int],
        assigns: Optional[Dict[int, int]] = None,
        mode_override: Optional[str] = None,
    ) -> List[Operation]:
        if assigns is None:
            assigns = {tid: (i % self.inst.L) + 1 for i, tid in enumerate(platform_seq)}

        mode = self.base_op_mode if mode_override is None else str(mode_override).lower()

        ops: List[Operation] = [Operation(Operation.TYPE_DEPOT)]

        if mode == "interleaved":
            active: Dict[int, Optional[int]] = {u: None for u in range(1, self.inst.L + 1)}
            for tid in platform_seq:
                u = int(assigns[tid])
                if active[u] is not None:
                    ops.append(Operation(Operation.TYPE_LAND, int(active[u]), u))
                ops.append(Operation(Operation.TYPE_TAKEOFF, int(tid), u))
                active[u] = int(tid)
            for u, tid in active.items():
                if tid is not None:
                    ops.append(Operation(Operation.TYPE_LAND, int(tid), int(u)))
        else:  # adjacent
            for tid in platform_seq:
                u = int(assigns[tid])
                ops.append(Operation(Operation.TYPE_TAKEOFF, int(tid), u))
                ops.append(Operation(Operation.TYPE_LAND, int(tid), u))

        ops.append(Operation(Operation.TYPE_DEPOT))
        return ops

    def _mosek_problem_status_str(self, M: Model) -> str:
        # Fusion API has slight differences by version, so be defensive.
        try:
            ps = M.getProblemStatus()
            return f"{ps}"
        except Exception:
            try:
                from mosek.fusion import SolutionType  # type: ignore
                ps = M.getProblemStatus(SolutionType.Default)
                return f"{ps}"
            except Exception:
                return "UNKNOWN_PROBLEM_STATUS"

    def solve_socp_mosek(
        self,
        ops: List[Operation],
        need_details: bool = False,
        assume_feasible: bool = False,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        if (not assume_feasible) and (not is_ops_feasible(ops, self.inst.L)):
            return {"obj": float("inf"), "status": "INFEASIBLE_OPS"}

        key: Optional[bytes] = None
        if use_cache and (not need_details):
            key = self.cache.ops_key(ops)
            cached = self.cache.get(key)
            if cached is not None:
                return {"obj": float(cached), "status": "CACHED"}

        n = len(ops)
        u_tk: Dict[int, int] = {}
        u_ld: Dict[int, int] = {}
        for idx, o in enumerate(ops):
            if o.op_type == Operation.TYPE_TAKEOFF and o.task_id is not None:
                u_tk[int(o.task_id)] = int(idx)
            elif o.op_type == Operation.TYPE_LAND and o.task_id is not None:
                u_ld[int(o.task_id)] = int(idx)
        tids = list(u_tk.keys())

        try:
            with Model("VMURP") as M:
                if self.mosek_threads > 0:
                    M.setSolverParam("numThreads", self.mosek_threads)

                if self.debug_mosek:
                    import sys
                    M.setLogHandler(sys.stdout)
                else:
                    M.setLogHandler(None)

                if self.use_mosek_bounds:
                    xL, xU = float(self._xb[0]), float(self._xb[1])
                    yL, yU = float(self._yb[0]), float(self._yb[1])
                    T_max = conservative_time_ub(self.inst, n_events=n, xb=self._xb, yb=self._yb)
                    px = M.variable(n, Domain.inRange(xL, xU))
                    py = M.variable(n, Domain.inRange(yL, yU))
                    t = M.variable(n, Domain.inRange(0.0, float(T_max)))
                    T_serv = M.variable(len(tids), Domain.inRange(0.0, float(T_max)))
                else:
                    px = M.variable(n)
                    py = M.variable(n)
                    t = M.variable(n, Domain.greaterThan(0.0))
                    T_serv = M.variable(len(tids), Domain.greaterThan(0.0))

                # depots
                M.constraint(px.index(0), Domain.equalsTo(self.inst.port_start[0]))
                M.constraint(py.index(0), Domain.equalsTo(self.inst.port_start[1]))
                M.constraint(t.index(0), Domain.equalsTo(0.0))
                M.constraint(px.index(n - 1), Domain.equalsTo(self.inst.port_end[0]))
                M.constraint(py.index(n - 1), Domain.equalsTo(self.inst.port_end[1]))

                # ship cones
                vv = float(self.inst.vv)
                for i in range(n - 1):
                    M.constraint(
                        Expr.vstack(
                            Expr.mul(vv, Expr.sub(t.index(i + 1), t.index(i))),
                            Expr.sub(px.index(i + 1), px.index(i)),
                            Expr.sub(py.index(i + 1), py.index(i)),
                        ),
                        Domain.inQCone(),
                    )

                # drone cones + sortie limit
                vd_out = float(self.vd_out)
                vd_in = float(self.vd_in)
                R = float(self.inst.R)
                tasks = self.inst.tasks
                for j, tid in enumerate(tids):
                    itk = u_tk.get(tid, None)
                    ild = u_ld.get(tid, None)
                    if itk is None or ild is None:
                        return {"obj": float("inf"), "status": "INCOMPLETE_OPS"}
                    tx, ty = tasks[int(tid)]

                    M.constraint(
                        Expr.vstack(
                            Expr.mul(vd_out, Expr.sub(T_serv.index(j), t.index(itk))),
                            Expr.sub(px.index(itk), tx),
                            Expr.sub(py.index(itk), ty),
                        ),
                        Domain.inQCone(),
                    )
                    M.constraint(
                        Expr.vstack(
                            Expr.mul(vd_in, Expr.sub(t.index(ild), T_serv.index(j))),
                            Expr.sub(px.index(ild), tx),
                            Expr.sub(py.index(ild), ty),
                        ),
                        Domain.inQCone(),
                    )
                    M.constraint(Expr.sub(t.index(ild), t.index(itk)), Domain.lessThan(R))

                # objective
                t_end = t.index(n - 1)
                ld_idx = [u_ld[tid] for tid in tids]
                tk_idx = [u_tk[tid] for tid in tids]
                drone_times = Expr.sum(Expr.sub(t.pick(ld_idx), t.pick(tk_idx)))
                obj = Expr.add(
                    Expr.mul(self.inst.cv * self.inst.vv, t_end),
                    Expr.mul(self.inst.cd * self.inst.vd, drone_times),
                )
                M.objective(ObjectiveSense.Minimize, obj)

                M.solve()

                solsta = M.getPrimalSolutionStatus()
                probsta = self._mosek_problem_status_str(M)

                near_opt = getattr(SolutionStatus, "NearOptimal", None)
                ok_statuses = {SolutionStatus.Optimal}
                if near_opt is not None:
                    ok_statuses.add(near_opt)

                if solsta in ok_statuses:
                    val = float(M.primalObjValue())
                    if use_cache and (not need_details) and key is not None and math.isfinite(val):
                        self.cache.put(key, val)
                    res: Dict[str, Any] = {"obj": val, "status": f"{solsta}", "prob_status": probsta}
                    if need_details:
                        res.update({"pos_x": list(px.level()), "pos_y": list(py.level()), "times": list(t.level())})
                    return res

                # infeasible / unknown / etc.
                return {"obj": float("inf"), "status": f"{solsta}", "prob_status": probsta}

        except Exception as e:
            out: Dict[str, Any] = {"obj": float("inf"), "status": f"MOSEK_ERROR({type(e).__name__}): {e}"}
            if self.debug_mosek:
                out["trace"] = traceback.format_exc(limit=12)
            return out

    def refine_ops(self, base_ops: List[Operation], base_obj: float) -> Tuple[float, List[Operation]]:
        best_obj = float(base_obj)
        best_ops = list(base_ops)

        if not math.isfinite(best_obj):
            best_obj = float(self.solve_socp_mosek(best_ops, need_details=False, assume_feasible=False).get("obj", float("inf")))

        for _ in range(max(0, self.refine_steps)):
            improved = False
            for _k in range(max(0, self.refine_evals)):
                cand = list(best_ops)

                ld_pos = [i for i, o in enumerate(cand) if o.op_type == Operation.TYPE_LAND]
                if not ld_pos:
                    break
                land_idx = self._rng.choice(ld_pos)
                land_op = cand.pop(land_idx)

                tk = None
                for j, o in enumerate(cand):
                    if o.op_type == Operation.TYPE_TAKEOFF and o.task_id == land_op.task_id:
                        tk = j
                        break
                if tk is None:
                    continue

                next_tk = None
                for j in range(tk + 1, len(cand)):
                    o = cand[j]
                    if o.op_type == Operation.TYPE_TAKEOFF and o.uav_id == land_op.uav_id:
                        next_tk = j
                        break
                if next_tk is None:
                    next_tk = len(cand) - 1

                lo = tk + 1
                hi = next_tk
                if lo > hi:
                    continue

                new_pos = self._rng.randint(lo, hi)
                cand.insert(new_pos, land_op)

                if not is_ops_feasible(cand, self.inst.L):
                    continue

                res = self.solve_socp_mosek(cand, need_details=False, assume_feasible=True, use_cache=True)
                obj = float(res.get("obj", float("inf")))

                if math.isfinite(obj) and (not math.isfinite(best_obj) or obj + 1e-9 < best_obj):
                    best_obj = obj
                    best_ops = cand
                    improved = True
                    break

            if not improved:
                break

        return best_obj, best_ops


# =============================================================================
# 5) LNS destroy/repair pieces
# =============================================================================

class LNSRepairManager:
    """LNS repair manager used by the PLNS-SA worker."""
    def __init__(self, inst: Instance, rng: random.Random, repair_mode: str = "exact", repair_k: int = 7):
        self.inst = inst
        self.rng = rng
        self.repair_mode = str(repair_mode).lower()
        self.repair_k = int(repair_k)

    def _delta_insert(self, seq: List[int], tid: int, pos: int) -> float:
        prev_node = -1 if pos == 0 else seq[pos - 1]
        next_node = -2 if pos == len(seq) else seq[pos]
        return self.inst.dist(prev_node, tid) + self.inst.dist(tid, next_node) - self.inst.dist(prev_node, next_node)

    def repair(self, partial: List[int], removed: List[int]) -> List[int]:
        curr = list(partial)
        self.rng.shuffle(removed)

        for tid in removed:
            m = len(curr) + 1
            if self.repair_mode == "exact":
                positions = range(m)
            else:
                k = min(max(1, self.repair_k), m)
                positions = self.rng.sample(range(m), k)

            best_pos = 0
            best_delta = float("inf")
            for p in positions:
                d = self._delta_insert(curr, tid, p)
                if d < best_delta:
                    best_delta = d
                    best_pos = p
            curr.insert(best_pos, tid)

        return curr


def _nn_init(inst: Instance, rng: random.Random) -> List[int]:
    unvisited = set(inst._task_ids)
    seq: List[int] = []
    curr = -1
    while unvisited:
        nxt = min(unvisited, key=lambda t: inst.dist(curr, t))
        seq.append(int(nxt))
        unvisited.remove(nxt)
        curr = int(nxt)
    if len(seq) >= 8:
        for _ in range(2):
            i = rng.randrange(len(seq))
            j = rng.randrange(len(seq))
            seq[i], seq[j] = seq[j], seq[i]
    return seq


# =============================================================================
# 6) Assignment helpers (used in PLNS-SA fallback and local refinement)
# =============================================================================

def _round_robin_assign(inst: Instance, platform_seq: List[int]) -> Dict[int, int]:
    return {tid: (i % inst.L) + 1 for i, tid in enumerate(platform_seq)}

def _block_assign(inst: Instance, platform_seq: List[int]) -> Dict[int, int]:
    n = len(platform_seq)
    if inst.L <= 1:
        return {tid: 1 for tid in platform_seq}
    block = math.ceil(n / inst.L)
    assigns: Dict[int, int] = {}
    u = 1
    cnt = 0
    for tid in platform_seq:
        assigns[tid] = u
        cnt += 1
        if cnt >= block and u < inst.L:
            u += 1
            cnt = 0
    return assigns

def _one_uav_assign(platform_seq: List[int]) -> Dict[int, int]:
    return {tid: 1 for tid in platform_seq}


def _assigns_from_ops(ops: List[Operation]) -> Dict[int, int]:
    """Recover the slot assignment actually realized by an OPS (so the (ops, assigns)
    pair stays self-consistent even when a fallback constructor was used)."""
    return {int(o.task_id): int(o.uav_id)
            for o in ops
            if o.op_type == Operation.TYPE_TAKEOFF and o.task_id is not None}


def _eval_seq_with_fallback(
    inst: Instance,
    evaluator: InnerEvaluator,
    platform_seq: List[int],
    fallback_ops: str,
) -> Tuple[float, List[Operation], str]:
    """
    Evaluate a platform sequence with a small set of assignment / mode fallbacks.
    Returns (best_obj, best_ops, tag).
    """
    # 1) try round_robin
    assigns = _round_robin_assign(inst, platform_seq)
    ops = evaluator.build_ops(platform_seq, assigns=assigns)
    res = evaluator.solve_socp_mosek(ops, need_details=False, assume_feasible=True, use_cache=True)
    obj = float(res.get("obj", float("inf")))
    if math.isfinite(obj):
        return obj, ops, "round_robin"

    # 2) try block (often reduces "single-task UAV" long windows)
    assigns = _block_assign(inst, platform_seq)
    ops = evaluator.build_ops(platform_seq, assigns=assigns)
    res = evaluator.solve_socp_mosek(ops, need_details=False, assume_feasible=True, use_cache=True)
    obj = float(res.get("obj", float("inf")))
    if math.isfinite(obj):
        return obj, ops, "block"

    # 3) fallback
    fb = str(fallback_ops).lower()
    if fb == "none":
        return float("inf"), ops, "none"

    if fb == "one_uav":
        assigns = _one_uav_assign(platform_seq)  # forces near-serial in interleaved mode
        ops = evaluator.build_ops(platform_seq, assigns=assigns, mode_override=None)
        res = evaluator.solve_socp_mosek(ops, need_details=False, assume_feasible=True, use_cache=True)
        obj = float(res.get("obj", float("inf")))
        return obj, ops, "one_uav"

    if fb == "adjacent":
        assigns = _one_uav_assign(platform_seq)
        ops = evaluator.build_ops(platform_seq, assigns=assigns, mode_override="adjacent")
        res = evaluator.solve_socp_mosek(ops, need_details=False, assume_feasible=True, use_cache=True)
        obj = float(res.get("obj", float("inf")))
        return obj, ops, "adjacent"

    # unknown fallback -> behave like none
    return float("inf"), ops, "none"


def _mutate_assigns(
    inst: Instance,
    assigns: Dict[int, int],
    rng: random.Random,
    move_frac: float,
) -> Dict[int, int]:
    """Reassign a random subset of tasks to uniform random slots (assignment move)."""
    out = dict(assigns)
    tids = list(out.keys())
    if not tids or inst.L <= 1:
        return out
    q = max(1, int(round(move_frac * len(tids))))
    for tid in rng.sample(tids, min(q, len(tids))):
        out[int(tid)] = rng.randint(1, inst.L)
    return out


def _loiter_move(
    inst: Instance,
    seq: List[int],
    assigns: Dict[int, int],
    rng: random.Random,
) -> Tuple[List[int], Dict[int, int]]:
    """
    Joint (pi, a) structural move targeting the loiter-optimal macro-structure that
    per-task random mutation cannot reach: pick k <= L-1 tasks as LOITERERS, give each
    a private slot (singleton => lands at the very end), move their takeoffs to the
    front of pi; chain the remaining tasks over the leftover slots (slot reuse =>
    selective early recovery). Constructor feasibility is preserved by construction.
    """
    n = len(seq)
    L = int(inst.L)
    if L < 2 or n < 3:
        return list(seq), dict(assigns)

    # global restructure for small instances; for larger ones mix 50/50 between the
    # global restructure (big escape jumps; won at n=40) and the windowed local
    # variant (bounded perturbation; keeps refined incumbents intact).
    if n <= max(8, 2 * L + 2) or rng.random() < 0.5:
        kmax = min(L - 1, n - 1)
        k = rng.randint(1, kmax)
        loiters = rng.sample(list(seq), k)
        loiter_set = set(loiters)
        rest = [t for t in seq if t not in loiter_set]
        new_seq = list(loiters) + rest
        new_assigns: Dict[int, int] = {}
        for i, t in enumerate(loiters):
            new_assigns[int(t)] = i + 1
        m = L - k
        for j, t in enumerate(rest):
            new_assigns[int(t)] = k + 1 + (j % m)
        return new_seq, new_assigns

    # medium/large instance: WINDOWED restructure -- apply the same loiter pattern to a
    # short pi-segment only, so the perturbation stays local (rolling loiter groups)
    # instead of collapsing the whole instance onto one chain.
    w = rng.randint(4, min(10, n))
    start = rng.randrange(0, n - w + 1)
    window = seq[start:start + w]
    k = min(rng.randint(1, L - 1), w - 1)
    loiters = rng.sample(window, k)
    loiter_set = set(loiters)
    rest_w = [t for t in window if t not in loiter_set]
    new_seq = seq[:start] + list(loiters) + rest_w + seq[start + w:]
    new_assigns = dict(assigns)
    for i, t in enumerate(loiters):
        new_assigns[int(t)] = i + 1
    m = L - k
    for j, t in enumerate(rest_w):
        new_assigns[int(t)] = k + 1 + (j % m)
    return new_seq, new_assigns


def _eval_seq_given_assigns(
    inst: Instance,
    evaluator: InnerEvaluator,
    platform_seq: List[int],
    assigns: Dict[int, int],
    fallback_ops: str,
) -> Tuple[float, List[Operation], str, Dict[int, int]]:
    """
    Evaluate a platform sequence under a GIVEN slot assignment (first-class search
    over (pi, a)); if infeasible, fall back through the legacy assignment ladder.
    Returns (obj, ops, tag, assigns_used).
    """
    ops = evaluator.build_ops(platform_seq, assigns=assigns)
    res = evaluator.solve_socp_mosek(ops, need_details=False, assume_feasible=True, use_cache=True)
    obj = float(res.get("obj", float("inf")))
    if math.isfinite(obj):
        return obj, ops, "given", dict(assigns)
    obj, ops, tag = _eval_seq_with_fallback(inst, evaluator, platform_seq, fallback_ops=fallback_ops)
    return obj, ops, tag, _assigns_from_ops(ops)


def palns_worker(seed: int, inst_dict: dict, params: dict) -> Dict[str, Any]:
    """Legacy function name for one PLNS-SA worker."""
    inst = Instance.from_dict(inst_dict)
    if params.get("R_override") is not None:
        inst.R = float(params["R_override"])

    dro = DROConfig(**params["dro_cfg"])
    vd_out, vd_in, hat_out, hat_in = dro.speeds(inst.vd)

    rng = random.Random(int(seed))
    evaluator = InnerEvaluator(
        inst,
        vd_out=vd_out,
        vd_in=vd_in,
        base_op_mode=params["base_op_mode"],
        refine_steps=int(params["refine_steps"]),
        refine_evals=int(params["refine_evals"]),
        mosek_threads=int(params["mosek_threads"]),
        seed=int(seed) + 7,
        cache_size=int(params.get("cache_size", 0)),
        use_mosek_bounds=bool(params.get("use_mosek_bounds", True)),
        debug_mosek=bool(params.get("debug_mosek", False)),
    )
    lns_repair = LNSRepairManager(inst, rng, repair_mode=params["repair_mode"], repair_k=int(params["repair_k"]))

    fallback_ops = str(params.get("fallback_ops", "one_uav"))

    # assignment-search extension: treat the slot assignment `a` as a first-class
    # search variable next to the task order `pi`. Default OFF => legacy behaviour
    # (round-robin assignment inside _eval_seq_with_fallback) with an identical
    # rng draw sequence, so published runs stay reproducible.
    assign_search = bool(params.get("assign_search", False))
    p_assign_move = float(params.get("p_assign_move", 0.5))
    assign_move_frac = float(params.get("assign_move_frac", 0.34))
    p_loiter_move = float(params.get("p_loiter_move", 0.25))

    # init
    curr_seq = _nn_init(inst, rng)
    curr_obj, curr_ops, init_tag = _eval_seq_with_fallback(inst, evaluator, curr_seq, fallback_ops=fallback_ops)
    curr_assigns = _assigns_from_ops(curr_ops) if curr_ops else _round_robin_assign(inst, curr_seq)

    # optional refine on init if feasible
    if math.isfinite(curr_obj):
        curr_obj, curr_ops = evaluator.refine_ops(curr_ops, curr_obj)

    best_seq = list(curr_seq)
    best_ops = list(curr_ops)
    best_obj = float(curr_obj)
    best_assigns = dict(curr_assigns)

    T = float(params["T_start"])
    alpha = float(params["alpha"])
    destroy_frac = float(params["destroy_frac"])
    search_style = str(params["search_style"]).lower()
    p_best = float(params["p_best"])

    refine_policy = str(params["refine_policy"]).lower()
    trigger_rel = float(params["refine_trigger_rel"])

    for _it in range(int(params["max_iter"])):
        if search_style == "best_only":
            base_seq = best_seq
            base_assigns = best_assigns
        elif search_style == "sa_current":
            base_seq = curr_seq
            base_assigns = curr_assigns
        else:
            if rng.random() < p_best:
                base_seq, base_assigns = best_seq, best_assigns
            else:
                base_seq, base_assigns = curr_seq, curr_assigns

        base_seq = list(base_seq)
        n = len(base_seq)
        if n <= 1:
            break

        rem_cnt = max(1, int(round(n * destroy_frac)))
        rem_cnt = min(rem_cnt, n - 1)
        removed = [base_seq.pop(rng.randrange(len(base_seq))) for _ in range(rem_cnt)]

        cand_seq = lns_repair.repair(base_seq, removed)

        if assign_search:
            cand_assigns = dict(base_assigns)
            u_move = rng.random()
            if u_move < p_loiter_move:
                cand_seq, cand_assigns = _loiter_move(inst, cand_seq, cand_assigns, rng)
            elif u_move < p_loiter_move + p_assign_move:
                cand_assigns = _mutate_assigns(inst, cand_assigns, rng, assign_move_frac)
            cand_obj, cand_ops, _tag, cand_assigns = _eval_seq_given_assigns(
                inst, evaluator, cand_seq, cand_assigns, fallback_ops=fallback_ops)
        else:
            cand_obj, cand_ops, _tag = _eval_seq_with_fallback(inst, evaluator, cand_seq, fallback_ops=fallback_ops)
            cand_assigns = _round_robin_assign(inst, cand_seq)

        do_refine = False
        if refine_policy == "always":
            do_refine = True
        elif refine_policy == "best":
            if math.isfinite(cand_obj) and (not math.isfinite(best_obj) or cand_obj <= best_obj * (1.0 + trigger_rel)):
                do_refine = True

        if do_refine and math.isfinite(cand_obj):
            cand_obj, cand_ops = evaluator.refine_ops(cand_ops, cand_obj)

        accept = False
        if math.isfinite(cand_obj):
            if (not math.isfinite(curr_obj)) or (cand_obj < curr_obj):
                accept = True
            else:
                try:
                    prob = math.exp((curr_obj - cand_obj) / max(1e-9, T))
                except OverflowError:
                    prob = 0.0
                if rng.random() < prob:
                    accept = True

        if accept:
            curr_seq, curr_ops, curr_obj = cand_seq, cand_ops, cand_obj
            curr_assigns = cand_assigns
            if (not math.isfinite(best_obj)) or (cand_obj < best_obj):
                best_seq, best_ops, best_obj = list(cand_seq), list(cand_ops), float(cand_obj)
                best_assigns = dict(cand_assigns)

        T *= alpha

    return {
        "best_obj": float(best_obj),
        "best_platform_seq": best_seq,
        "best_ops": best_ops,
        "best_assigns": {int(k): int(v) for k, v in best_assigns.items()},
        "seed": int(seed),
        "vd_out": float(vd_out),
        "vd_in": float(vd_in),
        "hat_out": float(hat_out),
        "hat_in": float(hat_in),
        "init_tag": str(init_tag),
    }


# =============================================================================
# 7) OPS-level local refinement
# =============================================================================

def _random_assign(inst: Instance, platform_seq: List[int], rng: random.Random) -> Dict[int, int]:
    return {tid: rng.randint(1, inst.L) for tid in platform_seq}


def ops_intensify(
    inst: Instance,
    evaluator: InnerEvaluator,
    platform_seq: List[int],
    base_ops: List[Operation],
    base_obj: float,
    time_limit: float,
    assign_trials: int,
    move_steps: int,
    move_candidates: int,
    seed: int,
) -> Tuple[float, List[Operation]]:
    t0 = time.perf_counter()
    rng = random.Random(int(seed) + 999)

    best_obj = float(base_obj)
    best_ops = list(base_ops)

    assigns_list: List[Tuple[str, Dict[int, int], Optional[str]]] = []
    assigns_list.append(("round_robin", _round_robin_assign(inst, platform_seq), None))
    assigns_list.append(("block", _block_assign(inst, platform_seq), None))
    assigns_list.append(("one_uav", _one_uav_assign(platform_seq), None))  # new: feasibility fallback
    for k in range(max(0, int(assign_trials))):
        assigns_list.append((f"rand{k+1}", _random_assign(inst, platform_seq, rng), None))

    for _tag, assigns, mode_override in assigns_list:
        if time.perf_counter() - t0 > time_limit:
            break
        ops = evaluator.build_ops(platform_seq, assigns=assigns, mode_override=mode_override)
        res = evaluator.solve_socp_mosek(ops, need_details=False, assume_feasible=True, use_cache=True)
        obj = float(res.get("obj", float("inf")))
        if math.isfinite(obj) and (not math.isfinite(best_obj) or obj + 1e-9 < best_obj):
            best_obj, best_ops = obj, list(ops)

    if (time.perf_counter() - t0) <= time_limit and math.isfinite(best_obj):
        obj2, ops2 = evaluator.refine_ops(best_ops, best_obj)
        if math.isfinite(obj2) and obj2 + 1e-9 < best_obj:
            best_obj, best_ops = obj2, ops2

    def feasible_range_after_removal(tmp_ops: List[Operation], land_op: Operation) -> Tuple[int, int]:
        tk = None
        for i, o in enumerate(tmp_ops):
            if o.op_type == Operation.TYPE_TAKEOFF and o.task_id == land_op.task_id:
                tk = i
                break
        if tk is None:
            raise ValueError("TAKEOFF not found for LAND task")

        next_tk = None
        for i in range(tk + 1, len(tmp_ops)):
            o = tmp_ops[i]
            if o.op_type == Operation.TYPE_TAKEOFF and o.uav_id == land_op.uav_id:
                next_tk = i
                break
        if next_tk is None:
            next_tk = len(tmp_ops) - 1
        return tk + 1, next_tk

    for _ in range(max(0, int(move_steps))):
        if time.perf_counter() - t0 > time_limit:
            break

        ops_cur = list(best_ops)
        ld_indices = [i for i, o in enumerate(ops_cur) if o.op_type == Operation.TYPE_LAND]
        if not ld_indices:
            break

        land_idx = rng.choice(ld_indices)
        land_op = ops_cur.pop(land_idx)

        try:
            lo, hi = feasible_range_after_removal(ops_cur, land_op)
        except Exception:
            continue

        lo = max(lo, 1)
        hi = min(hi, len(ops_cur) - 1)
        if lo > hi:
            continue

        land_center = min(max(lo, land_idx), hi)

        cand_positions = {lo, hi, land_center}
        for d in (-4, -3, -2, -1, 1, 2, 3, 4):
            cand_positions.add(min(hi, max(lo, land_center + d)))
        while len(cand_positions) < min(max(1, int(move_candidates)), hi - lo + 1):
            cand_positions.add(rng.randint(lo, hi))

        for p in cand_positions:
            if time.perf_counter() - t0 > time_limit:
                break
            cand_ops = list(ops_cur)
            cand_ops.insert(int(p), land_op)

            if not is_ops_feasible(cand_ops, inst.L):
                continue

            res = evaluator.solve_socp_mosek(cand_ops, need_details=False, assume_feasible=True, use_cache=True)
            obj = float(res.get("obj", float("inf")))
            if math.isfinite(obj) and obj + 1e-9 < best_obj:
                best_obj, best_ops = obj, list(cand_ops)
                break

    return best_obj, best_ops


# =============================================================================
# 8) Plotting
# =============================================================================

def plot_solution(inst: Instance, ops: List[Operation], details: Dict[str, Any], title: str) -> None:
    if not HAS_PLOT:
        print("[Plot] matplotlib not available.")
        return
    if not details or ("pos_x" not in details):
        print("[Plot] no details to plot.")
        return

    px, py = details["pos_x"], details["pos_y"]
    plt.figure(figsize=(11, 9))

    for tid, (tx, ty) in inst.tasks.items():
        if tid == 0:
            continue
        plt.scatter(tx, ty, marker="x", s=60)
        plt.text(tx + 0.8, ty + 0.8, f"T{tid}", fontsize=8)

    plt.plot(px, py, marker="o", linewidth=2.0, alpha=0.8, label="Vessel")

    tk_index = {o.task_id: i for i, o in enumerate(ops) if o.op_type == Operation.TYPE_TAKEOFF}
    ld_index = {o.task_id: i for i, o in enumerate(ops) if o.op_type == Operation.TYPE_LAND}
    for tid in tk_index.keys():
        if tid not in ld_index:
            continue
        itk, ild = tk_index[tid], ld_index[tid]
        tx, ty = inst.tasks[int(tid)]
        plt.plot([px[itk], tx, px[ild]], [py[itk], ty, py[ild]], linestyle="--", alpha=0.35)

    plt.scatter(inst.port_start[0], inst.port_start[1], marker="D", s=90, label="DepotStart")
    plt.scatter(inst.port_end[0], inst.port_end[1], marker="D", s=90, label="DepotEnd")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend()
    plt.show()


# =============================================================================
# 9) Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid VMURP v3 FIXED3 (PLNS-SA + OPS-level local refinement)")

    ap.add_argument("--json", required=True, help="Instance JSON file")
    ap.add_argument("--instance_index", type=int, default=0)
    ap.add_argument("--mode", type=str, default="hybrid", choices=["palns", "hybrid"])

    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max_iter", type=int, default=100)
    ap.add_argument("--mosek_threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--base_op_mode", type=str, default="interleaved", choices=["interleaved", "adjacent"])

    ap.add_argument("--repair_mode", type=str, default="exact", choices=["exact", "sampled"])
    ap.add_argument("--repair_k", type=int, default=7)
    ap.add_argument("--destroy_frac", type=float, default=0.30)
    ap.add_argument("--T_start", type=float, default=5000.0)
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--search_style", type=str, default="mix", choices=["mix", "best_only", "sa_current"])
    ap.add_argument("--p_best", type=float, default=0.75)

    ap.add_argument("--refine_policy", type=str, default="best", choices=["never", "best", "always"])
    ap.add_argument("--refine_steps", type=int, default=30)
    ap.add_argument("--refine_evals", type=int, default=1)
    ap.add_argument("--refine_trigger_rel", type=float, default=0.10)

    ap.add_argument("--cache_size", type=int, default=5000)
    ap.add_argument("--use_mosek_bounds", action="store_true")
    ap.add_argument("--no_mosek_bounds", action="store_true")

    # NEW: feasibility fallback
    ap.add_argument(
        "--fallback_ops",
        type=str,
        default="one_uav",
        choices=["none", "one_uav", "adjacent"],
        help="If interleaved/adjacent with standard assignments is infeasible, fallback to a safer ops construction.",
    )

    ap.add_argument("--debug_mosek", action="store_true")

    # DRO params
    ap.add_argument("--no_dro", action="store_true")
    ap.add_argument("--eta_lower", type=float, default=1.0)
    ap.add_argument("--eta_upper", type=float, default=1.2)
    ap.add_argument("--mu_plus", type=float, default=1.1)
    ap.add_argument("--sigma_plus", type=float, default=0.05)
    ap.add_argument("--eps_out", type=float, default=0.05)
    ap.add_argument("--eps_in", type=float, default=0.05)
    ap.add_argument("--eps_sortie", type=float, default=None)

    ap.add_argument("--R", type=float, default=None)

    ap.add_argument("--intensify_time_limit", type=float, default=40.0)
    ap.add_argument("--assign_trials", type=int, default=40)
    ap.add_argument("--move_steps", type=int, default=120)
    ap.add_argument("--move_candidates", type=int, default=10)

    ap.add_argument("--plot", action="store_true")

    args = ap.parse_args()

    if not HAS_MOSEK:
        print("[Fatal] MOSEK Fusion not available. Please install mosek and set license.")
        return

    use_bounds = True
    if bool(args.no_mosek_bounds):
        use_bounds = False
    elif bool(args.use_mosek_bounds):
        use_bounds = True

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)
    inst_raw = data.get("instances", [data])[int(args.instance_index)]
    inst = Instance.from_dict(inst_raw)
    if args.R is not None:
        inst.R = float(args.R)

    dro_cfg = DROConfig(
        enabled=not bool(args.no_dro),
        eta_lower=float(args.eta_lower),
        eta_upper=float(args.eta_upper),
        mu_plus=float(args.mu_plus),
        sigma_plus=float(args.sigma_plus),
        eps_out=float(args.eps_out),
        eps_in=float(args.eps_in),
        eps_sortie=None if args.eps_sortie is None else float(args.eps_sortie),
    )
    vd_out, vd_in, hat_out, hat_in = dro_cfg.speeds(inst.vd)

    print("\n=== Hybrid VMURP v3 (FIXED3) ===")
    print(f"Instance={inst.instance_id} | n={inst.n_tasks} | L={inst.L} | R={inst.R}")
    print(f"Ship(vv={inst.vv}, cv={inst.cv}) | Drone(vd={inst.vd}, cd={inst.cd})")
    print(f"DRO enabled={dro_cfg.enabled} | hat_out={hat_out:.4f} hat_in={hat_in:.4f} => vd_out={vd_out:.4f} vd_in={vd_in:.4f}")
    print(f"Mode={args.mode} | base_op_mode={args.base_op_mode} | workers={args.workers} | mosek_threads={args.mosek_threads}")
    print(f"Speed: cache_size={args.cache_size} | mosek_bounds={use_bounds} | fallback_ops={args.fallback_ops}")
    print("-" * 98)

    params = {
        "dro_cfg": dro_cfg.__dict__,
        "base_op_mode": str(args.base_op_mode),
        "repair_mode": str(args.repair_mode),
        "repair_k": int(args.repair_k),
        "destroy_frac": float(args.destroy_frac),
        "T_start": float(args.T_start),
        "alpha": float(args.alpha),
        "max_iter": int(args.max_iter),
        "mosek_threads": int(args.mosek_threads),
        "refine_policy": str(args.refine_policy),
        "refine_steps": int(args.refine_steps),
        "refine_evals": int(args.refine_evals),
        "refine_trigger_rel": float(args.refine_trigger_rel),
        "search_style": str(args.search_style),
        "p_best": float(args.p_best),
        "R_override": None if args.R is None else float(args.R),
        "cache_size": int(args.cache_size),
        "use_mosek_bounds": bool(use_bounds),
        "debug_mosek": bool(args.debug_mosek),
        "fallback_ops": str(args.fallback_ops),
    }

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(int(args.workers)) as pool:
        futures = [
            pool.apply_async(palns_worker, args=(int(args.seed) + i, inst_raw, params))
            for i in range(int(args.workers))
        ]
        results = [fu.get() for fu in futures]
    t1 = time.perf_counter()

    tag_count: Dict[str, int] = {}
    for r in results:
        tag = str(r.get("init_tag", "NA"))
        tag_count[tag] = tag_count.get(tag, 0) + 1

    best = min(results, key=lambda r: float(r["best_obj"]))
    palns_obj = float(best["best_obj"])
    palns_seq = list(best["best_platform_seq"])
    palns_ops = list(best["best_ops"])

    rep = ops_feasibility_report(palns_ops, inst.L)
    print(f"[PLNS-SA] done in {t1 - t0:.2f}s | best_obj={palns_obj:.6f} | seed={best['seed']}")
    print(f"[PLNS-SA] ops_feasible={rep['ok']} | max_active={rep['max_active']} (L={inst.L}) | nT={rep['n_takeoff']} nL={rep['n_land']}")
    print(f"[PLNS-SA] init_tag_dist={tag_count}")
    if not rep["ok"]:
        print(f"[PLNS-SA][Warn] infeasible ops: {rep['reason']}")
    if not math.isfinite(palns_obj):
        print("[PLNS-SA][Warn] best_obj still inf. Try: --workers 1 --debug_mosek and/or set --fallback_ops one_uav/adjacent.")

    final_obj = palns_obj
    final_ops = palns_ops

    evaluator_main = InnerEvaluator(
        inst,
        vd_out=vd_out,
        vd_in=vd_in,
        base_op_mode=str(args.base_op_mode),
        refine_steps=int(args.refine_steps),
        refine_evals=int(args.refine_evals),
        mosek_threads=int(args.mosek_threads),
        seed=int(args.seed) + 202,
        cache_size=int(args.cache_size),
        use_mosek_bounds=bool(use_bounds),
        debug_mosek=bool(args.debug_mosek),
    )

    if args.mode == "hybrid" and math.isfinite(palns_obj) and rep["ok"]:
        t2 = time.perf_counter()
        obj2, ops2 = ops_intensify(
            inst=inst,
            evaluator=evaluator_main,
            platform_seq=palns_seq,
            base_ops=palns_ops,
            base_obj=palns_obj,
            time_limit=float(args.intensify_time_limit),
            assign_trials=int(args.assign_trials),
            move_steps=int(args.move_steps),
            move_candidates=int(args.move_candidates),
            seed=int(args.seed),
        )
        t3 = time.perf_counter()
        rep2 = ops_feasibility_report(ops2, inst.L)
        print(f"[LocalRefine] done in {t3 - t2:.2f}s | best_obj={obj2:.6f} | improved={obj2 + 1e-9 < palns_obj}")
        print(f"[LocalRefine] ops_feasible={rep2['ok']} | max_active={rep2['max_active']} (L={inst.L})")
        if rep2["ok"] and math.isfinite(obj2) and obj2 + 1e-9 < final_obj:
            final_obj, final_ops = obj2, ops2

    print("\n=== Result ===")
    if math.isfinite(final_obj):
        print(f"obj={final_obj:.6f} | platform_seq_len={len(palns_seq)} | ops_len={len(final_ops)}")
    else:
        print(f"obj=inf | platform_seq_len={len(palns_seq)} | ops_len={len(final_ops)}")
    print(f"platform_seq(head)={palns_seq[:20]}")

    if args.plot and HAS_PLOT and math.isfinite(final_obj) and is_ops_feasible(final_ops, inst.L):
        det = evaluator_main.solve_socp_mosek(final_ops, need_details=True, assume_feasible=True, use_cache=False)
        title = f"{inst.instance_id} | obj={final_obj:.2f} | R={inst.R} | L={inst.L}"
        plot_solution(inst, final_ops, det, title)


if __name__ == "__main__":
    main()
