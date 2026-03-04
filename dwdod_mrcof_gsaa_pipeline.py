#!/usr/bin/env python3
"""
Reproducible reference implementation for the paper:
Optimization of Hospital Resource Scheduling Efficiency Based on Dynamic Weighted Distance Anomaly Detection Algorithm

This script implements:
1) DWD-OD (Dynamic Weighted Distance-Based Outlier Detection) with:
   - z-score standardization
   - dynamic weights based on device-type priority alpha and load fluctuation coefficient beta (sliding window)
   - Adaptive Mahalanobis Distance (AMD) using a dynamic diagonal weight matrix
   - sliding-window LOF-like scoring and a dynamic local density threshold

2) A runnable, self-contained MRCOF-style scheduling simulation with GSAA:
   - Stage-1: isolate anomalous tasks
   - Stage-2: optimize deployment of normal tasks onto nodes via GA + Simulated Annealing (Metropolis criterion)

Dataset (uploaded by the user):
/mnt/data/hospital_communication_energy_system.csv

Outputs:
- outputs/anomaly_scores.csv
- outputs/scheduling_plan.csv
- outputs/summary.json

Notes:
- This is a faithful engineering implementation aligned with the paper workflow, but OpenStack/Kubernetes actions
  are simulated as an assignment optimization problem to keep the code runnable in a standalone Python environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def safe_float(x) -> float:
    """Convert to float with fallback."""
    try:
        return float(x)
    except Exception:
        return float("nan")


def parse_blood_pressure(bp_str: str) -> Tuple[float, float]:
    """
    Parse blood pressure string like '(111, 84)' into (systolic, diastolic).
    """
    if pd.isna(bp_str):
        return (np.nan, np.nan)
    s = str(bp_str).strip()
    s = s.replace("(", "").replace(")", "")
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        return (np.nan, np.nan)
    return (safe_float(parts[0]), safe_float(parts[1]))


def zscore_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean/std for z-score normalization."""
    mu = np.nanmean(x, axis=0)
    sigma = np.nanstd(x, axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return mu, sigma


def zscore_transform(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Apply z-score normalization."""
    return (x - mu) / sigma


def robust_cov_inv(x: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """
    Compute an invertible covariance matrix inverse with ridge regularization.
    """
    x = np.asarray(x, dtype=float)
    # Use row-wise mean imputation for NaNs
    col_means = np.nanmean(x, axis=0)
    x2 = np.where(np.isnan(x), col_means, x)
    cov = np.cov(x2, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    cov = cov + ridge * np.eye(cov.shape[0], dtype=float)
    return np.linalg.inv(cov)


# -----------------------------
# DWD-OD: Dynamic Weighted Distance-Based Outlier Detection
# -----------------------------

@dataclass
class DWDODConfig:
    window_size: int = 240            # number of rows per window (engineering default for streaming)
    step_size: int = 60               # slide step
    k_neighbors: int = 10             # k in kNN neighborhood
    threshold_quantile: float = 0.95  # dynamic local density threshold quantile
    ridge: float = 1e-6               # covariance ridge


def build_alpha_weights(feature_names: List[str]) -> np.ndarray:
    """
    Build device-type priority alpha for each feature dimension.

    Engineering mapping:
    - Medical equipment and critical task indicators get higher alpha
    - HVAC is important but typically has higher variance
    - Lighting is lower power but should be monitored carefully
    - Vital signs features are critical for clinical response

    You may tune these values to match local hospital policies.
    """
    alpha_map: Dict[str, float] = {
        "Medical Equipment Power Usage (kWh)": 1.30,
        "HVAC Power Usage (kWh)": 1.10,
        "Lighting Power Usage (kWh)": 1.15,
        "Total Power Usage (kWh)": 1.20,
        "Energy Consumption (kWh)": 1.20,
        "Renewable Energy Usage (%)": 1.05,
        "HVAC Efficiency (%)": 1.05,
        "Room Temperature (°C)": 1.10,
        "Room Humidity (%)": 1.05,
        "Outdoor Temperature (°C)": 1.00,
        "Outdoor Humidity (%)": 1.00,
        "Temperature (°C)": 1.10,
        "Humidity (%)": 1.00,
        "Oxygen Level (%)": 1.25,
        "Heart Rate (bpm)": 1.20,
        "BP Systolic (mmHg)": 1.15,
        "BP Diastolic (mmHg)": 1.15,
        "Energy Saving Mode": 1.05,
        "System Health Check": 1.10,
    }

    alpha = np.ones(len(feature_names), dtype=float)
    for i, fn in enumerate(feature_names):
        alpha[i] = alpha_map.get(fn, 1.0)
    return alpha


def load_fluctuation_beta(window_raw: np.ndarray) -> np.ndarray:
    """
    Compute load fluctuation coefficient beta_k within the current window.
    Paper definition: beta = std / mean for each feature dimension.
    Use absolute mean to avoid sign issues and add epsilon.
    """
    eps = 1e-12
    mu = np.nanmean(window_raw, axis=0)
    sd = np.nanstd(window_raw, axis=0)
    denom = np.maximum(np.abs(mu), eps)
    beta = sd / denom
    beta = np.where(np.isnan(beta), 0.0, beta)
    return beta


def dynamic_weights(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Equation-style weight:
    w_k = alpha_k * (1 + beta_k / max(beta))
    """
    eps = 1e-12
    max_beta = float(np.max(beta)) if np.max(beta) > eps else 1.0
    w = alpha * (1.0 + beta / max_beta)
    return w


def amd_distance(
    xi: np.ndarray,
    xj: np.ndarray,
    cov_inv: np.ndarray,
    w: np.ndarray
) -> float:
    """
    Adaptive Mahalanobis Distance with dynamic weight matrix Wd = diag(w):
    d = sqrt( (xi-xj)^T * Wd * cov_inv * Wd * (xi-xj) )
    """
    dvec = (xi - xj).reshape(-1, 1)
    wd = np.diag(w)
    val = float(dvec.T @ wd @ cov_inv @ wd @ dvec)
    return math.sqrt(max(val, 0.0))


def knn_avg_distance(
    x: np.ndarray,
    idx: int,
    cov_inv: np.ndarray,
    w: np.ndarray,
    k: int
) -> float:
    """
    Compute average AMD distance from sample idx to its k nearest neighbors inside x.
    A straightforward O(n^2) approach is used for clarity and correctness.
    """
    n = x.shape[0]
    if n <= 1:
        return 0.0

    distances = []
    xi = x[idx]
    for j in range(n):
        if j == idx:
            continue
        d = amd_distance(xi, x[j], cov_inv, w)
        distances.append(d)

    distances.sort()
    k_eff = min(k, len(distances))
    if k_eff == 0:
        return 0.0
    return float(np.mean(distances[:k_eff]))


def dwdod_detect(
    df: pd.DataFrame,
    feature_cols: List[str],
    cfg: DWDODConfig
) -> pd.DataFrame:
    """
    Run sliding-window DWD-OD anomaly detection.

    Returns dataframe with:
    - anomaly_score (LOF-like)
    - threshold
    - is_anomaly
    """
    df = df.copy()
    df = df.sort_values("Timestamp").reset_index(drop=True)

    raw_all = df[feature_cols].to_numpy(dtype=float)
    mu, sigma = zscore_fit(raw_all)
    z_all = zscore_transform(raw_all, mu, sigma)

    alpha = build_alpha_weights(feature_cols)

    scores = np.full(df.shape[0], np.nan, dtype=float)
    thresholds = np.full(df.shape[0], np.nan, dtype=float)

    n = df.shape[0]
    wsize = cfg.window_size
    step = cfg.step_size
    k = cfg.k_neighbors

    for start in range(0, n, step):
        end = min(start + wsize, n)
        if end - start < max(20, k + 2):
            break

        win_raw = raw_all[start:end]
        win_z = z_all[start:end]

        beta = load_fluctuation_beta(win_raw)
        w = dynamic_weights(alpha, beta)

        cov_inv = robust_cov_inv(win_z, ridge=cfg.ridge)

        # Density proxy: density_i = 1 / (avg_k_dist + eps)
        avg_dist = np.zeros(end - start, dtype=float)
        density = np.zeros(end - start, dtype=float)

        for i in range(end - start):
            ad = knn_avg_distance(win_z, i, cov_inv, w, k)
            avg_dist[i] = ad
            density[i] = 1.0 / (ad + 1e-12)

        # Local density baseline
        rho_local = float(np.mean(density)) + 1e-12

        # LOF-like score aligned with paper’s idea:
        # Larger avg distance with respect to local density indicates anomaly.
        lof_like = avg_dist * rho_local

        # Dynamic local density threshold using quantile inside window
        th = float(np.quantile(lof_like, cfg.threshold_quantile))

        scores[start:end] = lof_like
        thresholds[start:end] = th

    df["anomaly_score"] = scores
    df["threshold"] = thresholds
    df["is_anomaly"] = (df["anomaly_score"] > df["threshold"]).astype(int)

    return df


# -----------------------------
# MRCOF Scheduling Simulation with GSAA
# -----------------------------

@dataclass
class Node:
    node_id: int
    compute_capacity: float          # abstract capacity units
    renewable_ratio: float           # 0..1
    base_latency_ms: float           # baseline latency in milliseconds


@dataclass
class GSAAConfig:
    population_size: int = 60
    generations: int = 120
    w1_latency: float = 0.55          # omega_1 in Eq.(6)
    w2_energy: float = 0.45           # omega_2 in Eq.(6)
    crossover_p0: float = 0.90
    mutation_p0: float = 0.10
    lambda_decay: float = 2.5
    u_rise: float = 2.0
    sa_temp0: float = 1.0
    sa_temp_min: float = 0.02


def build_nodes(num_nodes: int, df: pd.DataFrame) -> List[Node]:
    """
    Create a small cluster of nodes with different energy/latency characteristics.
    Renewable ratio is informed by dataset distribution to keep realism.
    """
    renewable = df["Renewable Energy Usage (%)"].to_numpy(dtype=float)
    renewable = np.clip(renewable / 100.0, 0.0, 1.0)
    r_mean = float(np.nanmean(renewable))
    r_std = float(np.nanstd(renewable))

    nodes: List[Node] = []
    for j in range(num_nodes):
        rr = float(np.clip(np.random.normal(r_mean, max(r_std, 0.05)), 0.0, 1.0))
        cap = float(np.random.uniform(0.8, 1.4))
        lat = float(np.random.uniform(8.0, 22.0))  # ms
        nodes.append(Node(node_id=j, compute_capacity=cap, renewable_ratio=rr, base_latency_ms=lat))
    return nodes


def energy_efficiency_score(node: Node, alpha: float = 0.6, beta: float = 0.4) -> float:
    """
    A practical form of Eq.(9) score:
    - compute utilization proxy uses capacity (higher is better)
    - renewable ratio uses node.renewable_ratio (higher is better)
    """
    return alpha * node.compute_capacity + beta * node.renewable_ratio


def task_priority_weight(row: pd.Series) -> float:
    """
    Build a task priority weight.
    Critical tasks should be scheduled to better nodes.

    Heuristic:
    - "System Health Check"==0 indicates risk; raise priority
    - "AI Predicted Health Status" == "Unhealthy" indicates criticality; raise priority
    """
    w = 1.0
    if int(row.get("System Health Check", 1)) == 0:
        w *= 1.25
    if str(row.get("AI Predicted Health Status", "")).strip().lower() == "unhealthy":
        w *= 1.30
    return w


def assignment_cost(
    assignment: np.ndarray,
    tasks: pd.DataFrame,
    nodes: List[Node],
    cfg: GSAAConfig
) -> float:
    """
    Fitness (to minimize) inspired by Eq.(6):
      omega_1 * sum delay + omega_2 * sum energy
    """
    # Track per-node load for queuing delay
    node_load = np.zeros(len(nodes), dtype=float)

    delay_sum = 0.0
    energy_sum = 0.0

    for i, (_, row) in enumerate(tasks.iterrows()):
        nid = int(assignment[i])
        node = nodes[nid]

        # Energy proxy: total power usage scaled by (1 - renewable_ratio)
        # Larger renewable ratio reduces effective grid energy impact.
        power = float(row.get("Total Power Usage (kWh)", row.get("Energy Consumption (kWh)", 0.0)))
        power = 0.0 if math.isnan(power) else max(power, 0.0)
        eff = energy_efficiency_score(node)
        eff = max(eff, 1e-6)

        # Energy term: higher efficiency reduces energy cost; more renewables reduce cost.
        energy_cost = power / eff * (1.0 - node.renewable_ratio + 0.05)

        # Delay proxy: base latency + queuing delay from load + penalty for critical tasks on low-score nodes
        pr = task_priority_weight(row)
        queue_delay = 3.0 * node_load[nid] / max(node.compute_capacity, 1e-6)  # ms
        score = energy_efficiency_score(node)
        critical_penalty = 0.0
        if pr > 1.2:
            critical_penalty = 10.0 * max(0.0, (1.0 - score))  # ms penalty

        delay = node.base_latency_ms + queue_delay + critical_penalty

        delay_sum += delay
        energy_sum += energy_cost

        # Update load: use a power-based proxy as "work"
        node_load[nid] += (power * pr) / 10.0

    return cfg.w1_latency * delay_sum + cfg.w2_energy * energy_sum


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One-point crossover for integer assignments."""
    n = len(parent1)
    if n < 2:
        return parent1.copy(), parent2.copy()
    point = random.randint(1, n - 1)
    c1 = np.concatenate([parent1[:point], parent2[point:]])
    c2 = np.concatenate([parent2[:point], parent1[point:]])
    return c1, c2


def mutate(child: np.ndarray, num_nodes: int, mutation_rate: float) -> np.ndarray:
    """Random reset mutation for assignment genes."""
    out = child.copy()
    for i in range(len(out)):
        if random.random() < mutation_rate:
            out[i] = random.randint(0, num_nodes - 1)
    return out


def adaptive_probabilities(gen: int, max_gen: int, cfg: GSAAConfig) -> Tuple[float, float]:
    """
    Adaptive crossover/mutation inspired by the paper's decay/rise behavior:
    pc = p0 * exp(-lambda * t)
    pm = p0 * (1 - exp(-u * t))
    where t in [0, 1].
    """
    t = gen / max(1, max_gen)
    pc = cfg.crossover_p0 * math.exp(-cfg.lambda_decay * t)
    pm = cfg.mutation_p0 * (1.0 - math.exp(-cfg.u_rise * t))
    pc = float(np.clip(pc, 0.05, 0.95))
    pm = float(np.clip(pm, 0.001, 0.35))
    return pc, pm


def sa_temperature(gen: int, max_gen: int, cfg: GSAAConfig) -> float:
    """Geometric cooling schedule."""
    t = gen / max(1, max_gen)
    temp = cfg.sa_temp0 * (cfg.sa_temp_min / cfg.sa_temp0) ** t
    return float(np.clip(temp, cfg.sa_temp_min, cfg.sa_temp0))


def gsaa_optimize(
    tasks: pd.DataFrame,
    nodes: List[Node],
    cfg: GSAAConfig
) -> Tuple[np.ndarray, float]:
    """
    Genetic Simulated Annealing Algorithm (GSAA) for task-to-node assignment.
    """
    num_nodes = len(nodes)
    num_tasks = tasks.shape[0]

    # Initialize population
    pop = [np.random.randint(0, num_nodes, size=num_tasks, dtype=int) for _ in range(cfg.population_size)]
    fitness = [assignment_cost(ind, tasks, nodes, cfg) for ind in pop]

    best_idx = int(np.argmin(fitness))
    best = pop[best_idx].copy()
    best_fit = float(fitness[best_idx])

    for gen in range(cfg.generations):
        pc, pm = adaptive_probabilities(gen, cfg.generations, cfg)
        temp = sa_temperature(gen, cfg.generations, cfg)

        # Tournament selection
        def tournament() -> np.ndarray:
            a, b = random.sample(range(cfg.population_size), 2)
            return pop[a] if fitness[a] < fitness[b] else pop[b]

        new_pop: List[np.ndarray] = []
        new_fit: List[float] = []

        while len(new_pop) < cfg.population_size:
            p1 = tournament()
            p2 = tournament()

            # Crossover
            if random.random() < pc:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            c1 = mutate(c1, num_nodes, pm)
            c2 = mutate(c2, num_nodes, pm)

            # Simulated annealing acceptance (Metropolis criterion)
            f1 = assignment_cost(c1, tasks, nodes, cfg)
            f2 = assignment_cost(c2, tasks, nodes, cfg)

            # Replace strategy: accept worse solutions with probability exp(-(delta)/T)
            def accept(child: np.ndarray, f_child: float, parent: np.ndarray, f_parent: float) -> Tuple[np.ndarray, float]:
                if f_child <= f_parent:
                    return child, f_child
                delta = f_child - f_parent
                prob = math.exp(-delta / max(temp, 1e-12))
                if random.random() < prob:
                    return child, f_child
                return parent.copy(), f_parent

            # Use each child's corresponding parent cost for acceptance
            f_p1 = assignment_cost(p1, tasks, nodes, cfg)
            f_p2 = assignment_cost(p2, tasks, nodes, cfg)

            a1, af1 = accept(c1, f1, p1, f_p1)
            a2, af2 = accept(c2, f2, p2, f_p2)

            new_pop.append(a1)
            new_fit.append(af1)
            if len(new_pop) < cfg.population_size:
                new_pop.append(a2)
                new_fit.append(af2)

        pop, fitness = new_pop, new_fit

        gen_best_idx = int(np.argmin(fitness))
        gen_best_fit = float(fitness[gen_best_idx])
        if gen_best_fit < best_fit:
            best_fit = gen_best_fit
            best = pop[gen_best_idx].copy()

    return best, best_fit


# -----------------------------
# Data loading and pipeline
# -----------------------------

def prepare_dataframe(csv_path: str) -> pd.DataFrame:
    """Load and clean the dataset for downstream processing."""
    df = pd.read_csv(csv_path)

    # Timestamp parsing
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Blood pressure parsing into two numeric columns
    sys_bp = []
    dia_bp = []
    for v in df["Blood Pressure (mmHg)"].values:
        s, d = parse_blood_pressure(v)
        sys_bp.append(s)
        dia_bp.append(d)
    df["BP Systolic (mmHg)"] = sys_bp
    df["BP Diastolic (mmHg)"] = dia_bp

    # Basic encoding for small integer/binary columns if missing type
    for col in ["Energy Saving Mode", "System Health Check", "Day of the Week"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def select_features(df: pd.DataFrame) -> List[str]:
    """
    Select feature columns for DWD-OD.
    Use numeric features that describe energy consumption and operating states.
    """
    preferred = [
        "Energy Consumption (kWh)",
        "Renewable Energy Usage (%)",
        "HVAC Power Usage (kWh)",
        "Lighting Power Usage (kWh)",
        "Medical Equipment Power Usage (kWh)",
        "Total Power Usage (kWh)",
        "HVAC Efficiency (%)",
        "Room Temperature (°C)",
        "Room Humidity (%)",
        "Outdoor Temperature (°C)",
        "Outdoor Humidity (%)",
        "Temperature (°C)",
        "Humidity (%)",
        "Oxygen Level (%)",
        "Heart Rate (bpm)",
        "BP Systolic (mmHg)",
        "BP Diastolic (mmHg)",
        "Energy Saving Mode",
        "System Health Check",
    ]
    cols = [c for c in preferred if c in df.columns]
    if len(cols) < 8:
        # Fallback: all numeric columns except obvious identifiers
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        drop_like = {"Day of the Week"}
        cols = [c for c in numeric_cols if c not in drop_like]
    return cols


def run_pipeline(
    csv_path: str,
    out_dir: str,
    seed: int,
    dwd_cfg: DWDODConfig,
    num_nodes: int,
    max_tasks_for_scheduling: int,
    gsaa_cfg: GSAAConfig
) -> None:
    set_seed(seed)
    ensure_dir(out_dir)

    df = prepare_dataframe(csv_path)
    feature_cols = select_features(df)

    # -------------------------
    # Part A: DWD-OD
    # -------------------------
    det = dwdod_detect(df, feature_cols, dwd_cfg)

    anomaly_out = det[["Timestamp", "Patient ID", "anomaly_score", "threshold", "is_anomaly"]].copy()
    anomaly_out_path = os.path.join(out_dir, "anomaly_scores.csv")
    anomaly_out.to_csv(anomaly_out_path, index=False)

    # -------------------------
    # Part B: MRCOF-style scheduling simulation
    # -------------------------
    # Stage-1: isolate anomalies
    normal_tasks = det[det["is_anomaly"] == 0].copy()
    anomalous_tasks = det[det["is_anomaly"] == 1].copy()

    # Keep scheduling size manageable and reproducible
    if normal_tasks.shape[0] > max_tasks_for_scheduling:
        normal_tasks = normal_tasks.sample(max_tasks_for_scheduling, random_state=seed).sort_values("Timestamp")

    nodes = build_nodes(num_nodes, det)

    # Stage-2: GSAA optimization for normal tasks
    best_assign, best_fit = gsaa_optimize(normal_tasks, nodes, gsaa_cfg)

    plan = normal_tasks[["Timestamp", "Patient ID"]].copy().reset_index(drop=True)
    plan["assigned_node"] = best_assign.astype(int)

    # Add node metadata for readability
    node_eff = [energy_efficiency_score(nodes[int(n)]) for n in plan["assigned_node"].values]
    node_ren = [nodes[int(n)].renewable_ratio for n in plan["assigned_node"].values]
    node_lat = [nodes[int(n)].base_latency_ms for n in plan["assigned_node"].values]
    plan["node_efficiency_score"] = node_eff
    plan["node_renewable_ratio"] = node_ren
    plan["node_base_latency_ms"] = node_lat

    plan_path = os.path.join(out_dir, "scheduling_plan.csv")
    plan.to_csv(plan_path, index=False)

    # -------------------------
    # Summary
    # -------------------------
    summary = {
        "dataset": os.path.basename(csv_path),
        "rows_total": int(df.shape[0]),
        "features_used_for_dwdod": feature_cols,
        "dwdod": {
            "window_size": dwd_cfg.window_size,
            "step_size": dwd_cfg.step_size,
            "k_neighbors": dwd_cfg.k_neighbors,
            "threshold_quantile": dwd_cfg.threshold_quantile,
            "anomalies_detected": int(det["is_anomaly"].sum()),
            "anomaly_rate": float(det["is_anomaly"].mean()),
        },
        "scheduling": {
            "num_nodes": num_nodes,
            "tasks_scheduled": int(normal_tasks.shape[0]),
            "isolated_anomalous_tasks": int(anomalous_tasks.shape[0]),
            "gsaa_best_fitness": float(best_fit),
            "gsaa": {
                "population_size": gsaa_cfg.population_size,
                "generations": gsaa_cfg.generations,
                "w1_latency": gsaa_cfg.w1_latency,
                "w2_energy": gsaa_cfg.w2_energy,
            },
            "nodes": [
                {
                    "node_id": n.node_id,
                    "compute_capacity": n.compute_capacity,
                    "renewable_ratio": n.renewable_ratio,
                    "base_latency_ms": n.base_latency_ms,
                    "efficiency_score": energy_efficiency_score(n),
                }
                for n in nodes
            ],
        },
        "outputs": {
            "anomaly_scores_csv": anomaly_out_path,
            "scheduling_plan_csv": plan_path,
        }
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Anomaly scores saved to: {anomaly_out_path}")
    print(f"Scheduling plan saved to: {plan_path}")
    print(f"Summary saved to: {summary_path}")


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DWD-OD + MRCOF(GSAA) runnable reference implementation")
    p.add_argument("--csv", type=str, default="/mnt/data/hospital_communication_energy_system.csv", help="Path to CSV dataset")
    p.add_argument("--out", type=str, default="outputs", help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # DWD-OD
    p.add_argument("--window-size", type=int, default=240, help="Sliding window size (rows)")
    p.add_argument("--step-size", type=int, default=60, help="Sliding window step (rows)")
    p.add_argument("--k", type=int, default=10, help="k nearest neighbors")
    p.add_argument("--th-quantile", type=float, default=0.95, help="Threshold quantile inside window")

    # Scheduling
    p.add_argument("--nodes", type=int, default=6, help="Number of cluster nodes for scheduling simulation")
    p.add_argument("--max-tasks", type=int, default=1200, help="Max normal tasks to schedule for runtime control")

    # GSAA
    p.add_argument("--pop", type=int, default=60, help="GSAA population size")
    p.add_argument("--gen", type=int, default=120, help="GSAA generations")
    p.add_argument("--w1", type=float, default=0.55, help="Latency weight (omega_1)")
    p.add_argument("--w2", type=float, default=0.45, help="Energy weight (omega_2)")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    dwd_cfg = DWDODConfig(
        window_size=args.window_size,
        step_size=args.step_size,
        k_neighbors=args.k,
        threshold_quantile=args.th_quantile,
    )

    gsaa_cfg = GSAAConfig(
        population_size=args.pop,
        generations=args.gen,
        w1_latency=args.w1,
        w2_energy=args.w2,
    )

    run_pipeline(
        csv_path=args.csv,
        out_dir=args.out,
        seed=args.seed,
        dwd_cfg=dwd_cfg,
        num_nodes=args.nodes,
        max_tasks_for_scheduling=args.max_tasks,
        gsaa_cfg=gsaa_cfg,
    )


if __name__ == "__main__":
    main()
