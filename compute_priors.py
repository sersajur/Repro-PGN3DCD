#!/usr/bin/env python3
"""
Compute per-point prior scalars for a pair of HKCD point clouds.

For each point in both clouds, computes:
  - D          geometric distance prior  (PGN3DCD original)
  - C          colour difference prior    (PGN3DCD original)
  - p          original PGN3DCD prior     = 0.5·D + 0.5·C
  - mu         Bayesian fused mean        (our approach)
  - var_total  Bayesian fused variance    (our approach)

Outputs two PLY files (one per cloud) with the original xyz/rgb/label
plus the five scalar fields above.  Open in CloudCompare to visualise.

Usage
-----
    python compute_priors.py \
        /data/HKCD/Test/sample_0/pointCloud0.ply \
        /data/HKCD/Test/sample_0/pointCloud1.ply \
        --out /tmp/priors

Dependencies: numpy, scipy, plyfile
    pip install numpy scipy plyfile
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement


# ── Constants (identical to SiamKPConvWithPriorUncertainty.py) ───────────
_EPS = 1e-6
_MAX_VAR = 0.25
POS_TH = 3.0   # geometric distance threshold  (metres)
F_TH = 0.6     # colour difference threshold    (normalised)


# ── Load / Save ─────────────────────────────────────────────────────────

def load_ply(path: str) -> dict:
    """Load an HKCD PLY file → dict with pos, rgb, label arrays."""
    ply = PlyData.read(path)
    v = ply["vertex"]
    pos = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)

    rgb = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.float64)
    rgb_norm = rgb / 255.0  # model works with [0, 1]

    # Label field
    label = None
    for name in ("scalar_cd_type", "cd_type"):
        if name in [p.name for p in v.properties]:
            label = np.array(v[name]).astype(np.int32)
            break
    if label is None:
        label = np.zeros(len(pos), dtype=np.int32)
        print(f"  warning: no label field in {path}, using zeros")

    return {
        "pos": pos,
        "rgb": rgb,           # uint8-scale (0-255) — for output PLY
        "rgb_norm": rgb_norm,  # [0,1] — for computation
        "label": label,
    }


def save_ply(path: str, pc: dict, scalars: dict):
    """Save point cloud with original fields + computed scalar fields."""
    n = len(pc["pos"])
    rgb = pc["rgb"].astype(np.uint8)

    # Build structured array
    dtype = [
        ("x", "f8"), ("y", "f8"), ("z", "f8"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("label", "i4"),
        ("scalar_D", "f4"),
        ("scalar_C", "f4"),
        ("scalar_p", "f4"),
        ("scalar_mu", "f4"),
        ("scalar_var_total", "f4"),
    ]
    arr = np.empty(n, dtype=dtype)
    arr["x"] = pc["pos"][:, 0]
    arr["y"] = pc["pos"][:, 1]
    arr["z"] = pc["pos"][:, 2]
    arr["red"] = rgb[:, 0]
    arr["green"] = rgb[:, 1]
    arr["blue"] = rgb[:, 2]
    arr["label"] = pc["label"]
    arr["scalar_D"] = scalars["D"].astype(np.float32)
    arr["scalar_C"] = scalars["C"].astype(np.float32)
    arr["scalar_p"] = scalars["p"].astype(np.float32)
    arr["scalar_mu"] = scalars["mu"].astype(np.float32)
    arr["scalar_var_total"] = scalars["var_total"].astype(np.float32)

    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(path)


# ── Computation ─────────────────────────────────────────────────────────

def compute_scalars(source: dict, target: dict) -> dict:
    """
    For each point in *target*, find its nearest neighbour in *source*
    and compute D, C, p, mu, var_total.

    This mirrors SiamKPConvWithPriorUncertainty.get_mask_v2:
        get_mask_v2(pos_1=source, pos_2=target, ...)
    """
    tree = cKDTree(source["pos"])
    dists, idxs = tree.query(target["pos"], k=1)

    # ── D: geometric distance prior ∈ [0, 1] ────────────────────────
    D = np.minimum(dists / POS_TH, 1.0).reshape(-1)

    # ── C: colour difference prior ∈ [0, 1] ─────────────────────────
    nearest_rgb = source["rgb_norm"][idxs]          # (N, 3)
    f_diff = np.abs(np.mean(target["rgb_norm"] - nearest_rgb, axis=1))
    C = np.minimum(f_diff / F_TH, 1.0)

    # ── p: original PGN3DCD prior (simple 50/50 blend) ───────────────
    p = 0.5 * D + 0.5 * C

    # ── Bayesian inverse-variance fusion (our approach) ──────────────
    var_D = np.clip(D * (1.0 - D), _EPS, None)
    var_C = np.clip(C * (1.0 - C), _EPS, None)

    w_D = 1.0 / var_D
    w_C = 1.0 / var_C
    w_sum = w_D + w_C

    mu = (w_D * D + w_C * C) / w_sum
    var_total = 1.0 / w_sum

    return {"D": D, "C": C, "p": p, "mu": mu, "var_total": var_total}


def print_stats(name: str, scalars: dict):
    """Print summary statistics."""
    print(f"\n  {name}:")
    for key in ("D", "C", "p", "mu", "var_total"):
        v = scalars[key]
        print(f"    {key:12s}  mean={v.mean():.4f}  std={v.std():.4f}"
              f"  min={v.min():.4f}  max={v.max():.4f}")
    vt = scalars["var_total"]
    print(f"    {'':12s}  %high(>0.10)={100*(vt>0.10).mean():.1f}%"
          f"  %vhigh(>0.20)={100*(vt>0.20).mean():.1f}%")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute per-point priors (D, C, p, mu, var_total) "
                    "for a pair of HKCD point clouds."
    )
    parser.add_argument("ply0", help="Path to pointCloud0.ply")
    parser.add_argument("ply1", help="Path to pointCloud1.ply")
    parser.add_argument("--out", required=True,
                        help="Output directory for result PLY files")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────
    print(f"Loading {args.ply0} ...")
    pc0 = load_ply(args.ply0)
    print(f"  {len(pc0['pos']):,} points")

    print(f"Loading {args.ply1} ...")
    pc1 = load_ply(args.ply1)
    print(f"  {len(pc1['pos']):,} points")

    # ── Compute for cloud 0 (target=pc0, source=pc1) ────────────────
    print("\nComputing scalars for pointCloud0 (source=pc1) ...")
    scalars0 = compute_scalars(source=pc1, target=pc0)
    print_stats("pointCloud0", scalars0)

    # ── Compute for cloud 1 (target=pc1, source=pc0) ────────────────
    print("\nComputing scalars for pointCloud1 (source=pc0) ...")
    scalars1 = compute_scalars(source=pc0, target=pc1)
    print_stats("pointCloud1", scalars1)

    # ── Save ─────────────────────────────────────────────────────────
    out0 = out_dir / "pointCloud0_priors.ply"
    out1 = out_dir / "pointCloud1_priors.ply"

    print(f"\nSaving {out0} ...")
    save_ply(str(out0), pc0, scalars0)

    print(f"Saving {out1} ...")
    save_ply(str(out1), pc1, scalars1)

    print(f"\nDone!  Open in CloudCompare:")
    print(f"  {out0}")
    print(f"  {out1}")
    print(f"\nIn CloudCompare: Properties → Active scalar field →"
          f" scalar_var_total → Color scale → coolwarm")


if __name__ == "__main__":
    main()
