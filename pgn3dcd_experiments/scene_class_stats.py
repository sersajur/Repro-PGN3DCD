"""Per-scene change/unchanged class statistics for an HKCD split.

Reads the raw cd_type label directly from each scene's pointCloud0.ply and
pointCloud1.ply (unambiguous scene=folder mapping; does NOT rely on the
preprocessed .pt index order, which can differ between OSes).

Usage:
    python scene_class_stats.py <split_dir>

    <split_dir>  Directory containing one sub-folder per scene, each with
                 pointCloud0.ply and pointCloud1.ply.

Prints a per-scene table (change points, total points, change ratio) plus the
aggregate ratio over the whole split.
"""
import os
import os.path as osp
import sys

import numpy as np
from plyfile import PlyData


def read_cd_type(ply_path):
    with open(ply_path, "rb") as fh:
        ply = PlyData.read(fh)
        data = ply.elements[0].data
        try:
            cd = data["scalar_cd_type"]
        except (ValueError, KeyError):
            cd = data["cd_type"]
    return np.asarray(cd).astype(np.int64)


def scene_counts(scene_dir):
    n_change = 0
    n_total = 0
    for fname in ("pointCloud0.ply", "pointCloud1.ply"):
        cd = read_cd_type(osp.join(scene_dir, fname))
        n_change += int((cd == 1).sum())
        n_total += int(cd.shape[0])
    return n_change, n_total


def main(split_dir):
    scenes = sorted(
        d.name for d in os.scandir(split_dir) if d.is_dir()
    )
    rows = []
    tot_change = 0
    tot_points = 0
    for name in scenes:
        n_change, n_total = scene_counts(osp.join(split_dir, name))
        ratio = n_change / n_total if n_total else 0.0
        rows.append((name, n_change, n_total, ratio))
        tot_change += n_change
        tot_points += n_total
        print(f"{name:14s}  change={n_change:>10,}  total={n_total:>11,}  ratio={ratio:7.4f}")

    global_ratio = tot_change / tot_points if tot_points else 0.0
    print("-" * 64)
    print(f"{'GLOBAL':14s}  change={tot_change:>10,}  total={tot_points:>11,}  ratio={global_ratio:7.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])