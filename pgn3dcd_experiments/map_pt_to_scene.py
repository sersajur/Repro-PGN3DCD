"""Map preprocessed pc0_<i>.pt files to scene names by geographic bbox.

The .pt index comes from os.scandir order, which is OS/filesystem-dependent, so
"pc0_0 = first alphabetically" is NOT reliable. This matches each pc0_<i>.pt to
the raw scene whose pointCloud0.ply has the closest XY bounding box —
GridSampling3D preserves spatial extent and each tile sits at distinct
coordinates, so the bbox is a robust, order-independent fingerprint.

Run it on the machine where the .pt files live (e.g. the GCP VM, inside the
container where /data is mounted).

Usage:
    python map_pt_to_scene.py <preprocessed_split_dir> <raw_split_dir>

    <preprocessed_split_dir>  dir holding pc0_<i>.pt (e.g. /data/HKCD/preprocessed_debug/Test)
    <raw_split_dir>           dir holding scene sub-folders (e.g. /data/HKCD/Test_subset)
"""
import glob
import os
import os.path as osp
import re
import sys

import numpy as np
from plyfile import PlyData

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from pt_to_ply import _load_data, _tensors  # noqa: E402


def pt_bbox(path):
    pos = _tensors(_load_data(path))["pos"].numpy()
    return pos.shape[0], (pos[:, 0].min(), pos[:, 0].max(), pos[:, 1].min(), pos[:, 1].max())


def raw_bbox(path):
    d = PlyData.read(path).elements[0].data
    return d.shape[0], (d["x"].min(), d["x"].max(), d["y"].min(), d["y"].max())


def main(pre_dir, raw_dir):
    pts = {}
    for p in glob.glob(osp.join(pre_dir, "pc0_*.pt")):
        idx = int(re.search(r"pc0_(\d+)\.pt", p).group(1))
        pts[idx] = pt_bbox(p)

    scenes = {}
    for d in sorted(os.scandir(raw_dir), key=lambda e: e.name):
        ply = osp.join(d.path, "pointCloud0.ply")
        if d.is_dir() and osp.isfile(ply):
            scenes[d.name] = raw_bbox(ply)

    print(f"{len(pts)} preprocessed .pt  vs  {len(scenes)} raw scenes\n")
    for idx in sorted(pts):
        n_pt, bb_pt = pts[idx]
        name, diff = min(
            ((nm, sum(abs(a - b) for a, b in zip(bb_pt, bb_raw))) for nm, (_, bb_raw) in scenes.items()),
            key=lambda x: x[1],
        )
        flag = "" if diff < 5.0 else "  <-- WEAK MATCH, check manually"
        print(f"  pc0_{idx}.pt (N={n_pt:>9,})  ->  {name}   (bbox Δ={diff:.2f} m){flag}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])