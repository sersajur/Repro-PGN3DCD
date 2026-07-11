"""Per-cylinder eval helpers.

Two use-cases share the SAME cylinder id scheme (`<scene>--<idx>`, idx = the
running per-scene index in test-dataloader order):

1. `CylinderDumper` — CSV-only, driven by the Trainer during eval. Writes
   `cylinders.csv` with per-cylinder stats. Cheap (no PLY, no capture hook).

2. `cylinder_arrays` / `write_ply` — reused by the standalone
   `pgn3dcd_experiments/dump_cylinders_by_id.py` to dump PLYs for chosen ids
   (with per-point mu/var/mask from the prior model's `_dump_io` hook).

CSV columns: cylinder_id, points_number, gt_unchanged, gt_changed,
predicted_unchanged, predicted_changed, correct, IOU_mean, IOU_changed,
IOU_unchanged. IoU is binary (class 0 = unchanged, class >=1 = changed) over the
cylinder's combined PC0+PC1 points; `correct` = count of points where the
(binary) prediction matches GT.
"""
import csv
import os

import numpy as np
from plyfile import PlyData, PlyElement

CSV_FIELDS = [
    "cylinder_id", "points_number", "gt_unchanged", "gt_changed",
    "predicted_unchanged", "predicted_changed", "correct",
    "IOU_mean", "IOU_changed", "IOU_unchanged",
]

# Per-cloud PLY fields (PC0 and PC1 go to separate files).
PLY_DTYPE = [
    ("x", "f4"), ("y", "f4"), ("z", "f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ("scalar_cylinder", "f4"),
    ("scalar_gt", "f4"), ("scalar_pred", "f4"), ("scalar_correct", "f4"),
    ("scalar_prob_change", "f4"),
    ("scalar_mu", "f4"), ("scalar_var", "f4"), ("scalar_mask", "f4"),
]


def scene_name(dataset, area):
    """Scene folder name for a given area index (e.g. '11-NE-12B')."""
    return dataset.filesPC0[area].split("/")[-2]


def stats_row(cyl_id, gt, pred):
    """One CSV row from combined (PC0+PC1) 1-D int arrays of gt / pred."""
    g = (gt >= 1).astype(np.int64)
    pr = (pred >= 1).astype(np.int64)

    def iou(c):
        union = int(((g == c) | (pr == c)).sum())
        inter = int(((g == c) & (pr == c)).sum())
        return inter / union if union else float("nan")

    iou0, iou1 = iou(0), iou(1)
    mean = float(np.nanmean([iou0, iou1]))
    fmt = lambda v: round(v, 4) if v == v else "nan"   # noqa: E731
    return {
        "cylinder_id": cyl_id,
        "points_number": int(gt.shape[0]),
        "gt_unchanged": int((g == 0).sum()),
        "gt_changed": int((g == 1).sum()),
        "predicted_unchanged": int((pr == 0).sum()),
        "predicted_changed": int((pr == 1).sum()),
        "correct": int((g == pr).sum()),
        "IOU_mean": fmt(mean),
        "IOU_changed": fmt(iou1),
        "IOU_unchanged": fmt(iou0),
    }


def _cloud_array(suf, cidx, predT, yT, idxT, xT, io, p, scene_pos_full):
    """Structured PLY array for one cloud (PC0 or PC1) of one cylinder, in scene coords."""
    idx = idxT.reshape(-1).long().cpu()
    pos = scene_pos_full[idx].numpy().astype(np.float32)
    rgb = np.clip(xT[:, 1:4].detach().cpu().numpy() * 255.0, 0, 255).round().astype(np.uint8)

    logp = predT.detach().cpu().float().numpy()
    prob = np.exp(logp)
    pred = logp.argmax(1)
    prob_change = 1.0 - prob[:, 0]
    y = yT.detach().cpu().numpy()
    n = pos.shape[0]

    # io may be absent (no hook), full (prior: mu/var/mask), or partial
    # (baseline: mask only). Any missing field is filled with NaN.
    sel = None
    if io is not None and f"batch{suf}" in io:
        sel = (io[f"batch{suf}"] == p).numpy()

    def _io(key):
        if sel is None or io is None or key not in io:
            return np.full(n, np.nan, np.float32)
        return io[key].reshape(-1).numpy()[sel]

    mu, var, mask = _io(f"mu{suf}"), _io(f"var{suf}"), _io(f"mask{suf}")

    arr = np.empty(n, dtype=PLY_DTYPE)
    arr["x"], arr["y"], arr["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    arr["scalar_cylinder"] = cidx
    arr["scalar_gt"] = y
    arr["scalar_pred"] = pred
    arr["scalar_correct"] = ((pred >= 1) == (y >= 1)).astype(np.float32)
    arr["scalar_prob_change"] = prob_change
    arr["scalar_mu"], arr["scalar_var"], arr["scalar_mask"] = mu, var, mask
    return arr


def cylinder_arrays(cidx, cyl, io, p, scene_pos0, scene_pos1):
    """(PC0 array, PC1 array) for one cylinder `cyl` (a Pair from to_data_list)."""
    a0 = _cloud_array("0", cidx, cyl.pred, cyl.y, cyl.idx, cyl.x, io, p, scene_pos0)
    a1 = _cloud_array("1", cidx, cyl.pred1, cyl.y_target, cyl.idx_target, cyl.x_target, io, p, scene_pos1)
    return a0, a1


def write_ply(arr, path):
    PlyData([PlyElement.describe(arr, "vertex")], byte_order="<").write(path)


class CylinderDumper:
    """CSV-only per-cylinder stats, accumulated over an eval pass."""

    def __init__(self, dataset, out_dir):
        self._ds = dataset
        os.makedirs(out_dir, exist_ok=True)
        self._csv_path = os.path.join(out_dir, "cylinders.csv")
        self._rows = []
        self._counter = {}     # scene_name -> next cylinder index

    def add_batch(self, model, data):
        out = model.get_output()
        data.pred = out[0]
        data.pred1 = out[1]
        for cyl in data.to_data_list(0):
            area = int(cyl.area)
            name = scene_name(self._ds, area)
            cidx = self._counter.get(name, 0)
            self._counter[name] = cidx + 1

            pred0 = cyl.pred.argmax(1).detach().cpu().numpy()
            pred1 = cyl.pred1.argmax(1).detach().cpu().numpy()
            y0 = cyl.y.detach().cpu().numpy()
            y1 = cyl.y_target.detach().cpu().numpy()
            gt = np.concatenate([y0, y1])
            pred = np.concatenate([pred0, pred1])
            self._rows.append(stats_row(f"{name}:{cidx}", gt, pred))

    def finalise(self):
        with open(self._csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(self._rows)
        print(f"[cyl-csv] {len(self._rows)} cylinders -> {self._csv_path}")
