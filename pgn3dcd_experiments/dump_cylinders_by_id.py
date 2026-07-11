"""Dump chosen test cylinders to PLY by their id (`<scene>:<idx>`, from cylinders.csv).

For each requested cylinder, writes two files in scene coordinates:
    <scene>_<idx>_pc0.ply  (before)   and   <scene>_<idx>_pc1.ply  (after)
each with per-point INPUTS (rgb, mu, var, mask) and network OUTPUTS
(pred, prob_change, correct vs gt).

The cylinder id counter matches CylinderDumper (per-scene, test-dataloader order),
so ids taken from cylinders.csv resolve to the same cylinders here.

Usage (in the GPU/CPU container, like eval):
    python pgn3dcd_experiments/dump_cylinders_by_id.py \
        training.checkpoint_dir=/output/PriorUnc_debug/train_... \
        model_name=SiamKPConvWithPriorUncertainty \
        training.weight_name=miou_ch \
        +cyl_ids="11-NE-12B:3,11-NE-12C:10" \
        +dump_dir=/output/PriorUnc_debug/picked_cylinders

Or read ids from a file (one `<scene>:<idx>` per line): +cyl_ids_file=/path/ids.txt
"""
import os
import sys

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from torch_points3d.trainer_SiamKPConv import Trainer
from torch_points3d.utils.repro import set_deterministic
from torch_points3d.metrics.cylinder_dump import scene_name, cylinder_arrays, write_ply


def _parse_ids(cfg):
    ids = set()
    path = cfg.get("cyl_ids_file", "")
    if path:
        with open(path) as fh:
            ids |= {ln.strip() for ln in fh if ln.strip()}
    raw = cfg.get("cyl_ids", "")
    if raw:
        ids |= {s.strip() for s in str(raw).split(",") if s.strip()}
    return ids


@hydra.main(config_path="../conf", config_name="evalSiamKPConvWithPriorUncertainty")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    set_deterministic(cfg.get("seed", 42))

    requested = _parse_ids(cfg)
    if not requested:
        sys.exit("Provide +cyl_ids=\"11-NE-12B:3,11-NE-12C:10\" and/or "
                 "+cyl_ids_file=<path> (ids come from cylinders.csv)")
    out_dir = cfg.get("dump_dir", "cylinder_dumps")
    os.makedirs(out_dir, exist_ok=True)

    trainer = Trainer(cfg)
    model = trainer._model
    model.eval()
    model._dump_io = True

    loader = trainer._dataset.test_dataloaders[0]
    dataset = loader.dataset

    counter, found, pos_cache = {}, set(), {}
    for data in loader:
        # Assign ids for this batch WITHOUT forward (advances the shared counter
        # exactly like CylinderDumper, so ids line up with cylinders.csv).
        hits = []
        for p in range(data.num_graphs):
            area = int(data.area[p])
            name = scene_name(dataset, area)
            cidx = counter.get(name, 0)
            counter[name] = cidx + 1
            cyl_id = f"{name}:{cidx}"
            if cyl_id in requested and cyl_id not in found:
                hits.append((cyl_id, p, area, cidx))
        if not hits:
            continue

        with torch.no_grad():
            model.set_input(data, trainer._device)
            with torch.cuda.amp.autocast(enabled=model.is_mixed_precision()):
                model.forward(epoch=0)
        io = getattr(model, "_dumped_io", None)   # None/partial for non-prior models
        data.pred = model.output0
        data.pred1 = model.output1
        data_l = data.to_data_list(0)

        for cyl_id, p, area, cidx in hits:
            if area not in pos_cache:
                p0, _, _, p1, _, _ = dataset._preproc_clouds_loader(area)
                pos_cache[area] = (p0, p1)
            sp0, sp1 = pos_cache[area]
            a0, a1 = cylinder_arrays(cidx, data_l[p], io, p, sp0, sp1)
            safe = cyl_id.replace(":", "_")
            write_ply(a0, os.path.join(out_dir, f"{safe}_pc0.ply"))
            write_ply(a1, os.path.join(out_dir, f"{safe}_pc1.ply"))
            found.add(cyl_id)
            print(f"dumped {cyl_id}: PC0={len(a0)} + PC1={len(a1)} pts -> {safe}_pc{{0,1}}.ply")

        if found >= requested:
            break

    missing = requested - found
    if missing:
        print(f"WARNING: {len(missing)} id(s) not found: {sorted(missing)}")
    print(f"Done: {len(found)}/{len(requested)} cylinders dumped to {out_dir}")

    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()