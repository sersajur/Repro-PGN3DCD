"""
Inspect a PGN3DCD checkpoint file and print its contents.
Optionally plot validation mIoU / mIoU_ch progression across epochs.

Usage:
    python pgn3dcd_experiments/inspect_checkpoint.py /path/to/checkpoint.pt
    python pgn3dcd_experiments/inspect_checkpoint.py /path/to/checkpoint.pt --plot val_miou.png
    python pgn3dcd_experiments/inspect_checkpoint.py  # uses CHECKPOINT_PATH env var

Inside Docker:
    docker compose --profile local run --rm local-cpu-eval \
        pgn3dcd_experiments/inspect_checkpoint.py /data/SiamEncFusionKPConv.pt
"""
import argparse
import sys
import os
import json
import torch
from omegaconf import OmegaConf


def sizeof_fmt(num_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def count_parameters(state_dict):
    total = sum(p.numel() for p in state_dict.values())
    trainable_size = sum(p.numel() * p.element_size() for p in state_dict.values())
    return total, trainable_size


def inspect_checkpoint(path):
    print(f"Checkpoint: {path}")
    print(f"File size:  {sizeof_fmt(os.path.getsize(path))}")
    print("=" * 70)

    ckp = torch.load(path, map_location="cpu")

    # --- Top-level keys ---
    print(f"\nTop-level keys: {sorted(ckp.keys())}")

    # --- Run config ---
    if "run_config" in ckp:
        print("\n" + "=" * 70)
        print("RUN CONFIG")
        print("=" * 70)
        cfg = OmegaConf.create(ckp["run_config"])
        print(OmegaConf.to_yaml(cfg))

    # --- Model weights ---
    if "models" in ckp:
        print("=" * 70)
        print("MODEL WEIGHTS")
        print("=" * 70)
        for name, state_dict in ckp["models"].items():
            num_params, size = count_parameters(state_dict)
            print(f"  {name:20s} — {num_params:,} parameters ({sizeof_fmt(size)})")

    # --- Training stats ---
    if "stats" in ckp:
        print("\n" + "=" * 70)
        print("TRAINING STATS")
        print("=" * 70)
        for stage, records in ckp["stats"].items():
            print(f"\n  [{stage}] — {len(records)} epoch(s)")
            if records:
                last = records[-1]
                print(f"  Last epoch stats:")
                for k, v in sorted(last.items()):
                    if isinstance(v, float):
                        print(f"    {k}: {v:.6f}")
                    else:
                        print(f"    {k}: {v}")

        # --- Best metrics summary ---
        val_stats = ckp["stats"].get("val", [])
        if val_stats:
            last_val = val_stats[-1]
            best_metrics = {k: v for k, v in last_val.items() if k.startswith("best_")}
            if best_metrics:
                print(f"\n  Best validation metrics (tracked across all epochs):")
                for k, v in sorted(best_metrics.items()):
                    metric_name = k[len("best_"):]
                    epoch_hit = next(
                        (r["epoch"] for r in val_stats if r.get(metric_name) == v),
                        None,
                    )
                    epoch_str = f" (epoch {epoch_hit})" if epoch_hit is not None else ""
                    if isinstance(v, float):
                        print(f"    {k}: {v:.6f}{epoch_str}")
                    else:
                        print(f"    {k}: {v}{epoch_str}")

    # --- Optimizer ---
    if "optimizer" in ckp:
        print("\n" + "=" * 70)
        print("OPTIMIZER")
        print("=" * 70)
        opt_name, opt_state = ckp["optimizer"]
        print(f"  Class: {opt_name}")
        param_groups = opt_state.get("param_groups", [])
        for i, pg in enumerate(param_groups):
            pg_info = {k: v for k, v in pg.items() if k != "params"}
            print(f"  Param group {i}: {pg_info}")

    # --- Schedulers ---
    if "schedulers" in ckp:
        print("\n" + "=" * 70)
        print("SCHEDULERS")
        print("=" * 70)
        for name, (sched_opt, sched_state) in ckp["schedulers"].items():
            print(f"  {name}:")
            print(f"    Config: {sched_opt}")
            print(f"    State:  {sched_state}")

    # --- Dataset properties ---
    if "dataset_properties" in ckp:
        print("\n" + "=" * 70)
        print("DATASET PROPERTIES")
        print("=" * 70)
        props = ckp["dataset_properties"]
        if props:
            print(f"  {json.dumps(props, indent=4, default=str)}")
        else:
            print("  (empty)")

    # --- Model props ---
    if "model_props" in ckp:
        print("\n" + "=" * 70)
        print("MODEL PROPS")
        print("=" * 70)
        for k, v in ckp["model_props"].items():
            print(f"  {k}: {v}")

    # --- Any other keys ---
    known_keys = {"run_config", "models", "stats", "optimizer", "schedulers",
                  "dataset_properties", "model_props"}
    extra_keys = set(ckp.keys()) - known_keys
    if extra_keys:
        print("\n" + "=" * 70)
        print("OTHER KEYS")
        print("=" * 70)
        for k in sorted(extra_keys):
            v = ckp[k]
            print(f"  {k}: {type(v).__name__} — {repr(v)[:200]}")


def plot_metrics(ckp, output_path):
    import matplotlib.pyplot as plt

    stats = ckp.get("stats", {})
    series = [
        ("val",   "miou",    "o", "-"),
        ("val",   "miou_ch", "s", "-"),
        ("train", "miou",    "o", "--"),
        ("train", "miou_ch", "s", "--"),
        ("train", "loss",    "x", ":"),
        ("val",   "loss",    "x", ":"),
    ]

    fig, (ax_metric, ax_loss) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for stage, key, marker, ls in series:
        records = stats.get(stage, [])
        if not records or key not in records[0]:
            print(f"  ! '{stage}.{key}' missing, skipping")
            continue
        epochs = [r.get("epoch") for r in records]
        values = [r.get(key) for r in records]
        target_ax = ax_loss if key == "loss" else ax_metric
        target_ax.plot(epochs, values, marker=marker, linestyle=ls, label=f"{stage}_{key}")

    ax_metric.set_ylabel("mIoU")
    ax_metric.set_title("Train / Val mIoU progression")
    ax_metric.grid(True, alpha=0.3)
    ax_metric.legend()

    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("checkpoint", help="Path to checkpoint .pt")
    parser.add_argument("--plot", metavar="PATH",
                        help="If given, save val mIoU/mIoU_ch progression plot to PATH")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    inspect_checkpoint(args.checkpoint)

    if args.plot:
        ckp = torch.load(args.checkpoint, map_location="cpu")
        plot_metrics(ckp, args.plot)
