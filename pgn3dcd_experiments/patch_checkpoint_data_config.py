"""
Inject defaults for fields BaseSiameseDataset reads from run_config.data
but never uses for change detection (num_points, tau_*, *_thresh, nameInPly).
Old checkpoints lack these fields; eval_SiamKPConv.py crashes restoring the
dataset from the checkpoint's frozen run_config without them.

Usage:
    python pgn3dcd_experiments/patch_checkpoint_data_config.py INPUT OUTPUT
"""
import argparse
import torch

# Taken from HKCDPair.yaml
DEFAULTS = {
    "num_points": 0,
    "tau_1": 0.1,
    "tau_2": 0.05,
    "trans_thresh": 0.15,
    "rot_thresh": 4,
    "nameInPly": "params",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("input", help="Input checkpoint path (.pt)")
    parser.add_argument("output", help="Output checkpoint path (.pt)")
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu")
    data_cfg = ckpt["run_config"]["data"]

    for key, val in DEFAULTS.items():
        if key not in data_cfg:
            data_cfg[key] = val
            print(f"  + {key} = {val!r}")

    torch.save(ckpt, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()