"""
Lightweight evaluation script that loads a checkpoint and runs inference
on the test set without using the full Trainer.

Configuration via environment variables:
    CHECKPOINT_PATH  - path to .pt file (default: /data/SiamEncFusionKPConv.pt)
    DATA_DIR         - path to HKCD dataset root (default: /data/HKCD)
    OUTPUT_DIR       - where to save results (default: /output)
    WEIGHT_NAME      - which weights to load: miou, miou_ch, acc, latest (default: miou)
    BATCH_SIZE       - batch size (default: 10)
    NUM_WORKERS      - dataloader workers (default: 2)
    MAX_BATCHES      - limit batches for quick test, 0 = all (default: 0)
    DEVICE           - auto, cpu, or cuda (default: auto)
"""
import warnings
warnings.filterwarnings("ignore")

import os
import copy
import torch
import logging
import numpy as np
from omegaconf import OmegaConf

from torch_points3d.datasets.dataset_factory import instantiate_dataset
from torch_points3d.models.model_factory import instantiate_model
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.utils.colors import COLORS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    # --- Configuration from env vars ---
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "/data/SiamEncFusionKPConv.pt")
    data_dir = os.environ.get("DATA_DIR", "/data/HKCD")
    output_dir = os.environ.get("OUTPUT_DIR", "/output")
    weight_name = os.environ.get("WEIGHT_NAME", "miou")
    batch_size = int(os.environ.get("BATCH_SIZE", "10"))
    num_workers = int(os.environ.get("NUM_WORKERS", "2"))
    max_batches = int(os.environ.get("MAX_BATCHES", "0"))
    device_str = os.environ.get("DEVICE", "auto")

    # --- Load checkpoint ---
    log.info("Loading checkpoint from %s", checkpoint_path)
    ckp = torch.load(checkpoint_path, map_location="cpu")

    run_config = OmegaConf.create(ckp["run_config"])

    # Override data paths to match container volumes
    run_config.data.dataTrainFile = os.path.join(data_dir, "Train/")
    run_config.data.dataValFile = os.path.join(data_dir, "Val/")
    run_config.data.dataTestFile = os.path.join(data_dir, "Test/")
    run_config.data.preprocessed_dir = os.path.join(data_dir, "preprocessed/")

    # Defaults required by BaseSiameseDataset but unused for change detection
    for key, val in [("num_points", -1), ("tau_1", 0.1), ("tau_2", 0.05),
                     ("trans_thresh", 0.1), ("rot_thresh", 5.0), ("nameInPly", "params")]:
        if key not in run_config.data:
            run_config.data[key] = val

    # --- Create dataset and model ---
    log.info("Creating dataset...")
    dataset = instantiate_dataset(run_config.data)

    log.info("Creating model...")
    model = instantiate_model(copy.deepcopy(run_config), dataset)

    # Load state dict
    models = ckp["models"]
    key = f"best_{weight_name}"
    if key in models:
        state_dict = models[key]
        log.info("Loading weights: %s", key)
    else:
        state_dict = models["latest"]
        log.info("Weight '%s' not found, loading 'latest'", weight_name)
    model.load_state_dict(state_dict, strict=False)

    dataset.create_dataloaders(
        model=model,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        precompute_multi_scale=False,
    )

    # --- Device ---
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    log.info("Device: %s", device)
    model = model.to(device)
    model.eval()

    # --- Run inference on test set ---
    if not dataset.has_test_loaders:
        log.error("No test data found!")
        return

    os.makedirs(output_dir, exist_ok=True)

    tracker = dataset.get_tracker(False, False, full_pc=True, full_res=True)

    for loader in dataset.test_dataloaders:
        stage_name = loader.dataset.name
        log.info("Evaluating on: %s", stage_name)
        tracker.reset(stage_name)

        with Ctq(loader) as tq_loader:
            for i, data in enumerate(tq_loader):
                if max_batches and i >= max_batches:
                    break
                with torch.no_grad():
                    model.set_input(data, device)
                    model.forward()
                    tracker.track(model, data=data, full_pc=True, full_res=True)
                tq_loader.set_postfix(**tracker.get_metrics(), color=COLORS.TEST_COLOR)

        tracker.finalise(full_pc=True, full_res=True, save_pc=True, name_test="", saving_path=output_dir)
        tracker.print_summary()


if __name__ == "__main__":
    main()
