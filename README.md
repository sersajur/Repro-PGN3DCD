# Repro-PGN3DCD

A fork of [PGN3DCD](https://github.com/zhanwenxiao/PGN3DCD) configured for reproducing the original evaluation results on the HKCD dataset. See [README_PGN3DCD.md](README_PGN3DCD.md) for the original documentation.

## What this fork adds

- **Dockerized environment** (CPU) with all dependencies pinned
- **Standalone evaluation script** (`eval_simple.py`) — no Trainer/Hydra overhead
- **Bug fixes** for running inference on CPU and with varying PLY field naming

## Repository structure

```
.
├── docker/
│   ├── Dockerfile.cpu          # Docker image for CPU inference
│   ├── Dockerfile.gpu          # Docker image for GPU inference
│   ├── install_system.sh       # System-level dependencies (from upstream)
│   └── install_python.sh       # Python/PyTorch installation (from upstream)
├── conf/
│   ├── evalSiamKPConv.yaml     # Eval config (paths updated for container use)
│   └── data/change_detection/
│       └── HKCDPair.yaml       # HKCD dataset config (paths updated for container use)
├── torch_points3d/
│   ├── datasets/change_detection/
│   │   └── HKCDPairCylinder.py # Fixed: support both 'cd_type' and 'scalar_cd_type' PLY fields
│   ├── models/change_detection/
│   │   └── SIFT_SKP_double_pc.py  # Fixed: float/double dtype mismatch in get_mask_v2
│   └── metrics/
│       └── hkCD_tracker.py     # Fixed: skip None entries in merge_avg_mappings
├── eval_simple.py              # Standalone eval script (configurable via env vars)
├── docker-compose.yml          # Compose config for local CPU evaluation
├── .env.example                # Template for local paths configuration
├── .dockerignore               # Excludes .git, checkpoints, data from Docker build
└── README_PGN3DCD.md           # Original upstream README
```

## File tree inside the Docker container

```
/ (container root)
├── venv/                       # Python 3.8 virtual environment with all dependencies
├── tp3d/                       # Project source code (WORKDIR, PYTHONPATH)
│   ├── eval_simple.py          # Entrypoint
│   ├── torch_points3d/         # Model, dataset, and metrics code
│   └── conf/                   # Hydra configs (used by dataset factory)
├── data/                       # Mounted volume with dataset and checkpoint
│   ├── SiamEncFusionKPConv.pt  # Pre-trained model checkpoint
│   └── HKCD/
│       ├── Train/              # Training split (used for preprocessing reference)
│       ├── Val/                # Validation split
│       └── Test/               # Test split (evaluation target)
└── output/                     # Mounted volume for evaluation results
    ├── res.txt                 # Metrics summary (per-area, average, cumulative)
    ├── cm.png                  # Confusion matrix (pointCloud0)
    ├── cm2.png                 # Confusion matrix (pointCloud1)
    └── <area_name>/            # Per-area predictions
        ├── pointCloud0.ply     # Predictions for time epoch 0
        └── pointCloud1.ply     # Predictions for time epoch 1
```

## Quick start

### 1. Clone and configure

```bash
git clone https://github.com/sersajur/Repro-PGN3DCD.git
cd Repro-PGN3DCD
cp .env.example .env
# Edit .env: set DATA_PATH to directory containing SiamEncFusionKPConv.pt and HKCD/
```

### 2. Run evaluation (CPU)

```bash
docker compose build eval-cpu
docker compose run --rm eval-cpu
```

Results will appear in `./output/`.

### 3. Configuration

All parameters are configurable via environment variables (set in `.env` or pass directly):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_PATH` | *(required)* | Host path to data directory |
| `OUTPUT_PATH` | `./output` | Host path for results |
| `CHECKPOINT_PATH` | `/data/SiamEncFusionKPConv.pt` | Container path to checkpoint |
| `DATA_DIR` | `/data/HKCD` | Container path to dataset |
| `OUTPUT_DIR` | `/output` | Container path for results |
| `WEIGHT_NAME` | `miou` | Weights to load: `miou`, `miou_ch`, `acc`, `latest` |
| `BATCH_SIZE` | `10` | Batch size |
| `NUM_WORKERS` | `2` | DataLoader workers |
| `MAX_BATCHES` | `0` | Limit batches (0 = all, useful for quick tests) |
| `DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
