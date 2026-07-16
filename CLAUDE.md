# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of **PGN3DCD** (prior-knowledge-guided 3D point-cloud change detection) built on the
**Torch-Points3D** framework (`torch_points3d/`). The task is binary/multi-class *change
segmentation* between two point clouds captured at different times (the **HKCD** dataset).
The active research line (branch `dev/prior-uncertainity-modeling`) replaces PGN3DCD's fixed
50/50 geometric+colour prior fusion with **Bayesian inverse-variance weighting** and feeds
per-point uncertainty as an extra input channel — see [prior_uncertainty_audit.md](pgn3dcd_doc/prior_uncertainty_audit.md)
(in Ukrainian) for the running experiment log.

## Running things

Everything runs through **Hydra** configs and, in practice, through **Docker Compose** (Python
3.8 / torch 1.8 / CUDA — pinned in the image; the host is not expected to have a working env).

Each experiment is a `train*.py` / `eval*.py` script paired 1:1 with a `conf/config*.yaml` /
`conf/eval*.yaml` file via `@hydra.main(config_name=...)`. To add a variant you clone both.

| Script | Config | Model |
|---|---|---|
| `trainSiamKPConv.py` / `eval_SiamKPConv.py` | `configSiamKPConv` / `evalSiamKPConv` | PGN3DCD baseline (`SiamEncFusionKPConv`) |
| `trainSiamKPConvWithPriorUncertainty.py` / `eval_SiamKPConvWithPriorUncertainty.py` | `...WithPriorUncertainty` | uncertainty variant |
| `eval_simple.py` | *(none — env vars)* | standalone loop, no Trainer/Hydra |

Hydra overrides are passed on the CLI, e.g.
`python3 trainSiamKPConv.py output_dir=/output/run1 training.epochs=5`.

### Docker Compose (primary workflow)

Services are grouped by `profiles: [local | gcp-cpu | gcp-gpu]`; volumes come from `.env`
(`cp .env.example .env` first — `CHECKPOINTS_DIR`, `DATA_PATH`, `OUTPUT_PATH`, `PROJECT_PATH`).
The repo is bind-mounted to `/tp3d` (WORKDIR/PYTHONPATH), checkpoints to `/checkpoints` (ro),
data to `/data`, results to `/output`.

```bash
docker compose --profile gcp-gpu run --rm gcp-gpu-train-uncertainty
docker compose --profile local  run --rm local-eval-uncertainty
```

The `*-fast*` services run tiny subset configs (`Train_debug`, few epochs) for smoke tests.

### `eval_simple.py`

A dependency-light eval path (no Trainer, no Hydra) driven entirely by env vars
(`CHECKPOINT_PATH`, `DATA_DIR`, `OUTPUT_DIR`, `WEIGHT_NAME`, `BATCH_SIZE`, `MAX_BATCHES`,
`DEVICE`). It rebuilds the dataset/model from the config baked into the checkpoint's
`run_config` and overrides only the data paths. Use it for quick CPU inference and when the
full Trainer stack is failing for infra reasons.

### Weight selection

Checkpoints store multiple bests (`best_miou`, `best_miou_ch`, `best_acc`, `latest`). Select
via `training.weight_name=` (Hydra) or `WEIGHT_NAME=` (eval_simple). `miou_ch` = mIoU on the
*changed* classes only and is the primary metric for this task.

## Static checks & tests

Tests are `unittest.TestCase` classes and are run with **`unittest`, not pytest** — pytest is
declared nowhere (`requirements.txt`, `pyproject.toml`) and is absent from the Docker image.
Run from the repo root (each test prepends it to `sys.path`), inside the container:

```bash
docker compose --profile gcp-gpu exec -T gcp-gpu-shell python3 -m unittest discover -s test  # full suite
docker compose --profile gcp-gpu exec -T gcp-gpu-shell python3 -m unittest test.test_kpconv -v         # one file
docker compose --profile gcp-gpu exec -T gcp-gpu-shell python3 -m unittest test.test_models.TestClass.test_x  # one test

make staticchecks          # flake8 (error subset only) + mypy torch_points3d
```

`mypy.ini` ignores missing imports and the MinkowskiEngine module.

## Architecture: how a model gets built and run

Torch-Points3D is a **factory + config** framework — you rarely instantiate classes directly:

- **`instantiate_dataset(cfg.data)`** (`datasets/dataset_factory.py`) and
  **`instantiate_model(cfg, dataset)`** (`models/model_factory.py`) read the Hydra config and
  reflectively construct classes by name. The `conf/models/change_detection/*.yaml` and
  `conf/data/change_detection/*.yaml` files define layer shapes (`down_conv`/`up_conv`,
  `FEAT`, etc.) that the model class reads at construction time.
- Change-detection models live in `torch_points3d/models/change_detection/`. The relevant
  inheritance chain is `SiamKPConvWithPriorUncertainty` → `SiamEncFusionKPConv` (defined in
  the confusingly-named `SIFT_SKP_double_pc.py`) → `UnwrappedUnetBasedModel`. The subclass
  overrides only `get_mask_v2` (prior fusion) and `forward`.
- **Model↔module wiring**: model modules pull KPConv conv-block classes by name via
  `getattr(modules_lib, name)`, where `modules_lib` is the *model module itself*. This is why
  model files carry `from torch_points3d.modules.KPConv import *` /
  `...core.base_conv.partial_dense import *` wildcard imports — **do not remove them**, they
  look dead but resolve `KPDualBlock`, `FPModule_PD`, etc.
- **Siamese forward** (see `SiamKPConvWithPriorUncertainty.forward`): two clouds go through
  two encoder stacks; at each level `torch_geometric.nn.knn` matches points across clouds and
  a per-level *difference* feature (`data0.x - data1.x[nn]`) is concatenated in and pushed onto
  a stack for the U-Net decoder. A per-point `mask` (from the prior) multiplicatively gates
  self-attention at every level. Both clouds are predicted (`output0`, `output1`).
- **Trainer**: `trainer_SiamKPConv.py` (`Trainer`) owns the train/eval loop, checkpointing,
  and the tracker; `train*.py`/`eval*.py` just build the config, seed determinism, and call
  `Trainer(cfg).train()` / `.eval()`.
- **Trackers** (`torch_points3d/metrics/`, e.g. `hkCD_tracker.py`) accumulate confusion
  matrices and, when `full_pc=True/full_res=True`, project cylinder-cropped predictions back
  to the full-resolution cloud and write per-area `.ply` predictions + confusion-matrix PNGs.
- **Data**: HKCD clouds are cropped into **cylinders** (`HKCDPairCylinder.py`) for training;
  full-res reconstruction happens in the tracker at eval time.

### Determinism

All `train*.py`/`eval*.py` call `set_deterministic(seed)` and stamp the environment
(`torch_points3d/utils/repro.py`). Note CPU↔GPU bit-exactness is explicitly *not* guaranteed —
only CPU↔CPU and GPU↔GPU stability. Preserve these seeding calls when editing entrypoints.

## Fork-specific fixes to be aware of

This fork patched several CPU/dtype/field-name issues in upstream (see [README.md](README.md)):
PLY label fields may be named either `cd_type` or `scalar_cd_type`; RGB may be 0–255 or already
normalized (the prior-fusion math is sensitive to this — see the audit doc); `get_mask_v2` had a
float/double mismatch. When touching dataset loading or prior math, check both PLY conventions.

## Utility scripts

`pgn3dcd_experiments/` holds analysis/debug tooling (not part of training): `inspect_checkpoint.py`,
`dump_cylinders_by_id.py` (per-cylinder μ/σ²/mask dump — enable via `model._dump_io = True`),
`make_train_subset.py` / `select_subset.py` (build the `Train_debug` fast subset),
`patch_checkpoint_data_config.py`, `pt_to_ply.py`. `compute_priors.py` (repo root) computes and
writes the D/C/μ/σ² prior scalars to PLY for CloudCompare visualization, mirroring the model's math.
