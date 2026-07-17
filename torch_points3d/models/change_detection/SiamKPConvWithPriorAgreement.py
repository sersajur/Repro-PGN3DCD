"""
SiamKPConvWithPriorAgreement
============================
FIX A for SiamKPConvWithPriorUncertainty (see
pgn3dcd_doc/prior_mask_math_uncertainty.md §8): the Bernoulli
inverse-variance fusion measured *extremeness* of the priors, not their
*reliability* — contradiction between sources collapsed to a confident
neutral, confidence never fell below 0.5, and the mask stopped
suppressing confidently-unchanged points.

This variant replaces the uncertainty model with the quantity the design
intent actually describes — **source agreement**:

    mu   = 0.5·D + 0.5·C          (baseline fusion — robust, bounded)
    d    = |D − C|                (disagreement in [0, 1])
    mask = (1 − d)·mu + d·0.5     (agree → follow; conflict → neutral)

Properties (locked in test/test_prior_agreement_mask.py):
  - when the sources agree (D == C), mask == baseline's 0.5·D + 0.5·C
    exactly — suppression of unchanged points is restored;
  - conflict (D=1, C=0) → mask = 0.5 with d = 1, honestly flagged, and
    for the first time distinguishable from ambiguity (D=C=0.5 → d = 0);
  - ∂mask/∂D ∈ [0, 1] — bounded sensitivity, no 0/0 singularity;
  - the (mu, d) pair is a lossless re-parametrisation of (D, C): the
    network receives both dimensions where the baseline keeps one.

Input layout is unchanged vs the uncertainty variant — [mu, R, G, B, d],
FEAT = 4 — so the architecture configs differ only in the class name.
The parent model is intentionally left untouched: checkpoint
SiamKPConvWithPriorUncertainty_5F.pt must keep evaluating with the
math it was trained under.

All changes are **non-parametric** — zero new learnable parameters.
"""

from typing import Any  # noqa: F401  (kept for parity with sibling models)
import torch

# These wildcard imports are required: model_factory.py passes this module
# as `modules_lib` and the base class looks up conv block classes (e.g.
# KPDualBlock, FPModule_PD) via getattr(modules_lib, name).
from torch_points3d.modules.KPConv import *          # noqa: F403
from torch_points3d.core.base_conv.partial_dense import *  # noqa: F403

from torch_points3d.models.change_detection.SiamKPConvWithPriorUncertainty import (
    SiamKPConvWithPriorUncertainty,
)


class SiamKPConvWithPriorAgreement(SiamKPConvWithPriorUncertainty):
    """Agreement-gated prior fusion; inherits the uncertainty variant's
    forward (which already routes the third return value into the extra
    input channel) and overrides only the mask computation."""

    def get_mask_v2(self, pos_1, pos_2, f_1, f_2, pos_th, f_th, knearest_idx_2):
        """Agreement-gated fusion of geometric (D) and colour (C) priors.

        Returns
        -------
        mu : Tensor (N, 1)
            Fused prior mean (fixed 50/50 — identical to the baseline).
        mask : Tensor (N, 1)
            Attention gate: mu where the sources agree, pulled to the
            neutral 0.5 in proportion to their disagreement.
        d : Tensor (N, 1)
            Disagreement |D − C| — appended as the extra input channel
            (slot the uncertainty variant used for σ²_total).
        """
        pos_1, pos_2 = pos_1.float(), pos_2.float()
        f_1, f_2 = f_1.float(), f_2.float()
        pos_th = pos_th.float() if torch.is_tensor(pos_th) else pos_th
        f_th = f_th.float() if torch.is_tensor(f_th) else f_th

        knearest_idx_2 = knearest_idx_2.reshape(knearest_idx_2.shape[0], -1, 1)

        # ── Geometric distance prior D ∈ [0, 1]  (same as parent) ─────
        nearest_pos = pos_1[knearest_idx_2[1, :, :], :]
        pos_2_ex = pos_2.unsqueeze(1).repeat(1, nearest_pos.shape[1], 1)
        dis_mat = torch.mean(
            torch.sqrt(torch.sum(torch.square(pos_2_ex - nearest_pos), dim=-1)),
            dim=-1,
        ).view(-1, 1).float()
        D = torch.where(dis_mat > pos_th, torch.ones_like(dis_mat), dis_mat / pos_th)

        # ── Colour difference prior C ∈ [0, 1]  (same as parent) ──────
        nearest_f = f_1[knearest_idx_2[1, :, :], :]
        f_2_ex = f_2.unsqueeze(1).repeat(1, nearest_f.shape[1], 1)
        f_mat = torch.abs(
            torch.mean(f_2_ex - nearest_f, dim=-1).view(-1, 1)
        ).float()
        C = torch.where(f_mat > f_th, torch.ones_like(f_mat), f_mat / f_th)

        # ── Agreement-gated fusion ─────────────────────────────────────
        mu = 0.5 * D + 0.5 * C
        d = (D - C).abs()
        mask = (1.0 - d) * mu + d * 0.5

        return mu, mask, d
