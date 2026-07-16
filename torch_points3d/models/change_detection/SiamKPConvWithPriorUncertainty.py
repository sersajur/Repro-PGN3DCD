"""
SiamKPConvWithPriorUncertainty
==============================
Extension of PGN3DCD's SiamEncFusionKPConv that replaces the fixed
50/50 prior fusion with **Bayesian inverse-variance weighting** and
feeds the per-point total uncertainty σ²_total as an additional input
channel to the encoder.

Key changes vs. the base model
------------------------------
1.  ``get_mask_v2``  →  ``get_mask_v2`` returns *(μ, mask, σ²_total)*
    instead of *(input, mask)*.
    - Bernoulli variance:  σ²_D = D·(1−D),  σ²_C = C·(1−C)
    - Inverse-variance weights: w_D = 1/σ²_D,  w_C = 1/σ²_C
    - Fused mean:  μ = (w_D·D + w_C·C) / (w_D + w_C)
    - Fused variance:  σ²_total = 1 / (w_D + w_C)
    - Confidence-modulated mask:
        confidence = 1 − σ²_total / 0.25
        mask = confidence · μ + (1 − confidence) · 0.5

2.  ``forward``  →  overwrites channel 0 (the ``add_ones`` bias channel)
    with μ — same slot the parent model uses for its 0.5·D + 0.5·C
    fusion — and appends σ²_total as a single extra feature channel.
    Input tensor goes from shape (N, FEAT) = (N, 4) to (N, FEAT+1) = (N, 5).

3.  Config:  FEAT = 4  (was 3) so that FEAT+1 = 5 covers
    [μ, R, G, B, σ²_total].
    The ones (bias) channel is reused as the μ slot instead of being
    kept as a constant — BatchNorm provides bias-emulation downstream,
    so the slot is better spent on an informative signal.

All changes are **non-parametric** — zero new learnable parameters.
"""

from typing import Any
import torch

# These wildcard imports are required: model_factory.py passes this module
# as `modules_lib` and the base class looks up conv block classes (e.g.
# KPDualBlock, FPModule_PD) via getattr(modules_lib, name).
from torch_points3d.modules.KPConv import *          # noqa: F403
from torch_points3d.core.base_conv.partial_dense import *  # noqa: F403

from torch_points3d.models.change_detection.SIFT_SKP_double_pc import (
    SiamEncFusionKPConv,
)
from torch_points3d.core.common_modules.base_modules import Identity

# Minimum variance clamp — avoids division-by-zero in inverse weighting
_EPS = 1e-6

# Maximum Bernoulli variance (at p = 0.5): used to normalise confidence
_MAX_VAR = 0.25


class SiamKPConvWithPriorUncertainty(SiamEncFusionKPConv):
    """Drop-in replacement for SiamEncFusionKPConv with uncertainty-aware prior fusion."""

    # ------------------------------------------------------------------
    # Prior mask computation  (overrides parent)
    # ------------------------------------------------------------------
    def get_mask_v2(self, pos_1, pos_2, f_1, f_2, pos_th, f_th, knearest_idx_2):
        """Bayesian inverse-variance weighted fusion of geometric (D)
        and colour (C) change priors.

        Returns
        -------
        mu : Tensor (N, 1)
            Fused prior mean per point.
        mask : Tensor (N, 1)
            Confidence-modulated mask fed into the attention layers.
        var_total : Tensor (N, 1)
            Fused variance per point — appended as extra input channel.
        """
        pos_1, pos_2 = pos_1.float(), pos_2.float()
        f_1, f_2 = f_1.float(), f_2.float()
        pos_th = pos_th.float() if torch.is_tensor(pos_th) else pos_th
        f_th = f_th.float() if torch.is_tensor(f_th) else f_th

        knearest_idx_2 = knearest_idx_2.reshape(knearest_idx_2.shape[0], -1, 1)

        # ── Geometric distance prior D ∈ [0, 1] ──────────────────────
        nearest_pos = pos_1[knearest_idx_2[1, :, :], :]
        pos_2_ex = pos_2.unsqueeze(1).repeat(1, nearest_pos.shape[1], 1)
        dis_mat = torch.mean(
            torch.sqrt(torch.sum(torch.square(pos_2_ex - nearest_pos), dim=-1)),
            dim=-1,
        ).view(-1, 1).float()
        D = torch.where(dis_mat > pos_th, torch.ones_like(dis_mat), dis_mat / pos_th)

        # ── Colour difference prior C ∈ [0, 1] ───────────────────────
        nearest_f = f_1[knearest_idx_2[1, :, :], :]
        f_2_ex = f_2.unsqueeze(1).repeat(1, nearest_f.shape[1], 1)
        f_mat = torch.abs(
            torch.mean(f_2_ex - nearest_f, dim=-1).view(-1, 1)
        ).float()
        C = torch.where(f_mat > f_th, torch.ones_like(f_mat), f_mat / f_th)

        # ── Bayesian inverse-variance fusion ──────────────────────────
        var_D = (D * (1.0 - D)).clamp(min=_EPS)
        var_C = (C * (1.0 - C)).clamp(min=_EPS)

        w_D = 1.0 / var_D
        w_C = 1.0 / var_C
        w_sum = w_D + w_C

        mu = (w_D * D + w_C * C) / w_sum            # fused mean
        var_total = 1.0 / w_sum                     # fused variance

        # ── Confidence-modulated mask ─────────────────────────────────
        confidence = (1.0 - var_total / _MAX_VAR).clamp(min=0.0, max=1.0)
        mask = confidence * mu + (1.0 - confidence) * 0.5

        return mu, mask, var_total

    # ------------------------------------------------------------------
    # Forward pass  (overrides parent)
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs) -> Any:
        """Identical to the base forward except:
        - Unpacks 3 values from ``get_mask_v2`` (mu, mask, var).
        - Replaces channel 0 with μ (instead of raw 0.5-blend).
        - Concatenates σ²_total as an extra input channel.
        """
        from torch_geometric.nn import knn as _knn  # avoid circular at module level

        stack_down = []
        pos_th1, pos_th2 = 3.0, 3.0
        f_th1, f_th2 = 0.6, 0.6

        data0 = self.input0
        data1 = self.input1
        color0, color1 = data0.x[:, 1:], data1.x[:, 1:]

        nn_list00 = _knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
        nn_list10 = _knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)

        # ── Uncertainty-aware prior fusion ────────────────────────────
        mu0, mask0, var0 = self.get_mask_v2(
            data1.pos, data0.pos, color1, color0, pos_th1, f_th1, nn_list00
        )
        mu1, mask1, var1 = self.get_mask_v2(
            data0.pos, data1.pos, color0, color1, pos_th2, f_th2, nn_list10
        )

        data0.mask = mask0
        data1.mask = mask1

        # Overwrite ones (bias channel) with μ; append σ²_total as extra channel.
        # Input: [ones, R, G, B] → [μ, R, G, B, σ²_total]
        # Frees the wasted ones-channel (which only emulated a bias term that
        # BatchNorm already provides downstream) and lets the first KPConv layer
        # spend all of its in-channel weights on informative signals.
        data0.x[:, 0] = mu0.squeeze(-1)
        data1.x[:, 0] = mu1.squeeze(-1)
        data0.x = torch.cat([data0.x, var0], dim=1)
        data1.x = torch.cat([data1.x, var1], dim=1)

        # ── Opt-in per-point capture for the by-id cylinder dump (zero cost off) ──
        # mu/var/mask are local to forward and discarded otherwise; batch aligns
        # them to points. Everything else the dump needs (pos via scene+idx, rgb,
        # gt) comes from the batch, so only these four (×PC0/PC1) are captured.
        # Enable via `model._dump_io = True`; read from `model._dumped_io`.
        if getattr(self, "_dump_io", False):
            _cpu = lambda t: t.detach().cpu().clone()   # noqa: E731
            self._dumped_io = {
                "mu0": _cpu(mu0), "var0": _cpu(var0), "mask0": _cpu(mask0), "batch0": _cpu(data0.batch),
                "mu1": _cpu(mu1), "var1": _cpu(var1), "mask1": _cpu(mask1), "batch1": _cpu(data1.batch),
            }

        # ── Encoder / decoder (unchanged from base) ──────────────────
        data0 = self.down_modules_1[0](data0, precomputed=self.pre_computed)
        data1 = self.down_modules_2[0](data1, precomputed=self.pre_computed_target)
        data0.x = self.self_att_list[0](data0.x, None, False, mask0)
        data1.x = self.self_att_list[0](data1.x, None, False, mask1)

        diff0 = data0.clone()
        diff0.x = data0.x - data1.x[nn_list00[1, :], :]
        diff1 = data1.clone()
        diff1.x = data1.x - data0.x[nn_list10[1, :], :]

        stack_down.append([diff0, diff1])
        data0.x = torch.cat((data0.x, diff0.x), axis=1)
        data1.x = torch.cat((data1.x, diff1.x), axis=1)

        for i in range(1, len(self.down_modules_1) - 1):
            data0 = self.down_modules_1[i](data0, precomputed=self.pre_computed)
            data1 = self.down_modules_2[i](data1, precomputed=self.pre_computed_target)
            mask0 = data0.mask
            mask1 = data1.mask
            data0.x = self.self_att_list[i](data0.x, None, False, mask0)
            data1.x = self.self_att_list[i](data1.x, None, False, mask1)

            nn_list00 = _knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
            nn_list10 = _knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)

            diff0 = data0.clone()
            diff0.x = data0.x - data1.x[nn_list00[1, :], :]
            diff1 = data1.clone()
            diff1.x = data1.x - data0.x[nn_list10[1, :], :]

            data0.x = torch.cat((data0.x, diff0.x), axis=1)
            data1.x = torch.cat((data1.x, diff1.x), axis=1)
            stack_down.append([diff0, diff1])

        # Bottleneck
        data0 = self.down_modules_1[-1](data0, precomputed=self.pre_computed)
        data1 = self.down_modules_2[-1](data1, precomputed=self.pre_computed_target)
        mask0 = data0.mask
        mask1 = data1.mask
        data0.x = self.self_att_list[-1](data0.x, None, False, mask0)
        data1.x = self.self_att_list[-1](data1.x, None, False, mask1)

        nn_list00 = _knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
        nn_list10 = _knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)

        diff0 = data0.clone()
        diff0.x = data0.x - data1.x[nn_list00[1, :], :]
        diff1 = data1.clone()
        diff1.x = data1.x - data0.x[nn_list10[1, :], :]

        data0.x = torch.cat((data0.x, diff0.x), axis=1)
        data1.x = torch.cat((data1.x, diff1.x), axis=1)

        innermost = False
        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append([diff0, diff1])
            data0 = self.inner_modules[0](data0)
            data1 = self.inner_modules[0](data1)
            innermost = True

        for i in range(len(self.up_modules_1)):
            if i == 0 and innermost:
                diff0, diff1 = stack_down.pop()
                data0 = self.up_modules_1[i]((data0, diff0))
                data1 = self.up_modules_2[i]((data1, diff1))
            else:
                diff0, diff1 = stack_down.pop()
                data0 = self.up_modules_1[i](
                    (data0, diff0), precomputed=self.upsample_target
                )
                data1 = self.up_modules_2[i](
                    (data1, diff1), precomputed=self.upsample_target
                )

        last_feature0 = data0.x
        last_feature1 = data1.x
        if self._use_category:
            self.output0 = self.FC_layer(last_feature0, self.category)
            self.output1 = self.FC_layer(last_feature1, self.category)
        else:
            self.output0 = self.FC_layer(last_feature0)
            self.output1 = self.FC_layer(last_feature1)

        if self.labels0 is not None and self.labels1 is not None:
            self.compute_loss()

        self.data_visual0 = self.input0
        self.data_visual0.pred = torch.max(self.output0, -1)[1]
        self.data_visual1 = self.input1
        self.data_visual1.pred = torch.max(self.output1, -1)[1]

        self.output = [self.output0, self.output1]

        return self.output0, self.output1
