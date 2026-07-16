"""Shared input construction for the prior-mask tests.

Both `get_mask_v2` implementations take the same arguments — the baseline
`SiamEncFusionKPConv` (`SIFT_SKP_double_pc.py`) and `SiamKPConvWithPriorUncertainty`
— so both test modules build their inputs here.

Neither implementation touches `self`, so both are exercised as unbound functions:
no config/dataset/model construction needed.

`make_inputs` produces inputs yielding *exact* D and C values, so the tests can
assert closed-form numbers rather than regression snapshots:
  - D = clamp(dist / POS_TH, 0, 1)       -> place the pair `D_target * POS_TH` apart
  - C = clamp(|mean(dRGB)| / F_TH, 0, 1) -> offset every colour channel by
    `C_target * F_TH` (the mean over channels then equals the offset)

POS_TH / F_TH mirror the literals hardcoded in both models' `forward()`.
Per the paper, each is 2x the dataset average of the quantity it normalises.
"""
import torch

POS_TH = 3.0
F_TH = 0.6


def make_inputs(d_targets, c_targets):
    """Build (pos_1, pos_2, f_1, f_2, knn_idx) yielding exactly the given D/C."""
    n = len(d_targets)
    pos_1 = torch.zeros(n, 3)
    pos_2 = torch.zeros(n, 3)
    pos_2[:, 0] = torch.tensor(d_targets) * POS_TH  # distance along x

    f_1 = torch.zeros(n, 3)
    f_2 = torch.tensor(c_targets).float().unsqueeze(1).repeat(1, 3) * F_TH

    return (pos_1, pos_2, f_1, f_2) + (knn_identity(n),)


def knn_identity(n):
    """torch_geometric.nn.knn layout: row 0 = query idx, row 1 = neighbour idx.

    Only row 1 is read by get_mask_v2; point i is matched to point i.
    """
    idx = torch.arange(n)
    return torch.stack([idx, idx], dim=0)
