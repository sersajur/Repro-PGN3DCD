"""Analytical checks on the baseline PGN3DCD prior mask.

Target: `SiamEncFusionKPConv.get_mask_v2` (`SIFT_SKP_double_pc.py:416-433`) — the
fixed 50/50 fusion `p = 0.5*D + 0.5*C` that the uncertainty variant replaces.

These lock in the claims written up in `pgn3dcd_doc/prior_mask_math_baseline.md`.
See `prior_mask_common.py` for how exact D/C values are constructed, and
`test_prior_uncertainty_mask.py` for the same treatment of the uncertainty variant.
"""
import os
import sys
import unittest

import torch

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from test.prior_mask_common import make_inputs, knn_identity, POS_TH, F_TH

from torch_points3d.models.change_detection.SIFT_SKP_double_pc import SiamEncFusionKPConv
from torch_points3d.models.change_detection.SiamKPConvWithPriorUncertainty import (
    SiamKPConvWithPriorUncertainty,
)


def _fuse(d_targets, c_targets):
    """Baseline fusion -> (input, mask), each flattened to (N,)."""
    pos_1, pos_2, f_1, f_2, knn_idx = make_inputs(d_targets, c_targets)
    inp, mask = SiamEncFusionKPConv.get_mask_v2(
        None, pos_1, pos_2, f_1, f_2, POS_TH, F_TH, knn_idx
    )
    return inp, mask


def _p(d_targets, c_targets):
    """Just the fused prior, flattened."""
    return _fuse(d_targets, c_targets)[0].view(-1)


class TestBaselinePriorComputation(unittest.TestCase):
    """Section 1 of the doc: how D and C are built."""

    def test_geometric_prior_is_distance_over_threshold(self):
        # C pinned to 0 so p == 0.5*D, isolating the geometric term.
        # dist = D_target * 3.0  ->  D = dist / 3.0 = D_target
        p = _p([0.0, 0.25, 0.5, 1.0], [0.0, 0.0, 0.0, 0.0])
        for got, d in zip(p.tolist(), [0.0, 0.25, 0.5, 1.0]):
            self.assertAlmostEqual(got, 0.5 * d, places=5)

    def test_geometric_prior_saturates_above_threshold(self):
        """Beyond pos_th the prior clamps to 1 rather than growing unbounded."""
        pos_1 = torch.zeros(2, 3)
        pos_2 = torch.zeros(2, 3)
        pos_2[:, 0] = torch.tensor([POS_TH, 4 * POS_TH])  # exactly at, and far past

        f = torch.zeros(2, 3)  # C = 0
        inp, _ = SiamEncFusionKPConv.get_mask_v2(
            None, pos_1, pos_2, f, f, POS_TH, F_TH, knn_identity(2)
        )
        # p = 0.5*D + 0.5*0  ->  D = 1 in both cases (not 1 and 4)
        self.assertAlmostEqual(inp.view(-1)[0].item(), 0.5, places=5)
        self.assertAlmostEqual(inp.view(-1)[1].item(), 0.5, places=5)

    def test_colour_prior_is_mean_channel_difference_over_threshold(self):
        # D pinned to 0 so p == 0.5*C, isolating the colour term.
        p = _p([0.0, 0.0, 0.0, 0.0], [0.0, 0.25, 0.5, 1.0])
        for got, c in zip(p.tolist(), [0.0, 0.25, 0.5, 1.0]):
            self.assertAlmostEqual(got, 0.5 * c, places=5)

    def test_colour_prior_saturates_above_threshold(self):
        pos = torch.zeros(1, 3)  # D = 0
        f_1 = torch.zeros(1, 3)
        f_2 = torch.full((1, 3), 5 * F_TH)  # way past the threshold
        inp, _ = SiamEncFusionKPConv.get_mask_v2(
            None, pos, pos, f_1, f_2, POS_TH, F_TH, knn_identity(1)
        )
        # p = 0.5*0 + 0.5*C  ->  C = 1 (not 5)
        self.assertAlmostEqual(inp.view(-1)[0].item(), 0.5, places=5)


class TestBaselineFusion(unittest.TestCase):
    """Sections 1 and 3: the fusion itself."""

    def test_fusion_is_fifty_fifty(self):
        cases = [(0.0, 1.0), (0.2, 0.6), (0.99, 0.5), (1.0, 0.0), (0.25, 0.30)]
        d, c = [x[0] for x in cases], [x[1] for x in cases]
        p = _p(d, c)
        for got, (di, ci) in zip(p.tolist(), cases):
            self.assertAlmostEqual(got, 0.5 * di + 0.5 * ci, places=5)

    def test_input_and_mask_are_the_same_object(self):
        """`mask = input` in the source — not a copy. The prior feeds the input
        channel and gates attention with the identical tensor."""
        inp, mask = _fuse([0.3], [0.7])
        self.assertIs(inp, mask)


class TestBaselineProperties(unittest.TestCase):
    """Section 4: why the fixed blend is well-behaved."""

    def test_error_in_one_source_enters_halved(self):
        """dp/dD = 0.5 exactly, for any D, C and any step -- the bounded
        sensitivity that the inverse-variance variant gives up."""
        for c in [0.0, 0.3, 0.8]:
            for d, delta in [(0.0, 0.2), (0.2, 0.2), (0.5, 0.4), (0.1, 0.05)]:
                before = _p([d], [c])[0].item()
                after = _p([d + delta], [c])[0].item()
                self.assertAlmostEqual(after - before, delta / 2.0, places=5)

    def test_monotone_in_both_priors(self):
        """More geometric (or colour) difference never lowers the prior."""
        grid = [i / 10.0 for i in range(11)]

        rising_d = _p(grid, [0.4] * len(grid)).tolist()
        self.assertEqual(rising_d, sorted(rising_d))

        rising_c = _p([0.4] * len(grid), grid).tolist()
        self.assertEqual(rising_c, sorted(rising_c))

    def test_defined_and_bounded_over_whole_domain(self):
        """No singularities: every (D, C) in [0,1]^2 maps to a finite p in [0,1].

        Contrast with the uncertainty variant, whose fusion is 0/0 at D=1, C=0.
        """
        grid = [i / 10.0 for i in range(11)]
        d = [x for x in grid for _ in grid]
        c = [y for _ in grid for y in grid]

        p = _p(d, c)
        self.assertTrue(torch.isfinite(p).all())
        self.assertGreaterEqual(p.min().item(), 0.0)
        self.assertLessEqual(p.max().item(), 1.0)


class TestBaselineColourPriorIsBrightnessOnly(unittest.TestCase):
    """Section 6: `C` is not a colour-difference prior -- it is |d(brightness)|.

        f_mat_2 = abs(mean(f_2_ex - nearest_f_2of1, dim=-1))      # line 428

    The mean over channels is taken on the *signed* difference, and only then
    abs -- so opposite-sign channel changes cancel out. Present in the
    uncertainty variant too (copied verbatim), so it does not skew the
    comparison between them -- but it does bound what either can see.
    """

    @staticmethod
    def _c(f_1, f_2):
        """Colour prior alone: identical positions -> D = 0 -> p = 0.5*C."""
        n = f_1.shape[0]
        pos = torch.zeros(n, 3)
        inp, _ = SiamEncFusionKPConv.get_mask_v2(
            None, pos, pos, f_1, f_2, POS_TH, F_TH, knn_identity(n)
        )
        return 2.0 * inp.view(-1)  # undo the 0.5 fusion weight

    def test_hue_flip_at_constant_brightness_is_invisible(self):
        red = torch.tensor([[1.0, 0.0, 0.0]])
        green = torch.tensor([[0.0, 1.0, 0.0]])
        # mean(-1, +1, 0) = 0 -> the prior reports "colour did not change"
        self.assertAlmostEqual(self._c(red, green)[0].item(), 0.0, places=6)

    def test_huge_hue_change_scores_below_a_tiny_brightness_change(self):
        red = torch.tensor([[1.0, 0.0, 0.0]])
        green = torch.tensor([[0.0, 1.0, 0.0]])
        dark = torch.tensor([[0.0, 0.0, 0.0]])
        dim = torch.tensor([[0.06, 0.06, 0.06]])

        hue_dist = torch.norm(green - red).item()        # ~1.414 in RGB space
        bright_dist = torch.norm(dim - dark).item()      # ~0.104
        self.assertGreater(hue_dist, 10 * bright_dist)   # ~13x further apart

        # ...yet the huge one scores 0 and the tiny one scores 0.1
        self.assertAlmostEqual(self._c(red, green)[0].item(), 0.0, places=6)
        self.assertAlmostEqual(self._c(dark, dim)[0].item(), 0.1, places=5)

    def test_prior_depends_only_on_mean_brightness(self):
        """Three wildly different hue shifts, one shared d(brightness) = 0.12."""
        f_1 = torch.tensor([
            [0.30, 0.30, 0.30],   # mean 0.30
            [0.90, 0.00, 0.00],   # mean 0.30
            [0.00, 0.45, 0.45],   # mean 0.30
        ])
        f_2 = torch.tensor([
            [0.42, 0.42, 0.42],   # mean 0.42
            [0.00, 0.90, 0.36],   # mean 0.42
            [0.81, 0.00, 0.45],   # mean 0.42
        ])
        expected = 0.12 / F_TH    # = 0.2
        for got in self._c(f_1, f_2).tolist():
            self.assertAlmostEqual(got, expected, places=5)


class TestBaselineVsInverseVariance(unittest.TestCase):
    """Section 3: 50/50 IS inverse-variance weighting, under sigma_D == sigma_C."""

    @staticmethod
    def _mu(d_targets, c_targets):
        pos_1, pos_2, f_1, f_2, knn_idx = make_inputs(d_targets, c_targets)
        mu, _, _ = SiamKPConvWithPriorUncertainty.get_mask_v2(
            None, pos_1, pos_2, f_1, f_2, POS_TH, F_TH, knn_idx
        )
        return mu.view(-1)

    def test_identical_when_bernoulli_variances_are_equal(self):
        """var_D == var_C  <=>  D == C  or  D == 1 - C.

        In both families the inverse-variance weights come out equal, so the
        weighted fusion collapses onto the baseline's fixed blend.
        """
        # D == C
        d1 = [0.1, 0.25, 0.5, 0.75]
        self.assertTrue(torch.allclose(self._mu(d1, d1), _p(d1, d1), atol=1e-5))

        # D == 1 - C  (endpoints excluded: 0/0 singularity at D=1, C=0)
        d2 = [0.3, 0.45, 0.7]
        c2 = [1 - x for x in d2]
        mu, p = self._mu(d2, c2), _p(d2, c2)
        self.assertTrue(torch.allclose(mu, p, atol=1e-5))
        # ...and both sit at the neutral midpoint
        self.assertTrue(torch.allclose(p, torch.full_like(p, 0.5), atol=1e-5))

    def test_degenerates_to_baseline_at_the_datasets_measured_means(self):
        """The paper sets each threshold to 2x the dataset average, which aims
        both priors at ~0.5 -- exactly where p(1-p) is maximal, equal and flat.

        At the measured HKCD means (D=0.25, C=0.30 per the audit's runtime
        table) the two fusions are already indistinguishable: the modification
        can only act in the tails.
        """
        mu = self._mu([0.25], [0.30])[0].item()
        p = _p([0.25], [0.30])[0].item()

        self.assertAlmostEqual(p, 0.275, places=5)
        self.assertAlmostEqual(mu, 0.27358, places=4)
        self.assertLess(abs(mu - p), 0.005)


if __name__ == "__main__":
    unittest.main()
