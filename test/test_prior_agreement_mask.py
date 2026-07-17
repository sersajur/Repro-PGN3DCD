"""Analytical checks on SiamKPConvWithPriorAgreement.get_mask_v2 (FIX A).

Design intent the fusion must satisfy (prior_mask_math_uncertainty.md §8):
  1. confident change      -> high, sharp mask
  2. confident no-change   -> low mask (suppress, like the baseline)
  3. uncertain / conflict  -> mask ~ 0.5, and the conflict visibly flagged

Inputs are built by `prior_mask_common.make_inputs` (exact D/C values), so all
expectations are closed-form numbers. Companion modules:
`test_prior_baseline_mask.py`, `test_prior_uncertainty_mask.py`.
"""
import os
import sys
import unittest

import torch

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from test.prior_mask_common import make_inputs, POS_TH, F_TH

from torch_points3d.models.change_detection.SIFT_SKP_double_pc import SiamEncFusionKPConv
from torch_points3d.models.change_detection.SiamKPConvWithPriorAgreement import (
    SiamKPConvWithPriorAgreement,
)


def _fuse(d_targets, c_targets):
    """Agreement fusion -> (mu, mask, d), each flattened to (N,)."""
    pos_1, pos_2, f_1, f_2, knn_idx = make_inputs(d_targets, c_targets)
    mu, mask, d = SiamKPConvWithPriorAgreement.get_mask_v2(
        None, pos_1, pos_2, f_1, f_2, POS_TH, F_TH, knn_idx
    )
    return mu.view(-1), mask.view(-1), d.view(-1)


def _baseline(d_targets, c_targets):
    pos_1, pos_2, f_1, f_2, knn_idx = make_inputs(d_targets, c_targets)
    inp, _ = SiamEncFusionKPConv.get_mask_v2(
        None, pos_1, pos_2, f_1, f_2, POS_TH, F_TH, knn_idx
    )
    return inp.view(-1)


class TestAgreementFusion(unittest.TestCase):
    def test_mu_is_exactly_the_baseline_fusion(self):
        """mu keeps the robust fixed 50/50 blend -- no inverse-variance."""
        cases_d = [0.0, 0.2, 0.99, 1.0, 0.25]
        cases_c = [1.0, 0.6, 0.50, 0.0, 0.30]
        mu, _, _ = _fuse(cases_d, cases_c)
        p = _baseline(cases_d, cases_c)
        self.assertTrue(torch.allclose(mu, p, atol=1e-5))

    def test_agreeing_sources_reproduce_baseline_mask_exactly(self):
        """Intent 1+2: when D == C, d = 0 and mask == mu == baseline p.
        In particular the confidently-unchanged point is suppressed again
        (the uncertainty variant raised it to 0.296)."""
        vals = [0.0, 0.2, 0.5, 0.9, 1.0]
        _, mask, d = _fuse(vals, vals)

        self.assertTrue(torch.allclose(d, torch.zeros_like(d), atol=1e-6))
        for got, v in zip(mask.tolist(), vals):
            self.assertAlmostEqual(got, v, places=5)

    def test_conflict_yields_neutral_mask_and_full_disagreement_flag(self):
        """Intent 3: D=1, C=0 -> mask = 0.5 AND d = 1. The neutral value now
        arrives with an honest 'sources conflict' signal, not a fake sigma^2~0
        'trust this'. No 0/0 anywhere -- plain arithmetic."""
        mu, mask, d = _fuse([1.0], [0.0])
        self.assertAlmostEqual(mu[0].item(), 0.5, places=6)
        self.assertAlmostEqual(mask[0].item(), 0.5, places=6)
        self.assertAlmostEqual(d[0].item(), 1.0, places=6)

    def test_contradiction_and_ambiguity_are_finally_distinguishable(self):
        """The (mu, d) pair separates what no single scalar could:
        contradiction = (0.5, 1), ambiguity = (0.5, 0)."""
        mu_x, mask_x, d_x = _fuse([1.0], [0.0])     # sources at war
        mu_a, mask_a, d_a = _fuse([0.5], [0.5])     # sources agree on "unsure"

        # same mu, same mask...
        self.assertAlmostEqual(mu_x[0].item(), mu_a[0].item(), places=6)
        self.assertAlmostEqual(mask_x[0].item(), mask_a[0].item(), places=6)
        # ...but the channel finally tells them apart
        self.assertAlmostEqual(d_x[0].item(), 1.0, places=6)
        self.assertAlmostEqual(d_a[0].item(), 0.0, places=6)

    def test_disagreement_pulls_mask_toward_neutral(self):
        """D=0.9, C=0.3: mu = 0.6, d = 0.6 -> mask = 0.4*0.6 + 0.6*0.5 = 0.54.
        The uncertainty variant went the other way (0.66, following the
        extreme source past the baseline's 0.60)."""
        mu, mask, d = _fuse([0.9], [0.3])
        self.assertAlmostEqual(mu[0].item(), 0.6, places=5)
        self.assertAlmostEqual(d[0].item(), 0.6, places=5)
        self.assertAlmostEqual(mask[0].item(), 0.54, places=5)

        # cautious: strictly between neutral and the baseline value
        self.assertGreater(mask[0].item(), 0.5)
        self.assertLess(mask[0].item(), _baseline([0.9], [0.3])[0].item())

    def test_channel_amplitude_is_visible(self):
        """The old sigma^2 channel lived at ~0.06; d spans [0, 1] and hits
        both ends on constructible inputs."""
        _, _, d = _fuse([1.0, 0.3], [0.0, 0.3])
        self.assertAlmostEqual(d.max().item(), 1.0, places=6)
        self.assertAlmostEqual(d.min().item(), 0.0, places=6)


class TestAgreementProperties(unittest.TestCase):
    """Same battery the other two fusions went through."""

    def test_monotone_in_both_priors(self):
        grid = [i / 50.0 for i in range(51)]
        for other in [0.05, 0.2, 0.5, 0.9]:
            rising_d = _fuse(grid, [other] * len(grid))[1].tolist()
            self.assertEqual(rising_d, sorted(rising_d), f"non-monotone in D at C={other}")
            rising_c = _fuse([other] * len(grid), grid)[1].tolist()
            self.assertEqual(rising_c, sorted(rising_c), f"non-monotone in C at D={other}")

    def test_sensitivity_is_bounded_by_one(self):
        """Closed form: dmask/dD = 1-D above the diagonal, D below it --
        always within [0, 1]. Baseline is a constant 0.5; the uncertainty
        variant peaked at ~67. Verified numerically over a fine grid."""
        step = 1.0 / 200
        grid = [i * step for i in range(201)]
        worst = 0.0
        for c in [0.05, 0.3, 0.5, 0.7, 0.99]:
            mask = _fuse(grid, [c] * len(grid))[1].tolist()
            for i in range(len(mask) - 1):
                worst = max(worst, (mask[i + 1] - mask[i]) / step)
        self.assertLessEqual(worst, 1.0 + 1e-3)

    def test_defined_and_bounded_over_whole_domain(self):
        """mask is a convex combination of mu and 0.5 -- finite and inside
        [0, 1] everywhere, corners included (no 0/0 to regularise away)."""
        grid = [i / 10.0 for i in range(11)]
        d = [x for x in grid for _ in grid]
        c = [y for _ in grid for y in grid]

        _, mask, _ = _fuse(d, c)
        self.assertTrue(torch.isfinite(mask).all())
        self.assertGreaterEqual(mask.min().item(), 0.0)
        self.assertLessEqual(mask.max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
