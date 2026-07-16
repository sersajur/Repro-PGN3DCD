"""Analytical checks on SiamKPConvWithPriorUncertainty.get_mask_v2.

`get_mask_v2` never touches `self`, so it is exercised as an unbound function —
no config/dataset/model construction needed.

Inputs are built by `prior_mask_common.make_inputs`, which produces *exact* D and
C values, so every expectation here is a closed-form number rather than a
regression snapshot. See `test_prior_baseline_mask.py` for the same treatment of
the baseline's fixed 50/50 fusion.
"""
import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from test.prior_mask_common import make_inputs, POS_TH, F_TH

from torch_points3d.models.change_detection.SiamKPConvWithPriorUncertainty import (
    SiamKPConvWithPriorUncertainty,
    _EPS,
    _MAX_VAR,
)


def _fuse(d_targets, c_targets):
    pos_1, pos_2, f_1, f_2, knn_idx = make_inputs(d_targets, c_targets)
    mu, mask, var = SiamKPConvWithPriorUncertainty.get_mask_v2(
        None, pos_1, pos_2, f_1, f_2, POS_TH, F_TH, knn_idx
    )
    return mu.view(-1), mask.view(-1), var.view(-1)


class TestPriorUncertaintyFusion(unittest.TestCase):
    def test_helper_reproduces_requested_priors(self):
        """Guard: a single-source case must fuse to that source's value."""
        # D == C == p  =>  weights are equal  =>  mu == p exactly.
        mu, _, _ = _fuse([0.2, 0.75], [0.2, 0.75])
        self.assertAlmostEqual(mu[0].item(), 0.2, places=5)
        self.assertAlmostEqual(mu[1].item(), 0.75, places=5)

    def test_agreeing_priors_pull_mask_toward_neutral(self):
        """Even with both priors agreeing, the mask is biased up toward 0.5.

        This is audit hypothesis F3, reproduced in closed form:
          var_D = var_C = 0.16 -> w = 6.25 each -> var_total = 0.08
          confidence = 1 - 0.08/0.25 = 0.68
          mask = 0.68*0.2 + 0.32*0.5 = 0.296   (vs mu = 0.2)
        """
        mu, mask, var = _fuse([0.2], [0.2])
        self.assertAlmostEqual(mu[0].item(), 0.2, places=5)
        self.assertAlmostEqual(var[0].item(), 0.08, places=5)
        self.assertAlmostEqual(mask[0].item(), 0.296, places=5)
        self.assertGreater(mask[0].item(), mu[0].item())

    def test_extreme_prior_dominates_moderate_one(self):
        """The more extreme prior hijacks the fusion, far past a 50/50 blend.

          D=0.99 -> var=0.0099 -> w_D=101.01 ;  C=0.5 -> var=0.25 -> w_C=4
          mu = (101.01*0.99 + 4*0.5) / 105.01 = 0.9713
        while the baseline's fixed fusion would give 0.5*0.99 + 0.5*0.5 = 0.745.
        """
        mu, _, _ = _fuse([0.99], [0.5])
        self.assertAlmostEqual(mu[0].item(), 0.97133, places=4)

        baseline = 0.5 * 0.99 + 0.5 * 0.5
        self.assertGreater(mu[0].item() - baseline, 0.2)

    def test_weight_tracks_extremeness_not_reliability(self):
        """sigma^2 is a deterministic function of the estimate itself.

        Both priors sit at the same distance from the decisive 0.5 point, yet
        the fused variance is identical whether they say "changed" or
        "unchanged" -- the weight cannot encode source trustworthiness.
        """
        _, _, var_low = _fuse([0.1], [0.1])
        _, _, var_high = _fuse([0.9], [0.9])
        self.assertAlmostEqual(var_low[0].item(), var_high[0].item(), places=5)

    def test_contradicting_priors_yield_confident_neutral_mask(self):
        """D=1 vs C=0: maximal disagreement, reported with maximal confidence.

        Both variances clamp to _EPS -> w = 1/_EPS each -> they cancel:
          mu = 0.5, var_total = _EPS/2, confidence ~ 1, mask = 0.5
        Real Bayesian fusion would *raise* uncertainty on disagreement.
        """
        mu, mask, var = _fuse([1.0], [0.0])

        self.assertAlmostEqual(mu[0].item(), 0.5, places=5)
        self.assertAlmostEqual(var[0].item(), _EPS / 2.0, places=9)
        self.assertAlmostEqual(mask[0].item(), 0.5, places=5)

        confidence = 1.0 - var[0].item() / _MAX_VAR
        self.assertGreater(confidence, 0.999)

    def test_mask_conflates_contradiction_with_ambiguity(self):
        """The headline failure: two opposite situations are indistinguishable.

        - contradiction (D=1, C=0): both sources decisive, and opposed
        - ambiguity    (D=0.5, C=0.5): both sources maximally undecided

        They produce the *same* mask, and the sigma^2 channel points the wrong
        way -- ~0 ("trust this") for the contradictory point vs 0.125 for the
        merely-undecided one.
        """
        _, mask_contra, var_contra = _fuse([1.0], [0.0])
        _, mask_ambig, var_ambig = _fuse([0.5], [0.5])

        self.assertAlmostEqual(mask_contra[0].item(), mask_ambig[0].item(), places=5)

        self.assertAlmostEqual(var_ambig[0].item(), 0.125, places=5)
        self.assertLess(var_contra[0].item(), var_ambig[0].item() / 1000)


def _closed_form(d, c):
    """mu = D*C*(2-D-C) / [D(1-D) + C(1-C)] -- see prior_mask_math_uncertainty.md.

    Derived by multiplying numerator and denominator of the inverse-variance
    fusion by D(1-D)C(1-C). Undefined (0/0) at D=1,C=0 and at D=0,C=1.
    """
    return d * c * (2 - d - c) / (d * (1 - d) + c * (1 - c))


class TestUncertaintyClosedForm(unittest.TestCase):
    def test_closed_form_matches_implementation(self):
        """The whole fusion collapses to a ratio of two polynomials -- no
        probability left in it. Interior points only: the implementation clamps
        the variances at _EPS, so it diverges from the algebra at the corners.
        """
        cases = [(0.25, 0.30), (0.99, 0.50), (0.20, 0.20), (0.90, 0.05), (0.70, 0.30)]
        mu, _, _ = _fuse([d for d, _ in cases], [c for _, c in cases])
        for got, (d, c) in zip(mu.tolist(), cases):
            self.assertAlmostEqual(got, _closed_form(d, c), places=5)


class TestUncertaintyProperties(unittest.TestCase):
    """Mirrors `TestBaselineProperties` -- same questions, different answers."""

    def test_monotone_in_both_priors(self):
        """Preserved: more geometric (or colour) difference never lowers mu."""
        grid = [i / 50.0 for i in range(51)]

        for c in [0.05, 0.2, 0.5, 0.9]:
            rising_d = _fuse(grid, [c] * len(grid))[0].tolist()
            self.assertEqual(rising_d, sorted(rising_d), f"non-monotone in D at C={c}")

        for d in [0.05, 0.2, 0.5, 0.9]:
            rising_c = _fuse([d] * len(grid), grid)[0].tolist()
            self.assertEqual(rising_c, sorted(rising_c), f"non-monotone in C at D={d}")

    def test_sensitivity_to_one_source_is_unbounded(self):
        """LOST vs baseline: dp/dD is a constant 0.5 there; here it is not
        bounded at all, so one source can drag the fusion the whole way.

        Worst case sits at D->0 with C extreme: the weight w_D = 1/(D(1-D))
        collapses from the _EPS clamp (1e6) down to ~200 over a half-percent
        step in D, handing the vote to C.
        """
        mu = _fuse([0.0, 0.005], [0.99, 0.99])[0]
        slope = (mu[1] - mu[0]).item() / 0.005

        self.assertAlmostEqual(mu[0].item(), 0.0, places=3)
        self.assertAlmostEqual(mu[1].item(), 0.3344, places=3)
        self.assertGreater(slope, 50.0)  # measured ~66.8 vs baseline's exact 0.5

    def test_limit_at_the_singularity_is_path_dependent(self):
        """D=1, C=0 is 0/0 in closed form -- and not a fillable hole: the value
        depends on the direction of approach, so no limit exists.

        The code returns 0.5 there, but that is an artefact of the _EPS clamp.
        """
        eps = [0.01, 0.005, 0.002]

        # Path A: C = eps      -> var_D == var_C -> equal weights -> mu = 0.5
        mu_a = _fuse([1 - e for e in eps], [e for e in eps])[0]
        for got in mu_a.tolist():
            self.assertAlmostEqual(got, 0.5, places=4)

        # Path B: C = eps**2   -> C's weight dwarfs D's -> mu -> 0
        mu_b = _fuse([1 - e for e in eps], [e ** 2 for e in eps])[0]
        for got, e in zip(mu_b.tolist(), eps):
            self.assertAlmostEqual(got, e, places=3)

        self.assertGreater(mu_a.min().item() - mu_b.max().item(), 0.4)

    def test_confidence_can_never_fall_below_one_half(self):
        """_MAX_VAR = 0.25 is the max variance of a *single* Bernoulli, but the
        *fused* variance cannot exceed 0.125: each w >= 4, so w_sum >= 8.

        So `confidence = 1 - var/0.25` only ever uses the top half of [0, 1],
        and the `.clamp(min=0.0)` in the source is an unreachable branch.
        """
        grid = [i / 20.0 for i in range(21)]
        d = [x for x in grid for _ in grid]
        c = [y for _ in grid for y in grid]

        _, _, var = _fuse(d, c)
        self.assertLessEqual(var.max().item(), 0.125 + 1e-6)

        confidence = (1.0 - var / _MAX_VAR).clamp(min=0.0, max=1.0)
        self.assertGreaterEqual(confidence.min().item(), 0.5 - 1e-6)

        # The ceiling is reached exactly where both priors are maximally undecided.
        _, _, var_mid = _fuse([0.5], [0.5])
        self.assertAlmostEqual(var_mid[0].item(), 0.125, places=5)


if __name__ == "__main__":
    unittest.main()
