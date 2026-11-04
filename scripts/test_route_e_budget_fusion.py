#!/usr/bin/env python3
"""CPU-only synthetic tests; no server cache, GPU, or held-out test data."""
import unittest
from copy import deepcopy

import numpy as np

import route_e_budget_fusion as f


class BudgetFusionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"sample_id": f"{'math' if i % 3 == 0 else 'gsm8k'}:{i}",
             "dataset": "math" if i % 3 == 0 else "gsm8k",
             "p_t": .70 + .01 * (i % 5), "p_n": .40 + .01 * (i % 5),
             "t_t": 2200. + 25*i, "t_n": 220. + 3*i}
            for i in range(30)]
        self.q = np.linspace(-2, 2, len(self.rows))
        self.c = np.linspace(2, -2, len(self.rows))
        self.lengths = np.array([[r['t_t'], r['t_n']] for r in self.rows])

    def test_alpha_endpoints_and_orientation(self):
        np.testing.assert_allclose(f.fusion_scores(self.q,self.c,0),self.c)
        np.testing.assert_allclose(f.fusion_scores(self.q,self.c,1),self.q)
        self.assertEqual(f.ranking(self.q,self.rows)[0],len(self.rows)-1)
        self.assertEqual(f.ranking(self.c,self.rows)[0],0)

    def test_heldout_folds_cover_exactly_once(self):
        groups=f.make_folds(self.rows,5,42)
        self.assertEqual(sorted(i for g in groups for i in g),list(range(len(self.rows))))
        self.assertTrue(all(len(g)>0 for g in groups))

    def test_budget_mode_does_not_peek_actual_lengths(self):
        a=f.policy_at_budget(self.rows,self.q,1200,'predicted',self.lengths)
        modified=deepcopy(self.rows)
        for row in modified:
            row['t_t'] += 5000  # actual cost changed; predictions are unchanged
        b=f.policy_at_budget(modified,self.q,1200,'predicted',self.lengths)
        np.testing.assert_allclose(a[1],b[1])
        self.assertNotAlmostEqual(a[0]['actual_tokens'],b[0]['actual_tokens'])
        self.assertAlmostEqual(a[0]['predicted_tokens'],b[0]['predicted_tokens'])

    def test_dense_exact_budget(self):
        r=f.policy_at_budget(self.rows,self.q,1200,'observed',self.lengths)
        self.assertAlmostEqual(r[0]['actual_tokens'],1200,places=5)
        self.assertTrue(np.all((r[1]>=0)&(r[1]<=1)))

    def test_oof_full_routing(self):
        sweep,choices,summary=f.run(self.rows,self.q,self.c,self.lengths,
                                   (1000,1400,1800,2200),f.ALPHAS,5,42,.0025,10)
        self.assertEqual(len(choices),5)
        self.assertEqual(len(summary),4)
        self.assertEqual(len(sweep),20)
        self.assertTrue(all('legacy_logistic_plus_cost' not in r['methods'] for r in summary))

    def test_development_fold_unreachable_budget_is_skipped(self):
        # A development subset can have a narrower cost range than the
        # full validation population; no extrapolation or held-out peeking.
        selection = f.choose_weights(
            self.rows[:18], self.q[:18], self.c[:18],
            (900, 1200, 2400, 3000), f.ALPHAS, .0025)
        self.assertIn(900, selection["available_budgets"])
        self.assertIn(3000, selection["unreachable_budgets"])
        self.assertEqual(selection["piecewise"]["high"],
                         selection["best_fixed"])
        # Selection does not use the held-out examples.
        self.assertEqual(selection, f.choose_weights(
            self.rows[:18], self.q[:18], self.c[:18],
            (900, 1200, 2400, 3000), f.ALPHAS, .0025))

    def test_development_all_budgets_unreachable_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "No jointly reachable"):
            f.choose_weights(self.rows, self.q, self.c,
                             (100000,), f.ALPHAS, .0025)

    def test_train_only_stats(self):
        mu,sigma=f.normalizer(np.array([1.,2.,3.]))
        self.assertEqual(mu,2.)
        self.assertAlmostEqual(sigma,np.sqrt(2./3.))
        self.assertAlmostEqual(f.standardize(np.array([2.]),(mu,sigma))[0],0.)
        with self.assertRaises(ValueError):
            f.normalizer(np.ones(5))


if __name__ == '__main__':
    unittest.main()