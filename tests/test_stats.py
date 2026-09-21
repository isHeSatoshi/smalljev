"""Stats unit tests — closed-form values, no weights."""
import unittest


class TestStats(unittest.TestCase):
    def test_wilson_known(self):
        from smalljev.stats import wilson_ci
        lo, hi = wilson_ci(60, 100)
        self.assertLess(lo, 0.6)
        self.assertGreater(hi, 0.6)
        self.assertAlmostEqual(lo, 0.502, delta=0.01)
        self.assertAlmostEqual(hi, 0.691, delta=0.01)

    def test_wilson_empty(self):
        from smalljev.stats import wilson_ci
        import math
        lo, hi = wilson_ci(0, 0)
        self.assertTrue(math.isnan(lo) and math.isnan(hi))

    def test_mcnemar_strong(self):
        from smalljev.stats import mcnemar_p
        self.assertLess(mcnemar_p(20, 2), 0.01)

    def test_mcnemar_null(self):
        from smalljev.stats import mcnemar_p
        self.assertGreater(mcnemar_p(10, 10), 0.5)

    def test_auroc_perfect(self):
        from smalljev.stats import auroc_score
        self.assertAlmostEqual(
            auroc_score([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)

    def test_auroc_chance(self):
        from smalljev.stats import auroc_score
        self.assertAlmostEqual(
            auroc_score([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]), 0.5)

    def test_ece_perfect(self):
        from smalljev.stats import ece_conf
        import numpy as np
        self.assertAlmostEqual(
            ece_conf(np.ones(10), np.ones(10)), 0.0, places=9)

    def test_risk_coverage(self):
        from smalljev.stats import risk_coverage
        import numpy as np
        rows = risk_coverage(np.array([0.9, 0.8, 0.3, 0.2]),
                             np.array([1, 1, 0, 0]), coverages=(0.5, 1.0))
        self.assertAlmostEqual(rows[0]["accuracy"], 1.0)
        self.assertAlmostEqual(rows[1]["accuracy"], 0.5)

    def test_reliability_bins(self):
        from smalljev.stats import reliability_bins
        import numpy as np
        rows = reliability_bins(np.array([0.95, 0.05]), np.array([1, 0]), n_bins=2)
        self.assertEqual(rows[0]["n"] + rows[1]["n"], 2)


if __name__ == "__main__":
    unittest.main()
