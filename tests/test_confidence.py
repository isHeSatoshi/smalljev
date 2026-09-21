"""Confidence-signal tests — pure torch/numpy, no weights.
Paper-grounded: layer-convergence (2510.01237 Eq.2), OOD-kNN abstention
(2503.23303 meta-learning), precision bands + coverage (routing), curriculum order.
"""
import unittest


class TestConvergence(unittest.TestCase):
    def test_constant_layers_ratio_one(self):
        from smalljev.confidence import layer_convergence
        import torch
        torch.manual_seed(0)
        h = torch.randn(4, 8)
        layers = [h.clone() for _ in range(6)]
        self.assertAlmostEqual(layer_convergence(layers), 1.0, places=5)

    def test_shrinking_variance_scores_high(self):
        from smalljev.confidence import layer_convergence
        import torch
        torch.manual_seed(0)
        layers = [torch.randn(4, 32) * (1.0 / (i + 1)) for i in range(8)]
        self.assertGreater(layer_convergence(layers), 2.0)

    def test_growing_variance_scores_low(self):
        from smalljev.confidence import layer_convergence
        import torch
        torch.manual_seed(0)
        layers = [torch.randn(4, 32) * (i + 1) for i in range(8)]
        self.assertLess(layer_convergence(layers), 1.0)


class TestOODBank(unittest.TestCase):
    def test_near_beats_far(self):
        from smalljev.confidence import OODBank
        import numpy as np
        rng = np.random.RandomState(0)
        refs = rng.randn(50, 8)
        bank = OODBank().fit(refs)
        near = refs[:5] + rng.randn(5, 8) * 0.01
        far = rng.randn(5, 8) * 20 + 100
        self.assertLess(bank.score(near).mean(), bank.score(far).mean())

    def test_score_shape(self):
        from smalljev.confidence import OODBank
        import numpy as np
        bank = OODBank().fit(np.random.RandomState(1).randn(20, 4))
        self.assertEqual(bank.score(np.random.RandomState(2).randn(7, 4)).shape, (7,))


class TestBands(unittest.TestCase):
    def test_threshold_for_precision(self):
        from smalljev.confidence import threshold_for_precision
        import numpy as np
        conf = np.array([0.99, 0.95, 0.9, 0.6, 0.55, 0.5])
        correct = np.array([1, 1, 1, 0, 1, 0])
        t = threshold_for_precision(conf, correct, target=0.9)
        self.assertGreaterEqual(t, 0.0)
        self.assertLessEqual(t, 1.0)
        sel = conf >= t
        self.assertGreaterEqual(correct[sel].mean(), 0.9 - 1e-9)

    def test_coverage_curve_perfect(self):
        from smalljev.confidence import coverage_curve
        import numpy as np
        conf = np.linspace(0.1, 1.0, 50)
        correct = np.ones(50)
        cov, acc = coverage_curve(conf, correct)
        self.assertTrue(bool((np.asarray(acc) == 1.0).all()))
        self.assertAlmostEqual(min(cov), 0.1, places=5)


class TestCurriculum(unittest.TestCase):
    def test_easy_first(self):
        from smalljev.confidence import order_easy_first
        import numpy as np
        # (p_true, correct): easy = correct & high margin
        ptr = np.array([0.99, 0.55, 0.51, 0.9])
        correct = np.array([1, 1, 0, 1])
        order = order_easy_first(ptr, correct)
        self.assertEqual(list(order[:2]), [0, 3])  # confident-correct first
        self.assertEqual(list(order[-1:]), [2])    # wrong & unsure last


if __name__ == "__main__":
    unittest.main()
