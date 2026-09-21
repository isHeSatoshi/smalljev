"""Calibration math tests — pure numpy, no weights needed."""
import unittest


class TestCalibration(unittest.TestCase):
    def test_ece_of_perfect_confident_predictions_is_zero(self):
        from smalljev.calibration import expected_calibration_error
        import numpy as np
        probs = np.array([1.0, 1.0, 1.0, 1.0])
        labels = np.array([1, 1, 1, 1])
        self.assertAlmostEqual(
            expected_calibration_error(probs, labels, n_bins=5), 0.0, places=9)

    def test_brier_score_bounds(self):
        from smalljev.calibration import brier_score
        import numpy as np
        # perfect: 0.0 ; worst: 1.0
        self.assertAlmostEqual(
            brier_score(np.array([1.0, 0.0]), np.array([1, 0])), 0.0)
        self.assertAlmostEqual(
            brier_score(np.array([0.0, 1.0]), np.array([1, 0])), 1.0)

    def test_vector_scaling_fixes_systematic_bias(self):
        from smalljev.calibration import (apply_vector_scale, fit_temperature,
                                          fit_vector_scale, negative_log_likelihood,
                                          apply_temperature)
        import numpy as np
        rng = np.random.RandomState(1)
        # balanced truth, but logits carry a constant +1.5 bias toward class 0
        y = np.tile([0, 1], 150)
        base = np.zeros((300, 2))
        base[np.arange(300), y] = 2.0
        logits = base + np.array([1.5, 0.0]) + rng.randn(300, 2) * 0.3
        fit_l, fit_y = logits[:200], y[:200]
        tst_l, tst_y = logits[200:], y[200:]
        nll_raw = negative_log_likelihood(apply_temperature(tst_l, 1.0), tst_y)
        t = fit_temperature(fit_l, fit_y)
        nll_temp = negative_log_likelihood(apply_temperature(tst_l, t), tst_y)
        t2, b = fit_vector_scale(fit_l, fit_y)
        nll_vec = negative_log_likelihood(apply_vector_scale(tst_l, t2, b), tst_y)
        self.assertLess(nll_vec, nll_temp - 0.05)  # scalar temp cannot fix bias
        self.assertLess(nll_vec, nll_raw - 0.2)

    def test_vector_scaling_matches_temperature_without_bias(self):
        from smalljev.calibration import (apply_temperature, apply_vector_scale,
                                          fit_temperature, fit_vector_scale,
                                          negative_log_likelihood)
        import numpy as np
        rng = np.random.RandomState(2)
        logits = rng.randn(240, 3) * 1.5
        labels = np.argmax(logits + rng.randn(240, 3), axis=1)
        t = fit_temperature(logits[:160], labels[:160])
        t2, b = fit_vector_scale(logits[:160], labels[:160])
        n1 = negative_log_likelihood(apply_temperature(logits[160:], t), labels[160:])
        n2 = negative_log_likelihood(apply_vector_scale(logits[160:], t2, b), labels[160:])
        self.assertAlmostEqual(n1, n2, delta=0.05)

    def test_temperature_scaling_reduces_nll_when_overconfident(self):
        from smalljev.calibration import fit_temperature, apply_temperature
        import numpy as np
        rng = np.random.RandomState(0)
        # overconfident wrong-ish logits: big magnitudes, ~70% correct
        logits = rng.randn(200, 3) * 4.0
        labels = np.argmax(logits + rng.randn(200, 3) * 2.0, axis=1)
        from smalljev.calibration import negative_log_likelihood
        nll_before = negative_log_likelihood(apply_temperature(logits, 1.0), labels)
        temp = fit_temperature(logits, labels)
        nll_after = negative_log_likelihood(apply_temperature(logits, temp), labels)
        self.assertGreater(temp, 1.0)  # must soften overconfidence
        self.assertLess(nll_after, nll_before)


if __name__ == "__main__":
    unittest.main()
