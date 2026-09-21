"""End-to-end API tests on the stub backend — no weights needed.

Desired API (from the project brief):

    {
      "intent": {"values": [...], "probabilities": [...]},
      "urgent": {"values": [...], "probabilities": [...]},
      "risk":   {"value": 0.73},
    }

Plus a `_meta` block reporting generated_tokens == 0 (no autoregression).
"""
import unittest


STATE = "Customer was charged twice for the same order and has already contacted support."

QUESTIONS = {
    "intent": {"type": "choice", "question": "What is the primary issue?",
               "choices": ["duplicate charge", "late delivery",
                           "account problem", "other"]},
    "urgent": {"type": "choice", "question": "Should this be escalated to a human?",
               "choices": ["yes", "no"]},
    "risk": {"type": "score", "question": "How risky is this situation?",
             "low": 0.0, "high": 1.0},
}


class TestDecideStub(unittest.TestCase):
    def test_returns_typed_json_shape(self):
        from smalljev import decide
        out = decide(STATE, QUESTIONS, backend="stub")
        self.assertEqual(out["intent"]["values"],
                         ["duplicate charge", "late delivery",
                          "account problem", "other"])
        self.assertEqual(len(out["intent"]["probabilities"]), 4)
        self.assertEqual(out["urgent"]["values"], ["yes", "no"])
        self.assertIsInstance(out["risk"]["value"], float)
        self.assertGreaterEqual(out["risk"]["value"], 0.0)
        self.assertLessEqual(out["risk"]["value"], 1.0)

    def test_probabilities_sum_to_one(self):
        from smalljev import decide
        out = decide(STATE, QUESTIONS, backend="stub")
        for key in ("intent", "urgent"):
            total = sum(out[key]["probabilities"])
            self.assertAlmostEqual(total, 1.0, places=5)

    def test_deterministic_given_same_input(self):
        from smalljev import decide
        first = decide(STATE, QUESTIONS, backend="stub")
        second = decide(STATE, QUESTIONS, backend="stub")
        self.assertEqual(first, second)

    def test_zero_generated_tokens(self):
        from smalljev import decide
        out = decide(STATE, QUESTIONS, backend="stub")
        self.assertIn("_meta", out)
        self.assertEqual(out["_meta"]["generated_tokens"], 0)

    def test_noul_returns_probability(self):
        from smalljev import decide
        out = decide(STATE, {"churn": {"type": "noul",
                                       "question": "Is this customer about to churn?"}},
                     backend="stub")
        p = out["churn"]["probability"]
        self.assertIsInstance(p, float)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)
        self.assertEqual(out["_meta"]["generated_tokens"], 0)

    def test_supports_unseen_question_without_retraining(self):
        from smalljev import decide
        new_q = dict(QUESTIONS)
        new_q["tone"] = {"type": "choice",
                         "question": "What is the customer tone?",
                         "choices": ["angry", "neutral", "polite"]}
        out = decide(STATE, new_q, backend="stub")
        self.assertEqual(out["tone"]["values"], ["angry", "neutral", "polite"])
        self.assertAlmostEqual(sum(out["tone"]["probabilities"]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
