"""Spec validation tests — RED first, no production code exists yet."""
import unittest


STATE = "Customer was charged twice for the same order and has already contacted support."


class TestSpecValidation(unittest.TestCase):
    def test_choice_requires_at_least_two_options(self):
        from smalljev.spec import ChoiceQuestion
        with self.assertRaises(ValueError):
            ChoiceQuestion(name="intent", question="What is it?",
                           choices=["only-one"])

    def test_choice_caps_at_255_options(self):
        from smalljev.spec import ChoiceQuestion
        with self.assertRaises(ValueError):
            ChoiceQuestion(name="big", question="Pick one.",
                           choices=[f"opt-{i}" for i in range(256)])

    def test_score_requires_ordered_levels(self):
        from smalljev.spec import ScoreQuestion
        with self.assertRaises(ValueError):
            ScoreQuestion(name="risk", question="How risky?",
                          levels=["low", "high"], values=[1.0, 0.0])

    def test_duplicate_question_names_rejected(self):
        from smalljev.spec import build_spec
        with self.assertRaises(ValueError):
            build_spec([
                {"name": "a", "type": "choice", "question": "Q1",
                 "choices": ["x", "y"]},
                {"name": "a", "type": "choice", "question": "Q2",
                 "choices": ["x", "y"]},
            ])

    def test_noul_spec_builds(self):
        from smalljev.spec import build_spec, NoulQuestion
        spec = build_spec([{"name": "churn", "type": "noul",
                            "question": "Is this customer about to churn?"}])
        self.assertIsInstance(spec["churn"], NoulQuestion)
        self.assertEqual(spec["churn"].question, "Is this customer about to churn?")


if __name__ == "__main__":
    unittest.main()
