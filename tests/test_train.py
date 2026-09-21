"""Tests for decision-SFT example construction (pure, CPU, cached tokenizer)."""
import unittest


class TestSFTExample(unittest.TestCase):
    def test_labels_mask_prompt_keep_answer(self):
        from smalljev.train_utils import build_sft_example
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-2B-Base",
                                            trust_remote_code=False)
        prompt = "State: x\nQuestion: pick?\nOptions:\nA. foo\nB. bar\nAnswer with a single letter:"
        ex = build_sft_example(tok, prompt, "B")
        self.assertEqual(ex["input_ids"][-1], ex["labels"][-1])  # answer kept
        self.assertTrue(all(l == -100 for l in ex["labels"][:-1]))  # prompt masked
        self.assertGreater(len(ex["input_ids"]), 5)

    def test_bandit_reward_sign(self):
        from smalljev.train_utils import bandit_reward
        self.assertEqual(bandit_reward(0, 0), 1.0)
        self.assertEqual(bandit_reward(1, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
