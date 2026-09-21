"""Pure helpers for decision tuning (no GPU, no model weights)."""


def answer_token_id(tokenizer, letter):
    toks = tokenizer.encode(f" {letter}", add_special_tokens=False)
    if len(toks) != 1:
        toks = tokenizer.encode(letter, add_special_tokens=False)
    assert len(toks) == 1, f"letter {letter!r} is not a single token"
    return toks[0]


def build_sft_example(tokenizer, prompt, letter):
    """SFT example where loss applies ONLY to the single answer-letter token."""
    ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    ans = answer_token_id(tokenizer, letter)
    labels = [-100] * len(ids) + [ans]
    return {"input_ids": ids + [ans], "labels": labels}


def bandit_reward(sampled, label):
    return 1.0 if sampled == label else 0.0
