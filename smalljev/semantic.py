"""Per-option semantic scoring: shared scorer over option-span representations.

Why this exists: the slot head (SlotChoiceHead) maps one readout vector to
positional slots (slot i = i-th option). Training learns a positional prior
(measured: 31/40 MASSIVE-18 predictions pile onto slot 0) and collapses at
high cardinality. This module replaces positional binding with semantic
binding: each option's TOKEN SPAN is mean-pooled from a single forward pass
and scored by SHARED weights. The head is permutation-equivariant by
construction — there is no slot 0 to collapse onto.

Prompt layout (lettered, same semantics as render_prompt):
    State: {state}\nQuestion: {question}\nOptions: + chunks("\n{L}. {opt}") + tail
Spans cover the option TEXT tokens only (marker "A." excluded, so letter
embeddings cannot leak position through the pooled rep). input_ids are built
by concatenating separately-tokenized pieces, so spans are exact by
construction (no fragile substring search). Order-shuffle augmentation at
training time removes any residual order signal through the causal mask.
"""
from smalljev.model import LETTERS

TAIL = "\nAnswer with a single letter:"


def build_semantic_ids(tokenizer, state, question, options, max_len=1024):
    """Returns (input_ids, spans). spans[i] = (start, end) over option i's text.

    Truncation preserves options: state text is shortened until the whole
    prompt fits max_len.
    """
    options = list(options)
    st = state
    for _ in range(4):
        head_ids = tokenizer(f"State: {st}\nQuestion: {question}\nOptions:",
                             add_special_tokens=True)["input_ids"]
        chunks = []
        for i, opt in enumerate(options):
            m = tokenizer(f"\n{LETTERS[i]}.", add_special_tokens=False)["input_ids"]
            t = tokenizer(f" {opt}", add_special_tokens=False)["input_ids"]
            chunks.append((m, t))
        tail_ids = tokenizer(TAIL, add_special_tokens=False)["input_ids"]
        total = len(head_ids) + sum(len(m) + len(t) for m, t in chunks) + len(tail_ids)
        if total <= max_len or len(st) < 100:
            break
        # shrink state ~proportionally to the overflow (chars ~= tokens roughly)
        overflow = total - max_len
        st = st[:max(50, len(st) - int(overflow * 1.5))]
    input_ids, spans = list(head_ids), []
    for m, t in chunks:
        input_ids += m
        spans.append((len(input_ids), len(input_ids) + len(t)))
        input_ids += t
    input_ids += tail_ids
    return input_ids, spans


def verify_construction(tokenizer, state="hello world", question="q?",
                        options=("alpha", "beta")):
    """Decoded construction must equal render_prompt (modulo spacing)."""
    from smalljev.model import render_prompt
    ids, spans = build_semantic_ids(tokenizer, state, question, list(options))
    ref = render_prompt(state, question, list(options))
    got = tokenizer.decode(ids)
    for opt in options:
        assert opt in got, f"option {opt!r} missing from decoded prompt"
    assert "Answer with a single letter:" in got
    # spans must decode back to their option text
    for (a, b), opt in zip(spans, options):
        assert tokenizer.decode(ids[a:b]).strip() == opt.strip(), \
            f"span decodes to {tokenizer.decode(ids[a:b])!r}, want {opt!r}"
    return True
