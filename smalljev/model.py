"""Backends: deterministic stub (no weights) + HF verbalizer (one forward pass, zero generation).

The HF backend is the v0 proof of the architecture: a frozen causal decoder repurposed as a
decision scorer. It NEVER calls model.generate() — a single forward() yields logits at the
readout position, which are restricted to option verbalizers and softmaxt into probabilities.
"""
import hashlib

import numpy as np

from .calibration import softmax
from .spec import ChoiceQuestion, NoulQuestion, ScoreQuestion

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

DEFAULT_BACKBONE = "openbmb/MiniCPM5-2B-Base"  # Apache-2.0, LlamaForCausalLM, 2.5B, 131K ctx


def render_prompt(state, question, options):
    """Single source of truth for the decision prompt (inference AND training)."""
    lines = [f"State: {state}", f"Question: {question}", "Options:"]
    lines += [f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
    lines += ["Answer with a single letter:"]
    return "\n".join(lines)


def _kind_of(q):
    if isinstance(q, ChoiceQuestion):
        return "choice"
    if isinstance(q, NoulQuestion):
        return "noul"
    return "score"


def _stub_logits(key, n):
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "little")
    return np.random.RandomState(seed).randn(n)


def _stub_unit(key):
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") / 2.0 ** 64


class StubBackend:
    """Deterministic stand-in: exercises template -> softmax -> JSON plumbing sans weights."""
    name = "stub"
    generated_tokens = 0

    def score_all(self, state, spec, temperature=1.0):
        out = {}
        for qname, q in spec.items():
            if isinstance(q, NoulQuestion):
                out[qname] = np.float64(_stub_unit(f"{state}||{q.question}||noul"))
            elif isinstance(q, ChoiceQuestion):
                logits = _stub_logits(f"{state}||{q.question}||{'|'.join(q.choices)}",
                                      len(q.choices))
                out[qname] = softmax(logits / temperature)
            else:
                logits = _stub_logits(f"{state}||{q.question}||score", len(q.levels))
                out[qname] = softmax(logits / temperature)
        return out


class HFBackend:
    """Frozen causal LM + verbalizer readout. One forward() per question batch, zero generation."""

    def __init__(self, model_id=DEFAULT_BACKBONE, adapter_id=None, heads_ckpt=None,
                 sem_ckpt=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto" if device == "cuda" else None,
            trust_remote_code=False,
            attn_implementation="eager",  # required: v1 packed sampler passes 4D masks
        ).eval()
        if adapter_id is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_id).eval()
            self.name = f"{model_id}+{adapter_id}"
        else:
            self.name = model_id
        self.device = next(self.model.parameters()).device
        self.generated_tokens = 0
        self._verbalizer_cache = {}
        self.heads = None
        if heads_ckpt is not None:
            from .heads import HeadsBundle, SlotChoiceHead
            try:
                self.heads = HeadsBundle.load(heads_ckpt)
            except KeyError:
                # legacy v1-v3 file: SlotChoiceHead only; wrap, Noul/Score stay verbalizer
                legacy = SlotChoiceHead.load(heads_ckpt)
                self.heads = HeadsBundle(legacy.slot.in_features, legacy.max_slots, 8)
                self.heads.choice.load_state_dict(legacy.state_dict())
            self.heads.to(self.device).eval()
        self.sem_head = None
        if sem_ckpt is not None:
            from .heads import OptionScorerHead
            self.sem_head = OptionScorerHead.load(sem_ckpt).to(self.device).eval()

    def _prompt(self, state, question, options):
        return render_prompt(state, question, options)

    def _branch_suffix(self, question, options):
        lines = [f"Question: {question}", "Options:"]
        lines += [f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
        lines += ["Answer with a single letter:"]
        return "\n" + "\n".join(lines)

    def packed_forward(self, state, suffixes, mask="block", need_hidden=False):
        """Raw packed forward: [STATE][S1]...[Sk] with a block-isolation or plain
        causal mask. Returns (logits[T, V], branch_bounds). Used to PROVE isolation:
        branch-1 output must equal the causal-mask run exactly, and must be invariant
        to other branches' content/order."""
        from .parallel import build_block_mask, to_additive_4d
        torch = self.torch
        state_ids = self.tokenizer(f"State: {state}", add_special_tokens=True)["input_ids"]
        branch_ids = [self.tokenizer(s, add_special_tokens=False)["input_ids"]
                      for s in suffixes]
        input_ids = list(state_ids) + [t for ids in branch_ids for t in ids]
        if mask == "block":
            allow = build_block_mask(len(state_ids), [len(b) for b in branch_ids])
            attn = to_additive_4d(allow).to(self.device)
        elif mask == "causal":
            attn = None  # model applies standard causal masking
        else:
            raise ValueError(f"mask must be 'block' or 'causal', got {mask!r}.")
        ids = torch.tensor([input_ids], device=self.device)
        with torch.no_grad():
            fw = self.model(input_ids=ids, attention_mask=attn, use_cache=False,
                            output_hidden_states=need_hidden)
            logits = fw.logits[0]
            hidden = fw.hidden_states[-1][0] if need_hidden else None
        bounds, pos = [], len(state_ids)
        for b in branch_ids:
            bounds.append((pos, pos + len(b)))
            pos += len(b)
        return logits, bounds, hidden

    def score_all_packed(self, state, spec, temperature=1.0):
        """        v1 sampler: STATE encoded ONCE, branches share the prefix via a block-
        isolation mask (parallel.build_block_mask). One forward pass total."""
        items = []
        for qname, q in spec.items():
            kind = _kind_of(q)
            if kind == "noul":
                options = ["yes", "no"]
            else:
                options = list(q.choices) if kind == "choice" else list(q.levels)
            items.append((qname, self._branch_suffix(q.question, options), options, kind))
        logits, bounds, hidden = self.packed_forward(
            state, [s for _, s, _, _ in items], mask="block",
            need_hidden=self.heads is not None)
        out = {}
        for (qname, _, opts, kind), (a, b) in zip(items, bounds):
            if self.heads is not None and kind == "choice":
                out[qname] = self._head_probs(hidden[b - 1], len(opts))
            elif self.heads is not None and kind == "noul":
                out[qname] = self._head_noul(hidden[b - 1])
            elif self.heads is not None and kind == "score":
                out[qname] = self._head_score_probs(hidden[b - 1], len(opts))
            elif kind == "noul":
                ids_v = self._verbalizer_ids(2)
                out[qname] = float(softmax(logits[b - 1, ids_v].float().cpu().numpy() / temperature)[0])
            else:
                ids_v = self._verbalizer_ids(len(opts))
                out[qname] = softmax(logits[b - 1, ids_v].float().cpu().numpy() / temperature)
        assert self.generated_tokens == 0
        return out

    def _verbalizer_ids(self, n):
        """Token ids for option letters A..; resolved once per tokenizer."""
        if n not in self._verbalizer_cache:
            ids = []
            for i in range(n):
                letter = LETTERS[i]
                for cand in (f" {letter}", letter):
                    toks = self.tokenizer.encode(cand, add_special_tokens=False)
                    if len(toks) == 1:
                        ids.append(toks[0])
                        break
                else:
                    raise ValueError(f"Letter {letter!r} is not a single token.")
            self._verbalizer_cache[n] = ids
        return self._verbalizer_cache[n]

    def _readout_rows(self, enc):
        """Last non-pad position per batch row (right-padded batches)."""
        mask = enc["attention_mask"]
        return (mask.sum(1) - 1).tolist()

    def _head_probs(self, h, k):
        import numpy as np
        with self.torch.no_grad():
            p = self.heads.choice.probs(h.float(), k).float().cpu().numpy()
        return np.asarray(p, dtype=np.float64)

    def _head_noul(self, h):
        with self.torch.no_grad():
            p = self.heads.noul.prob(h.float()).float().cpu().numpy()
        return float(np.asarray(p, dtype=np.float64).reshape(-1)[0])

    def _head_score_probs(self, h, n_levels):
        import numpy as np
        with self.torch.no_grad():
            p = self.heads.score.probs(h.float(), n_levels).float().cpu().numpy()
        return np.asarray(p, dtype=np.float64)

    def score_all(self, state, spec, temperature=1.0):
        torch = self.torch
        # items: (qname, prompt, options, kind). Noul without heads falls back to a
        # yes/no verbalizer (P("yes") = probability); with heads it uses the bit.
        items = []
        for qname, q in spec.items():
            kind = _kind_of(q)
            if kind == "noul":
                options = ["yes", "no"]
            else:
                options = list(q.choices) if kind == "choice" else list(q.levels)
            items.append((qname, self._prompt(state, q.question, options), options, kind))
        enc = self.tokenizer([p for _, p, _, _ in items],
                             return_tensors="pt", padding=True).to(self.device)
        use_heads = self.heads is not None
        with torch.no_grad():
            # SINGLE forward pass for the whole batch. generate() is never called.
            fw = self.model(**enc, output_hidden_states=use_heads, use_cache=False)
            last_idx = self._readout_rows(enc)
            ar = torch.arange(len(items), device=self.device)
            logits = fw.logits[ar, last_idx, :]
            hidden = fw.hidden_states[-1][ar, last_idx, :] if use_heads else None
        out = {}
        for idx, (qname, _, opts, kind) in enumerate(items):
            if use_heads and kind == "choice":
                out[qname] = self._head_probs(hidden[idx], len(opts))
            elif use_heads and kind == "noul":
                out[qname] = self._head_noul(hidden[idx])
            elif use_heads and kind == "score":
                out[qname] = self._head_score_probs(hidden[idx], len(opts))
            elif kind == "noul":
                ids = self._verbalizer_ids(2)
                opt_logits = logits[idx, ids].float().cpu().numpy() / temperature
                out[qname] = float(softmax(opt_logits)[0])
            else:
                ids = self._verbalizer_ids(len(opts))
                opt_logits = logits[idx, ids].float().cpu().numpy() / temperature
                out[qname] = softmax(opt_logits)
        assert self.generated_tokens == 0
        return out

    def _decode_branch(self, kind, opts, logit_row, hidden_row, temperature=1.0):
        """Shared readout: learned heads when present, verbalizer otherwise."""
        if self.heads is not None and kind == "choice":
            return self._head_probs(hidden_row, len(opts))
        if self.heads is not None and kind == "noul":
            return self._head_noul(hidden_row)
        if self.heads is not None and kind == "score":
            return self._head_score_probs(hidden_row, len(opts))
        if kind == "noul":
            ids_v = self._verbalizer_ids(2)
            return float(softmax(logit_row[ids_v].float().cpu().numpy() / temperature)[0])
        ids_v = self._verbalizer_ids(len(opts))
        return softmax(logit_row[ids_v].float().cpu().numpy() / temperature)

    def packed_hidden_all(self, state, suffixes):
        """All layer hidden states for packed branches (confidence features).

        Returns (bounds, layers) with layers = list of L (T, H) tensors on device.
        Caller slices readout positions. Heavy (~L*T*H); use on demand only.
        """
        from .parallel import build_block_mask, to_additive_4d
        torch = self.torch
        state_ids = self.tokenizer(f"State: {state}", add_special_tokens=True)["input_ids"]
        branch_ids = [self.tokenizer(s, add_special_tokens=False)["input_ids"]
                      for s in suffixes]
        allow = build_block_mask(len(state_ids), [len(b) for b in branch_ids])
        attn = to_additive_4d(allow).to(self.device)
        ids = torch.tensor([state_ids + [t for b in branch_ids for t in b]],
                           device=self.device)
        with torch.no_grad():
            fw = self.model(input_ids=ids, attention_mask=attn, use_cache=False,
                            output_hidden_states=True)
        bounds, pos = [], len(state_ids)
        for b in branch_ids:
            bounds.append((pos, pos + len(b)))
            pos += len(b)
        return bounds, [h[0] for h in fw.hidden_states]

    def score_all_broadcast(self, state, spec, temperature=1.0):
        """Batched prefix-cache sampler: prefill STATE once, then score ALL branch
        suffixes in ONE forward against batch-expanded shared KVs. Each row attends
        the shared prefix + only its own suffix (block isolation across rows).
        Same math as the block mask; attention is O(k*Lb*(Ls+Lb)) instead of
        O((Ls+k*Lb)^2), and state KVs are never recomputed."""
        import numpy as np
        from .parallel import BLOCKED
        torch = self.torch
        items = []
        for qname, q in spec.items():
            kind = _kind_of(q)
            if kind == "noul":
                options = ["yes", "no"]
            else:
                options = list(q.choices) if kind == "choice" else list(q.levels)
            items.append((qname, self.tokenizer(
                self._branch_suffix(q.question, options),
                add_special_tokens=False)["input_ids"], options, kind))
        state_ids = self.tokenizer(f"State: {state}", add_special_tokens=True)["input_ids"]
        use_heads = self.heads is not None
        out = {}
        with torch.no_grad():
            s = torch.tensor([state_ids], device=self.device)
            base_past = self.model(input_ids=s, use_cache=True,
                                   output_hidden_states=False).past_key_values
            k = len(items)
            max_lb = max(len(ids) for _, ids, _, _ in items)
            pad = self.tokenizer.pad_token_id
            batch = torch.tensor(
                [[*ids, *[pad] * (max_lb - len(ids))] for _, ids, _, _ in items],
                device=self.device)
            # expand shared prefix KVs across the batch (one call, no per-branch copy)
            base_past = base_past.batch_repeat_interleave(k)
            # 4D isolation mask: row r, suffix query j attends prefix + own suffix[<=j]
            big = np.zeros((k, max_lb, len(state_ids) + max_lb), dtype=bool)
            for r, (_, ids, _, _) in enumerate(items):
                lr = len(ids)
                big[r, :lr, :len(state_ids)] = True
                for j in range(lr):
                    big[r, j, len(state_ids):len(state_ids) + j + 1] = True
            attn = torch.where(torch.from_numpy(big), 0.0, BLOCKED
                               ).unsqueeze(1).to(self.device, dtype=torch.float32)
            fw = self.model(input_ids=batch, past_key_values=base_past,
                            attention_mask=attn, use_cache=False,
                            output_hidden_states=use_heads)
            for r, (qname, ids, opts, kind) in enumerate(items):
                pos = len(ids) - 1
                hidden_row = fw.hidden_states[-1][r, pos, :] if use_heads else None
                out[qname] = self._decode_branch(kind, opts, fw.logits[r, pos, :],
                                                 hidden_row, temperature)
        assert self.generated_tokens == 0
        return out

    def score_all_semantic(self, state, spec, temperature=1.0):
        """Per-option semantic scoring: each option's token span is mean-pooled
        from ONE forward pass; the SHARED sem_head turns spans into Choice logits.
        No slot positions, no letter priors — binding is purely semantic.
        Noul/Score questions fall back to the standard paths."""
        import numpy as np
        torch = self.torch
        from .semantic import build_semantic_ids
        assert self.sem_head is not None, "semantic scoring needs sem_ckpt"
        out = {}
        for qname, q in spec.items():
            if not isinstance(q, ChoiceQuestion):
                sub = self.score_all(state, {qname: q}, temperature=temperature)
                out[qname] = sub[qname]
                continue
            options = list(q.choices)
            input_ids, spans = build_semantic_ids(self.tokenizer, state, q.question,
                                                  options)
            ids = torch.tensor([input_ids], device=self.device)
            with torch.no_grad():
                h = self.model(input_ids=ids, use_cache=False,
                               output_hidden_states=True).hidden_states[-1][0]
            reps = []
            for (a, b) in spans:
                if b <= a:
                    reps.append(h[-1, :])
                else:
                    reps.append(h[a:b, :].float().mean(0))
            H = torch.stack(reps).unsqueeze(0).float()
            with torch.no_grad():
                dev = next(self.sem_head.parameters()).device
                p = self.sem_head.probs(H.to(dev)).float().cpu().numpy()[0]
            out[qname] = np.asarray(p / p.sum(), dtype=np.float64)
        assert self.generated_tokens == 0
        return out


def semantic_ids(tokenizer, state, question, options):
    """Legacy shim: exact-span construction now lives in smalljev.semantic."""
    from .semantic import build_semantic_ids
    return build_semantic_ids(tokenizer, state, question, list(options))


def get_backend(backend):
    if backend == "stub" or backend is None:
        return StubBackend()
    if isinstance(backend, str):
        return HFBackend(backend)
    return backend
