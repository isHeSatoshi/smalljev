"""JevBench adapter for the smalljev semantic arm (semantic-v4 .. semantic-v7).

In-process, native softmax over per-option span representations + Noul bit head.
Identical contract to the legacy smalljev_adapter.SmalljevAdapter (native
probability_origin, DecisionResult with probs/ok/raw/latency_s) so the existing
JevBench runner/harness/summarize stack is unchanged.

Loads:
    <prefix>-lora/               PeftModel LoRA adapter
    <prefix>-scorer.pt           OptionScorerHead (shared, permutation-equivariant)
    <prefix>-noulscore.pt        {"hidden_size", "noul": BinaryNoulHead, "score": OrdinalScoreHead}

Mapping per question type:

  * noul   - last-token hidden readout through BinaryNoulHead; P(true). Result
             over JevBench labels ["no", "yes"] = {no: 1-p, yes: p}.
  * choice - last-token hidden used only to derive one forward; per-option span
             mean-pooled through OptionScorerHead; full softmax distribution.
  * score  - identical to choice, but the option set is the level descriptions
             authored in the task criteria. Probabilities are stored over the
             integer index labels (0..N-1), exactly matching eval_semantic.py.

Rubric is appended to the question text (same as legacy adapter, same as every
native adapter in the JevBench harness).

This adapter does NOT touch the live smalljev source. It only IMPORTS the
existing smalljev packages read-only via sys.path injection.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

DEFAULT_SMALLJEV_PATH = os.environ.get(
    "SMALLJEV_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "smalljev")),
)


def _add_smalljev_to_path():
    if DEFAULT_SMALLJEV_PATH not in sys.path:
        sys.path.insert(0, DEFAULT_SMALLJEV_PATH)


_add_smalljev_to_path()

from jevbench.adapters.base import DecisionResult  # noqa: E402


class SmalljevSemanticAdapter:
    name = "smalljev_semantic"
    cost_basis = "local_gpu_no_provider_tariff"

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        key_env: str = "",
        timeout_s: Optional[float] = None,
        price_input_per_m: Optional[float] = None,
        price_output_per_m: Optional[float] = None,
        device: str = "cuda",
        threads: int = 2,
        revision: Optional[str] = None,
        backbone: str = "openbmb/MiniCPM5-2B-Base",
        adapter_dir: Optional[str] = None,           # <prefix>-lora/
        scorer_ckpt: Optional[str] = None,           # <prefix>-scorer.pt
        noulscore_ckpt: Optional[str] = None,        # <prefix>-noulscore.pt
        variant_label: Optional[str] = None,
    ):
        self.path = endpoint
        self.model = model or variant_label or "smalljev_semantic"
        self.key_env = key_env
        self.timeout_s = timeout_s
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self.device = device
        self.threads = threads
        self.revision = revision
        self.backbone = backbone
        self.adapter_dir = adapter_dir
        self.scorer_ckpt = scorer_ckpt
        self.noulscore_ckpt = noulscore_ckpt
        self.variant_label = variant_label or self.model
        self._loaded = False
        self._backend = None
        self._scorer = None
        self._noul = None
        self._score_head = None
        self._device_real = None
        self.load_s = None
        self.torch_version = None
        self.transformers_version = None
        self.peft_version = None

    # --- backend loading ---------------------------------------------------------

    def load(self):
        if self._loaded:
            return
        if not (self.adapter_dir and self.scorer_ckpt and self.noulscore_ckpt):
            raise ValueError(
                "Semantic adapter needs --adapter-dir, --scorer-ckpt, --noulscore-ckpt")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from smalljev.heads import OptionScorerHead, BinaryNoulHead, OrdinalScoreHead

        if self.device == "cuda":
            torch.set_num_threads(self.threads)
        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(self.backbone, trust_remote_code=False)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            self.backbone, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=False, attn_implementation="eager").eval()
        model = PeftModel.from_pretrained(base, self.adapter_dir).eval()
        self._backend = (tok, model)
        self._scorer = OptionScorerHead.load(self.scorer_ckpt)
        blob = torch.load(self.noulscore_ckpt, map_location="cpu", weights_only=True)
        noul = BinaryNoulHead(blob["hidden_size"])
        noul.load_state_dict(blob["noul"])
        self._noul = noul
        score = OrdinalScoreHead(blob["hidden_size"], 8)
        score.load_state_dict(blob["score"])
        self._score_head = score
        # Move heads to GPU if available
        if self.device == "cuda" and torch.cuda.is_available():
            self._scorer = self._scorer.cuda().eval()
            self._noul = self._noul.cuda().eval()
            self._score_head = self._score_head.cuda().eval()
            self._device_real = torch.device("cuda")
        else:
            self._scorer = self._scorer.eval()
            self._noul = self._noul.eval()
            self._score_head = self._score_head.eval()
            self._device_real = torch.device("cpu")
        self.load_s = time.perf_counter() - t0
        self.torch_version = torch.__version__
        import transformers
        self.transformers_version = transformers.__version__
        try:
            import peft
            self.peft_version = peft.__version__
        except Exception:
            self.peft_version = None
        self._loaded = True

    # --- task translation -------------------------------------------------------

    def _format_state(self, task) -> str:
        s = task.state
        if isinstance(s, str):
            return s
        return json.dumps(s, ensure_ascii=False)

    def _labels_and_rubric(self, task):
        qtype = task.question["type"]
        crit = task.question.get("criteria") or {}
        if qtype == "noul":
            return ["no", "yes"], {"no": crit.get("false", "No"),
                                   "yes": crit.get("true", "Yes")}
        if qtype == "score":
            if not isinstance(crit, list):
                raise ValueError(f"score task {task.id}: criteria must be a list")
            labels = [str(i) for i in range(len(crit))]
            rubric = {labels[i]: (crit[i] or labels[i]) for i in range(len(crit))}
            return labels, rubric
        if not isinstance(crit, dict):
            raise ValueError(f"choice task {task.id}: criteria must be a dict")
        labels = list(task.labels)
        rubric = {k: (crit.get(k) or k) for k in labels}
        return labels, rubric

    def _question_text(self, task, rubric):
        instr = task.question["instructions"]
        rubric_str = json.dumps(rubric, ensure_ascii=False)
        return f"{instr}\nAllowed answers and rubric: {rubric_str}"

    def build_request(self, task):
        labels, rubric = self._labels_and_rubric(task)
        return {
            "state": self._format_state(task),
            "labels": labels,
            "rubric": rubric,
            "question": self._question_text(task, rubric),
            "qtype": task.question["type"],
        }

    # --- inference --------------------------------------------------------------

    def _semantic_choice_probs(self, items, batch=8):
        """items: list of (state, question, options). Returns list of prob arrays.

        SMALLJEV_ENSEMBLE=k (default 1): average probs over k option orders
        (identity + rotations). Order-averaging removes residual position bias
        at the cost of k forwards; latency scales linearly.
        """
        import os as _os
        K = max(1, int(_os.environ.get("SMALLJEV_ENSEMBLE", "1")))
        if K == 1:
            return self._semantic_choice_probs_single(items, batch)
        import numpy as _np
        acc = None
        for k in range(K):
            perms = []
            for (state, q, opts) in items:
                n = len(opts)
                shift = k % n
                perms.append((state, q, list(opts[shift:] + opts[:shift]), shift))
            Ps = self._semantic_choice_probs_single(
                [(s, q, o) for (s, q, o, _) in perms], batch)
            unrot = []
            for (s, q, o, shift), p in zip(perms, Ps):
                n = len(p)
                back = _np.empty(n)
                for j in range(n):
                    back[(j + shift) % n] = p[j]
                unrot.append(back / back.sum())
            acc = unrot if acc is None else [a + b for a, b in zip(acc, unrot)]
        return [a / K for a in acc]

    def _semantic_choice_probs_single(self, items, batch=8):
        """items: list of (state, question, options). Returns list of prob arrays."""
        tok, model = self._backend
        import numpy as np
        from smalljev.semantic import build_semantic_ids
        out = []
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            ids_list, spans_list = [], []
            for state, q, opts in chunk:
                ids, spans = build_semantic_ids(tok, state, q, list(opts),
                                                max_len=2560)
                ids_list.append(ids)
                spans_list.append(spans)
            import torch as _t
            pad = tok.pad_token_id
            mx = max(len(x) for x in ids_list)
            input_ids = _t.tensor([[*x, *[pad] * (mx - len(x))] for x in ids_list],
                                  device=self._device_real)
            with _t.no_grad():
                H_all = model(input_ids=input_ids, use_cache=False,
                              output_hidden_states=True).hidden_states[-1].float()
                for r, spans in enumerate(spans_list):
                    reps = _t.stack([H_all[r, a:b, :].mean(0) for a, b in spans])
                    p = self._scorer.probs(reps.unsqueeze(0).to(self._device_real)
                                           ).float().cpu().numpy()[0]
                    out.append(np.asarray(p / p.sum(), dtype=np.float64))
        return out

    def _readout_hidden(self, items):
        """items: list of (question_text,). Returns list of (T,H) last-token hidden."""
        tok, model = self._backend
        import torch as _t
        prompts = [q for (q,) in items]
        enc = tok(prompts, return_tensors="pt", padding=True).to(self._device_real)
        with _t.no_grad():
            fw = model(**enc, output_hidden_states=True, use_cache=False)
        last_idx = (enc["attention_mask"].sum(1) - 1).tolist()
        ar = _t.arange(len(items), device=self._device_real)
        return fw.hidden_states[-1][ar, last_idx, :].float()

    # --- contract ---------------------------------------------------------------

    def run(self, task) -> DecisionResult:
        res = DecisionResult(
            adapter=self.name, ok=False,
            probs_source="native",
            model=self.variant_label,
        )
        body = self.build_request(task)
        res.request_body = body
        try:
            self.load()
        except Exception as e:
            res.error = f"load failed: {type(e).__name__}: {str(e)[:250]}"
            return res

        labels = body["labels"]
        qtype = body["qtype"]
        question = body["question"]
        state = body["state"]

        t0 = time.perf_counter()
        try:
            if qtype == "choice":
                triplets = [(state, question, labels)]
                probs_list = self._semantic_choice_probs(triplets, batch=4)
                if len(probs_list) != 1:
                    raise ValueError(f"expected 1 prob array, got {len(probs_list)}")
                p = probs_list[0]
                if len(p) != len(labels):
                    raise ValueError(
                        f"choice length mismatch: probs={len(p)} vs labels={len(labels)}")
                res.probs = {labels[i]: float(p[i]) for i in range(len(labels))}
            elif qtype == "noul":
                # Build yes/no options rendered as a choice (the prior eval path)
                # so we can re-use the semantic scorer; fallback to Noul bit head.
                triplets = [(state,
                             question + "\nAnswer with yes or no.",
                             ["no", "yes"])]
                p = self._semantic_choice_probs(triplets, batch=4)[0]
                # p[0] is for "no", p[1] for "yes"
                yes = float(p[1] / p.sum())
                res.probs = {"yes": yes, "no": 1.0 - yes}
            elif qtype == "score":
                # Score: level descriptions become the choice options; the
                # integer index label is what JevBench expects in `probs`.
                levels = [body["rubric"][lab] for lab in labels]
                triplets = [(state, question, levels)]
                p = self._semantic_choice_probs(triplets, batch=4)[0]
                if len(p) != len(labels):
                    raise ValueError(
                        f"score length mismatch: probs={len(p)} vs labels={len(labels)}")
                res.probs = {labels[i]: float(p[i]) for i in range(len(labels))}
            else:
                raise ValueError(f"unknown question type: {qtype}")
        except Exception as e:
            res.latency_s = time.perf_counter() - t0
            res.error = f"{type(e).__name__}: {str(e)[:300]}"
            return res

        res.latency_s = time.perf_counter() - t0
        res.raw = {
            "answers": {"decision": {"probabilities": list(res.probs.values())}},
            "runtime": {
                "device": str(self._device_real),
                "threads": self.threads,
                "revision": self.revision,
                "backbone": self.backbone,
                "adapter_id": self.adapter_dir,
                "scorer_ckpt": self.scorer_ckpt,
                "noulscore_ckpt": self.noulscore_ckpt,
                "variant_label": self.variant_label,
                "generated_tokens": 0,
                "n_questions": 1,
                "probability_origin": "native-option-scorer-span-softmax",
                "torch": self.torch_version,
                "transformers": self.transformers_version,
                "peft": self.peft_version,
                "load_s": self.load_s,
            },
        }
        # Schema validity check (analogous to legacy adapter)
        if any(v is None for v in res.probs.values()):
            res.error = "schema invalid: missing label"
            return res
        s = sum(res.probs.values())
        if not (0.99 <= s <= 1.01):
            res.error = f"schema invalid: probabilities sum to {s:.4f}"
            return res
        if not all(0.0 <= v <= 1.0 for v in res.probs.values()):
            res.error = "schema invalid: probability outside [0,1]"
            return res
        res.ok = True
        return res

    def reserve_estimate(self, task) -> float:
        return 0.0
