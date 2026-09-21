"""JevBench adapter for smalljev (in-process, native softmax over typed decision heads).

This adapter translates canonical JevBench tasks into smalljev `decide()` calls and
returns a JevBench `DecisionResult` with the model's native probability distribution
over the exact label set. Nothing is generated; the distribution is the model's own
softmax over the declared option slots / Noul bit / Score levels (i.e. probability_origin
= native).

The adapter is intentionally narrow: it never modifies the smalljev workspace, never
trains anything, and never repairs an invalid distribution.

Mapping per question type:

  * noul   - The criterion text becomes the question statement; the adapter asks
             decide() for a Noul probability P(true). The result is stored as
             {"yes": p, "no": 1-p}. JevBench labels are exactly ["no", "yes"].
  * choice - The ordered labels are passed verbatim as `choices` (preserving order).
             The adapter returns {label: probability} for every label.
  * score  - The ordered level descriptions become `levels` and the integer indices
             become the `values`. The adapter returns {level_index: probability}.

Rubric treatment (per the IMPLEMENTATION.md "rubric reaches every adapter" test):

  Every native adapter appends the exact same rubric to the instruction text so that
  no native adapter sees less rubric information than another. For smalljev, the
  rubric is encoded as a concise "criteria: <map>" suffix on the question so that the
  text contains the same semantic content as the JevBench criteria field.

State truncation:

  The model is a 2.5B-param MiniCPM5 decoder with 131K context. We do not impose any
  state token cap; smalljev's tokenizer will handle long states. If the model
  truncates internally, it will be observable via runtime state.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Optional

# Read-only path to the live smalljev source. Set via SMALLJEV_PATH environment
# variable; default points at the user's smalljev checkout.
DEFAULT_SMALLJEV_PATH = os.environ.get(
    "SMALLJEV_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "smalljev")),
)


def _add_smalljev_to_path():
    """Inject the smalljev package directory into sys.path."""
    if DEFAULT_SMALLJEV_PATH not in sys.path:
        sys.path.insert(0, DEFAULT_SMALLJEV_PATH)


_add_smalljev_to_path()

from jevbench.adapters.base import DecisionResult  # noqa: E402


class SmalljevAdapter:
    """JevBench adapter for smalljev.

    cost_basis = local GPU, no provider tariff. The harness will compute a price if
    both `price_input_per_m` and `price_output_per_m` are provided; otherwise the
    per-decision cost stays null. We never fabricate a price.
    """

    name = "smalljev"
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
        backbone: Optional[str] = None,
        adapter_id: Optional[str] = None,
        heads_ckpt: Optional[str] = None,
        variant_label: Optional[str] = None,
    ):
        self.path = endpoint
        self.model = model or variant_label or "smalljev"
        self.key_env = key_env
        self.timeout_s = timeout_s
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self.device = device
        self.threads = threads
        self.revision = revision
        self.backbone = backbone
        self.adapter_id = adapter_id
        self.heads_ckpt = heads_ckpt
        self.variant_label = variant_label or self.model
        self._backend = None
        self.torch_version = None
        self.python = sys.version.split()[0]
        self.load_s: Optional[float] = None

    # -- backbone: smalljev.decide(state, questions, backend) ------

    def load(self):
        if self._backend is None:
            import torch  # noqa: F401
            from smalljev.model import HFBackend

            if self.device == "cuda":
                torch.set_num_threads(self.threads)
            t0 = time.perf_counter()
            self._backend = HFBackend(
                model_id=self.backbone,
                adapter_id=self.adapter_id,
                heads_ckpt=self.heads_ckpt,
            )
            self.load_s = time.perf_counter() - t0
            self.torch_version = torch.__version__
            import transformers
            self.transformers_version = transformers.__version__
        return self._backend

    # -- translation helpers ----------------------------------------

    def _format_state(self, task) -> str:
        """Coerce task.state to a string. Dict states are JSON-encoded."""
        s = task.state
        if isinstance(s, str):
            return s
        return json.dumps(s, ensure_ascii=False)

    def _labels_and_rubric(self, task):
        """Return (ordered_labels, rubric_dict) in the shape the smalljev spec expects."""
        qtype = task.question["type"]
        crit = task.question.get("criteria") or {}
        if qtype == "noul":
            # labels are ["no", "yes"]; rubric true/false -> short text
            rubric = {"no": crit.get("false", "No"), "yes": crit.get("true", "Yes")}
            return ["no", "yes"], rubric
        if qtype == "score":
            # criteria is a list of level descriptions, labels are integer indices as strings
            if not isinstance(crit, list):
                raise ValueError(f"score task {task.id}: criteria must be a list, got {type(crit)}")
            labels = [str(i) for i in range(len(crit))]
            rubric = {labels[i]: (crit[i] or labels[i]) for i in range(len(crit))}
            return labels, rubric
        # choice: criteria is {label: description}
        labels = list(task.labels)
        if not isinstance(crit, dict):
            raise ValueError(f"choice task {task.id}: criteria must be a dict, got {type(crit)}")
        # We do NOT reorder the labels -- the JevBench `labels` field is already in
        # the order the gold answer was authored with. The adapter preserves that.
        rubric = {k: (crit.get(k) or k) for k in labels}
        return labels, rubric

    def _build_questions(self, task, labels, rubric) -> dict:
        """Translate a single JevBench task into the smalljev spec dict.

        The rubric is appended to the question text (same as other native adapters)
        so the model receives the same rubric information as any other native flavour.
        The labels are preserved exactly as authored.
        """
        qtype = task.question["type"]
        instr = task.question["instructions"]
        rubric_str = json.dumps(rubric, ensure_ascii=False)
        question_text = f"{instr}\nAllowed answers and rubric: {rubric_str}"

        qname = "decision"
        if qtype == "noul":
            return {qname: {"type": "noul", "question": question_text}}
        if qtype == "choice":
            return {qname: {"type": "choice", "question": question_text, "choices": labels}}
        if qtype == "score":
            # Build an ordinal ScoreQuestion: levels are the level descriptions (which
            # is the rubric text); values are the integer indices so they are strictly
            # increasing. The model's OrdinalScoreHead treats each slot i as level i.
            levels = [rubric[lab] for lab in labels]
            values = [float(i) for i in range(len(labels))]
            return {qname: {"type": "score", "question": question_text,
                             "levels": levels, "values": values}}
        raise ValueError(f"unknown question type: {qtype}")

    def build_request(self, task) -> dict:
        labels, rubric = self._labels_and_rubric(task)
        questions = self._build_questions(task, labels, rubric)
        state = self._format_state(task)
        return {"state": state, "questions": questions, "labels": labels}

    # -- adapter contract ------------------------------------------

    def run(self, task) -> DecisionResult:
        res = DecisionResult(
            adapter=self.name, ok=False,
            probs_source="native",
            model=self.variant_label,
        )
        body = self.build_request(task)
        res.request_body = body
        try:
            backend = self.load()
        except Exception as e:
            res.error = f"load failed: {type(e).__name__}: {str(e)[:250]}"
            return res

        # Time the actual decide() call (not the load time).
        try:
            from smalljev import decide
        except Exception as e:
            res.error = f"import failed: {type(e).__name__}: {str(e)[:250]}"
            return res

        t0 = time.perf_counter()
        try:
            out = decide(body["state"], body["questions"], backend=backend)
        except Exception as e:
            res.latency_s = time.perf_counter() - t0
            res.error = f"{type(e).__name__}: {str(e)[:300]}"
            return res
        res.latency_s = time.perf_counter() - t0

        ans = out.get("decision") or {}
        meta = out.get("_meta", {})
        res.raw = {
            "answers": {"decision": ans},
            "runtime": {
                "device": self.device,
                "threads": self.threads,
                "revision": self.revision,
                "backbone": self.backbone,
                "adapter_id": self.adapter_id,
                "heads_ckpt": self.heads_ckpt,
                "variant_label": self.variant_label,
                "generated_tokens": meta.get("generated_tokens"),
                "n_questions": meta.get("n_questions"),
                "probability_origin": "native-option-slot-softmax",
                "torch": self.torch_version,
                "transformers": getattr(self, "transformers_version", None),
                "load_s": self.load_s,
            },
        }

        # Translate smalljev result -> JevBench probability dict over exact labels.
        try:
            qtype = task.question["type"]
            labels = body["labels"]
            if qtype == "noul":
                p = float(ans.get("probability"))
                if not (0.0 <= p <= 1.0):
                    raise ValueError(f"noul out of range: {p}")
                res.probs = {"yes": p, "no": 1.0 - p}
            elif qtype == "choice":
                probs = ans.get("probabilities")
                if probs is None:
                    raise ValueError("missing probabilities")
                if list(ans.get("values", [])) != labels:
                    raise ValueError(
                        f"label order mismatch: returned {ans.get('values')!r} "
                        f"vs expected {labels!r}"
                    )
                res.probs = {labels[i]: float(probs[i]) for i in range(len(labels))}
            elif qtype == "score":
                probs = ans.get("probabilities")
                levels = ans.get("levels")
                if probs is None or levels is None:
                    raise ValueError("missing score probabilities/levels")
                # Build the expected label list from the criteria (in the same order
                # the adapter constructed them).
                crit = task.question.get("criteria") or []
                expected_labels = [str(i) for i in range(len(crit))]
                # The model's `levels` are the level descriptions (not the indices),
                # so we map back: find each expected label by matching level text.
                if len(probs) != len(expected_labels):
                    raise ValueError(
                        f"score length mismatch: got {len(probs)} probs, expected "
                        f"{len(expected_labels)}"
                    )
                res.probs = {expected_labels[i]: float(probs[i])
                             for i in range(len(expected_labels))}
            else:
                raise ValueError(f"unknown question type: {qtype}")
        except (KeyError, TypeError, ValueError) as e:
            res.error = f"answer parse failed: {e}"
            return res

        res.ok = True
        return res

    def reserve_estimate(self, task) -> float:
        # Local GPU has no per-token tariff. The runner will keep cost_usd = None.
        return 0.0