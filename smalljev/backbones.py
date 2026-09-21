"""Backbone abstraction for the encoder experiment: every arm exposes encode().

Readout semantics follow each architecture's native convention (verified from
cards): CLS for BERT-family true encoders, mean-pool for CLS-less bidirectional
models, last-token for decoder-style representation models and causal decoders.
Same HeadsBundle plugs into any readout.
"""
import torch


def cls_readout(h):
    return h[:, 0, :]


def mean_readout(h, mask):
    w = mask.unsqueeze(-1).to(h.dtype)
    return (h * w).sum(1) / w.sum(1).clamp_min(1e-6)


def last_readout(h, mask, device):
    last = mask.sum(1) - 1
    return h[torch.arange(len(last), device=device), last, :]

BACKBONES = {
    "minicpm5": {
        "model_id": "openbmb/MiniCPM5-2B-Base", "license": "Apache-2.0",
        "params_b": 2.52, "langs": "en+zh+", "arch": "causal-decoder",
        "readout": "last", "context": 131072},
    "bidirlm": {
        "model_id": "BidirLM/BidirLM-1.7B-Base", "license": "Apache-2.0",
        "params_b": 1.7, "langs": "88", "arch": "true-encoder",
        "readout": "mean", "context": 32768,
        "revision": "transformers-v4", "trust_remote_code": True},
    "ettin-1b": {
        "model_id": r"D:\Project\smalljev\evals\weights\ettin-encoder-1b",
        "hub_id": "jhu-clsp/ettin-encoder-1b", "license": "MIT",
        "params_b": 1.0, "langs": "en", "arch": "true-encoder",
        "readout": "cls", "context": 8192},
    "ettin-from-dec-1b": {
        "model_id": r"D:\Project\smalljev\evals\weights\ettin-enc-from-dec-1b",
        "hub_id": "jhu-clsp/ettin-enc-from-dec-1b", "license": "MIT",
        "params_b": 1.0, "langs": "en", "arch": "true-encoder-decoder-pretrained",
        "readout": "cls", "context": 8192},
    "modernbert-base": {
        "model_id": "answerdotai/ModernBERT-base", "license": "Apache-2.0",
        "params_b": 0.15, "langs": "en", "arch": "true-encoder",
        "readout": "cls", "context": 8192},
    "modernbert-large": {
        "model_id": "answerdotai/ModernBERT-large", "license": "Apache-2.0",
        "params_b": 0.40, "langs": "en", "arch": "true-encoder",
        "readout": "cls", "context": 8192},
    "f2llm-4b": {
        "model_id": "codefuse-ai/F2LLM-v2-4B", "license": "Apache-2.0",
        "params_b": 4.0, "langs": "200+", "arch": "decoder-repr",
        "readout": "last", "context": 32768},
    "harrier-0.6b": {
        "model_id": "microsoft/harrier-oss-v1-0.6b", "license": "MIT",
        "params_b": 0.6, "langs": "multi", "arch": "decoder-repr",
        "readout": "last", "context": 32768},
}


class Backbone:
    def __init__(self, key, device="cuda", dtype=None):
        from transformers import AutoModel, AutoTokenizer
        self.spec = BACKBONES[key]
        self.key = key
        dtype = dtype or (torch.bfloat16 if device == "cuda" else torch.float32)
        kw = {}
        if self.spec.get("revision"):
            kw["revision"] = self.spec["revision"]
        if self.spec.get("trust_remote_code"):
            kw["trust_remote_code"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec["model_id"], **kw)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            self.spec["model_id"], torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None, **kw).eval()
        self.device = next(self.model.parameters()).device
        self.hidden_size = self.model.config.hidden_size
        self.readout = self.spec["readout"]

    def encode(self, texts, max_len=512, train=False):
        """Joint-encode texts -> readout vectors (B, H) on device (model dtype)."""
        enc = self.tokenizer(list(texts), return_tensors="pt", padding=True,
                             truncation=True, max_length=max_len).to(self.device)
        enc.pop("token_type_ids", None)
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            kw = {} if self.spec["arch"].startswith("true-encoder") else {"use_cache": False}
            h = self.model(**enc, **kw).last_hidden_state
        mask = enc["attention_mask"]
        if self.readout == "cls":
            return cls_readout(h)
        if self.readout == "mean":
            return mean_readout(h, mask)
        # last meaningful token
        return last_readout(h, mask, self.device)

    def joint_text(self, state, question, options):
        return f"{state} [SEP] {question} Options: " + " ".join(f"({c})" for c in options)
