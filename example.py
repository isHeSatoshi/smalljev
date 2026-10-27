from smalljev import decide
from smalljev.model import HFBackend

backend = HFBackend(
    model_id="./weights/base",
    adapter_id="./weights/semantic-v9/semantic-v9-lora",
    sem_ckpt="./weights/semantic-v9/semantic-v9-scorer.pt"
)

out = decide(
    "the package arrived broken and the customer wants a refund",
    {"intent":   {"type": "choice", "question": "primary issue?",
                  "choices": ["damaged", "wrong_item", "late", "other"]},
     "escalate": {"type": "noul",   "question": "needs a human right now"},
     "urgency":  {"type": "score",  "question": "how urgent?",
                  "levels": ["low", "medium", "high"],
                  "values": (0.1, 0.5, 0.9)}},
    backend=backend, temperature=1.0
)

print("\n\n\n\n\n-----------------------------")
print(out)
