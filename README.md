# smalljev

the open TypeSafe Jev that runs on your mama's phone. 2.5B params, one forward pass, zero generated tokens.

---

## install + run

```bash
# 1. clone + install
git clone https://github.com/isHeSatoshi/smalljev
cd smalljev
pip install -e ".[bench,test]"

# 2. download the model weights + LoRA from HuggingFace
huggingface-cli download isHeSatoshi/smalljev-semantic-v9 \
    --local-dir ./weights/semantic-v9

# 3. run
python -c "
from smalljev import decide
from smalljev.model import HFBackend

backend = HFBackend(
    model_id='openbmb/MiniCPM5-2B-Base',
    adapter_id='isHeSatoshi/smalljev-semantic-v9',
    sem_ckpt='isHeSatoshi/smalljev-semantic-v9',
)

out = decide(
    'the package arrived broken and the customer wants a refund',
    {'intent':   {'type': 'choice', 'question': 'primary issue?',
                  'choices': ['damaged', 'wrong_item', 'late', 'other']},
     'escalate': {'type': 'noul',   'question': 'needs a human right now'},
     'urgency':  {'type': 'score',  'question': 'how urgent?',
                  'levels': ['low', 'medium', 'high']}},
    backend=backend,
)
print(out)
"
```

needs CUDA GPU. backbone (~5 GB) auto-downloads from HF on first call.

**license** apache-2.0.
