"""Debug: isolate import crash."""
import sys, os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

print("1. import torch", flush=True)
import torch
print(f"   OK: {torch.__version__}", flush=True)

print("2. import transformers", flush=True)
import transformers
print(f"   OK: {transformers.__version__}", flush=True)

print("3. import peft", flush=True)
import peft
print(f"   OK: {peft.__version__}", flush=True)

print("4. import accelerate", flush=True)
import accelerate
print(f"   OK: {accelerate.__version__}", flush=True)

print("5. Basic model load test", flush=True)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "checkpoints/qwen3.5-9b-sft-distill-merged",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
print(f"   OK: {type(model).__name__}", flush=True)

print("ALL DONE", flush=True)
