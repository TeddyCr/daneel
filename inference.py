import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from device import resolve_device

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "./adapter"


def load(model: str = BASE_MODEL, adapter: str = ADAPTER, device: str = "auto"):
    dev, dtype = resolve_device(device)
    base_model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, device_map=dev)
    tokenizer = AutoTokenizer.from_pretrained(model)
    return PeftModel.from_pretrained(base_model, adapter), tokenizer, dev


def run(model, tokenizer, messages, tools=None, device="mps", max_new_tokens=512):
    inputs = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()


def infer(
    message: str,
    system: str | None = None,
    tools: list | None = None,
    use_adapter: bool = True,
    model: str = BASE_MODEL,
    adapter: str = ADAPTER,
    device: str = "auto",
    max_new_tokens: int = 512,
) -> str:
    loaded, tokenizer, dev = load(model, adapter, device)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": message})

    if use_adapter:
        return run(loaded, tokenizer, messages, tools, dev, max_new_tokens)

    with loaded.disable_adapter():
        return run(loaded, tokenizer, messages, tools, dev, max_new_tokens)
