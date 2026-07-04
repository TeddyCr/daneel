import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
import torch


def load_model(
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code: bool = True,
        device: str = "mps",
        dtype: torch.dtype = torch.float16
    ):
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device,
        trust_remote_code=trust_remote_code,
    )


CHAT_TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_template_qwen.jinja")


def load_tokenizer(
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device_map: str = "mps",
        padding_side: str = "right",
        trust_remote_code: bool = True,
        chat_template_path: str = CHAT_TEMPLATE_PATH
    ):
    # Override the stock template with chat_template_qwen.jinja, which wraps
    # assistant turns in {% generation %} markers so SFTConfig(assistant_only_loss=True)
    # can mask loss to the assistant reply only.
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        device_map=device_map,
        trust_remote_code=trust_remote_code
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side

    with open(chat_template_path) as f:
        tokenizer.chat_template = f.read()

    return tokenizer

def lora_config(
        task_type: TaskType = TaskType.CAUSAL_LM,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: list = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias: str = "none"
):
    """
    Configures the LoRA (Low-Rank Adaptation) settings for fine-tuning the model.
    The LoraConfig class is used to specify the parameters for LoRA, including the
    task type (TaskType.CAUSAL_LM), inference mode (inference_mode=True), rank (r=8),
    alpha (lora_alpha=16), and dropout rate (lora_dropout=0.05).
    """
    return LoraConfig(
        task_type=task_type,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
    )
    
