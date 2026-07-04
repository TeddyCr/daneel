import logging
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback

from dataset import load_dataset
from device import resolve_device
from inference import BASE_MODEL
from model import CHAT_TEMPLATE_PATH, load_model, load_tokenizer, lora_config

logger = logging.getLogger(__name__)

SMALL_PRESET = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 5,
    "logging_steps": 5,
    "max_length": 2560,
    "eval_steps": 25,
}


def get_sft_config(
        output_dir: str = "./weights/checkpoints",
        num_train_epochs: int = 10,
        per_device_train_batch_size: int = 1,
        per_device_eval_batch_size: int = 1,
        gradient_accumulation_steps: int = 16,
        learning_rate: float = 2e-4,
        lr_scheduler_type: str = "cosine",
        warmup_steps: int = 50,
        bf16: bool = False,
        fp16: bool = False,
        greater_is_better=False,
        metric_for_best_model: str = "eval_loss",
        logging_steps: int = 10,
        eval_strategy: str = "steps",
        eval_steps: int = 50,
        save_strategy: str = "steps",
        save_steps: int = 50,
        dataloader_pin_memory: bool = False,
        gradient_checkpointing: bool = True,
        optim: str = "adamw_torch",
        max_length: int = 2560,
        load_best_model_at_end=True,
    ) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        bf16=bf16,
        fp16=fp16,
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        dataloader_pin_memory=dataloader_pin_memory,
        optim=optim,
        gradient_checkpointing=gradient_checkpointing,
        max_length=max_length,
        load_best_model_at_end=load_best_model_at_end,
        greater_is_better=greater_is_better,
        metric_for_best_model=metric_for_best_model,
        assistant_only_loss=True,
    )


def train(
        small: bool = False,
        device: str = "auto",
        model_id: str = BASE_MODEL,
        chat_template: str | None = None,
        training_set: str = "training_set.jsonl",
        adapter_out: str = "./adapter",
        **overrides,
    ):
    dev, dtype = resolve_device(device)
    logger.info("Training %s on %s (small=%s)", model_id, dev, small)

    model = load_model(model_id=model_id, device=dev, dtype=dtype)
    tokenizer = load_tokenizer(
        model_id=model_id,
        device_map=dev,
        chat_template_path=chat_template or CHAT_TEMPLATE_PATH,
    )
    lora = lora_config()

    config_args = {**SMALL_PRESET, **overrides} if small else overrides
    config = get_sft_config(**config_args)

    dataset = load_dataset(source=training_set)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_set, eval_set = split["train"], split["test"]
    if small:
        train_set = train_set.shuffle(seed=42).select(range(50))
        eval_set = eval_set.shuffle(seed=42).select(range(10))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_set,
        eval_dataset=eval_set,
        args=config,
        peft_config=lora,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001),
        ],
    )
    trainer.train()
    trainer.save_model(adapter_out)
    logger.info("Saved adapter to %s", adapter_out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
