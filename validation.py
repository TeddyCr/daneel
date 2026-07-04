import json
import logging
import os

from inference import BASE_MODEL, load, run

logger = logging.getLogger(__name__)

VALIDATION_SET = "./validation_set.jsonl"
LOG_DIR = "./eval_logs"


def prompt_messages(messages):
    user_idx = next(i for i, m in enumerate(messages) if m["role"] == "user")
    return messages[:user_idx + 1]


def expected_message(messages):
    user_idx = next(i for i, m in enumerate(messages) if m["role"] == "user")
    return messages[user_idx + 1]


def evaluate(model, tokenizer, records, label, device):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{label}.jsonl")

    with open(log_path, "w") as log_file:
        for i, rec in enumerate(records, 1):
            messages = rec["messages"]
            tools = rec.get("tools")
            prompt = prompt_messages(messages)

            output = run(model, tokenizer, prompt, tools, device)

            log_file.write(json.dumps({
                "index": i,
                "model": label,
                "input": prompt,
                "output": output,
                "expected": expected_message(messages),
            }, ensure_ascii=False) + "\n")
            log_file.flush()

            logger.info("[%s] %d/%d", label, i, len(records))

    logger.info("[%s] wrote per-example log to %s", label, log_path)


def validate(validation_set: str = VALIDATION_SET, device: str = "auto", model_id: str = BASE_MODEL):
    with open(validation_set) as f:
        records = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d validation records.", len(records))

    model, tokenizer, dev = load(model=model_id, device=device)

    evaluate(model, tokenizer, records, "adapter", dev)

    with model.disable_adapter():
        evaluate(model, tokenizer, records, "base", dev)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    validate()
