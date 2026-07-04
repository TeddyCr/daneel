# daneel

LoRA fine-tuning, validation, and inference for a Qwen2.5-1.5B agent adapter.

All functionality is exposed through the `daneel` CLI with three commands: `train`, `validate`, and `infer`.

## Install

```bash
pip install -e .
```

This installs the `daneel` console command. You can also run it without installing:

```bash
python cli.py <command> ...
```

## Common options

Every command takes:

- `--device {auto,cuda,mps,cpu}` (default `auto`) — `auto` prefers CUDA, then Apple MPS, then CPU. Requesting an unavailable device raises an error. The dtype is picked to match (cuda→bfloat16, mps→float16, cpu→float32).
- `--model` (default `Qwen/Qwen2.5-1.5B-Instruct`) — the base model, given as a Hugging Face model id or a local path.

## `daneel train`

Fine-tune the LoRA adapter and save it to `./adapter`.

```bash
daneel train                          # full run, auto device
daneel train --small                  # quick run on a tiny subset
daneel train --device mps
daneel train --model Qwen/Qwen2.5-3B-Instruct
daneel train --training-set ./my_data.jsonl --chat-template ./my_template.jinja
daneel train --epochs 5 --lr 1e-4 --batch-size 2
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--small` | off | Quick run on 50 train / 10 eval examples with a lightweight preset |
| `--model` | `Qwen/Qwen2.5-1.5B-Instruct` | base model id or path |
| `--chat-template` | bundled Qwen template | path to a chat template (`.jinja`) |
| `--training-set` | `training_set.jsonl` | path to the training jsonl |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `--epochs` | 10 | `num_train_epochs` |
| `--lr` | 2e-4 | `learning_rate` |
| `--batch-size` | 1 | `per_device_train_batch_size` |
| `--grad-accum` | 16 | `gradient_accumulation_steps` |
| `--max-length` | 2560 | max sequence length |
| `--warmup-steps` | 50 | LR warmup steps |
| `--output-dir` | `./weights/checkpoints` | checkpoint directory |

Anything not exposed as a flag (scheduler, eval/save cadence, optimizer, `assistant_only_loss`) keeps its default in `training.py`.

## `daneel validate`

Run the validation set through both the adapter and the base model, writing per-example logs to `./eval_logs/adapter.jsonl` and `./eval_logs/base.jsonl`.

```bash
daneel validate
daneel validate --validation-set ./validation_set.jsonl --device mps
```

Each log line contains the input messages, the raw model output, and the gold assistant turn.

## `daneel infer`

Answer a single message and print the reply to stdout.

```bash
daneel infer "What's the weather in Paris?"
daneel infer "Summarize this" --system "You are concise."
daneel infer "hello" --no-adapter            # base model only
daneel infer "hello" --max-new-tokens 256
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--system` | none | Optional system prompt |
| `--no-adapter` | off | Use the base model without the LoRA adapter |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `--max-new-tokens` | 512 | Generation cap |

## Logging

Status output uses the `logging` module. Set verbosity with the global `--log-level` (before the command):

```bash
daneel --log-level DEBUG train --small
```

## Project layout

| File | Purpose |
|---|---|
| `cli.py` | argparse entrypoint (`daneel`) |
| `device.py` | device/dtype resolution |
| `dataset.py` | dataset loading / JSONL → Arrow conversion |
| `model.py` | base model, tokenizer, LoRA config |
| `training.py` | SFT config + `train()` |
| `validation.py` | adapter-vs-base evaluation |
| `inference.py` | model loading, generation, `infer()` |
