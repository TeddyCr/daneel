import argparse
import logging

DEVICES = ["auto", "cuda", "mps", "cpu"]
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _train(args):
    from training import train

    overrides = {}
    if args.epochs is not None:
        overrides["num_train_epochs"] = args.epochs
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.batch_size is not None:
        overrides["per_device_train_batch_size"] = args.batch_size
    if args.grad_accum is not None:
        overrides["gradient_accumulation_steps"] = args.grad_accum
    if args.max_length is not None:
        overrides["max_length"] = args.max_length
    if args.warmup_steps is not None:
        overrides["warmup_steps"] = args.warmup_steps
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir

    train(
        small=args.small,
        device=args.device,
        model_id=args.model,
        chat_template=args.chat_template,
        training_set=args.training_set,
        **overrides,
    )


def _validate(args):
    from validation import validate

    validate(validation_set=args.validation_set, device=args.device, model_id=args.model)


def _infer(args):
    from inference import infer

    print(infer(
        args.message,
        system=args.system,
        use_adapter=not args.no_adapter,
        model=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    ))


def build_parser():
    parser = argparse.ArgumentParser(prog="daneel")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="fine-tune the LoRA adapter")
    t.add_argument("--small", action="store_true", help="quick run on a tiny subset")
    t.add_argument("--model", default=DEFAULT_MODEL, help="base model id or path")
    t.add_argument("--chat-template", help="path to a chat template (.jinja); defaults to the bundled Qwen template")
    t.add_argument("--training-set", default="training_set.jsonl", help="path to the training jsonl")
    t.add_argument("--device", choices=DEVICES, default="auto")
    t.add_argument("--epochs", type=int)
    t.add_argument("--lr", type=float)
    t.add_argument("--batch-size", type=int)
    t.add_argument("--grad-accum", type=int)
    t.add_argument("--max-length", type=int)
    t.add_argument("--warmup-steps", type=int)
    t.add_argument("--output-dir")
    t.set_defaults(func=_train)

    v = sub.add_parser("validate", help="evaluate adapter vs base on the validation set")
    v.add_argument("--validation-set", default="./validation_set.jsonl")
    v.add_argument("--model", default=DEFAULT_MODEL, help="base model id or path")
    v.add_argument("--device", choices=DEVICES, default="auto")
    v.set_defaults(func=_validate)

    i = sub.add_parser("infer", help="answer a single message with the model")
    i.add_argument("message")
    i.add_argument("--system")
    i.add_argument("--no-adapter", action="store_true", help="use the base model only")
    i.add_argument("--model", default=DEFAULT_MODEL, help="base model id or path")
    i.add_argument("--device", choices=DEVICES, default="auto")
    i.add_argument("--max-new-tokens", type=int, default=512)
    i.set_defaults(func=_infer)

    return parser


def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
