"""评测本地模型，并保存 PPL 基线。"""

import argparse
import json
import os
from pathlib import Path


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


from utils.model_utils import load_model_and_tokenizer
from utils.data_utils import (
    load_local_text_dataset,
    sample_token_windows,
    tokenize_text_stream,
)
from utils.eval_utils import evaluate_ppl


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--eval-files", nargs="+", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--device", required=True)

    parser.add_argument(
        "--sampling",
        choices=["concat", "document"],
        required=True,
    )

    parser.add_argument("--text-column", default="text")
    parser.add_argument("--nsamples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=None)

    parser.add_argument(
        "--include-tail",
        action="store_true",
        help="保留不足 seqlen 的尾块。",
    )

    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    dataset = load_local_text_dataset(
        {"eval": args.eval_files},
        cache_dir=args.cache_dir,
        text_column=args.text_column,
    )

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device=args.device,
    )

    if args.sampling == "concat":
        eval_ids = tokenize_text_stream(
            dataset["eval"],
            tokenizer,
            text_column=args.text_column,
        )
    else:
        sampled = sample_token_windows(
            dataset["eval"],
            tokenizer,
            mode="document",
            nsamples=args.nsamples,
            seqlen=args.seqlen,
            seed=args.seed,
            text_column=args.text_column,
        )

        eval_ids = sampled["input_ids"]

    result = evaluate_ppl(
        model,
        eval_ids,
        seqlen=args.seqlen,
        stride=args.stride,
        drop_last=not args.include_tail,
        max_windows=args.max_windows,
    )

    result.update({
        "model": args.model,
        "tokenizer": str(tokenizer.name_or_path),
        "eval_files": args.eval_files,
        "sampling": args.sampling,
        "text_column": args.text_column,
        "add_special_tokens": True,
        "nsamples": (
            args.nsamples
            if args.sampling == "document"
            else None
        ),
        "seed": args.seed,
    })

    content = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )

    print(content)

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            content + "\n",
            encoding="utf-8",
        )

        print(f"结果已保存：{args.output.resolve()}")


if __name__ == "__main__":
    main()