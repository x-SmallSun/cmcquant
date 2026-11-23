"""
离线检查模型文件、分词器，以及可选的模型前向计算。

从项目根目录运行：
    python -m scripts.check_models
"""

import argparse
import gc
import os
import sys
import traceback
from pathlib import Path


# 在首次导入 Hugging Face 相关库之前设置离线模式。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


from utils.model_utils import (
    DEFAULT_MODELS_ROOT,
    MODEL_NAMES,
    get_decoder_layers,
    inspect_model_files,
    iter_quantizable_linears,
    load_model_and_tokenizer,
    load_tokenizer,
)


def check_one_model(model_name, args):
    """检查一个模型；默认不加载权重张量。"""
    info = inspect_model_files(
        model_name,
        args.models_root,
    )

    config = info["config"]

    # 文件存在之外，再确认分词器确实能工作。
    tokenizer = load_tokenizer(info["path"])

    test_text = "The quick brown fox jumps over the lazy dog."
    token_ids = tokenizer.encode(test_text)

    if not token_ids:
        raise ValueError("分词器产生了空 token 序列。")

    vocab_size = config["vocab_size"]

    if min(token_ids) < 0 or max(token_ids) >= vocab_size:
        raise ValueError(
            "测试文本的 token 编号超出了模型词表，"
            "请检查分词器是否与模型匹配。"
        )

    print(
        f"[文件检查通过] {model_name}\n"
        f"  model_type: {config['model_type']}\n"
        f"  Transformer blocks: {config['num_hidden_layers']}\n"
        f"  hidden_size: {config['hidden_size']}\n"
        f"  vocab_size: {vocab_size}\n"
        f"  权重格式: {info['weight_format']}\n"
        f"  权重文件数: {len(info['weight_files'])}\n"
        f"  权重文件大小: {info['weight_gib']:.3f} GiB",
        flush=True,
    )

    if not args.load:
        return

    import torch

    model, tokenizer = load_model_and_tokenizer(
        info["path"],
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    blocks = get_decoder_layers(model)

    linear_names = [
        name
        for name, _ in iter_quantizable_linears(model)
    ]

    # 当前适配的原始架构：
    # Llama / Qwen2.5：每个 block 有 7 个线性层。
    # OPT：每个 block 有 6 个线性层。
    linears_per_block = (
        6
        if config["model_type"] == "opt"
        else 7
    )

    if len(blocks) != config["num_hidden_layers"]:
        raise ValueError(
            "实际 block 数量与 config.json 不一致。"
        )

    expected_linear_count = len(blocks) * linears_per_block

    if len(linear_names) != expected_linear_count:
        raise ValueError(
            f"线性层数量不符合预期："
            f"实际 {len(linear_names)}，"
            f"预期 {expected_linear_count}。"
        )

    inputs = tokenizer(
        test_text,
        return_tensors="pt",
        truncation=True,
        max_length=args.seq_len,
        return_attention_mask=True,
    )

    input_device = model.get_input_embeddings().weight.device

    # 只传入这些架构都支持的输入字段。
    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
        if key in {"input_ids", "attention_mask"}
    }

    # 仅本次检查不计算梯度，不影响以后训练量化参数。
    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=False,
        )

        logits = outputs.logits

    expected_shape = (
        *inputs["input_ids"].shape,
        model.config.vocab_size,
    )

    if tuple(logits.shape) != expected_shape:
        raise ValueError(
            f"logits 形状异常：{tuple(logits.shape)}，"
            f"预期：{expected_shape}"
        )

    if not torch.isfinite(logits).all().item():
        raise ValueError("模型输出包含 NaN 或 Inf。")

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"[前向检查通过] {model_name}\n"
        f"  设备: {input_device}\n"
        f"  参数精度: {next(model.parameters()).dtype}\n"
        f"  参数量: {parameter_count / 1e9:.3f} B\n"
        f"  待量化线性层数: {len(linear_names)}\n"
        f"  输入形状: {tuple(inputs['input_ids'].shape)}\n"
        f"  logits 形状: {tuple(logits.shape)}",
        flush=True,
    )

    print("  前几个待量化线性层：")

    for name in linear_names[:7]:
        print(f"    {name}")

    # 返回后，model、inputs、outputs 等局部变量离开作用域。
    # 脚本外层随后执行垃圾回收。


def main():
    parser = argparse.ArgumentParser(
        description="检查本地模型，按需加载并执行一次前向计算。"
    )

    parser.add_argument(
        "--models-root",
        type=Path,
        default=DEFAULT_MODELS_ROOT,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )

    parser.add_argument(
        "--load",
        action="store_true",
        help="加载权重并执行前向计算；默认只检查文件和分词器。",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="例如 cpu、cuda:0、cuda:1。",
    )

    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )

    parser.add_argument(
        "--attn-implementation",
        choices=["sdpa", "eager"],
        default="sdpa",
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="失败时打印完整错误调用栈。",
    )

    args = parser.parse_args()

    if args.seq_len < 1:
        parser.error("--seq-len 必须大于 0。")

    failed_count = 0

    for model_name in args.models:
        try:
            check_one_model(model_name, args)

        except Exception as error:
            failed_count += 1

            print(
                f"[失败] {model_name}: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )

            if args.verbose:
                traceback.print_exc()

        # 不保存模型对象，多个模型依次检查。
        gc.collect()
        print()

    success_count = len(args.models) - failed_count

    print(
        f"检查结束："
        f"{success_count} 个通过，"
        f"{failed_count} 个失败。"
    )

    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())