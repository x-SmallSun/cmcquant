"""
通用本地文本数据工具。

支持：
    Parquet
    JSON / JSONL
    JSON.gz / JSONL.gz
    TXT

所有路径由调用者传入。
所有生成的 token 张量保存在 CPU 上。
"""

from pathlib import Path
import glob
import random

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


def _infer_file_format(path):
    """根据文件后缀推断 datasets 的读取格式。"""
    name = str(path).lower()

    if name.endswith(".parquet"):
        return "parquet"

    if name.endswith(
        (".json", ".jsonl", ".json.gz", ".jsonl.gz")
    ):
        return "json"

    if name.endswith((".txt", ".txt.gz")):
        return "text"

    raise ValueError(
        f"无法推断文件格式：{path}，"
        "请显式传入 file_format。"
    )


def load_local_text_dataset(
    data_files,
    *,
    cache_dir,
    file_format=None,
    text_column="text",
):
    """
    读取本地文本数据，返回 DatasetDict。

    data_files 示例：
        {
            "train": "/somewhere/train-*.parquet",
            "test": "/somewhere/test-*.parquet",
        }

    每个划分也可以传入文件列表：
        {
            "train": [
                "/somewhere/part1.json.gz",
                "/somewhere/part2.json.gz",
            ]
        }

    支持通配符，文件按路径排序后读取。
    """
    if not data_files:
        raise ValueError("data_files 不能为空。")

    resolved_files = {}

    for split, patterns in data_files.items():
        if isinstance(patterns, (str, Path)):
            patterns = [patterns]

        files = []

        for pattern in patterns:
            pattern = str(Path(pattern).expanduser())

            matches = [
                Path(path).resolve()
                for path in glob.glob(
                    pattern,
                    recursive=True,
                )
                if Path(path).is_file()
            ]

            if not matches:
                raise FileNotFoundError(
                    f"{split} 没有匹配到文件：{pattern}"
                )

            files.extend(matches)

        # 去重并固定文件顺序。
        files = sorted(set(files))

        if not files:
            raise ValueError(
                f"{split} 没有数据文件。"
            )

        for path in files:
            if path.stat().st_size == 0:
                raise ValueError(
                    f"数据文件为空：{path}"
                )

        resolved_files[split] = [
            str(path) for path in files
        ]

    if file_format is None:
        formats = {
            _infer_file_format(path)
            for files in resolved_files.values()
            for path in files
        }

        if len(formats) != 1:
            raise ValueError(
                "一次加载中的文件必须使用同一种格式。"
                "不同格式的数据集请分别调用。"
            )

        file_format = formats.pop()

    if file_format not in {"parquet", "json", "text"}:
        raise ValueError(
            f"暂不支持 file_format={file_format!r}"
        )

    dataset = load_dataset(
        file_format,
        data_files=resolved_files,
        cache_dir=str(
            Path(cache_dir).expanduser().resolve()
        ),

        # 使用磁盘上的 Arrow 缓存，
        # 不要求把整个数据集复制到内存。
        keep_in_memory=False,
    )

    for split in resolved_files:
        if len(dataset[split]) == 0:
            raise ValueError(
                f"{split} 划分没有数据。"
            )

        if text_column not in dataset[split].column_names:
            raise ValueError(
                f"{split} 中没有字段 {text_column!r}，"
                f"实际字段：{dataset[split].column_names}"
            )

    return dataset


def _encode_text(
    text,
    tokenizer,
    add_special_tokens,
):
    """对一段文本分词，返回 Python token ID 列表。"""
    if not isinstance(text, str):
        raise TypeError("文本字段必须是字符串。")

    return tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )["input_ids"]


def tokenize_text_stream(
    dataset_split,
    tokenizer,
    *,
    text_column="text",
    separator="\n\n",
    add_special_tokens=True,
):
    """
    拼接一个数据划分并分词。

    适用于 WikiText-2 等较小语料的完整评测。
    返回 CPU torch.long 张量，形状为 [1, N]。

    C4 的文档窗口评测请使用 sample_token_windows。
    """
    text = separator.join(
        dataset_split[text_column]
    )

    if not text.strip():
        raise ValueError("当前划分没有有效文本。")

    token_ids = _encode_text(
        text,
        tokenizer,
        add_special_tokens,
    )

    if len(token_ids) < 2:
        raise ValueError("分词后的 token 数不足。")

    return torch.tensor(
        [token_ids],
        dtype=torch.long,
        device="cpu",
    )


def sample_token_windows(
    dataset_split,
    tokenizer,
    *,
    mode,
    nsamples=128,
    seqlen=2048,
    seed=0,
    text_column="text",
    separator="\n\n",
    add_special_tokens=True,
    max_attempts=None,
):
    """
    采样定长 token 窗口。

    mode="concat"：
        拼接整个划分后，随机截取 token 窗口。

    mode="document"：
        随机选择文档；
        文档分词后足够长，才从中截取窗口。

    返回：
        {
            "input_ids": [nsamples, seqlen],
            "metadata": {...},
        }

    两种模式都允许重复采样。
    document 模式的窗口不会跨越文档边界。
    """
    if mode not in {"concat", "document"}:
        raise ValueError(
            "mode 只能是 concat 或 document。"
        )

    if nsamples <= 0 or seqlen <= 0:
        raise ValueError(
            "nsamples 和 seqlen 必须大于 0。"
        )

    if len(dataset_split) == 0:
        raise ValueError("不能从空数据集中采样。")

    rng = random.Random(seed)

    windows = []
    origins = []

    if mode == "concat":
        text = separator.join(
            dataset_split[text_column]
        )

        token_ids = _encode_text(
            text,
            tokenizer,
            add_special_tokens,
        )

        if len(token_ids) < seqlen:
            raise ValueError(
                f"总 token 数为 {len(token_ids)}，"
                f"不足以截取长度 {seqlen}。"
            )

        for _ in range(nsamples):
            start = rng.randint(
                0,
                len(token_ids) - seqlen,
            )

            windows.append(
                token_ids[start:start + seqlen]
            )

            origins.append({
                "row_index": None,
                "token_start": start,
            })

    else:
        if max_attempts is None:
            max_attempts = max(
                1000,
                nsamples * 100,
            )

        if max_attempts <= 0:
            raise ValueError(
                "max_attempts 必须大于 0。"
            )

        attempts = 0

        while (
            len(windows) < nsamples
            and attempts < max_attempts
        ):
            attempts += 1

            row_index = rng.randrange(
                len(dataset_split)
            )

            text = dataset_split[row_index][
                text_column
            ]

            if not isinstance(text, str):
                raise TypeError(
                    f"第 {row_index} 行的文本不是字符串。"
                )

            if not text.strip():
                continue

            token_ids = _encode_text(
                text,
                tokenizer,
                add_special_tokens,
            )

            if len(token_ids) < seqlen:
                continue

            start = rng.randint(
                0,
                len(token_ids) - seqlen,
            )

            windows.append(
                token_ids[start:start + seqlen]
            )

            origins.append({
                "row_index": row_index,
                "token_start": start,
            })

        if len(windows) < nsamples:
            raise RuntimeError(
                f"尝试 {attempts} 次后，只得到 "
                f"{len(windows)}/{nsamples} 个窗口。"
                "请检查文档长度，或增大 max_attempts。"
            )

    return {
        "input_ids": torch.tensor(
            windows,
            dtype=torch.long,
            device="cpu",
        ),

        "metadata": {
            "mode": mode,
            "source_split": str(
                getattr(dataset_split, "split", "")
            ),
            "tokenizer": str(
                tokenizer.name_or_path
            ),
            "nsamples": nsamples,
            "seqlen": seqlen,
            "seed": seed,
            "text_column": text_column,
            "add_special_tokens": add_special_tokens,
            "separator": (
                separator
                if mode == "concat"
                else None
            ),
            "origins": origins,
        },
    }


class CalibrationDataset(Dataset):
    """把二维 token 张量包装成模型输入字典。"""

    def __init__(self, input_ids):
        if (
            input_ids.ndim != 2
            or input_ids.shape[0] == 0
            or input_ids.shape[1] == 0
        ):
            raise ValueError(
                "input_ids 必须是非空二维张量。"
            )

        if (
            input_ids.dtype != torch.long
            or input_ids.device.type != "cpu"
        ):
            raise ValueError(
                "input_ids 必须是 CPU 上的 torch.long。"
            )

        self.input_ids = input_ids

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, index):
        input_ids = self.input_ids[index]

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(
                input_ids
            ),
        }


def make_calibration_loader(
    input_ids,
    *,
    batch_size=1,
):
    """构造校准 DataLoader，保持样本顺序。"""
    if batch_size <= 0:
        raise ValueError(
            "batch_size 必须大于 0。"
        )

    return DataLoader(
        CalibrationDataset(input_ids),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )