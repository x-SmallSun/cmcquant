"""
本地模型加载和结构适配。

支持：
    Llama-2、Llama-3、Llama-3.2
    OPT
    Qwen2.5

注意：
    PyTorch 中 nn.Linear.weight 的形状是：
        [out_features, in_features]

    线性层计算为：
        Y = X @ weight.T + bias
"""

from pathlib import Path
import json


# 根据本文件的位置确定项目根目录，不依赖终端当前所在目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_ROOT = PROJECT_ROOT / "models"

MODEL_NAMES = (
    "Llama-3.2-1B",
    "Llama-2-7b-hf",
    "Meta-Llama-3-8B",
    "opt-1.3b",
    "opt-6.7b",
    "Qwen2.5-1.5B",
    "Qwen2.5-7B",
)

# Qwen2.5 在 Transformers 中的 model_type 是 qwen2。
SUPPORTED_MODEL_TYPES = {"llama", "opt", "qwen2"}


def read_json(path):
    """读取一个 JSON 对象。"""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        content = json.load(file)

    if not isinstance(content, dict):
        raise ValueError(f"JSON 顶层不是对象：{path}")

    return content


def resolve_model_path(
    model_name_or_path,
    models_root=DEFAULT_MODELS_ROOT,
):
    """
    将模型名称或本地路径转换成绝对路径。

    示例：
        resolve_model_path("Llama-3.2-1B")

        resolve_model_path(
            "/data/tyxiao/cmcquant/models/Llama-3.2-1B"
        )

    找不到目录时直接报错，不尝试联网下载。
    """
    supplied_path = Path(model_name_or_path).expanduser()

    # 优先接受调用者直接传入的已有本地目录。
    if supplied_path.is_dir():
        return supplied_path.resolve()

    # 否则认为传入的是 models 目录下的模型名称。
    model_path = Path(models_root).expanduser() / supplied_path

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"找不到模型目录：{model_path}\n"
            "请检查模型名称和 models_root。"
        )

    return model_path.resolve()


def inspect_model_files(
    model_name_or_path,
    models_root=DEFAULT_MODELS_ROOT,
):
    """
    检查模型配置、权重文件和分片索引。

    支持：
        model.safetensors
        model.safetensors.index.json

        pytorch_model.bin
        pytorch_model.bin.index.json

    如果同时存在两套格式，优先选择 safetensors。
    只检查被选中的格式，避免未下载的另一套权重索引造成误报。

    这里只检查文件存在且非空，不会读取完整权重张量，
    也不等于完成了文件哈希校验。
    """
    model_path = resolve_model_path(
        model_name_or_path,
        models_root,
    )

    config = read_json(model_path / "config.json")

    model_type = config.get("model_type")

    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"暂不支持 model_type={model_type!r}，"
            "当前支持 llama、opt、qwen2。"
        )

    # 每项分别表示：
    # 文件名、权重格式、是否为分片索引。
    candidates = (
        ("model.safetensors", "safetensors", False),
        ("model.safetensors.index.json", "safetensors", True),
        ("pytorch_model.bin", "pytorch", False),
        ("pytorch_model.bin.index.json", "pytorch", True),
    )

    for filename, weight_format, is_index in candidates:
        selected_file = model_path / filename

        if not selected_file.is_file():
            continue

        if is_index:
            index = read_json(selected_file)
            weight_map = index.get("weight_map")

            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(
                    f"权重索引缺少有效的 weight_map：{selected_file}"
                )

            shard_names = list(weight_map.values())

            if not all(isinstance(name, str) for name in shard_names):
                raise ValueError(
                    f"权重索引中的分片名必须是字符串：{selected_file}"
                )

            # 多个参数可能位于同一个分片，因此需要去重。
            weight_files = [
                (model_path / name).resolve()
                for name in sorted(set(shard_names))
            ]

            # 索引只能引用当前模型目录内的文件。
            if any(
                not path.is_relative_to(model_path)
                for path in weight_files
            ):
                raise ValueError(
                    f"索引引用了模型目录外的文件：{selected_file}"
                )
        else:
            weight_files = [selected_file]

        for weight_file in weight_files:
            if not weight_file.is_file():
                raise FileNotFoundError(
                    f"缺少权重分片：{weight_file}"
                )

            if weight_file.stat().st_size == 0:
                raise ValueError(
                    f"权重文件为空：{weight_file}"
                )

        total_bytes = sum(
            path.stat().st_size
            for path in weight_files
        )

        return {
            "path": str(model_path),
            "config": config,
            "weight_format": weight_format,
            "weight_files": [
                path.name for path in weight_files
            ],
            "weight_gib": total_bytes / (1024 ** 3),
        }

    raise FileNotFoundError(
        f"{model_path} 中没有找到可识别的模型权重。\n"
        "需要 safetensors、pytorch_model.bin，"
        "或者它们对应的分片索引。"
    )


def load_tokenizer(
    model_name_or_path,
    models_root=DEFAULT_MODELS_ROOT,
):
    """加载本地分词器，不联网、不扩充分词器词表。"""
    from transformers import AutoTokenizer

    model_path = resolve_model_path(
        model_name_or_path,
        models_root,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )

    # 部分 Llama 模型没有单独的 padding token。
    # 复用已有 eos token，不新增 token，也不修改模型 embedding。
    if (
        tokenizer.pad_token_id is None
        and tokenizer.eos_token_id is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model_and_tokenizer(
    model_name_or_path,
    models_root=DEFAULT_MODELS_ROOT,
    *,
    device="cpu",
    dtype="auto",
    attn_implementation="sdpa",
    freeze=True,
):
    """
    将模型加载到指定的单个设备。

    参数：
        device:
            "cpu"、"cuda:0"、"cuda:1" 等。

        dtype:
            "auto"、"float16"、"bfloat16"、"float32"。

            auto 在 GPU 上沿用检查点配置的精度；
            auto 在 CPU 上使用 float32。

        attn_implementation:
            "sdpa"：使用 PyTorch SDPA。
            "eager"：普通注意力实现，便于调试。

        freeze:
            True 时冻结原始模型参数。

    返回：
        model, tokenizer

    注意：
        冻结原始参数不会关闭全局自动求导。
        后续新建的可学习裁剪参数仍然可以训练。
    """
    import torch
    from transformers import AutoModelForCausalLM

    info = inspect_model_files(
        model_name_or_path,
        models_root,
    )

    target_device = torch.device(device)

    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("当前只支持 cpu 或 cuda:N。")

    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA 不可用，请检查 PyTorch 安装和显卡驱动。"
            )

        gpu_index = (
            0
            if target_device.index is None
            else target_device.index
        )

        if gpu_index >= torch.cuda.device_count():
            raise ValueError(
                f"GPU 编号 {gpu_index} 超出可见 GPU 范围。"
            )

        target_device = torch.device(f"cuda:{gpu_index}")

    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if dtype == "auto":
        selected_dtype = (
            torch.float32
            if target_device.type == "cpu"
            else "auto"
        )
    elif dtype in dtype_mapping:
        selected_dtype = dtype_mapping[dtype]
    else:
        raise ValueError(f"不支持的 dtype：{dtype}")

    if attn_implementation not in {"sdpa", "eager"}:
        raise ValueError(
            "attn_implementation 只能为 sdpa 或 eager。"
        )

    tokenizer = load_tokenizer(info["path"])

    model = AutoModelForCausalLM.from_pretrained(
        info["path"],
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=(
            info["weight_format"] == "safetensors"
        ),
        dtype=selected_dtype,

        # 整个模型只放在指定设备上，不自动分配到其他 GPU。
        device_map={"": str(target_device)},
        low_cpu_mem_usage=True,

        attn_implementation=attn_implementation,
    )

    # 校准和局部重建通常不需要生成时使用的 KV cache。
    model.config.use_cache = False

    # 关闭 dropout 等训练行为，但不关闭自动求导。
    model.eval()

    if freeze:
        model.requires_grad_(False)

    return model, tokenizer


def get_decoder_layers(model):
    """
    获取模型的 Transformer blocks。

    Llama / Qwen2.5：
        model.model.layers

    OPT：
        model.model.decoder.layers
    """
    model_type = model.config.model_type

    if model_type in {"llama", "qwen2"}:
        return model.model.layers

    if model_type == "opt":
        return model.model.decoder.layers

    raise ValueError(
        f"暂不支持 model_type={model_type!r}"
    )


def get_block_linears(block):
    """
    获取一个 Transformer block 内的所有 nn.Linear。

    返回：
        {
            相对模块名: 线性层对象,
            ...
        }

    这里只查找，不修改层。
    """
    import torch.nn as nn

    return {
        name: module
        for name, module in block.named_modules()
        if isinstance(module, nn.Linear)
    }


def iter_quantizable_linears(model):
    """
    遍历所有 Transformer blocks 内的线性层。

    每次返回：
        完整模块名, 线性层对象

    默认不包含：
        embedding
        lm_head
        归一化层
        Transformer blocks 外的其他模块
    """
    blocks = get_decoder_layers(model)

    prefix = (
        "model.decoder.layers"
        if model.config.model_type == "opt"
        else "model.layers"
    )

    for block_index, block in enumerate(blocks):
        for relative_name, linear in get_block_linears(block).items():
            full_name = (
                f"{prefix}.{block_index}.{relative_name}"
            )

            yield full_name, linear