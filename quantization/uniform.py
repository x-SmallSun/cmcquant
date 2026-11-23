"""按组执行 MinMax RTN 均匀量化，使用浮点偏移。"""

import torch

from quantization.grouping import (
    group_weight,
    ungroup_weight,
)


_SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
}


@torch.no_grad()
def quantize_uniform(
    weight: torch.Tensor,
    n_bits: int = 4,
    group_size: int = 128,
):
    """
    将浮点权重量化为整数编号，返回量化状态字典。

    参数：
        weight：
            二维浮点权重：
            [out_features, in_features]

        n_bits：
            量化编号位宽，支持 2～8 bit。

        group_size：
            每组权重数量，必须整除输入维度。
            -1 表示每个输出通道的整行权重作为一组。

    量化：
        offset = 每组权重的最小值
        scale = (maximum - minimum) / (2**n_bits - 1)
        q = clamp(round((weight - offset) / scale))

    反量化：
        weight_hat = q * scale + offset

    说明：
        使用各组的真实最小值和最大值。
        不强制量化范围包含 0。
        scale 和 offset 使用 float32 保存。
        不修改原始 weight。
        当前用于离线量化，不计算梯度。
    """
    if (
        not isinstance(n_bits, int)
        or isinstance(n_bits, bool)
    ):
        raise TypeError("n_bits 必须是整数。")

    if not 2 <= n_bits <= 8:
        raise ValueError("当前支持 2～8 bit。")

    if weight.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "weight 必须是 float16、bfloat16 或 float32。"
        )

    if not torch.isfinite(weight).all().item():
        raise ValueError("weight 包含 NaN 或 Inf。")

    # 沿输入维度分组，不补零。
    # groups：[输出通道数, 每行组数, 每组权重数]
    groups, group_info = group_weight(
        weight,
        group_size,
    )

    # 用 float32 计算量化参数。
    values = groups.float()

    # 每组的真实最小值和最大值。
    # 形状均为：[输出通道数, 每行组数, 1]
    minimum = values.amin(
        dim=-1,
        keepdim=True,
    )

    maximum = values.amax(
        dim=-1,
        keepdim=True,
    )

    qmin = 0
    qmax = 2 ** n_bits - 1

    # 每组量化网格的间距。
    ranges = maximum - minimum
    scales = ranges / qmax

    # 常数组的范围为 0。
    # 令 scale=1 后，该组所有编号都是 0；
    # 反量化时恢复为 offset，即原来的常数。
    scales = torch.where(
        ranges == 0,
        torch.ones_like(scales),
        scales,
    )

    if not torch.isfinite(scales).all().item():
        raise ValueError(
            "量化尺度在 float32 下发生溢出，"
            "请检查权重范围。"
        )

    if (scales <= 0).any().item():
        raise ValueError(
            "非零权重范围对应的尺度在 float32 下发生下溢。"
        )

    # 浮点偏移直接使用每组最小值。
    # 不取整，也不限制其取值范围。
    offsets = minimum

    # 减去偏移 → 缩放 → 取整 → 限制整数编号范围。
    qweight = torch.round(
        (values - offsets) / scales
    ).clamp(qmin, qmax)

    # 当前使用 uint8 保存编号。
    # 例如 4 bit 量化只使用 0～15，
    # 实际低 bit 打包后续再实现。
    qweight = qweight.to(torch.uint8)

    return {
        "qweight": qweight,
        "scales": scales,
        "offsets": offsets,
        "n_bits": n_bits,
        "qmin": qmin,
        "qmax": qmax,
        "group_info": group_info,
        "original_dtype": weight.dtype,
    }


@torch.no_grad()
def dequantize_uniform(
    state: dict,
    dtype=None,
):
    """
    根据量化状态重建浮点权重。

    计算规则：
        weight_hat = qweight * scale + offset

    返回形状：
        [out_features, in_features]

    dtype=None 时恢复为原始权重的浮点类型。
    """
    if dtype is None:
        dtype = state["original_dtype"]

    if dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "重建类型必须是 float16、bfloat16 或 float32。"
        )

    qweight = state["qweight"]
    device = qweight.device

    scales = state["scales"].to(
        device=device,
        dtype=torch.float32,
    )

    offsets = state["offsets"].to(
        device=device,
        dtype=torch.float32,
    )

    # 在 float32 下进行反量化。
    # scales 和 offsets 沿组内维度广播。
    reconstructed_groups = (
        qweight.float() * scales + offsets
    )

    # 从三维分组形式恢复成二维权重。
    reconstructed = ungroup_weight(
        reconstructed_groups,
        state["group_info"],
    )

    return reconstructed.to(dtype=dtype)


@torch.no_grad()
def fake_quantize_uniform(
    weight: torch.Tensor,
    n_bits: int = 4,
    group_size: int = 128,
):
    """
    量化后立即反量化，返回浮点重建权重。

    返回值与原始 weight 的形状、浮点类型一致。
    当前不提供 STE，不用于量化感知训练。
    """
    state = quantize_uniform(
        weight,
        n_bits=n_bits,
        group_size=group_size,
    )

    return dequantize_uniform(state)