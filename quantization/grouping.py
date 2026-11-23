"""沿二维权重矩阵的输入维度分组，不进行补零。"""

import torch


def group_weight(
    weight: torch.Tensor,
    group_size: int = 128,
):
    """
    参数：
        weight：
            二维权重，形状为 [out_features, in_features]。

        group_size：
            每组权重数量，必须整除 in_features。
            -1 表示每个输出通道的整行权重作为一组。

    返回：
        groups：
            形状为：
            [out_features, groups_per_row, group_size]

        info：
            保存原始形状和分组信息的字典。
    """
    assert weight.ndim == 2, (
        f"weight 必须是二维张量，实际形状为 {tuple(weight.shape)}"
    )

    out_features, in_features = weight.shape

    assert out_features > 0 and in_features > 0, (
        "weight 不能为空。"
    )

    assert (
        isinstance(group_size, int)
        and not isinstance(group_size, bool)
    ), "group_size 必须是整数。"

    if group_size == -1:
        group_size = in_features

    assert group_size > 0, (
        "group_size 必须大于 0，或等于 -1。"
    )

    assert in_features % group_size == 0, (
        f"输入维度 {in_features} 必须能被 "
        f"group_size={group_size} 整除。"
    )

    groups_per_row = in_features // group_size

    # clone 避免后续修改分组数据时影响原始权重。
    groups = weight.reshape(
        out_features,
        groups_per_row,
        group_size,
    ).clone()

    info = {
        "original_shape": (out_features, in_features),
        "group_size": group_size,
        "groups_per_row": groups_per_row,
    }

    return groups, info


def ungroup_weight(
    groups: torch.Tensor,
    info: dict,
):
    """根据分组信息，将权重恢复为原来的二维形状。"""
    out_features, in_features = info["original_shape"]

    expected_shape = (
        out_features,
        info["groups_per_row"],
        info["group_size"],
    )

    assert tuple(groups.shape) == expected_shape, (
        f"分组形状应为 {expected_shape}，"
        f"实际为 {tuple(groups.shape)}"
    )

    return groups.reshape(
        out_features,
        in_features,
    ).contiguous()