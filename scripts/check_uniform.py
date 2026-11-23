"""在 CPU 上检查分组和均匀量化，并演示线性层输出误差。"""

import torch
import torch.nn.functional as F

from quantization.grouping import (
    group_weight,
    ungroup_weight,
)
from quantization.uniform import (
    quantize_uniform,
    dequantize_uniform,
    fake_quantize_uniform,
)


def check_grouping():
    """检查分组、恢复、数据独立性和整除约束。"""
    weight = torch.arange(
        16,
        dtype=torch.float32,
        device="cpu",
    ).reshape(2, 8)

    original = weight.clone()

    groups, info = group_weight(
        weight,
        group_size=4,
    )

    assert groups.shape == (2, 2, 4)

    expected_groups = torch.tensor(
        [
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            [[8, 9, 10, 11], [12, 13, 14, 15]],
        ],
        dtype=torch.float32,
        device="cpu",
    )

    torch.testing.assert_close(
        groups,
        expected_groups,
    )

    restored = ungroup_weight(groups, info)

    torch.testing.assert_close(
        restored,
        original,
        rtol=0,
        atol=0,
    )

    # 修改分组数据，不应影响原始权重。
    groups[0, 0, 0] = 100

    torch.testing.assert_close(
        weight,
        original,
        rtol=0,
        atol=0,
    )

    # group_size=-1：每行作为一组。
    row_groups, row_info = group_weight(
        weight,
        group_size=-1,
    )

    assert row_groups.shape == (2, 1, 8)

    torch.testing.assert_close(
        ungroup_weight(row_groups, row_info),
        original,
        rtol=0,
        atol=0,
    )

    # 输入维度 8 不能被 3 整除，应当拒绝。
    try:
        group_weight(weight, group_size=3)
    except AssertionError:
        pass
    else:
        raise AssertionError(
            "未拒绝不能整除输入维度的 group_size。"
        )

    print("[通过] 分组、恢复、数据独立性和整除约束")


def check_known_values():
    """使用可手算的例子检查编号、偏移和重建结果。"""
    weight = torch.tensor(
        [
            [
                2.0, 2.6, 4.4, 5.0,
                -5.0, -4.4, -2.6, -2.0,
            ],
            [
                0.0, 1.0, 2.0, 3.0,
                -3.0, -2.0, -1.0, 0.0,
            ],
        ],
        dtype=torch.float32,
        device="cpu",
    )

    original = weight.clone()

    # 每组 4 个元素，使用 2 bit，即 4 个量化级别。
    # 四组的范围宽度均为 3，因此 scale 均为 1。
    state = quantize_uniform(
        weight,
        n_bits=2,
        group_size=4,
    )

    expected_codes = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.uint8,
        device="cpu",
    ).reshape(1, 1, 4).expand(2, 2, 4)

    expected_offsets = torch.tensor(
        [
            [[2.0], [-5.0]],
            [[0.0], [-3.0]],
        ],
        dtype=torch.float32,
        device="cpu",
    )

    expected_weight = torch.tensor(
        [
            [
                2.0, 3.0, 4.0, 5.0,
                -5.0, -4.0, -3.0, -2.0,
            ],
            [
                0.0, 1.0, 2.0, 3.0,
                -3.0, -2.0, -1.0, 0.0,
            ],
        ],
        dtype=torch.float32,
        device="cpu",
    )

    torch.testing.assert_close(
        state["qweight"],
        expected_codes,
        rtol=0,
        atol=0,
    )

    torch.testing.assert_close(
        state["offsets"],
        expected_offsets,
        rtol=0,
        atol=0,
    )

    torch.testing.assert_close(
        state["scales"],
        torch.ones_like(expected_offsets),
        rtol=0,
        atol=0,
    )

    reconstructed = dequantize_uniform(state)

    torch.testing.assert_close(
        reconstructed,
        expected_weight,
        rtol=0,
        atol=0,
    )

    torch.testing.assert_close(
        weight,
        original,
        rtol=0,
        atol=0,
    )

    print("[通过] 手算例子的编号、浮点偏移和重建结果")


def check_constant_groups():
    """检查全零组、正常数组和负常数组。"""
    weight = torch.tensor(
        [
            [
                0.0, 0.0, 0.0, 0.0,
                2.5, 2.5, 2.5, 2.5,
            ],
            [
                -3.0, -3.0, -3.0, -3.0,
                0.125, 0.125, 0.125, 0.125,
            ],
        ],
        dtype=torch.float32,
        device="cpu",
    )

    state = quantize_uniform(
        weight,
        n_bits=4,
        group_size=4,
    )

    # 常数组统一使用编号 0、尺度 1。
    assert torch.all(state["qweight"] == 0).item()
    assert torch.all(state["scales"] == 1).item()

    reconstructed = dequantize_uniform(state)

    assert torch.isfinite(reconstructed).all().item()

    torch.testing.assert_close(
        reconstructed,
        weight,
        rtol=0,
        atol=0,
    )

    print("[通过] 全零组和非零常数组")


def check_dtypes_and_bits():
    """检查不同浮点类型、位宽和分组方式。"""
    generator = torch.Generator(
        device="cpu"
    ).manual_seed(2026)

    source = torch.randn(
        3,
        256,
        generator=generator,
        device="cpu",
    )

    checked = 0

    for dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        for n_bits in (2, 3, 4, 8):
            for group_size in (128, -1):
                weight = source.to(dtype=dtype)
                original = weight.clone()

                state = quantize_uniform(
                    weight,
                    n_bits=n_bits,
                    group_size=group_size,
                )

                qweight = state["qweight"]

                assert qweight.dtype == torch.uint8
                assert state["scales"].dtype == torch.float32
                assert state["offsets"].dtype == torch.float32

                assert qweight.min().item() >= 0
                assert qweight.max().item() <= 2 ** n_bits - 1

                reconstructed = dequantize_uniform(state)

                assert reconstructed.shape == weight.shape
                assert reconstructed.dtype == dtype
                assert reconstructed.device.type == "cpu"
                assert torch.isfinite(reconstructed).all().item()

                # fake_quantize 应与显式量化再反量化一致。
                fake_weight = fake_quantize_uniform(
                    weight,
                    n_bits=n_bits,
                    group_size=group_size,
                )

                torch.testing.assert_close(
                    fake_weight,
                    reconstructed,
                    rtol=0,
                    atol=0,
                )

                # 使用 float32 重建，排除最终转回
                # float16/bfloat16 时产生的额外舍入误差。
                reconstructed_f32 = dequantize_uniform(
                    state,
                    dtype=torch.float32,
                )

                group_errors = (
                    reconstructed_f32 - weight.float()
                ).abs().reshape(qweight.shape)

                max_errors = group_errors.amax(
                    dim=-1,
                    keepdim=True,
                )

                # 最近邻取整的误差应不超过半个步长，
                # 另留少量 float32 运算误差余量。
                tolerance = (
                    8
                    * torch.finfo(torch.float32).eps
                    * max(
                        1.0,
                        weight.float().abs().max().item(),
                    )
                )

                error_bound = (
                    state["scales"] / 2 + tolerance
                )

                assert torch.all(
                    max_errors <= error_bound
                ).item(), (
                    f"重建误差超出范围：dtype={dtype}, "
                    f"n_bits={n_bits}, group_size={group_size}"
                )

                torch.testing.assert_close(
                    weight,
                    original,
                    rtol=0,
                    atol=0,
                )

                checked += 1

    print(f"[通过] 浮点类型、位宽和分组组合，共 {checked} 组")


@torch.no_grad()
def print_linear_errors():
    """比较量化前后的权重误差和线性层输出误差。"""
    generator = torch.Generator(
        device="cpu"
    ).manual_seed(2026)

    # 权重：[out_features, in_features]
    weight = torch.randn(
        32,
        256,
        generator=generator,
        device="cpu",
    ) * 0.02

    bias = torch.randn(
        32,
        generator=generator,
        device="cpu",
    ) * 0.01

    # 输入：[batch_size, sequence_length, in_features]
    inputs = torch.randn(
        2,
        16,
        256,
        generator=generator,
        device="cpu",
    )

    reference_output = F.linear(
        inputs,
        weight,
        bias,
    )

    weight_energy = weight.square().mean().item()
    output_energy = reference_output.square().mean().item()

    print("\nCPU 随机线性层误差，group_size=128")
    print(
        f"{'bit':>4} "
        f"{'权重 MSE':>14} "
        f"{'权重 NMSE':>14} "
        f"{'输出 MSE':>14} "
        f"{'输出 NMSE':>14}"
    )

    for n_bits in (2, 3, 4, 8):
        reconstructed_weight = fake_quantize_uniform(
            weight,
            n_bits=n_bits,
            group_size=128,
        )

        quantized_output = F.linear(
            inputs,
            reconstructed_weight,
            bias,
        )

        weight_mse = F.mse_loss(
            reconstructed_weight,
            weight,
        ).item()

        output_mse = F.mse_loss(
            quantized_output,
            reference_output,
        ).item()

        weight_nmse = weight_mse / weight_energy
        output_nmse = output_mse / output_energy

        print(
            f"{n_bits:>4d} "
            f"{weight_mse:>14.6e} "
            f"{weight_nmse:>14.6e} "
            f"{output_mse:>14.6e} "
            f"{output_nmse:>14.6e}"
        )


def main():
    # 小张量检查使用一个 CPU 线程即可。
    torch.set_num_threads(1)

    check_grouping()
    check_known_values()
    check_constant_groups()
    check_dtypes_and_bits()

    print("\n全部正确性检查通过。")

    print_linear_errors()


if __name__ == "__main__":
    main()