"""
模型评测与输出重建误差。

PPL 输入：
    [1, N]：一条较长的连续序列，例如 WikiText-2。
    [nsamples, seqlen]：多个独立窗口，例如 C4。

不同的行始终作为独立序列处理，不跨行连接上下文。
"""

import math

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def _valid_output_rows(
    prediction,
    reference,
    valid_mask=None,
):
    """
    将输出整理为 [有效位置数, 特征维度]。

    支持：
        [N, D]
        [B, T, D]

    valid_mask 的形状必须等于输出除最后一维之外的形状。
    reference 会 detach，prediction 保留梯度。
    """
    if prediction.shape != reference.shape:
        raise ValueError(
            f"输出形状不同：{prediction.shape} "
            f"和 {reference.shape}"
        )

    if prediction.ndim < 2 or prediction.shape[-1] == 0:
        raise ValueError(
            "输出至少应为二维，最后一维是特征维度。"
        )

    if (
        not prediction.is_floating_point()
        or not reference.is_floating_point()
    ):
        raise TypeError("请传入浮点输出，而不是 token ID。")

    feature_dim = prediction.shape[-1]

    pred = prediction.reshape(-1, feature_dim)

    ref = reference.detach().to(
        prediction.device
    ).reshape(-1, feature_dim)

    if valid_mask is not None:
        if tuple(valid_mask.shape) != tuple(
            prediction.shape[:-1]
        ):
            raise ValueError(
                "valid_mask 的形状必须等于 "
                "prediction.shape[:-1]。"
            )

        keep = valid_mask.to(
            device=prediction.device,
            dtype=torch.bool,
        ).reshape(-1)

        pred = pred[keep]
        ref = ref[keep]

    # 使用 float32 计算误差，同时保留 prediction 的梯度。
    return pred.float(), ref.float()


def reconstruction_loss(
    prediction,
    reference,
    valid_mask=None,
):
    """
    可反向传播的输出 MSE。

    prediction：
        量化层或量化 block 的输出。

    reference：
        原始层或原始 block 的参考输出。

    返回：
        对有效输出元素取平均的标量损失。
    """
    pred, ref = _valid_output_rows(
        prediction,
        reference,
        valid_mask,
    )

    if pred.numel() == 0:
        raise ValueError("没有有效输出元素。")

    return (pred - ref).square().mean()


class OutputErrorMeter:
    """
    跨多个 batch 累计输出误差。

    MSE：
        sum((prediction - reference)^2) / 有效元素数

    NMSE：
        sum((prediction - reference)^2)
        / sum(reference^2)

    累计总和后再计算，避免不同大小的 batch 被错误地等权平均。
    """

    def __init__(self):
        self.error_square_sum = 0.0
        self.reference_square_sum = 0.0
        self.num_elements = 0
        self.num_valid_positions = 0

    @torch.no_grad()
    def update(
        self,
        prediction,
        reference,
        valid_mask=None,
    ):
        pred, ref = _valid_output_rows(
            prediction,
            reference,
            valid_mask,
        )

        if pred.numel() == 0:
            return

        error_sum = (
            (pred - ref)
            .square()
            .sum(dtype=torch.float64)
            .item()
        )

        reference_sum = (
            ref.square()
            .sum(dtype=torch.float64)
            .item()
        )

        if not (
            math.isfinite(error_sum)
            and math.isfinite(reference_sum)
        ):
            raise ValueError("输出误差包含 NaN 或 Inf。")

        self.error_square_sum += error_sum
        self.reference_square_sum += reference_sum
        self.num_elements += pred.numel()
        self.num_valid_positions += pred.shape[0]

    def compute(self):
        if self.num_elements == 0:
            raise ValueError("尚未累计有效输出。")

        mse = (
            self.error_square_sum
            / self.num_elements
        )

        if self.reference_square_sum == 0:
            nmse = (
                0.0
                if self.error_square_sum == 0
                else float("inf")
            )
        else:
            nmse = (
                self.error_square_sum
                / self.reference_square_sum
            )

        return {
            "mse": mse,
            "nmse": nmse,
            "num_elements": self.num_elements,
            "num_valid_positions": self.num_valid_positions,
        }


def _iter_windows(
    length,
    seqlen,
    stride,
    drop_last,
):
    """
    返回：
        begin：窗口起点
        end：窗口终点，不包含 end
        first_target：本窗口第一个需要计分的 token 的局部位置

    局部位置 0 没有前一个 token，因此不计分。
    滑窗中已经计分的重叠 token 也不重复计分。
    """
    previous_end = 0

    for begin in range(0, length, stride):
        end = min(begin + seqlen, length)

        if end - begin < 2:
            break

        if drop_last and end - begin < seqlen:
            break

        first_target = max(
            previous_end,
            begin + 1,
        )

        if first_target < end:
            yield (
                begin,
                end,
                first_target - begin,
            )

        previous_end = end

        if end == length:
            break


def _iter_sequences(input_ids, attention_mask):
    """逐行返回独立序列；如果存在 padding，则先去掉两端 padding。"""
    for row_index, row in enumerate(input_ids):
        if attention_mask is not None:
            mask = attention_mask[row_index]
            positions = mask.nonzero(
                as_tuple=False
            ).flatten()

            if positions.numel() == 0:
                continue

            begin = positions[0].item()
            end = positions[-1].item() + 1

            if not mask[begin:end].all().item():
                raise ValueError(
                    "attention_mask 只支持两端 padding，"
                    "不支持序列内部出现无效位置。"
                )

            row = row[begin:end]

        yield row_index, row


@torch.inference_mode()
def evaluate_ppl(
    model,
    input_ids,
    *,
    seqlen=2048,
    stride=None,
    drop_last=True,
    attention_mask=None,
    max_windows=None,
    show_progress=True,
):
    """
    计算自回归语言模型的 PPL。

    input_ids：
        二维 torch.long 张量，每一行是独立序列。

    stride：
        None：使用 seqlen，即独立分块。
        小于 seqlen：使用带重叠的滑动窗口。

    drop_last：
        True：忽略不足 seqlen 的尾块。
        False：保留能够产生预测目标的短尾块。

    max_windows：
        仅用于快速检查，限制总共评测的窗口数。

    attention_mask：
        可选，形状与 input_ids 相同。
        当前 data_utils 产生的数据没有 padding，可以不传。
    """
    if (
        input_ids.ndim != 2
        or input_ids.dtype != torch.long
    ):
        raise ValueError(
            "input_ids 必须是二维 torch.long 张量。"
        )

    if seqlen < 2:
        raise ValueError("seqlen 必须至少为 2。")

    stride = seqlen if stride is None else stride

    if not 1 <= stride <= seqlen:
        raise ValueError(
            "stride 必须满足 1 <= stride <= seqlen。"
        )

    if max_windows is not None and max_windows <= 0:
        raise ValueError(
            "max_windows 必须大于 0。"
        )

    input_ids = input_ids.detach().cpu()

    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask 与 input_ids 形状必须相同。"
            )

        attention_mask = attention_mask.detach().cpu()

        if not (
            (attention_mask == 0)
            | (attention_mask == 1)
        ).all().item():
            raise ValueError(
                "attention_mask 只能包含 0 或 1。"
            )

        attention_mask = attention_mask.bool()

    input_device = (
        model.get_input_embeddings().weight.device
    )

    if input_device.type == "meta":
        raise ValueError(
            "输入 embedding 尚未加载到实际设备。"
        )

    context_limit = getattr(
        model.config,
        "max_position_embeddings",
        None,
    )

    if (
        isinstance(context_limit, int)
        and seqlen > context_limit
    ):
        raise ValueError(
            f"seqlen={seqlen} 超过配置中的上下文长度 "
            f"{context_limit}。"
        )

    def planned_windows():
        for row_index, sequence in _iter_sequences(
            input_ids,
            attention_mask,
        ):
            for begin, end, first_target in _iter_windows(
                sequence.numel(),
                seqlen,
                stride,
                drop_last,
            ):
                yield (
                    row_index,
                    sequence[begin:end].unsqueeze(0),
                    first_target,
                )

    # 保存各模块原来的模式，评测结束后恢复。
    training_states = [
        (module, module.training)
        for module in model.modules()
    ]

    total_nll = 0.0
    num_predicted_tokens = 0
    num_windows = 0
    partial = False

    progress = tqdm(
        total=max_windows,
        desc="PPL",
        unit="window",
        disable=not show_progress,
    )

    model.eval()

    try:
        for row_index, window, first_target in planned_windows():
            if (
                max_windows is not None
                and num_windows >= max_windows
            ):
                partial = True
                break

            window = window.to(input_device)

            logits = model(
                input_ids=window,
                use_cache=False,
                return_dict=True,
                output_hidden_states=False,
                output_attentions=False,
            ).logits

            if logits.shape[:2] != window.shape:
                raise ValueError(
                    "模型必须返回每个输入位置的 logits。"
                )

            # 位置 t 的 logits 用于预测位置 t+1 的 token。
            #
            # 只截取本窗口尚未计分的目标，避免滑窗重复计分。
            scores = logits[
                :, first_target - 1:-1, :
            ].float()

            targets = window[
                :, first_target:
            ].to(logits.device)

            nll = F.cross_entropy(
                scores.reshape(-1, scores.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )

            value = nll.item()

            if not math.isfinite(value):
                raise ValueError(
                    f"第 {row_index} 条序列的 "
                    "NLL 出现 NaN 或 Inf。"
                )

            total_nll += value
            num_predicted_tokens += targets.numel()
            num_windows += 1

            progress.update(1)

            del logits, scores, targets, window, nll

    finally:
        progress.close()

        for module, training in training_states:
            module.training = training

    if num_predicted_tokens == 0:
        raise ValueError(
            "没有可评测的预测 token。"
            "请检查序列长度，或设置 drop_last=False。"
        )

    avg_nll = total_nll / num_predicted_tokens

    try:
        ppl = math.exp(avg_nll)
    except OverflowError:
        ppl = float("inf")

    return {
        "ppl": ppl,
        "avg_nll": avg_nll,
        "total_nll": total_nll,
        "num_predicted_tokens": num_predicted_tokens,
        "num_windows": num_windows,
        "seqlen": seqlen,
        "stride": stride,
        "drop_last": drop_last,
        "partial": partial,
    }