"""
统计模型复杂度和运行耗时。

示例：
    python profile_stats.py --datasets PEMS-BAY METR-LA
    python profile_stats.py --datasets PEMS08 --variants TFPredictor --max-train-batches 0

备注：
    - TFPredictor 使用仓库原始的 KFGN_Mamba 实现。
    - TFPredictor-Transformer 保持相同输入输出维度、KFGN 图模块、层数和残差骨架，
      只把 Mamba mixer 替换为 Transformer mixer，用于更公平的消融对比。
    - 默认用 20 个 batch 外推单个 epoch 的训练耗时；如果希望完整实测一个 epoch，
      请设置 --max-train-batches 0。
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import os
import sys
import time
import types
import zipfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 如果服务器环境没有安装 einops，这里提供脚本所需的最小兼容实现。
# 这样 profile_stats.py 可以在缺少 einops 的环境里继续运行。
try:
    import einops  # noqa: F401
except ImportError:
    einops_stub = types.ModuleType("einops")

    def _repeat(tensor, pattern, **axes_lengths):
        if pattern == "n -> d n":
            return tensor.unsqueeze(0).repeat(axes_lengths["d"], 1)
        raise NotImplementedError(f"当前兼容版 einops.repeat 不支持该模式: {pattern}")

    def _rearrange(tensor, pattern, **axes_lengths):
        if pattern == "b l d_in -> b d_in l":
            return tensor.permute(0, 2, 1)
        if pattern == "b d_in l -> b l d_in":
            return tensor.permute(0, 2, 1)
        raise NotImplementedError(f"当前兼容版 einops.rearrange 不支持该模式: {pattern}")

    def _einsum(*args):
        pattern = args[-1]
        tensors = args[:-1]
        patterns = {
            "b l d_in, d_in n -> b l d_in n": "bld,dn->bldn",
            "b l d_in, b l n, b l d_in -> b l d_in n": "bld,bln,bld->bldn",
            "b d_in n, b n -> b d_in": "bdn,bn->bd",
        }
        if pattern not in patterns:
            raise NotImplementedError(f"当前兼容版 einops.einsum 不支持该模式: {pattern}")
        return torch.einsum(patterns[pattern], *tensors)

    einops_stub.repeat = _repeat
    einops_stub.rearrange = _rearrange
    einops_stub.einsum = _einsum
    sys.modules["einops"] = einops_stub

from TFPredictor import KFGN, KFGN_Mamba, ModelArgs, RMSNorm


# 当前脚本所在目录，用于拼接数据集路径和输出路径。
ROOT = Path(__file__).resolve().parent


# 数据集路径配置。
# 优先读取解压后的 csv/npy；如果文件不存在，会自动从 zip 中读取。
DATASETS = {
    "PEMS-BAY": {
        "aliases": {"PEMS-BAY", "PEMSBAY", "PEMSBY", "PEMS-BY", "PEMBY"},
        "csv": ROOT / "pemby" / "pemsby_flow.csv",
        "adj": ROOT / "pemby" / "pemsby-dtw-288-1-.npy",
        "zip": ROOT / "pemby" / "pemby.zip",
        "zip_csv": "pemsby_flow.csv",
        "zip_adj": "pemsby-dtw-288-1-.npy",
    },
    "METR-LA": {
        "aliases": {"METR-LA", "METRLA", "metr-la"},
        "csv": ROOT / "metr-la" / "metr-la_flow.csv",
        "adj": ROOT / "metr-la" / "metr-la-dtw-288-1-.npy",
        "zip": ROOT / "metr-la" / "metr-la.zip",
        "zip_csv": "metr-la_flow.csv",
        "zip_adj": "metr-la-dtw-288-1-.npy",
    },
    "PEMS04": {
        "aliases": {"PEMS04", "pems04"},
        "csv": ROOT / "PEMS04" / "pems04_flow.csv",
        "adj": ROOT / "PEMS04" / "pems04_adj.npy",
    },
    "PEMS08": {
        "aliases": {"PEMS08", "pems08"},
        "csv": ROOT / "PEMS08" / "pems08_flow.csv",
        "adj": ROOT / "PEMS08" / "PEMS08-dtw-288-1-.npy",
    },
}


@dataclass
class ProfileRow:
    """单行统计结果，对应最终表格中的一行。"""

    dataset: str
    variant: str
    seq_len: int
    pred_len: int
    train_input_len: int
    inference_input_len: int
    forecast_head: str
    fused_selective_scan: bool
    params_m: float
    flops_g: float | None
    train_time_per_epoch_s: float
    inference_time_s: float
    train_batches_measured: int
    train_batches_total: int
    inference_batches_measured: int
    device: str
    flops_method: str


class TransformerMixer(nn.Module):
    """Transformer 替换模块：用于替换原始 TFPredictor 中的 MambaBlock。"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.0) -> None:
        super().__init__()
        # 多头自注意力，batch_first=True 表示输入形状为 [batch, seq_len, dim]。
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        # 与原模型风格保持一致，使用 RMSNorm 而不是 LayerNorm。
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        # FFN 宽度由 --transformer-ffn-multiplier 控制，可用于参数量匹配实验。
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 标准 Transformer 子层：Attention 残差 + FFN 残差。
        attn_input = self.attn_norm(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class KFGNTransformerResidualBlock(nn.Module):
    """公平对照块：保留 KFGN，只把后续 mixer 从 Mamba 换成 Transformer。"""

    def __init__(
        self,
        args: ModelArgs,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # 每层都保留和原始 ResidualBlock 一样的 KFGN 图建模模块。
        self.kfgn = KFGN(K=args.K, A=args.A, feature_size=args.feature_size)
        self.mixer = TransformerMixer(
            d_model=args.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.norm = RMSNorm(args.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对齐原始 ResidualBlock: norm -> KFGN -> mixer -> residual。
        x1 = self.norm(x)
        x2 = self.kfgn(x1)
        x3 = self.mixer(x2)
        return x3 + x1


class TFPredictorTransformer(nn.Module):
    """KFGN + Transformer 版本，用作 TFPredictor 的公平结构对照。"""

    def __init__(
        self,
        args: ModelArgs,
        nhead: int,
        ffn_multiplier: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        nhead = choose_valid_nhead(args.d_model, nhead)
        # FFN 宽度 = d_model * multiplier；例如 6.0 可让参数量更接近 Mamba 版本。
        dim_feedforward = max(1, int(round(args.d_model * ffn_multiplier)))
        self.args = args
        self.nhead = nhead
        self.ffn_multiplier = ffn_multiplier
        self.dim_feedforward = dim_feedforward
        self.encode = nn.Linear(args.features, args.d_model)
        self.encoder_layers = nn.ModuleList(
            [
                KFGNTransformerResidualBlock(
                    args,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(args.n_layer)
            ]
        )
        self.encoder_norm = RMSNorm(args.d_model)
        self.decode = nn.Linear(args.d_model, args.features)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 输入输出形状与原始 KFGN_Mamba 保持一致。
        x = self.encode(input_ids)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.encoder_norm(x)
        return self.decode(x)


class TemporalForecastWrapper(nn.Module):
    """时间维预测头：把模型输出长度从 seq_len 映射到 pred_len。"""

    def __init__(self, backbone: nn.Module, seq_len: int, pred_len: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.temporal_proj = nn.Linear(seq_len, pred_len)
        self._init_tail_projection()

    def _init_tail_projection(self) -> None:
        """初始化为近似“取最后 pred_len 个时间步”，让训练开始更稳定。"""

        with torch.no_grad():
            self.temporal_proj.weight.zero_()
            self.temporal_proj.bias.zero_()
            copy_len = min(self.seq_len, self.pred_len)
            for i in range(copy_len):
                self.temporal_proj.weight[self.pred_len - copy_len + i, self.seq_len - copy_len + i] = 1.0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.backbone(input_ids)
        if x.size(1) != self.seq_len:
            raise RuntimeError(f"时间预测头期望长度 {self.seq_len}，实际得到 {x.size(1)}。")
        x = self.temporal_proj(x.transpose(1, 2)).transpose(1, 2)
        return x


def choose_valid_nhead(d_model: int, requested: int) -> int:
    """选择能整除 d_model 的注意力头数，避免 MultiheadAttention 报错。"""

    for candidate in [requested, 8, 4, 3, 2, 1]:
        if candidate > 0 and d_model % candidate == 0:
            return candidate
    return 1


def patch_cuda_for_cpu(device: torch.device) -> None:
    """兼容 CPU 运行：屏蔽原始模型中硬编码的 .cuda() 调用。"""

    if device.type != "cpu" or torch.cuda.is_available():
        return

    if getattr(torch.Tensor, "_profile_stats_cuda_patched", False):
        return

    def _cuda_noop(self, *args, **kwargs):
        return self

    torch.Tensor.cuda = _cuda_noop  # type: ignore[method-assign]
    torch.Tensor._profile_stats_cuda_patched = True  # type: ignore[attr-defined]


def canonical_dataset(name: str) -> str:
    """把用户输入的数据集别名统一转换为脚本内部使用的标准名称。"""

    probe = name.strip()
    for canonical, spec in DATASETS.items():
        if probe in spec["aliases"] or probe.upper() in {x.upper() for x in spec["aliases"]}:
            return canonical
    supported = ", ".join(DATASETS)
    raise ValueError(f"未知数据集 {name!r}。支持的数据集：{supported}")


def open_from_file_or_zip(spec: dict, kind: str):
    """打开数据文件；如果普通文件不存在，则从 zip 压缩包中读取。"""

    path = spec[kind]
    if path.exists():
        return path.open("rb")

    zip_path = spec.get("zip")
    zip_member = spec.get(f"zip_{kind}")
    if zip_path and zip_path.exists() and zip_member:
        archive = zipfile.ZipFile(zip_path)
        member = archive.open(zip_member, "r")
        return _ZipMemberWrapper(archive, member)

    raise FileNotFoundError(f"未找到数据文件：{path}，也未找到 zip 内文件：{zip_path}:{zip_member}")


class _ZipMemberWrapper:
    """让 zip 内部文件也能像普通文件一样用于 with 上下文。"""

    def __init__(self, archive: zipfile.ZipFile, member) -> None:
        self.archive = archive
        self.member = member

    def __enter__(self):
        return self.member

    def __exit__(self, exc_type, exc, tb):
        self.member.close()
        self.archive.close()


def load_matrix_csv(handle) -> np.ndarray:
    """读取交通流 CSV 矩阵，跳过第一行列名。"""

    text = io.TextIOWrapper(handle, encoding="utf-8")
    return np.loadtxt(text, delimiter=",", skiprows=1, dtype=np.float32)


def load_dataset_arrays(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """读取指定数据集的流量矩阵和邻接矩阵。"""

    spec = DATASETS[dataset]
    with open_from_file_or_zip(spec, "csv") as csv_handle:
        speed_matrix = load_matrix_csv(csv_handle)
    with open_from_file_or_zip(spec, "adj") as adj_handle:
        adj = np.load(adj_handle)
    return speed_matrix, adj.astype(np.float32)


def fft_transform_with_timescale(data: np.ndarray, timescale: int, n_freq: int = 3) -> np.ndarray:
    """复用原 prepare.py 的频域特征构造逻辑。"""

    data_fft = torch.fft.rfft(torch.tensor(data, dtype=torch.float32), dim=1)
    frequencies = torch.abs(data_fft)
    n_freq = min(n_freq, frequencies.size(1))
    top_frequencies, _ = torch.topk(frequencies, n_freq, dim=1)
    top_frequencies_mean = top_frequencies.mean(dim=1)

    segmented_means = []
    num_segments = data.shape[1] // timescale
    for i in range(num_segments):
        segment = top_frequencies_mean[:, i * timescale : (i + 1) * timescale]
        segmented_means.append(segment.mean(dim=1))
    if data.shape[1] % timescale != 0 or not segmented_means:
        remaining = top_frequencies_mean[:, num_segments * timescale :]
        segmented_means.append(remaining.mean(dim=1))
    return torch.stack(segmented_means, dim=1).numpy()


def prepare_dataset(
    speed_matrix: np.ndarray,
    batch_size: int,
    seq_len: int,
    pred_len: int,
    feature_limit: int,
    train_propotion: float = 0.8,
    valid_propotion: float = 0.1,
    timescale: int = 48,
) -> tuple[DataLoader, DataLoader, DataLoader, float]:
    """构造训练、验证、测试 DataLoader，保持与原项目数据处理逻辑一致。"""

    time_len = speed_matrix.shape[0]
    max_speed = float(np.max(speed_matrix))
    min_speed = float(np.min(speed_matrix))
    speed_matrix = (speed_matrix - min_speed) / (max_speed - min_speed)

    # 按滑动窗口构造输入序列和预测标签。
    speed_sequences = []
    speed_labels = []
    for i in range(time_len - seq_len - pred_len):
        speed_sequences.append(speed_matrix[i : i + seq_len])
        speed_labels.append(speed_matrix[i + seq_len : i + seq_len + pred_len])

    speed_sequences = np.asarray(speed_sequences, dtype=np.float32)
    speed_labels = np.asarray(speed_labels, dtype=np.float32)

    # 拼接频域特征，并根据邻接矩阵节点数裁剪到模型可用维度。
    fft_features = fft_transform_with_timescale(speed_sequences, timescale)
    fft_features = np.expand_dims(fft_features, axis=1).repeat(seq_len, axis=1)
    speed_sequences = np.concatenate((speed_sequences, fft_features), axis=2)
    if feature_limit > 0 and speed_sequences.shape[2] > feature_limit:
        speed_sequences = speed_sequences[:, :, :feature_limit]
    if feature_limit > 0 and speed_labels.shape[2] > feature_limit:
        speed_labels = speed_labels[:, :, :feature_limit]

    sample_size = speed_sequences.shape[0]
    # 按原项目比例划分 train/valid/test。
    train_index = int(sample_size * train_propotion)
    valid_index = int(sample_size * (train_propotion + valid_propotion))

    train_data = torch.tensor(speed_sequences[:train_index], dtype=torch.float32)
    train_label = torch.tensor(speed_labels[:train_index], dtype=torch.float32)
    valid_data = torch.tensor(speed_sequences[train_index:valid_index], dtype=torch.float32)
    valid_label = torch.tensor(speed_labels[train_index:valid_index], dtype=torch.float32)
    test_data = torch.tensor(speed_sequences[valid_index:], dtype=torch.float32)
    test_label = torch.tensor(speed_labels[valid_index:], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_data, train_label), batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(TensorDataset(valid_data, valid_label), batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, valid_loader, test_loader, max_speed


def build_model(
    variant: str,
    adj: np.ndarray,
    features: int,
    seq_len: int,
    pred_len: int,
    k: int,
    layers: int,
    transformer_heads: int,
    transformer_ffn_multiplier: float,
    forecast_head_mode: str,
    device: torch.device,
) -> tuple[nn.Module, str]:
    """根据 variant 构造需要统计的模型。"""

    forecast_head = "none"
    if variant == "TFPredictor":
        args = ModelArgs(
            K=k,
            # A 在构造阶段保持 CPU，因为原始 KFGN 初始化时会创建 CPU 中间张量；
            # 模型构造完成后再整体移动到目标设备。
            A=torch.tensor(adj, dtype=torch.float32),
            feature_size=features,
            d_model=features,
            n_layer=layers,
            features=features,
        )
        model = KFGN_Mamba(args)

    elif variant == "TFPredictor-Transformer":
        # Transformer 对照版本同样保留 KFGN、层数和输入输出维度。
        args = ModelArgs(
            K=k,
            A=torch.tensor(adj, dtype=torch.float32),
            feature_size=features,
            d_model=features,
            n_layer=layers,
            features=features,
        )
        model = TFPredictorTransformer(
            args,
            nhead=transformer_heads,
            ffn_multiplier=transformer_ffn_multiplier,
        )

    else:
        raise ValueError(f"未知模型变体 {variant!r}")

    should_add_head = forecast_head_mode == "always" or (
        forecast_head_mode == "auto" and seq_len != pred_len
    )
    if should_add_head:
        model = TemporalForecastWrapper(model, seq_len=seq_len, pred_len=pred_len)
        forecast_head = "temporal_linear"
    elif seq_len != pred_len:
        raise ValueError("seq_len 和 pred_len 不一致时需要预测头，请使用 --forecast-head auto 或 always。")

    return model.to(device), forecast_head


def count_params_m(model: nn.Module) -> float:
    """统计可训练参数量，单位为百万参数 M。"""

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1_000_000


def has_fused_selective_scan(model: nn.Module) -> bool:
    """检查当前模型中是否存在启用 fused selective scan 的 MambaBlock。"""

    return any(getattr(module, "use_fused_selective_scan", False) for module in model.modules())


def estimate_flops_g(model: nn.Module, example: torch.Tensor) -> tuple[float | None, str]:
    """估算单个 batch 的 FLOPs，优先使用 thop，失败时尝试 PyTorch profiler。"""

    # 在模型副本上统计 FLOPs，避免 thop 注册 hook 后污染后续训练计时。
    try:
        profile_model = copy.deepcopy(model).to(example.device)
    except Exception:
        profile_model = model

    model_was_training = profile_model.training
    profile_model.eval()
    try:
        from thop import profile

        flops, _ = profile(profile_model, inputs=(example,), verbose=False)
        return float(flops) / 1_000_000_000, "thop"
    except Exception as thop_error:
        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if example.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.no_grad(), torch.profiler.profile(activities=activities, with_flops=True) as prof:
                profile_model(example)
            flops = sum(event.flops for event in prof.key_averages() if event.flops)
            if flops:
                return float(flops) / 1_000_000_000, "torch.profiler"
            return None, f"unavailable ({type(thop_error).__name__})"
        except Exception as profiler_error:
            return None, f"unavailable ({type(thop_error).__name__}, {type(profiler_error).__name__})"
    finally:
        profile_model.train(model_was_training)


def sync_if_needed(device: torch.device) -> None:
    """CUDA 计时时同步 GPU，避免异步执行导致耗时偏小。"""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def iter_limited(loader: DataLoader, max_batches: int) -> Iterable:
    """遍历 DataLoader，可通过 max_batches 限制统计批次数。"""

    for index, batch in enumerate(loader):
        if max_batches > 0 and index >= max_batches:
            break
        yield batch


def measure_train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    max_batches: int,
    device: torch.device,
    lr: float,
    expected_input_len: int,
    use_amp: bool,
) -> tuple[float, int, int]:
    """测量训练一个 epoch 的耗时；可用部分 batch 外推完整 epoch。"""

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    total_batches = len(train_loader)
    measured_batches = 0

    sync_if_needed(device)
    start = time.perf_counter()
    for inputs, labels in iter_limited(train_loader, max_batches):
        if inputs.size(1) != expected_input_len:
            raise RuntimeError(f"训练输入长度应为 {expected_input_len}，实际为 {inputs.size(1)}。")
        inputs = inputs.to(device)
        labels = labels.to(device).squeeze()
        optimizer.zero_grad(set_to_none=True)
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp and device.type == "cuda" else nullcontext()
        with autocast_context:
            pred = model(inputs)
            loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        measured_batches += 1
    sync_if_needed(device)
    elapsed = time.perf_counter() - start

    if measured_batches == 0:
        raise RuntimeError("没有统计到训练 batch，请检查 batch size 和数据集长度。")

    # max_batches=0 表示完整 epoch 实测；否则按平均 batch 时间外推完整 epoch。
    if max_batches == 0 or measured_batches == total_batches:
        return elapsed, measured_batches, total_batches
    return elapsed / measured_batches * total_batches, measured_batches, total_batches


def measure_inference(
    model: nn.Module,
    test_loader: DataLoader,
    max_batches: int,
    warmup_batches: int,
    device: torch.device,
    expected_input_len: int,
    use_amp: bool,
) -> tuple[float, int]:
    """测量测试集推理耗时；可选择 warmup 批次和统计批次数。"""

    model.eval()
    measured_batches = 0

    with torch.inference_mode():
        for inputs, _ in iter_limited(test_loader, warmup_batches):
            if inputs.size(1) != expected_input_len:
                raise RuntimeError(f"推理 warmup 输入长度应为 {expected_input_len}，实际为 {inputs.size(1)}。")
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp and device.type == "cuda" else nullcontext()
            with autocast_context:
                model(inputs.to(device))
        sync_if_needed(device)

        start = time.perf_counter()
        for inputs, _ in iter_limited(test_loader, max_batches):
            if inputs.size(1) != expected_input_len:
                raise RuntimeError(f"推理输入长度应为 {expected_input_len}，实际为 {inputs.size(1)}。")
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp and device.type == "cuda" else nullcontext()
            with autocast_context:
                model(inputs.to(device))
            measured_batches += 1
        sync_if_needed(device)
        elapsed = time.perf_counter() - start

    if measured_batches == 0:
        raise RuntimeError("没有统计到推理 batch，请检查 batch size 和数据集长度。")
    return elapsed, measured_batches


def profile_one(
    dataset: str,
    variant: str,
    seq_len: int,
    pred_len: int,
    args: argparse.Namespace,
    device: torch.device,
) -> ProfileRow:
    """完成单个 数据集/模型变体 的完整统计流程。"""

    speed_matrix, adj = load_dataset_arrays(dataset)
    feature_limit = args.feature_limit or int(adj.shape[0])
    train_loader, _, test_loader, _ = prepare_dataset(
        speed_matrix=speed_matrix,
        batch_size=args.batch_size,
        seq_len=seq_len,
        pred_len=pred_len,
        feature_limit=feature_limit,
        timescale=args.timescale,
    )

    patch_cuda_for_cpu(device)
    model, forecast_head = build_model(
        variant=variant,
        adj=adj[:feature_limit, :feature_limit],
        features=feature_limit,
        seq_len=seq_len,
        pred_len=pred_len,
        k=args.k,
        layers=args.layers,
        transformer_heads=args.transformer_heads,
        transformer_ffn_multiplier=args.transformer_ffn_multiplier,
        forecast_head_mode=args.forecast_head,
        device=device,
    )

    example_inputs, _ = next(iter(train_loader))
    example_inputs = example_inputs.to(device)

    # 依次统计参数量、FLOPs、训练耗时和推理耗时。
    fused_selective_scan = has_fused_selective_scan(model)
    params_m = count_params_m(model)
    flops_g, flops_method = estimate_flops_g(model, example_inputs)
    train_time, train_measured, train_total = measure_train_epoch(
        model=model,
        train_loader=train_loader,
        max_batches=args.max_train_batches,
        device=device,
        lr=args.lr,
        expected_input_len=seq_len,
        use_amp=args.amp,
    )
    inference_time, inference_measured = measure_inference(
        model=model,
        test_loader=test_loader,
        max_batches=args.max_inference_batches,
        warmup_batches=args.warmup_batches,
        device=device,
        expected_input_len=seq_len,
        use_amp=args.amp,
    )

    row = ProfileRow(
        dataset=dataset,
        variant=variant,
        seq_len=seq_len,
        pred_len=pred_len,
        train_input_len=seq_len,
        inference_input_len=seq_len,
        forecast_head=forecast_head,
        fused_selective_scan=fused_selective_scan,
        params_m=params_m,
        flops_g=flops_g,
        train_time_per_epoch_s=train_time,
        inference_time_s=inference_time,
        train_batches_measured=train_measured,
        train_batches_total=train_total,
        inference_batches_measured=inference_measured,
        device=str(device),
        flops_method=flops_method,
    )
    del model, train_loader, test_loader, example_inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def format_float(value: float | None, digits: int = 4) -> str:
    """格式化浮点数，None 或 NaN 输出为 NA。"""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def print_markdown(rows: list[ProfileRow]) -> None:
    """在终端打印 Markdown 表格，方便直接复制到论文或文档。"""

    print()
    print("| Dataset | Variant | Train Input Len | Inference Input Len | Pred Len | Forecast Head | Fused Scan | Params (M) | FLOPs (G) | Training Time per Epoch (s) | Inference Time (s) |")
    print("|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| "
            f"{row.dataset} | {row.variant} | "
            f"{row.train_input_len} | {row.inference_input_len} | {row.pred_len} | {row.forecast_head} | "
            f"{row.fused_selective_scan} | "
            f"{format_float(row.params_m)} | {format_float(row.flops_g)} | "
            f"{format_float(row.train_time_per_epoch_s, 2)} | {format_float(row.inference_time_s, 2)} |"
        )


def write_outputs(rows: list[ProfileRow], output_prefix: Path) -> None:
    """把统计结果保存为 CSV 和 JSON。"""

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    json_path = output_prefix.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(row) for row in rows], handle, indent=2, ensure_ascii=False)

    print(f"\n已保存：{csv_path}")
    print(f"已保存：{json_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="统计 TFPredictor 的参数量、FLOPs 和运行耗时。")
    parser.add_argument("--datasets", nargs="+", default=["PEMS-BAY", "METR-LA"])
    parser.add_argument("--variants", nargs="+", default=["TFPredictor", "TFPredictor-Transformer"])
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=None, help="批量测试多个输入序列长度，例如：--seq-lens 96 198 368。")
    parser.add_argument("--pred-len", type=int, default=5)
    parser.add_argument("--pred-lens", nargs="+", type=int, default=None, help="批量测试多个预测长度；可与 --seq-lens 按位置配对。")
    parser.add_argument(
        "--match-pred-len-to-seq-len",
        action="store_true",
        help="让每组 pred_len 自动等于对应的 seq_len，例如 96->96、198->198。",
    )
    parser.add_argument("--timescale", type=int, default=48)
    parser.add_argument("--feature-limit", type=int, default=0, help="0 表示使用邻接矩阵大小作为特征数。")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--transformer-heads", type=int, default=1)
    parser.add_argument(
        "--transformer-ffn-multiplier",
        type=float,
        default=2.0,
        help="Transformer FFN 隐藏层宽度倍率；参数匹配实验建议使用约 6.0。",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true", help="使用 CUDA 混合精度以降低长序列训练显存占用。")
    parser.add_argument(
        "--forecast-head",
        choices=["auto", "always", "never"],
        default="auto",
        help="时间预测头策略；auto 表示 seq_len != pred_len 时自动添加。",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=20,
        help="用于估算一个 epoch 的 batch 数；0 表示完整实测一个 epoch。",
    )
    parser.add_argument(
        "--max-inference-batches",
        type=int,
        default=0,
        help="用于推理计时的 batch 数；0 表示完整测试集。",
    )
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "profile_results")
    return parser.parse_args()


def resolve_seq_pred_pairs(args: argparse.Namespace) -> list[tuple[int, int]]:
    """根据命令行参数生成 seq_len 和 pred_len 的配对列表。"""

    seq_lens = args.seq_lens if args.seq_lens else [args.seq_len]
    if args.match_pred_len_to_seq_len:
        return [(seq_len, seq_len) for seq_len in seq_lens]

    if args.pred_lens:
        if len(args.pred_lens) == 1:
            return [(seq_len, args.pred_lens[0]) for seq_len in seq_lens]
        if len(args.pred_lens) != len(seq_lens):
            raise ValueError("--pred-lens 的数量必须等于 --seq-lens，或者只提供一个 pred_len 供所有 seq_len 共用。")
        return list(zip(seq_lens, args.pred_lens))

    return [(seq_len, args.pred_len) for seq_len in seq_lens]


def main() -> None:
    """脚本入口：解析参数，逐个数据集和模型变体进行统计。"""

    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已指定使用 CUDA，但 torch.cuda.is_available() 返回 False。")

    os.environ.setdefault("PYTHONHASHSEED", "0")
    torch.manual_seed(0)
    np.random.seed(0)

    rows = []
    datasets = [canonical_dataset(name) for name in args.datasets]
    seq_pred_pairs = resolve_seq_pred_pairs(args)
    for dataset in datasets:
        for variant in args.variants:
            for seq_len, pred_len in seq_pred_pairs:
                print(f"正在统计 {dataset} / {variant}，seq_len={seq_len}，pred_len={pred_len}，设备：{device} ...", flush=True)
                rows.append(profile_one(dataset, variant, seq_len, pred_len, args, device))

    print_markdown(rows)
    write_outputs(rows, args.output_prefix)


if __name__ == "__main__":
    main()
