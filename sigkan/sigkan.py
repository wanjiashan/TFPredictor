import numpy as np
import torch
import torch.nn as nn
import iisignature
from sigkan import KANLinear, GRKAN, GRN

class SigKAN(nn.Module):
    def __init__(self, unit, sig_level, dropout=0.):
        super(SigKAN, self).__init__()
        self.unit = unit
        self.sig_level = sig_level
        # 准备签名计算的设置
        self.sig_setup = iisignature.prepare(unit, self.sig_level)
        self.kan_layer = KANLinear(unit, dropout=dropout, use_bias=False, use_layernorm=False)
        self.sig_to_weight = GRKAN(unit, activation='softmax', dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        try:
            batch_size, seq_len, num_features = inputs.shape

            # 计算签名特征
            sig_list = []
            for i in range(batch_size):
                sample = inputs[i].detach().cpu().numpy()
                sig = iisignature.logsig(sample, self.sig_setup)
                sig_list.append(sig)

            sig_tensor = torch.tensor(np.array(sig_list), dtype=torch.float32).cuda()

            # 调整 sig_tensor 形状为 (batch_size, num_features)
            sig_tensor = sig_tensor.view(batch_size, -1)

            # 确保 sig_tensor 的维度与 self.unit 匹配
            if sig_tensor.shape[1] > self.unit:
                sig_tensor = sig_tensor[:, :self.unit]

            # 生成权重
            weights = self.sig_to_weight(sig_tensor)

            # 如果 weights 形状不正确，需要进行 reshape 或 squeeze 操作
            if len(weights.shape) > 2:
                weights = weights.mean(dim=1)

            weights = weights.view(batch_size, 1, num_features).expand(-1, seq_len, -1)

            # 通过 KAN 层处理输入
            kan_out = self.kan_layer(inputs)
            kan_out = self.dropout(kan_out)

            return kan_out * weights

        except Exception as e:
            raise e


    # def forward(self, inputs):
    #     try:
    #         batch_size, seq_len, num_features = inputs.shape
    #
    #         # 计算签名特征
    #         sig_list = []
    #         for i in range(batch_size):
    #             sample = inputs[i].detach().cpu().numpy()
    #             sig = iisignature.logsig(sample, self.sig_setup)
    #             sig_list.append(sig)
    #
    #         sig_tensor = torch.tensor(np.array(sig_list), dtype=torch.float32).cuda()
    #
    #         # 调整 sig_tensor 形状为 (batch_size, num_features)
    #         sig_tensor = sig_tensor.view(batch_size, -1)
    #
    #         # 确保 sig_tensor 的维度与 self.unit 匹配
    #         if sig_tensor.shape[1] > self.unit:
    #             sig_tensor = sig_tensor[:, :self.unit]
    #
    #         # 生成权重
    #         weights = self.sig_to_weight(sig_tensor)
    #
    #         # 检查 weights 的形状是否正确
    #         print(f"Sig tensor shape: {sig_tensor.shape}")
    #         print(f"Weights shape before view: {weights.shape}")
    #
    #         # 如果 weights 形状不正确，需要进行 reshape 或 squeeze 操作
    #         if len(weights.shape) > 2:
    #             weights = weights.mean(dim=1)
    #
    #         assert weights.shape == (batch_size, num_features), f"Unexpected weights shape: {weights.shape}"
    #
    #         # 将 weights 调整为 [batch_size, 1, num_features]，然后扩展为 [batch_size, seq_len, num_features]
    #         weights = weights.view(batch_size, 1, num_features).expand(-1, seq_len, -1)
    #
    #         # 打印调试信息
    #         print(f"Inputs shape: {inputs.shape}")
    #         print(f"Weights shape after expand: {weights.shape}")
    #
    #         # 通过 KAN 层处理输入
    #         kan_out = self.kan_layer(inputs)
    #         kan_out = self.dropout(kan_out)
    #
    #         # 打印调试信息
    #         print(f"Kan out shape: {kan_out.shape}")
    #
    #         return kan_out * weights
    #
    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    #         raise e


class SigDense(nn.Module):
    def __init__(self, unit, sig_level, dropout=0.):
        super(SigDense, self).__init__()
        self.unit = unit
        self.sig_level = sig_level
        self.sig_setup = iisignature.prepare([unit, unit], self.sig_level)  # 使用 iisignature 初始化签名层
        self.dense_layer = nn.Linear(unit, unit)
        self.sig_to_weight = GRN(unit, activation='softmax', dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.time_weigthing_kernel = None

    def build(self, input_shape):
        seq_length = input_shape[1]
        self.time_weigthing_kernel = nn.Parameter(torch.ones(seq_length, 1))

    def forward(self, inputs):
        if self.time_weigthing_kernel is None:
            self.build(inputs.shape)

        # 对每个时间步加权
        inputs = self.time_weigthing_kernel * inputs

        # 计算签名特征
        batch_size = inputs.size(0)
        inputs_reshaped = inputs.view(-1, inputs.size(1), inputs.size(2))
        sig = torch.stack([torch.tensor(iisignature.sig(inputs_reshaped[i].cpu().numpy(), self.sig_setup))
                           for i in range(inputs_reshaped.size(0))])
        sig = sig.view(batch_size, -1)

        # 通过 GRN 层生成权重
        weights = self.sig_to_weight(sig)

        # 通过 Dense 层处理输入
        dense_out = self.dense_layer(inputs)

        # Dropout
        dense_out = self.dropout(dense_out)

        # 加权输出
        return dense_out * weights.unsqueeze(1)
