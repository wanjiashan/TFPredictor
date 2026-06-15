import torch
import torch.nn as nn
import torch.nn.functional as F
from sigkan import KANLinear
import torch.nn as nn
import torch
import torch.nn as nn

class AddAndNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(AddAndNorm, self).__init__()
        self.norm_layer = nn.LayerNorm(normalized_shape)

    def forward(self, inputs):
        # 确保输入列表中每个张量的形状一致
        skip, gating_output = inputs

        # 检查并对齐 skip 和 gating_output 的形状
        if skip.dim() == 2 and gating_output.dim() == 3:
            # 扩展 skip 的形状以匹配 gating_output
            skip = skip.unsqueeze(1).expand_as(gating_output)
        elif gating_output.dim() == 2 and skip.dim() == 3:
            # 扩展 gating_output 的形状以匹配 skip
            gating_output = gating_output.unsqueeze(1).expand_as(skip)

        # 堆叠并在第一个维度上求和
        tmp = torch.stack([skip, gating_output], dim=0).sum(dim=0)
        tmp = self.norm_layer(tmp)
        return tmp


import torch
import torch.nn as nn

class Gate(nn.Module):
    def __init__(self, hidden_layer_size=None):
        super(Gate, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dense_layer = None
        self.gated_layer = None

    def build(self, input_shape, device):
        if self.hidden_layer_size is None:
            self.hidden_layer_size = input_shape[-1]
        self.dense_layer = nn.Linear(input_shape[-1], self.hidden_layer_size).to(device)
        self.gated_layer = nn.Linear(input_shape[-1], self.hidden_layer_size).to(device)

    def forward(self, inputs):
        if self.dense_layer is None:
            self.build(inputs.shape, inputs.device)
        dense_output = self.dense_layer(inputs)
        gated_output = torch.sigmoid(self.gated_layer(inputs))
        return dense_output * gated_output


import torch.nn.functional as F


class GRKAN(nn.Module):
    def __init__(self, hidden_layer_size, output_size=None, activation=None, dropout=0.1, use_bias=False, **kwargs):
        super(GRKAN, self).__init__(**kwargs)
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        self.dropout_value = dropout
        self.use_bias = use_bias

        # 使用 `torch.nn.functional` 中的激活函数
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.output_size is None:
            self.output_size = self.hidden_layer_size
        self.skip_layer = nn.Linear(self.output_size, self.output_size)

        self.hidden_layer_1 = KANLinear(self.hidden_layer_size, base_activation='elu', dropout=self.dropout_value,
                                        use_bias=self.use_bias, use_layernorm=False)
        self.hidden_layer_2 = KANLinear(self.hidden_layer_size, dropout=self.dropout_value, use_bias=self.use_bias,
                                        use_layernorm=False)
        self.gate_layer = Gate(self.output_size)
        self.add_and_norm_layer = AddAndNorm(self.output_size)

    import torch.nn.functional as F

    def forward(self, inputs):
        if self.skip_layer is None:
            skip = inputs
        else:
            skip = self.skip_layer(inputs)

        hidden = self.hidden_layer_1(inputs)
        hidden = self.hidden_layer_2(hidden)
        hidden = self.dropout(hidden)
        gating_output = self.gate_layer(hidden)
        output = self.add_and_norm_layer([skip, gating_output])

        if self.activation is not None:
            if isinstance(self.activation, torch.nn.Softmax):
                output = self.activation(output, dim=-1)  # 指定 dim 参数
            else:
                output = self.activation(output)

        return output


class GRN(nn.Module):
    def __init__(self, hidden_layer_size, output_size=None, activation=None, dropout=0.1):
        super(GRN, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size or hidden_layer_size
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.skip_layer = nn.Linear(hidden_layer_size, self.output_size)

        self.hidden_layer_1 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.hidden_layer_2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.gate_layer = Gate(self.output_size)
        self.add_and_norm_layer = AddAndNorm()

    def forward(self, inputs):
        skip = self.skip_layer(inputs)

        hidden = F.elu(self.hidden_layer_1(inputs))
        hidden = self.hidden_layer_2(hidden)
        hidden = self.dropout(hidden)
        gating_output = self.gate_layer(hidden)
        output = self.add_and_norm_layer([skip, gating_output])

        if self.activation:
            output = self.activation(output)
        return output
