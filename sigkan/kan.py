import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GridInitializer:
    def __init__(self, grid_range, grid_size, spline_order):
        self.grid_range = grid_range
        self.grid_size = grid_size
        self.spline_order = spline_order

    def __call__(self, shape):
        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        grid = torch.linspace(
            start=-self.spline_order * h + self.grid_range[0],
            end=(self.grid_size + self.spline_order) * h + self.grid_range[0],
            steps=self.grid_size + 2 * self.spline_order + 1
        )
        grid = torch.tile(grid.unsqueeze(0).unsqueeze(0), [shape[0], shape[1], 1])
        return grid




class KANLinear(nn.Module):
    def __init__(
        self,
        units,
        grid_size=3,
        spline_order=3,
        base_activation='silu',
        grid_range=[-1, 1],
        dropout=0.,
        use_bias=True,
        use_layernorm=True
    ):
        super(KANLinear, self).__init__()
        self.units = units
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = getattr(F, base_activation)
        self.grid_range = grid_range
        self.use_bias = use_bias
        self.use_layernorm = use_layernorm

        if self.use_layernorm:
            self.layer_norm = nn.LayerNorm(normalized_shape=None)

        self.dropout = nn.Dropout(dropout)
        self.grid_initializer = GridInitializer(grid_range, grid_size, spline_order)

        # 延迟初始化的权重参数
        self.base_weight = None
        self.spline_weight = None
        self.base_bias = None

    def forward(self, x):
        # 检查并初始化参数
        if self.base_weight is None:
            self.in_features = x.size(-1)
            device = x.device  # 获取输入张量所在的设备
            self.base_weight = nn.Parameter(torch.empty((self.units, self.in_features), device=device))
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

            self.spline_weight = nn.Parameter(torch.empty((self.units, self.in_features * (self.grid_size + self.spline_order)), device=device))
            nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

            if self.use_bias:
                self.base_bias = nn.Parameter(torch.zeros(self.units, device=device))

            # 初始化 grid 并移动到设备
            self.grid = nn.Parameter(self.grid_initializer([1, self.in_features]).to(device), requires_grad=False)

        if self.use_layernorm:
            x = self.layer_norm(x)

        # 确保所有操作在同一设备上进行
        base_output = torch.matmul(self.base_activation(x), self.base_weight.t())
        if self.use_bias:
            base_output += self.base_bias

        spline_output = torch.matmul(self.b_splines(x), self.spline_weight.t())
        return self.dropout(base_output) + self.dropout(spline_output)

    def b_splines(self, x):
        batch_size = x.size(0)
        x_expanded = x.unsqueeze(-1)  # shape: [batch_size, ..., in_features, 1]

        grid_expanded = self.grid.expand(batch_size, -1, -1)

        # 遍历输入的其他维度，调整 grid_expanded 的维度
        for dim in range(1, len(x.shape) - 1):
            grid_expanded = grid_expanded.unsqueeze(1).expand(-1, x.size(dim), -1, -1)

        bases = ((x_expanded >= grid_expanded[..., :-1]) & (x_expanded < grid_expanded[..., 1:])).float()

        for k in range(1, self.spline_order + 1):
            left_denominator = grid_expanded[..., k:-1] - grid_expanded[..., :-(k + 1)]
            right_denominator = grid_expanded[..., k + 1:] - grid_expanded[..., 1:-k]

            left = (x_expanded - grid_expanded[..., :-(k + 1)]) / left_denominator
            right = (grid_expanded[..., k + 1:] - x_expanded) / right_denominator
            bases = left * bases[..., :-1] + right * bases[..., 1:]

        bases = bases.view(batch_size, -1, bases.size(-2) * bases.size(-1))
        return bases


