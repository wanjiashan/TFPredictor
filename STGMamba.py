from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from modules import DynamicFilterGNN
from sigkan import KANLinear, GRKAN, GRN, SigKAN

class KFGN(nn.Module):
    def __init__(self, K, A, feature_size, sig_level, Clamp_A=True, dropout=0.):
        super(KFGN, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size
        self.K = K
        self.A_list = []

        D_inverse = torch.diag(1 / torch.sum(A, 0))
        norm_A = torch.matmul(D_inverse, A)
        A = norm_A

        A_temp = torch.eye(feature_size, feature_size)
        for i in range(K):
            A_temp = torch.matmul(A_temp, A)
            if Clamp_A:
                A_temp = torch.clamp(A_temp, max=1.)
            self.A_list.append(A_temp)

        self.gc_list = nn.ModuleList([DynamicFilterGNN(feature_size, feature_size, self.A_list[i], bias=False) for i in range(K)])
        gc_input_size = self.feature_size * K

        self.fl = nn.Linear(gc_input_size + self.hidden_size, self.hidden_size)
        self.il = nn.Linear(gc_input_size + self.hidden_size, self.hidden_size)
        self.ol = nn.Linear(gc_input_size + self.hidden_size, self.hidden_size)
        self.Cl = nn.Linear(gc_input_size + self.hidden_size, self.hidden_size)

        self.Neighbor_weight = Parameter(torch.FloatTensor(feature_size))
        stdv = 1. / math.sqrt(feature_size)
        self.Neighbor_weight.data.uniform_(-stdv, stdv)

        # SigKAN 部分替换 LSTM
        self.sigkan = SigKAN(unit=feature_size, sig_level=sig_level, dropout=dropout)

        # Kalman Filtering 相关参数
        self.c = torch.nn.Parameter(torch.Tensor([1]))

    def forward(self, input, Hidden_State=None, Cell_State=None):
        batch_size, time_steps, _ = input.shape
        if Hidden_State is None:
            Hidden_State = Variable(torch.zeros(batch_size, self.feature_size).cuda())
        if Cell_State is None:
            Cell_State = Variable(torch.zeros(batch_size, self.feature_size).cuda())

        Hidden_State = Hidden_State.unsqueeze(1).expand(-1, time_steps, -1)
        Cell_State = Cell_State.unsqueeze(1).expand(-1, time_steps, -1)

        x = input
        # print(x.shape)  # 打印 x 的形状

        gc = self.gc_list[0](x)
        for i in range(1, self.K):
            gc = torch.cat((gc, self.gc_list[i](x)), 1)

        combined = torch.cat((gc, Hidden_State), 1)
        # print("Combined shape after cat: ", combined.shape)

        # 确保在 view 操作时，新形状的元素数量一致
        dim1, dim2, dim3 = combined.shape
        combined = combined.view(dim1, time_steps, -1)
        # print("Combined shape after view: ", combined.shape)

        f = torch.sigmoid(self.fl(combined))
        i = torch.sigmoid(self.il(combined))
        o = torch.sigmoid(self.ol(combined))
        C = torch.tanh(self.Cl(combined))

        NC = torch.mul(Cell_State,
                       torch.mv(Variable(self.A_list[-1], requires_grad=False).cuda(), self.Neighbor_weight))
        Cell_State = f * NC + i * C
        Hidden_State = o * torch.tanh(Cell_State)

        # SigKAN 替换 LSTM
        sigkan_out = self.sigkan(input)

        # Kalman Filtering
        var1, var2 = torch.var(input), torch.var(gc)

        pred = (Hidden_State * var1 * self.c + sigkan_out * var2) / (var1 + var2 * self.c)

        return pred


# Mamba Network

@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    features: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False
    K: int = 3
    A: torch.Tensor = None
    feature_size: int = None
    sig_level: int = 2
    dropout: float = 0.0

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class KFGN_Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.kfgn = KFGN(K=args.K, A=args.A, feature_size=args.feature_size, sig_level=args.sig_level, dropout=args.dropout)
        self.encode = nn.Linear(args.features, args.d_model)
        self.encoder_layers = nn.ModuleList([ResidualBlock(args, self.kfgn) for _ in range(args.n_layer)])
        self.encoder_norm = RMSNorm(args.d_model)
        self.decode = nn.Linear(args.d_model, args.features)

    def forward(self, input_ids):
        x = self.encode(input_ids)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.encoder_norm(x)
        x = self.decode(x)
        return x



# Residual Block in Mamba Model
class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs, kfgn: KFGN):
        super().__init__()
        self.args = args
        # 传递 sig_level 参数
        self.kfgn = KFGN(K=args.K, A=args.A, feature_size=args.feature_size, sig_level=args.sig_level,
                         dropout=args.dropout)
        self.mixer = MambaBlock(args, self.kfgn)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        x0 = x
        x1 = self.norm(x)
        x2 = self.kfgn(x1)
        x3 = self.mixer(x2)
        output = x3 + x1

        return output


class MambaBlock(nn.Module):    
    def __init__(self, args: ModelArgs, kfgn: KFGN):
        super().__init__()
        self.args = args
        self.kfgn = kfgn

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        

    def forward(self, x):
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape
    
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        # This is the new version of Selective Scan Algorithm named as "Graph Selective Scan"
        #In Graph Selective Scan, we use the Feed-Forward graph information from KFGN, and incorporate the Feed-Forward information with "delta"
        temp_adj = self.kfgn.gc_list[-1].get_transformed_adjacency()        
        temp_adj_padded = torch.ones(d_in, d_in, device=temp_adj.device)       
        temp_adj_padded[:temp_adj.size(0), :temp_adj.size(1)] = temp_adj
        
        delta_p = torch.matmul(delta,temp_adj_padded)
        
        # The fused param delta_p will participate in the following upgrading of deltaA and deltaB_u
        deltaA = torch.exp(einsum(delta_p, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta_p, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
    
        return y

    
class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

