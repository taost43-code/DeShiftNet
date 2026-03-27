import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

# --- Helper Functions ---

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

def _init_weights(module, name='', scheme=''):
    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # Default initialization
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def channel_shuffle(x, groups):
    b, c, h, w = x.size()
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(b, -1, h, w)
    return x

def named_apply(fn, module, name='', depth_first=True, include_root=False):
    if include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_fullname = name + '.' + child_name if name else child_name
        if depth_first:
            named_apply(fn, child_module, name=child_fullname, depth_first=depth_first, include_root=True)
            continue
        fn(module=child_module, name=child_fullname)

# --- Basic Blocks ---

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class DownSingle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = SingleConv(in_channels, out_channels)
    def forward(self, x):
        return self.conv(self.pool(x))

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x):
        return self.conv(self.pool(x))

class DWConv(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class DeformableShiftMLP(nn.Module):
    """
    Content-aware axial mixing Token-MLP used in UNext_EMCAD.
    Applies soft-shift along height and width with learned gating.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop: float = 0., shift_size: int = 5, use_standard_shift: bool = False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self.use_standard_shift = bool(use_standard_shift)
        # token MLP
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        # content-aware gating: height on input channels, width on hidden channels
        self.gate_h = nn.Conv2d(in_features, shift_size, kernel_size=1, stride=1, padding=0)
        self.gate_w = nn.Conv2d(hidden_features, shift_size, kernel_size=1, stride=1, padding=0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _soft_shift_h(self, feat_padded):
        B, C, Hp, Wp = feat_padded.shape
        logits = self.gate_h(feat_padded)  # (B, S, Hp, Wp)
        weights = torch.softmax(logits, dim=1)
        acc = 0.0
        idx = 0
        for s in range(-self.pad, self.pad + 1):
            rolled = torch.roll(feat_padded, shifts=s, dims=2)
            w = weights[:, idx:idx+1, :, :]
            acc = acc + rolled * w
            idx += 1
        acc = torch.narrow(acc, 2, self.pad, Hp - 2 * self.pad)
        acc = torch.narrow(acc, 3, self.pad, Wp - 2 * self.pad)
        return acc

    def _soft_shift_w(self, feat_padded):
        B, C, Hp, Wp = feat_padded.shape
        logits = self.gate_w(feat_padded)  # (B, S, Hp, Wp)
        weights = torch.softmax(logits, dim=1)
        acc = 0.0
        idx = 0
        for s in range(-self.pad, self.pad + 1):
            rolled = torch.roll(feat_padded, shifts=s, dims=3)
            w = weights[:, idx:idx+1, :, :]
            acc = acc + rolled * w
            idx += 1
        acc = torch.narrow(acc, 2, self.pad, Hp - 2 * self.pad)
        acc = torch.narrow(acc, 3, self.pad, Wp - 2 * self.pad)
        return acc

    def forward(self, x, H, W):
        # x: (B, N, C=in_features)
        B, N, C = x.shape

        if self.use_standard_shift:
            xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
            xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
            xs = torch.chunk(xn, self.shift_size, 1)
            x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad, H)
            x_s = torch.narrow(x_cat, 3, self.pad, W)
            x_s = x_s.reshape(B, C, H * W).contiguous()
            x_shift_r = x_s.transpose(1, 2)

            tokens = self.fc1(x_shift_r)
            tokens = self.dwconv(tokens, H, W)
            tokens = self.act(tokens)
            tokens = self.drop(tokens)

            Ch = self.hidden_features
            xn = tokens.transpose(1, 2).view(B, Ch, H, W).contiguous()
            xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
            xs = torch.chunk(xn, self.shift_size, 1)
            x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad, H)
            x_s = torch.narrow(x_cat, 3, self.pad, W)
            x_s = x_s.reshape(B, Ch, H * W).contiguous()
            x_shift_c = x_s.transpose(1, 2)

            out = self.fc2(x_shift_c)
            out = self.drop(out)
            return out

        # Deformable content-aware axial soft-shift path
        fmap = x.transpose(1, 2).view(B, C, H, W).contiguous()
        fmap_p = F.pad(fmap, (self.pad, self.pad, self.pad, self.pad), mode="constant", value=0)
        mixed_h = self._soft_shift_h(fmap_p)
        tokens = mixed_h.reshape(B, C, H * W).transpose(1, 2).contiguous()

        tokens = self.fc1(tokens)
        tokens = self.dwconv(tokens, H, W)
        tokens = self.act(tokens)
        tokens = self.drop(tokens)

        Ch = self.hidden_features
        fmap2 = tokens.transpose(1, 2).view(B, Ch, H, W).contiguous()
        fmap2_p = F.pad(fmap2, (self.pad, self.pad, self.pad, self.pad), mode="constant", value=0)
        mixed_w = self._soft_shift_w(fmap2_p)
        tokens2 = mixed_w.reshape(B, Ch, H * W).transpose(1, 2).contiguous()
        tokens2 = self.fc2(tokens2)
        tokens2 = self.drop(tokens2)
        return tokens2

class DownDeformMLP(nn.Module):
    """UNet Down block variant replacing heavy 3x3 DoubleConv with 1x1 projection + DeformableShiftMLP."""
    def __init__(self, in_channels: int, out_channels: int, mlp_ratio: float = 1.0, shift_size: int = 5, use_standard_shift: bool = False):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        hidden = max(1, int(out_channels * mlp_ratio))
        self.norm = nn.LayerNorm(out_channels)
        self.mlp = DeformableShiftMLP(in_features=out_channels, hidden_features=hidden, out_features=out_channels,
                                      shift_size=shift_size, use_standard_shift=use_standard_shift)

    def forward(self, x):
        x = self.pool(x)
        x = self.proj(x)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        tokens = self.mlp(tokens, H, W)
        y = tokens.transpose(1, 2).view(B, C, H, W).contiguous()
        # residual for stability
        return x + y

# --- EMCAD Components ---

class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights('normal')
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))
        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out)

class CAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=7, activation='relu'):
        super(CAG, self).__init__()
        # assert kernel_size in (3, 5, 7, 9, 11)
        padding = kernel_size // 2
        self.q_proj = nn.Conv2d(F_g, F_int, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(F_l, F_int, kernel_size=1, bias=False)
        # local aggregation on correlation map
        self.local_corr = nn.Conv2d(F_int, F_int, kernel_size=kernel_size, padding=padding, groups=F_int, bias=False)
        self.bn = nn.BatchNorm2d(F_int)
        self.act = act_layer(activation, inplace=True)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.init_weights('normal')
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
    def forward(self, g, x):
        q = self.q_proj(g)
        k = self.k_proj(x)
        # channel-wise correlation
        corr = q * k
        corr = self.local_corr(corr)
        corr = self.bn(corr)
        corr = self.act(corr)
        gate = self.psi(corr)
        return x * gate

class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()
        # assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights('normal')
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(LGAG, self).__init__()
        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)
        self.init_weights('normal')
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.init_weights('normal')
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

class DeformShiftTokBranch(nn.Module):
    def __init__(self, in_channels, activation='relu6', max_shift=2):
        super(DeformShiftTokBranch, self).__init__()
        self.in_channels = in_channels
        self.max_shift = int(max_shift)
        self.offset = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1, bias=True)
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = act_layer(activation, inplace=True)
        self.pw = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        B, C, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=x.device),
            torch.linspace(-1.0, 1.0, W, device=x.device),
            indexing='ij'
        )
        base_grid = torch.stack((xx, yy), dim=-1)
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)
        offs = torch.tanh(self.offset(x)) * self.max_shift
        off_x = offs[:, 0, :, :] / ((W - 1) / 2)
        off_y = offs[:, 1, :, :] / ((H - 1) / 2)
        grid = torch.stack((base_grid[..., 0] + off_x, base_grid[..., 1] + off_y), dim=-1)
        y = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        y = self.dw(y)
        y = self.bn(y)
        y = self.act(y)
        y = self.pw(y)
        return y

class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True, use_deform_tok_branch: bool = False, deform_max_shift: int = 2):
        super(MSDC, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.stride = stride
        self.dw_parallel = dw_parallel
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            if dw_parallel:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, kernel_size=k, stride=stride, padding=padding, groups=in_channels, bias=False),
                        nn.BatchNorm2d(in_channels),
                        act_layer(activation, inplace=True)
                    )
                )
            else:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, kernel_size=k, stride=stride, padding=padding, bias=False),
                        nn.BatchNorm2d(in_channels),
                        act_layer(activation, inplace=True)
                    )
                )
        # 变形 Token 分支：仅在 stride=1 的块中启用，将原 3 分支扩展为 4 分支
        if bool(use_deform_tok_branch) and int(self.stride) == 1:
            self.branches.append(DeformShiftTokBranch(in_channels=in_channels, activation=activation, max_shift=deform_max_shift))
        self.init_weights('normal')
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
    def forward(self, x):
        out = 0
        for b in self.branches:
            out = out + b(x)
        return out

class MSCB(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True, add=True, activation='relu6', use_deform_tok_branch: bool = False, deform_max_shift: int = 2):
        super(MSCB, self).__init__()
        mid_channels = int(in_channels * expansion_factor)
        self.add = add
        self.msdconv = MSDC(in_channels, kernel_sizes, stride, activation=activation, dw_parallel=dw_parallel, use_deform_tok_branch=use_deform_tok_branch, deform_max_shift=deform_max_shift)
        self.pw1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = act_layer(activation, inplace=True)
        self.pw2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weights('normal')
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
    def forward(self, x):
        y = self.msdconv(x)
        y = self.pw1(y)
        y = self.bn1(y)
        y = self.act(y)
        y = self.pw2(y)
        y = self.bn2(y)
        if self.add:
            y = y + x
        return y

def MSCBLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True, add=True, activation='relu6', use_deform_tok_branch: bool = False, deform_max_shift: int = 2):
    convs = []
    mscb = MSCB(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation, use_deform_tok_branch=use_deform_tok_branch, deform_max_shift=deform_max_shift)
    convs.append(mscb)
    if n > 1:
        for i in range(1, n):
            mscb = MSCB(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation, use_deform_tok_branch=use_deform_tok_branch, deform_max_shift=deform_max_shift)
            convs.append(mscb)
    conv = nn.Sequential(*convs)
    return conv

class EMCAD(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64], kernel_sizes=[1, 3, 5], expansion_factor=6, dw_parallel=True, add=True, lgag_ks=3, activation='relu6',
                 ablate_mscam: bool = False, ablate_lgag: bool = False, ablate_eucb: bool = False,
                 use_deform_tok_branch: bool = False, deform_max_shift: int = 2,
                 use_cag: bool = False, cag_ks: int = 7):
        super(EMCAD, self).__init__()
        eucb_ks = 3
        # Ablation flags
        self.ablate_mscam = bool(ablate_mscam)
        self.ablate_lgag = bool(ablate_lgag)
        self.ablate_eucb = bool(ablate_eucb)
        # New switches
        self.use_deform_tok_branch = bool(use_deform_tok_branch)
        self.deform_max_shift = int(deform_max_shift)
        self.use_cag = bool(use_cag)
        self.cag_ks = int(cag_ks)

        # MSCAM-related blocks (propagate deformable token branch)
        self.mscb4 = MSCBLayer(channels[0], channels[0], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                               dw_parallel=dw_parallel, add=add, activation=activation,
                               use_deform_tok_branch=self.use_deform_tok_branch, deform_max_shift=self.deform_max_shift)
        self.mscb3 = MSCBLayer(channels[1], channels[1], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                               dw_parallel=dw_parallel, add=add, activation=activation,
                               use_deform_tok_branch=self.use_deform_tok_branch, deform_max_shift=self.deform_max_shift)
        self.mscb2 = MSCBLayer(channels[2], channels[2], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                               dw_parallel=dw_parallel, add=add, activation=activation,
                               use_deform_tok_branch=self.use_deform_tok_branch, deform_max_shift=self.deform_max_shift)
        self.mscb1 = MSCBLayer(channels[3], channels[3], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                               dw_parallel=dw_parallel, add=add, activation=activation,
                               use_deform_tok_branch=self.use_deform_tok_branch, deform_max_shift=self.deform_max_shift)
        self.cab4 = CAB(channels[0])
        self.cab3 = CAB(channels[1])
        self.cab2 = CAB(channels[2])
        self.cab1 = CAB(channels[3])
        self.sab = SAB()

        # EUCB upsample blocks
        self.eucb3 = EUCB(in_channels=channels[0], out_channels=channels[1], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.eucb2 = EUCB(in_channels=channels[1], out_channels=channels[2], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks // 2)

        # Alternatives for EUCB when ablated (simple bilinear + 1x1 conv)
        self.eucb3_alt_pw = nn.Conv2d(channels[0], channels[1], kernel_size=1)
        self.eucb2_alt_pw = nn.Conv2d(channels[1], channels[2], kernel_size=1)
        self.eucb1_alt_pw = nn.Conv2d(channels[2], channels[3], kernel_size=1)

        # Gating module: Default uses LGAG, switch to CAG when use_cag=True (keeps output channels consistent with tensor size)
        self.lgag3 = LGAG(F_g=channels[1], F_l=channels[1], F_int=max(1, channels[1] // 2), kernel_size=lgag_ks, groups=1)
        self.lgag2 = LGAG(F_g=channels[2], F_l=channels[2], F_int=max(1, channels[2] // 2), kernel_size=lgag_ks, groups=1)
        self.lgag1 = LGAG(F_g=channels[3], F_l=channels[3], F_int=max(1, int(channels[3] / 2)), kernel_size=lgag_ks, groups=1)
        if self.use_cag:
            self.cag3 = CAG(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2, kernel_size=self.cag_ks, activation='relu')
            self.cag2 = CAG(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2, kernel_size=self.cag_ks, activation='relu')
            self.cag1 = CAG(F_g=channels[3], F_l=channels[3], F_int=int(channels[3] / 2), kernel_size=self.cag_ks, activation='relu')

    def forward(self, x, skips):
        # Stage 4: MSCAM4
        if self.ablate_mscam:
            d4 = self.mscb4(x)
        else:
            d4 = self.cab4(x) * x
            d4 = self.sab(d4) * d4
            d4 = self.mscb4(d4)

        # Stage 3: EUCB3 and Gate3
        if self.ablate_eucb:
            d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
            d3 = self.eucb3_alt_pw(d3)
        else:
            d3 = self.eucb3(d4)

        # Stage 3: Gating after EUCB3, can choose LGAG or CAG; skip gating and add directly when ablate_lgag=True
        if self.ablate_lgag:
            d3 = d3 + skips[0]
        else:
            gate3 = self.cag3 if self.use_cag else self.lgag3
            x3 = gate3(g=d3, x=skips[0])
            d3 = d3 + x3

        if self.ablate_mscam:
            d3 = self.mscb3(d3)
        else:
            d3 = self.cab3(d3) * d3
            d3 = self.sab(d3) * d3
            d3 = self.mscb3(d3)

        # Stage 2: EUCB2 and Gate2
        if self.ablate_eucb:
            d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
            d2 = self.eucb2_alt_pw(d2)
        else:
            d2 = self.eucb2(d3)

        # Stage 2: Gating after EUCB2, same as above
        if self.ablate_lgag:
            d2 = d2 + skips[1]
        else:
            gate2 = self.cag2 if self.use_cag else self.lgag2
            x2 = gate2(g=d2, x=skips[1])
            d2 = d2 + x2

        if self.ablate_mscam:
            d2 = self.mscb2(d2)
        else:
            d2 = self.cab2(d2) * d2
            d2 = self.sab(d2) * d2
            d2 = self.mscb2(d2)

        # Stage 1: EUCB1 and Gate1
        if self.ablate_eucb:
            d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
            d1 = self.eucb1_alt_pw(d1)
        else:
            d1 = self.eucb1(d2)

        # Stage 1: Gating after EUCB1, same as above
        if self.ablate_lgag:
            d1 = d1 + skips[2]
        else:
            gate1 = self.cag1 if self.use_cag else self.lgag1
            x1 = gate1(g=d1, x=skips[2])
            d1 = d1 + x1

        if self.ablate_mscam:
            d1 = self.mscb1(d1)
        else:
            d1 = self.cab1(d1) * d1
            d1 = self.sab(d1) * d1
            d1 = self.mscb1(d1)

        return [d4, d3, d2, d1]

class DeShiftNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, channels=(32, 64, 128, 160, 256), use_deform_shift_mlp=True, shift_size=7, use_cag=False, cag_ks=7, use_deform_tok_branch=False, deform_max_shift=2, ablate_mscam=False, ablate_lgag=False, ablate_eucb=False, deep_supervision=True):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.deep_supervision = deep_supervision
        self.use_deform_shift_mlp = bool(use_deform_shift_mlp)
        # Adapt single channel input to 3 channels to match pretrained weights/decoder channels
        self.conv_in = nn.Sequential(
            nn.Conv2d(1, 3, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        # Encoder: First three layers use UNext style SingleConv to significantly reduce FLOPs, last two layers can switch to DeformableShiftMLP version
        self.enc1 = SingleConv(input_channels, c1)
        self.enc2 = DownSingle(c1, c2)
        self.enc3 = DownSingle(c2, c3)
        if self.use_deform_shift_mlp:
            self.enc4 = DownDeformMLP(c3, c4, mlp_ratio=1.0, shift_size=shift_size, use_standard_shift=False)
            self.enc5 = DownDeformMLP(c4, c5, mlp_ratio=1.0, shift_size=shift_size, use_standard_shift=False)
        else:
            self.enc4 = Down(c3, c4)
            self.enc5 = Down(c4, c5)
        # EMCAD Decoder: Can switch LGAG->CAG, and can enable deform token branch in MSDC
        self.decoder = EMCAD(
            channels=[c5, c4, c3, c2], kernel_sizes=[1, 3, 5], expansion_factor=2,
            dw_parallel=True, add=True, lgag_ks=3, activation='relu',
            ablate_mscam=ablate_mscam, ablate_lgag=ablate_lgag, ablate_eucb=ablate_eucb,
            use_deform_tok_branch=use_deform_tok_branch, deform_max_shift=deform_max_shift,
            use_cag=use_cag, cag_ks=cag_ks
        )
        # Prediction heads (deep supervision and final output)
        self.out_head4 = nn.Conv2d(c5, num_classes, 1)
        self.out_head3 = nn.Conv2d(c4, num_classes, 1)
        self.out_head2 = nn.Conv2d(c3, num_classes, 1)
        self.out_head1 = nn.Conv2d(c2, num_classes, 1)
    def forward(self, x):
        if x.size(1) == 1:
            x = self.conv_in(x)
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x4 = self.enc4(s3)
        x5 = self.enc5(x4)
        d4, d3, d2, d1 = self.decoder(x5, [x4, s3, s2])
        p4 = self.out_head4(d4)
        p3 = self.out_head3(d3)
        p2 = self.out_head2(d2)
        p1 = self.out_head1(d1)
        H0, W0 = x.size(2), x.size(3)
        p4 = F.interpolate(p4, size=(H0, W0), mode='bilinear')
        p3 = F.interpolate(p3, size=(H0, W0), mode='bilinear')
        p2 = F.interpolate(p2, size=(H0, W0), mode='bilinear')
        p1 = F.interpolate(p1, size=(H0, W0), mode='bilinear')
        if self.deep_supervision:
            return [p4, p3, p2, p1]
        return p1
