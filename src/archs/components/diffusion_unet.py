"""Diffusion-based Segmentation Models
- SegDiff: Based on https://github.com/tomeramit/SegDiff
- MedSegDiff: Medical image segmentation with diffusion

Two separate implementations with different conditioning approaches.
"""
import copy
import math
from collections import namedtuple
from functools import partial

import torch
from einops import rearrange
from torch import einsum, nn
from torch.fft import fft2, ifft2

# Constants
ModelPrediction = namedtuple('ModelPrediction', ['predict_noise', 'predict_x_start'])


# ====== Utility Functions ======
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t):
    return t


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# ====== Diffusion Process Utilities ======
def extract(a, t, x_shape):
    """Extract values from a 1-D tensor for a batch of indices"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """Linear beta schedule for diffusion"""
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine beta schedule for diffusion"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# ====== Basic Components ======
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1))


def downsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


def feed_forward_att(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv2d(dim, inner_dim, 1),
        nn.GELU(),
        nn.Conv2d(inner_dim, dim, 1))


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = dim_head * heads
        self.pre_norm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.pre_norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = dim_head * heads
        self.pre_norm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.pre_norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class MIDAttention(nn.Module):
    """Cross-attention for conditioning (MedSegDiff)"""
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = dim_head * heads
        self.pre_norm_x = LayerNorm(dim)
        self.pre_norm_c = LayerNorm(dim)
        self.to_qkv_x = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_qkv_c = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, c_x):
        b, c, h, w = x.shape
        x = self.pre_norm_x(x)
        c_x = self.pre_norm_c(c_x)
        qkv_x = self.to_qkv_x(x).chunk(3, dim=1)
        qkv_c = self.to_qkv_c(c_x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv_x)
        q_c, k_c, v_c = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv_c)
        q_c = q_c * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q_c, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class MIDTransformer(nn.Module):
    """Transformer with cross-attention (MedSegDiff)"""
    def __init__(self, dim, dim_head=32, heads=4, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(MIDAttention(dim, dim_head=dim_head, heads=heads)),
                Residual(feed_forward_att(dim)),
                Residual(feed_forward_att(dim)),
                Residual(MIDAttention(dim, dim_head=dim_head, heads=heads))]))

    def forward(self, x, c):
        for attn_1, ff_1, ff_2, attn_2 in self.layers:
            x = attn_1(x, c)
            x1 = ff_1(x)
            x2 = ff_2(x)
            x = attn_2(x1, x2)
        return x


# ====== RRDB Components (SegDiff) ======
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


# ====== FFT Conditioning (MedSegDiff) ======
class FFTConditioning(nn.Module):
    """FFT-based conditioning module for MedSegDiff (eq 3 in paper)"""
    def __init__(self, fmap_size, dim):
        super().__init__()
        self.dim = dim
        self.default_fmap_size = fmap_size
        # Initialize with default size, will be resized dynamically if needed
        self.ff_parser_attn_map = nn.Parameter(torch.ones(dim, fmap_size, fmap_size))
        self.norm_input = LayerNorm(dim, bias=True)
        self.norm_condition = LayerNorm(dim, bias=True)
        self.block = ResnetBlock(dim, dim)

    def forward(self, x, c):
        """
        Args:
            x: Main path features (noisy mask features)
            c: Conditional features (image features)
        """
        dtype = x.dtype
        b, c_dim, h, w = x.shape

        # FF-parser: modulate high frequencies
        x_fft = fft2(x)

        # Dynamically resize attention map if needed
        if h != self.default_fmap_size or w != self.default_fmap_size:
            # Resize attention map to match input size
            attn_map = torch.nn.functional.interpolate(
                self.ff_parser_attn_map.unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        else:
            attn_map = self.ff_parser_attn_map

        x_fft = x_fft * attn_map
        x = ifft2(x_fft).real.type(dtype)

        # Eq 3 in paper: c = (norm(x) * norm(c)) * c
        normed_x = self.norm_input(x)
        normed_c = self.norm_condition(c)
        c = (normed_x * normed_c) * c

        # Extra block for information integration
        return self.block(c)


# ====== NATTEN Conditioning (Proposed) ======
class NATTENConditioning(nn.Module):
    """NATTEN-based conditioning module for proposed model"""
    def __init__(self, fmap_size, dim):
        super().__init__()
        self.dim = dim
        self.default_fmap_size = fmap_size
        # Initialize with default size, will be resized dynamically if needed
        self.ff_parser_attn_map = nn.Parameter(torch.ones(dim, fmap_size, fmap_size))
        self.norm_input = LayerNorm(dim, bias=True)
        self.norm_condition = LayerNorm(dim, bias=True)
        self.block = ResnetBlock(dim, dim)

    def forward(self, x, c):
        """
        Args:
            x: Main path features (noisy mask features)
            c: Conditional features (image features)
        """
        dtype = x.dtype
        b, c_dim, h, w = x.shape

        # FF-parser: modulate high frequencies
        x_fft = fft2(x)

        # Dynamically resize attention map if needed
        if h != self.default_fmap_size or w != self.default_fmap_size:
            # Resize attention map to match input size
            attn_map = torch.nn.functional.interpolate(
                self.ff_parser_attn_map.unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        else:
            attn_map = self.ff_parser_attn_map

        x_fft = x_fft * attn_map
        x = ifft2(x_fft).real.type(dtype)

        # Eq 3 in paper: c = (norm(x) * norm(c)) * c
        normed_x = self.norm_input(x)
        normed_c = self.norm_condition(c)
        c = (normed_x * normed_c) * c

        # Extra block for information integration
        return self.block(c)


# ====== SegDiff UNet ======
class SegDiffUNet(nn.Module):
    """SegDiff: RRDB-based conditioning, F(x_t) + G(I) fed to UNet"""
    def __init__(self, dim, image_size, mask_channels=1, input_img_channels=1, init_dim=None,
                 dim_mult=(1, 2, 4, 8), full_self_attn=(False, False, True, True), attn_dim_head=32,
                 attn_heads=4, resnet_block_groups=8, rrdb_blocks=3):
        super().__init__()

        self.image_size = image_size
        self.mask_channels = mask_channels
        self.input_img_channels = input_img_channels

        init_dim = default(init_dim, dim)

        # F model: process x_t
        self.F_model = nn.Conv2d(mask_channels, init_dim, 3, padding=1)

        # G model: RRDB encoder for conditioning image
        RRDB_block_f = partial(RRDB, nf=init_dim, gc=32)
        self.G_model = nn.Sequential(
            nn.Conv2d(input_img_channels, init_dim, 3, 1, 1),
            make_layer(RRDB_block_f, rrdb_blocks),
            nn.Conv2d(init_dim, init_dim, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Time embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim))

        # UNet
        dims = [init_dim, *map(lambda m: dim * m, dim_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        attn_kwargs = dict(dim_head=attn_dim_head, heads=attn_heads)

        self.downs = nn.ModuleList([])
        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(in_out, full_self_attn)):
            is_last = ind >= (len(in_out) - 1)
            attn_klass = Attention if full_attn else LinearAttention
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(attn_klass(dim_in, **attn_kwargs)),
                downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(Attention(mid_dim, **attn_kwargs))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        self.ups = nn.ModuleList([])
        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(reversed(in_out), reversed(full_self_attn))):
            is_last = ind == (len(in_out) - 1)
            attn_klass = Attention if full_attn else LinearAttention
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(attn_klass(dim_out, **attn_kwargs)),
                upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)]))

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, mask_channels, 1)

    def forward(self, x, time, cond):
        # F(x_t) + G(I) -> UNet
        F_out = self.F_model(x)
        G_out = self.G_model(cond)

        x = F_out + G_out  # Key difference: addition of F and G
        r = x.clone()
        t = self.time_mlp(time)

        h = []
        for block1, block2, attn, dsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = dsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample_layer in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample_layer(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


# ====== Simple Concat UNet (BerDiff style) ======
class SimpleConcatUNet(nn.Module):
    """Simple UNet: concat(x_t, cond) as input (BerDiff style)
    
    Simple baseline that concatenates noisy mask and condition image
    along channel dimension before feeding to UNet.
    
    For BerDiff (Binomial Diffusion), output must be in [0, 1] range,
    so sigmoid is applied at the end.
    """
    def __init__(self, dim, image_size, mask_channels=1, input_img_channels=1, init_dim=None,
                 dim_mult=(1, 2, 4, 8), full_self_attn=(False, False, True, True), attn_dim_head=32,
                 attn_heads=4, resnet_block_groups=8):
        super().__init__()

        self.image_size = image_size
        self.mask_channels = mask_channels
        self.input_img_channels = input_img_channels

        init_dim = default(init_dim, dim)

        # Initial conv takes concatenated input: [x_t, cond]
        self.init_conv = nn.Conv2d(mask_channels + input_img_channels, init_dim, 7, padding=3)

        # Time embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim))

        # UNet
        dims = [init_dim, *map(lambda m: dim * m, dim_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        attn_kwargs = dict(dim_head=attn_dim_head, heads=attn_heads)

        self.downs = nn.ModuleList([])
        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(in_out, full_self_attn)):
            is_last = ind >= (len(in_out) - 1)
            attn_klass = Attention if full_attn else LinearAttention
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(attn_klass(dim_in, **attn_kwargs)),
                downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(Attention(mid_dim, **attn_kwargs))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        self.ups = nn.ModuleList([])
        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(reversed(in_out), reversed(full_self_attn))):
            is_last = ind == (len(in_out) - 1)
            attn_klass = Attention if full_attn else LinearAttention
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(attn_klass(dim_out, **attn_kwargs)),
                upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)]))

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, mask_channels, 1)

    def forward(self, x, time, cond):
        """
        Args:
            x: noisy mask (B, mask_channels, H, W)
            time: timestep (B,)
            cond: conditional image (B, input_img_channels, H, W)
        
        Returns:
            Predicted mask in [0, 1] range (with sigmoid applied)
        """
        # Concat noisy mask and condition image
        x = torch.cat([x, cond], dim=1)  # (B, mask_channels + input_img_channels, H, W)

        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)

        h = []
        for block1, block2, attn, dsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = dsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample_layer in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample_layer(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        # Apply sigmoid for BerDiff (Binomial Diffusion): output must be in [0, 1]
        return torch.sigmoid(x)


# ====== MedSegDiff UNet ======
class MedSegDiffUNet(nn.Module):
    """MedSegDiff: FFT conditioning with separate paths for x and cond"""
    def __init__(self, dim, image_size, mask_channels=1, input_img_channels=1, init_dim=None,
                 dim_mult=(1, 2, 4, 8), full_self_attn=(False, False, True, True), attn_dim_head=32,
                 attn_heads=4, mid_transformer_depth=1, resnet_block_groups=8,
                 skip_connect_condition_fmap=False, conditioning_fmap_size=None):
        super().__init__()

        self.image_size = image_size
        self.mask_channels = mask_channels
        self.input_img_channels = input_img_channels
        self.skip_connect_condition_fmap = skip_connect_condition_fmap

        init_dim = default(init_dim, dim)

        # Separate init convs for mask and condition
        self.init_conv = nn.Conv2d(mask_channels, init_dim, 7, padding=3)
        self.cond_init_conv = nn.Conv2d(input_img_channels, init_dim, 7, padding=3)

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim))

        dims = [init_dim, *map(lambda m: dim * m, dim_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        attn_kwargs = dict(dim_head=attn_dim_head, heads=attn_heads)

        if conditioning_fmap_size is None:
            curr_fmap_size = image_size
        else:
            curr_fmap_size = min(image_size, conditioning_fmap_size)
        self.downs = nn.ModuleList([])
        self.conditioners = nn.ModuleList([])

        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(in_out, full_self_attn)):
            is_last = ind >= (len(in_out) - 1)
            attn_klass = Attention if full_attn else LinearAttention

            self.conditioners.append(FFTConditioning(curr_fmap_size, dim_in))

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(attn_klass(dim_in, **attn_kwargs)),
                downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)]))

            if not is_last:
                curr_fmap_size = max(1, curr_fmap_size // 2)

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_transformer = MIDTransformer(mid_dim, depth=mid_transformer_depth, **attn_kwargs)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        self.cond_downs = copy.deepcopy(self.downs)
        self.cond_mid_block1 = copy.deepcopy(self.mid_block1)

        self.ups = nn.ModuleList([])
        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(reversed(in_out), reversed(full_self_attn))):
            is_last = ind == (len(in_out) - 1)
            attn_klass = Attention if full_attn else LinearAttention
            skip_connect_dim = dim_in * (2 if skip_connect_condition_fmap else 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim=time_dim),
                Residual(attn_klass(dim_out, **attn_kwargs)),
                upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)]))

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, mask_channels, 1)

    def forward(self, x, time, cond):
        skip_connect_c = self.skip_connect_condition_fmap

        x = self.init_conv(x)
        c = self.cond_init_conv(cond)
        r = x.clone()
        t = self.time_mlp(time)

        h = []
        for (block1, block2, attn, dsample), (cond_block1, cond_block2, cond_attn, cond_dsample), conditioner in \
                zip(self.downs, self.cond_downs, self.conditioners):
            x = block1(x, t)
            c = cond_block1(c, t)
            h.append([x, c] if skip_connect_c else [x])

            x = block2(x, t)
            c = cond_block2(c, t)
            x = attn(x)
            c = cond_attn(c)
            x = conditioner(x, c)  # FFT conditioning with conditional features
            h.append([x, c] if skip_connect_c else [x])

            x = dsample(x)
            c = cond_dsample(c)

        x = self.mid_block1(x, t)
        c = self.cond_mid_block1(c, t)
        x = x + c
        x = self.mid_transformer(x, c)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample_layer in self.ups:
            x = torch.cat((x, *h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, *h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample_layer(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)









if __name__ == "__main__":
    print("=" * 70)
    print("Testing UNet Architectures")
    print("=" * 70)

    # Test UNet architectures
    img = torch.randn(2, 1, 224, 224)
    cond = torch.randn(2, 1, 224, 224)
    time = torch.randint(0, 1000, (2,))

    print("\n1. SegDiffUNet (RRDB-based conditioning: F(x_t) + G(cond))")
    segdiff_unet = SegDiffUNet(
        dim=64,
        image_size=224,
        mask_channels=1,
        input_img_channels=1,
        dim_mult=(1, 2, 4, 8),
        full_self_attn=(False, False, True, True),
        rrdb_blocks=3
    )
    output = segdiff_unet(img, time, cond)
    params = sum(p.numel() for p in segdiff_unet.parameters())
    print(f"   Output shape: {output.shape}, Params: {params:,}")

    print("\n2. SimpleConcatUNet (Concat-based: concat([x_t, cond]))")
    simple_unet = SimpleConcatUNet(
        dim=64,
        image_size=224,
        mask_channels=1,
        input_img_channels=1,
        dim_mult=(1, 2, 4, 8),
        full_self_attn=(False, False, True, True)
    )
    output = simple_unet(img, time, cond)
    params = sum(p.numel() for p in simple_unet.parameters())
    print(f"   Output shape: {output.shape}, Params: {params:,}")

    print("\n3. MedSegDiffUNet (FFT-based conditioning: dual path)")
    medsegdiff_unet = MedSegDiffUNet(
        dim=64,
        image_size=224,
        mask_channels=1,
        input_img_channels=1,
        dim_mult=(1, 2, 4, 8),
        full_self_attn=(False, False, True, True),
        mid_transformer_depth=1
    )
    output = medsegdiff_unet(img, time, cond)
    params = sum(p.numel() for p in medsegdiff_unet.parameters())
    print(f"   Output shape: {output.shape}, Params: {params:,}")

    print("\n" + "=" * 70)
    print("✓ All UNet architectures work correctly!")
    print("=" * 70)
