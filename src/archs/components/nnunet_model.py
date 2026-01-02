"""nnUNet-style 2D UNet backbone.

This is a lightweight adaptation that mirrors the core nnUNet design
(LeakyReLU + InstanceNorm, two 3x3 convs per stage, encoder/decoder with
skip connections). It is registered as a supervised model so it can be
used with the existing data modules and training pipeline.
"""

from typing import List

import torch
import torch.nn as nn

from src.registry.models import register_model


def conv3x3(in_channels: int, out_channels: int, bias: bool = True) -> nn.Conv2d:
    """3x3 conv with padding."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)


class DoubleConv(nn.Module):
    """Two consecutive conv-norm-activation blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float = 1e-5,
        leaky_relu_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.InstanceNorm2d(out_channels, eps=norm_eps, affine=True),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
            conv3x3(out_channels, out_channels),
            nn.InstanceNorm2d(out_channels, eps=norm_eps, affine=True),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downsampling with max-pool followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsampling + skip connection + DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if needed to handle odd input sizes.
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@register_model(
    name="nnunet",
    task="supervised",
    params=24_000_000,
    speed="medium",
    description="nnUNet-style 2D UNet (LeakyReLU + InstanceNorm) adapted to project data format",
    paper_url="https://github.com/MIC-DKFZ/nnUNet",
    default_lr=2e-4,
    default_epochs=500,
)
class NnUNet2D(nn.Module):
    """Simplified nnUNet-style 2D UNet for segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        num_stages: int = 5,
    ) -> None:
        super().__init__()

        # Encoder
        features: List[int] = [
            min(base_channels * (2**i), base_channels * 16) for i in range(num_stages)
        ]
        self.inc = DoubleConv(in_channels, features[0])
        self.down_blocks = nn.ModuleList(
            Down(features[i], features[i + 1]) for i in range(num_stages - 1)
        )

        # Decoder
        self.up_blocks = nn.ModuleList(
            Up(features[i + 1], features[i]) for i in reversed(range(num_stages - 1))
        )
        self.outc = OutConv(features[0], num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        # Kaiming initialization similar to nnUNet defaults.
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, a=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.inc(x)
        skips = [x0]
        x_enc = x0
        for down in self.down_blocks:
            x_enc = down(x_enc)
            skips.append(x_enc)

        x_dec = x_enc
        for skip, up in zip(reversed(skips[:-1]), self.up_blocks):
            x_dec = up(x_dec, skip)

        return self.outc(x_dec)
