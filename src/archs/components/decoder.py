"""Decoder components for OCT segmentation."""

import torch
import torch.nn as nn

from .attention import CBAM


class OCTDecoder(nn.Module):
    """Decoder for OCT segmentation using upsampling and skip connections."""

    def __init__(self, in_channels: int = 512, num_classes: int = 2):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, 256, 2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.attention1 = CBAM(256)
        self.attention2 = CBAM(128)
        self.attention3 = CBAM(64)

        self.final_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x: torch.Tensor, skip_connections: tuple) -> torch.Tensor:
        c1, c2, c3 = skip_connections

        # Decoder path
        x = self.up1(x)
        x = torch.cat([x, c3], dim=1)
        x = self.conv1(x)
        x = self.attention1(x)

        x = self.up2(x)
        x = torch.cat([x, c2], dim=1)
        x = self.conv2(x)
        x = self.attention2(x)

        x = self.up3(x)
        x = torch.cat([x, c1], dim=1)
        x = self.conv3(x)
        x = self.attention3(x)

        x = self.final_conv(x)
        return x


class TransformerDecoder(nn.Module):
    """Transformer-based decoder for OCT segmentation."""

    def __init__(self, embed_dim: int = 768, num_classes: int = 2,
                 patch_size: int = 16, img_size: int = 224):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2

        self.decoder_embed = nn.Linear(embed_dim, embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.decoder_blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(embed_dim, nhead=8, dim_feedforward=embed_dim*4)
            for _ in range(4)
        ])

        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size**2 * num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Remove cls token
        x = x[:, 1:, :]  # B, num_patches, embed_dim

        # Decoder embedding
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed

        # Transformer decoder blocks
        for block in self.decoder_blocks:
            x = block(x, x)

        x = self.decoder_norm(x)

        # Prediction head
        x = self.decoder_pred(x)

        # Reshape to image
        B, num_patches, patch_dim = x.shape
        x = x.reshape(B, self.num_patches, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.reshape(B, -1, self.img_size, self.img_size)

        return x
