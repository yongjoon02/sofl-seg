"""CSNet: Channel and Spatial attention Network for OCT segmentation.

Official implementation from: https://github.com/iMED-Lab/CS-Net
Paper: "CS-Net: Channel and Spatial Attention Network for Curvilinear Structure Segmentation"
"""

import torch
import torch.nn as nn


def initialize_weights(*models):
    """Initialize model weights using Kaiming initialization."""
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResEncoder(nn.Module):
    """Residual Encoder Block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = out + residual
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    """Decoder Block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SpatialAttentionBlock(nn.Module):
    """Spatial Attention Block using asymmetric convolutions."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor (B, C, H, W)
        Returns:
            affinity value + x
        """
        B, C, H, W = x.size()

        # Query: [B, C//8, H, W] -> [B, H*W, C//8]
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)

        # Key: [B, C//8, H, W] -> [B, C//8, H*W]
        proj_key = self.key(x).view(B, -1, W * H)

        # Affinity: [B, H*W, H*W]
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)

        # Value: [B, C, H, W] -> [B, C, H*W]
        proj_value = self.value(x).view(B, -1, H * W)

        # Weighted sum: [B, C, H*W]
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)

        # Residual connection with learnable weight
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock(nn.Module):
    """Channel Attention Block."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor (B, C, H, W)
        Returns:
            affinity value + x
        """
        B, C, H, W = x.size()

        # Query: [B, C, H*W]
        proj_query = x.view(B, C, -1)

        # Key: [B, H*W, C]
        proj_key = x.view(B, C, -1).permute(0, 2, 1)

        # Affinity: [B, C, C]
        affinity = torch.matmul(proj_query, proj_key)

        # Max pooling trick for better performance
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)

        # Value: [B, C, H*W]
        proj_value = x.view(B, C, -1)

        # Weighted sum: [B, C, H*W]
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)

        # Residual connection with learnable weight
        out = self.gamma * weights + x
        return out


class AffinityAttention(nn.Module):
    """Affinity Attention Module combining Spatial and Channel Attention."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.sab = SpatialAttentionBlock(in_channels)
        self.cab = ChannelAttentionBlock(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor
        Returns:
            sab + cab (combined spatial and channel attention)
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = sab + cab
        return out


class CSNet(nn.Module):
    """CS-Net: Channel and Spatial Attention Network.
    
    Official implementation for curvilinear structure segmentation (vessel, road, etc.)
    
    Args:
        in_channels: number of input channels (default: 1 for grayscale)
        num_classes: number of output classes (default: 2 for binary segmentation)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()

        # Encoder with residual blocks
        self.enc_input = ResEncoder(in_channels, 32)
        self.encoder1 = ResEncoder(32, 64)
        self.encoder2 = ResEncoder(64, 128)
        self.encoder3 = ResEncoder(128, 256)
        self.encoder4 = ResEncoder(256, 512)

        # Downsampling
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        # Affinity Attention (bottleneck)
        self.affinity_attention = AffinityAttention(512)

        # Decoder
        self.decoder4 = Decoder(512, 256)
        self.decoder3 = Decoder(256, 128)
        self.decoder2 = Decoder(128, 64)
        self.decoder1 = Decoder(64, 32)

        # Upsampling (transposed convolution)
        self.deconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        # Final output layer
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

        # Initialize weights
        initialize_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        enc_input = self.enc_input(x)
        down1 = self.downsample(enc_input)

        enc1 = self.encoder1(down1)
        down2 = self.downsample(enc1)

        enc2 = self.encoder2(down2)
        down3 = self.downsample(enc2)

        enc3 = self.encoder3(down3)
        down4 = self.downsample(enc3)

        # Bottleneck with attention
        input_feature = self.encoder4(down4)
        attention = self.affinity_attention(input_feature)
        attention_fuse = input_feature + attention

        # Decoder path with skip connections
        up4 = self.deconv4(attention_fuse)
        up4 = torch.cat((enc3, up4), dim=1)
        dec4 = self.decoder4(up4)

        up3 = self.deconv3(dec4)
        up3 = torch.cat((enc2, up3), dim=1)
        dec3 = self.decoder3(up3)

        up2 = self.deconv2(dec3)
        up2 = torch.cat((enc1, up2), dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.deconv1(dec2)
        up1 = torch.cat((enc_input, up1), dim=1)
        dec1 = self.decoder1(up1)

        # Final output (no sigmoid - will be applied in loss function)
        final = self.final(dec1)

        return final


if __name__ == "__main__":
    # Test the model
    model = CSNet(in_channels=1, num_classes=2)
    x = torch.randn(2, 1, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
