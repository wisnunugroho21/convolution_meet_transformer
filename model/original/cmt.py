import torch.nn as nn
from torch import Tensor

from model.original.attention import MultiHeadAttention

class LocalPerceptionUnit(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size = 3, stride = 1, padding = 1, groups = dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv(x)
        return x + x1

class LightweightMultiHeadAttention(nn.Module):
    def __init__(self, heads: int, dim: int, k: int) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size = k, stride = k, groups = dim)
        )

        self.att = MultiHeadAttention(heads = heads, d_model = dim)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        key_value = self.conv(x)

        query       = x.flatten(2).permute(0, 2, 1)
        key_value   = key_value.flatten(2).permute(0, 2, 1)

        out = self.att(query, key_value, key_value)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        return out

class InvertedResidualFFN(nn.Module):
    def __init__(self, dim: int, b: int) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim * b, kernel_size = 1, stride = 1),
            nn.GELU(),
            nn.BatchNorm2d(dim * b)            
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim * b, dim * b, kernel_size = 3, stride = 1, padding = 1, groups = dim * b)
        )

        self.conv3 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm2d(dim * b),
            nn.Conv2d(dim * b, dim, kernel_size = 1, stride = 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x21 = x2 + x1        
        return self.conv3(x21)

class CMTBlock(nn.Module):
    def __init__(self, dim: int, head: int, k: int, b: int) -> None:
        super().__init__()

        self.lpu    = LocalPerceptionUnit(dim)
        self.lmhsa  = LightweightMultiHeadAttention(head, dim, k)
        self.irffn  = InvertedResidualFFN(dim, b)

        self.ln1 = nn.GroupNorm(1, dim)
        self.ln2 = nn.GroupNorm(1, dim)

    def forward(self, x: Tensor) -> Tensor:
        x1  = self.lpu(x)

        x21 = self.lmhsa(self.ln1(x1))
        x2  = x21 + x1

        x31 = self.irffn(self.ln2(x2))
        x3  = x31 + x2

        return x3