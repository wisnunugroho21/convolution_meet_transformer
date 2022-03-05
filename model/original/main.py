import math
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.original.cmt import CMTBlock

class MainModel(nn.Module):
    def __init__(self, num_class) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.GELU(),
            nn.BatchNorm2d(32)
        )

        self.stage1 = nn.Sequential(*[CMTBlock(64, head = 1, k = 8, b = 4) for _ in range(3)])
        self.stage2 = nn.Sequential(*[CMTBlock(128, head = 2, k = 4, b = 4) for _ in range(3)])
        self.stage3 = nn.Sequential(*[CMTBlock(256, head = 4, k = 2, b = 6) for _ in range(3)])
        self.stage4 = nn.Sequential(*[CMTBlock(512, head = 8, k = 1, b = 4) for _ in range(3)])

        self.downsampler1 = nn.Conv2d(32, 64, kernel_size = 2, stride = 2)
        self.downsampler2 = nn.Conv2d(64, 128, kernel_size = 2, stride = 2)
        self.downsampler3 = nn.Conv2d(128, 256, kernel_size = 2, stride = 2)
        self.downsampler4 = nn.Conv2d(256, 512, kernel_size = 2, stride = 2)

        self.out = nn.Sequential(
            nn.AvgPool2d(kernel_size = 1),
            nn.Flatten(),
            nn.Linear(512, 1280),
            nn.GELU(),
            nn.Linear(1280, 1000),
            nn.GELU(),
            nn.Linear(1000, num_class)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        x = self.downsampler1(x)
        x = self.stage1(x)

        x = self.downsampler2(x)
        x = self.stage2(x)

        x = self.downsampler3(x)
        x = self.stage3(x)

        x = self.downsampler4(x)
        x = self.stage4(x)

        x = self.out(x)

        return x