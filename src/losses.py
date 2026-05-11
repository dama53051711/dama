import torch
from torch import nn


class HingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        target = 2 * target - 1  # From {0, 1} to {-1, +1}
        return torch.clamp_min(1 - target * output, 0).mean()


class SmoothedHingeLoss(nn.Module):
    """
    https://home.ttic.edu/~nati/Publications/RennieSrebroIJCAI05.pdf
    """

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        z = (2 * target - 1) * output
        v1 = (0.5 - z)[z <= 0].sum()
        v2 = ((1 - z) ** 2 / 2)[(z > 0) & (z < 1)].sum()
        return (v1 + v2) / len(output)


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = nn.BCELoss()

    def forward(self, output, target):
        return self.loss_func(torch.sigmoid(output), target.float())
