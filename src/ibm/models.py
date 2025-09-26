import torch.nn as nn


class AddConstant(nn.Module):
    def __init__(self, c: float = 1.0):
        """Example module that adds a constant value to the input.

        Args:
            c (float): Constant to be added. Defaults to 1.0.
        """
        super().__init__()
        self.c = c

    def forward(self, x):
        return x + self.c
