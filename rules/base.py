from abc import ABC, abstractmethod

import torch
from torch import nn, FloatTensor


class Rule(nn.Module, ABC):
    def __init__(self, num_args: int, var_shape: tuple, embedding_size: int) -> None:
        super().__init__()
        self.num_args = num_args
        self.var_shape = var_shape
        self.embedding_size = embedding_size
        self.embedding = nn.Parameter(torch.randn(embedding_size))

    def forward(self, vars: FloatTensor) -> FloatTensor:
        assert vars.shape[1:] == (self.num_args, *self.var_shape)
        return self.rule(vars)

    @abstractmethod
    def apply(self, vars: FloatTensor) -> FloatTensor:
        pass
