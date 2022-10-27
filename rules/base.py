from abc import ABC, abstractmethod
from typing_extensions import Self

import torch
from torch import nn, Tensor


class Rule(nn.Module, ABC):
    def __init__(
        self,
        num_args: int,
        var_shape: tuple,
        embedding_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.num_args = num_args
        self.var_shape = var_shape

        self.embedding_size = embedding_size
        self.embedding = nn.Parameter(torch.randn(embedding_size))

        self.device = device
        _ = self.to(device)

    def forward(self, vars: Tensor) -> Tensor:
        assert vars.shape[1:] == (self.num_args, *self.var_shape)
        output_var = self.apply(vars)
        assert output_var.shape[1:] == self.var_shape
        return output_var

    @abstractmethod
    def apply(self, vars: Tensor) -> Tensor:
        pass

    def to(self, device: torch.device, *args, **kwargs) -> Self:
        self.device = device
        return super().to(device=device, *args, **kwargs)
