from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn
from torch import FloatTensor

from environment.states import State
from environment.actions import BackwardAction


class BackwardPolicy(nn.Module, ABC):
    def __init__(self, uses_representation: bool = False) -> None:
        super().__init__()
        self.uses_representation = uses_representation

    @abstractmethod
    def forward(
        self, state: State, representation: Any | None = None
    ) -> tuple[BackwardAction, FloatTensor]:
        if self.uses_representation:
            assert representation is not None, "Policy requires a state representation"
        else:
            assert representation is None, "Policy does not use a state representation"
