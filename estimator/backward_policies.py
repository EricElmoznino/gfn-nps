from abc import ABC, abstractmethod

import torch
from torch import nn
from torch import FloatTensor

from environment.states import State
from environment.actions import BackwardAction


class BackwardPolicy(nn.Module, ABC):
    @abstractmethod
    def __init__(self, example_state: State, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def forward(self, state: State) -> tuple[BackwardAction, FloatTensor]:
        pass
