from abc import ABC, abstractmethod

import torch
from torch import nn
from torch import FloatTensor

from gfn_parameterization.states import State


class ForwardPolicy(nn.Module, ABC):
    @abstractmethod
    def __init__(self, example_state: State, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def forward(self, state: State) -> tuple[State, FloatTensor]:
        pass
