from abc import ABC, abstractmethod
from typing import Any

from torch import nn
from torch.nn import functional as F
from torch import FloatTensor, LongTensor, BoolTensor

from environment.states import State, DAGState
from environment.actions import ForwardAction, DAGForwardAction
from estimator.state_representation import StateRepresentation


class ForwardPolicy(nn.Module, ABC):
    def __init__(self, uses_representation: bool = False) -> None:
        super().__init__()
        self.uses_representation = uses_representation

    @abstractmethod
    def forward(
        self, state: State, representation: Any | None = None
    ) -> tuple[ForwardAction, FloatTensor]:
        if self.uses_representation:
            assert representation is not None, "Policy requires a state representation"
        else:
            assert representation is None, "Policy does not use a state representation"


####################################################
#################### Subclasses ####################
####################################################


class DAGForwardPolicy(ForwardPolicy, ABC):
    def __init__(self, example_state: DAGState, representation_size: int) -> None:
        super().__init__(uses_representation=True)

        self.representation_size = representation_size
        self.num_rules = len(example_state.rules)

    def forward(
        self,
        state: DAGState,
        representation: tuple[FloatTensor, BoolTensor],
    ) -> tuple[DAGForwardAction, FloatTensor]:
        var_repr, var_mask = representation
        # TODO: continue
