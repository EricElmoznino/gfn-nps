from abc import ABC, abstractmethod
from copy import deepcopy
from typing_extensions import Self

import torch
from torch import nn
from torch import Tensor, BoolTensor, LongTensor

from rules import Rule


class State(nn.Module, ABC):
    """
    Abstract base class for a GFN state.

    *Note*: This only subclasses nn.Module so that we can easily
    move the state between the cpu/gpu using self.register_buffer().
    If there is a more elegant way to achieve this, we should.
    """

    def __init__(
        self,
        batch_size: int,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.register_buffer("done", torch.zeros(batch_size, dtype=torch.bool))

    @abstractmethod
    def clone_instance(self) -> Self:
        pass

    def clone(self) -> Self:
        state = self.clone_instance()
        for attr, value in self._buffers.items():
            state._buffers[attr] = value.clone()
        return state


####################################################
#################### Subclasses ####################
####################################################


class DAGState(State):
    def __init__(
        self,
        max_actions: int,
        initial_vars: Tensor,
        rules: list[Rule],
    ) -> None:
        batch_size = initial_vars.shape[0]
        super().__init__(batch_size=batch_size)

        self.max_actions = max_actions
        self.rules = rules

        self._num_init_vars = initial_vars.shape[1]
        self.register_buffer(
            "_num_actions",
            torch.zeros(batch_size, dtype=torch.long),
        )

        # Stores the variables and applied rules in the graph
        self.register_buffer(
            "vars",
            torch.concat(
                [
                    initial_vars,
                    torch.zeros(
                        batch_size,
                        max_actions,
                        *initial_vars.shape[2:],  # var shape
                        dtype=initial_vars.dtype,
                    ),
                ],
                dim=1,
            ),
        )
        self.register_buffer(
            "applied_rules",
            torch.zeros(
                batch_size,
                max_actions,
                dtype=torch.long,
            ),
        )

        # Stores the connectivity of the graph
        self.register_buffer(
            "vars_to_rules",
            torch.zeros(
                batch_size,
                self.vars.shape[1],
                max_actions,
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "rules_to_vars",
            torch.zeros(
                batch_size,
                max_actions,
                self.vars.shape[1],
                dtype=torch.long,
            ),
        )

    @property
    def var_shape(self) -> tuple:
        return tuple(self.vars.shape[2:])

    @property
    def num_vars(self) -> LongTensor:
        return self._num_init_vars + self._num_actions

    @property
    def num_actions(self) -> LongTensor:
        return self._num_actions

    @property
    def var_mask(self) -> BoolTensor:
        var_range = torch.arange(self.vars.shape[1], device=self.vars.device)
        return var_range.unsqueeze(0) < self.num_vars.unsqueeze(1)

    @property
    def applied_rule_mask(self) -> BoolTensor:
        applied_rules_range = torch.arange(self.max_actions, device=self.vars.device)
        return applied_rules_range.unsqueeze(0) < self.num_actions.unsqueeze(1)

    @property
    def leaf_mask(self) -> BoolTensor:
        leafs = self.var_mask & (self.vars_to_rules.sum(dim=2) == 0)
        return leafs

    def clone_instance(self) -> Self:
        state = type(self)(
            self.max_actions, self.vars[:, : self._num_init_vars], self.rules
        )
        return state


class SingleOutputDAGState(DAGState):
    def output(self) -> tuple[Tensor, BoolTensor]:
        num_leafs = self.leaf_mask.sum(dim=1)
        is_valid = num_leafs == 1

        out = torch.zeros_like(self.vars[:, 0])
        selected_vars = self.leaf_mask[is_valid].long().argmax(dim=1)
        out[is_valid] = self.vars[is_valid, selected_vars]

        return out, is_valid
