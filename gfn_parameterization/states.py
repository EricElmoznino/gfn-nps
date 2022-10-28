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

    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass

    @abstractmethod
    def forward_action(self, *args, **kwargs) -> Self:
        pass

    @abstractmethod
    def backward_action(self, *args, **kwargs) -> Self:
        pass


class DAGState(State):
    def __init__(
        self,
        max_actions: int,
        initial_vars: Tensor,
        rules: list[Rule],
    ) -> None:
        super().__init__()

        self.max_actions = max_actions
        self.rules = rules

        batch_size = initial_vars.shape[0]
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
    def batch_size(self) -> int:
        return self.vars.shape[0]

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

    def forward_action(
        self,
        rule_indices: torch.LongTensor,
        arg_indices: LongTensor,
    ) -> Self:
        state = self.clone()

        sample_indices = torch.arange(state.batch_size, device=state.vars.device)
        updated_indices = sample_indices[rule_indices != -1]
        assert len(updated_indices) > 0, "Must be updating at least one sample"
        assert (
            state.num_actions[updated_indices].max() < state.max_actions
        ), "One of the samples being updated is already at max_actions"
        assert (
            arg_indices.shape[1] > 0
            and ((arg_indices[updated_indices] != -1).sum(dim=1) > 0).all()
        ), "One of the samples being updated has no rule arguments"
        assert (
            arg_indices.max(dim=1).values < state.num_vars[updated_indices]
        ).all(), "One of the samples being updated is using an argument beyong the number of variables available"

        # Apply the rules
        # *Note*: We run each rule on all its inputs. This looks
        # serial, but it will actually parallelize automatically if
        # the rules are on different GPUs because GPU operations are
        # asynchronous.
        def get_rule_args(rule_index: int) -> Tensor:
            # Selects the arguments for a rule in the correct order
            indices = rule_sample_indices[rule_index]
            num_samples = len(indices)
            if num_samples == 0:
                return None
            num_args = state.rules[rule_index].num_args
            args = state.vars[
                indices.unsqueeze(1),
                arg_indices[indices, :num_args],
            ]
            return args

        rule_sample_indices = [sample_indices[rule_indices == i] 
                               for i in range(len(state.rules))]  # fmt: skip
        rule_args = [get_rule_args(i) 
                     for i in range(len(state.rules))]  # fmt: skip
        rule_outputs = [rule(args.to(rule.device)) if args is not None else None
                        for rule, args in zip(state.rules, rule_args)]  # fmt: skip

        # Add the new variables to the graph
        for indices, outputs in zip(rule_sample_indices, rule_outputs):
            if outputs is not None:
                outputs = outputs.to(state.vars.device)
                state.vars[indices, state.num_vars[indices]] = outputs

        # Add the new rule nodes to the graph
        state.applied_rules[
            updated_indices,
            state.num_actions[updated_indices],
        ] = rule_indices[updated_indices]

        # Update the connectivity
        for i in range(arg_indices.shape[1]):
            arg_num = i + 1
            arg_num_var_indices = arg_indices[:, i]
            arg_num_sample_indices = sample_indices[arg_num_var_indices != -1]
            if len(arg_num_sample_indices) == 0:
                break
            state.vars_to_rules[
                arg_num_sample_indices,
                arg_num_var_indices[arg_num_sample_indices],
                state.num_actions[arg_num_sample_indices],
            ] = arg_num
        state.rules_to_vars[
            updated_indices,
            state.num_actions[updated_indices],
            state.num_vars[updated_indices],
        ] = 1

        # Increment the number of actions
        state._num_actions[updated_indices] += 1

        return state

    def backward_action(self, removal_indices: torch.LongTensor) -> Self:
        state = self.clone()

        sample_indices = torch.arange(state.batch_size, device=state.vars.device)
        updated_indices = sample_indices[removal_indices != -1]
        assert (
            len(updated_indices) > 0
        ), "Must be removing variable for at least one sample"
        assert (
            state.num_actions[updated_indices].min() > 0
        ), "Cannot remove more variables than have been added"
        removal_indices = removal_indices[updated_indices]
        assert state.leaf_mask[
            updated_indices, removal_indices
        ].all(), "Can only remove leaf variables"
        assert (
            removal_indices[updated_indices] >= state._num_init_vars
        ).all(), "Cannot remove initial variables"

        def shift_down(x, idx, dim):
            x_range = torch.arange(x.shape[dim], device=x.device)
            x_range = x_range[
                [None if i != dim else slice(None) for i in range(x.ndim)]
            ]
            idx = idx[[slice(None)] + [None] * (x.ndim - 1)]
            next_x = x_range > idx
            next_x = next_x.narrow(dim, 1, x.shape[dim] - 1)
            x_shifted = ~next_x * x.narrow(
                dim, 0, x.shape[dim] - 1
            ) + next_x * x.narrow(dim, 1, x.shape[dim] - 1)
            x_shifted = torch.cat(
                [x_shifted, torch.zeros_like(x.narrow(dim, 0, 1))], dim=dim
            )
            return x_shifted

        # Remove the variable and the parent rule node that produced it
        # (we shift all variables and rules that come after removal_indices down by 1)
        state.vars[updated_indices] = shift_down(
            state.vars[updated_indices], removal_indices, dim=1
        )
        parent_rule = state.rules_to_vars[updated_indices, :, removal_indices].argmax(
            dim=1
        )
        state.applied_rules[updated_indices] = shift_down(
            state.applied_rules[updated_indices], parent_rule, dim=1
        )

        # Update the connectivity
        # (we shift all variables and rules that come after removal_indices down by 1)
        state.vars_to_rules[updated_indices] = shift_down(
            state.vars_to_rules[updated_indices], removal_indices, dim=1
        )
        state.vars_to_rules[updated_indices] = shift_down(
            state.vars_to_rules[updated_indices], parent_rule, dim=2
        )
        state.rules_to_vars[updated_indices] = shift_down(
            state.rules_to_vars[updated_indices], parent_rule, dim=1
        )
        state.rules_to_vars[updated_indices] = shift_down(
            state.rules_to_vars[updated_indices], removal_indices, dim=2
        )

        # Decrement the number of actions
        state._num_actions[updated_indices] -= 1

        return state

    def clone(self) -> Self:
        state = type(self)(
            self.max_actions, self.vars[:, : self._num_init_vars], self.rules
        )
        for attr, value in self._buffers.items():
            state._buffers[attr] = value.clone()
        return state


class SingleOutputDAGState(DAGState):
    def output(self) -> tuple[Tensor, BoolTensor]:
        num_leafs = self.leaf_mask.sum(dim=1)
        is_valid = num_leafs == 1

        out = torch.zeros_like(self.vars[:, 0])
        selected_vars = self.leaf_mask[is_valid].long().argmax(dim=1)
        out[is_valid] = self.vars[is_valid, selected_vars]

        return out, is_valid
