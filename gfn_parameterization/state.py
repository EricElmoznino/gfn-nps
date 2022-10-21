from abc import ABC, abstractmethod
from typing_extensions import Self

import torch
from torch import nn
from torch import BoolTensor, LongTensor, FloatTensor

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
        initial_vars: FloatTensor,
        rules: list[Rule],
    ) -> None:
        self.max_actions = max_actions
        self.rules = rules

        batch_size = initial_vars.shape[0]
        self.register_buffer(
            "_num_init_vars",
            torch.full(
                (batch_size,),
                fill_value=initial_vars.shape[1],
                dtype=torch.long,
            ),
        )
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
    def leaf_mask(self) -> BoolTensor:
        var_range = torch.arange(self.vars.shape[1], device=self.vars.device)
        is_var = var_range.unsqueeze(0) < self.num_vars.unsqueeze(1)
        leafs = is_var & self.vars_to_rules.sum(dim=2) == 0
        return leafs

    def forward_action(
        self,
        rule_indices: torch.LongTensor,
        arg_mask: BoolTensor,
        arg_order: LongTensor,
    ) -> Self:
        sample_indices = torch.arange(self.batch_size, device=self.vars.device)
        updated_indices = sample_indices[rule_indices != -1]
        assert len(updated_indices) > 0
        assert self.num_actions[updated_indices].max() < self.max_actions

        # Apply the rules
        # *Note*: We run each rule on all its inputs. This looks
        # serial, but it will actually parallelize automatically if
        # the rules are on different GPUs because GPU operations are
        # asynchronous.
        def get_rule_args(indices: LongTensor) -> FloatTensor:
            if len(indices) == 0:
                return None
            args = self.vars[indices]
            args = args[arg_mask[indices]]
            args = args[arg_order[indices, : args.shape[1]]]
            return args

        rule_sample_indices = [sample_indices[rule_indices == i] 
                               for i in range(len(self.rules))]  # fmt: skip
        rule_args = [get_rule_args(indices) 
                     for indices in rule_sample_indices]  # fmt: skip
        rule_outputs = [rule(args.to(rule.device)) if args is not None else None
                        for rule, args in zip(self.rules, rule_args)]  # fmt: skip

        # Add the new variables to the graph
        for indices, outputs in zip(rule_sample_indices, rule_outputs):
            if outputs is not None:
                outputs = outputs.to(self.vars.device)
                self.vars[indices, self.num_vars[indices]] = outputs

        # Create filtered versions of variables for convenience
        num_actions = self.num_actions[updated_indices]
        num_vars = self.num_vars[updated_indices]
        rules_indices = rule_indices[updated_indices]
        arg_mask = arg_mask[updated_indices]

        # Add the new rule nodes to the graph
        self.applied_rules[updated_indices, num_actions] = rules_indices

        # Update the connectivity
        mask = torch.zeros_like(self.vars_to_rules, dtype=torch.bool)
        mask[updated_indices, :, num_actions] = arg_mask
        self.vars_to_rules[mask] = 1
        self.rules_to_vars[updated_indices, num_actions, num_vars] = 1

        # Increment the number of actions
        self._num_actions[updated_indices] += 1

    def backward_action(self, removal_indices: torch.LongTensor) -> Self:
        assert len(removal_indices) > 0
        modifying = torch.arange(self.batch_size, device=self.vars.device)
        modifying = modifying[removal_indices != -1]
        assert self.num_actions[modifying].min() > 0
        removal_indices = removal_indices[modifying]
        assert self.leaf_mask[modifying, removal_indices].all()

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
        self.vars[modifying] = shift_down(self.vars[modifying], removal_indices, dim=1)
        parent_rule = self.rules_to_vars[modifying, :, removal_indices].argmax(dim=1)
        self.applied_rules[modifying] = shift_down(
            self.appied_rules[modifying], parent_rule, dim=1
        )

        # Update the connectivity
        # (we shift all variables and rules that come after removal_indices down by 1)
        self.vars_to_rules[modifying] = shift_down(
            self.vars_to_rules[modifying], removal_indices, dim=1
        )
        self.vars_to_rule[modifying] = shift_down(
            self.vars_to_rules[modifying], parent_rule, dim=2
        )
        self.rules_to_vars[modifying] = shift_down(
            self.rules_to_vars[modifying], parent_rule, dim=1
        )
        self.rules_to_vars[modifying] = shift_down(
            self.rules_to_vars[modifying], removal_indices, dim=2
        )

        # Decrement the number of actions
        self._num_actions[modifying] -= 1


class SingleOutputDAGState(DAGState):
    def output(self) -> tuple(FloatTensor, BoolTensor):
        num_leafs = self.leaf_mask.sum(dim=1)
        is_valid = num_leafs == 1

        out = torch.full_like(self.vars[:, 0], fill_value=torch.nan)
        selected_vars = self.leaf_mask[is_valid].long().argmax(dim=1)
        out[is_valid] = self.vars[is_valid, selected_vars]

        return out, is_valid
