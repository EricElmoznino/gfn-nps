from abc import ABC, abstractmethod

import torch
from torch import Tensor, BoolTensor

from environment.states import State, DAGState, SingleOutputDAGState
from environment.actions import (
    ForwardAction,
    BackwardAction,
    DAGForwardAction,
    DAGBackwardAction,
)
from rules import Rule


class Environment(ABC):
    """Abstract class for environments"""

    def __init__(self) -> None:
        super().__init__()

        self.state: State | None = None

    @abstractmethod
    def reset(self, batch_size: int) -> State:
        pass

    @abstractmethod
    def step(self, action: ForwardAction) -> State:
        assert self.state is not None, "Environment has not been reset"
        next_state = self.state.clone()
        next_state.done |= action.terminating
        return next_state

    @abstractmethod
    def backward_step(self, action: BackwardAction) -> State:
        assert self.state is not None, "Environment has not been reset"
        prev_state = self.state.clone()
        prev_state.done &= ~action.updating
        return prev_state

    @property
    def done(self) -> BoolTensor:
        return self.state.done


####################################################
#################### Subclasses ####################
####################################################


class DAGEnvironment(Environment):
    def __init__(self, rules: list[Rule]) -> None:
        super().__init__()
        self.state: DAGState | None = None
        self.rules = rules

    def step(self, action: DAGForwardAction) -> DAGState:
        next_state: DAGState = super().step(action)

        # If we're just terminating episodes, no further processing is needed
        if action.terminating.any() and not action.updating.any():
            self.state = next_state
            return next_state.clone()

        # Convenience variables
        rule_indices, arg_indices = action.rule_indices, action.arg_indices
        sample_indices = torch.arange(
            next_state.batch_size, device=next_state.vars.device
        )
        updated_indices = sample_indices[action.updating]

        # Check that the action is valid on this state
        assert not (
            next_state.done & action.updating
        ).any(), "Cannot update a sample that has already terminated"
        assert (
            arg_indices.max(dim=1).values < next_state.num_vars[updated_indices]
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
            num_args = next_state.rules[rule_index].num_args
            args = next_state.vars[
                indices.unsqueeze(1),
                arg_indices[indices, :num_args],
            ]
            return args

        rule_sample_indices = [sample_indices[rule_indices == i] 
                               for i in range(len(next_state.rules))]  # fmt: skip
        rule_args = [get_rule_args(i) 
                     for i in range(len(next_state.rules))]  # fmt: skip
        rule_outputs = [rule(args.to(rule.device)) if args is not None else None
                        for rule, args in zip(next_state.rules, rule_args)]  # fmt: skip

        # Add the new variables to the graph
        for indices, outputs in zip(rule_sample_indices, rule_outputs):
            if outputs is not None:
                outputs = outputs.to(next_state.vars.device)
                next_state.vars[indices, next_state.num_vars[indices]] = outputs

        # Add the new rule nodes to the graph
        next_state.applied_rules[
            updated_indices,
            next_state.num_actions[updated_indices],
        ] = rule_indices[updated_indices]

        # Update the connectivity
        for i in range(arg_indices.shape[1]):
            arg_num = i + 1
            arg_num_var_indices = arg_indices[:, i]
            arg_num_sample_indices = sample_indices[arg_num_var_indices != -1]
            if len(arg_num_sample_indices) == 0:
                break
            next_state.vars_to_rules[
                arg_num_sample_indices,
                arg_num_var_indices[arg_num_sample_indices],
                next_state.num_actions[arg_num_sample_indices],
            ] = arg_num
        next_state.rules_to_vars[
            updated_indices,
            next_state.num_actions[updated_indices],
            next_state.num_vars[updated_indices],
        ] = 1

        # Increment the number of actions
        next_state._num_actions[updated_indices] += 1
        next_state.done |= next_state.num_actions == next_state.max_actions

        self.state = next_state
        return next_state.clone()

    def backward_step(self, action: DAGBackwardAction) -> DAGState:
        prev_state = super().backward_step(action)

        # Convenience variables
        removal_indices = action.removal_indices
        sample_indices = torch.arange(
            prev_state.batch_size, device=prev_state.vars.device
        )
        updated_indices = sample_indices[action.updating]
        removal_indices = removal_indices[updated_indices]

        # Check that the action is valid on this state
        assert (
            prev_state.num_actions[updated_indices].min() > 0
        ), "Cannot remove more variables than have been added"
        assert prev_state.leaf_mask[
            updated_indices, removal_indices
        ].all(), "Can only remove leaf variables"
        assert (
            removal_indices >= prev_state._num_init_vars
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
        prev_state.vars[updated_indices] = shift_down(
            prev_state.vars[updated_indices], removal_indices, dim=1
        )
        parent_rule = prev_state.rules_to_vars[
            updated_indices, :, removal_indices
        ].argmax(dim=1)
        prev_state.applied_rules[updated_indices] = shift_down(
            prev_state.applied_rules[updated_indices], parent_rule, dim=1
        )

        # Update the connectivity
        # (we shift all variables and rules that come after removal_indices down by 1)
        prev_state.vars_to_rules[updated_indices] = shift_down(
            prev_state.vars_to_rules[updated_indices], removal_indices, dim=1
        )
        prev_state.vars_to_rules[updated_indices] = shift_down(
            prev_state.vars_to_rules[updated_indices], parent_rule, dim=2
        )
        prev_state.rules_to_vars[updated_indices] = shift_down(
            prev_state.rules_to_vars[updated_indices], parent_rule, dim=1
        )
        prev_state.rules_to_vars[updated_indices] = shift_down(
            prev_state.rules_to_vars[updated_indices], removal_indices, dim=2
        )

        # Decrement the number of actions
        prev_state._num_actions[updated_indices] -= 1

        self.state = prev_state
        return prev_state.clone()


class SingleOutputDAGEnvironment(DAGEnvironment):
    def __init__(self, rules: list[Rule]) -> None:
        super().__init__(rules=rules)
        self.state: SingleOutputDAGState | None = None

    def step(self, action: DAGForwardAction) -> SingleOutputDAGState:
        assert self.state is not None, "Environment has not been reset"
        _, is_valid = self.state.output()
        assert (
            is_valid >= action.terminating
        ).all(), "Cannot terminate on invalid output"
        return super().step(action)
