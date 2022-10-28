from types import NoneType
import pytest

import torch

from gfn_parameterization.states import DAGState, SingleOutputDAGState
from tests.rules.test_base import DummyRule, DummyRuleToken


class TestDAGState:

    state_float = DAGState(
        max_actions=4,
        initial_vars=torch.randn(2, 2, 3, 4),
        rules=[
            DummyRule(
                num_args=i % 3 + 1,
                var_shape=(3, 4),
                embedding_size=32,
            )
            for i in range(3)
        ],
    )
    state_int = DAGState(
        max_actions=4,
        initial_vars=torch.ones(2, 2, dtype=torch.long),
        rules=[
            DummyRuleToken(
                num_args=i % 3 + 1,
                var_shape=(),
                embedding_size=32,
            )
            for i in range(3)
        ],
    )

    def test___init__(self) -> None:
        assert self.state_float.batch_size == 2
        assert self.state_float.var_shape == (3, 4)
        assert (self.state_float.num_vars == torch.ones(2, dtype=torch.long) * 2).all()
        assert (self.state_float.num_actions == torch.zeros(2, dtype=torch.long)).all()
        assert (
            self.state_float.var_mask
            == torch.tensor([[True, True, False, False, False, False]] * 2)
        ).all()
        assert (
            self.state_float.applied_rule_mask
            == torch.zeros_like(self.state_float.applied_rules, dtype=torch.bool)
        ).all()
        assert (
            self.state_float.leaf_mask
            == torch.tensor([[True, True, False, False, False, False]] * 2)
        ).all()
        assert self.state_float.vars.shape == (2, 4 + 2, 3, 4)
        assert self.state_float.applied_rules.shape == (2, 4)
        assert self.state_float.vars_to_rules.shape == (2, 4 + 2, 4)
        assert self.state_float.rules_to_vars.shape == (2, 4, 4 + 2)

        assert self.state_int.batch_size == 2
        assert self.state_int.var_shape == ()
        assert (self.state_int.num_vars == torch.ones(2, dtype=torch.long) * 2).all()
        assert (self.state_int.num_actions == torch.zeros(2, dtype=torch.long)).all()
        assert (
            self.state_int.var_mask
            == torch.tensor([[True, True, False, False, False, False]] * 2)
        ).all()
        assert (
            self.state_int.applied_rule_mask
            == torch.zeros_like(self.state_int.applied_rules, dtype=torch.bool)
        ).all()
        assert (
            self.state_int.leaf_mask
            == torch.tensor([[True, True, False, False, False, False]] * 2)
        ).all()
        assert self.state_int.vars.shape == (2, 4 + 2)
        assert self.state_int.applied_rules.shape == (2, 4)
        assert self.state_int.vars_to_rules.shape == (2, 4 + 2, 4)
        assert self.state_int.rules_to_vars.shape == (2, 4, 4 + 2)

    def test_clone(self) -> None:
        state_clone = self.state_float.clone()
        for attr, value in state_clone._buffers.items():
            assert (value == self.state_float._buffers[attr]).all()
            assert value is not self.state_float._buffers[attr]

    @pytest.mark.parametrize("state", [state_float, state_int])
    def test_forward_action_valid(self, state: DAGState) -> None:
        true_leaf_mask = state.leaf_mask.clone()
        true_vars = state.vars.clone()
        true_applied_rules = state.applied_rules.clone()
        true_vars_to_rules = state.vars_to_rules.clone()
        true_rules_to_vars = state.rules_to_vars.clone()

        # b0: r1(v1, v0) -> v2
        # b1: r0(v0) -> v2
        state = state.forward_action(
            rule_indices=torch.tensor([1, 0]),
            arg_indices=torch.tensor([[1, 0, -1], [0, -1, -1]]),
        )
        assert (state.num_vars == torch.ones(2, dtype=torch.long) * 3).all()
        assert (state.num_actions == torch.ones(2, dtype=torch.long)).all()
        true_leaf_mask[0, [0, 1]] = False
        true_leaf_mask[1, 0] = False
        true_leaf_mask[[0, 1], 2] = True
        assert (state.leaf_mask == true_leaf_mask).all()
        true_vars[range(2), state.num_vars - 1] = state.vars[
            range(2), state.num_vars - 1
        ]
        assert (state.vars == true_vars).all()
        true_applied_rules[0, 0] = 1
        true_applied_rules[1, 0] = 0
        assert (state.applied_rules == true_applied_rules).all()
        true_vars_to_rules[0, 0, 0] = 2
        true_vars_to_rules[0, 1, 0] = 1
        true_vars_to_rules[1, 0, 0] = 1
        assert (state.vars_to_rules == true_vars_to_rules).all()
        true_rules_to_vars[[0, 1], 0, 2] = 1
        assert (state.rules_to_vars == true_rules_to_vars).all()

        # b0: r1(v1, v0) -> v2
        #     r2(v2, v1, v0) -> v3
        # b1: r0(v0) -> v2
        state = state.forward_action(
            rule_indices=torch.tensor([2, -1]),
            arg_indices=torch.tensor([[2, 1, 0], [-1, -1, -1]]),
        )
        assert (state.num_vars == torch.tensor([4, 3])).all()
        assert (state.num_actions == torch.tensor([2, 1])).all()
        true_leaf_mask[0, 2] = False
        true_leaf_mask[0, 3] = True
        assert (state.leaf_mask == true_leaf_mask).all()
        true_vars[0, 3] = state.vars[0, 3]
        assert (state.vars == true_vars).all()
        true_applied_rules[0, 1] = 2
        assert (state.applied_rules == true_applied_rules).all()
        true_vars_to_rules[0, 0, 1] = 3
        true_vars_to_rules[0, 1, 1] = 2
        true_vars_to_rules[0, 2, 1] = 1
        assert (state.vars_to_rules == true_vars_to_rules).all()
        true_rules_to_vars[0, 1, 3] = 1
        assert (state.rules_to_vars == true_rules_to_vars).all()

    @pytest.mark.parametrize("state", [state_float, state_int])
    def test_forward_action_invalid(self, state: DAGState) -> None:
        # Not applying rules to any samples
        with pytest.raises(
            AssertionError, match="Must be updating at least one sample"
        ):
            _ = state.forward_action(
                rule_indices=torch.tensor([-1, -1]),
                arg_indices=torch.tensor([[-1], [-1]]),
            )

        # No arguments for a rule
        with pytest.raises(AssertionError, match="has no rule arguments"):
            _ = state.forward_action(
                rule_indices=torch.tensor([0, -1]),
                arg_indices=torch.tensor([[-1], [-1]]),
            )
        with pytest.raises(AssertionError, match="has no rule arguments"):
            _ = state.forward_action(
                rule_indices=torch.tensor([0, -1]),
                arg_indices=torch.tensor([[], []]),
            )

        # Out of bounds argument
        with pytest.raises(
            AssertionError,
            match="using an argument beyong the number of variables available",
        ):
            _ = state.forward_action(
                rule_indices=torch.tensor([1, -1]),
                arg_indices=torch.tensor([[0, 2], [-1, -1]]),
            )

        # Reached maximum number of actions
        arg_mask = torch.zeros(2, 6, dtype=torch.bool)
        arg_mask[0, 0] = True
        for _ in range(state.max_actions):
            state = state.forward_action(
                rule_indices=torch.tensor([0, -1]),
                arg_indices=torch.tensor([[0], [-1]]),
            )
        with pytest.raises(
            AssertionError,
            match="already at max_actions",
        ):
            _ = state.forward_action(
                rule_indices=torch.tensor([0, -1]),
                arg_indices=torch.tensor([[0], [-1]]),
            )

    @pytest.mark.parametrize("state", [state_float, state_int])
    def test_backward_action_valid(self, state: DAGState):
        # b0: v0
        #     r0(v1) -> v2
        # b1: r1(v0, v1) -> v2
        #     r1(v0, v1) -> v3
        state = state.forward_action(
            rule_indices=torch.tensor([0, 1]),
            arg_indices=torch.tensor([[1, -1], [0, 1]]),
        )
        state = state.forward_action(
            rule_indices=torch.tensor([-1, 1]),
            arg_indices=torch.tensor([[-1, -1], [0, 1]]),
        )

        # b0: v0, v1
        # b1: r1(v0, v1) -> v3
        new_state = state.backward_action(removal_indices=torch.tensor([2, 2]))
        assert (new_state.num_vars == state.num_vars - 1).all()
        assert (new_state.num_actions == state.num_actions - 1).all()
        true_leaf_mask = torch.zeros_like(state.leaf_mask)
        true_leaf_mask[0, [0, 1]] = True
        true_leaf_mask[1, 2] = True
        assert (new_state.leaf_mask == true_leaf_mask).all()
        assert (new_state.vars[0, 0:2] == state.vars[0, 0:2]).all()
        assert (new_state.vars[0, 2:] == 0).all()
        assert (new_state.vars[1, 0:3] == state.vars[1, [0, 1, 3]]).all()
        true_applied_rules = torch.zeros_like(state.applied_rules)
        true_applied_rules[1, 0] = 1
        assert (new_state.applied_rules == true_applied_rules).all()
        true_vars_to_rules = torch.zeros_like(state.vars_to_rules)
        true_vars_to_rules[1, 0, 0] = 1
        true_vars_to_rules[1, 1, 0] = 2
        assert (new_state.vars_to_rules == true_vars_to_rules).all()
        true_rules_to_vars = torch.zeros_like(state.rules_to_vars)
        true_rules_to_vars[1, 0, 2] = 1
        assert (new_state.rules_to_vars == true_rules_to_vars).all()

    @pytest.mark.parametrize("state", [state_float, state_int])
    def test_backward_action_invalid(self, state: DAGState):
        # b0: v0, v1
        # b1: r1(v0, v1) -> v2
        state = state.forward_action(
            rule_indices=torch.tensor([-1, 1]),
            arg_indices=torch.tensor([[-1, -1], [0, 1]]),
        )

        # Not removing any variables
        with pytest.raises(
            AssertionError, match="Must be removing variable for at least one sample"
        ):
            _ = state.backward_action(removal_indices=torch.tensor([-1, -1]))

        # Trying to remove more variables than were added
        with pytest.raises(
            AssertionError, match="Cannot remove more variables than have been added"
        ):
            _ = state.backward_action(removal_indices=torch.tensor([0, -1]))

        # b0: v0, r0(v1) -> v2
        # b1: r1(v0, v1) -> v2
        #     r1(v1, v2) -> v3
        state = state.forward_action(
            rule_indices=torch.tensor([0, 1]),
            arg_indices=torch.tensor([[1, -1], [1, 2]]),
        )

        # Trying to remove non-leaf variable
        with pytest.raises(AssertionError, match="Can only remove leaf variables"):
            _ = state.backward_action(removal_indices=torch.tensor([-1, 2]))

        # Trying to remove initial variable]))
        with pytest.raises(AssertionError, match="Cannot remove initial variables"):
            _ = state.backward_action(removal_indices=torch.tensor([0, -1]))

    @pytest.mark.parametrize("state", [state_float])
    def test_forward_grad_flow(self, state: DAGState):
        state.vars.requires_grad = True

        # b0: v0
        #     r0(v1) -> v2
        # b1: r1(v0, v1) -> v2
        #     r1(v0, v1) -> v3
        state_next = state.forward_action(
            rule_indices=torch.tensor([0, 1]),
            arg_indices=torch.tensor([[1, -1], [0, 1]]),
        )
        state_next = state_next.forward_action(
            rule_indices=torch.tensor([-1, 1]),
            arg_indices=torch.tensor([[-1, -1], [0, 1]]),
        )

        state_next.vars.sum().backward()
        assert state.vars.grad is not None


class TestSingleOutputDAGState:

    state_float = SingleOutputDAGState(
        max_actions=4,
        initial_vars=torch.randn(2, 2, 3, 4),
        rules=[
            DummyRule(
                num_args=i % 3 + 1,
                var_shape=(3, 4),
                embedding_size=32,
            )
            for i in range(3)
        ],
    )
    state_int = SingleOutputDAGState(
        max_actions=4,
        initial_vars=torch.ones(2, 2, dtype=torch.long),
        rules=[
            DummyRuleToken(
                num_args=i % 3 + 1,
                var_shape=(),
                embedding_size=32,
            )
            for i in range(3)
        ],
    )

    @pytest.mark.parametrize("state", [state_float, state_int])
    def test_output(self, state: SingleOutputDAGState):
        # b0: r1(v0, v1) -> v2
        # b1: r1(v0, v1) -> v2
        #     r1(v0, v1) -> v3
        state = state.forward_action(
            rule_indices=torch.tensor([1, 1]),
            arg_indices=torch.tensor([[0, 1], [0, 1]]),
        )
        state = state.forward_action(
            rule_indices=torch.tensor([-1, 1]),
            arg_indices=torch.tensor([[-1, -1], [0, 1]]),
        )

        outputs, is_valid = state.output()
        assert (is_valid == torch.tensor([True, False])).all()
        assert (outputs[0] == state.vars[0, 2]).all()
        assert (outputs[1] == 0).all()
