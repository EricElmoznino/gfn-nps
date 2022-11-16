import pytest

import torch

from environment.envs import SingleOutputDAGEnvironment
from environment.states import SingleOutputDAGState
from environment.actions import DAGForwardAction, DAGBackwardAction
from tests.rules.test_base import DummyRule, DummyRuleToken


class DummyEnvironment(SingleOutputDAGEnvironment):
    def __init__(self, token_rule: bool = False, max_actions: int = 4) -> None:
        if token_rule:
            rules = [
                DummyRuleToken(
                    num_args=i % 3 + 1,
                    var_shape=(),
                    embedding_size=32,
                )
                for i in range(3)
            ]
        else:
            rules = [
                DummyRule(
                    num_args=i % 3 + 1,
                    var_shape=(3, 4),
                    embedding_size=32,
                )
                for i in range(3)
            ]
        super().__init__(rules)

        self.token_rule = token_rule
        self.max_actions = max_actions

    def reset(self, batch_size: int = 2) -> SingleOutputDAGState:
        if self.token_rule:
            initial_vars = torch.ones(batch_size, 2, dtype=torch.long)
        else:
            initial_vars = torch.randn(batch_size, 2, 3, 4)
        self.state = SingleOutputDAGState(
            max_actions=self.max_actions,
            initial_vars=initial_vars,
            rules=self.rules,
        )
        return self.state.clone()


class TestSingleOutputDAGEnvironment:

    env_float = DummyEnvironment(token_rule=False)
    env_int = DummyEnvironment(token_rule=True)

    def test___init__(self) -> None:
        env = self.env_float
        assert len(env.rules) == 3
        assert env.state is None
        with pytest.raises(AttributeError):
            _ = env.done

    def test_reset(self) -> None:
        env = self.env_float
        state = env.reset()
        assert state is not None and env.state is not None
        assert state is not env.state
        for attr in state._buffers:
            assert (env.state._buffers[attr] == state._buffers[attr]).all()

    @pytest.mark.parametrize("env", [env_float, env_int])
    def test_step(self, env: DummyEnvironment) -> None:
        state = env.reset()
        true_done = state.done.clone()
        true_leaf_mask = state.leaf_mask.clone()
        true_vars = state.vars.clone()
        true_applied_rules = state.applied_rules.clone()
        true_vars_to_rules = state.vars_to_rules.clone()
        true_rules_to_vars = state.rules_to_vars.clone()

        # b0: r1(v1, v0) -> v2
        # b1: r0(v0) -> v2
        action = DAGForwardAction(
            terminating=torch.tensor([False, False]),
            rule_indices=torch.tensor([1, 0]),
            arg_indices=torch.tensor([[1, 0, -1], [0, -1, -1]]),
        )
        state = env.step(action)
        assert (state.done == true_done).all()
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
        assert state is not None and env.state is not None
        assert state is not env.state
        for attr in state._buffers:
            assert (env.state._buffers[attr] == state._buffers[attr]).all()

        # b0: r1(v1, v0) -> v2
        #     r2(v2, v1, v0) -> v3
        # b1: r0(v0) -> v2
        action = DAGForwardAction(
            terminating=torch.tensor([False, False]),
            rule_indices=torch.tensor([2, -1]),
            arg_indices=torch.tensor([[2, 1, 0], [-1, -1, -1]]),
        )
        state = env.step(action)
        assert (state.done == true_done).all()
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

        # b0: r1(v1, v0) -> v2
        #     r2(v2, v1, v0) -> v3
        #     r1(v2, v3) -> v4
        #     terminate
        # b1: r0(v0) -> v2
        action = DAGForwardAction(
            terminating=torch.tensor([False, False]),
            rule_indices=torch.tensor([1, -1]),
            arg_indices=torch.tensor([[2, 3], [-1, -1]]),
        )
        state = env.step(action)
        action = DAGForwardAction(
            terminating=torch.tensor([True, False]),
            rule_indices=torch.tensor([-1, -1]),
            arg_indices=torch.tensor([[], []]),
        )
        state = env.step(action)
        true_done[0] = True
        assert (state.done == true_done).all()

    @pytest.mark.parametrize("env", [env_float, env_int])
    def test_step_invalid(self, env: DummyEnvironment) -> None:
        _ = env.reset()

        # Terminating on invalid output
        with pytest.raises(AssertionError, match="Cannot terminate on invalid output"):
            action = DAGForwardAction(
                terminating=torch.tensor([True, False]),
                rule_indices=torch.tensor([-1, -1]),
                arg_indices=torch.tensor([[-1], [-1]]),
            )
            _ = env.step(action)

        # Out of bounds argument
        with pytest.raises(
            AssertionError,
            match="using an argument beyong the number of variables available",
        ):
            action = DAGForwardAction(
                terminating=torch.tensor([False, False]),
                rule_indices=torch.tensor([1, -1]),
                arg_indices=torch.tensor([[0, 2], [-1, -1]]),
            )
            _ = env.step(action)

        # Reached maximum number of actions
        arg_mask = torch.zeros(2, 6, dtype=torch.bool)
        arg_mask[0, 0] = True
        action = DAGForwardAction(
            terminating=torch.tensor([False, False]),
            rule_indices=torch.tensor([0, -1]),
            arg_indices=torch.tensor([[0], [-1]]),
        )
        for _ in range(env.max_actions):
            _ = env.step(action)
        with pytest.raises(
            AssertionError,
            match="sample that has already terminated",
        ):
            _ = env.step(action)

        # Applying a rule to a terminated sample
        _ = env.reset()
        action = DAGForwardAction(
            terminating=torch.tensor([False, False]),
            rule_indices=torch.tensor([1, -1]),
            arg_indices=torch.tensor([[1, 0, -1], [-1, -1, -1]]),
        )
        _ = env.step(action)
        action = DAGForwardAction(
            terminating=torch.tensor([True, False]),
            rule_indices=torch.tensor([-1, -1]),
            arg_indices=torch.tensor([[], []]),
        )
        _ = env.step(action)
        with pytest.raises(
            AssertionError,
            match="sample that has already terminated",
        ):
            action = DAGForwardAction(
                terminating=torch.tensor([False, False]),
                rule_indices=torch.tensor([0, -1]),
                arg_indices=torch.tensor([[0], [-1]]),
            )
            _ = env.step(action)

    @pytest.mark.parametrize("env", [env_float, env_int])
    def test_backward_step(self, env: DummyEnvironment) -> None:
        _ = env.reset()

        # b0: v0
        #     r0(v1) -> v2
        # b1: r1(v0, v1) -> v2
        #     r1(v0, v1) -> v3
        action = DAGForwardAction(
            terminating=torch.tensor([False, False]),
            rule_indices=torch.tensor([0, 1]),
            arg_indices=torch.tensor([[1, -1], [0, 1]]),
        )
        _ = env.step(action)
        action = DAGForwardAction(
            terminating=torch.tensor([False, False]),
            rule_indices=torch.tensor([-1, 1]),
            arg_indices=torch.tensor([[-1, -1], [0, 1]]),
        )
        state = env.step(action)

        # b0: v0, v1
        # b1: r1(v0, v1) -> v3
        action = DAGBackwardAction(removal_indices=torch.tensor([2, 2]))
        prev_state = env.backward_step(action)
        assert (prev_state.num_vars == state.num_vars - 1).all()
        assert (prev_state.num_actions == state.num_actions - 1).all()
        true_leaf_mask = torch.zeros_like(state.leaf_mask)
        true_leaf_mask[0, [0, 1]] = True
        true_leaf_mask[1, 2] = True
        assert (prev_state.leaf_mask == true_leaf_mask).all()
        assert (prev_state.vars[0, 0:2] == state.vars[0, 0:2]).all()
        assert (prev_state.vars[0, 2:] == 0).all()
        assert (prev_state.vars[1, 0:3] == state.vars[1, [0, 1, 3]]).all()
        true_applied_rules = torch.zeros_like(state.applied_rules)
        true_applied_rules[1, 0] = 1
        assert (prev_state.applied_rules == true_applied_rules).all()
        true_vars_to_rules = torch.zeros_like(state.vars_to_rules)
        true_vars_to_rules[1, 0, 0] = 1
        true_vars_to_rules[1, 1, 0] = 2
        assert (prev_state.vars_to_rules == true_vars_to_rules).all()
        true_rules_to_vars = torch.zeros_like(state.rules_to_vars)
        true_rules_to_vars[1, 0, 2] = 1
        assert (prev_state.rules_to_vars == true_rules_to_vars).all()

        # b0: v0, v1
        # b1: r1(v0, v1) -> v3
        #     terminate
        action = DAGForwardAction(
            terminating=torch.tensor([False, True]),
            rule_indices=torch.tensor([-1, -1]),
            arg_indices=torch.tensor([[], []]),
        )
        state = env.step(action)

        # b0: v0, v1
        # b1: r1(v0, v1) -> v3
        action = DAGBackwardAction(removal_indices=torch.tensor([-1, 2]))
        prev_state = env.backward_step(action)
        assert not prev_state.done.any()
        assert (prev_state.done != state.done).any()

    @pytest.mark.parametrize("env", [env_float, env_int])
    def test_backward_step_invalid(self, env: DummyEnvironment) -> None:
        _ = env.reset()

        # b0: v0, v1
        # b1: r1(v0, v1) -> v2
        action = DAGForwardAction(
            terminating=torch.tensor([False, False]),
            rule_indices=torch.tensor([-1, 1]),
            arg_indices=torch.tensor([[-1, -1], [0, 1]]),
        )
        _ = env.step(action)

        # Trying to remove more variables than were added
        with pytest.raises(
            AssertionError, match="Cannot remove more variables than have been added"
        ):
            action = DAGBackwardAction(removal_indices=torch.tensor([0, 1]))
            _ = env.backward_step(action)

        # b0: v0, r0(v1) -> v2
        # b1: r1(v0, v1) -> v2
        #     r1(v1, v2) -> v3
        action = DAGForwardAction(
            terminating=torch.tensor([False, False]),
            rule_indices=torch.tensor([0, 1]),
            arg_indices=torch.tensor([[1, -1], [1, 2]]),
        )
        _ = env.step(action)

        # Trying to remove non-leaf variable
        with pytest.raises(AssertionError, match="Can only remove leaf variables"):
            action = DAGBackwardAction(removal_indices=torch.tensor([-1, 2]))
            _ = env.backward_step(action)

        # Trying to remove initial variable]))
        with pytest.raises(AssertionError, match="Cannot remove initial variables"):
            action = DAGBackwardAction(removal_indices=torch.tensor([0, -1]))
            _ = env.backward_step(action)

    @pytest.mark.parametrize("env", [env_float])
    def test_grad_flow(self, env: DummyEnvironment) -> None:
        _ = env.reset()
        state = env.state
        state.vars.requires_grad = True

        # b0: v0
        #     r0(v1) -> v2
        # b1: r1(v0, v1) -> v2
        #     r1(v0, v1) -> v3
        action = DAGForwardAction(
            terminating=torch.tensor([False, False]),
            rule_indices=torch.tensor([0, 1]),
            arg_indices=torch.tensor([[1, -1], [0, 1]]),
        )
        _ = env.step(action)
        action = DAGForwardAction(
            terminating=torch.tensor([False, False]),
            rule_indices=torch.tensor([-1, 1]),
            arg_indices=torch.tensor([[-1, -1], [0, 1]]),
        )
        state_next = env.step(action)

        state_next.vars.sum().backward()
        assert state.vars.grad is not None

    @pytest.mark.parametrize("env", [env_float, env_int])
    def test_output(self, env: DummyEnvironment) -> None:
        _ = env.reset()

        # b0: r1(v0, v1) -> v2
        # b1: r1(v0, v1) -> v2
        #     r1(v0, v1) -> v3
        action = DAGForwardAction(
            terminating=torch.tensor([False, False]),
            rule_indices=torch.tensor([1, 1]),
            arg_indices=torch.tensor([[0, 1], [0, 1]]),
        )
        _ = env.step(action)
        action = DAGForwardAction(
            terminating=torch.tensor([False, False]),
            rule_indices=torch.tensor([-1, 1]),
            arg_indices=torch.tensor([[-1, -1], [0, 1]]),
        )
        state = env.step(action)

        outputs, is_valid = state.output()
        assert (is_valid == torch.tensor([True, False])).all()
        assert (outputs[0] == state.vars[0, 2]).all()
        assert (outputs[1] == 0).all()
