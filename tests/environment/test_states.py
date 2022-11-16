import torch

from environment.states import DAGState, SingleOutputDAGState
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
