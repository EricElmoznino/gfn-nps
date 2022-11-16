import pytest

import torch

from environment.actions import DAGForwardAction, DAGBackwardAction


class TestDAGForwardAction:
    def test_init(self):
        terminating = torch.tensor([True, False, False])
        rule_indices = torch.tensor([0, 1, -1])
        arg_indices = torch.tensor([[0, 1], [1, 0], [-1, -1]])
        action = DAGForwardAction(terminating, rule_indices, arg_indices)
        assert action.batch_size == 3
        assert (action.terminating == terminating).all()
        assert (action.rule_indices == rule_indices).all()
        assert (action.arg_indices == arg_indices).all()
        assert (action.updating == torch.tensor([False, True, False])).all()

        terminating = torch.tensor([True, False, False])
        rule_indices = torch.tensor([-1, -1, -1])
        arg_indices = torch.tensor([[], [], []])
        assert DAGForwardAction(terminating, rule_indices, arg_indices) is not None

    def test_init_invalid(self):
        terminating = torch.tensor([False, False, False])
        rule_indices = torch.tensor([-1, -1, -1])
        arg_indices = torch.tensor([[], [], []])
        with pytest.raises(
            AssertionError, match="Must be updating at least one sample"
        ):
            _ = DAGForwardAction(terminating, rule_indices, arg_indices)
        rule_indices[0] = 0
        with pytest.raises(
            AssertionError,
            match="One of the samples being updated has no rule arguments",
        ):
            _ = DAGForwardAction(terminating, rule_indices, arg_indices)
        arg_indices = torch.tensor([[-1], [-1], [-1]])
        with pytest.raises(
            AssertionError,
            match="One of the samples being updated has no rule arguments",
        ):
            _ = DAGForwardAction(terminating, rule_indices, arg_indices)


class TestDAGBackwardAction:
    def test_init(self):
        removal_indices = torch.tensor([0, 1, -1])
        action = DAGBackwardAction(removal_indices)
        assert action.batch_size == 3
        assert (action.removal_indices == removal_indices).all()
        assert (action.updating == torch.tensor([True, True, False])).all()

    def test_init_invalid(self):
        removal_indices = torch.tensor([-1, -1, -1])
        with pytest.raises(
            AssertionError, match="Must be updating at least one sample"
        ):
            _ = DAGBackwardAction(removal_indices)