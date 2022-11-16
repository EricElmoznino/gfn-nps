from abc import ABC, abstractmethod

from torch import BoolTensor, LongTensor


class ForwardAction(ABC):
    """Abstract class for actions"""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass

    @property
    @abstractmethod
    def terminating(self) -> BoolTensor:
        pass

    @property
    @abstractmethod
    def updating(self) -> BoolTensor:
        pass


class BackwardAction(ABC):
    """Abstract class for actions"""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass

    @property
    @abstractmethod
    def updating(self) -> BoolTensor:
        pass


####################################################
#################### Subclasses ####################
####################################################


class DAGForwardAction(ForwardAction):
    def __init__(
        self,
        terminating: BoolTensor,
        rule_indices: LongTensor,
        arg_indices: LongTensor,
    ) -> None:
        super().__init__()
        self._terminating = terminating
        self.rule_indices = rule_indices
        self.arg_indices = arg_indices

        # Check that the action is valid
        assert (
            self.updating.any() or self.terminating.any()
        ), "Must be updating at least one sample"
        assert (
            not self.updating.any()
            or arg_indices.shape[1] > 0
            and ((arg_indices[self.updating] != -1).sum(dim=1) > 0).all()
        ), "One of the samples being updated has no rule arguments"

    @property
    def batch_size(self) -> int:
        return len(self._terminating)

    @property
    def terminating(self) -> BoolTensor:
        return self._terminating

    @property
    def updating(self) -> BoolTensor:
        updating_indices = self.rule_indices != -1
        updating_indices &= ~self.terminating  # Don't change terminated state's data
        return updating_indices


class DAGBackwardAction(BackwardAction):
    def __init__(self, removal_indices: LongTensor) -> None:
        super().__init__()
        self.removal_indices = removal_indices

        # Check that the action is valid
        assert self.updating.any(), "Must be updating at least one sample"

    @property
    def batch_size(self) -> int:
        return len(self.removal_indices)

    @property
    def updating(self) -> BoolTensor:
        return self.removal_indices != -1
