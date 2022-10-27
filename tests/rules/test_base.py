import pytest

import torch
from torch import LongTensor, nn
from torch import FloatTensor

from rules import Rule


class DummyRule(Rule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.network = nn.Sequential(
            nn.Linear(self.var_shape[-1], 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.var_shape[-1]),
        )

    def apply(self, vars: FloatTensor) -> FloatTensor:
        return self.network(vars).mean(dim=1)


class DummyRuleToken(Rule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.var_shape == ()

        self.network = nn.Sequential(
            nn.Linear(self.num_args, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def apply(self, vars: LongTensor) -> LongTensor:
        vars = vars.float()
        return self.network(vars).squeeze(dim=1).long()


class TestRule:
    def test_embedding(self) -> None:
        rule = DummyRule(num_args=2, var_shape=(3, 4), embedding_size=32)
        assert rule.embedding.shape == (rule.embedding_size,) == (32,)

    def test_apply(self) -> None:
        rule = DummyRule(num_args=2, var_shape=(3, 4), embedding_size=32)
        vars = torch.randn(5, 2, 3, 4)
        output_var = rule(vars)
        assert output_var.shape == (vars.shape[0], *vars.shape[2:])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_to(self) -> None:
        rule = DummyRule(num_args=2, var_shape=(3, 4), embedding_size=32)

        rule.to(torch.device("cuda"))
        assert rule.device == torch.device("cuda")
        for param in rule.parameters():
            assert param.device == torch.device("cuda")

        vars = torch.randn(5, 2, 3, 4).cuda()
        output_var = rule(vars)
        assert output_var.device == torch.device("cuda")

        rule.to(torch.device("cpu"))
        assert rule.device == torch.device("cpu")
        for param in rule.parameters():
            assert param.device == torch.device("cpu")

        vars = torch.randn(5, 2, 3, 4)
        output_var = rule(vars)
        assert output_var.device == torch.device("cpu")
