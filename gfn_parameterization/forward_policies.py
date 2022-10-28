from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
from torch import FloatTensor, LongTensor, BoolTensor

from gfn_parameterization.states import State, DAGState


class ForwardPolicy(nn.Module, ABC):
    @abstractmethod
    def __init__(self, example_state: State, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def forward(self, state: State) -> tuple[State, FloatTensor]:
        pass


class DAGForwardPolicy(ForwardPolicy, ABC):
    def __init__(
        self,
        example_state: DAGState,
        node_embedding_size: int,
    ) -> None:
        super().__init__()

        self.var_shape = example_state.var_shape
        self.num_rules = len(example_state.rules)
        self.var_size = int(np.prod(self.var_shape))
        self.node_embedding_size = node_embedding_size

        if node_embedding_size != self.var_size:
            self.var_node_embedding = nn.Linear(self.var_size, node_embedding_size)
        else:
            self.var_node_embedding = nn.Identity()

        example_rule = example_state.rules[0]
        if node_embedding_size != example_rule.embedding_size:
            self.rule_node_embedding = nn.Linear(
                example_rule.embedding_size, node_embedding_size
            )
        else:
            self.rule_node_embedding = nn.Identity()

    def forward(self, state: DAGState) -> tuple[DAGState, FloatTensor]:

        pass

    def make_graph(self, state: DAGState) -> tuple[FloatTensor, LongTensor, BoolTensor]:
        """
        Create the full node feature matrix and adjacency matrix with edge attributes.
        Edge attributes are correspond to:
          0: rule -> var
          1..N: var (n'th argument) -> rule

        Args:
            state (DAGState): State to make graph from

        Returns:
            tuple[FloatTensor, LongTensor, BoolTensor]: Node features matrix,
            adjacency matrix, and mask for unfilled nodes
        """
        var_nodes = state.vars.view(state.batch_size, -1, self.var_size)
        var_nodes = self.var_node_embedding(var_nodes)
        rule_nodes = torch.stack([rule.embedding for rule in state.rules], dim=1)
        rule_nodes = self.rule_node_embedding(rule_nodes)
        nodes = torch.cat([var_nodes, rule_nodes], dim=1)

        mask = torch.concat([state.var_mask, state.rule_mask], dim=1)

    @property
    @abstractmethod
    def representation_shape(self) -> tuple[int, ...]:
        pass

    @abstractmethod
    def representation(
        self, nodes: FloatTensor, adjacency: LongTensor, node_mask: BoolTensor
    ) -> FloatTensor:
        pass
