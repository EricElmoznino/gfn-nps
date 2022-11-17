from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch import FloatTensor, BoolTensor
import numpy as np

from environment.states import State, DAGState


class StateRepresentation(nn.Module, ABC):
    @abstractmethod
    def forward(self, state: State) -> Any:
        pass


####################################################
#################### Subclasses ####################
####################################################


class DAGRepresentation(StateRepresentation, ABC):
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

        if node_embedding_size != self.var_size + 2:
            self.var_node_embedding = nn.Linear(self.var_size, node_embedding_size - 2)
        else:
            self.var_node_embedding = nn.Identity()

        example_rule = example_state.rules[0]
        if node_embedding_size != example_rule.embedding_size + 2:
            self.rule_node_embedding = nn.Linear(
                example_rule.embedding_size, node_embedding_size - 2
            )
        else:
            self.rule_node_embedding = nn.Identity()

    def forward(self, state: DAGState) -> tuple[FloatTensor, BoolTensor]:
        nodes, adjacency, node_mask = self.make_graph(state)
        var_repr = self.representation(nodes, adjacency, node_mask)
        var_repr = nodes[:, state.vars.shape[1]]
        return var_repr, state.var_mask

    def make_graph(
        self, state: DAGState
    ) -> tuple[FloatTensor, FloatTensor, BoolTensor]:
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
        # Embed the variables and rules
        var_nodes = state.vars.view(state.batch_size, -1, self.var_size)
        var_nodes = self.var_node_embedding(var_nodes)
        rule_nodes = torch.stack([rule.embedding for rule in state.rules], dim=1)
        rule_nodes = self.rule_node_embedding(rule_nodes)

        # Add binary variable/rule identifiers through concatenation
        var_nodes = torch.cat(
            [
                var_nodes,
                torch.zeros_like(var_nodes[:, :, 0]),
                torch.ones_like(var_nodes[:, :, 0]),
            ],
            dim=2,
        )
        rule_nodes = torch.cat(
            [
                rule_nodes,
                torch.ones_like(rule_nodes[:, :, 0]),
                torch.zeros_like(rule_nodes[:, :, 0]),
            ],
            dim=2,
        )

        # Combine the variable/rule nodes through concatenation
        nodes = torch.cat([var_nodes, rule_nodes], dim=1)

        # Create the adjacency matrix with one-hot edge features
        # 1. Edge attribute =0 for rule -> var
        # 2. Edge attribute =arg_num for var -> rule
        num_nodes, num_vars, num_rules = (
            nodes.shape[1],
            var_nodes.shape[1],
            rule_nodes.shape[1],
        )
        max_rule_args = max([rule.num_args for rule in state.rules])
        adjacency = torch.zeros(
            state.batch_size,
            num_nodes,
            num_nodes,
            max_rule_args + 1,
            dtype=torch.float,
            device=nodes.device,
        )
        adjacency[:, :num_vars, -num_rules] = F.one_hot(
            state.vars_to_rules, max_rule_args + 1
        )
        adjacency[:, -num_rules:, :num_vars, 0] = state.rules_to_vars.float()

        # Combine the variable/rule node masks through concatenation
        node_mask = torch.concat([state.var_mask, state.applied_rule_mask], dim=1)

        return nodes, adjacency, node_mask

    @property
    @abstractmethod
    def representation_size(self) -> int:
        pass

    @abstractmethod
    def representation(
        self, nodes: FloatTensor, adjacency: FloatTensor, node_mask: BoolTensor
    ) -> FloatTensor:
        pass
