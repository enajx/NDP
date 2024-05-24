import torch
import torch.distributions as td
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph


class GeneratedNetwork(MessagePassing):
    def __init__(
        self,
        graph,
        aggr="add",
        operations=[torch.add, torch.subtract, torch.multiply],
        activations=[torch.nn.Identity(), torch.relu, torch.tanh],
    ):
        super().__init__(aggr=aggr)  # "Add" aggregation (Step 5).
        self.graph = graph

        self.value_channel = 0
        self.replication_channel = 1

        self.operations = operations
        self.activations = activations
        self.num_operations = len(self.operations)
        self.num_activations = len(self.activations)

        self.operation_channels = [2, self.num_operations + 2]
        self.activation_channels = [
            self.operation_channels[-1],
            self.num_activations + self.operation_channels[-1],
        ]

        self.topological_order = self.graph.topological_sort()

    def compute_propagation(self, x, edge_index):
        # Step 4-5: Start propagating messages.

        out = (
            torch.ones(
                x[:, self.value_channel : self.value_channel + 1].size(),
                device=x.device,
            )
            * x[:, self.value_channel : self.value_channel + 1]
        )
        output_node_idx = self.graph.num_input_nodes
        count = 0
        for node in self.topological_order[self.graph.num_input_nodes :]:
            operation_logits = x[
                node, self.operation_channels[0] : self.operation_channels[1]
            ]
            operations_dist = td.one_hot_categorical.OneHotCategorical(
                logits=operation_logits
            )
            operation_probs = operations_dist.sample().squeeze()

            if count < output_node_idx:
                activation_logits = x[
                    node, self.activation_channels[0] : self.activation_channels[1]
                ]

                activations_dist = td.one_hot_categorical.OneHotCategorical(
                    logits=activation_logits
                )
                activation_probs = activations_dist.sample().squeeze()
            else:
                activation_probs = torch.zeros((self.num_activations,), device=x.device)
                activation_probs[0] = 1.0

            _, edges, _, _ = k_hop_subgraph(node, 1, edge_index)

            node_propagate = self.propagate(
                edges,
                x=out,
                operation_probs=operation_probs,
                activation_probs=activation_probs,
            )
            out[node] = out[node] + node_propagate[node] - out[node]
            count += 1
        return out

    def message(self, x_i, x_j, operation_probs, activation_probs):
        # x_i has shape [E, out_channels]
        # x_j has shape [E, out_channels]

        operation_outputs = []
        for op in self.operations:
            operation_outputs.append(op(x_i, x_j))  # [E, out_channels]

        operation_outputs = torch.stack(operation_outputs) * operation_probs.view(
            (self.num_operations, 1, 1)
        )  # [num_operations, E, out_channels]
        operation_outputs = torch.sum(operation_outputs, dim=0)  # [E, out_channels]

        activation_outputs = []
        for act in self.activations:
            activation_outputs.append(act(operation_outputs))  # [E, out_channels]

        activation_outputs = torch.stack(activation_outputs) * activation_probs.view(
            (self.num_activations, 1, 1)
        )  # [num_activations, E, out_channels]
        activation_outputs = torch.sum(activation_outputs, dim=0)  # [E, out_channels]

        return activation_outputs

    def forward(self, inputs=None):
        data = self.graph.to_data()
        with torch.no_grad():
            if inputs is not None:
                data.x[: self.graph.num_input_nodes][:, 0] = (
                    data.x[: self.graph.num_input_nodes][:, 0] * 0.0 + inputs
                )
        nodes = self.compute_propagation(data.x, data.edge_index)
        return nodes[self.graph.output_nodes], nodes

    def batch_forward(self, inputs):
        output = []
        nodes = []
        for inp in inputs:
            batch_output, batch_nodes = self.forward(inp)
            output.append(batch_output)
            nodes.append(batch_nodes)
        return torch.stack(output).squeeze(), torch.stack(nodes).squeeze()
