import copy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torch_geometric.data import Data

from growing_nn.graph.generated_network import GeneratedNetwork


class DirectedGraph:
    def __init__(self, nodes, edge_dict, num_input_nodes, num_output_nodes):
        self.nodes = nodes
        self.edge_dict = edge_dict
        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes

        node_indices = np.arange(self.nodes.size(0))
        self.input_nodes = list(node_indices[: self.num_input_nodes])
        self.output_nodes = list(
            node_indices[
                self.num_input_nodes : self.num_input_nodes + self.num_output_nodes
            ]
        )

    def add_edges(self, new_edges):
        for node in new_edges:
            if node not in self.edge_dict:
                self.edge_dict[node] = []
            destinations = new_edges[node]
            for d in destinations:
                if d not in self.edge_dict[node]:
                    self.edge_dict[node].append(d)

    def add_nodes(self, nodes):
        self.nodes = torch.vstack((self.nodes, nodes))

    def to_data(self):
        edges = []

        for node in self.edge_dict:
            destinations = self.edge_dict[node]
            for d in destinations:
                edges.append([node, d])

        edges = torch.tensor(edges).long().t().contiguous().to(self.nodes.device)
        return Data(
            x=self.nodes * torch.ones(self.nodes.size(), device=self.nodes.device),
            edge_index=edges,
        )

    # A recursive function used by topologicalSort
    def topologicalSortUtil(self, v, visited, stack):

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        if v in self.edge_dict:
            for i in self.edge_dict[v]:
                if not visited[i]:
                    self.topologicalSortUtil(i, visited, stack)

        # Push current vertex to stack which stores result
        stack.append(v)

    # The function to do Topological Sort. It uses recursive
    # topologicalSortUtil()
    def topological_sort(self):
        # Mark all the vertices as not visited
        self.V = self.nodes.size(0)
        visited = [False] * self.V
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(self.V):
            if not visited[i]:
                self.topologicalSortUtil(i, visited, stack)

        # Print contents of the stack
        return stack[::-1]

    def plot(self, labels=None, fig=None, node_colors=None):
        data = self.to_data()
        G = torch_geometric.utils.to_networkx(data, to_undirected=False)

        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=LR")

        if fig is None:
            fig = plt.figure()
        canvas = FigureCanvas(fig)

        if node_colors is None:
            node_colors = ["blue"] * self.nodes.size(0)
            for i in self.input_nodes:
                node_colors[i] = "green"
            for i in self.output_nodes:
                node_colors[i] = "red"

        nx.draw_networkx_nodes(G, pos, node_color=node_colors)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos, labels=labels)

        canvas.draw()  # draw the canvas, cache the renderer

        image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image

    def generate_network(self, *args, **kwargs):
        return GeneratedNetwork(self, *args, **kwargs)

    def copy(self):
        nodes = self.nodes * torch.ones(self.nodes.size(), device=self.nodes.device)
        edge_dict = copy.deepcopy(self.edge_dict)
        return DirectedGraph(
            nodes, edge_dict, self.num_input_nodes, self.num_output_nodes
        )
