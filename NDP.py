import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import networkx as nx
from scipy import stats, sparse
from numpy.random import default_rng

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_grad_enabled(False)


def MLP(input_dim, output_dim, hidden_layers_dims, activation, last_layer_activated, bias):
    """This function creates a multi-layer perceptron.

    Args:
        input_dim (int): The input dimension of the MLP.
        output_dim (int): The output dimension of the MLP.
        hidden_layers_dims (list): A list of integers that represent the number of hidden layers and their dimensions.
        growth_model_last_layer_activated (bool): Whether the last layer should be activated.
        activation (torch.nn.functional, optional): The activation function to use. Defaults to tanh.

    Returns:
        torch.nn.Module: The MLP.
    """
    layers = []
    layers.append(torch.nn.Linear(input_dim, hidden_layers_dims[0], bias=bias))
    layers.append(activation)
    for i in range(1, len(hidden_layers_dims)):
        layers.append(torch.nn.Linear(hidden_layers_dims[i - 1], hidden_layers_dims[i], bias=bias))
        layers.append(activation)
    layers.append(torch.nn.Linear(hidden_layers_dims[-1], output_dim, bias=bias))
    if last_layer_activated:
        layers.append(activation)
    return torch.nn.Sequential(*layers)


# Generate intial graph random matrix
def generate_initial_graph(network_size, sparsity, binary_connectivity, undirected, seed):
    """This function generates a random initial graph.

    Args:
        network_size (int): The intial number of nodes in the network.
        sparsity (float): The initial sparsity of the network.
        binary_connectivity (bool): Whether the network has binary weights.
        undirected (bool): Whether the network is undirected.

    Returns:
        G: networkx.Graph: The initial graph
        W: np.array: The initial adjacency matrix.
    """
    nb_disjoint_initial_graphs = np.inf
    while nb_disjoint_initial_graphs > 1:
        maxWeight = 1
        minWeight = -1
        rng = default_rng(seed)
        if binary_connectivity:
            rvs = stats.uniform(loc=0, scale=1).rvs
            W = np.rint(sparse.random(network_size, network_size, density=sparsity, data_rvs=rvs, random_state=rng).toarray())
        else:
            rvs = stats.uniform(loc=minWeight, scale=maxWeight - minWeight).rvs
            W = sparse.random(network_size, network_size, density=sparsity, data_rvs=rvs, random_state=rng).toarray()  # rows are outbounds, columns are inbounds
        disjoint_initial_graphs = [e for e in nx.connected_components(nx.from_numpy_array(W))]
        nb_disjoint_initial_graphs = len(disjoint_initial_graphs)

    if undirected:
        G = nx.from_numpy_array(W, create_using=nx.Graph)
    else:
        G = nx.from_numpy_array(W, create_using=nx.DiGraph)

    return G


def propagate_features(
    network_state: np.array,
    W: np.array,
    network_thinking_time: int,
    recurrent_activation_function: str,
    additive_update: bool,
    persistent_observation: np.array = None,
    feature_transformation_model=None,
):
    """
    Propagate the network state through the network.

    Args:
        network_state (np.array): vector of network state, i.e. the features of the nodes / node embeddings
        W (np.array): adjacency matrix
        network_thinking_time (int): number of steps to propagate the network state
        recurrent_activation_function (str)): activation function to use for network state propagation
        additive_update (bool): whether to use additive update or not
        persistent_observation (np.array) : observation array

    Returns:
        np.array: the updated network state
    """
    with torch.no_grad():
        network_state = torch.tensor(network_state, dtype=torch.float64)
        persistent_observation = torch.tensor(persistent_observation, dtype=torch.float64) if persistent_observation is not None else None
        W = torch.tensor(W, dtype=torch.float64)

        if recurrent_activation_function is None:
            activation = None
        elif recurrent_activation_function == "tanh":
            activation = np.tanh
        else:
            raise ValueError("Activation function not available.")

        for step in range(network_thinking_time):
            if additive_update:
                network_state += W.T @ network_state
            else:
                network_state = W.T @ network_state

            if feature_transformation_model is not None:
                network_state = feature_transformation_model(network_state)
            elif activation is not None:
                network_state = activation(network_state)

            if persistent_observation is not None:
                network_state[: persistent_observation.shape[0]] = persistent_observation

        return network_state.detach().numpy()


def query_pairs_of_node_embeddings(W: np.array, network_state: np.array, self_link_allowed: bool = False):
    """This function returns a dictionary (and its array version) of the concatenated node embedd1ings for every node pair.

    Args:
        W (np.array): The adjacency matrix.
        network_state (np.array): The network state.
        self_link_allowed (bool, optional): whether the output array/dict contains concatenated embedding of nodes with self-links. Defaults to False.

    Returns:
        node_embeddings_concatenated_dict (dict): A dictionary of the concatenated node embeddings for every node pair.
        node_embeddings_concatenated_array (np.array): An array of the concatenated node embeddings for every node pair.
    """
    node_embeddings_concatenated_dict = {}
    idx = np.arange(len(W))

    # Make every node-node pair appear only once
    W = abs(W)
    links = np.clip((np.tril(W) + np.triu(W).T), 0, 1)
    if not self_link_allowed:
        np.fill_diagonal(links, 0)

    # print(f'Number of pair-wise undirected links: {np.sum(links)}')
    for i in range(len(W)):
        nbr = links[i] > 0  # mask of neighbors

        for j in idx[nbr]:
            concatenated_features = np.concatenate([network_state[i], network_state[j]])
            node_embeddings_concatenated_dict[len(node_embeddings_concatenated_dict)] = {
                "from_node": i,
                "to_node": j,
                "concatenated_features": concatenated_features,
            }

    return node_embeddings_concatenated_dict, np.array([node_embeddings_concatenated_dict[e]["concatenated_features"] for e in node_embeddings_concatenated_dict])


def predict_new_nodes(growth_decision_model, embeddings_for_growth_model, node_embedding_size):
    """This function predicts the new nodes based on the concatenated node embeddings.

    Args:
        growth_decision_model (torch.nn.Module): The model deciding whether two nodes should be add a new node.
        node_embeddings_concatenated_array (np.array): An array of the concatenated node embeddings for every node pair.
        node_embedding_size (int): The size of the node embeddings.

    Returns:
        new_nodes_predictions (np.array): An array of the predictions of the growth decision model.
        # predictions_probabilities (np.array): An array of the probabilities of the growth decision model.
    """
    new_nodes_predictions = []
    with torch.no_grad():
        predictions_probabilities = growth_decision_model(torch.tensor(embeddings_for_growth_model, dtype=torch.float64)).detach().numpy()
        new_nodes_predictions = (predictions_probabilities > 0).squeeze()

    return new_nodes_predictions


def update_weights(G, network_state, model, config):
    """This function updates the weights of the edges based on the network state.

    Args:
        G (nx.Graph): The graph.
        network_state (np.array): The network state.
        model (torch.nn.Module): The model predicting the weights of the edges.
        config (dict): The configuration dictionary.

    Returns:
        nx.Graph: The graph with updated weights.
    """

    with torch.no_grad():
        for i, j in G.edges():
            G[i][j]["weight"] = model(torch.tensor(np.concatenate([network_state[i], network_state[j]]), dtype=torch.float64)).detach().numpy()[0]

    return G


def add_new_nodes(
    G,
    config,
    network_state,
    node_embeddings_concatenated_dict,
    new_nodes_predictions,
    node_based_growth,
    node_pairs_based_growth,
):
    """

    Args:
        G (nx.Graph): The graph.
        network_state (np.array): The network state.
        node_embeddings_concatenated_dict (dict): A dictionary of the concatenated node embeddings for every node pair.
        new_nodes_predictions (np.array): An array of the predictions of the growth decision model.
        # predicted_weights  (np.array): An array of the predicted weights of the new edges.
        binary_connectivity (bool): Whether the graph weights are binary or not.
        draw_graphs (bool, optional): Whether to draw the graphs or not. Defaults to False.
        animate_graph (bool, optional): Whether to animate the graph or not. Defaults to False.
    """

    current_graph_size = len(G)
    if node_pairs_based_growth:
        for idx_edge in node_embeddings_concatenated_dict:
            if new_nodes_predictions[idx_edge]:
                target_connections = (
                    node_embeddings_concatenated_dict[idx_edge]["from_node"],
                    node_embeddings_concatenated_dict[idx_edge]["to_node"],
                )

                # Find current neighbors — used to update the new node embedding
                neighbors = np.unique(np.concatenate([[n for n in nx.all_neighbors(G, target_connections[0])], [n for n in nx.all_neighbors(G, target_connections[1])]]))

                # Add new node and edges
                G.add_node(current_graph_size)
                G.add_edge(target_connections[0], current_graph_size, weight=1 if config["binary_connectivity"] else 0)
                G.add_edge(current_graph_size, target_connections[1], weight=1 if config["binary_connectivity"] else 0)

                # Expand node embeddings
                network_state = np.concatenate([network_state, np.expand_dims(np.mean(network_state[neighbors], axis=0), axis=0)])

                current_graph_size += 1

    elif node_based_growth:
        # Find current neighbors at the beginning of this growth cycle
        if len(G) == 1:
            neighbors = np.array([[0]])
        else:
            neighbors = []
            for idx_node in range(len(G)):
                neighbors_idx = [n for n in nx.all_neighbors(G, idx_node)]
                neighbors_idx.append(idx_node)
                neighbors.append(np.unique(neighbors_idx))

        # Add new nodes and unweighted edges
        for idx_node in range(len(G)):
            if new_nodes_predictions.shape == ():
                new_nodes_predictions = new_nodes_predictions.reshape(1)

            if new_nodes_predictions[idx_node]:
                if len(neighbors) != 0:
                    # Add new node
                    G.add_node(current_graph_size)
                    for neighbor in neighbors[idx_node]:
                        if nx.is_directed(G):
                            G.add_edge(neighbor, current_graph_size, weight=1)
                            G.add_edge(current_graph_size, neighbor, weight=1)
                        else:
                            G.add_edge(neighbor, current_graph_size, weight=1)
                    current_graph_size += 1

                    # Expand node embeddings
                    network_state = np.concatenate([network_state, np.expand_dims(np.mean(network_state[neighbors[idx_node]], axis=0), axis=0)])

    return G, network_state
