import numpy as np
from matplotlib import pyplot
import pathlib
import torch
import gymnasium as gym
import networkx as nx
import random
import os


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    Load a checkpoint from disk .
    """
    checkpoint = torch.load(checkpoint_fpath + "model_with_checkpoint.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint["epoch"], checkpoint["train_loss_"], checkpoint["val_loss_"]


def count_parameters(model):
    """
    Return total model of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plots_losses_save(path, train_loss, val_loss, test_loss, loss_function):
    """
    Plot loss and validation loss .

    """
    pyplot.plot(train_loss, label="Training loss")
    pyplot.plot(val_loss, label="Validation loss")
    pyplot.plot(len(val_loss) - 1, [test_loss], marker="*", linestyle="None", markersize=10, color="red", label="Test loss")
    pyplot.text(len(val_loss) - 1, test_loss, str(test_loss)[:8])
    pyplot.xlabel("Epoch", fontsize=12)
    pyplot.ylabel(loss_function + " loss per image (mean)", fontsize=12)
    pyplot.legend(loc="best", prop={"size": 14})
    pyplot.savefig(path + "/losses.pdf", dpi=300)
    pyplot.clf()
    pyplot.close()


def plot_loss_save(path, train_loss, test_loss, loss_function):
    pyplot.plot(train_loss, label="Training loss")
    pyplot.plot(len(train_loss) - 1, [test_loss], marker="*", linestyle="None", markersize=10, color="red", label="Test loss")
    pyplot.xlabel("Epoch", fontsize=12)
    pyplot.ylabel(loss_function + " loss per image (mean)", fontsize=12)
    pyplot.legend(loc="best", prop={"size": 14})
    pyplot.savefig(path + "/losses.pdf", dpi=300)
    pyplot.clf()
    pyplot.close()
    pyplot.plot(
        np.arange(int(0.1 * len(train_loss)), len(train_loss)),
        train_loss[int(0.1 * len(train_loss)) :],
        label="Training loss",
    )
    pyplot.plot(len(train_loss) - 1, [test_loss], marker="*", linestyle="None", markersize=10, color="red", label="Test loss")
    pyplot.xlabel("Epoch", fontsize=12)
    pyplot.ylabel(loss_function + " loss per image (mean)", fontsize=12)
    pyplot.legend(loc="best", prop={"size": 14})
    pyplot.savefig(path + "/losses_without_first_10_percent.pdf", dpi=300)
    pyplot.clf()
    pyplot.close()


def plot_lr(path, lr):
    pyplot.plot(lr, label="Learning rate")
    pyplot.xlabel("Epoch", fontsize=12)
    pyplot.ylabel("Learning rate", fontsize=12)
    pyplot.tight_layout()
    pyplot.savefig(path + "/lr.pdf", dpi=300)
    pyplot.clf()
    pyplot.close()


def dimensions_env(environment):
    """
    Look up observation and action space dimension
    """
    from gymnasium.spaces import Discrete, Box

    env = gym.make(environment)
    if len(env.observation_space.shape) == 3:  # Pixel-based environment
        pixel_env = True
        input_dim = 3
    elif len(env.observation_space.shape) == 1:  # State-based environment
        pixel_env = False
        input_dim = env.observation_space.shape[0]
    elif isinstance(env.observation_space, Discrete):
        pixel_env = False
        input_dim = env.observation_space.n
    else:
        raise ValueError("Observation space not supported")

    if isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]
    elif isinstance(env.action_space, Discrete):
        action_dim = env.action_space.n
    else:
        raise ValueError("Action space not supported")

    return input_dim, action_dim, pixel_env


def x0_sampling(dist, nb_params):
    if dist == "U[0,1]":
        return np.random.rand(nb_params)
    elif dist == "U[-1,1]":
        return 2 * np.random.rand(nb_params) - 1
    elif dist == "N[0,1]":
        return np.random.randn(nb_params)
    else:
        raise ValueError("Distribution not available")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def nx_layout(_G, layout: str):
    if layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(_G)
    elif layout == "planar":
        pos = nx.planar_layout(_G)
    elif layout == "shell":
        pos = nx.shell_layout(_G)
    elif layout == "spring":
        pos = nx.spring_layout(_G)
    elif layout == "spectral":
        pos = nx.spectral_layout(_G)
    elif layout == "random_fixed":
        for node in _G.nodes:
            posXY = np.random.default_rng(node).uniform(-1, +1, (2,))
            _G.nodes[node]["pos"] = [posXY[0], posXY[1]]
        pos = nx.get_node_attributes(_G, "pos")
    elif layout == "diagonal":
        for node in _G.nodes:
            _G.nodes[node]["pos"] = [node, node]
        pos = nx.get_node_attributes(_G, "pos")
    else:
        pos = None

    return pos


def visualise_graph(evolved_parameters, config, plot_title, env_rollout, logtocloud=True):
    from train_backend import fitness_functional
    from celluloid import Camera

    pathlib.Path(config["_path"] + "/graph_animations/").mkdir(parents=True, exist_ok=True)

    fig = pyplot.figure(figsize=(12, 8))
    fig.tight_layout()
    pyplot.box(False)
    camera = Camera(fig)
    config["celluloid_camera"] = camera

    fitness = fitness_functional(config, animate_graph_growth=not env_rollout, animate_graph_rollout=env_rollout, solution_id=plot_title)
    fitness(evolved_parameters)

    animation = camera.animate()
    fps = 16 if env_rollout else 3
    animation.save(config["_path"] + "/graph_animations/" + plot_title + ".mp4", fps=fps)
    pyplot.show()


def extended_neighbors(G, G_nodes=None):
    """
    Find direct and extended neighbors for each node in the graph

    Args:
        G (NetworkX graph): Graph
        G_nodes ([G.nodes], optional): List of nodes for which we want to find their neighbors. Defaults to None.

    Returns:
        [dict]: Dictionary with direct and extended neighbors for each node
    """
    neighbors = {}
    node_list = G.nodes if G_nodes is None else G_nodes
    for node in node_list:
        connections_idx = np.unique([n for n in nx.all_neighbors(G, node)])
        extended_neighbors = np.empty(0, dtype=int)
        for node_extended in connections_idx:
            connections_idx_extended = np.unique([n for n in nx.all_neighbors(G, node_extended)])
            extended_neighbors = np.unique(np.concatenate((extended_neighbors, connections_idx_extended)))
        neighbors[node] = {"direct_neighbors": connections_idx, "extended_neighbors": extended_neighbors}

    return neighbors


def direct_neighbors(G, G_nodes=None):
    """
    Input can be either networkX G.nodes or a list of nodes [1,2,3,4]
    Return the extended dictionary with direct neighbors for each node
    """
    neighbors = {}
    node_list = G.nodes if G_nodes is None else G_nodes
    for node in node_list:
        connections_idx = np.unique([n for n in nx.all_neighbors(G, node)])
        extended_neighbors = np.empty(0, dtype=int)
        neighbors[node] = {"direct_neighbors": connections_idx}

    return neighbors


def seed_python_numpy_torch_cuda(seed: int):
    if seed is None:
        rng = np.random.default_rng()
        seed = int(rng.integers(2**32, size=1))
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def animate_graph(
    G,
    network_state,
    celluloid_camera,
    layout,
    arrows,
    nodes_role_dims,
    font_size=8,
    print_labels=True,
    roullout=False,
    rollout_timestep=None,
    growth_cycle=0,
):
    labels = dict([(i, np.round(vector_embedding, 3)) for i, vector_embedding in enumerate(network_state)]) if print_labels else None
    if nodes_role_dims is not None:
        observation_dim, action_dim = nodes_role_dims
        color_map_nodes = [("indianred" if node < observation_dim else ("slategray" if node >= len(G) - action_dim else "white")) for node in G.nodes()]
    else:
        color_map_nodes = ["white" for node in G.nodes()]
    # Color yellow the node of the action_dim with the highnest activation
    if roullout:
        color_map_nodes[np.argmax(network_state[-action_dim:]) - action_dim] = "orange"

    nx.draw_networkx(
        G,
        edgecolors="black",
        labels=labels,
        font_size=font_size,
        node_size=500,
        node_color=color_map_nodes,
        pos=nx_layout(G, layout),
        arrows=arrows,
        width=[max(0.01, abs(G[u][v]["weight"])) for u, v in G.edges()],
    )

    edges_labels = dict([((u, v), np.round(G[u][v]["weight"], 3)) for u, v in G.edges()])
    nx.draw_networkx_edge_labels(
        G,
        pos=nx_layout(G, layout),
        edge_labels=edges_labels,
        font_color="black",
        font_size=7,
        alpha=0.5,
    )

    if roullout:
        pyplot.text(1, -1.4, "Env timestep " + str(rollout_timestep), fontsize=12)
    else:
        pyplot.text(0.5, -0.7, "Growth cycle: " + str(int(growth_cycle)) + ", Graph size " + str(len(G)), fontsize=12)
    celluloid_camera.snap()


def environment_max_reward(env_name):
    if env_name == "LunarLander-v2":
        return 200
    elif env_name == "CartPole-v1":
        return 500
    elif env_name == "SmallWorldNetwork":
        return 20
    else:
        raise NotImplementedError


if __name__ == "__main__":
    pass
