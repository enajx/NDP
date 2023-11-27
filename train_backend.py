import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import networkx as nx
from scipy import stats, sparse
from numpy.random import default_rng
from matplotlib import pyplot
import copy
import time
import powerlaw
from scipy.stats import kstest
from typing import List, Tuple, Dict, Union, Optional, Callable

from NDP import *
from optimizers import CMAES
from utils import dimensions_env, animate_graph, seed_python_numpy_torch_cuda, environment_max_reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_grad_enabled(False)


def env_rollout(G: nx.graph, config: dict, render=False, animate_graph_rollout: bool = False, solution_id: str = None, seed: int = None) -> float:
    if animate_graph_rollout:
        graph = copy.deepcopy(G)

    try:
        diameter = nx.diameter(G.to_undirected())
    except:
        diameter = int(np.sqrt(len(G)))
        print(f"WARNING: Graph is not connected due to prunning. Diameter manually set to {diameter}.")
    policy_connectivity = nx.to_numpy_array(G)

    # Instatiate the environment
    if render:
        env = gym.make(config["environment"], render_mode="rgb_array")
        env = RecordVideo(env=env, video_folder=config["_path"] + "/env_renders/" + solution_id + str(time.time()))
    else:
        env = gym.make(config["environment"])

    # Resize and normilise input for pixel environments
    if config["pixel_env"] == True:
        raise NotImplementedError

    observation, info = env.reset(seed=seed, options={})
    done = False
    episodeReward = 0
    network_state = np.zeros(policy_connectivity.shape[0])
    network_thinking_time = diameter + config["network_thinking_time_extra_rollout"]
    timestep = 0
    while not done:
        # For obaservation ∈ gym.spaces.Discrete, we one-hot encode the observation
        if isinstance(env.observation_space, gym.spaces.Discrete):
            observation = (observation == torch.arange(env.observation_space.n)).float()
        # Swap axes to the correct order for pytorch
        if config["pixel_env"]:
            raise NotImplementedError

        # Visualise the network's information flow
        if animate_graph_rollout:
            animate_graph(
                G=graph,
                network_state=network_state,
                celluloid_camera=config["celluloid_camera"],
                layout=config["layout"],
                arrows=config["arrows"],
                nodes_role_dims=(config["observation_dim"], config["action_dim"]),
                font_size=8,
                print_labels=True,
                roullout=True,
                rollout_timestep=timestep,
            )

        # Represent observation as a feature vector (node embedding) of the network
        network_state[: config["observation_dim"]] = observation
        persistent_observation = observation if config["persistent_observation_rollout"] else None

        # Let the network update its internal state
        network_state = propagate_features(
            network_state=network_state,
            W=policy_connectivity,
            network_thinking_time=network_thinking_time,
            recurrent_activation_function=config["recurrent_activation_function"],
            additive_update=config["additive_update"],
            persistent_observation=persistent_observation,
            feature_transformation_model=None,
        )

        # Select action from the output nodes
        action = network_state[-config["action_dim"] :]

        # Bound the action or convert it to a discrete action
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        elif isinstance(env.action_space, gym.spaces.Discrete):
            action = np.argmax(action)

        # Forward step in the envionrment
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if "Bullet" in config["environment"]:
            reward = env.unwrapped.rewards[1]  # Distance walked
        # Save reward
        episodeReward += reward

        # Render the environment
        if render:
            env.render()

        timestep += 1

    # print(episodeReward)
    return episodeReward


def fitness_functional(config: dict, render=False, animate_graph_growth=False, animate_graph_rollout=False, solution_id=None, checksum=False) -> Callable[np.ndarray, float]:  # type: ignore
    def fitness(evolved_parameters: np.array) -> float:
        """
        Evaluate an agent 'evolved_parameters' in an environment 'environment' during a lifetime.
        Returns the negative episodic fitness of the agent.
        """

        # To average out the growth process stochasticity
        mean_reward = 0
        for _ in range(config["nb_growth_evals"]):
            if config["nb_growth_evals"] > 1:
                config["seed"] = None

            seed_python_numpy_torch_cuda(config["seed"])

            # Define networks
            if config["shared_intial_graph_bool"]:
                G = config["shared_intial_graph"]
            else:
                G = generate_initial_graph(config["initial_network_size"], config["initial_sparsity"], config["binary_connectivity"], config["undirected"], seed=None)

            # Initialise the network state
            if config["coevolve_initial_embeddings"]:
                initial_network_state = np.expand_dims(evolved_parameters[: config["node_embedding_size"]], axis=0)
            elif (not config["shared_intial_embedding"]) and config["initial_embeddings_random"]:
                initial_network_state = np.random.default_rng(None).uniform(-1, +1, (config["initial_network_size"], config["node_embedding_size"]))
            elif config["shared_intial_embedding"] and config["initial_embeddings_random"]:
                initial_network_state = config["initial_network_state"]
            else:
                initial_network_state = np.ones((config["initial_network_size"], config["node_embedding_size"]))

            # Create the growth-decision network
            mlp_growth_model = MLP(
                input_dim=config["input_size_growth_model"],
                output_dim=1,
                hidden_layers_dims=config["mlp_growth_hidden_layers_dims"],
                last_layer_activated=config["growth_model_last_layer_activated"],
                activation=torch.nn.Tanh(),
                bias=config["growth_model_bias"],
            )

            # Create the network that transforms node embeddings
            if config["NN_transform_node_embedding_during_growth"]:
                mlp_feature_transformation = MLP(
                    input_dim=config["node_embedding_size"],
                    output_dim=config["node_embedding_size"],
                    hidden_layers_dims=config["mlp_embedding_transform_hidden_layers_dims"],
                    last_layer_activated=config["transform_model_last_layer_activated"],
                    activation=torch.nn.Tanh(),
                    bias=config["transform_model_bias"],
                )

            # Create the network that dermines weights based on pair of node embeddings for node-based-growth with non-binary connectivity
            if config["node_based_growth"] and not config["binary_connectivity"]:
                mlp_weight_values = MLP(
                    input_dim=2 * config["node_embedding_size"],
                    output_dim=1,
                    hidden_layers_dims=config["mlp_weight_values_hidden_layers_dims"],
                    last_layer_activated=config["mlp_weight_values_last_layer_activated"],
                    activation=torch.nn.Tanh(),
                    bias=config["mlp_weight_values_bias"],
                )

            # Create indeces for unpaacking the evolved parameters
            n1 = config["node_embedding_size"] if config["coevolve_initial_embeddings"] else 0
            n2 = n1 + config["nb_params_growth_model"]
            n3 = n2 + config["nb_params_feature_transformation"]
            n4 = n3 + config["nb_params_mlp_weight_values"]

            # Load evolved weights into the growth-decision network
            torch.nn.utils.vector_to_parameters(torch.tensor(evolved_parameters[n1:n2], dtype=torch.float64, requires_grad=False), mlp_growth_model.parameters())

            # Load evolved weights into network that updates the node embeddings
            if config["NN_transform_node_embedding_during_growth"]:
                torch.nn.utils.vector_to_parameters(
                    torch.tensor(
                        evolved_parameters[n2:n3],
                        dtype=torch.float64,
                        requires_grad=False,
                    ),
                    mlp_feature_transformation.parameters(),
                )

            # Load evolved weights into network that dermines weights based on pair of node embeddings for node-based-growth with non-binary connectivity
            if config["node_based_growth"] and not config["binary_connectivity"]:
                torch.nn.utils.vector_to_parameters(
                    torch.tensor(
                        evolved_parameters[n3:n4],
                        dtype=torch.float64,
                        requires_grad=False,
                    ),
                    mlp_weight_values.parameters(),
                )

            network_state = copy.deepcopy(initial_network_state)
            obs_action_dim_tuple = None if "Network" in config["environment"] else (2, 2) if "gate" in config["environment"] else (config["observation_dim"], config["action_dim"])

            if render or checksum:
                ns = np.sum(network_state)
                gs = np.sum(nx.to_numpy_array(G))
                print(f"\nChecksum of initial network state before growth: {ns}")
                print(f"Checksum of graph G before growth: {gs}")
                print(f"The final grown graph has {len(G)} nodes and {len(G.edges)} edges.")
                config["checksum_best_State_grown"] = ns
                config["checksum_best_Graph_grown"] = gs

            for growth_cycle_nb in range(config["number_of_growth_cycles"]):
                # Draw the graph

                if animate_graph_growth and growth_cycle_nb == 0:
                    animate_graph(
                        G=G,
                        network_state=network_state,
                        celluloid_camera=config["celluloid_camera"],
                        layout=config["layout"],
                        arrows=config["arrows"],
                        nodes_role_dims=obs_action_dim_tuple,
                        print_labels=True,
                        growth_cycle=growth_cycle_nb,
                    )

                # Define network thinking time based on its current diameter
                try:
                    diameter = nx.diameter(G.to_undirected())
                except:
                    diameter = int(np.sqrt(len(G)))
                    print(f"\nWARNING: Graph is not connected due to prunning. Diameter manually set to {diameter}.")
                network_thinking_time = diameter + config["network_thinking_time_extra_growth"]

                # Local propagation of node features — i.e thinking time
                network_state = propagate_features(
                    network_state=network_state,
                    W=nx.to_numpy_array(G),
                    network_thinking_time=network_thinking_time,
                    recurrent_activation_function=config["recurrent_activation_function"],
                    additive_update=config["additive_update"],
                    persistent_observation=None,
                    feature_transformation_model=mlp_feature_transformation if config["NN_transform_node_embedding_during_growth"] else None,
                )

                if animate_graph_growth:
                    animate_graph(
                        G=G,
                        network_state=network_state,
                        celluloid_camera=config["celluloid_camera"],
                        layout=config["layout"],
                        arrows=config["arrows"],
                        nodes_role_dims=obs_action_dim_tuple,
                        print_labels=True,
                        growth_cycle=growth_cycle_nb,
                    )

                # Query node/edges embeddings
                if config["node_pairs_based_growth"]:
                    # Query pairs of node embeddings
                    node_embeddings_concatenated_dict, embeddings_for_growth_model = query_pairs_of_node_embeddings(
                        W=nx.to_numpy_array(G),
                        network_state=network_state,
                        self_link_allowed=config["self_link_allowed_during_querying"],
                    )
                elif config["node_based_growth"]:
                    embeddings_for_growth_model = network_state
                    node_embeddings_concatenated_dict = None
                elif config["edge_based_growth"]:
                    raise NotImplementedError

                # Predict new nodes
                new_nodes_predictions = predict_new_nodes(mlp_growth_model, embeddings_for_growth_model, config["node_embedding_size"])

                # Add new nodes and increase the network_state vector accordingly
                G, network_state = add_new_nodes(
                    G=copy.deepcopy(G),
                    config=config,
                    network_state=network_state,
                    node_embeddings_concatenated_dict=node_embeddings_concatenated_dict,
                    new_nodes_predictions=new_nodes_predictions,
                    node_based_growth=config["node_based_growth"],
                    node_pairs_based_growth=config["node_pairs_based_growth"],
                )

                if animate_graph_growth:
                    animate_graph(
                        G=G,
                        network_state=network_state,
                        celluloid_camera=config["celluloid_camera"],
                        layout=config["layout"],
                        arrows=config["arrows"],
                        nodes_role_dims=obs_action_dim_tuple,
                        print_labels=True,
                        growth_cycle=growth_cycle_nb,
                    )

                if len(G) > 1 and not config["binary_connectivity"]:
                    G = update_weights(G=G, network_state=network_state, model=mlp_weight_values, config=config)
                    if animate_graph_growth:
                        animate_graph(
                            G=G,
                            network_state=network_state,
                            celluloid_camera=config["celluloid_camera"],
                            layout=config["layout"],
                            arrows=config["arrows"],
                            nodes_role_dims=obs_action_dim_tuple,
                            print_labels=True,
                            growth_cycle=growth_cycle_nb,
                        )

                if config["prunning_phase"]:
                    edges_to_be_removed = [(a, b) for a, b, attrs in G.edges(data=True) if abs(attrs["weight"]) <= config["prunning_threshold"]]
                    G.remove_edges_from(edges_to_be_removed)
                    if animate_graph_growth:
                        animate_graph(
                            G=G,
                            network_state=network_state,
                            celluloid_camera=config["celluloid_camera"],
                            layout=config["layout"],
                            arrows=config["arrows"],
                            nodes_role_dims=obs_action_dim_tuple,
                            print_labels=True,
                            growth_cycle=growth_cycle_nb,
                        )

            if render or checksum:
                ns = np.sum(network_state)
                gs = np.sum(nx.to_numpy_array(G))
                print(f"Checksum of network state after growth: {ns}")
                print(f"Checksum of graph G after growth: {gs}")
                print(f"The final grown graph has {len(G)} nodes and {len(G.edges)} edges.")
                config["checksum_best_State_grown"] = ns
                config["checksum_best_Graph_grown"] = gs

            if animate_graph_growth:
                figWeights, axWeights = pyplot.subplots(1, 1, figsize=(12, 8))
                axWeights.set_title("Weights distributions")
                axWeights.hist(np.array([G[i][j]["weight"] for (i, j) in G.edges()]), bins=20)
                figWeights.savefig(config["_path"] + "/weights_" + solution_id + ".png")

            # Run the environment
            if not animate_graph_growth:  # Here we don't want to render the environment if we are just animating the graph growth when calling visualise_graph()
                # If the graph is too small, we don't want to evaluate it and return bad score directly
                if len(G) < config["min_network_size"]:
                    if render:
                        print("\nNetwork too small")
                    if config["maximise"]:
                        return len(G) - config["min_network_size"]
                    else:
                        return config["min_network_size"] - len(G)

                mean_episode_reward = 0
                for _ in range(config["nb_episode_evals"]):
                    if config["environment"] == "SmallWorldNetwork":
                        episode_reward = small_world_ness_fitness(G=G, niter=5, nrand=10, seed=config["seed"], sigma=config["sigma"], omega=config["omega"], render=render)
                    elif "gate" in config["environment"]:
                        episode_reward = bool_gates_fitness(G=G, config=config, render=render, animate_graph_rollout=animate_graph_rollout)
                    elif config["environment"] == "ScaleFreeNetwork":
                        episode_reward = scalefree_fitness(G=G, ks_test=config["ks_test"], render=render)
                    else:
                        seed_env_eval = int(np.random.default_rng(config["env_seed"]).integers(2**32, size=1)[0])
                        animate_graph_rollout_ = True if (_ == 0 and animate_graph_rollout) else False
                        episode_reward = env_rollout(G=G, config=config, render=render, animate_graph_rollout=animate_graph_rollout_, solution_id=solution_id, seed=seed_env_eval)
                    mean_episode_reward += episode_reward

                if render:
                    print(f"\n(mean) Episode reward for {config['nb_episode_evals']} env rollouts: {mean_episode_reward / config['nb_episode_evals']}")

                mean_reward += mean_episode_reward / config["nb_episode_evals"]

        mean_reward /= config["nb_growth_evals"]
        if render:
            print(f"\n-------\n\n(mean) Episodes reward for {config['nb_episode_evals']*config['nb_growth_evals']} runs: {mean_reward}")
            print("\n---------------------------------------------\n")

        if config["fewer_edges"]:
            # print(f'Current reward: {mean_reward}')
            env_max_reward = environment_max_reward(config["environment"])
            sparsity_penalty = (len(G.edges) / len(G) ** 2) * (env_max_reward)
            mean_reward -= sparsity_penalty

        if config["fewer_nodes"]:
            env_max_reward = environment_max_reward(config["environment"])
            nb_nodes = len(G)
            nb_nodes_penalty = 10 * nb_nodes * (env_max_reward)
            mean_reward -= nb_nodes_penalty

        if config["balanced_weights"]:
            env_max_reward = environment_max_reward(config["environment"])
            mean_weights = abs(np.array([G[i][j]["weight"] for (i, j) in G.edges()]).mean())
            unbalance_penalty = mean_weights * env_max_reward
            mean_reward -= unbalance_penalty

        return mean_reward

    return fitness


def bool_gates_fitness(G: nx.Graph, config: dict, render=False, animate_graph_rollout: bool = False):
    """
    Returns a scalar measure of how many elements of the boolean gate truth table are correctly predicted by the graph G.
    """
    # Truth table
    if config["environment"] == "XOR_gate":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        Y = np.array([0, 1, 1, 0])
    elif config["environment"] == "NAND_gate":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        Y = np.array([1, 1, 1, 0])
    elif config["environment"] == "AND_gate":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        Y = np.array([0, 0, 0, 1])
    elif config["environment"] == "OR_gate":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        Y = np.array([0, 1, 1, 1])
    elif config["environment"] == "NOR_gate":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        Y = np.array([1, 0, 0, 0])
    elif config["environment"] == "XNOR_gate":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        Y = np.array([1, 0, 0, 1])

    if animate_graph_rollout:
        graph = copy.deepcopy(G)

    try:
        diameter = nx.diameter(G.to_undirected())
    except:
        diameter = int(np.sqrt(len(G)))
        print(f"WARNING: Graph is not connected due to prunning. Diameter manually set to {diameter}.")
    policy_connectivity = nx.to_numpy_array(G)

    network_state = np.zeros(policy_connectivity.shape[0])
    network_thinking_time = diameter + config["network_thinking_time_extra_rollout"]
    timestep = 0
    fitness = 0
    for idx, x in enumerate(X):
        # Visualise the network's information flow
        if animate_graph_rollout:
            animate_graph(
                G=graph,
                network_state=network_state,
                celluloid_camera=config["celluloid_camera"],
                layout=config["layout"],
                arrows=config["arrows"],
                nodes_role_dims=(2, 2),
                font_size=8,
                print_labels=True,
                roullout=True,
                rollout_timestep=timestep,
            )

        # Represent observation as a feature vector (node embedding) of the network
        network_state[:2] = x
        persistent_observation = x if config["persistent_observation_rollout"] else None

        # Let the network update its internal state
        network_state = propagate_features(
            network_state=network_state,
            W=policy_connectivity,
            network_thinking_time=network_thinking_time,
            recurrent_activation_function=config["recurrent_activation_function"],
            additive_update=config["additive_update"],
            persistent_observation=persistent_observation,
            feature_transformation_model=None,
        )

        # Select action from the output nodes
        bool_prediction = np.argmax(network_state[-2:])

        if bool_prediction == Y[idx]:
            fitness += 1

        timestep += 1

    if render:
        print(f"{config['environment']} gate fitness: {fitness}")
    return fitness


def small_world_ness_fitness(G: nx.Graph, niter=5, nrand=10, seed=None, sigma=True, omega=False, render=False):
    """
    High fitness value means the graph has small-worldness.
    Returns a scalar measure of small-world-ness of a graph G.
    Omega ought to be close to zero and/or sigma > 1
    """
    if sigma:
        sigma_ = nx.sigma(G, niter=niter, nrand=nrand, seed=seed)
    else:
        sigma_ = 0
    if omega:
        abs_omega_inv_ = 1 / abs(nx.omega(G, niter=niter, nrand=nrand, seed=seed))
    else:
        abs_omega_inv_ = 0

    if render:
        s = nx.sigma(G, niter=niter, nrand=nrand, seed=seed)
        o = nx.omega(G, niter=niter, nrand=nrand, seed=seed)
        print(f"\nSigma (smallworldness if > 1): {s}")
        print(f"Omega (smallworldness if ≈ 0): {o}")

    return abs_omega_inv_ + sigma_


def scalefree_fitness(G: nx.Graph, ks_test=False, render=False):
    """
    High fitness value means the graph is scale-free.
    If ks_test:
        Returns 1 minus Kolmogorov-Smirnov test statistic for powerlaw [0,1]
    else:
        Returns loglikelihood ratio R, and its p-value p a graph G being scale-free. [-inf, inf]
        G is scale-free if R > 0 and p < 0.05.
    """

    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    fit = powerlaw.Fit(degree_sequence, xmin=1)

    if render:
        pyplot.figure(figsize=(10, 6))
        fig1 = fit.plot_pdf(color="b", linewidth=2, label="data")
        fit.power_law.plot_pdf(color="g", linestyle="--", ax=fig1, label="powerlaw")
        pyplot.legend()

        pyplot.figure(figsize=(10, 6))
        fig2 = fit.plot_ccdf(linewidth=3, color="black", label="data distribution")
        fit.power_law.plot_ccdf(ax=fig2, color="red", linestyle="--", label="powerlaw")
        fit.lognormal.plot_ccdf(ax=fig2, color="green", linestyle="--", label="lognormal")
        fit.stretched_exponential.plot_ccdf(ax=fig2, color="blue", linestyle="--", label="stretched_exponential")
        fit.exponential.plot_ccdf(ax=fig2, color="pink", linestyle="--", label="exponential")
        pyplot.legend()
        pyplot.show()

    if ks_test:
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        test, p = kstest(degree_sequence, "powerlaw", args=(alpha, xmin), N=len(degree_sequence))
        return 1 - test
    else:
        # returns the cummulative sums of  loglikelihood ratio between each pair of distribution fits
        R1, p1 = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)
        R2, p2 = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
        R3, p3 = fit.distribution_compare("power_law", "stretched_exponential", normalized_ratio=True)
        R4, p4 = fit.distribution_compare("power_law", "lognormal_positive", normalized_ratio=True)
        return R1 + R2 + R3 + R4


def train_model(config):
    if "Network" in config["environment"]:
        config["min_network_size"] = config["min_size_grownNetwork"]
        if config["extra_nodes"] == -1:
            config["initial_network_size"] = 1
        else:
            config["initial_network_size"] = config["extra_nodes"]
    elif "gate" in config["environment"]:
        config["min_network_size"] = 4
        if config["extra_nodes"] == -1:
            config["initial_network_size"] = 1
        else:
            config["initial_network_size"] = config["extra_nodes"]
    else:
        # Figure out environment dimennsions and type
        observation_dim, action_dim, pixel_env = dimensions_env(config["environment"])
        config["observation_dim"] = observation_dim
        config["action_dim"] = action_dim
        config["pixel_env"] = pixel_env
        if pixel_env:
            raise NotImplementedError
        else:
            if config["extra_nodes"] == -1:
                config["initial_network_size"] = 1
            else:
                config["initial_network_size"] = observation_dim + action_dim + config["extra_nodes"]
        config["min_network_size"] = observation_dim + action_dim

    # Find number of trainable parameters
    config["nb_params_coevolve_initial_embeddings"] = config["node_embedding_size"] if config["coevolve_initial_embeddings"] else 0

    config["input_size_growth_model"] = config["node_embedding_size"] * 2 if config["node_pairs_based_growth"] else config["node_embedding_size"]
    mlp_growth_model = MLP(
        input_dim=config["input_size_growth_model"],
        output_dim=1,
        hidden_layers_dims=config["mlp_growth_hidden_layers_dims"],
        last_layer_activated=config["growth_model_last_layer_activated"],
        activation=torch.nn.Tanh(),
        bias=config["growth_model_bias"],
    )
    config["nb_params_growth_model"] = torch.nn.utils.parameters_to_vector(mlp_growth_model.parameters()).detach().numpy().shape[0]

    if config["NN_transform_node_embedding_during_growth"]:
        mlp_feature_transformation = MLP(
            input_dim=config["node_embedding_size"],
            output_dim=config["node_embedding_size"],
            hidden_layers_dims=config["mlp_embedding_transform_hidden_layers_dims"],
            last_layer_activated=config["transform_model_last_layer_activated"],
            activation=torch.nn.Tanh(),
            bias=config["transform_model_bias"],
        )
        config["nb_params_feature_transformation"] = torch.nn.utils.parameters_to_vector(mlp_feature_transformation.parameters()).detach().numpy().shape[0]
    else:
        config["nb_params_feature_transformation"] = 0

    if config["node_based_growth"] and not config["binary_connectivity"]:
        output_dim_mlp = 1 if config["undirected"] else 2
        mlp_weight_values = MLP(
            input_dim=2 * config["node_embedding_size"],
            output_dim=1,
            hidden_layers_dims=config["mlp_weight_values_hidden_layers_dims"],
            last_layer_activated=config["mlp_weight_values_last_layer_activated"],
            activation=torch.nn.Tanh(),
            bias=config["mlp_weight_values_bias"],
        )
        config["nb_params_mlp_weight_values"] = torch.nn.utils.parameters_to_vector(mlp_weight_values.parameters()).detach().numpy().shape[0]
    else:
        config["nb_params_mlp_weight_values"] = 0

    config["nb_trainable_parameters"] = (
        config["nb_params_coevolve_initial_embeddings"] + config["nb_params_growth_model"] + config["nb_params_feature_transformation"] + config["nb_params_mlp_weight_values"]
    )
    print(f"The growth model has {config['nb_trainable_parameters']} trainable parameters")

    # Generate initial graph
    if config["shared_intial_graph_bool"]:
        config["shared_intial_graph"] = generate_initial_graph(config["initial_network_size"], config["initial_sparsity"], config["binary_connectivity"], config["undirected"], config["seed"])

    # Generate initial node embedding
    if not config["coevolve_initial_embeddings"] and config["shared_intial_embedding"] and config["initial_embeddings_random"]:
        config["initial_network_state"] = np.random.default_rng(config["seed"]).uniform(-1, +1, (config["initial_network_size"], config["node_embedding_size"]))

    fitness = fitness_functional(config)

    # Run optimiser
    if config["optimizer"] == "CMAES":
        solution_best, solution_centroid, early_stopping_executed, logger = CMAES(config, fitness)
    else:
        raise NotImplementedError

    return solution_best, solution_centroid, early_stopping_executed, logger


if __name__ == "__main__":
    pass
