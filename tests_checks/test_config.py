import pytest
import yaml
from operator import xor


def only_one_true(*args):
    return sum(args) == 1


def test_embedding_shared_graph(config):
    a = config["shared_intial_embedding"]
    b = config["shared_intial_graph_bool"]
    if a:
        assert b


def test_embedding_shared_graph(config):
    a = config["extra_nodes"]
    b = config["initial_sparsity"]
    if a == -1:
        assert b == 1


def random_seed_and_reval(config):
    a = config["nb_episode_evals"]
    b = config["env_seed"]
    if a > 1 and b is not None:
        raise ValueError("If you want to re-evaluate the solution, you must set env_seed to None")


def growth_mode(config):
    a = config["node_based_growth"]
    b = config["node_pairs_based_growth"]
    c = config["edge_based_growth"]
    c = config["probabilistic_growth"]
    assert only_one_true(a, b, c)


def smallworldnet(config):
    if config["environment"] == "SmallWorldNet":
        assert config["persistent_observation_rollout"] == False


def smallworldnetBinary(config):
    if config["environment"] == "SmallWorldNet":
        assert config["binary_connectivity"] == True


def refavals(config):
    a = config["evolution_feval_check_N"]
    b = config["nb_episode_evals"]
    assert a % b == 0


def coevolve_initial_embeddings(config):
    if config["coevolve_initial_embeddings"]:
        assert config["initial_embeddings_random"] == False


def coevolve_initial_embeddings_2(config):
    if config["coevolve_initial_embeddings"]:
        if config["extra_nodes"] != -1:
            raise NotImplementedError("coevolve_initial_embeddings is not implemented for extra_nodes != -1")


def all_config_checks(config):
    growth_mode(config)
    random_seed_and_reval(config)
    test_embedding_shared_graph(config)
    smallworldnet(config)
    smallworldnetBinary(config)
    refavals(config)
    coevolve_initial_embeddings(config)
    coevolve_initial_embeddings_2(config)
