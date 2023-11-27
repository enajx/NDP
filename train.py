import pathlib
import time
import yaml
import numpy as np
import warnings
import torch

from train_backend import train_model
from utils import seed_python_numpy_torch_cuda, visualise_graph

from tests_checks.test_config import all_config_checks

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(suppress=True)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64)


def train(config):
    pathlib.Path(config["_path"]).mkdir(parents=True, exist_ok=False)

    ########
    # Seed #
    ########
    config["seed"] = np.random.randint(10**7) if config["seed"] is None else config["seed"]
    seed_python_numpy_torch_cuda(config["seed"])
    print("\nSeed: ", config["seed"])
    print("Env_seed: ", config["env_seed"])

    ########################
    ### Launch traininig ###
    ########################
    solution_best, solution_centroid, early_stopping_executed, logger_df = train_model(config)

    if not early_stopping_executed:
        # Save results
        print(f"\nSaving models and config file — Run ID {config['id']}")
        print(f"\nFinal model has {config['nb_trainable_parameters']} parameters")
        if config["save_model"]:
            # Save solution and rewards
            np.save(path + "/" + "solution_centroid", solution_centroid)
            np.save(path + "/" + "solution_best", solution_best)
            if logger_df is not None:
                logger_df.to_csv(path + "/" + "logger.csv")

        # Save config file
        with open(config["_path"] + "/" + "config.yml", "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

        # Visaulise graph development
        if config["visualise_network"]:
            config["nb_episode_evals"] = 1
            config["nb_growth_evals"] = 1
            print(f"\nGenerating graph visualisations...")
            pathlib.Path(config["_path"] + "/graph_animations/").mkdir(parents=True, exist_ok=False)
            print(f"Generating growth visualisations")
            visualise_graph(solution_best, config, "Graph development — Best solution", env_rollout=False)
            visualise_graph(solution_centroid, config, "Graph development — Centroid solution", env_rollout=False)
            if "Network" not in config["environment"]:
                print(f"Generating rollout visualisations")
                visualise_graph(solution_best, config, "Information propagation during rollout — Best solution", env_rollout=True)
                visualise_graph(solution_centroid, config, "Information propagation during rollout — Centroid solution", env_rollout=True)

        # Render
        if config["render"]:
            from train_backend import fitness_functional

            pathlib.Path(config["_path"] + "/env_renders/").mkdir(parents=True, exist_ok=False)
            config["nb_episode_evals"] = 1
            config["nb_growth_evals"] = 3
            print(f"\nRunning environment for best solution...")
            fitness = fitness_functional(config, render=True, solution_id="best")
            fitness(solution_best)
            print(f"\nRunning environment for centroid solution...")
            print(f"\nRunning environment for centroid solution...")
            fitness = fitness_functional(config, render=True, solution_id="centroid")

        print(f"\nBYE! - ID: {config['id']}")

    else:
        print(f"\nEARLY STOPPING EXECUTED\nNothing will remain, bye!\n")


if __name__ == "__main__":
    # Load configuration file
    import argparse

    parser = argparse.ArgumentParser(description="Configuration file path")
    parser.add_argument("--conf", type=str, default="run_experiment.yaml", metavar="", help="Path to yaml configuration file")
    args = parser.parse_args()
    with open(args.conf) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Check config file makes sense
    all_config_checks(config)

    # Create local path to save results locally if not resuming training from checkpoint
    config["id"] = str(int(time.time()))
    print(f"Model ID: {config['id']}")
    path = "saved_models" + "/" + str(config["id"])
    config["_path"] = path

    print(f"\nConfig:\n{config}")

    # Launch training
    train(config)
