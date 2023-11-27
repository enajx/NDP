import numpy as np
import yaml

from train_backend import fitness_functional
from utils import seed_python_numpy_torch_cuda, visualise_graph

if __name__ == "__main__":
    # Load configuration file
    import argparse

    parser = argparse.ArgumentParser(description="id training")
    parser.add_argument("--id", type=str, default="1701101416", metavar="", help="Run id, e.g. 1701100614")
    parser.add_argument("--rollouts", type=int, default=1, metavar="", help="Number of evaluation rollouts")
    parser.add_argument(
        "--random_seed",
        type=bool,
        default=0,
        metavar="",
        help="If true, it uses a new seed different from training. This makes initial embeddings random at each evaluation if shared option was set to False.",
    )
    parser.add_argument("--generate_animations", type=bool, default=1, metavar="", help="Generate graph animations")
    parser.add_argument("--layout", type=str, default="random_fixed", metavar="", help="Layout: random_fixed shell, spectral, spring, kamada_kawai, planar")
    args = parser.parse_args()

    path = "saved_models/" + args.id

    with open(path + "/config.yml") as file:
        config = yaml.load(file, Loader=yaml.Loader)

    if args.random_seed:
        print(f"\nUsing random seed for evaluation")
        config["seed"] = None
    seed_python_numpy_torch_cuda(config["seed"])
    print("\nSeed: ", config["seed"])
    print("Env seed: ", config["env_seed"])

    solution_best = np.load(path + "/" + "solution_best.npy")
    config["layout"] = args.layout
    config["nb_episode_evals"] = 1
    config["nb_growth_evals"] = args.rollouts
    fitness = fitness_functional(config=config, render=True, solution_id="best")
    fitness(solution_best)
    if args.generate_animations:
        print(f"\nGenerating growth visualisations")
        visualise_graph(solution_best, config, "REVAL: Graph development — Best solution", env_rollout=False, logtocloud=False)
        if "Network" not in config["environment"]:
            print(f"\nGenerating rollout visualisations")
            visualise_graph(solution_best, config, "REVAL: Information propagation during rollout — Best solution", env_rollout=True, logtocloud=False)
    print(f"\nSumcheck best: {np.sum(solution_best)}")
