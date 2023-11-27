import numpy as np
import psutil
import time
from utils import x0_sampling
import cma
import pandas as pd
from multiprocess import Pool


def CMAES(config, fitness):
    nb_parameters = config["nb_trainable_parameters"]
    x0 = x0_sampling(config["x0_dist"], nb_parameters)
    es = cma.CMAEvolutionStrategy(
        x0,
        config["sigma_init"],
        {
            "verb_disp": config["print_every"],
            "popsize": config["popsize"],
            "maxiter": config["generations"],
            "seed": config["seed"],
            "CMA_elitist": config["CMA_elitist"],
            "minstd": config["minstd"],
        },
    )

    print("\n.......................................................")
    print("\nInitilisating CMA-ES with", nb_parameters, "trainable parameters \n")

    print("\n ♪┏(°.°)┛┗(°.°)┓ Starting Evolution ┗(°.°)┛┏(°.°)┓ ♪ \n")
    tic = time.time()

    objective_solution_best = np.Inf
    objective_solution_centroid = np.Inf
    objectives_centroid = []
    objectives_best = []
    gen = 0
    early_stopping_executed = False

    # Physical cores in the machine
    num_cores = psutil.cpu_count(logical=False) if config["threads"] == -1 else config["threads"]
    print(f"\nUsing {num_cores} cores\n")

    # Optimisation loop
    while not es.stop() or gen < config["generations"]:
        try:
            # Generate candidate solutions
            X = es.ask()

            # Evaluate in parallel
            if num_cores > 1:
                with Pool(num_cores) as pool:
                    fitvals = pool.map_async(fitness, X).get()
            else:
                fitvals = [fitness(x) for x in X]

            # Correct sign of fevals needed cause this CMA implementation only minimises
            if config["maximise"]:
                fitvals = [-fitval for fitval in fitvals]

            # Inform CMA optimizer of fitness results
            es.tell(X, fitvals)

            if gen % config["print_every"] == 0:
                es.disp()

            # Store best solution
            objective_current_best_sol = es.best.f
            objectives_best.append(objective_current_best_sol)
            if objective_current_best_sol <= objective_solution_best:
                objective_solution_best = objective_current_best_sol
                solution_best = es.best.x

            # Store best mean solution
            objective_current_centroid_sol = es.fit.fit.mean()
            objectives_centroid.append(objective_current_centroid_sol)
            if objective_current_centroid_sol <= objective_solution_centroid:
                objective_solution_centroid = objective_current_centroid_sol
                solution_centroid = es.mean

            gen += 1

            if gen % config["evolution_feval_check_every"] == 0:
                test_fevals = []
                print("\n" + 30 * "v")
                for _ in range(config["evolution_feval_check_N"] // config["nb_episode_evals"]):
                    checksum = True if _ == 0 else False
                    # feval = fitness(solution_best, config, render=False, solution_id="best_check_"+str(config['evolution_feval_check_N']), checksum=checksum)
                    feval = fitness(solution_best)
                    test_fevals.append(feval)
                print(f"\nEvaluated {config['nb_episode_evals']*(config['evolution_feval_check_N']//config['nb_episode_evals'])} times the best solution found so far. Mean: {np.mean(test_fevals)}")
                print("\n" + 30 * "^" + "\n")

        # Allows to interrupt optimation with Ctrl+C
        except KeyboardInterrupt:  # Only works with python mp
            time.sleep(5)
            print("\n" + 20 * "*")
            print(f"\nCaught Ctrl+C!\nStopping evolution\n")
            print(20 * "*" + "\n")
            break

        # Early stopping of evolution
        if config["early_stopping"]:
            if gen == config["early_stopping_conditions"]["generation"] and objective_current_best_sol > config["early_stopping_conditions"]["objective_value"]:
                print(f"\nObjective too high {objective_current_best_sol} (reward too low) at generation {gen}.\nUnpromising run! Stopping evolution.\n")
                print(20 * "*")
                early_stopping_executed = True
                break

        # # Stopping evolution if loss flattening
        if config["flattening_stopping"] and gen > config["flattening_stopping_conditions"]["min_generation"]:
            std_objective = np.std(objectives_best[: -config["flattening_stopping_conditions"]["last_generations"]])
            if std_objective < config["flattening_stopping_conditions"]["min_std"]:
                print(f"\nObjective flattening!\nStd last {config['flattening_stopping_conditions']['last_generations']} generation is: {std_objective}\nStopping evolution.\n")
                print(20 * "*")
                break

    # losses/Loss arrays
    objectives_centroid = np.array(objectives_centroid)
    objectives_best = np.array(objectives_best)
    # solution = es.result # unsed since ask&tell

    toc = time.time()
    config["training time"] = str(int(toc - tic)) + " seconds"
    print("\nEvolution took: ", int(toc - tic), " seconds\n")
    print(f"========Optimizer output:==========================")
    print(f"Best single loss found was {objective_solution_best}")
    print(f"Best population centroid loss found was {objective_solution_centroid}")
    print(f"===================================================\n")

    # Create dataframe for logging objective values
    logger_pd = pd.DataFrame({"pop_best_eval": objectives_best, "mean_eval": objectives_centroid})

    return solution_best, solution_centroid, early_stopping_executed, logger_pd
