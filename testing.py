import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from joblib import Parallel, delayed
import shutil
import itertools
import csv

from evoman.environment import Environment
from demo_controller import player_controller

from scipy.stats import wilcoxon




def load_final_solution(experiment_name, suffix=""):
    solution_path = os.path.join(experiment_name, f'best_solution{suffix}.txt')

    try:
        with open(solution_path, 'r') as file_aux:
            lines = file_aux.readlines()

            # Combine the lines of the best solution into a single string, excluding "Best Solution:" prefix
            best_solution_lines = []
            reading_solution = False
            for line in lines:
                if "Best Solution:" in line:
                    best_solution_lines.append(line.split(":", 1)[1].strip())  # Start reading after 'Best Solution:'
                    reading_solution = True
                elif reading_solution and "Best Fitness:" not in line:
                    best_solution_lines.append(line.strip())  # Continue reading the solution part
                elif "Best Fitness:" in line:
                    break  # Stop reading when reaching 'Best Fitness:'

            best_solution_str = ' '.join(best_solution_lines)
            best_solution = np.fromstring(best_solution_str.strip('[]'), sep=' ')
            # Extract the best fitness and convert to float
            for line in lines:
                if "Best Fitness:" in line:
                    best_fitness = float(line.split(":", 1)[1].strip())
                    break
            print(f"Best solution loaded successfully from {solution_path}")
            return best_solution, best_fitness

    except FileNotFoundError:
        print(f"File not found: {solution_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return None, None






def setup_environment(experiment_name, controller, enemies, multiplemode="yes", visuals=False) -> Environment:
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    return Environment(
        experiment_name=experiment_name,
        playermode="ai",
        enemies=enemies,
        player_controller=controller,
        speed="normal",
        enemymode="static",
        multiplemode=multiplemode,
        level=2,
        visuals=visuals,
    )


exp_name = "island_random_best_10_50_[[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], [1, 4, 7]]_True"

controller = player_controller(10)

best_solution, best_fitness = load_final_solution(exp_name, suffix="_outside_loop")

env = setup_environment("test", controller, [1, 2, 3, 4, 5, 6, 7, 8], visuals=True)

# f, p, e, t = env.play(pcont=best_solution)

print("last gen")
# print(f"Fitness: {f}, Player life: {p}, Enemy life: {e}, Time: {t}")


best_solution_outside_loop,best_fitness_outside_loop = load_final_solution(exp_name, suffix="_outside_loop")
f, p, e, t = env.play(pcont=best_solution_outside_loop)

print("outside loop")
print(f"Fitness: {f}, Player life: {p}, Enemy life: {e}, Time: {t}")
