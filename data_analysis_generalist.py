import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller
from joblib import Parallel, delayed

from scipy.stats import wilcoxon




def extract_category(folder_name):
    """
    Extracts the category from the folder name based on the enemies and method used.
    """
    if "random_best" in folder_name:
        if "[1, 5, 7]" in folder_name:
            return "random_best_1_5_7"
        elif "[1, 2, 3, 4, 5, 6, 7, 8]" in folder_name:
            return "random_best_1_8"
    elif "diversity" in folder_name:
        if "[1, 5, 7]" in folder_name:
            return "diversity_1_5_7"
        elif "[1, 2, 3, 4, 5, 6, 7, 8]" in folder_name:
            return "diversity_1_8"
    return "unknown"


def read_results(file_path):
    """
    Reads the results_island_es.txt file and extracts Best, Mean, and SD fitness.
    """
    best_fitness = []
    mean_fitness = []
    sd_fitness = []
    generations = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            if "Generation" in line:
                # Parse the fitness values
                parts = line.strip().split(", ")
                gen = parts[0].split(":")[0]
                gen = gen.split(" ")[-1]

                generations.append(int(gen))
                best_fitness.append(float(parts[0].split(":")[-1]))
                mean_fitness.append(float(parts[1].split(":")[-1]))
                sd_fitness.append(float(parts[2].split(":")[-1]))


    return generations, best_fitness, mean_fitness, sd_fitness
def categorize_and_aggregate(root_folder, label):
    """
    Goes through each folder in the root directory, reads the results, and aggregates them into separate dataframes
    for Best, Mean, and Standard Deviation fitness. Each row represents a generation, and each column represents a run.
    """
    best_fitness_data = []
    mean_fitness_data = []
    sd_fitness_data = []

    # Walk through folders
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            category = extract_category(folder)
            if category == label:
                print(f"Processing folder: {folder}")
                results_file = os.path.join(folder_path, "results_island_es.txt")
                generations, best_fitness, mean_fitness, sd_fitness = read_results(results_file)
                best_fitness_data.append(best_fitness)
                mean_fitness_data.append(mean_fitness)
                sd_fitness_data.append(sd_fitness)

    # Create dataframes
    best_fitness_df = pd.DataFrame(best_fitness_data).T
    mean_fitness_df = pd.DataFrame(mean_fitness_data).T
    sd_fitness_df = pd.DataFrame(sd_fitness_data).T
    return best_fitness_df, mean_fitness_df, sd_fitness_df


# function that plots best and mean fitness over generations
def plot_fitness_over_generations(best_df, mean_df, title):
    mean_over_runs_best = best_df.mean(axis=1)
    mean_over_runs_mean = mean_df.mean(axis=1)

    # get the sd of the best fitness and mean fitness
    sd_over_runs_best = best_df.std(axis=1)
    sd_over_runs_mean = mean_df.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_over_runs_best, label="Best Fitness")
    plt.plot(mean_over_runs_mean, label="Mean Fitness", linestyle='--')
    plt.fill_between(mean_over_runs_best.index, mean_over_runs_best - sd_over_runs_best, mean_over_runs_best + sd_over_runs_best, alpha=0.2)
    plt.fill_between(mean_over_runs_mean.index, mean_over_runs_mean - sd_over_runs_mean, mean_over_runs_mean + sd_over_runs_mean, alpha=0.2)
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.ylim(0, 100)
    plt.xlim(0, 200)
    plt.legend()
    plt.show()


def load_final_solutions(root_directory, label, suffix=""):
    """
    Goes through the root directory, finds experiments that match the specified label, and loads the best solutions.
    """
    best_solutions = []
    best_fitnesses = []

    # Walk through the folders in the root directory
    for folder in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder)
        if os.path.isdir(folder_path):
            # Check if the folder matches the desired category
            category = extract_category(folder)
            if category == label:
                # Construct the path to the best solution file
                solution_path = os.path.join(folder_path, f'best_solution{suffix}.txt')

                try:
                    with open(solution_path, 'r') as file_aux:
                        lines = file_aux.readlines()

                        # Combine the lines of the best solution into a single string, excluding "Best Solution:" prefix
                        best_solution_lines = []
                        reading_solution = False
                        for line in lines:
                            if "Best Solution:" in line:
                                best_solution_lines.append(
                                    line.split(":", 1)[1].strip())  # Start reading after 'Best Solution:'
                                reading_solution = True
                            elif reading_solution and "Best Fitness:" not in line:
                                best_solution_lines.append(line.strip())  # Continue reading the solution part
                            elif "Best Fitness:" in line:
                                break  # Stop reading when reaching 'Best Fitness:'

                        best_solution_str = ' '.join(best_solution_lines)
                        best_solution = np.fromstring(best_solution_str.strip('[]'), sep=' ')

                        # Extract the best fitness and convert to float
                        best_fitness = None
                        for line in lines:
                            if "Best Fitness:" in line:
                                best_fitness = float(line.split(":", 1)[1].strip())
                                break

                        if best_solution is not None and best_fitness is not None:
                            best_solutions.append(best_solution)
                            best_fitnesses.append(best_fitness)
                            print(f"Best solution loaded successfully from {solution_path}")

                except FileNotFoundError:
                    print(f"File not found: {solution_path}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")

    return best_solutions, best_fitnesses






def setup_environment(experiment_name, controller, enemies, multiplemode="yes", visuals=False) -> Environment:
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    return Environment(
        experiment_name=experiment_name,
        playermode="ai",
        enemies=enemies,
        player_controller=controller,
        speed="fastest",
        enemymode="static",
        multiplemode=multiplemode,
        level=2,
        visuals=visuals,
    )




def test_solutions(solutions_list):
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    controller = player_controller(10)
    env = setup_environment("test", controller, [1, 2, 3, 4, 5, 6, 7, 8], visuals=False)

    gains = []
    player_lives = []
    enemy_lives = []
    times = []
    fitnesses = []
    for solution in solutions_list:
        gains_current = []
        player_lives_current = []
        enemy_lives_current = []
        times_current = []
        fitnesses_current = []
        for i in range(5):
            f, p, e, t = env.play(pcont=solution)
            gains_current.append(p - e)
            player_lives_current.append(p)
            enemy_lives_current.append(e)
            times_current.append(t)
            fitnesses_current.append(f)
        gains.append(np.mean(gains_current))
        player_lives.append(np.mean(player_lives_current))
        enemy_lives.append(np.mean(enemy_lives_current))
        times.append(np.mean(times_current))
        fitnesses.append(np.mean(fitnesses_current))


    print(f"Gains: {gains}")
    return gains



# function that box plots 4 lists of gains on the same subplot
def plot_gains(gains_ea1_enemy1, gains_ea1_enemy2, gains_ea2_enemy1, gains_ea2_enemy2, title):
    plt.figure(figsize=(10, 6))
    plt.boxplot([gains_ea1_enemy1, gains_ea1_enemy2, gains_ea2_enemy1, gains_ea2_enemy2])
    plt.title(title)
    plt.xticks([1, 2, 3, 4], ["EA1 [1,2,3,4,5,6,7,8]", "EA1 [1,5,7]", "EA2 [1,2,3,4,5,6,7,8]", "EA2 [1,5,7]"])
    plt.ylabel("Gains")
    plt.ylim(-100, 100)

    # Perform Wilcoxon signed-rank test if significantly different from 0
    for i, gains in enumerate([gains_ea1_enemy1, gains_ea1_enemy2, gains_ea2_enemy1, gains_ea2_enemy2]):
        w, p = wilcoxon(gains, alternative='two-sided')
        if p < 0.05:
            plt.text(i + 1, 90, f"p={p:.3f}", ha='center', va='center', backgroundcolor='white', color='red')
        else:
            plt.text(i + 1, 90, f"p={p:.3f}", ha='center', va='center', backgroundcolor='white')
    plt.show()




# Example usage:
root_folder = "final_runs_generalist"
random_best_1_8_best, random_best_1_8_mean, random_best_1_8_sd = categorize_and_aggregate(root_folder, "random_best_1_8")
random_best_1_5_7_best, random_best_1_5_7_mean, random_best_1_5_7_sd = categorize_and_aggregate(root_folder, "random_best_1_5_7")
diversity_1_8_best, diversity_1_8_mean, diversity_1_8_sd = categorize_and_aggregate(root_folder, "diversity_1_8")
diversity_1_5_7_best, diversity_1_5_7_mean, diversity_1_5_7_sd = categorize_and_aggregate(root_folder, "diversity_1_5_7")

# Plot the fitness over generations
plot_fitness_over_generations(random_best_1_8_best, random_best_1_8_mean, "Random Best 1-8")
plot_fitness_over_generations(random_best_1_5_7_best, random_best_1_5_7_mean, "Random Best 1-5-7")
plot_fitness_over_generations(diversity_1_8_best, diversity_1_8_mean, "Diversity 1-8")
plot_fitness_over_generations(diversity_1_5_7_best, diversity_1_5_7_mean, "Diversity 1-5-7")

###################### GAINS ######################
# ---------------------------- EA1
best_solutions_random_best_1_8_loop, best_fitnesses_random_best_1_8_loop = load_final_solutions(root_folder, "random_best_1_8", suffix="_outside_loop")
best_solutions_random_best_1_8_all, best_fitnesses_random_best_1_8_all = load_final_solutions(root_folder, "random_best_1_8", suffix="_all")
best_solutions_random_best_1_8_subset, best_fitnesses_random_best_1_8_subset = load_final_solutions(root_folder, "random_best_1_8", suffix="")

best_solutions_random_best_1_5_7_loop, best_fitnesses_random_best_1_5_7_loop = load_final_solutions(root_folder, "random_best_1_5_7", suffix="_outside_loop")
best_solutions_random_best_1_5_7_all, best_fitnesses_random_best_1_5_7_all = load_final_solutions(root_folder, "random_best_1_5_7", suffix="_all")
best_solutions_random_best_1_5_7_subset, best_fitnesses_random_best_1_5_7_subset = load_final_solutions(root_folder, "random_best_1_5_7", suffix="")

# ---------------------------- EA2
best_solutions_diversity_1_8_loop, best_fitnesses_diversity_1_8_loop = load_final_solutions(root_folder, "diversity_1_8", suffix="_outside_loop")
best_solutions_diversity_1_8_all, best_fitnesses_diversity_1_8_all = load_final_solutions(root_folder, "diversity_1_8", suffix="_all")
best_solutions_diversity_1_8_subset, best_fitnesses_diversity_1_8_subset = load_final_solutions(root_folder, "diversity_1_8", suffix="")

best_solutions_diversity_1_5_7_loop, best_fitnesses_diversity_1_5_7_loop = load_final_solutions(root_folder, "diversity_1_5_7", suffix="_outside_loop")
best_solutions_diversity_1_5_7_all, best_fitnesses_diversity_1_5_7_all = load_final_solutions(root_folder, "diversity_1_5_7", suffix="_all")
best_solutions_diversity_1_5_7_subset, best_fitnesses_diversity_1_5_7_subset = load_final_solutions(root_folder, "diversity_1_5_7", suffix="")



# Test the best solutions
gains_random_best_1_8_loop = test_solutions(best_solutions_random_best_1_8_loop)
gains_random_best_1_5_7_loop = test_solutions(best_solutions_random_best_1_5_7_loop)
gains_diversity_1_8_loop = test_solutions(best_solutions_diversity_1_8_loop)
gains_diversity_1_5_7_loop = test_solutions(best_solutions_diversity_1_5_7_loop)

# Plot the gains
plot_gains(gains_random_best_1_8_loop, gains_random_best_1_5_7_loop, gains_diversity_1_8_loop, gains_diversity_1_5_7_loop, "Gains From Soulutions Kept Outside Loop")


gains_diversity_1_5_7_subset = test_solutions(best_solutions_diversity_1_5_7_subset)
gains_diversity_1_8_subset = test_solutions(best_solutions_diversity_1_8_subset)
gains_random_best_1_5_7_subset = test_solutions(best_solutions_random_best_1_5_7_subset)
gains_random_best_1_8_subset = test_solutions(best_solutions_random_best_1_8_subset)

plot_gains(gains_random_best_1_8_subset, gains_random_best_1_5_7_subset, gains_diversity_1_8_subset, gains_diversity_1_5_7_subset, "Gains From Best Solutions")


gains_diversity_1_5_7_all = test_solutions(best_solutions_diversity_1_5_7_all)
gains_diversity_1_8_all = test_solutions(best_solutions_diversity_1_8_all)
gains_random_best_1_5_7_all = test_solutions(best_solutions_random_best_1_5_7_all)
gains_random_best_1_8_all = test_solutions(best_solutions_random_best_1_8_all)

plot_gains(gains_random_best_1_8_all, gains_random_best_1_5_7_all, gains_diversity_1_8_all, gains_diversity_1_5_7_all, "Gains From All Solutions")