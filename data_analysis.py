import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Function to extract diversity results from a text file
def extract_diversity(file_path):
    with open(file_path, 'r') as f:
        results = [float(line.split(':')[-1]) for line in f.readlines()]
    return results


# Function to extract fitness results with Best Fitness, Mean Fitness, and Std Dev Fitness
def extract_fitness_extended(file_path):
    best_fitness = []
    mean_fitness = []
    std_fitness = []

    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = line.split(',')
            best_fitness.append(float(parts[0].split(':')[-1]))
            mean_fitness.append(float(parts[1].split(':')[-1]))
            std_fitness.append(float(parts[2].split(':')[-1]))

    return best_fitness, mean_fitness, std_fitness


# Function to process all iterations and generate DataFrames for diversity and fitness
def process_results_extended(base_dir, enemy, iterations=10, generations=50):

    # Initialize
    diversity_data = []
    best_fitness_data = []
    mean_fitness_data = []
    std_fitness_data = []

    # check if base dir contains es or fs
    if 'fs' in base_dir:
        base = "fs"
    elif 'es' in base_dir:
        base = "es"

    # Loop over the iterations (10)
    for i in range(1, iterations + 1):
        # Paths for diversity and fitness results
        diversity_file = os.path.join(base_dir, f'{base}_{i}_enemy{enemy}', 'diversity_results.txt')
        fitness_file = os.path.join(base_dir, f'{base}_{i}_enemy{enemy}', 'fitness_results.txt')


        # Extract results
        diversity_data.append(extract_diversity(diversity_file))

        best_fitness, mean_fitness, std_fitness = extract_fitness_extended(fitness_file)
        best_fitness_data.append(best_fitness)
        mean_fitness_data.append(mean_fitness)
        std_fitness_data.append(std_fitness)


    diversity_df = pd.DataFrame(diversity_data)
    best_fitness_df = pd.DataFrame(best_fitness_data)
    mean_fitness_df = pd.DataFrame(mean_fitness_data)
    std_fitness_df = pd.DataFrame(std_fitness_data)

    return diversity_df, best_fitness_df, mean_fitness_df, std_fitness_df



# plotting all fitness results on the same plot
def plot_fitness_results(best_fitness_df_with, mean_fitness_df_with, best_fitness_df_without, mean_fitness_df_without, enemy):

    generations = best_fitness_df_with.shape[1]

    # Calculate the mean and standard deviation for EA with fitness sharing
    best_fitness_mean_with = best_fitness_df_with.mean(axis=0)
    best_fitness_std_with = best_fitness_df_with.std(axis=0)
    mean_fitness_mean_with = mean_fitness_df_with.mean(axis=0)
    mean_fitness_std_with = mean_fitness_df_with.std(axis=0)

    # Calculate the mean and standard deviation for EA without fitness sharing
    best_fitness_mean_without = best_fitness_df_without.mean(axis=0)
    best_fitness_std_without = best_fitness_df_without.std(axis=0)
    mean_fitness_mean_without = mean_fitness_df_without.mean(axis=0)
    mean_fitness_std_without = mean_fitness_df_without.std(axis=0)

    # x-axis
    gen_range = np.arange(1, generations + 1)

    plt.figure(figsize=(10, 6))
    line_width = 1

    # best fitness with fitness sharing
    plt.plot(gen_range, best_fitness_mean_with, label='Best Fitness (With Sharing)', color='blue', linestyle='-', linewidth=line_width)
    plt.fill_between(gen_range, best_fitness_mean_with - best_fitness_std_with, best_fitness_mean_with + best_fitness_std_with,
                     color='blue', alpha=0.3)

    # mean fitness with fitness sharing
    plt.plot(gen_range, mean_fitness_mean_with, label='Mean Fitness (With Sharing)', color='orange', linestyle='--', linewidth=line_width)
    plt.fill_between(gen_range, mean_fitness_mean_with - mean_fitness_std_with, mean_fitness_mean_with + mean_fitness_std_with,
                     color='orange', alpha=0.3)

    # best fitness without fitness sharing
    plt.plot(gen_range, best_fitness_mean_without, label='Best Fitness (Without Sharing)', color='green', linestyle='-', linewidth=line_width)
    plt.fill_between(gen_range, best_fitness_mean_without - best_fitness_std_without, best_fitness_mean_without + best_fitness_std_without,
                     color='green', alpha=0.3)

    # mean fitness without fitness sharing
    plt.plot(gen_range, mean_fitness_mean_without, label='Mean Fitness (Without Sharing)', color='red', linestyle='--', linewidth=line_width)
    plt.fill_between(gen_range, mean_fitness_mean_without - mean_fitness_std_without, mean_fitness_mean_without + mean_fitness_std_without,
                     color='red', alpha=0.3)

    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Mean and Best Fitness Over Generations with Standard Deviations')
    plt.suptitle(f'Enemy {enemy}')
    plt.legend()
    plt.grid(True)

    # Set x-axis limits to start where the lines start and end at the last generation
    # plt.xlim(gen_range[0], gen_range[-1])

    plt.tight_layout()
    # save the plot
    plt.savefig(f"fitness_enemy{enemy}.png")
    plt.show()


def plot_diversity_results(diversity_df_with, diversity_df_without, enemy):

    generations = diversity_df_with.shape[1]

    # with fs
    diversity_mean_with = diversity_df_with.mean(axis=0)
    diversity_std_with = diversity_df_with.std(axis=0)

    # without fs
    diversity_mean_without = diversity_df_without.mean(axis=0)
    diversity_std_without = diversity_df_without.std(axis=0)

    # x-axis
    gen_range = np.arange(1, generations + 1)

    plt.figure(figsize=(10, 6))

    line_width = 1

    # Diversity with fs
    plt.plot(gen_range, diversity_mean_with, label='Diversity (With Sharing)', color='blue', linewidth=line_width)
    plt.fill_between(gen_range, diversity_mean_with - diversity_std_with, diversity_mean_with + diversity_std_with,
                     color='blue', alpha=0.3)

    # Diversity without fs
    plt.plot(gen_range, diversity_mean_without, label='Diversity (Without Sharing)', color='red', linewidth=line_width)
    plt.fill_between(gen_range, diversity_mean_without - diversity_std_without, diversity_mean_without + diversity_std_without,
                     color='red', alpha=0.3)

    plt.xlabel('Generations')
    plt.ylabel('Diversity')
    plt.title('Diversity Over Generations with Standard Deviations')
    plt.suptitle(f'Enemy {enemy}')
    plt.legend()
    plt.grid(True)

    # Set x-axis limits to start where the lines start and end at the last generation
    # plt.xlim(gen_range[0], gen_range[-1])

    plt.tight_layout()

    plt.savefig(f"diversity_enemy{enemy}.png")
    plt.show()

diversity_df_es, best_fitness_df_es, mean_fitness_df_es, std_fitness_df_es = process_results_extended(base_dir='es_enemy8', enemy=8, iterations=10, generations=50)
diversity_df_fs, best_fitness_df_fs, mean_fitness_df_fs, std_fitness_df_fs = process_results_extended(base_dir='fs_enemy8', enemy=8, iterations=10, generations=50)


plot_fitness_results(best_fitness_df_es, mean_fitness_df_es, best_fitness_df_fs, mean_fitness_df_fs, enemy=8)

plot_diversity_results(diversity_df_es, diversity_df_fs, enemy=8)




