import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Function to extract diversity results from a text file
def extract_diversity(file_path):
    """
    Extracts the genotypic diversity values from the file.

    Args:
    - file_path (str): Path to the diversity_results.txt file.

    Returns:
    - list of float: Diversity values for each generation.
    """
    with open(file_path, 'r') as f:
        results = [float(line.split(':')[-1]) for line in f.readlines()]
    return results


# Function to extract fitness results with Best Fitness, Mean Fitness, and Std Dev Fitness
def extract_fitness_extended(file_path):
    """
    Extracts the best fitness, mean fitness, and standard deviation fitness values from the file.

    Args:
    - file_path (str): Path to the fitness_results.txt file.

    Returns:
    - tuple of lists: Best Fitness, Mean Fitness, and Std Dev Fitness values for each generation.
    """
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
    """
    Processes diversity and fitness results (including mean and std dev) over multiple iterations
    and stores them in DataFrames.

    Args:
    - base_dir (str): Base directory containing iteration folders.
    - iterations (int): Number of iterations.
    - generations (int): Number of generations.

    Returns:
    - pd.DataFrame: DataFrame of diversity results (n_iterations x generations).
    - pd.DataFrame: DataFrame of best fitness results (n_iterations x generations).
    - pd.DataFrame: DataFrame of mean fitness results (n_iterations x generations).
    - pd.DataFrame: DataFrame of std dev fitness results (n_iterations x generations).
    """
    # Initialize lists to store data
    diversity_data = []
    best_fitness_data = []
    mean_fitness_data = []
    std_fitness_data = []

    # check if base dir contains es or fs
    if 'fs' in base_dir:
        base = "fs"
    elif 'es' in base_dir:
        base = "es"

    # Loop over the iterations
    for i in range(1, iterations + 1):
        # Paths for diversity and fitness results
        diversity_file = os.path.join(base_dir, f'{base}_{i}_enemy{enemy}', 'diversity_results.txt')
        fitness_file = os.path.join(base_dir, f'{base}_{i}_enemy{enemy}', 'fitness_results.txt')

        # Extract diversity results
        diversity_data.append(extract_diversity(diversity_file))

        # Extract fitness results (best, mean, std)
        best_fitness, mean_fitness, std_fitness = extract_fitness_extended(fitness_file)
        best_fitness_data.append(best_fitness)
        mean_fitness_data.append(mean_fitness)
        std_fitness_data.append(std_fitness)

    # Convert lists to DataFrames
    diversity_df = pd.DataFrame(diversity_data)
    best_fitness_df = pd.DataFrame(best_fitness_data)
    mean_fitness_df = pd.DataFrame(mean_fitness_data)
    std_fitness_df = pd.DataFrame(std_fitness_data)

    return diversity_df, best_fitness_df, mean_fitness_df, std_fitness_df




def plot_fitness_results(best_fitness_df_with, mean_fitness_df_with, best_fitness_df_without, mean_fitness_df_without, enemy):
    """
    Plots the mean of the best and mean fitness across generations, along with standard deviations,
    for both EA with fitness sharing and EA without fitness sharing.

    Args:
    - best_fitness_df_with (pd.DataFrame): Best fitness results (n_iterations x generations) for EA with fitness sharing.
    - mean_fitness_df_with (pd.DataFrame): Mean fitness results (n_iterations x generations) for EA with fitness sharing.
    - best_fitness_df_without (pd.DataFrame): Best fitness results (n_iterations x generations) for EA without fitness sharing.
    - mean_fitness_df_without (pd.DataFrame): Mean fitness results (n_iterations x generations) for EA without fitness sharing.
    """
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

    # Generate an array for generations
    gen_range = np.arange(1, generations + 1)

    # Plotting
    plt.figure(figsize=(10, 6))

    line_width = 1

    # Plot the best fitness for EA with fitness sharing
    plt.plot(gen_range, best_fitness_mean_with, label='Best Fitness (With Sharing)', color='blue', linestyle='-', linewidth=line_width)
    plt.fill_between(gen_range, best_fitness_mean_with - best_fitness_std_with, best_fitness_mean_with + best_fitness_std_with,
                     color='blue', alpha=0.3)

    # Plot the mean fitness for EA with fitness sharing
    plt.plot(gen_range, mean_fitness_mean_with, label='Mean Fitness (With Sharing)', color='orange', linestyle='--', linewidth=line_width)
    plt.fill_between(gen_range, mean_fitness_mean_with - mean_fitness_std_with, mean_fitness_mean_with + mean_fitness_std_with,
                     color='orange', alpha=0.3)

    # Plot the best fitness for EA without fitness sharing
    plt.plot(gen_range, best_fitness_mean_without, label='Best Fitness (Without Sharing)', color='green', linestyle='-', linewidth=line_width)
    plt.fill_between(gen_range, best_fitness_mean_without - best_fitness_std_without, best_fitness_mean_without + best_fitness_std_without,
                     color='green', alpha=0.3)

    # Plot the mean fitness for EA without fitness sharing
    plt.plot(gen_range, mean_fitness_mean_without, label='Mean Fitness (Without Sharing)', color='red', linestyle='--', linewidth=line_width)
    plt.fill_between(gen_range, mean_fitness_mean_without - mean_fitness_std_without, mean_fitness_mean_without + mean_fitness_std_without,
                     color='red', alpha=0.3)

    # Add labels and title
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Mean and Best Fitness Over Generations with Standard Deviations')
    plt.suptitle(f'Enemy {enemy}')
    plt.legend()
    plt.grid(True)

    # Set x-axis limits to start where the lines start and end at the last generation
    # plt.xlim(gen_range[0], gen_range[-1])

    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_diversity_results(diversity_df_with, diversity_df_without, enemy):
    """
    Plots the mean diversity across generations, along with standard deviations,
    for both EA with fitness sharing and EA without fitness sharing.

    Args:
    - diversity_df_with (pd.DataFrame): Diversity results (n_iterations x generations) for EA with fitness sharing.
    - diversity_df_without (pd.DataFrame): Diversity results (n_iterations x generations) for EA without fitness sharing.
    """
    generations = diversity_df_with.shape[1]

    # Calculate the mean and standard deviation for EA with fitness sharing
    diversity_mean_with = diversity_df_with.mean(axis=0)
    diversity_std_with = diversity_df_with.std(axis=0)

    # Calculate the mean and standard deviation for EA without fitness sharing
    diversity_mean_without = diversity_df_without.mean(axis=0)
    diversity_std_without = diversity_df_without.std(axis=0)

    # Generate an array for generations
    gen_range = np.arange(1, generations + 1)

    # Plotting
    plt.figure(figsize=(10, 6))

    line_width = 1

    # Plot the diversity for EA with fitness sharing
    plt.plot(gen_range, diversity_mean_with, label='Diversity (With Sharing)', color='blue', linewidth=line_width)
    plt.fill_between(gen_range, diversity_mean_with - diversity_std_with, diversity_mean_with + diversity_std_with,
                     color='blue', alpha=0.3)

    # Plot the diversity for EA without fitness sharing
    plt.plot(gen_range, diversity_mean_without, label='Diversity (Without Sharing)', color='red', linewidth=line_width)
    plt.fill_between(gen_range, diversity_mean_without - diversity_std_without, diversity_mean_without + diversity_std_without,
                     color='red', alpha=0.3)

    # Add labels and title
    plt.xlabel('Generations')
    plt.ylabel('Diversity')
    plt.title('Diversity Over Generations with Standard Deviations')
    plt.suptitle(f'Enemy {enemy}')
    plt.legend()
    plt.grid(True)

    # Set x-axis limits to start where the lines start and end at the last generation
    # plt.xlim(gen_range[0], gen_range[-1])

    plt.tight_layout()
    # Show the plot
    plt.show()

diversity_df_es, best_fitness_df_es, mean_fitness_df_es, std_fitness_df_es = process_results_extended(base_dir='es_enemy3', enemy=3, iterations=10, generations=50)
diversity_df_fs, best_fitness_df_fs, mean_fitness_df_fs, std_fitness_df_fs = process_results_extended(base_dir='fs_enemy3', enemy=3, iterations=10, generations=50)


plot_fitness_results(best_fitness_df_es, mean_fitness_df_es, best_fitness_df_fs, mean_fitness_df_fs, enemy=3)
plot_diversity_results(diversity_df_es, diversity_df_fs, enemy=3)



