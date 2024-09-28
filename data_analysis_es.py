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

    # Loop over the iterations
    for i in range(1, iterations + 1):
        # Paths for diversity and fitness results
        diversity_file = os.path.join(base_dir, f'es_{i}_enemy{enemy}', 'diversity_results.txt')
        fitness_file = os.path.join(base_dir, f'es_{i}_enemy{enemy}', 'fitness_results.txt')

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

# Example usage:
diversity_df, best_fitness_df, mean_fitness_df, std_fitness_df = process_results_extended(base_dir='es_enemy8', enemy=8, iterations=10, generations=50)
print(diversity_df)
print(best_fitness_df)
print(mean_fitness_df)
print(std_fitness_df)



def plot_fitness_results(best_fitness_df, mean_fitness_df, enemy, algorithm):
    """
    Plots the mean of the best and mean fitness across generations, along with standard deviations.
    The mean is plotted as a dashed line, and the best is plotted as a solid line.

    Args:
    - best_fitness_df (pd.DataFrame): DataFrame containing best fitness results (n_iterations x generations).
    - mean_fitness_df (pd.DataFrame): DataFrame containing mean fitness results (n_iterations x generations).
    """
    generations = best_fitness_df.shape[1]

    # Calculate the mean and standard deviation across the iterations (axis=0 means along columns)
    best_fitness_mean = best_fitness_df.mean(axis=0)
    best_fitness_std = best_fitness_df.std(axis=0)

    mean_fitness_mean = mean_fitness_df.mean(axis=0)
    mean_fitness_std = mean_fitness_df.std(axis=0)

    # Generate an array for generations
    gen_range = np.arange(1, generations + 1)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot the best fitness
    plt.plot(gen_range, best_fitness_mean, label='Best Fitness', color='blue', linestyle='-', linewidth=2)
    plt.fill_between(gen_range, best_fitness_mean - best_fitness_std, best_fitness_mean + best_fitness_std,
                     color='blue', alpha=0.3)

    # Plot the mean fitness
    plt.plot(gen_range, mean_fitness_mean, label='Mean Fitness', color='orange', linestyle='--', linewidth=2)
    plt.fill_between(gen_range, mean_fitness_mean - mean_fitness_std, mean_fitness_mean + mean_fitness_std,
                     color='orange', alpha=0.3)

    # Add labels and title
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Mean and Best Fitness Over Generations with Standard Deviations')
    plt.suptitle(f'Enemy {enemy} - {algorithm}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_diversity_results(diversity_df, enemy, algorithm):
    """
    Plots the mean of the best and mean fitness across generations, along with standard deviations.
    The mean is plotted as a dashed line, and the best is plotted as a solid line.

    Args:
    - best_fitness_df (pd.DataFrame): DataFrame containing best fitness results (n_iterations x generations).
    - mean_fitness_df (pd.DataFrame): DataFrame containing mean fitness results (n_iterations x generations).
    """
    generations = diversity_df.shape[1]

    diversity_mean = diversity_df.mean(axis=0)
    diversity_std = diversity_df.std(axis=0)

    # Generate an array for generations
    gen_range = np.arange(1, generations + 1)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot the best fitness
    plt.plot(gen_range, diversity_mean, label='Diversity', color='red', linewidth=2)
    plt.fill_between(gen_range, diversity_mean - diversity_std, diversity_mean + diversity_std,
                     color='blue', alpha=0.3)

    # Add labels and title
    plt.xlabel('Generations')
    plt.ylabel('Diversity')
    plt.title('Diversity Over Generations with Standard Deviations')
    plt.suptitle(f'Enemy {enemy} - {algorithm}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example usage:
plot_fitness_results(best_fitness_df, mean_fitness_df, enemy=8, algorithm='Without FS')
plot_diversity_results(diversity_df, enemy=8, algorithm='Without FS')