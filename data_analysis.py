import os
import pandas as pd


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
def process_results_extended(base_dir,enemy, iterations=10, generations=50):
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
        diversity_file = os.path.join(base_dir, f'fs_{i}_enemy{enemy}', 'diversity_results.txt')
        fitness_file = os.path.join(base_dir, f'fs_{i}_enemy{enemy}', 'fitness_results.txt')

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
diversity_df, best_fitness_df, mean_fitness_df, std_fitness_df = process_results_extended(base_dir='fs_enemy3', enemy=3, iterations=10, generations=50)
print(diversity_df)
print(best_fitness_df)
print(mean_fitness_df)
print(std_fitness_df)