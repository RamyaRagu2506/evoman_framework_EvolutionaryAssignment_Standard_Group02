import os
import numpy as np
import pandas as pd
from evoman.environment import Environment
from memetic_controller import player_controller

def load_weights(file_path):
    """
    Loads the neural network weights from the specified file.
    
    Args:
    - file_path (str): Path to the file containing the network weights.
    
    Returns:
    - np.array: Array of loaded weights.
    """
    return np.loadtxt(file_path)

def simulate_enemy_fight(weights, enemy=4, n_hidden_neurons=10):
    """
    Simulates a fight between the player (with the given weights) and the specified enemy.
    
    Args:
    - weights (np.array): Neural network weights for the player.
    - enemy (int): The enemy number to fight against (default is enemy 4).
    - n_hidden_neurons (int): Number of hidden neurons in the player's controller.
    
    Returns:
    - tuple: Fitness, player life, enemy life, time.
    """
    # Set up the EvoMan environment for enemy 4
    env = Environment(
        enemies=[enemy],
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False
    )
    
    # Run the simulation with the given weights
    f, p, e, t = env.play(pcont=weights)

    individual_gain = p - e #player energy - enemey energy 
    
    # Return the results of the fight
    return f, p, e, t,individual_gain

def run_simulations_for_all(base_dir, enemy=4, n_hidden_neurons=10):
    """
    Runs the simulation for all best.txt files in the subfolders of base_dir.
    
    Args:
    - base_dir (str): Path to the main directory (e.g., fs_enemy4) that contains subfolders.
    - enemy (int): The enemy number to fight against (default is enemy 4).
    - n_hidden_neurons (int): Number of hidden neurons in the player's controller.
    
    Returns:
    - pd.DataFrame: DataFrame with simulation results.
    """
    results = []
    best_individual_weights = None
    max_individual_gain = float('-inf')  # Initialize the maximum gain as negative infinity

    # Loop over all subdirectories (e.g., fs_1_enemy4, fs_2_enemy4, etc.)
    for i in range(1, 11):  # Assuming there are 10 subfolders
        subfolder = os.path.join(base_dir, f'fs_{i}_enemy{enemy}')
        best_weights_path = os.path.join(subfolder, 'best.txt')

        # Check if the best.txt file exists
        if os.path.exists(best_weights_path):
            # Load the best weights
            best_weights = load_weights(best_weights_path)

            # Run the simulation
            fitness, player_life, enemy_life, time, individual_gain = simulate_enemy_fight(best_weights, enemy, n_hidden_neurons)

            # Store the results in the list
            results.append({
                'Iteration': i,
                'Fitness': fitness,
                'Player_Life': player_life,
                'Enemy_Life': enemy_life,
                'Time': time,
                'Individual_Gain': individual_gain
            })

            # Update the best individual if this individual has the highest gain
            if individual_gain > max_individual_gain:
                max_individual_gain = individual_gain
                best_individual_weights = best_weights

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file in the base directory
    results_file = os.path.join(base_dir, 'simulation_results_fs.xls')
    results_df.to_csv(results_file, index=False)

    # Save the weights of the individual with the maximum gain
    if best_individual_weights is not None:
        best_individual_file = os.path.join(base_dir, 'best_individual.txt')
        np.savetxt(best_individual_file, best_individual_weights)
        print(f"Best individual weights saved to {best_individual_file}")

    return results_df

# Example usage:
base_directory = "fs_enemy4" # Replace this with the actual path to the es_enemy4 directory
simulation_results_df = run_simulations_for_all(base_directory, enemy=4, n_hidden_neurons=10)

# Output the simulation results
print(simulation_results_df)
