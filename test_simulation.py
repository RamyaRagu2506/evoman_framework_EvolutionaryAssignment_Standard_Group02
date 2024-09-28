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
    - tuple: Fitness, player life, enemy life, time, individual_gain.
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

    individual_gain = p - e  # player energy - enemy energy
    
    # Return the results of the fight
    return f, p, e, t, individual_gain

def simulate_multiple_times_best(weights_path, output_dir, enemy=4, n_hidden_neurons=10, n_runs=5):
    """
    Simulates a fight against the specified enemy multiple times using the weights in the provided file,
    and stores the results in an Excel file in the specified output directory.
    
    Args:
    - weights_path (str): Path to the file containing the network weights.
    - output_dir (str): Directory where the results will be saved.
    - enemy (int): The enemy number to fight against (default is enemy 4).
    - n_hidden_neurons (int): Number of hidden neurons in the player's controller.
    - n_runs (int): Number of times to run the simulation (default is 5).
    
    Returns:
    - pd.DataFrame: DataFrame containing the simulation results.
    """
    # Load the best weights
    weights = load_weights(weights_path)
    
    results = []

    # Run the simulation for n_runs times
    for run in range(n_runs):
        fitness, player_life, enemy_life, time, individual_gain = simulate_enemy_fight(weights, enemy, n_hidden_neurons)
        
        # Store the results
        results.append({
            'Run': run + 1,
            'Fitness': fitness,
            'Player_Life': player_life,
            'Enemy_Life': enemy_life,
            'Time': time,
            'Individual_Gain': individual_gain
        })
        print(f"Run {run + 1} - Fitness: {fitness}, Player Life: {player_life}, Enemy Life: {enemy_life}, Time: {time}, Individual Gain: {individual_gain}")

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the results to an Excel file in the test folder
    excel_file_path = os.path.join(output_dir, 'es_enemy_8.xlsx')
    results_df.to_excel(excel_file_path, index=False)

    print(f"Results saved to {excel_file_path}")
    
    return results_df

# Example usage:
best_individual_path = 'es_enemy8/best_individual.txt'  # Replace with actual path to the best_individual.txt file
output_directory = 'test'  # Replace with actual path to the test folder
simulation_results = simulate_multiple_times_best(best_individual_path, output_dir=output_directory, enemy=8, n_hidden_neurons=10, n_runs=5)

