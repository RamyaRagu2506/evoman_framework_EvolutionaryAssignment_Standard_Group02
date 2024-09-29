import os
import numpy as np
import pandas as pd
from evoman.environment import Environment
from memetic_controller import player_controller

def load_weights(file_path):
    
    return np.loadtxt(file_path)

def simulate_enemy_fight(weights, enemy, n_hidden_neurons=10):

    # Setting up the EvoMan environment for enemy 
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

def run_simulations_for_all(base_dir, enemy, n_hidden_neurons=10):
    try:
        results = []
        individual_gains = [] 

        # Loop over all sub_directories (e.g., es_1_enemy4, es_2_enemy4, etc.)
        for i in range(1, 11):  # 10 sub_folders
            subfolder = os.path.join(base_dir, f'es_{i}_enemy{enemy}')
            best_weights_path = os.path.join(subfolder, 'best.txt')

            # Check if the best.txt file exists
            if os.path.exists(best_weights_path):
                # Load the best weights
                best_weights = load_weights(best_weights_path)

                # Run the simulation 5 times
                for x in range(5):
                    # Run the simulation
                    fitness, player_life, enemy_life, time, individual_gain = simulate_enemy_fight(best_weights, enemy, n_hidden_neurons)
                    individual_gains.append(individual_gain)  # Store the individual gain for each simulation made 

                # Calculate the average individual gain
                avg_individual_gain = np.mean(individual_gains)

                # Store the results in the list
                results.append({
                    'Iteration': i,
                    'Fitness': fitness,
                    'Player_Life': player_life,
                    'Enemy_Life': enemy_life,
                    'Time': time,
                    'Avg_Individual_Gain': avg_individual_gain
                })

        # Convert the results to a DataFrame
        results_df = pd.DataFrame(results)

        # Save the results to a CSV file in the base directory
        results_file = os.path.join(base_dir, 'simulation_results_es.xlsx')
        results_df.to_excel(results_file, index=False)

        return results_df
    except Exception as e:
        print(f"An error occurred: {e} Check the parameters involved")

base_directory = "es_enemy3" 
enemy = 3
simulation_results_df = run_simulations_for_all(base_directory, enemy, n_hidden_neurons=10)

# Output the simulation results
print(simulation_results_df)
