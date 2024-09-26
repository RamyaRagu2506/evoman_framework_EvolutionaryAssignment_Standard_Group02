#######################################################################################
# EvoMan FrameWork - V1.0 2016                                  					  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.          #
#        specialist solutions for each enemy (game)                                  #
# Author: Karine Miras        			                              				  #
# karine.smiras@gmail.com     				                                          #
#######################################################################################

# Imports
import sys
import os
import argparse
import numpy as np
import time
from joblib import Parallel, delayed
from time import sleep

# EvoMan Framework Imports
from evoman.environment import Environment
from memetic_controller import player_controller
import matplotlib.pyplot as plt
import pickle

# Global Configuration
DEFAULT_HIDDEN_NEURONS = 10
DEFAULT_POP_SIZE = 200
DEFAULT_GENS = 100
DEFAULT_ENEMY = 3
DEFAULT_VARS = 265 # total no. of weights in consideration 
DEFAULT_TAU = 1 / np.sqrt(2 * np.sqrt(DEFAULT_VARS)) # global mutation factor 
DEFAULT_TAU_PRIME = 1/np.sqrt(2* (DEFAULT_VARS)) # local mutation factor 
DEFAULT_ALPHA = 0.5
REPLACEMENT_FACTOR = 4  # 1/REPLACEMENT_FACTOR of the population will be replaced with random solutions (doomsday)
LOCAL_SEARCH_ITER = 5  # Number of iterations for local hill climbing search
DEFAULT_EPSILON = 1e-8 # prevent the step size in mutation to shrink 

# Argument Parsing
def get_args():
    parser = argparse.ArgumentParser(description='EvoMan Memetic Algorithm Experiment')
    parser.add_argument('--experiment_name', type=str, default='experiment_test',
                        help='Name of the experiment directory')
    parser.add_argument('--run_mode', type=str, choices=['train', 'test'], default='train',
                        help='Run mode: "train" or "test"')
    parser.add_argument('--enemies', type=int, default=DEFAULT_ENEMY, help='ID of the enemy to fight')
    parser.add_argument('--pop_size', type=int, default=DEFAULT_POP_SIZE, help='Population size for evolution')
    parser.add_argument('--gens', type=int, default=DEFAULT_GENS, help='Number of generations for evolution')
    parser.add_argument('--n_hidden_neurons', type=int, default=DEFAULT_HIDDEN_NEURONS,
                        help='Number of hidden neurons in the neural network controller')
    return parser.parse_args()


# Simulation function
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f


# Environment setup
def setup_environment(experiment_name, enemies, controller):
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    return Environment(
        experiment_name=experiment_name,
        playermode="ai",
        enemies=[enemies],
        player_controller=controller,
        speed="fastest",
        enemymode="static",
        level=2,
        visuals=False
    )


# Fitness evaluation using parallel workers
def evaluate_fitnesses(env, population):
    fitnesses = Parallel(n_jobs=-1)(
        delayed(run_game_in_worker)(env.experiment_name, env.player_controller, ind) for ind in population
    )
    return fitnesses


def run_game_in_worker(experiment_name, controller, ind):
    env = setup_environment(experiment_name, DEFAULT_ENEMY, controller)
    return simulation(env, ind)


# Initialize population
def init_population(pop_size, env, n_vars, dom_l, dom_u):
    pop = np.random.uniform(low=dom_l, high=dom_u, size=(pop_size, n_vars))
    step_sizes = np.random.uniform(low=-0.5, high=0.5, size=(pop_size, n_vars))
    fit_pop = evaluate_fitnesses(env, pop)
    print(f"INITIAL POPULATION: Best Fitness: {round(max(fit_pop), 6)} - Mean Fitness: {round(np.mean(fit_pop), 6)} - Std Fitness: {round(np.std(fit_pop), 6)}")
    print("INITIAL POPULATION: step size metrics: mean: ", np.mean(step_sizes), "std: ", np.std(step_sizes))
    return pop, step_sizes, fit_pop

# Save the population and fitness values (solution state)
def save_population_state(population, fitness, generation, experiment_name):
    population_path = os.path.join(experiment_name, 'memetic_population.pkl')
    with open(population_path, 'wb') as f:
        pickle.dump([population, fitness, generation], f)
    print("Population state saved successfully.")

# Load the population and fitness values (solution state)
def load_population_state(experiment_name):
    population_path = os.path.join(experiment_name, 'memetic_population.pkl')
    with open(population_path, 'rb') as f:
        population, fitness, generation = pickle.load(f)
    print("Population state loaded successfully.")
    return population, fitness, generation

# Parent selection methods
def select_parents_tournament(pop, fit_pop, tournament_size=10):
    fit_pop = np.array(fit_pop)
    tournament_indices = np.random.randint(0, pop.shape[0], tournament_size)
    tournament = fit_pop[tournament_indices]
    best_parent_idx = np.argmax(tournament)
    best_parent = pop[tournament_indices[best_parent_idx]]
    return tournament_indices[best_parent_idx], best_parent


# Recombination: Blend Recombination
def blend_recombination(step_sizes, pop, fit_pop, n_vars, alpha=DEFAULT_ALPHA):
    n_offspring = np.random.randint(DEFAULT_POP_SIZE + 1, DEFAULT_POP_SIZE * 2)
    offspring = np.zeros((n_offspring, n_vars))
    offspring_step_size = np.zeros((n_offspring, n_vars))

    for i in range(n_offspring):
        parent_idx1, parent1 = select_parents_tournament(pop, fit_pop)
        parent_idx2, parent2 = select_parents_tournament(pop, fit_pop)
        difference = np.abs(parent1 - parent2)
        min_values = np.minimum(parent1, parent2) - difference * alpha
        max_values = np.maximum(parent1, parent2) + difference * alpha
        offspring[i] = np.random.uniform(min_values, max_values)
        offspring_step_size[i] = np.mean(np.stack((step_sizes[parent_idx1], step_sizes[parent_idx2])), axis=0)
    return offspring, offspring_step_size


# Mutation: Gaussian mutation
def gaussian_mutation(individual, step_size, tau= DEFAULT_TAU, tau_prime= DEFAULT_TAU_PRIME, epsilon=1e-8):
# Global mutation on step sizes (applies to all dimensions)
    global_mutation = np.exp(tau * np.random.randn())
    
    # Local mutation on each step size (applies to each dimension independently)
    local_mutation = tau_prime * np.random.randn(*step_size.shape)
    
    # Update the step sizes with both global and local mutation
    new_step_size = step_size * global_mutation + local_mutation
    
    # Apply the boundary condition to prevent step sizes from becoming too small
    new_step_size[new_step_size < epsilon] = epsilon
    
    # Mutate the individual (xi) using the updated step sizes
    new_individual = individual + new_step_size * np.random.randn(*individual.shape)
    
    return new_individual, new_step_size


# Local Search: Hill Climbing
def hill_climb(env, individual, mutation_rate, n_iterations=LOCAL_SEARCH_ITER):
    best_individual = individual.copy()
    best_fitness = simulation(env, best_individual)
    for _ in range(n_iterations):
        new_individual = individual + mutation_rate * np.random.randn(*individual.shape)
        new_fitness = simulation(env, new_individual)
        if new_fitness > best_fitness:
            best_fitness = new_fitness
            best_individual = new_individual
    return best_individual, best_fitness


# Survivor selection with elitism
def survivor_selection_elitism(pop, fit_pop, step_sizes, fit_offspring, offspring, offspring_step_size, pop_size):
    parents_and_offspring = np.concatenate((pop, offspring), axis=0)
    parents_and_offspring_step_sizes = np.concatenate((step_sizes, offspring_step_size), axis=0)
    parents_and_offspring_fitnesses = np.concatenate((fit_pop, fit_offspring), axis=0)
    elite_idx = np.argsort(parents_and_offspring_fitnesses)[-pop_size:]
    return parents_and_offspring[elite_idx], parents_and_offspring_fitnesses[elite_idx], parents_and_offspring_step_sizes[elite_idx]


# Plot fitness over generations and save as image
def plot_fitness(generations, best_fitness_list, mean_fitness_list, std_fitness_list, experiment_name):
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness_list, label='Best Fitness', color='b', marker='o')
    plt.plot(generations, mean_fitness_list, label='Mean Fitness', color='g', marker='x')
    plt.plot(generations, std_fitness_list, label='Standard Deviation', color='r', marker='s')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations - Memetic Algorithm')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to the experiment directory
    plot_path = os.path.join(experiment_name, 'fitness_over_generations.png')
    plt.savefig(plot_path)
    plt.show()

# Function to save results to a file
def save_generation_results(experiment_name, generation, best_fitness, mean_fitness, std_fitness):
    results_path = os.path.join(experiment_name, 'results_memetic.txt')
    with open(results_path, 'a') as file_aux:
        file_aux.write(f"Generation {generation + 1}: Best Fitness: {best_fitness:.6f}, Mean Fitness: {mean_fitness:.6f}, Standard Deviation Fitness: {std_fitness:.6f}\n")

# Save the final best solution and fitness
def save_final_solution(experiment_name, best_solution, best_fitness):
    solution_path = os.path.join(experiment_name, 'best_solution.txt')
    with open(solution_path, 'w') as file_aux:
        file_aux.write(f"Best Solution: {best_solution}\n")
        file_aux.write(f"Best Fitness: {best_fitness:.6f}\n")

# Main Memetic Algorithm with Evolutionary Strategy and Hill Climbing
def memetic_algorithm(env, pop, fit_pop, npop, gens, ini_g, n_vars, mutation_rate, experiment_name):
    print(f"Starting Memetic Algorithm with {npop} individuals and {gens} generations...\n")
    
    best_fitness_list = []
    mean_fitness_list = []
    std_fitness_list = []
    step_sizes = np.random.uniform(0.1, 0.5, size=(npop, n_vars))
    overall_best_fitness = None  # Initialize to track the best fitness across all generations
    no_generations_without_improvement = 0  # Counter for stagnation

    for generation in range(ini_g, gens):
        print(f"\n========== Generation {generation + 1}/{gens} ==========")

        # Recombination and Mutation
        offspring, offspring_step_sizes = blend_recombination(step_sizes, pop, fit_pop, n_vars)
        
        # Apply Gaussian mutation to offspring
        for i in range(len(offspring)):
            offspring[i], offspring_step_sizes[i] = gaussian_mutation(offspring[i], offspring_step_sizes[i], mutation_rate)
        
        # Evaluate offspring
        fit_offspring = evaluate_fitnesses(env, offspring)

        # Apply Local Search (Hill Climbing) to offspring
        for i in range(len(offspring)):
            refined_individual, refined_fitness = hill_climb(env, offspring[i], mutation_rate)
            offspring[i], fit_offspring[i] = refined_individual, refined_fitness

        # Survivor selection with elitism
        pop, fit_pop, step_sizes = survivor_selection_elitism(pop, fit_pop, step_sizes, fit_offspring, offspring, offspring_step_sizes, npop)
        
        # Track fitness for plotting
        best_fitness = np.max(fit_pop)
        mean_fitness = np.mean(fit_pop)
        std_fitness = np.std(fit_pop)
        best_fitness_list.append(best_fitness)
        mean_fitness_list.append(mean_fitness)
        std_fitness_list.append(std_fitness)
        save_generation_results(experiment_name, generation, best_fitness, mean_fitness, std_fitness)

        print(f"\nSummary of Generation {generation + 1}:")
        print(f"  - Best Fitness: {best_fitness:.6f}")
        print(f"  - Mean Fitness: {mean_fitness:.6f}")
        print(f"  - Standard Deviation Fitness : {std_fitness:.6f}")
        print(f"======================================")

    best_individual_idx = np.argmax(fit_pop)
    return pop[np.argmax(fit_pop)], np.max(fit_pop), best_fitness_list, mean_fitness_list, std_fitness_list


def main():
    # Parameters
    npop = 100
    gens = 30
    mutation_rate = 0.1
    n_hidden_neurons = 10
    dom_l, dom_u = -1, 1

    # Set up environment
    experiment_name = "memetic_optimization_es_enemy3"
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    env = Environment(experiment_name=experiment_name,
                      enemies=[3],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    print(f"Environment setup with {n_vars} variables per individual.")

    # Check if previous evolution exists, else start new evolution
    if os.path.exists(experiment_name + '/memetic_population_de.pkl'):
        # Continue evolution
        print("\nCONTINUING EVOLUTION\n")
        pop, fit_pop, ini_g = load_population_state(experiment_name)
    else:
        # New evolution
        print("\nNEW EVOLUTION\n")
        pop, step_sizes, fit_pop = init_population(npop, env, n_vars, dom_l, dom_u)
        fit_pop = evaluate_fitnesses(env, pop)
        ini_g = 0

    # Run the Memetic Algorithm with DE
    print("\nRunning the Memetic Algorithm with DE...")
    best_solution, best_fitness, best_fitness_list, mean_fitness_list, std_fitness_list = memetic_algorithm(
        env, pop, fit_pop, npop, gens, ini_g, n_vars, mutation_rate, experiment_name)

    # Output final results
    print(f"\nBest solution found after {gens} generations:\n{best_solution}")
    print(f"Best fitness achieved: {best_fitness}")

    # Save the final best solution and its fitness
    save_population_state(pop, fit_pop, gens, experiment_name)

    # Plot the fitness over generations
    generations = list(range(ini_g + 1, gens + 1))
    plot_fitness(generations, best_fitness_list, mean_fitness_list, std_fitness_list, experiment_name)
    np.savetxt(experiment_name + '/best.txt', best_solution)

if __name__ == '__main__':
    main()
