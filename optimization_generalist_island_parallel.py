import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import random

import memetic_controller
from evoman.environment import Environment
from demo_controller import player_controller
import memetic_controller

# Global Configuration
DEFAULT_HIDDEN_NEURONS = 10
DEFAULT_POP_SIZE = 200 # Population size per island
DEFAULT_GENS = 50
DEFAULT_VARS = 265
DEFAULT_TAU = 1 / np.sqrt(2 * np.sqrt(DEFAULT_VARS))
DEFAULT_TAU_PRIME = 1 / np.sqrt(2 * (DEFAULT_VARS))
DEFAULT_ALPHA = 0.5
LOCAL_SEARCH_ITER = 5
DEFAULT_EPSILON = 1e-8
DEFAULT_ENEMY = [2,3]
n_islands = 6
mutation_rate = 0.8
migration_interval = 20
migration_size = 2
migration_type = "diversity"  # Can be "similarity" or "diversity" 
mu = 0 # mean of initial population
sigma = 0.21 #std for initial population
headless = True
dom_l, dom_u = -1, 1

from joblib import Parallel, delayed
def setup_environment(experiment_name, controller, enemies=DEFAULT_ENEMY) -> Environment:
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    return Environment(
        experiment_name=experiment_name,
        playermode="ai",
        enemies=enemies,
        player_controller=controller,
        speed="fastest",
        enemymode="static",
        multiplemode="yes",
        level=2,
        visuals=False,
    )


# Fitness evaluation using parallel workers
def evaluate_fitnesses(env, population):
    fitnesses = Parallel(n_jobs=-1)(
        delayed(run_game_in_worker)(env.experiment_name, env.player_controller, ind) for ind in population
    )
    return fitnesses


def run_game_in_worker(experiment_name, controller, ind):
    env = setup_environment(experiment_name, controller, enemies=DEFAULT_ENEMY)
    return simulation(env, ind, enemies=DEFAULT_ENEMY)

# Fitness evaluation using parallel workers


# Simulation function
def simulation(env, x, enemies=DEFAULT_ENEMY):
    env.update_parameter('enemies', enemies)
    env.update_parameter('multiplemode', 'yes')
    f, p, e, t = env.play(pcont=x)
    return f

# Recombination (Blend Crossover)
def blend_recombination(step_sizes, pop, fit_pop, n_vars, alpha=DEFAULT_ALPHA):
    n_offspring = len(pop)
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

# Mutation (Gaussian Mutation)
def gaussian_mutation(individual, step_size, tau=DEFAULT_TAU, tau_prime=DEFAULT_TAU_PRIME, epsilon=DEFAULT_EPSILON):
    global_mutation = np.exp(tau * np.random.randn())
    local_mutation = tau_prime * np.random.randn(*step_size.shape)
    new_step_size = step_size * global_mutation + local_mutation
    new_step_size[new_step_size < epsilon] = epsilon
    new_individual = individual + new_step_size * np.random.randn(*individual.shape)
    return new_individual, new_step_size

# Parent selection using tournament
def select_parents_tournament(pop, fit_pop, tournament_size=10):
    fit_pop = np.array(fit_pop)
    tournament_indices = np.random.randint(0, len(pop), tournament_size)
    best_parent_idx = tournament_indices[np.argmax(fit_pop[tournament_indices])]
    best_parent = pop[best_parent_idx]
    return best_parent_idx, best_parent

# Survivor selection (elitism)
def survivor_selection_elitism(pop, fit_pop, step_sizes, fit_offspring, offspring, offspring_step_size, pop_size):
    combined_population = np.concatenate((pop, offspring), axis=0)
    combined_step_sizes = np.concatenate((step_sizes, offspring_step_size), axis=0)
    combined_fitness = np.concatenate((fit_pop, fit_offspring), axis=0)
    elite_indices = np.argsort(combined_fitness)[-pop_size:]
    return combined_population[elite_indices], combined_fitness[elite_indices], combined_step_sizes[elite_indices]

# Migration logic (similarity and diversity-based)
def similarity(source_island, destination_best, migration_size):
    source_island_copy = source_island.copy()
    most_similar = []
    for _ in range(migration_size):
        similarity_score = float('inf')
        for index, individual in enumerate(source_island_copy):
            difference = np.abs(destination_best - individual)
            sum_diff = np.sum(difference)
            if sum_diff < similarity_score:
                similarity_score = sum_diff
                most_similar_ind = individual
                most_similar_index = index
        most_similar.append(most_similar_ind)
        source_island_copy = np.delete(source_island_copy, most_similar_index, axis=0)
    return most_similar

def diversity(source_island, destination_best, migration_size):
    source_island_copy = source_island.copy()
    most_diverse = []
    for _ in range(migration_size):
        diversity_score = 0
        for index, individual in enumerate(source_island_copy):
            difference = np.abs(destination_best - individual)
            sum_diff = np.sum(difference)
            if sum_diff > diversity_score:
                diversity_score = sum_diff
                most_diverse_ind = individual
                most_diverse_index = index
        most_diverse.append(most_diverse_ind)
        source_island_copy = np.delete(source_island_copy, most_diverse_index, axis=0)
    return most_diverse

# Migration between islands
def migrate(world_population, world_pop_fit, migration_size, migration_type):
    for i, island in enumerate(world_population):
        best_island_index = np.argmax(world_pop_fit[i])
        best_individual = island[best_island_index]
        other_islands = [world_population[j] for j in range(len(world_population)) if j != i]
        source_island = random.choice(other_islands)
        if migration_type == "similarity":
            migrants = similarity(source_island, best_individual, migration_size)
        else:
            migrants = diversity(source_island, best_individual, migration_size)
        worst_indices = np.argsort(world_pop_fit[i])[:migration_size]
        for idx, migrant in zip(worst_indices, migrants):
            world_population[i][idx] = migrant
    return world_population

# Evolutionary Strategy applied per island
def individual_island_run(env,island_population, pop_fit, step_sizes, mutation_rate):
    offspring, offspring_step_sizes = blend_recombination(step_sizes, island_population, pop_fit, DEFAULT_VARS)
    for i in range(len(offspring)):
        offspring[i], offspring_step_sizes[i] = gaussian_mutation(offspring[i], offspring_step_sizes[i], mutation_rate)
    fit_offspring = evaluate_fitnesses(env, offspring)
    return survivor_selection_elitism(island_population, pop_fit, step_sizes, fit_offspring, offspring, offspring_step_sizes, DEFAULT_POP_SIZE)

# Parallel island execution
def parallel_island_run(env, world_population, pop_fit, step_sizes, mutation_rate):
    for i in range(n_islands):
        world_population[i], pop_fit[i], step_sizes[i] = individual_island_run(env, world_population[i], pop_fit[i],
                                                                               step_sizes[i], mutation_rate)
    return world_population, pop_fit, step_sizes

# Function to save the population and fitness values (solution state)
def save_population_state(population, fitness, generation, experiment_name):
    population_path = os.path.join(experiment_name, 'island_population.pkl')
    with open(population_path, 'wb') as f:
        pickle.dump([population, fitness, generation], f)
    print("Population state saved successfully.")

# Function to load the population and fitness values (solution state)
def load_population_state(experiment_name):
    population_path = os.path.join(experiment_name, 'island_population.pkl')
    with open(population_path, 'rb') as f:
        population, fitness, generation = pickle.load(f)
    print("Population state loaded successfully.")
    return population, fitness, generation

# Function to plot fitness over generations and save as an image
def plot_fitness(generations, best_fitness_list, mean_fitness_list, std_fitness_list, experiment_name):
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness_list, label='Best Fitness', color='b', marker='o')
    plt.plot(generations, mean_fitness_list, label='Mean Fitness', color='g', marker='x')
    plt.plot(generations, std_fitness_list, label='Standard Deviation', color='r', marker='s')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations - Island Evolutionary Strategy')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to the experiment directory
    plot_path = os.path.join(experiment_name, 'fitness_over_generations.png')
    plt.savefig(plot_path)
    plt.show()

# Function to save results to a file
def save_generation_results(experiment_name, generation, best_fitness, mean_fitness, std_fitness):
    results_path = os.path.join(experiment_name, 'results_island_es.txt')
    with open(results_path, 'a') as file_aux:
        file_aux.write(f"Generation {generation + 1}: Best Fitness: {best_fitness:.6f}, Mean Fitness: {mean_fitness:.6f}, Standard Deviation Fitness: {std_fitness:.6f}\n")

# Function to save the final best solution and fitness
def save_final_solution(experiment_name, best_solution, best_fitness):
    solution_path = os.path.join(experiment_name, 'best_solution.txt')
    with open(solution_path, 'w') as file_aux:
        file_aux.write(f"Best Solution: {best_solution}\n")
        file_aux.write(f"Best Fitness: {best_fitness:.6f}\n")


def test_solution_against_all_enemies(winner):
    print("winner shape", winner.shape)
    print("winner", winner)
    all_enemies = [1, 2, 3, 4, 5, 6, 7, 8]
    controller = memetic_controller.player_controller(n_hidden=10)
    controller.set(winner, 10)  # Set the weights once

    total_gains = []
    total_fitnesses = []

    for enemy in all_enemies:
        env = setup_environment("testing_against_all8", controller, enemies=[enemy])
        env.update_parameter('speed', 'fastest')
        env.update_parameter('multimode', 'yes')
        env.update_parameter("visuals", False)

        fitness, player_life, enemy_life, _ = env.play(pcont=controller)
        gain = player_life - enemy_life
        
        total_fitnesses.append(fitness)
        total_gains.append(gain)
        
        print(f"Enemy {enemy}: Fitness = {fitness}, Gain = {gain}")

    # box plot of total gains
    plt.boxplot(total_gains)
    plt.title("Total Gains Against All Enemies")
    plt.show()

    print(f"Total Gains: {total_gains}")
    print(f"Total Fitnesses: {total_fitnesses}")

    return total_fitnesses, total_gains
# Main function 
def main():
    ini = time.time()


    # Set up environment
    experiment_name = f"island_demo_diversity_try_{DEFAULT_ENEMY}"
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    controller = player_controller(DEFAULT_HIDDEN_NEURONS)
    env = env = setup_environment(experiment_name, controller, DEFAULT_ENEMY)


    n_weights= (env.get_num_sensors() + 1) * DEFAULT_HIDDEN_NEURONS + (DEFAULT_HIDDEN_NEURONS + 1) * 5
    print(f"Environment setup with {n_weights} variables per individual.")


    # Check if a previous state exists, otherwise start a new evolution
    if not os.path.exists(experiment_name + '/island_population.pkl'):
        print('NEW EVOLUTION')
        world_population = [np.clip(np.random.normal(mu, sigma, size=(DEFAULT_POP_SIZE, n_weights)),dom_l,dom_u) for _ in range(n_islands)]
        step_sizes = [np.random.normal(0.05, 0.01, size=(DEFAULT_POP_SIZE, n_weights)) for _ in range(n_islands)]
        world_pop_fit = [evaluate_fitnesses(env, one_island_pop) for one_island_pop in world_population]
        ini_g = 0  # Start from generation 0
        best_fitness_list, mean_fitness_list, std_fitness_list = [], [], []
    else:
        print('CONTINUING EVOLUTION')
        world_population, world_pop_fit, ini_g = load_population_state(experiment_name)
        best_fitness_list, mean_fitness_list, std_fitness_list = [], [], []
        step_sizes = [np.random.normal(0.05, 0.01, size=(DEFAULT_POP_SIZE, n_weights)) for _ in range(n_islands)]

    # Run the evolution process for generations
    for gen in range(ini_g, DEFAULT_GENS):
        print(f"\nGeneration {gen}")
        world_population, world_pop_fit, step_sizes = parallel_island_run(env, world_population, world_pop_fit, step_sizes, mutation_rate)

        # Compute fitness statistics
        best_fitness = np.max([np.max(fit) for fit in world_pop_fit])
        mean_fitness = np.mean([np.mean(fit) for fit in world_pop_fit])
        std_fitness = np.std([np.std(fit) for fit in world_pop_fit])

        # Store fitness results for plotting
        best_fitness_list.append(best_fitness)
        mean_fitness_list.append(mean_fitness)
        std_fitness_list.append(std_fitness)

        # Save results for each generation
        save_generation_results(experiment_name, gen, best_fitness, mean_fitness, std_fitness)

        # Perform migration at intervals
        if gen % migration_interval == 0:
            world_population = migrate(world_population, world_pop_fit, migration_size, migration_type)
            world_pop_fit = [evaluate_fitnesses(env, one_island_pop) for one_island_pop in world_population]

        # Save the population state every generation
        save_population_state(world_population, world_pop_fit, gen, experiment_name)

    # Plot and save the fitness over generations
    generations = list(range(ini_g + 1, DEFAULT_GENS + 1))
    plot_fitness(generations, best_fitness_list, mean_fitness_list, std_fitness_list, experiment_name)

    # Find the best solution across all islands
    best_island_idx = np.argmax([np.max(fit) for fit in world_pop_fit])
    best_individual_idx = np.argmax(world_pop_fit[best_island_idx])
    best_solution = world_population[best_island_idx][best_individual_idx]
    best_fitness = world_pop_fit[best_island_idx][best_individual_idx]

    print(f"Best solution shape: {best_solution.shape}")
    print(f"Best fitness: {best_fitness}")

    # Test the best solution
    test_solution_against_all_enemies(best_solution)

    # Save the final solution
    save_final_solution(experiment_name, best_solution, best_fitness)

    print(f"\nExecution time: {round((time.time() - ini) / 60, 2)} minutes")

if __name__ == "__main__":
    main()
