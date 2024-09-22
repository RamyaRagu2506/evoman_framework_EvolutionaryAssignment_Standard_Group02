import sys
import time
import numpy as np
import random
import os
from memetic_controller import player_controller
from evoman.environment import Environment
import matplotlib.pyplot as plt
import pickle

# Island Model Parameters
num_islands = 5  # Number of islands
migration_interval = 5  # Number of generations before migration
migration_size = 2  # Number of individuals to migrate

# Save the population and fitness values (solution state)
def save_population_state(populations, fitnesses, generation, experiment_name):
    for island in range(num_islands):
        population_path = os.path.join(experiment_name, f'memetic_population_island_{island}.pkl')
        with open(population_path, 'wb') as f:
            pickle.dump([populations[island], fitnesses[island], generation], f)
        print(f"Island {island}: Population state saved successfully.")

# Load the population and fitness values (solution state)
def load_population_state(experiment_name):
    populations = []
    fitnesses = []
    for island in range(num_islands):
        population_path = os.path.join(experiment_name, f'memetic_population_island_{island}.pkl')
        with open(population_path, 'rb') as f:
            population, fitness, generation = pickle.load(f)
        populations.append(population)
        fitnesses.append(fitness)
        print(f"Island {island}: Population state loaded successfully.")
    return populations, fitnesses, generation

# Initialize population
def initialize_population(npop, n_vars, dom_l, dom_u):
    return np.random.uniform(dom_l, dom_u, (npop, n_vars))

# Evaluate fitness of the population
def evaluate_population(env, pop):
    return np.array([env.play(pcont=individual)[0] for individual in pop])

# Selection based on fitness (roulette wheel selection)
def selection(pop, fit_pop):
    min_fitness = np.min(fit_pop)
    if min_fitness < 0:
        fit_pop = fit_pop - min_fitness + 1e-6  # Shift fitness values to be positive
    fit_sum = np.sum(fit_pop)
    probs = fit_pop / fit_sum if fit_sum > 0 else np.ones(len(fit_pop)) / len(fit_pop)
    selected_idx = np.random.choice(np.arange(len(pop)), size=len(pop), p=probs)
    return pop[selected_idx]

# Crossover
def crossover(parent1, parent2, n_vars):
    cross_point = np.random.randint(1, n_vars)
    return np.concatenate((parent1[:cross_point], parent2[cross_point:]))

# Mutation
def mutation(offspring, mutation_rate, dom_l, dom_u):
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            offspring[i] += np.random.uniform(dom_l, dom_u)
    return offspring

# Local Search (Hill Climbing)
def hill_climb(env, individual, mutation_rate, n_iterations=5):
    best_fitness = env.play(pcont=individual)[0]
    best_individual = individual.copy()
    for _ in range(n_iterations):
        new_individual = mutation(individual.copy(), mutation_rate, -1, 1)
        new_fitness = env.play(pcont=new_individual)[0]
        if new_fitness > best_fitness:
            best_fitness = new_fitness
            best_individual = new_individual
    return best_individual, best_fitness

# Migration: Exchange individuals between islands
def migrate_islands(populations, fitnesses):
    for i in range(num_islands):
        source_island = populations[i]
        target_island = populations[(i + 1) % num_islands]
        source_fit = fitnesses[i]
        
        # Select the best individuals to migrate
        best_individuals = source_island[np.argsort(source_fit)[-migration_size:]]
        
        # Replace random individuals in the target island with these best individuals
        replace_indices = np.random.choice(len(target_island), migration_size, replace=False)
        target_island[replace_indices] = best_individuals

    return populations

# Plot fitness over generations and save as image
def plot_fitness(generations, best_fitness_list, mean_fitness_list, experiment_name):
    plt.figure(figsize=(10, 6))
    for i in range(num_islands):
        plt.plot(generations, best_fitness_list[i], label=f'Best Fitness (Island {i+1})', marker='o')
        plt.plot(generations, mean_fitness_list[i], label=f'Mean Fitness (Island {i+1})', marker='x')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations - Island Model Memetic Algorithm')
    plt.legend()
    plt.grid(True)

    # Save the plot to the experiment directory
    plot_path = os.path.join(experiment_name, 'fitness_over_generations.png')
    plt.savefig(plot_path)
    plt.show()

# Memetic Algorithm with Island Model
def island_model_memetic_algorithm(env, populations, fitnesses, npop, gens, ini_g, n_vars, dom_l, dom_u, mutation_rate, experiment_name):
    best_fitness_list = [[] for _ in range(num_islands)]
    mean_fitness_list = [[] for _ in range(num_islands)]

    for generation in range(ini_g, gens):
        print(f"\n========== Generation {generation + 1}/{gens} ==========")
        
        for island in range(num_islands):
            print(f"\nIsland {island + 1}:")

            # Evaluate fitness and store fitness stats
            fit_pop = fitnesses[island]
            best_fitness_list[island].append(np.max(fit_pop))
            mean_fitness_list[island].append(np.mean(fit_pop))

            # Selection
            selected_pop = selection(populations[island], fit_pop)

            # Crossover and mutation
            offspring = []
            for i in range(0, npop, 2):
                parent1, parent2 = selected_pop[i], selected_pop[i + 1]
                child1 = crossover(parent1, parent2, n_vars)
                child2 = crossover(parent2, parent1, n_vars)
                offspring.append(mutation(child1, mutation_rate, dom_l, dom_u))
                offspring.append(mutation(child2, mutation_rate, dom_l, dom_u))

            offspring = np.array(offspring)

            # Local search
            refined_offspring = []
            for individual in offspring:
                refined_individual, _ = hill_climb(env, individual, mutation_rate)
                refined_offspring.append(refined_individual)

            populations[island] = np.array(refined_offspring)
            fitnesses[island] = evaluate_population(env, populations[island])

        # Perform migration every 'migration_interval' generations
        if (generation + 1) % migration_interval == 0:
            populations = migrate_islands(populations, fitnesses)

    return populations, fitnesses, best_fitness_list, mean_fitness_list

# Main function
def main():
    npop = 100
    gens = 30
    mutation_rate = 0.1
    n_hidden_neurons = 10
    dom_l, dom_u = -1, 1

    # Set up environment
    experiment_name = "island_memetic_optimization"
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    env = Environment(experiment_name=experiment_name,
                      enemies=[2],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    print(f"Environment setup with {n_vars} variables per individual.")

    # Check if previous evolution exists, else start new evolution
    if all([os.path.exists(experiment_name + f'/memetic_population_island_{i}.pkl') for i in range(num_islands)]):
        # Continue evolution
        print("\nCONTINUING EVOLUTION\n")
        populations, fitnesses, ini_g = load_population_state(experiment_name)
    else:
        # New evolution
        print("\nNEW EVOLUTION\n")
        populations = []
        fitnesses = []
        for island in range(num_islands):
            print(f"\nInitializing Island {island + 1}:")
            pop = initialize_population(npop, n_vars, dom_l, dom_u)
            fit_pop = evaluate_population(env, pop)
            populations.append(pop)
            fitnesses.append(fit_pop)
        ini_g = 0

    # Run the Island Model Memetic Algorithm
    print("\nRunning the Island Model Memetic Algorithm...")
    final_populations, final_fitnesses, best_fitness_list, mean_fitness_list = island_model_memetic_algorithm(
        env, populations, fitnesses, npop, gens, ini_g, n_vars, dom_l, dom_u, mutation_rate, experiment_name)

    # Save the population state for future continuation
    save_population_state(final_populations, final_fitnesses, gens, experiment_name)

    # Plot the fitness over generations
    generations = list(range(ini_g + 1, gens + 1))
    plot_fitness(generations, best_fitness_list, mean_fitness_list, experiment_name)

if __name__ == "__main__":
    main()
