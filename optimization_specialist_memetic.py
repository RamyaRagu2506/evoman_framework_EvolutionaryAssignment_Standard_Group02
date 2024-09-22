import sys
import time
import numpy as np
import random
import os
from memetic_controller import player_controller
from evoman.environment import Environment
import matplotlib.pyplot as plt
import pickle

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

# Global search (GA): Initialize population
def initialize_population(npop, n_vars, dom_l, dom_u):
    print("Initializing population...")
    return np.random.uniform(dom_l, dom_u, (npop, n_vars))

# Evaluate fitness of the population
def evaluate_population(env, pop):
    print("Evaluating fitness for the population...")
    return np.array([env.play(pcont=individual)[0] for individual in pop])

# Selection based on fitness (roulette wheel selection)
def selection(pop, fit_pop):
    print("Selecting individuals based on fitness...")
    min_fitness = np.min(fit_pop)
    if min_fitness < 0:
        fit_pop = fit_pop - min_fitness + 1e-6  # Shift fitness values to be positive
    fit_sum = np.sum(fit_pop)
    if fit_sum == 0:
        probs = np.ones(len(fit_pop)) / len(fit_pop)  # Handle zero-sum fitness
    else:
        probs = fit_pop / fit_sum
    selected_idx = np.random.choice(np.arange(len(pop)), size=len(pop), p=probs)
    return pop[selected_idx]

# Crossover (recombination)
def crossover(parent1, parent2, n_vars):
    cross_point = np.random.randint(1, n_vars)
    return np.concatenate((parent1[:cross_point], parent2[cross_point:]))

# Mutation
def mutation(offspring, mutation_rate, dom_l, dom_u):
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            offspring[i] += np.random.uniform(dom_l, dom_u)
    return offspring

# Local Search: Simple Hill Climbing
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

# Plot fitness over generations and save as image
def plot_fitness(generations, best_fitness_list, mean_fitness_list, experiment_name):
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness_list, label='Best Fitness', color='b', marker='o')
    plt.plot(generations, mean_fitness_list, label='Mean Fitness', color='g', marker='x')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations - Memetic Algorithm')
    plt.legend()
    plt.grid(True)
    
    # Save the plot as an image
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

# Memetic Algorithm: Global search (GA) + local search (hill climbing)
def memetic_algorithm(env, pop, fit_pop, npop, gens, ini_g, n_vars, dom_l, dom_u, mutation_rate, experiment_name):
    print(f"Starting Memetic Algorithm with {npop} individuals and {gens} generations...\n")
    
    best_fitness_list = []
    mean_fitness_list = []

    for generation in range(ini_g, gens):
        print(f"\n========== Generation {generation + 1}/{gens} ==========")
        print(f"Best fitness in current generation: {np.max(fit_pop):.6f}")
        print(f"Mean fitness in current generation: {np.mean(fit_pop):.6f}")
        print(f"Standard Deviation fitness in current generation: {np.std(fit_pop):.6f}\n")

        # Store fitness values for plotting
        best_fitness_list.append(np.max(fit_pop))
        mean_fitness_list.append(np.mean(fit_pop))

        # Selection
        print("Performing selection...")
        selected_pop = selection(pop, fit_pop)

        # Create new population with crossover and mutation
        print("Generating offspring with crossover and mutation...")
        offspring = []
        for i in range(0, npop, 2):
            parent1, parent2 = selected_pop[i], selected_pop[i + 1]
            child1 = crossover(parent1, parent2, n_vars)
            child2 = crossover(parent2, parent1, n_vars)
            offspring.append(mutation(child1, mutation_rate, dom_l, dom_u))
            offspring.append(mutation(child2, mutation_rate, dom_l, dom_u))

        offspring = np.array(offspring)

        # Local search (hill climbing) applied to offspring
        print("Refining offspring with local search (hill climbing)...")
        refined_offspring = []
        for individual in offspring:
            refined_individual, refined_fitness = hill_climb(env, individual, mutation_rate)
            refined_offspring.append(refined_individual)

        pop = np.array(refined_offspring)
        fit_pop = evaluate_population(env, pop)

        # Log results for the generation
        best_fitness = np.max(fit_pop)
        mean_fitness = np.mean(fit_pop)
        std_fitness = np.std(fit_pop)
        save_generation_results(experiment_name, generation, best_fitness, mean_fitness, std_fitness)

        print(f"\nSummary of Generation {generation + 1}:")
        print(f"  - Best Fitness: {best_fitness:.6f}")
        print(f"  - Mean Fitness: {mean_fitness:.6f}")
        print(f"  - Standard Deviation Fitness : {std_fitness:.6f}")
        print(f"======================================")

    best_individual_idx = np.argmax(fit_pop)
    return pop[best_individual_idx], fit_pop[best_individual_idx], best_fitness_list, mean_fitness_list

# Main function to run the memetic algorithm
def main():
    # Parameters
    npop = 100
    gens = 30
    mutation_rate = 0.1
    n_hidden_neurons = 10
    dom_l, dom_u = -1, 1

    # Set up environment
    experiment_name = "memetic_optimization"
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
    if os.path.exists(experiment_name + '/memetic_population.pkl'):
        # Continue evolution
        print("\nCONTINUING EVOLUTION\n")
        pop, fit_pop, ini_g = load_population_state(experiment_name)
    else:
        # New evolution
        print("\nNEW EVOLUTION\n")
        pop = initialize_population(npop, n_vars, dom_l, dom_u)
        fit_pop = evaluate_population(env, pop)
        ini_g = 0

    # Run the Memetic Algorithm
    print("\nRunning the Memetic Algorithm...")
    best_solution, best_fitness, best_fitness_list, mean_fitness_list = memetic_algorithm(env, pop, fit_pop, npop, gens, ini_g, n_vars, dom_l, dom_u, mutation_rate, experiment_name)

    # Output final results
    print(f"\nBest solution found after {gens} generations:\n{best_solution}")
    print(f"Best fitness achieved: {best_fitness}")

    # Save the final best solution and its fitness
    save_final_solution(experiment_name, best_solution, best_fitness)

    # Save the population state for future continuation
    save_population_state(pop, fit_pop, gens, experiment_name)

    # Plot the fitness over generations
    generations = list(range(ini_g + 1, gens + 1))
    plot_fitness(generations, best_fitness_list, mean_fitness_list, experiment_name)


if __name__ == "__main__":
    main()
