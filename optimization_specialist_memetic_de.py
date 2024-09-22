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
    population_path = os.path.join(experiment_name, 'memetic_population_de.pkl')
    with open(population_path, 'wb') as f:
        pickle.dump([population, fitness, generation], f)
    print("Population state saved successfully.")

# Load the population and fitness values (solution state)
def load_population_state(experiment_name):
    population_path = os.path.join(experiment_name, 'memetic_population_de.pkl')
    with open(population_path, 'rb') as f:
        population, fitness, generation = pickle.load(f)
    print("Population state loaded successfully.")
    return population, fitness, generation

# Global search (DE): Initialize population
def initialize_population(npop, n_vars, dom_l, dom_u):
    print("Initializing population...")
    return np.random.uniform(dom_l, dom_u, (npop, n_vars))

# Evaluate fitness of the population
def evaluate_population(env, pop):
    print("Evaluating fitness for the population...")
    return np.array([env.play(pcont=individual)[0] for individual in pop])

# Differential Evolution mutation and crossover
def de_mutation(pop, F=0.8):
    """Differential Evolution mutation: mutation using a weighted difference."""
    new_pop = []
    for i in range(len(pop)):
        candidates = list(range(0, len(pop)))
        candidates.remove(i)
        a, b, c = pop[random.sample(candidates, 3)]
        mutant = np.clip(a + F * (b - c), -1, 1)  # Mutation step
        new_pop.append(mutant)
    return np.array(new_pop)

def de_crossover(pop, mutated_pop, crossover_rate=0.9):
    """Differential Evolution crossover: combines target and mutant vectors."""
    offspring = []
    for i in range(len(pop)):
        trial = np.copy(pop[i])
        for j in range(len(pop[i])):
            if random.random() < crossover_rate or j == random.randint(0, len(pop[i]) - 1):
                trial[j] = mutated_pop[i][j]  # Cross genes from mutant
        offspring.append(trial)
    return np.array(offspring)

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
    plt.title('Fitness over Generations - Memetic Algorithm with DE')
    plt.legend()
    plt.grid(True)
    
    # Save the plot as an image
    plot_path = os.path.join(experiment_name, 'fitness_over_generations.png')
    plt.savefig(plot_path)
    plt.show()

# Function to save results to a file
def save_generation_results(experiment_name, generation, best_fitness, mean_fitness, std_fitness):
    results_path = os.path.join(experiment_name, 'results_memetic_de.txt')
    with open(results_path, 'a') as file_aux:
        file_aux.write(f"Generation {generation + 1}: Best Fitness: {best_fitness:.6f}, Mean Fitness: {mean_fitness:.6f}, Standard Deviation Fitness: {std_fitness:.6f}\n")

# Save the final best solution and fitness
def save_final_solution(experiment_name, best_solution, best_fitness):
    solution_path = os.path.join(experiment_name, 'best_solution_de.txt')
    with open(solution_path, 'w') as file_aux:
        file_aux.write(f"Best Solution: {best_solution}\n")
        file_aux.write(f"Best Fitness: {best_fitness:.6f}\n")

# Memetic Algorithm: Global search (DE) + local search (hill climbing)
def memetic_algorithm_de(env, pop, fit_pop, npop, gens, ini_g, n_vars, mutation_rate, experiment_name):
    print(f"Starting Memetic Algorithm with DE and {npop} individuals over {gens} generations...\n")
    
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

        # Mutation (Differential Evolution mutation)
        mutated_pop = de_mutation(pop)

        # Crossover (Differential Evolution crossover)
        offspring = de_crossover(pop, mutated_pop)

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

# Main function to run the memetic algorithm with DE
def main():
    # Parameters
    npop = 100
    gens = 30
    mutation_rate = 0.1
    n_hidden_neurons = 10
    dom_l, dom_u = -1, 1

    # Set up environment
    experiment_name = "memetic_optimization_de_enemy2"
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
    if os.path.exists(experiment_name + '/memetic_population_de.pkl'):
        # Continue evolution
        print("\nCONTINUING EVOLUTION\n")
        pop, fit_pop, ini_g = load_population_state(experiment_name)
    else:
        # New evolution
        print("\nNEW EVOLUTION\n")
        pop = initialize_population(npop, n_vars, dom_l, dom_u)
        fit_pop = evaluate_population(env, pop)
        ini_g = 0

    # Run the Memetic Algorithm with DE
    print("\nRunning the Memetic Algorithm with DE...")
    best_solution, best_fitness, best_fitness_list, mean_fitness_list = memetic_algorithm_de(env, pop, fit_pop, npop, gens, ini_g, n_vars, mutation_rate, experiment_name)

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
