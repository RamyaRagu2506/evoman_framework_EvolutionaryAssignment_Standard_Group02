# optimization_specialist_memetic.py

import sys
import time
import numpy as np
import random
import os
from memetic_controller import player_controller
from evoman.environment import Environment


# Global search (GA): Initialize population
def initialize_population(npop, n_vars, dom_l, dom_u):
    return np.random.uniform(dom_l, dom_u, (npop, n_vars))


# Evaluate fitness of the population
def evaluate_population(env, pop):
    return np.array([env.play(pcont=individual)[0] for individual in pop])


# Selection based on fitness (roulette wheel selection)
def selection(pop, fit_pop):
    fit_sum = np.sum(fit_pop)
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
def hill_climb(env, individual, mutation_rate, n_iterations=10):
    best_fitness = env.play(pcont=individual)[0]
    best_individual = individual.copy()

    for _ in range(n_iterations):
        new_individual = mutation(individual.copy(), mutation_rate, -1, 1)
        new_fitness = env.play(pcont=new_individual)[0]
        if new_fitness > best_fitness:
            best_fitness = new_fitness
            best_individual = new_individual

    return best_individual, best_fitness


# Memetic Algorithm: Global search (GA) + local search (hill climbing)
def memetic_algorithm(env, npop, gens, n_vars, dom_l, dom_u, mutation_rate):
    # Initialize population
    pop = initialize_population(npop, n_vars, dom_l, dom_u)
    fit_pop = evaluate_population(env, pop)

    for generation in range(gens):
        print(f"\nGeneration {generation + 1}")

        # Selection
        selected_pop = selection(pop, fit_pop)

        # Create new population with crossover and mutation
        offspring = []
        for i in range(0, npop, 2):
            parent1, parent2 = selected_pop[i], selected_pop[i + 1]
            child1 = crossover(parent1, parent2, n_vars)
            child2 = crossover(parent2, parent1, n_vars)
            offspring.append(mutation(child1, mutation_rate, dom_l, dom_u))
            offspring.append(mutation(child2, mutation_rate, dom_l, dom_u))

        offspring = np.array(offspring)

        # Local search (hill climbing) applied to offspring
        refined_offspring = []
        for individual in offspring:
            refined_individual, _ = hill_climb(env, individual, mutation_rate)
            refined_offspring.append(refined_individual)

        pop = np.array(refined_offspring)
        fit_pop = evaluate_population(env, pop)

        # Log results
        best_fitness = np.max(fit_pop)
        mean_fitness = np.mean(fit_pop)
        print(f"Best Fitness: {best_fitness}, Mean Fitness: {mean_fitness}")

    best_individual_idx = np.argmax(fit_pop)
    return pop[best_individual_idx], fit_pop[best_individual_idx]


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
                      visuals=True)

    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    # Run the Memetic Algorithm
    best_solution, best_fitness = memetic_algorithm(env, npop, gens, n_vars, dom_l, dom_u, mutation_rate)
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness achieved: {best_fitness}")


if __name__ == "__main__":
    main()
