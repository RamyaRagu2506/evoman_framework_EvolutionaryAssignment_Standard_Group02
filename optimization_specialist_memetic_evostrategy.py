import sys
import numpy as np
import random
import os
from memetic_controller import player_controller
from evoman.environment import Environment
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed

# Global Configuration
DEFAULT_HIDDEN_NEURONS = 10
DEFAULT_POP_SIZE = 100
DEFAULT_GENS = 30
DEFAULT_ENEMY = 2
DEFAULT_TAU = 1 / np.sqrt(DEFAULT_POP_SIZE)
DEFAULT_ALPHA = 0.5
REPLACEMENT_FACTOR = 4  # For doomsday replacement
LOCAL_SEARCH_ITER = 5  # Number of iterations for hill climbing

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

# Simulation function
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

# Save population and fitness state
def save_population_state(pop, fitness, generation, experiment_name):
    path = os.path.join(experiment_name, 'memetic_population.pkl')
    with open(path, 'wb') as f:
        pickle.dump([pop, fitness, generation], f)
    print("Population state saved successfully.")

# Load population and fitness state
def load_population_state(experiment_name):
    path = os.path.join(experiment_name, 'memetic_population.pkl')
    with open(path, 'rb') as f:
        pop, fitness, generation = pickle.load(f)
    print("Population state loaded successfully.")
    return pop, fitness, generation

# Fitness evaluation
def evaluate_population(env, pop):
    return np.array([env.play(pcont=individual)[0] for individual in pop])


def evaluate_fitnesses(env, population):
    fitnesses = Parallel(n_jobs=-1)(
        delayed(run_game_in_worker)(env.experiment_name, env.player_controller, ind) for ind in population
    )
    return fitnesses

def run_game_in_worker(experiment_name, controller, ind):
    os.environ["SDL_VIDEODRIVER"] = "dummy"  # to not open the pygame window
    env = setup_environment(experiment_name, DEFAULT_ENEMY, controller)
    return simulation(env, ind)

# Parent selection methods
def select_parents_tournament(pop, fit_pop, tournament_size=10):
    fit_pop = np.array(fit_pop)
    parents = []
    for _ in range(len(pop)):
        tournament_indices = np.random.randint(0, pop.shape[0], tournament_size)
        tournament = fit_pop[tournament_indices]
        best_parent_idx = np.argmax(tournament)
        parents.append(tournament_indices[best_parent_idx])
    return parents

# Recombination: Blend Recombination
def blend_recombination(parent1, parent2, alpha=DEFAULT_ALPHA):
    diff = np.abs(parent1 - parent2)
    min_vals = np.minimum(parent1, parent2) - diff * alpha
    max_vals = np.maximum(parent1, parent2) + diff * alpha
    return np.random.uniform(min_vals, max_vals)

# Mutation: Gaussian
def gaussian_mutation(individual, step_size, tau=DEFAULT_TAU):
    new_step_size = step_size * np.exp(tau * np.random.randn(*step_size.shape))
    new_individual = individual + new_step_size * np.random.randn(*individual.shape)
    return new_individual, new_step_size

# Survivor Selection: Elitism with Doomsday Selection
def survivor_selection_elitism(pop, fit_pop, offspring, fit_offspring, step_sizes, offspring_step_sizes, pop_size):
    combined_pop = np.concatenate((pop, offspring), axis=0)
    combined_fitness = np.concatenate((fit_pop, fit_offspring), axis=0)
    combined_step_sizes = np.concatenate((step_sizes, offspring_step_sizes), axis=0)

    elite_idx = np.argsort(combined_fitness)[-pop_size:]  # Select best
    return combined_pop[elite_idx], combined_fitness[elite_idx], combined_step_sizes[elite_idx]

def doomsday_selection(pop, fit_pop, step_sizes, pop_size, replacement_factor=REPLACEMENT_FACTOR):
    worst_indices = np.argsort(fit_pop)[:pop_size // replacement_factor]
    pop[worst_indices] = np.random.uniform(-1, 1, size=pop[worst_indices].shape)
    step_sizes[worst_indices] = np.random.uniform(0.1, 0.5, size=step_sizes[worst_indices].shape)
    return pop, step_sizes

# Local Search: Hill Climbing
def hill_climb(env, individual, mutation_rate, n_iterations=LOCAL_SEARCH_ITER):
    best_individual = individual.copy()
    best_fitness = simulation(env, best_individual)
    for _ in range(n_iterations):
        new_individual = individual + mutation_rate * np.random.randn(*individual.shape)
        new_fitness = simulation(env, new_individual)
        if new_fitness > best_fitness:
            best_individual, best_fitness = new_individual, new_fitness
    return best_individual, best_fitness

# Plot fitness over generations
def plot_fitness(generations, best_fitness_list, mean_fitness_list, experiment_name):
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness_list, label='Best Fitness', color='b', marker='o')
    plt.plot(generations, mean_fitness_list, label='Mean Fitness', color='g', marker='x')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations - Memetic Algorithm')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(experiment_name, 'fitness_over_generations.png')
    plt.savefig(plot_path)
    plt.show()

# Save generation results
def save_generation_results(experiment_name, generation, best_fitness, mean_fitness, std_fitness):
    results_path = os.path.join(experiment_name, 'results_memetic.txt')
    with open(results_path, 'a') as file_aux:
        file_aux.write(f"Generation {generation + 1}: Best Fitness: {best_fitness:.6f}, Mean Fitness: {mean_fitness:.6f}, Std Fitness: {std_fitness:.6f}\n")

# Memetic Algorithm: Global search (ES) + Local Search (Hill Climbing)
def memetic_algorithm(env, pop, fit_pop, npop, gens, ini_g, n_vars, dom_l, dom_u, mutation_rate, experiment_name):
    print(f"Starting Memetic Algorithm with {npop} individuals and {gens} generations...\n")
    
    best_fitness_list = []
    mean_fitness_list = []

    step_sizes = np.random.uniform(0.1, 0.5, size=(npop, n_vars))  # Mutation step sizes

    for generation in range(ini_g, gens):
        print(f"\n========== Generation {generation + 1}/{gens} ==========")
        
        # Parent selection - tournament 
        Parents = select_parents_tournament(pop, fit_pop)

        # Recombination and Mutation
        offspring = []
        offspring_step_sizes = []
        for i in range(0, npop, 2):
            parent1_idx= Parents[i], 
            parent2_idx =Parents[i + 1]
            parent1 = pop [parent1_idx]
            parent2 = pop [parent2_idx]
            child1 = blend_recombination(parent1, parent2)
            child2 = blend_recombination(parent2, parent1)
            child1, step1 = gaussian_mutation(child1, step_sizes[parent1_idx])
            child2, step2 = gaussian_mutation(child2, step_sizes[parent2_idx])
            offspring.append(child1)
            offspring.append(child2)
            offspring_step_sizes.append(step1)
            offspring_step_sizes.append(step2)

        offspring = np.array(offspring)
        offspring_step_sizes = np.array(offspring_step_sizes)

        # Evaluate offspring
        # fit_offspring = evaluate_population(env, offspring)
        fit_offspring = evaluate_fitnesses(env, offspring)

        # Apply local search (Hill Climbing) to offspring
        for i, individual in enumerate(offspring):
            refined_individual, refined_fitness = hill_climb(env, individual, mutation_rate)
            offspring[i], fit_offspring[i] = refined_individual, refined_fitness

        # Survivor Selection with Elitism and Doomsday
        pop, fit_pop, step_sizes = survivor_selection_elitism(pop, fit_pop, offspring, fit_offspring, step_sizes, offspring_step_sizes, npop)

        # Occasionally apply doomsday
        if generation % 10 == 0:
            print("Applying Doomsday Selection...")
            pop, step_sizes = doomsday_selection(pop, fit_pop, step_sizes, npop)

        # Log results for plotting
        best_fitness = np.max(fit_pop)
        mean_fitness = np.mean(fit_pop)
        std_fitness = np.std(fit_pop)
        best_fitness_list.append(best_fitness)
        mean_fitness_list.append(mean_fitness)
        
        print(f"Generation {generation + 1}: Best Fitness: {best_fitness:.6f}, Mean Fitness: {mean_fitness:.6f}, Std Deviation: {std_fitness:.6f}")
        save_generation_results(experiment_name, generation, best_fitness, mean_fitness, std_fitness)

    # Return the best solution
    best_idx = np.argmax(fit_pop)
    return pop[best_idx], fit_pop[best_idx], best_fitness_list, mean_fitness_list

# Main function to run the memetic algorithm
def main():
    npop = 100
    gens = 30
    mutation_rate = 0.1
    n_hidden_neurons = 10
    dom_l, dom_u = -1, 1

    # Set up environment
    experiment_name = "memetic_optimization_es_v2_enemy2"
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    env = setup_environment(experiment_name, DEFAULT_ENEMY, player_controller(n_hidden_neurons))
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
        pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
        # fit_pop = evaluate_population(env, pop)
        fit_pop = evaluate_fitnesses(env, pop)
        ini_g = 0

    # Run the Memetic Algorithm
    best_solution, best_fitness, best_fitness_list, mean_fitness_list = memetic_algorithm(
        env, pop, fit_pop, npop, gens, ini_g, n_vars, dom_l, dom_u, mutation_rate, experiment_name)

    # Output final results
    print(f"\nBest solution found after {gens} generations:\n{best_solution}")
    print(f"Best fitness achieved: {best_fitness}")

    # Save the final best solution and its fitness
    np.savetxt(os.path.join(experiment_name, 'best_solution.txt'), best_solution)
    
    # Save the population state for future continuation
    save_population_state(pop, fit_pop, gens, experiment_name)

    # Plot the fitness over generations
    generations = list(range(ini_g + 1, gens + 1))
    plot_fitness(generations, best_fitness_list, mean_fitness_list, experiment_name)


if __name__ == "__main__":
    main()
