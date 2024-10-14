import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from joblib import Parallel, delayed
import shutil
import itertools
import csv

from evoman.environment import Environment
from demo_controller import player_controller


# Global Configuration
DEFAULT_HIDDEN_NEURONS = 10
DEFAULT_VARS = 265
DEFAULT_POP_SIZE = 100  # Population size per island
DEFAULT_GENS = 60 # Number of generations

# recombinantion parameters
COMMA_STRAT = True


# Tournament selection parameters
DEFAULT_TOURNAMENT_SIZE = 10 # Number of individuals in the tournament for parent selection

# params for mutation
mutation_rate = 1 # not really used anywhere
DEFAULT_TAU = 1 / np.sqrt(2 * np.sqrt(DEFAULT_VARS))
DEFAULT_TAU_PRIME = 1 / np.sqrt(2 * (DEFAULT_VARS))
DEFAULT_ALPHA = 0.5
DEFAULT_EPSILON = 1e-8

# Island model parameters
ISLAND_ENEMIES = [[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8], [1,4,6]] # enemies for each island
n_islands = len(ISLAND_ENEMIES)
migration_interval = 10 # Every n generations
migration_size = 50 # Number of individuals to migrate
migration_type = "random_best" # Can be "similarity" or "diversity" or "best" or "random_best"
TOP_PERCENT = 0.5 # percentage of the population to consider for migration if choosing random individuals
INIT_PERIOD = 30 #DEFAULT_GENS//2 - migration_interval # number of generations with the first set of enemies
NEW_ISLAND_ENEMIES = ISLAND_ENEMIES# [[1,2,4],[7,6,2],[1,2,4],[1,2,4]] # enemies for each island after the initial period (should be the same length as island_enemies)
CHANGE_ENEMIES_AFTER_INIT = False # change the enemies after the initial period


MIGRATE_ENEMIES=False # if True, the islands will migrate to the next enemy set after the initial period, make sure to set migration type to "best" and migration size is pop suze

# Initialisation parameters
INIT_POP_MU = 0  # mean of initial population
INIT_POP_SIGMA = 0.5  # std for initial population
STEPSIZE_MU = 0 # mean of step size
STEPSIZE_SIGMA = 0.5 # std for step size
dom_l, dom_u = -1, 1



ALL_ENEMIES = [1, 2, 3, 4, 5, 6, 7, 8] # now only used to initialise the environment
GET_STATS_AGAINST_ALL = False # get statistics against all enemies per generation
# why you would want this is because you want to see how well the population performs against all enemies not just the ones they were trained on
# you want to turn it off if you want faster training and only see at the end how it performed against all enemies


EXPERIMENT_NAME = f"island_{migration_type}_{migration_interval}_{migration_size}_{ISLAND_ENEMIES}_{COMMA_STRAT}"


headless = True # set to False to see the game (along with the visuals=True parameter in the environment setup)
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"



def setup_environment(experiment_name, controller, enemies, multiplemode="yes", visuals=False) -> Environment:
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    return Environment(
        experiment_name=experiment_name,
        playermode="ai",
        enemies=enemies,
        player_controller=controller,
        speed="fastest",
        enemymode="static",
        multiplemode=multiplemode,
        level=2,
        visuals=visuals,
    )
# function that initialises the environments for each island
def setup_island_environments(experiment_name, controller, enemies, multiplemode="yes", visuals=False):
    # check that the number of islands is the same as the number of enemy sets
    assert len(enemies) == n_islands, f"Number of islands and enemy sets must be the same. {n_islands} islands, {len(enemies)} enemy sets."

    envs = []
    for i, enemy in enumerate(enemies):
        env = setup_environment(f"{experiment_name}_island_{i}", controller, enemy, multiplemode, visuals)
        envs.append(env)
    return envs


# Fitness evaluation using parallel workers
def evaluate_fitnesses(env, population):
    fitnesses = Parallel(n_jobs=-1)(
        delayed(run_game_in_worker)(env.experiment_name, env.player_controller, env.enemies, ind) for ind in population
    )
    return fitnesses


def run_game_in_worker(experiment_name, controller, enemies, ind):
    env = setup_environment(experiment_name, controller, enemies)
    return simulation(env, ind)


# Simulation function
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f


# Recombination (Blend Crossover)
def blend_recombination(step_sizes, pop, fit_pop, n_vars, alpha=DEFAULT_ALPHA):
    if COMMA_STRAT:
        n_offspring = 3*len(pop)
    else:
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

# Parent selection using tournament
def select_parents_tournament(pop, fit_pop, tournament_size=DEFAULT_TOURNAMENT_SIZE):
    fit_pop = np.array(fit_pop)
    tournament_indices = np.random.randint(0, len(pop), tournament_size)
    best_parent_idx = tournament_indices[np.argmax(fit_pop[tournament_indices])]
    best_parent = pop[best_parent_idx]
    return best_parent_idx, best_parent

# Mutation (Gaussian Mutation)
def gaussian_mutation(individual, step_size, tau=DEFAULT_TAU, tau_prime=DEFAULT_TAU_PRIME, epsilon=DEFAULT_EPSILON):
    global_mutation = np.exp(tau * np.random.randn())
    local_mutation = tau_prime * np.random.randn(*step_size.shape)
    new_step_size = step_size * global_mutation + local_mutation # Update step size
    new_step_size[new_step_size < epsilon] = epsilon # Ensure step size is not too small
    new_individual = individual + new_step_size * np.random.randn(*individual.shape) # Apply mutation
    return new_individual, new_step_size





# Survivor selection (elitism)
def survivor_selection_elitism(pop, fit_pop, step_sizes, fit_offspring, offspring, offspring_step_size, pop_size):
    if COMMA_STRAT:
        offspring = np.array(offspring)
        offspring_step_size = np.array(offspring_step_size)
        fit_offspring = np.array(fit_offspring)
        elite_indices = np.argsort(fit_offspring)[-pop_size:]  # Sort and select best offspring only
        # Return only the top offspring
        return offspring[elite_indices], fit_offspring[elite_indices], offspring_step_size[elite_indices]

    else:
        combined_population = np.concatenate((pop, offspring), axis=0)
        combined_step_sizes = np.concatenate((step_sizes, offspring_step_size), axis=0)
        combined_fitness = np.concatenate((fit_pop, fit_offspring), axis=0)
        elite_indices = np.argsort(combined_fitness)[-pop_size:] # Select the best individuals of the combined population size pop_size
        return combined_population[elite_indices], combined_fitness[elite_indices], combined_step_sizes[elite_indices]
def blend_recombination(step_sizes, pop, fit_pop, n_vars, alpha=DEFAULT_ALPHA):
    if COMMA_STRAT:
        n_offspring = 3*len(pop)
    else:
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

# Migration logic (similarity and diversity-based)
def similarity(source_island, source_island_fit,destination_best, migration_size, TOP_PERCENT=TOP_PERCENT):
    """
    Find the most similar individuals in the source island to the best individual in the destination island
    """
    source_island_copy = source_island.copy()
    # get the best TOP_PERCENT
    best_indices = np.argsort(source_island_fit)[-migration_size:]
    source_island_copy = source_island[best_indices]


    most_similar = []
    for _ in range(migration_size):
        similarity_score = float('inf')
        most_similar_index = None  # Track the index of the most similar individual
        most_similar_ind = None

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


def diversity(source_island, source_island_fit, destination_best, migration_size):
    """
    Find the most diverse individuals in the source island to the best individual in the destination island
    """
    source_island_copy = source_island.copy()
    # get the best TOP_PERCENT
    best_indices = np.argsort(source_island_fit)[-migration_size:]
    source_island_copy = source_island[best_indices]

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

# function that returns the best individuals from source island
def best_individuals(source_island, source_island_fit, migration_size):
    """
    Return the best individuals from the source island
    """
    best_indices = np.argsort(source_island_fit)[-migration_size:]
    best_individuals = source_island[best_indices]
    return best_individuals

def pick_random_individuals(source_island, source_island_fit, migration_size, top_percent=TOP_PERCENT):
    """
    Pick random individuals from the source island from the top x percent
    """
    # get the top x percent of the population
    best_indices = np.argsort(source_island_fit)[-migration_size:]
    source_island_best = source_island[best_indices]

    random_individuals = random.choices(source_island_best, k=migration_size)

    return random_individuals

# Migration between islands
def migrate(world_population, world_pop_fit, migration_size, migration_type):
    for i, island in enumerate(world_population):
        best_individual_index = np.argmax(world_pop_fit[i])
        best_individual = island[best_individual_index] # Get the best individual of the island
        other_islands = [world_population[j] for j in range(len(world_population)) if j != i]
        source_island = random.choice(other_islands) # Choose a random island to migrate from (excluding the current island)

        if MIGRATE_ENEMIES:
            try:
                source_island = world_population[i+1]
            except IndexError:
                source_island = world_population[0]

        if migration_type == "similarity":
            source_island_index = [j for j in range(len(world_population)) if (world_population[j] == source_island).all()][0]
            migrants = similarity(source_island, world_pop_fit[source_island_index], best_individual, migration_size)
        elif migration_type == "diversity":
            source_island_index = [j for j in range(len(world_population)) if (world_population[j] == source_island).all()][0]
            migrants = diversity(source_island, world_pop_fit[source_island_index], best_individual, migration_size)
        elif migration_type == "best":
            source_island_index = [j for j in range(len(world_population)) if (world_population[j] == source_island).all()][0]
            migrants = best_individuals(source_island, world_pop_fit[source_island_index], migration_size)
        elif migration_type == "random_best":
            source_island_index = [j for j in range(len(world_population)) if (world_population[j] == source_island).all()][0]
            migrants = pick_random_individuals(source_island, world_pop_fit[source_island_index], migration_size)
        else:
            raise ValueError("Invalid migration type.")

        # Replace the worst individuals in the current island with the migrants
        worst_indices = np.argsort(world_pop_fit[i])[:migration_size]
        for idx, migrant in zip(worst_indices, migrants):
            world_population[i][idx] = migrant
    return world_population


# Evolutionary Strategy applied per island
def individual_island_run(island_env, island_population, pop_fit, step_sizes, mutation_rate, island_index):
    """
    Run the evolutionary strategy on a single island with its own population
    1. Recombination
    2. Mutation
    3. Survivor Selection
    """

    offspring, offspring_step_sizes = blend_recombination(step_sizes, island_population, pop_fit, DEFAULT_VARS)
    for i in range(len(offspring)):
        offspring[i], offspring_step_sizes[i] = gaussian_mutation(offspring[i], offspring_step_sizes[i])
    fit_offspring = evaluate_fitnesses(island_env, offspring)
    return survivor_selection_elitism(island_population, pop_fit, step_sizes, fit_offspring, offspring,
                                      offspring_step_sizes, DEFAULT_POP_SIZE)


# Parallel island execution
def parallel_island_run(envs, world_population, pop_fit, step_sizes, mutation_rate):
    for i in range(n_islands):
        world_population[i], pop_fit[i], step_sizes[i] = individual_island_run(envs[i], world_population[i], pop_fit[i],
                                                                               step_sizes[i], mutation_rate, island_index=i)
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
def plot_fitness(generations, best_fitness_list, mean_fitness_list, std_fitness_list, experiment_name, title="Fitness_Over_Generations"):
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness_list, label='Best Fitness', color='b', marker='o')
    plt.plot(generations, mean_fitness_list, label='Mean Fitness', color='g', marker='x')
    plt.fill_between(generations, np.array(mean_fitness_list) - np.array(std_fitness_list),
                        np.array(mean_fitness_list) + np.array(std_fitness_list), color='r', alpha=0.2, label='Std Dev Fitness')
    # plt.plot(generations, std_fitness_list, label='Standard Deviation', color='r', marker='s')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title(title)
    plt.suptitle(experiment_name)
    plt.legend()
    plt.grid(True)

    # Save the plot to the experiment directory
    plot_path = os.path.join(experiment_name, f'{title}.png')
    plt.savefig(plot_path)
    plt.show()


# Function to save results to a file
def save_generation_results(experiment_name, generation, best_fitness, mean_fitness, std_fitness):
    results_path = os.path.join(experiment_name, 'results_island_es.txt')
    with open(results_path, 'a') as file_aux:
        file_aux.write(
            f"Generation {generation + 1}: Best Fitness: {best_fitness:.6f}, Mean Fitness: {mean_fitness:.6f}, Standard Deviation Fitness: {std_fitness:.6f}\n")


# Function to save the final best solution and fitness
def save_final_solution(experiment_name, best_solution, best_fitness):
    solution_path = os.path.join(experiment_name, 'best_solution.txt')
    with open(solution_path, 'w') as file_aux:
        file_aux.write(f"Best Solution: {best_solution}\n")
        file_aux.write(f"Best Fitness: {best_fitness:.6f}\n")


def test_solution_against_all_enemies_loop(winner, title="Total gain against all enemies"):
    """
    Test the solution against all enemies in a loop with multipplemode disabled
    """

    all_enemies = [1, 2, 3, 4, 5, 6, 7, 8]
    controller = player_controller(DEFAULT_HIDDEN_NEURONS)
    controller.set(winner, n_inputs=20)  # Set the weights once

    total_gains = []
    total_fitnesses = []

    for enemy in all_enemies:
        # Set up the environment
        env_test_single = setup_environment("test_env", controller, enemies=[enemy], multiplemode="no", visuals=False)


        # Play the game using the controller with the loaded weights
        fitness, player_life, enemy_life, _ = env_test_single.play(pcont=winner)  # Pass weights, not the controller
        total_gain = player_life - enemy_life

        total_fitnesses.append(fitness)
        total_gains.append(total_gain)

        print(f"Enemy {enemy}: Fitness = {fitness}, Gain = {total_gain}")

    print(title)
    print(f"average fitness: {np.mean(total_fitnesses)}, average gain: {np.mean(total_gains)}")


    # count how many wins (gain > 0)
    wins = sum([1 for gain in total_gains if gain > 0])
    print(f"Wins: {wins} out of 8")
    # Plot the results
    plt.boxplot(total_gains)


    plt.title(title)
    plt.suptitle(f"Wins: {wins} out of 8 - trained on {ISLAND_ENEMIES}", fontsize=10)

    plt.show()

    return total_fitnesses, total_gains


def test_solution_against_all_enemies_multiplemode(winner):
    """
    Test the solution against all enemies with multiplemode enabled
    """

    all_enemies = [1, 2, 3, 4, 5, 6, 7, 8]
    controller = player_controller(DEFAULT_HIDDEN_NEURONS)
    controller.set(winner, n_inputs=20)  # Set the weights once

    # Set up the environment
    env_test_multiple = setup_environment("test_env_multiplemode", controller, enemies=all_enemies, multiplemode="yes", visuals=False)
    # env_test_multiple.update_parameter('speed', 'normal') # Set the speed to normal if you want to visualise

    # Play the game using the controller with the loaded weights
    fitnesses = []
    gains = []
    for _ in range(10):
        total_fitness, player_life, enemy_life, _ = env_test_multiple.play(pcont=winner)
        total_gain = player_life - enemy_life
        fitnesses.append(total_fitness)
        gains.append(total_gain)

   # box plot of total gains - always exactly the same?
   #  plt.boxplot(gains)
   #  plt.title("Total Gains Against All Enemies (10 games)")
   #  plt.suptitle(f"{EXPERIMENT_NAME}")
   #  plt.show()
    print(f"Multiplemode test: Mean Fitness = {np.mean(fitnesses)}, Mean Gain = {np.mean(gains)}")

    return total_fitness, total_gain




def evaluate_fitnesses_against_all(world_population):
    controller = player_controller(DEFAULT_HIDDEN_NEURONS)
    all_enemies_env = setup_environment("all_enemies", controller, ALL_ENEMIES, multiplemode="yes", visuals=False)
    fitness_against_all = []
    for island in world_population:
        island_fitnesses = evaluate_fitnesses(all_enemies_env, island)
        fitness_against_all.append(island_fitnesses)
    return fitness_against_all



# Main function
def main():
    GET_STATS_AGAINST_ALL = True
    ini = time.time()

    # Set up environment
    experiment_name = EXPERIMENT_NAME
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    controller = player_controller(DEFAULT_HIDDEN_NEURONS)
    # env = setup_environment(experiment_name, controller, ALL_ENEMIES)
    envs = setup_island_environments(experiment_name, controller, ISLAND_ENEMIES, visuals=False) # initialise the environments for each island
    n_weights = (envs[0].get_num_sensors() + 1) * DEFAULT_HIDDEN_NEURONS + (DEFAULT_HIDDEN_NEURONS + 1) * 5
    print(f"Environment setup with {n_weights} variables per individual.")

    print("Name of the experiment: ", experiment_name)
    all_enemies_env = setup_environment("all_enemies", controller, ALL_ENEMIES, multiplemode="yes", visuals=False)
    # Check if a previous state exists, otherwise start a new evolution
    if not os.path.exists(experiment_name + '/island_population.pkl'):
        print('NEW EVOLUTION')
        world_population = [np.clip(np.random.normal(INIT_POP_MU, INIT_POP_SIGMA, size=(DEFAULT_POP_SIZE, n_weights)), dom_l, dom_u) for _
                            in range(n_islands)]
        step_sizes = [np.random.normal(STEPSIZE_MU, STEPSIZE_SIGMA, size=(DEFAULT_POP_SIZE, n_weights)) for _ in range(n_islands)]
        world_pop_fit = [evaluate_fitnesses(envs[i], one_island_pop) for i, one_island_pop in enumerate(world_population)] # evaluate fitness for each island with different enemies
        ini_g = 0  # Start from generation 0
        best_fitness_list, mean_fitness_list, std_fitness_list = [], [], []
    else:
        print('CONTINUING EVOLUTION')
        world_population, world_pop_fit, ini_g = load_population_state(experiment_name)
        best_fitness_list, mean_fitness_list, std_fitness_list = [], [], []
        step_sizes = [np.random.normal(STEPSIZE_MU, STEPSIZE_SIGMA, size=(DEFAULT_POP_SIZE, n_weights)) for _ in range(n_islands)]

    best_fitness_against_all_list = []
    mean_fitness_against_all_list = []
    std_fitness_against_all_list = []

    best_fitness_outside_loop_All = None

    # Run the evolution process for generations
    for gen in range(ini_g, DEFAULT_GENS):

        world_population, world_pop_fit, step_sizes = parallel_island_run(envs, world_population, world_pop_fit,
                                                                          step_sizes, mutation_rate)

        # Compute fitness statistics
        best_fitness = np.max([np.max(fit) for fit in world_pop_fit])
        mean_fitness = np.mean([np.mean(fit) for fit in world_pop_fit])
        std_fitness = np.std([np.std(fit) for fit in world_pop_fit])

        # print(f"generation {gen}, enemies used: {envs[0].enemies}")
        print(f"\nGeneration {gen}, Best Fitness: {best_fitness:.6f}, Mean Fitness: {mean_fitness:.6f}, Standard Deviation Fitness: {std_fitness:.6f}")


            # get the statistics when testing against all enemies
        if GET_STATS_AGAINST_ALL:
            fitnesses_against_all = evaluate_fitnesses_against_all(world_population)
            best_fitness_against_all = np.max([np.max(fit) for fit in fitnesses_against_all])
            mean_fitness_against_all = np.mean([np.mean(fit) for fit in fitnesses_against_all])
            std_fitness_against_all = np.std([np.std(fit) for fit in fitnesses_against_all])
            print(f"Best Fitness Against All Enemies: {best_fitness_against_all:.6f}, Mean Fitness Against All Enemies: {mean_fitness_against_all:.6f}, Standard Deviation Fitness Against All Enemies: {std_fitness_against_all:.6f}")

            best_fitness_against_all_list.append(best_fitness_against_all)
            mean_fitness_against_all_list.append(mean_fitness_against_all)
            std_fitness_against_all_list.append(std_fitness_against_all)

            # Save the best solution OUTSIDE LOOP
            if best_fitness_outside_loop_All is None or best_fitness_against_all > best_fitness_outside_loop_All:
                best_island_idx_all = np.argmax([np.max(fit) for fit in fitnesses_against_all])
                best_individual_idx_all = np.argmax(fitnesses_against_all[best_island_idx_all])
                best_solution_outside_loop_All = world_population[best_island_idx_all][best_individual_idx_all].copy()  # Save the actual solution

        # Store fitness results for plotting
        best_fitness_list.append(best_fitness)
        mean_fitness_list.append(mean_fitness)
        std_fitness_list.append(std_fitness)

        # Save results for each generation
        save_generation_results(experiment_name, gen, best_fitness, mean_fitness, std_fitness)

        # Perform migration at intervals
        if gen == INIT_PERIOD and CHANGE_ENEMIES_AFTER_INIT:
            print("Initial period completed. Changing enemies")
            envs = setup_island_environments(experiment_name, controller, NEW_ISLAND_ENEMIES, visuals=False) # initialise the environments for each island


        if gen % migration_interval == 0 and gen >= INIT_PERIOD:
            world_population = migrate(world_population, world_pop_fit, migration_size, migration_type)
            world_pop_fit = [evaluate_fitnesses(envs[i], one_island_pop) for i, one_island_pop in enumerate(world_population)]
            print(f"Migration at generation {gen} completed.")

        # Save the population state every generation
        save_population_state(world_population, world_pop_fit, gen, experiment_name)

    # Plot and save the fitness over generations
    generations = list(range(ini_g + 1, DEFAULT_GENS + 1))
    plot_fitness(generations, best_fitness_list, mean_fitness_list, std_fitness_list, experiment_name, title = "Fitness Over Generations")

    # Plot and save the fitness against all enemies over generations
    if GET_STATS_AGAINST_ALL:
        plot_fitness(generations, best_fitness_against_all_list, mean_fitness_against_all_list, std_fitness_against_all_list, experiment_name, title = "Fitness_Against_All_Enemies")

    # Find the best solution across all islands
    best_island_idx = np.argmax([np.max(fit) for fit in world_pop_fit])
    best_individual_idx = np.argmax(world_pop_fit[best_island_idx])
    best_solution = world_population[best_island_idx][best_individual_idx]
    best_fitness = world_pop_fit[best_island_idx][best_individual_idx]

    # find the best solution across all islands based on the best fitness against all enemies
    if GET_STATS_AGAINST_ALL:
        best_island_idx_all = np.argmax([np.max(fit) for fit in fitnesses_against_all])
        best_individual_idx_all = np.argmax(fitnesses_against_all[best_island_idx_all])
        best_solution_all = world_population[best_island_idx_all][best_individual_idx_all]
        best_fitness_all = fitnesses_against_all[best_island_idx_all][best_individual_idx_all]


        # print(f"Best solution shape against all enemies: {best_solution_all.shape}")
        # print(f"Best fitness against all enemies: {best_fitness_all}")

    # print(f"Best solution shape: {best_solution.shape}")
    # print(f"Best fitness: {best_fitness}")

    # Test the best solution
    total_fitnesses, total_gains = test_solution_against_all_enemies_loop(best_solution, title="picked based on subset")
    test_solution_against_all_enemies_multiplemode(best_solution)

    if GET_STATS_AGAINST_ALL:
        print("Testing the best solution obtained by checking on all enemies")
        total_fitnesses_all, total_gains_all = test_solution_against_all_enemies_loop(best_solution_all, title="picked based on all enemies")
        test_solution_against_all_enemies_multiplemode(best_solution_all)

        test_solution_against_all_enemies_loop(best_solution_outside_loop_All, title="outside loop")


    # Save the final solution
    save_final_solution(experiment_name, best_solution, best_fitness)

    # delete folders
    for i in range(n_islands):
        shutil.rmtree(f"{EXPERIMENT_NAME}_island_{i}")
    shutil.rmtree("test_env")
    shutil.rmtree("test_env_multiplemode")
    # shutil.rmtree(EXPERIMENT_NAME)




    print(f"\nExecution time: {round((time.time() - ini) / 60, 2)} minutes")
    return total_gains, total_gains_all


if __name__ == "__main__":
    # grid_search_island_model_parameters()
    main()





# grid search
#
# def set_global_params(ISLAND_ENEMIES_GRID, migration_interval_GRID, migration_size_GRID, migration_type_GRID, TOP_PERCENT_GRID, INIT_PERIOD_GRID, COMMA_STRAT_GRID,
#                       DEFAULT_POP_SIZE_GRID, DEFAULT_GENS_GRID, INIT_POP_MU_GRID, INIT_POP_SIGMA_GRID, STEPSIZE_MU_GRID, STEPSIZE_SIGMA_GRID, experiment_name_grid):
#     global n_islands, MIGRATE_ENEMIES, headless, migration_interval, migration_size, migration_type
#     global TOP_PERCENT, INIT_PERIOD, COMMA_STRAT, DEFAULT_POP_SIZE, DEFAULT_GENS
#     global INIT_POP_MU, INIT_POP_SIGMA, STEPSIZE_MU, STEPSIZE_SIGMA, EXPERIMENT_NAME
#
#     # Update island-related parameters
#     n_islands = len(ISLAND_ENEMIES_GRID)
#     MIGRATE_ENEMIES = False  # We are not migrating enemies
#     headless = True  # Ensure the execution is headless (no display)
#
#     # Update the parameters globally
#     migration_interval = migration_interval_GRID
#     migration_size = migration_size_GRID
#     migration_type = migration_type_GRID
#     TOP_PERCENT = TOP_PERCENT_GRID
#     INIT_PERIOD = INIT_PERIOD_GRID
#     COMMA_STRAT = COMMA_STRAT_GRID
#
#     # Update initialization parameters globally
#     DEFAULT_POP_SIZE = DEFAULT_POP_SIZE_GRID
#     DEFAULT_GENS = DEFAULT_GENS_GRID
#     INIT_POP_MU = INIT_POP_MU_GRID
#     INIT_POP_SIGMA = INIT_POP_SIGMA_GRID
#     STEPSIZE_MU = STEPSIZE_MU_GRID
#     STEPSIZE_SIGMA = STEPSIZE_SIGMA_GRID
#
#     # Update the experiment name
#     EXPERIMENT_NAME = experiment_name_grid
#
#     print(f"Global parameters set: \n"
#           f"ISLAND_ENEMIES: {ISLAND_ENEMIES_GRID}, \n"
#           f"migration_interval: {migration_interval}, \n"
#           f"migration_size: {migration_size}, \n"
#           f"migration_type: {migration_type}, \n"
#           f"TOP_PERCENT: {TOP_PERCENT}, \n"
#           f"INIT_PERIOD: {INIT_PERIOD}, \n"
#           f"COMMA_STRAT: {COMMA_STRAT}, \n"
#           f"DEFAULT_POP_SIZE: {DEFAULT_POP_SIZE}, \n"
#           f"DEFAULT_GENS: {DEFAULT_GENS}, \n"
#           f"INIT_POP_MU: {INIT_POP_MU}, \n"
#           f"INIT_POP_SIGMA: {INIT_POP_SIGMA}, \n"
#           f"STEPSIZE_MU: {STEPSIZE_MU}, \n"
#           f"STEPSIZE_SIGMA: {STEPSIZE_SIGMA}, \n"
#           f"EXPERIMENT_NAME: {EXPERIMENT_NAME}")
#
# def grid_search_island_model_parameters():
#     # Define the hyperparameter grid
#     param_grid = {
#         'ISLAND_ENEMIES_GRID': [
#             [[2,4], [2,6], [7,8], [1,5]],
#              [[1,2,5], [2,5,6], [1,7,8], [1,5,6]],
#              [[2,3,4,8], [1,2,5,6], [1,5,7,8], [2,4,7,8]],
#              [[2,4],[7,8], [1,5], [3,6]],
#         ],
#         'migration_interval_GRID': [10, 20],
#         'migration_size_GRID': [10, 20, 30],
#         'migration_type_GRID': ['similarity', 'diversity', 'best', 'random_best'],
#         'TOP_PERCENT_GRID': [0.25, 0.5, 0.75],
#         'INIT_PERIOD_GRID': [0],
#         "COMMA_STRAT": [True, False],
#     }
#
#     # Prepare to store results
#     output_file = 'grid_search_results_total_gain_FINAL.txt'
#     output_file_csv = 'grid_search_results_total_gain_FINAL.csv'
#
#     # Generate all combinations of parameters from the grid
#     grid_combinations = list(itertools.product(*param_grid.values()))
#
#     # Iterate over all combinations
#     for combo in grid_combinations:
#         ISLAND_ENEMIES_GRID, migration_interval_GRID, migration_size_GRID, migration_type_GRID, TOP_PERCENT_GRID, INIT_PERIOD_GRID, COMMA_STRAT_GRID = combo
#
#         experiment_name_grid = f"grid_search_{migration_type_GRID}_{migration_interval_GRID}_{migration_size_GRID}_{ISLAND_ENEMIES_GRID}_{TOP_PERCENT_GRID}_{INIT_PERIOD_GRID}_{COMMA_STRAT_GRID}"
#
#         # Set global parameters using the helper function
#         set_global_params(ISLAND_ENEMIES_GRID, migration_interval_GRID, migration_size_GRID, migration_type_GRID, TOP_PERCENT_GRID, INIT_PERIOD_GRID, COMMA_STRAT_GRID,
#                           DEFAULT_POP_SIZE, DEFAULT_GENS, INIT_POP_MU, INIT_POP_SIGMA, STEPSIZE_MU, STEPSIZE_SIGMA, experiment_name_grid)
#
#         # Run the evolutionary strategy with this set of parameters
#         try:
#             # Run the main function, which executes the experiment with the current parameters
#             total_gains, total_gains_all = main()  # Assuming that `main()` now returns the total gain
#             no_wins = sum([1 for gain in total_gains if gain > 0])
#             no_wins_all = sum([1 for gain in total_gains_all if gain > 0])
#
#             # Log results to file
#             with open(output_file, 'a') as f:
#                 f.write(f"Parameters: {combo}\n")
#                 f.write(f"Total Gain: {total_gains}\n")
#                 f.write(f"Total Gain chosen on all Enemies: {total_gains_all}\n")
#                 f.write(f"Number of wins: {no_wins}\n")
#                 f.write(f"Number of wins chosen on all enemies: {no_wins_all}\n")
#                 f.write("============================================================\n")
#             # Open the CSV file in append mode
#             with open(output_file_csv, 'a', newline='') as csvfile:
#                 csvwriter = csv.writer(csvfile)
#
#                 # Write the header if the file is empty
#                 if csvfile.tell() == 0:
#                     csvwriter.writerow(['ISLAND_ENEMIES', 'migration_interval', 'migration_size', 'migration_type',
#                                         'TOP_PERCENT', 'INIT_PERIOD', "COMMA_STRAT",'Total Gain', 'Total Gain chosen on All Enemies', 'Number of Wins', 'Number of Wins Chosen on All Enemies'])
#
#                 # Log results to CSV file
#                 csvwriter.writerow(
#                     [ISLAND_ENEMIES_GRID, migration_interval_GRID, migration_size_GRID, migration_type_GRID,
#                      TOP_PERCENT_GRID, INIT_PERIOD_GRID, COMMA_STRAT_GRID, total_gains, total_gains_all, no_wins, no_wins_all])
#
#
#         except Exception as e:
#             # If there's an error in this combination, log it and continue
#             print(f"Error with parameters {combo}: {e}")
#             with open(output_file, 'a') as f:
#                 f.write(f"Error with parameters {combo}: {e}\n")
#                 f.write("============================================================\n")
#             # Open the CSV file in append
#             with open(output_file, 'a', newline='') as csvfile:
#                 csvwriter = csv.writer(csvfile)
#                 csvwriter.writerow(
#                     [ISLAND_ENEMIES_GRID, migration_interval_GRID, migration_size_GRID, migration_type_GRID,
#                      TOP_PERCENT_GRID, INIT_PERIOD_GRID, f"Error: {e}"])
#
#
#         # shutil.rmtree(experiment_name_grid) # Delete the experiment folder after each run
#
#
#     return