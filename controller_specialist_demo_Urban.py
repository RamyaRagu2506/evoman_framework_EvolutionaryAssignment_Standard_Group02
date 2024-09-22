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


# Disable the Pygame window
os.environ["SDL_VIDEODRIVER"] = "dummy"

# EvoMan Framework Imports
from evoman.environment import Environment
from demo_controller import player_controller

# Global Configuration
DEFAULT_HIDDEN_NEURONS = 10
DEFAULT_POP_SIZE = 100
DEFAULT_GENS = 100
DEFAULT_ENEMY = 8
DEFAULT_TAU = 1 / np.sqrt(DEFAULT_POP_SIZE)
DEFAULT_ALPHA = 0.5


# Argument Parsing
def get_args():
    parser = argparse.ArgumentParser(description='EvoMan Genetic Algorithm Experiment')
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


# Parent selection methods
def select_parents_tournament(pop, fit_pop, tournament_size=10):
    fit_pop = np.array(fit_pop)
    tournament_indices = np.random.randint(0, pop.shape[0], tournament_size)
    tournament = fit_pop[tournament_indices]
    best_parent_idx = np.argmax(tournament)
    best_parent = pop[tournament_indices[best_parent_idx]]
    return tournament_indices[best_parent_idx], best_parent


# Recombination methods
def blend_recombination(step_sizes, pop, fit_pop, n_vars, alpha=DEFAULT_ALPHA):
    n_offspring = np.random.randint(DEFAULT_POP_SIZE + 1, DEFAULT_POP_SIZE * 2)
    offspring = np.zeros((n_offspring, n_vars))
    offspring_step_size = np.zeros((n_offspring, n_vars))

    for i in range(n_offspring):
        parent_idx1, parent1 = select_parents_tournament(pop, fit_pop)
        parent_idx2, parent2 = select_parents_tournament(pop, fit_pop)
        differece = np.abs(parent1 - parent2)
        min_values = np.minimum(parent1, parent2) - differece * alpha
        max_values = np.maximum(parent1, parent2) + differece * alpha
        offspring[i] = np.random.uniform(min_values, max_values)
        offspring_step_size[i] = np.mean(np.stack((step_sizes[parent_idx1], step_sizes[parent_idx2])), axis=0)
    return offspring, offspring_step_size


# Mutation
def mutate(individual, step_size, tau=DEFAULT_TAU):
    new_step_size = step_size * np.exp(tau * np.random.randn(*step_size.shape))
    new_individual = individual + new_step_size * np.random.randn(*individual.shape)
    return new_individual, new_step_size


# Survivor selection
def survivor_selection(pop, fit_pop, step_sizes, pop_size):
    elite_idx = np.argsort(fit_pop)[-pop_size:]
    pop = pop[elite_idx]
    step_sizes = step_sizes[elite_idx]
    return pop, step_sizes

def init_population(pop_size, env, n_vars):
    pop = np.random.uniform(low=-1, high=1, size=(pop_size, n_vars))
    step_sizes = np.random.uniform(low=0.1, high=0.2, size=(pop_size, n_vars))
    fit_pop = evaluate_fitnesses(env, pop)
    return pop, step_sizes, fit_pop

# Main genetic algorithm function
def genetic_algorithm(env, pop_size, gens, tau):
    n_vars = (env.get_num_sensors() + 1) * DEFAULT_HIDDEN_NEURONS + (DEFAULT_HIDDEN_NEURONS + 1) * 5

    # initialize population
    pop, step_sizes, fit_pop = init_population(pop_size, env, n_vars)

    # keep track of the best solution
    best_solution = pop[np.argmax(fit_pop)]
    best_fitness = np.max(fit_pop)

    for gen in range(gens):
        offspring, offspring_step_size = blend_recombination(step_sizes, pop, fit_pop, n_vars)
        offspring, offspring_step_size = mutate(offspring, offspring_step_size, tau)
        fit_offspring = evaluate_fitnesses(env, offspring)
        pop, step_sizes = survivor_selection(offspring, fit_offspring, offspring_step_size, pop_size)
        fit_pop = evaluate_fitnesses(env, pop)

        if np.max(fit_pop) > best_fitness:
            best_solution = pop[np.argmax(fit_pop)]
            best_fitness = np.max(fit_pop)
            print(f"New best solution found at generation {gen} with fitness: {best_fitness}")
        else: # add the best solution to the population
            sorted_indices = np.argsort(fit_pop) # sort the population by fitness
            # replace the worst solution with the best solution
            pop[sorted_indices[0]] = best_solution
            step_sizes[sorted_indices[0]] = step_sizes[np.argmax(fit_pop)]
            fit_pop[sorted_indices[0]] = best_fitness

        print(f"Generation {gen} - Best Fitness: {round(max(fit_pop),6)} - Mean Fitness: {round(np.mean(fit_pop), 6)} - Std Fitness: {round(np.std(fit_pop),6)}")

    return best_solution, best_fitness


# Main function
def main():
    args = get_args()

    # Controller and Environment Setup
    controller = player_controller(args.n_hidden_neurons)
    env = setup_environment(args.experiment_name, args.enemies, controller)

    if args.run_mode == 'test':
        bsol = np.loadtxt(args.experiment_name + '/best.txt')
        print('RUNNING SAVED BEST SOLUTION')
        env.update_parameter('speed', 'normal')
        evaluate_fitnesses(env, [bsol])
        sys.exit(0)

    print('STARTING EVOLUTION')
    best_solution, best_fitness = genetic_algorithm(env, args.pop_size, args.gens, DEFAULT_TAU)
    print(f'Best solution found with fitness: {best_fitness}')

    # Save best solution
    np.savetxt(args.experiment_name + '/best.txt', best_solution)


if __name__ == '__main__':
    main()