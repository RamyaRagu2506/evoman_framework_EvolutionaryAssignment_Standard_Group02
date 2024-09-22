import sys
sys.stdout.flush = lambda: None  # Disable output buffering

# EvoMan Framework - V1.0 2016
# DEMO: Neuroevolution - Genetic Algorithm neural network.

# imports framework
import sys
from evoman.environment import Environment
from neat_demo_controller import player_controller

# imports other libs
import numpy as np
import os
import neat
import pickle

# runs simulation
def simulation(env, controller):
    f, p, e, t = env.play(pcont=controller)
    return f

# Evaluation function for NEAT genomes
def eval_genomes(genomes, config, env):
    for genome_id, genome in genomes:
        # Create a neural network from the genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # Try to simulate the game and assign fitness, set default if it fails
        try:
            genome.fitness = simulation(env, net)
        except Exception as e:
            print(f"Error during simulation for genome {genome_id}: {e}", flush=True)
            genome.fitness = 0  # Assign default fitness if there's an error


# Initialize NEAT population
def initialize_neat_population(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    
    # Initialize population
    neat_population = neat.Population(config)
    
    # Add reporters
    neat_population.add_reporter(neat.StdOutReporter(True))
    stats_neat = neat.StatisticsReporter()
    neat_population.add_reporter(stats_neat)

    return neat_population, config

# Save the best genome after each generation
def save_best_genome(genome, experiment_name, generation):
    genome_path = os.path.join(experiment_name, f'best_genome_gen_{generation}.pkl')
    with open(genome_path, 'wb') as f:
        pickle.dump(genome, f)

# Save the NEAT population state
def save_population_state(population, experiment_name):
    population_path = os.path.join(experiment_name, 'neat_state.pkl')
    with open(population_path, 'wb') as f:
        pickle.dump(population, f)

# Save generation results
def save_results(experiment_name, generation, best_fitness, mean_fitness, std_fitness):
    file_path = os.path.join(experiment_name, 'results_neat.txt')
    with open(file_path, 'a') as file_aux:
        file_aux.write(f"\nGeneration {generation}: Best Fitness: {best_fitness}, Mean Fitness: {mean_fitness}, Std Fitness: {std_fitness}\n")

# Main NEAT evolution function
def run_neat_evolution(population, generations, config, env, experiment_name):
    for generation in range(generations):
        print(f"\nRunning generation {generation + 1}...", flush=True)
        
        # Run NEAT for one generation
        population.run(lambda genomes, config: eval_genomes(genomes, config, env), 1)

        # Ensure that all genomes have valid fitness before finding the best one
        valid_genomes = [g for g in population.population.values() if g.fitness is not None]
        
        if valid_genomes:  # Check if there's any valid genome
            best_genome = max(valid_genomes, key=lambda genome: genome.fitness)
            best_fitness = best_genome.fitness
            print(f"Generation {generation + 1}: Best Fitness: {best_fitness}", flush=True)

            # Log the best genome's fitness and save the results
            mean_fitness = np.mean([genome.fitness for genome in valid_genomes])
            std_fitness = np.std([genome.fitness for genome in valid_genomes])
            save_results(experiment_name, generation, best_fitness, mean_fitness, std_fitness)

            # Save the best genome
            save_best_genome(best_genome, experiment_name, generation)

            # Save the population state
            save_population_state(population, experiment_name)

    return best_genome


def main():
    print("Starting NEAT Evolution...", flush=True)

    # Parameters
    gens = 30
    experiment_name = 'neat_optimization'
    config_file = 'neat_config.txt'

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Initialize the environment
    n_hidden_neurons = 10
    env = Environment(experiment_name=experiment_name,
                      enemies=[7],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=True)

    print("Initializing NEAT population...", flush=True)

    # Initialize NEAT population
    if not os.path.exists(experiment_name + '/neat_state.pkl'):
        print('\nNEW EVOLUTION\n', flush=True)
        pop_neat, config_neat = initialize_neat_population(config_file)
        ini_g = 0
    else:
        print('\nCONTINUING EVOLUTION\n', flush=True)
        with open(experiment_name + '/neat_state.pkl', 'rb') as f:
            pop_neat = pickle.load(f)
        ini_g = pop_neat.generation

    # Run NEAT evolution
    winner_neat = run_neat_evolution(pop_neat, gens - ini_g, config_neat, env, experiment_name)

    # Output the final best genome
    print(f"Best genome after {gens} generations: {winner_neat}", flush=True)

if __name__ == '__main__':
    main()
