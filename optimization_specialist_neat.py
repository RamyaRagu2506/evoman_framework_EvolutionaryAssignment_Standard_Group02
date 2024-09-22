import sys
sys.stdout.flush = lambda: None  # Disable output buffering

# EvoMan Framework - V1.0 2016
# DEMO: Neuroevolution - Genetic Algorithm neural network.

# imports framework
import sys
from evoman.environment import Environment
from neat_controller import player_controller

# imports other libs
import numpy as np
import os
import neat
import pickle
import matplotlib.pyplot as plt


# Runs simulation
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
            print(f"Genome {genome_id} Fitness: {genome.fitness:.6f}", flush=True)
        except Exception as e:
            print(f"Error during simulation for genome {genome_id}: {e}", flush=True)
            genome.fitness = 0  # Assign default fitness if there's an error

# Initialize NEAT population
def initialize_neat_population(config_file):
    # https://neat-python.readthedocs.io/en/latest/xor_example.html
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    
    # Initialize population
    neat_population = neat.Population(config)
    
    # Add reporters
    neat_population.add_reporter(neat.StdOutReporter(True))
    stats_neat = neat.StatisticsReporter()
    neat_population.add_reporter(stats_neat)

    print("NEAT Population initialized successfully.", flush=True)
    return neat_population, config

# Save the best genome after each generation
def save_best_genome(genome, experiment_name, generation):
    genome_path = os.path.join(experiment_name, f'best_genome_gen_{generation}.pkl')
    with open(genome_path, 'wb') as f:
        pickle.dump(genome, f)
    print(f"Best genome saved for generation {generation + 1}.", flush=True)

# Save the NEAT population state
def save_population_state(population, experiment_name, generation):
    population_path = os.path.join(experiment_name, 'neat_state.pkl')
    with open(population_path, 'wb') as f:
        pickle.dump([population, generation], f)
    print("Population state saved successfully.", flush=True)

# Load the NEAT population state
def load_population_state(experiment_name):
    population_path = os.path.join(experiment_name, 'neat_state.pkl')
    with open(population_path, 'rb') as f:
        population, generation = pickle.load(f)
    print("Population state loaded successfully.", flush=True)
    return population, generation

# Save generation results
def save_results(experiment_name, generation, best_fitness, mean_fitness, std_fitness):
    file_path = os.path.join(experiment_name, 'results_neat.txt')
    with open(file_path, 'a') as file_aux:
        file_aux.write(f"\nGeneration {generation + 1}: Best Fitness: {best_fitness:.6f}, "
                       f"Mean Fitness: {mean_fitness:.6f}, Std Fitness: {std_fitness:.6f}\n")
    print(f"Results saved for generation {generation + 1}.", flush=True)

# Plot fitness over generations and save as image
def plot_fitness(generations, best_fitness_list, mean_fitness_list, experiment_name):
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness_list, label='Best Fitness', color='b', marker='o')
    plt.plot(generations, mean_fitness_list, label='Mean Fitness', color='g', marker='x')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations - NEAT Algorithm')
    plt.legend()
    plt.grid(True)

    # Save the plot to the experiment directory
    plot_path = os.path.join(experiment_name, 'fitness_over_generations.png')
    plt.savefig(plot_path)
    plt.show()

# Main NEAT evolution function
def run_neat_evolution(population, generations, config, env, experiment_name):
    # Lists to store fitness values for plotting
    best_fitness_list = []
    mean_fitness_list = []
    
    for generation in range(generations):
        print(f"\n========== Running Generation {generation + 1} ==========", flush=True)
        
        # Run NEAT for one generation
        population.run(lambda genomes, config: eval_genomes(genomes, config, env), 1)

        # Ensure that all genomes have valid fitness before finding the best one
        valid_genomes = [g for g in population.population.values() if g.fitness is not None]
        
        if valid_genomes:  # Check if there's any valid genome
            best_genome = max(valid_genomes, key=lambda genome: genome.fitness)
            best_fitness = best_genome.fitness
            mean_fitness = np.mean([genome.fitness for genome in valid_genomes])
            std_fitness = np.std([genome.fitness for genome in valid_genomes])

            # Log and print fitness values
            print(f"Generation {generation + 1} Summary: ", flush=True)
            print(f"  - Best Fitness: {best_fitness:.6f}", flush=True)
            print(f"  - Mean Fitness: {mean_fitness:.6f}", flush=True)
            print(f"  - Std Deviation: {std_fitness:.6f}", flush=True)

            # Save fitness values for plotting
            best_fitness_list.append(best_fitness)
            mean_fitness_list.append(mean_fitness)

            # Save the generation results
            save_results(experiment_name, generation, best_fitness, mean_fitness, std_fitness)

            # Save the best genome
            save_best_genome(best_genome, experiment_name, generation)

            # Save the population state
            save_population_state(population, experiment_name, generation)
    
    # Return the best genome and fitness data for plotting
    generations_list = list(range(1, generations + 1))
    return best_genome, best_fitness_list, mean_fitness_list, generations_list

# Main function to run the NEAT algorithm
def main():
    print("Starting NEAT Evolution...", flush=True)

    # Parameters
    gens = 30
    experiment_name = 'neat_optimization_v2_enemy8'
    config_file = 'neat_config.txt'

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Initialize the environment
    n_hidden_neurons = 10
    env = Environment(experiment_name=experiment_name,
                      enemies=[8],  # Adjust the enemy as needed
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    print("Environment setup complete.", flush=True)

    # Check if previous evolution exists, else start new evolution
    if not os.path.exists(experiment_name + '/neat_state.pkl'):
        print('\nNEW EVOLUTION\n', flush=True)
        pop_neat, config_neat = initialize_neat_population(config_file)
        ini_g = 0
    else:
        print('\nCONTINUING EVOLUTION\n', flush=True)
        pop_neat, ini_g = load_population_state(experiment_name)

    # Run NEAT evolution
    best_genome, best_fitness_list, mean_fitness_list, generations = run_neat_evolution(
        pop_neat, gens - ini_g, config_neat, env, experiment_name)

    # Output the final best genome
    print(f"\nBest genome after {gens} generations: {best_genome}", flush=True)

    # Plot the fitness over generations
    plot_fitness(generations, best_fitness_list, mean_fitness_list, experiment_name)

if __name__ == '__main__':
    main()
