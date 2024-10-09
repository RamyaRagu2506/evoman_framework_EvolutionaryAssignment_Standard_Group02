import neat
import os
from evoman.environment import Environment
from joblib import Parallel, delayed
from neat_controller import NEATController

NUM_CORES = os.cpu_count() # Get cores of your system for parallel evaluation
print(f"Number of cores on your device: {NUM_CORES}")

DEFAULT_EXPERIMENT_NAME = 'neat_evoman'
DEFAULT_ENEMY = [1,5]  # You can adjust the enemies here
DEFAULT_GENERATIONS = 100



# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"



# Environment setup
def setup_environment(experiment_name, enemies, controller):
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    return Environment(
        experiment_name=experiment_name,
        playermode="ai",
        enemies=enemies,
        player_controller=controller,
        speed="fastest",
        enemymode="static",
        level=2,
        visuals=False,
        multiplemode="yes",
    )


# Simulation function
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f



# NEAT configuration loader
def load_neat_config(config_file):
    config_path = config_file
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    return config


# Fitness function to be used by NEAT
def fitness_function(genomes, config):
    # Create an empty environment to pass around
    env = setup_environment(DEFAULT_EXPERIMENT_NAME, DEFAULT_ENEMY, None)

    # Parallelize fitness evaluation
    fitnesses = evaluate_fitnesses(env, genomes, config)

    # Assign fitness back to each genome
    for (genome_id, genome), fitness in zip(genomes, fitnesses):
        genome.fitness = fitness


# Fitness evaluation using parallel workers
def evaluate_fitnesses(env, genomes, config):
    fitnesses = Parallel(n_jobs=-1)(
        delayed(run_game_in_worker)(env.experiment_name, neat.nn.FeedForwardNetwork.create, genome, config) for
        genome_id, genome in genomes
    )
    return fitnesses


def eval_genome(genome, config):
    # Create a neural network for the given genome
    network = neat.nn.FeedForwardNetwork.create(genome, config)

    # Wrap the network in the NEATController (assuming it's already defined)
    controller = NEATController(network)

    # Set up the Evoman environment with the controller
    env = setup_environment(DEFAULT_EXPERIMENT_NAME, DEFAULT_ENEMY, controller)

    # Run the game and get the fitness score
    fitness = simulation(env, controller)

    return fitness  # Return the fitness of the genome
def run_game_in_worker(experiment_name, controller_func, genome, config):
    # Create a new instance of the neural network for each genome
    network = controller_func(genome, config)  # Create the actual FeedForwardNetwork instance

    # Wrap the network in the custom NEATController class
    controller = NEATController(network)

    # Set up the environment with the new NEATController
    env = setup_environment(experiment_name, DEFAULT_ENEMY, controller)

    # Run the simulation and return the fitness
    return simulation(env, controller)


def run_neat():
    # Load config from neat_config.txt
    config = load_neat_config("neat_config.txt")

    # Initialise the population
    population = neat.Population(config)

    # Add reporters to get progress information
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Set up ParallelEvaluator (change num_workers to match your system)
    evaluator = neat.ParallelEvaluator(num_workers=NUM_CORES, eval_function=eval_genome)

    # Run the NEAT algorithm using the parallel evaluator
    winner = population.run(evaluator.evaluate, n=DEFAULT_GENERATIONS)


    # After evolution, print the best genome
    print(f"Best genome: {winner}")
    return winner


if __name__ == "__main__":
    winner = run_neat()

    print("Winner genome:", winner)
