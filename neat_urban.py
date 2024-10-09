import neat
import os
import matplotlib.pyplot as plt
from evoman.environment import Environment
from joblib import Parallel, delayed
from neat_controller import NEATController

NUM_CORES = os.cpu_count() # Get cores of your system for parallel evaluation

DEFAULT_EXPERIMENT_NAME = 'neat_evoman'
DEFAULT_ENEMY = [1,2,3,4,5,6,7,8]  # You can adjust the enemies here
DEFAULT_GENERATIONS = 100



# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"



# Environment setup
def setup_environment(experiment_name, controller, enemies=DEFAULT_ENEMY) -> Environment:
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

# NEAT onfiguration loader
def load_neat_config(config_file):
    config_path = config_file
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    return config



# Simulation function
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

def eval_fitnesses(genome, config):
    # Create a neural network for the given genome
    network = neat.nn.FeedForwardNetwork.create(genome, config)

    # Wrap the network in the NEATController (assuming it's already defined)
    controller = NEATController(network)

    # Set up the Evoman environment with the controller
    env = setup_environment(DEFAULT_EXPERIMENT_NAME, controller, enemies=DEFAULT_ENEMY)

    # Run the game and get the fitness score
    fitness = simulation(env, controller)

    return fitness  # Return the fitness of the genome


def test_solution_against_all_enemies(winner, config):

    all_enemies = [1, 2, 3, 4, 5, 6, 7, 8]

    winner_network = neat.nn.FeedForwardNetwork.create(winner, config) # Create the neural network from the winning genome

    controller = NEATController(winner_network)
    env = setup_environment(DEFAULT_EXPERIMENT_NAME, controller, enemies=all_enemies)

    # Run the game and get the fitness score
    fitness = simulation(env, controller)

    return fitness

# Function to plot fitness statistics (max, avg, std) over generations
def plot_fitness_statistics(stats, title=f"enemies: {DEFAULT_ENEMY}"):
    # Get the generation numbers
    generations = list(range(len(stats.most_fit_genomes)))

    # Extract max, avg, and std fitness values over generations
    max_fitness = [genome.fitness for genome in stats.most_fit_genomes]
    avg_fitness = stats.get_fitness_mean()  # Already a list of floats
    std_fitness = stats.get_fitness_stdev()  # Already a list of floats

    plt.figure(figsize=(10, 6))

    plt.plot(generations, max_fitness, label='Max Fitness', color='blue')
    plt.plot(generations, avg_fitness, label='Avg Fitness', color='green')
    # plt.plot(generations, std_fitness, label='Std Dev Fitness', color='red')

    plt.fill_between(generations, [avg - std for avg, std in zip(avg_fitness, std_fitness)],
                     [avg + std for avg, std in zip(avg_fitness, std_fitness)], color='red', alpha=0.2, label='Std Dev Fitness')

    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.suptitle(title)
    plt.title('Max, Avg, and Std Dev Fitness Over Generations')
    plt.legend()

    plt.grid(True)
    plt.show()
def run_neat():
    # Load config
    config = load_neat_config("neat_config.txt")

    # Initialise the population
    population = neat.Population(config)

    # Add reporters to get progress information
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Set up ParallelEvaluator
    evaluator = neat.ParallelEvaluator(num_workers=NUM_CORES, eval_function=eval_fitnesses)

    # Run the NEAT algorithm using the parallel evaluator
    winner = population.run(evaluator.evaluate, n=DEFAULT_GENERATIONS)


    # After evolution, print the best genome
    print(f"Best genome: {winner}")

    # Test the best genome against all enemies
    total_fitness = test_solution_against_all_enemies(winner, config)

    plot_fitness_statistics(stats)

    return winner, total_fitness


if __name__ == "__main__":
    print(f"Number of cores on your device: {NUM_CORES}")
    winner, total_fitness = run_neat()
    print("total fitness: ", total_fitness)

