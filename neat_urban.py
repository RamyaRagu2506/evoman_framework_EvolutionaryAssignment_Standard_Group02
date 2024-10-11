import neat
import os
import matplotlib.pyplot as plt
from evoman.environment import Environment
from joblib import Parallel, delayed
from neat_controller import NEATController

NUM_CORES = os.cpu_count() # Get cores of your system for parallel evaluation

DEFAULT_EXPERIMENT_NAME = 'neat_evoman'
DEFAULT_ENEMY = [7, 8, 5]  # You can adjust the enemies here
DEFAULT_GENERATIONS = 50



# choose this for not using visuals and thus making experiments faster
headless = False
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"



# Environment setup
def setup_environment(experiment_name, controller, enemies=DEFAULT_ENEMY, MULTIPLEMODE="yes") -> Environment:
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
        multiplemode=MULTIPLEMODE,
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
    env.update_parameter('speed', 'fastest')
    env.update_parameter("visuals", True)

    # Run the game and get the fitness score
    total_gains = []
    total_fitnesses = []
    for i in range(10):
        total_fitness, player_life, enemy_life, _ = env.play(pcont=controller)
        total_gain = player_life - enemy_life
        total_fitnesses.append(total_fitness)
        total_gains.append(total_gain)

    # box plot of total gains
    plt.boxplot(total_gains)
    plt.title("Total Gains Against All Enemies")
    plt.show()

    # -800 to 800
    # not sure about fitness
    # grade depends on how many enemies you beat

    # calculate gain for each enemy
    gain_against_enemies = {}
    for enemy in all_enemies:
        env = setup_environment(DEFAULT_EXPERIMENT_NAME, controller, enemies=[enemy],MULTIPLEMODE="no")
        env.update_parameter('speed', 'fastest')
        env.update_parameter("visuals", False)


        f, p, e, t = env.play(pcont=controller)
        print(f"Enemy {enemy}: {f}")
        gain = p - e
        # -100 to 100
        gain_against_enemies[enemy] = gain


    # and calculate the average gain for the boxplot
    # count wins if gain > 0
    # gain will affect the grade, not fitness


    return total_fitness, total_gain, gain_against_enemies

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
    total_fitness, total_gain, gain_against_enemies = test_solution_against_all_enemies(winner, config)

    plot_fitness_statistics(stats)

    return winner, total_fitness, total_gain, gain_against_enemies


if __name__ == "__main__":
    print(f"Number of cores on your device: {NUM_CORES}")
    winner, total_fitness, total_gain, gain_against_enemies = run_neat()
    print("total fitness: ", total_fitness)
    print("total gain: ", total_gain)
    print("gain against enemies: ", gain_against_enemies)


