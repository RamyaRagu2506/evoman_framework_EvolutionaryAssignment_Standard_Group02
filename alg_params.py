import numpy as np
# alg parameters
dom_u = 1 #not used for now 
dom_l = -1 #not used for now
mu = 0 #mean for initial population
sigma = 0.21 #std for initial population
npop = 70 #size of the population
n_gens = 100 #number of generations
mutation_rate = 0.8 #mutation rate
last_best = 0 #not used for now
tournament_size = 2 #tournament size for parent selection
exit_local_optimum = False #boolean

max_time = 1000 #max time for each simulation

#  Other parameters
# seed = np.random.randint(0, 1000)
seed = 1010
experiment_name = 'diversity_run_enemy3_'+str(seed)
run_mode = 'train' # train or test
experiment_test_name = 'crossover_baseline_enemy1_111'

n_hidden_neurons = 128

# algorithm parameters
# mutation_type = 'uniform' # or 'gaussian'
# TODO: make crossover a choice 
# crossover_type = 0 # 0 for uniform crossover, 1 for one point crossover, 2 for two point crossover

# for multiple islands
n_islands = 6 # number of islands
# n_migrations = 10 # number of migrations between islands
migration_size = 2 # number of individuals to migrate between islands 
migration_interval = 20 # number of generations between migrations
migration_type = "diversity" #"similarity" # or "diversity"

alg_args = {'dom_u': dom_u, 'dom_l': dom_l, 'npop': npop, 'n_gens': n_gens, 'mutation_rate': mutation_rate, 'last_best': last_best, 'experiment_name': experiment_name, 'tournament_size': tournament_size, 'seed': seed, 'run_mode': run_mode, 'n_hidden_neurons': n_hidden_neurons, 'n_islands': n_islands, 'migration_size': migration_size, 'migration_interval': migration_interval, 'migration_type': migration_type, 'max_time': max_time, 'exit_local_optimum': exit_local_optimum}