#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                              			  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        specialist solutions for each enemy (game)                                   #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
import sys, os


from evoman.environment import Environment
from demo_controller import player_controller
from joblib import Parallel, delayed

# imports other libs
import numpy as np
import time
# Update the number of neurons for this specific example
n_hidden_neurons = 10

experiment_name = 'blend_recombination_test1'
controller = player_controller(n_hidden_neurons)
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)




# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  enemies=[8],   #DECIDE ON ENEMY
				  player_controller=controller,
			  	  speed="fastest",
				  enemymode="static",
				  level=2,
				  visuals=False)


env.state_to_log() # checks environment state
ini = time.time()  

run_mode = 'train' # train or test

n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5


no_parents = 10
pop_size = 100
gens = 30
tau = 1/np.sqrt(pop_size)    #according to book p.59

def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    # f - fitness, p - player life, e - enemy life, t - time
    return f

def run_game_in_worker(experiment_name, controller, ind):
    # Recreate or reinitialize the environment from env_config inside the worker
    env = Environment(experiment_name=experiment_name,
                      playermode="ai",
                      enemies=[8],  # DECIDE ON ENEMY
                      player_controller=controller,
                      speed="fastest",
                      enemymode="static",
                      level=2,
                      visuals=False)
    return simulation(env, ind)

def evaluate_fitnesses(env, population):
    ''' Evaluates fitness of each individual in the population of solutions
    parallelized for efficiency'''
    # Instead of passing the full environment, pass only the configuration or parameters needed to reinitialize it
    name  = env.experiment_name
    contr = env.player_controller
    fitnesses = Parallel(n_jobs=-1)(
        delayed(run_game_in_worker)(name, contr, ind) for ind in population
    )
    return fitnesses
def evaluate(x): # evaluates the weights for the network (x)
    # x is population
    return np.array(list(map(lambda y: simulation(env,y), x)))
#returns an array of weights per individual

#pop.shape[0] number of individuals/rows in the pop


def select_parents_uniform_random(pop, no_parents): #global
    #selects random parents
    parent_idx = np.random.randint(0, pop.shape[0], no_parents)
    parents = pop[parent_idx]
    return parent_idx, parents

def select_parents_tournament(pop, fit_pop, tournament_size=10): #global
    #selects parents based on tournament selection, returns the best parent

    # Convert fit_pop to a NumPy array
    fit_pop = np.array(fit_pop)
    # select random individuals from pop
    tournament_indices = np.random.randint(0, pop.shape[0], tournament_size).flatten() # list of indices in tournament
    tournament = fit_pop[tournament_indices]
    # select the best individual from the tournament

    best_parent_idx = np.argmax(tournament) # index of the best parent relative to the tournament
    best_parent = pop[tournament_indices[best_parent_idx]] # the best parent

    return tournament_indices[best_parent_idx], best_parent # returns index of best parent and the parent itself



def recombination(no_parents, step_sizes, pop):  #global
    #due to comma stratefy, the no of offspring is at least same size as pop.size
    n_offspring = np.random.randint(pop_size+1, pop_size * 2)
    
	#init arrays
    offspring =  np.zeros( (n_offspring, n_vars) )
    offspring_step_size = np.zeros( (n_offspring, n_vars)) # sigma
    
    for i in range(n_offspring):
        # Get random parents
        parent_idx, parents = np.random.randint(0, pop.shape[0], no_parents)

        # Get the mean per gene for offspring
        offspring[i] = np.mean(parents, axis=0)
        # Get the mean per step size for offspring
        offspring_step_size[i] = np.mean(step_sizes[parent_idx], axis=0)
    
    return offspring, offspring_step_size

def blend_recombination(step_sizes, pop, fit_pop, alpha=0.5):  #global, alpha is a scaling parameter
    n_offspring = np.random.randint(pop_size + 1, pop_size * 2)

    offspring =  np.zeros( (n_offspring, n_vars) )
    offspring_step_size = np.zeros( (n_offspring, n_vars)) # sigma

    for i in range(n_offspring):
        parent_idx1, parent1 = select_parents_tournament(pop, fit_pop)
        parent_idx2, parent2 = select_parents_tournament(pop, fit_pop)


        differece = np.abs(parent1 - parent2)
        # find the minimum and maximum values for each gene
        min_values = np.minimum(parent1, parent2) - differece * alpha
        max_values = np.maximum(parent1, parent2) + differece * alpha
        # generate offspring
        offspring[i] = np.random.uniform(min_values, max_values)

        # generate offspring step size (sigma) - no reason for this design choice (for now)
        offspring_step_size[i] = np.mean(np.stack((step_sizes[parent_idx1], step_sizes[parent_idx2])), axis=0)

    return offspring, offspring_step_size



def mutate(individual, step_size, tau): # FORMULA FROM BOOK 4.2,4.3
    #updates step size, according to book formula page 58
    new_step_size = step_size * np.exp(tau * np.random.randn(*step_size.shape))
    
    #Gaussian perturbation
    new_individual = individual + new_step_size * np.random.randn(*individual.shape)
    return new_individual, new_step_size


def survivor_selection(pop, fit_pop, step_sizes, pop_size):
    #selects indices of the best individuals
    elite_idx = np.argsort(fit_pop)[-pop_size:] # -pop_size: selects the best individuals (at the end of the list) of size pop_size

    
	#selects the best individuals
    pop = pop[elite_idx]
    step_sizes = step_sizes[elite_idx]
    
    return pop, step_sizes


ini_g = 0
best = None
mean = None
std = None

# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)


# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')

    pop = np.random.uniform(low=-1, high=1, size=(pop_size, n_vars))
    step_sizes = np.random.uniform(low=0.1, high=0.2, size=(pop_size, n_vars))
    fit_pop = evaluate(pop)

else:

    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]
    step_sizes = env.solutions[2]

    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    # finds last generation number
    file_aux  = open(experiment_name+'/gen.txt','r')
    ini_g = int(file_aux.readline())
    file_aux.close()
    
# Save results for the first population
file_aux = open(experiment_name + '/results.txt', 'a')

if best is not None:
	file_aux.write('\n\ngen best mean std')
	print('\n GENERATION ' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
	file_aux.write('\n' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
file_aux.close()

# fit_pop = evaluate(pop)
fit_pop = evaluate_fitnesses(env, pop)
best = np.argmax(fit_pop)
mean = np.mean(fit_pop)
std = np.std(fit_pop)
ini_g = 0
solutions = [pop, fit_pop, step_sizes]
env.update_solutions(solutions)  


best_current_solution_index = np.argmax(fit_pop)
best_current_solution = pop[best_current_solution_index]
	
for i in range(ini_g+1, gens):
    # offspring, offspring_step_size = recombination(no_parents, step_sizes, pop) # for uniform random recombination
    offspring, offspring_step_size = blend_recombination(step_sizes, pop, fit_pop, alpha=0.5) # for blend recombination

    offspring, offspring_step_size = mutate(offspring, offspring_step_size, tau)
    # f_offspring = evaluate(offspring)
    f_offspring = evaluate_fitnesses(env, offspring)
	#comma strategy
    pop, step_sizes = survivor_selection(offspring, f_offspring, offspring_step_size, pop_size)
    # fit_pop = evaluate(pop)
    fit_pop = evaluate_fitnesses(env, pop)


	#from other file 
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    # Save the results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    # Save the generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()

    # Save the best solution
    np.savetxt(experiment_name+'/best.txt',pop[best])

    # Save the simulation state
    solutions = [pop, fit_pop, step_sizes]  # Make sure to include step sizes if needed
    env.update_solutions(solutions)
    env.save_state()
    


# Test saved demo solutions for each enemy
for en in range(1, 9):
	# Update the enemy
	env.update_parameter('enemies', [en])

	# Load specialist controller
	sol = np.loadtxt('solutions_demo/demo_' + str(en) + '.txt')
	print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY ' + str(en) + ' \n')
	env.play(sol)

fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')

file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()


env.state_to_log() # checks environment state