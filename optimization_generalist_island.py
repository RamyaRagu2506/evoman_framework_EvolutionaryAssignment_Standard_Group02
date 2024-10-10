###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller
from alg_params import *

# imports other libs
import numpy as np
import os
import time
import random
import copy


#DIVERSITY OF THE POPULATION AS IMPORTANT AS THE FITNESS OF THE INDIVIDUALS

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

# ini = time.time()  # sets time marker


# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


# def mutation(pop_to_mutate, mut, exit_local_optimum): #this is a dummy mutation function
#     mutated_pop = []

#     return mutated_pop

def mutation(pop_to_mutate, mut, exit_local_optimum):
    mut_pop=pop_to_mutate.copy()
    if np.random.random()<mut: # chance of mutation
        for e in range(len(pop_to_mutate)):
            if exit_local_optimum:
                mut_e=mut_pop[e]+np.random.normal(0,5) #5 up for interpertation
            else:
                mut_e=mut_pop[e]+np.random.normal(0,1)
            mut_pop[e]=np.clip(mut_e,dom_l,dom_u) #clip because domain issues
        
    return np.array(mut_pop)
    

    # mut_pop=pop_to_mutate.copy
    # for e in range(len(pop)):
    #     if exit_local_optimum:
    #         mut_e=mut_pop[e]+np.random.normal(0,5) #5 up for interpertation
    #     else:
    #         mut_e=mut_pop[e]+np.random.normal(0,1)
    #     mut_pop[e]=np.clip(mut_e,dom_l,dom_u) #clip because domain issues
    # return np.array(mut_pop)
    # return np.array(mutated_pop)


def parent_selection(population, pop_fitness):
    tournament = np.random.randint(0, len(population)-1, size=(tournament_size))
    print("len(population) in parent selection: ", len(population))
    print("tournament in parent selection: ", tournament)
    # if pop_fitness is None:
    #     fitness = np.array([evaluate_player(EVALUATION_IT, list(population[t])) for t in tournament])
    # else:
    np.sort(pop_fitness)
    fitness = np.array([pop_fitness[t] for t in tournament])
    # print("fitness in parent selection: ", fitness)
    # print("fitness len in parent selection: ", len(fitness))
    # parents = np.copy(population[tournament[fitness.argmin()]])
    parents = np.array([population[t] for t in tournament])
    # print("parents in parent selection: ", parents)
    # print("parents len in parent selection: ", len(parents))

    return parents[0], parents[1]

def crossover(parent1, parent2):
    offspring1, offspring2 = np.copy(parent1),np.copy(parent2)
    crossover_type= np.random.choice([0,1,2])  #uni, single, double-point
    if crossover_type== 0:  
        for i in range(len(parent1)):
            if np.random.random()<0.5:  
                offspring1[i], offspring2[i] =parent2[i], parent1[i]
    elif crossover_type ==1:  #one-point
        point= np.random.randint(1,len(parent1))  
        offspring1[:point], offspring2[:point] = parent1[:point],parent2[:point]
        offspring1[point:], offspring2[point:] = parent2[point:], parent1[point:]
    elif crossover_type== 2:  #two-point
        point1 =np.random.randint(1,len(parent1) - 1)
        point2= np.random.randint(point1,len(parent1))
        offspring1[point1:point2],offspring2[point1:point2]=parent2[point1:point2],parent1[point1:point2]
    return offspring1,offspring2

# # kills the worst genomes, and replace with new best/random solutions
# def doomsday(pop,fit_pop): #this is a dummy doomsday function

#     worst = int(npop/4)  # a quarter of the population
#     order = np.argsort(fit_pop)
#     orderasc = order[0:worst]

#     for o in orderasc:
#         for j in range(0,n_vars):
#             pro = np.random.uniform(0,1)
#             if np.random.uniform(0,1)  <= pro:
#                 pop[o][j] = np.random.uniform(dom_l, dom_u) # random dna, uniform dist.
#             else:
#                 pop[o][j] = pop[order[-1:]][0][j] # dna from best

#         fit_pop[o]=evaluate([pop[o]])

#     return pop,fit_pop


def similarity(source_island, destination_best, migration_size):
    source_island_copy = source_island.copy()
    most_similar = []
    for i in range(migration_size):
        similarity = 9999
        for index in range(len(source_island_copy)):
            individual = source_island_copy[index]
            difference = [abs(a - b) for a, b in zip(destination_best, individual)]
            sum_diff = sum(difference)
            if sum_diff <= similarity:
                similarity = sum_diff
                most_similar_ind = individual
                most_similar_ind_index = index
        most_similar.append(most_similar_ind)
        # print("source_island_copy shape before del: ", source_island_copy.shape)
        # print("most_similar_ind_index: ", most_similar_ind_index)
        source_island_copy = np.delete(source_island_copy, most_similar_ind_index, axis=0)
        # print("source_island_copy shape after del: ", source_island_copy.shape)

    return most_similar

def diversity(source_island, destination_best, migration_size):
    source_island_copy = source_island.copy()
    most_diverse = []
    for i in range(migration_size):
        diversity = 0
        for index in range(len(source_island_copy)):
            individual = source_island_copy[index]
            difference = [abs(a - b) for a, b in zip(destination_best, individual)]
            sum_diff = sum(difference)
            if sum_diff >= diversity:
                diversity = sum_diff
                most_diverse_ind = individual
                most_diverse_ind_index = index
        most_diverse.append(most_diverse_ind)
        source_island_copy = np.delete(source_island_copy, most_diverse_ind_index, axis=0)

    return most_diverse

def migrate(world_population, world_pop_fit, migration_size, migration_type):
        migrant_groups = []

        for i in range(len(world_population)):
            # TODO: create a selection function for the migrants
            # it is random now
            # migrant_groups.append({
            #     # "individuals": island.select(migration_size),
            #     "individuals": np.random.choice(world_population[island], migration_size),
            #     "destination": np.random.randint(n_islands)
            # })
            island = world_population[i]

            island_best = np.argmax(world_pop_fit[i])
            # print("island_best: ", island_best)
            # print("island[island_best]: ", island[island_best])
            world_without_destination = world_population.copy()
            world_without_destination.pop(i)
            source = random.choice(world_without_destination)
            # print("source: ", source.shape)
            if migration_type == "similarity":
                migrants = similarity(source_island=source, destination_best=island[island_best], migration_size=migration_size)
                # remove the worst individuals from the source island
                world_pop_fit_copy = world_pop_fit.copy()
                for k in range(migration_size):
                    # delete the worst individual at the destination island
                    # island = np.delete(island, island[np.random.randint(0, len(island))])
                    island_worst = np.argmin(world_pop_fit_copy[i])
                    # print("island_worst: ", island_worst)
                    # print("island before del: ", island.shape)
                    # island = np.vstack((island[:island_worst],island[island_worst+1:]))
                    island = np.delete(island, island_worst, axis=0)
                    # print("island after del: ", island.shape)
                    world_pop_fit_copy = np.delete(world_pop_fit_copy, island_worst)

                # TODO DONE: add the migrants to the destination island
                # island = np.concatenate(island, migrants[k], axis=0)
                island = np.vstack((island, migrants))
                # island.append(migrants)
                # print("island after append: ", island.shape)

            elif migration_type == "diversity":
                migrants = diversity(source_island=source, destination_best=island[island_best], migration_size=migration_size)
                world_pop_fit_copy = world_pop_fit.copy()
                for k in range(migration_size):
                    island_worst = np.argmin(world_pop_fit_copy[i])
                    island = np.delete(island, island_worst, axis=0)
                    world_pop_fit_copy = np.delete(world_pop_fit_copy, island_worst)

                island = np.vstack((island, migrants))



def individual_island_run(island_population, pop_fit, mutation_rate, exit_local_optimum):
        # self.sort()
        print(f"Island Population : {len(island_population)}........")
        parent_1, parent_2 = parent_selection(island_population, pop_fitness=pop_fit)

        child_1, child_2 = crossover(parent_1, parent_2)

        child_1_mutated = mutation(child_1, mutation_rate, exit_local_optimum)
        child_2_mutated = mutation(child_2, mutation_rate, exit_local_optimum)

        # child_1.reevaluate()
        # child_2.reevaluate()
        child_1_mutated = child_1_mutated.reshape(1, -1)
        child_2_mutated = child_2_mutated.reshape(1, -1)


        delete1 = np.argmin(pop_fit)
        print("before delete1: ", island_population.shape)
        updated_island_population = np.delete(island_population, delete1, axis=0)
        print("after delete1: ", updated_island_population.shape)
        delete2 = np.argmin(pop_fit)
        updated_island_population = np.delete(updated_island_population, delete2, axis=0)
        print("after delete2: ", updated_island_population.shape)
        updated_island_population = np.append(updated_island_population, child_1_mutated, axis=0)
        print("after append1: ", updated_island_population.shape)
        updated_island_population = np.append(updated_island_population, child_2_mutated, axis=0)
        print("after append2: ", updated_island_population.shape)

        return updated_island_population

def parallel_island_run(world_population, pop_fit, mutation_rate, exit_local_optimum):
    for i in range(n_islands):
        print(f"Debugging World population of {i} , Island {i+1}: {len(world_population[i])}........")
        new_island_population = individual_island_run(island_population=world_population[i], pop_fit=pop_fit[i], mutation_rate=mutation_rate, exit_local_optimum=exit_local_optimum)

        # print("new_island_population: ", new_island_population)
        # print("world_population[i before update]", world_population[i])
        world_population[i] = new_island_population
        # print("world_population[i after update]", world_population[i])
    
    return world_population

def main():
    ini = time.time()  # sets time marker

    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    # experiment_name = 'optimization_test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[3],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)

    env.state_to_log() # checks environment state

    # number of weights for multilayer with 10 hidden neurons
    n_weights = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    # start writing your own code from here

    if not os.path.exists(experiment_name+'/evoman_solstate'):
        print( '\n NEW EVOLUTION \n')

        # initial population using normal distribution centered around 0
        # TODO DONE create world population, below is only for one island 
        # pop = np.random.normal(mu, sigma, size=(npop, n_weights))
        world_population = [np.random.normal(mu, sigma, size=(npop, n_weights)) for i in range(n_islands)]
        print("len(world_population): ", len(world_population))
        print("world_population[i] len: ", [len(world_population[i]) for i in range(n_islands)])
        world_pop_fit = [evaluate(env, one_island_pop_fit) for one_island_pop_fit in world_population] #TODO: evaluate function
        # flattened_world_population = np.array(world_population).flatten()
        flattened_world_population = np.concatenate([world_population[i] for i in range(n_islands)], axis=0)
        # flattened_world_pop_fit = np.array(world_pop_fit).flatten()
        flattened_world_pop_fit =  np.concatenate([world_pop_fit[i] for i in range(n_islands)], axis=0)
        print("len(flattened_world_population): ", len(flattened_world_population))
        print("len(flattened_world_pop_fit): ", len(flattened_world_pop_fit))
        # best_islands = [np.argmax(one_island_pop_fit) for one_island_pop_fit in world_pop_fit]
        # best_islands = np.argmax(world_pop_fit, axis=1)
        best_overall = np.argmax(flattened_world_pop_fit)
        # mean = [np.mean(one_island_pop_fit) for one_island_pop_fit in world_pop_fit]
        mean = np.mean(flattened_world_pop_fit) 
        # std = np.std(pop_fit)
        # std = [np.std(one_island_pop_fit) for one_island_pop_fit in world_pop_fit]
        std = np.std(flattened_world_pop_fit)
        ini_g = 0
        solutions = [flattened_world_population, flattened_world_pop_fit]
        env.update_solutions(solutions)

    else:
        print( '\n CONTINUING EVOLUTION \n')

        env.load_state()
        pop = env.solutions[0]
        pop_fit = env.solutions[1]

        best = np.argmax(pop_fit)
        mean = np.mean(pop_fit)
        std = np.std(pop_fit)

        # finds last generation number
        file_aux  = open(experiment_name+'/gen.txt','r')
        ini_g = int(file_aux.readline())
        file_aux.close()

    # saves results for first pop
    file_aux  = open(experiment_name+'/results.txt','a')
    file_aux.write('\n\ngen best mean std')
    print( '\n GENERATION '+str(ini_g)+' '+str(round(flattened_world_pop_fit[best_overall],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(ini_g)+' '+str(round(flattened_world_pop_fit[best_overall],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    # # used for doomsday
    # last_sol = flattened_world_pop_fit[best_overall]
    # notimproved = 0

    #TODO: for loop here for the generations
    for i in range(ini_g, n_gens):
        print(f"Generation {i}........")
        # print("world_population: ", len(world_population[i])) # will have error after n_islands

        updated_world_population = parallel_island_run(world_population, world_pop_fit, mutation_rate, exit_local_optimum)
        print("updated_world_population len: ", len(updated_world_population))
        print("updated_world_population each island len: ", [len(i) for i in updated_world_population])
        updated_world_pop_fit = [evaluate(env, updated_island_pop_fit) for updated_island_pop_fit in updated_world_population]

        if i % migration_interval == 0:
            migrate(updated_world_population, updated_world_pop_fit, migration_size, migration_type)


        flattened_updated_world_population = np.concatenate([updated_world_population[i] for i in range(n_islands)], axis=0)
        print("len(flattened_updated_world_population): ", len(flattened_updated_world_population))
        print("flattened_updated_world_population[0]: ", flattened_updated_world_population[0])
        flattened_updated_world_pop_fit = evaluate(env, flattened_updated_world_population)
        best = np.argmax(flattened_updated_world_pop_fit)
        std  =  np.std(flattened_updated_world_pop_fit)
        mean = np.mean(flattened_updated_world_pop_fit)


        # saves results
        file_aux  = open(experiment_name+'/results.txt','a')
        print( '\n GENERATION '+str(i)+' '+str(round(flattened_updated_world_pop_fit[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        file_aux.write('\n'+str(i)+' '+str(round(flattened_updated_world_pop_fit[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
        file_aux.close()

        # saves generation number
        file_aux  = open(experiment_name+'/gen.txt','w')
        file_aux.write(str(i))
        file_aux.close()

        # saves file with the best solution
        print( '\n BEST SOLUTION:'+str(flattened_updated_world_pop_fit[best])+' '+str(flattened_updated_world_population[best])+'\n')
        np.savetxt(experiment_name+'/best.txt',flattened_updated_world_population[best])

        # saves simulation state
        solutions = [flattened_updated_world_population, flattened_updated_world_pop_fit]
        env.update_solutions(solutions)
        env.save_state()


    fim = time.time() # prints total execution time for experiment
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
    print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')


    file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
    file.close()


    env.state_to_log() # checks environment state




if __name__ == '__main__':
    main()