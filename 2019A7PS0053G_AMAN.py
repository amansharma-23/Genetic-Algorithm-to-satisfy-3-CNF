from CNF_Creator import *
import numpy
import matplotlib.pyplot as plot
import time

population_size = 20
number_of_variables = 50
number_of_parents_mating = 2
crossover_point = 25

def calculate_fitness(population,sentence):
    fitness = []
    for solution in population:          #iterating over every member of the population
        fit_value = calculate_fit_value(solution,sentence)      
        fitness.append(fit_value)
    return fitness

def calculate_fit_value(solution,sentence):
    satisfied_clauses = 0
    for clause in sentence:          #iterating over every clause in sentence
        satisfied = False             
        for literal in clause:
            if(literal < 0):
                literal = literal * -1
                if(solution[literal -1] < 0):
                    satisfied = True
                    break
            else:
                if(solution[literal - 1] > 0):
                    satisfied = True
                    break
        if(satisfied):
            satisfied_clauses = satisfied_clauses + 1         #conunting the number of clauses satisfied
    return satisfied_clauses

def crossover(parent1, parent2):
    #generating a random crossover point
    crossover_point = numpy.random.randint(low=5,high=number_of_variables-5,size=1)

    #creating children by crossing-over parents
    child1 = numpy.append(parent1[:crossover_point[0]],parent2[crossover_point[0]:50])
    child2 = numpy.append(parent2[:crossover_point[0]],parent1[crossover_point[0]:50])
    
    # for crossover_point in range(5,45):
    #     child3 = numpy.append(parent1[:crossover_point],parent2[crossover_point:50])
    #     if(calculate_fit_value(child1,sentence)<calculate_fit_value(child3,sentence)):
    #             child1=child3
    #     child4 = numpy.append(parent2[:crossover_point],parent1[crossover_point:50])
    #     if(calculate_fit_value(child2,sentence)<calculate_fit_value(child4,sentence)):
    #             child2=child4

    return [child1,child2]

def calculate_probability(fitness):
    #calculating total fitness
    total_fitness = 0
    for i in fitness:
        total_fitness+=i      

    #calcualting probability of being picked according to population      
    probability = []
    for i in fitness:
        probability.append(i/total_fitness)
    return probability

def mutate(solution):
    #generating a random point of mutation
    mutation_at = numpy.random.randint(low=0,high=number_of_variables,size=1)

    #mutation
    solution[mutation_at] = solution[mutation_at]*-1
    return solution

#textbook algorithm
def genetic_algorithm(population,sentence):
    next_gen = numpy.empty([population_size,number_of_variables])    #array to store next generation
    fitness = calculate_fitness(population,sentence)    #calculating fitness
    probability = calculate_probability(fitness)    #caculating probability of being selected on the basis of fitness
    indices = numpy.arange(population.shape[0])
    for i in range(0,20,2):
        parents = population[numpy.random.choice(indices,2,p=probability,replace=False)]     #choosing parents
        children = crossover(parents[0],parents[1])     #crossing-over

        #causing mutation with low probability
        temp = numpy.random.randint(low=0,high=10,size=1)
        if(temp<1): 
            children[0] = mutate(children[0])
            children[1] = mutate(children[1])

        #adding children to next generation
        next_gen[i]=children[0]
        next_gen[i+1]=children[1]
    return next_gen

#improved algorithim
def improved_genetic_algorithm(population,sentence):
    next_gen = numpy.empty([population_size,number_of_variables])   #array to store next generation
    indices = numpy.arange(population.shape[0])     
    fitness = calculate_fitness(population,sentence)    #calculating fitness
    probability = calculate_probability(fitness)    #caculating probability of being selected on the basis of fitness
    for i in range(0,10):
        parents = population[numpy.random.choice(indices,2,p=probability,replace=False)]  #choosing parents
        children = crossover(parents[0],parents[1])
        temp = numpy.random.randint(low=0,high=10,size=1)

        #double mutation with low probability
        if(temp<1):
            children[0] = mutate(children[0])
            children[1] = mutate(children[1])
            children[0] = mutate(children[0])
            children[1] = mutate(children[1])
        
        #single mutation with higher probability
        elif(temp<4):
            children[0] = mutate(children[0])
            children[1] = mutate(children[1])

        #choosing the child with greater fitness value
        if(calculate_fit_value(children[1],sentence)>calculate_fit_value(children[0],sentence)):
            next_gen[i]=children[1]
        else:
            next_gen[i]=children[0]

    #choosing parents for next generation
    for i in range(10,20):
        good_parent = population[numpy.random.choice(indices,1,p=probability,replace=False)]
        temp = numpy.random.randint(low=0,high=10,size=1)
        
        #mutating parents
        if(temp<3):
            good_parent[0] = mutate(good_parent[0])
            # if(temp<2):
            #     good_parent[0] = mutate(good_parent[0])
        next_gen[i] = good_parent[0]
    return next_gen

def main():
    cnfC = CNF_Creator(n=50) # n is number of symbols in the 3-CNF sentence
    # sentence = cnfC.CreateRandomSentence(m=120) # m is number of clauses in the 3-CNF sentence
    # print('Random sentence : ',sentence)

    sentence = cnfC.ReadCNFfromCSVfile()
    # print('\nSentence from CSV file : ',sentence)


#   CODE BEGINS

    #creating initial population
    origin = numpy.arange(1,51)
    population = numpy.empty([population_size,number_of_variables])
    for i in range(population_size):
        population[i] = origin

    fitness = calculate_fitness(population,sentence)

    initial_time = time.time()

    best_model = numpy.empty(number_of_variables)
    fitness_of_best_model = -1
    fitness_of_last_best_model = fitness_of_best_model
    number_of_generations_since_last_best_model = 0

    for i in range(0,10000):        #iterating for a large number of generations
        population = improved_genetic_algorithm(population,sentence)
        fitness = calculate_fitness(population,sentence)
        for j in range(population_size):
            if(fitness[j]>fitness_of_best_model):
                best_model=population[j]
                fitness_of_best_model=fitness[j]
        if(fitness_of_best_model==fitness_of_last_best_model):
            number_of_generations_since_last_best_model+=1
        else:
            number_of_generations_since_last_best_model=0
            fitness_of_last_best_model=fitness_of_best_model
        if(number_of_generations_since_last_best_model>1000):      #terminating if the best values does not change over a large number of generations
            break
        if(fitness_of_best_model==len(sentence)):
            break
        # print("generation: ",i,"fitness of best model so far: ",100 * fitness_of_best_model/len(sentence))
        if(time.time()-initial_time>45):        #terminating if the runtime exceeeds 45 seconds
            break

    final_time = time.time()


    print('\n\n')
    print('Roll No : 2019A7PS0053G')
    print('Number of clauses in CSV file : ',len(sentence))
    print('Best model : ',best_model)
    print('Fitness value of best model : ',100*fitness_of_best_model/len(sentence))
    print('Time taken : ',final_time-initial_time)
    print('\n\n')
    

    # CODE FOR PLOTTING GRAPHS
    # average_time_values = []
    # average_fitness_values = []
    # x_axis_values = []
    # for m in range(100,320,20):
    #     total_fitness_value = 0
    #     total_time_taken = 0
    #     for i in range(10):
    #         print("m: ",m,"i: ",i)
    #         sentence = cnfC.CreateRandomSentence(m)
    #         origin = numpy.arange(1,51)
    #         population = numpy.empty([population_size,number_of_variables])
    #         for i in range(population_size):
    #             population[i] = origin

    #         fitness = calculate_fitness(population,sentence)

    #         initial_time = time.time()

    #         best_model = numpy.empty(number_of_variables)
    #         fitness_of_best_model = -1
    #         fitness_of_last_best_model = fitness_of_best_model
    #         number_of_generations_since_last_best_model = 0

    #         for i in range(0,10000):
    #             population = improved_genetic_algorithm(population,sentence)
    #             fitness = calculate_fitness(population,sentence)
    #             for j in range(population_size):
    #                 if(fitness[j]>fitness_of_best_model):
    #                     best_model=population[j]
    #                     fitness_of_best_model=fitness[j]
    #             if(fitness_of_best_model==fitness_of_last_best_model):
    #                 number_of_generations_since_last_best_model+=1
    #             else:
    #                 number_of_generations_since_last_best_model=0
    #                 fitness_of_last_best_model=fitness_of_best_model
    #             if(number_of_generations_since_last_best_model>500):
    #                 break
    #             if(fitness_of_best_model==len(sentence)):
    #                 break
    #             # print("generation: ",i,"fitness of best model so far: ",100 * fitness_of_best_model/len(sentence))
    #             if(time.time()-initial_time>45):
    #                 break

    #         final_time = time.time()
    #         total_fitness_value += 100*fitness_of_best_model/len(sentence)
    #         total_time_taken +=final_time-initial_time
    #     average_fitness_value = total_fitness_value/10
    #     average_time_taken = total_time_taken/10
    #     average_time_values.append(average_time_taken)
    #     average_fitness_values.append(average_fitness_value)
    #     x_axis_values.append(m)
    # plot.plot(x_axis_values,average_fitness_values,label="Average Fitness Value")
    # plot.ylabel('Average Fitness Value of Best Model Found')
    # plot.xlabel('Number of Clauses')
    # plot.legend()
    # plot.title("Average fitness value of best model found for different sentence lengths")
    # plot.show()
    # plot.plot(x_axis_values,average_time_values,label="Average Time Taken")
    # plot.ylabel('Average Time Taken')
    # plot.xlabel('Number of Clauses')
    # plot.legend()
    # plot.title("Average time taken for different sentence lengths")
    # plot.show()
    # plot.show()

if __name__=='__main__':
    main()