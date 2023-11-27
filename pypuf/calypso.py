from pypuf.simulation import XORArbiterPUF, InterposePUF
from pypuf.io import random_inputs
import random
import numpy as np
import math
import copy
import argparse
from multiprocessing import Process
import multiprocessing
import pandas as pd
from numpy.random import default_rng
from pypuf.metrics import uniqueness, uniqueness_data
import pypuf.metrics
import lppuf
from datetime import datetime

parser = argparse.ArgumentParser(description ='Evolutionary algorithm based attack on PUFs.')
parser.add_argument('--target-degree', help="The number of XORs to be used in the XOR arbiter PUF", required=True)
parser.add_argument('--cut-length', help="The cut length to use", required=True)
parser.add_argument('--challenge-num', help="Number of challenges to consider", required=True)
parser.add_argument('--proc', help="Number of processors to split the load on", required=True)
parser.add_argument('--population', help="Population size",required=True)
parser.add_argument('-landscape-evolution',help="Perform landscape evolution (default=False)", action='store_true')
parser.add_argument('-aeomebic-reproduction', help="Perform aeomebic reproduction (default=False)", action='store_true')
args = parser.parse_args()

CHALLENGE_NUM = int(args.challenge_num)
PUF_LENGTH = int(args.target_degree)
CHALLENGE_LENGTH = 64
CUT_LENGTH = int(args.cut_length)

# Thread pool
POPULATION_SIZE = int(args.population)
PROC = int(args.proc)

# Switches
landscape_evolution_switch = args.landscape_evolution
aeomebic_reproduction_switch = args.aeomebic_reproduction

class Chromosome():
    def __init__(self, n, k, external_weights=None, external_biases=None):
        self.weight_vector = None
        self.bias_vector = None
        if(external_weights is None):
            self.weight_vector = np.random.normal(loc=0, scale=0.5, size=(k, n))
        else:
            self.weight_vector = external_weights

        if(external_biases is None):
            self.bias_vector = np.random.normal(loc=0, scale=0.5, size=(k,))
        else:
            self.bias_vector = external_biases

        self.generation = 0
        self.puf = None
        self.fitness = 0
        self.age = 0
        self.mutation_indices = []
        self.bias = 0


    def set_generation(self, generation):
        self.generation = generation

    def print_parameters(self):
        print(self.weight_vector)
        print(self.bias_vector)

    def generate_puf(self):
        if(self.puf is not None):
            del self.puf
        self.puf = XORArbiterPUF(n=CHALLENGE_LENGTH, k=PUF_LENGTH, noisiness=0,
                        external_weights=self.weight_vector, external_biases=self.bias_vector)

    def evaluate_puf_fitness(self, golden_challenge_set, target_responses, optimal_bias):
        self.generate_puf()
        predicted_responses = self.puf.eval(golden_challenge_set)
        ands = 0
        ors = 0
        for index in range(len(predicted_responses)):
            if(predicted_responses[index] == target_responses[index]):
                ands = ands + 1
        return ands / (2 * len(predicted_responses) - ands)

class GeneticAlgoWrapper:
    def __init__(self, targetPUF, challenge_set, response_golden_set, n, k, target_response_set, optimal_bias):
        self.max_population_size = POPULATION_SIZE
        self.initial_max_population_size = self.max_population_size
        self.n = n
        self.k = k
        self.targetPUF = targetPUF
        self.population = []
        self.generate_initial_population()
        self.golden_challenge_set = challenge_set
        self.golden_response_set = response_golden_set
        self.target_responses = target_response_set
        self.last_new_find = 0
        self.mutation_std = 0.5
        self.optimal_bias = optimal_bias
        self.crossover_indexes = []

    def generate_initial_population(self):
        for _ in range(self.max_population_size):
            c = Chromosome(self.n, self.k)
            c.generate_puf()
            self.population.append(c)



    def evaluate_subset_fitness(self, member):
        predicted_responses = member.puf.eval(self.golden_challenge_set)
        ands = 0
        ors = 0
        for index in range(len(predicted_responses)):
            if(predicted_responses[index] == self.target_responses[index]):
                ands = ands + 1
        member.fitness = ands / (2 * len(predicted_responses) - ands)
        member.age = member.age + 1
        member.bias = pypuf.metrics.bias_data(predicted_responses)
        return member

    def evaluate_subset_fitness_acc(self, member):
        predicted_responses = member.puf.eval(self.golden_challenge_set)
        matches = np.sum(predicted_responses == self.target_responses)
        member.age = member.age + 1
        member.bias = pypuf.metrics.bias_data(predicted_responses)
        member.fitness = matches/len(predicted_responses) - np.abs(np.abs(self.optimal_bias) - np.abs(member.bias))
        return member

    def evaluate_population_fitness(self):
        global PROC
        pool = multiprocessing.Pool(processes=PROC)
        self.population = pool.map(self.evaluate_subset_fitness, self.population, chunksize=POPULATION_SIZE // PROC)

    def sort_population_members(self):
        self.population.sort(key=lambda x: x.fitness)
        self.population.reverse()


    def mutate_children_round_robin(self, children):
        mean_list = [-0.4, -0.2, 0, 0.1, 0.3]
        std_list = [0.1, 0.2, 0.3, 0.4, 0.5]

        bias_mutation_rate =  1 / (self.k)

        for child in children:
            mutation_indices = []
            indices_to_change = random.randint(1, CUT_LENGTH - 1)
            for _ in range(indices_to_change):
                if(len(child.mutation_indices) == 0):
                    for j in range(self.n * self.k):
                        child.mutation_indices.append(j)
                mutation_indices.append(child.mutation_indices.pop(random.randint(0, len(child.mutation_indices) - 1) ))

            mutation_std = std_list[random.randint(0, len(std_list) - 1)]
            mean = mean_list[random.randint(0, len(mean_list) - 1)]

            original_weight_vector = copy.deepcopy(child.weight_vector)
            original_bias_vector = copy.deepcopy(child.bias_vector)
            child.weight_vector = child.weight_vector.reshape((self.k * self.n))
            for index in mutation_indices:
                child.weight_vector[index] = np.random.normal(loc=mean, scale=mutation_std)
            child.weight_vector = child.weight_vector.reshape((self.k, self.n))

            for _ in range(1):
                if(random.random() > 0.3):
                    continue
                index = random.randint(0, self.k - 1)
                child.bias_vector[index] = np.random.normal(loc=mean, scale=self.mutation_std)

            evaluated_fitness = child.evaluate_puf_fitness(self.golden_challenge_set, self.target_responses, self.optimal_bias)
            if(evaluated_fitness < child.fitness):
                child.weight_vector = original_weight_vector
                child.bias_vector = original_bias_vector
                child.generate_puf()
            else:
                child.fitness = evaluated_fitness

    def compute_average_population_fitness(self):
        total_fitness = 0
        for member in self.population:
            total_fitness = total_fitness + member.fitness
        return total_fitness/len(self.population)


    def evaluate_pop_bias(self):
        total_bias = 0
        b = []
        for member in self.population:
            total_bias = total_bias + np.abs(member.bias)
            b.append(np.abs(member.bias))
        return total_bias / len(self.population)

    def compute_majority_voting(self, test_challenges, test_responses):
        majority_vote = np.array([0] * len(test_responses))
        for index in range(int(len(self.population))):
            pred = self.population[index].puf.eval(test_challenges)
            majority_vote = majority_vote + pred
        majority_vote[majority_vote < 0] = -1
        majority_vote[majority_vote > 0] = 1
        majority_vote[majority_vote == 0] = 1
        return np.sum(majority_vote == test_responses) / len(test_challenges)


    def compute_majority_voting_test(self, test_challenges, test_responses):
        majority_vote = np.array([0] * len(test_responses))
        for index in range(int(len(self.population) * 0.1)):
            pred = self.population[index].puf.eval(test_challenges)
            majority_vote = majority_vote + pred
        majority_vote[majority_vote < 0] = -1
        majority_vote[majority_vote > 0] = 1
        majority_vote[majority_vote == 0] = 1
        return np.sum(majority_vote == test_responses) / len(test_challenges)

    def attack(self):
        GENERATION = 1
        new_challenge_set = random_inputs(n=CHALLENGE_LENGTH, N=100000, seed=random.randint(0, 600))
        new_response_set = self.targetPUF.eval(new_challenge_set)
        while True:
            random.seed(datetime.now().timestamp())
            np.random.seed(int(datetime.now().timestamp()) + random.randint(0, 10000))

            self.evaluate_population_fitness()
            self.sort_population_members()
            max_fitness = np.sum(self.target_responses == self.population[0].puf.eval(self.golden_challenge_set))/len(self.golden_challenge_set)
            max_fitness_iteration = np.sum(self.target_responses == self.population[0].puf.eval(self.golden_challenge_set))/len(self.golden_challenge_set)
            if(GENERATION % 10 == 0):
                max_observed_fitness = max_fitness_iteration
                print("-----------------")
                print("[!!!] Generational fitness on a test challenge set of 100000 samples:", self.compute_majority_voting_test(new_challenge_set, new_response_set))
                print("-----------------")


            print("Generational high: ", max_fitness,". Avg pop fitness: {avg_fitness}. Maj vote:: {vote}. Avg. bias={bias}.".format(avg_fitness=self.compute_average_population_fitness(), vote=self.compute_majority_voting(self.golden_challenge_set, self.target_responses), bias=self.evaluate_pop_bias()))
            GENERATION = GENERATION + 1
            self.mutate_children_round_robin(self.population)

            if(landscape_evolution_switch and GENERATION % 10 == 0):
                print("Landscape evolution")
                for _ in range(int(0.25 * len(self.golden_challenge_set))):
                    self.golden_challenge_set[random.randint(0, len(self.golden_challenge_set))] = random_inputs(n=CHALLENGE_LENGTH, N=1, seed=random.randint(0, 60000)).reshape((CHALLENGE_LENGTH, ))
                self.target_responses = targetPUF.eval(challenges)

            if(aeomebic_reproduction_switch and GENERATION % 50 == 0):
                print("Aeomebic reproduction")
                self.population.pop(len(self.population) - 1)
                self.population.append(copy.deepcopy(self.population[0]))

challenges=random_inputs(n=CHALLENGE_LENGTH, N=CHALLENGE_NUM, seed=random.randint(0, 60000))
puf_seed = random.randint(0, 50)
targetPUF = XORArbiterPUF(n=CHALLENGE_LENGTH, k=PUF_LENGTH, seed=random.randint(0, 1000))
responses = targetPUF.eval(challenges)
target_vector = []
optimal_bias = pypuf.metrics.bias_data(responses)
print("Bias: ", optimal_bias)
geneticAlgoWrapper = GeneticAlgoWrapper(targetPUF, challenges, target_vector, CHALLENGE_LENGTH, PUF_LENGTH, responses, optimal_bias)
geneticAlgoWrapper.attack()
