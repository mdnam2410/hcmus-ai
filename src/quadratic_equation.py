"""Solve quadratic equation using genetic algorithm.
"""

import copy
import math
import random
import struct
import numpy as np
import matplotlib.pyplot as plt
import operator

GENERATIONS = 10
MUTATION_PROBABILITY = 0.30
POPULATION_SIZE = 1000
SOLUTION_RANGE = 1000
PRECISION = 1000000

t = map(int, input().split())
A, B, C = t


def float_to_binary(idv: float):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('>f', idv))

def binary_to_float(s):
    b = int(s, 2).to_bytes(len(s) // 8, byteorder='big')
    return struct.unpack('>f', b)[0]

def fitness(idv: float):
    """Computes the fitness score of the individual

    Args:
        idv (float): An individual

    Returns:
        float: Fitness score
    """
    global A, B, C
    try:
        f = 1 / abs(A * idv ** 2 + B * idv + C)
        if f >= PRECISION:
            return math.inf
        return f
    except ZeroDivisionError:
        return math.inf

def reproduce(idv1: str, idv2: str):
    cross_point = random.randint(0, len(idv1) - 1)
    child = copy.deepcopy(idv1[0:cross_point]) + copy.deepcopy(idv2[cross_point:])
    return child
    

def mutate(idv: str):
    for c in idv:
        if np.random.choice([0, 1], p=[1 - MUTATION_PROBABILITY, MUTATION_PROBABILITY]) == 1:
            c = '1' if c == '0' else '0'

def best_individual(population):
    t = list(map(fitness, population))
    i, _ = max(enumerate(t), key=operator.itemgetter(1))
    return population[i]

def solve():
    population = np.random.uniform(-SOLUTION_RANGE, SOLUTION_RANGE, size=POPULATION_SIZE)
    has_goal = False
    goals = []

    # For statistics
    max_fitness_scores = []
    best_individuals = []
    prev = 0
    for i in range(GENERATIONS):
        print('Generation', i)
        max_fitness_scores.append(prev)
        best_individuals.append(best_individual(population))

        # Computes fitness score for each individual
        fitness_scores = np.array([fitness(x) for x in population])

        # Checks whether the population contains goals
        for i, idv in enumerate(fitness_scores):
            if math.isinf(idv):
                has_goal = True
                goals.append(population[i])

        if has_goal:
            return max_fitness_scores, goals, best_individuals
        else:
            has_goal = False
        
        # Computes probability associated to each individual
        s = np.sum(fitness_scores)
        probabilites = np.array([x / s for x in fitness_scores])

        new_population = np.empty(shape=(POPULATION_SIZE,))
        for j in range(POPULATION_SIZE):
            # Chooses two random individuals to reproduce
            parent1 = np.random.choice(population, p=probabilites)
            parent2 = np.random.choice(population, p=probabilites)

            # Reproduces
            child = reproduce(float_to_binary(parent1), float_to_binary(parent2))

            # Mutates
            mutate(child)

            child = binary_to_float(child)
            new_population[j] = child
            
        population = new_population
        prev = np.max(fitness_scores)
    
    return max_fitness_scores, goals, best_individuals

if __name__ == '__main__':
    scores, goal, best_individuals = solve()
    print(goal)
    # plt.plot(scores)
    plt.plot(best_individuals)
    plt.show()