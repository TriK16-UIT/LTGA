import numpy as np
from typing import List, Tuple, Optional, Callable, Union, Type
import random
from tqdm import tqdm

from ..measures.base_measure import BaseMeasure

class LTGA:
    def __init__(self,
                 problem,
                 measure: BaseMeasure,
                 population_size: int = 50,
                 max_evaluations: int = 10000,
                 tournament_size: int = 2,
                 verbose: bool = True):
              
        self.problem = problem
        self.measure = measure
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.tournament_size = tournament_size
        self.verbose = verbose
        
        self.evaluations = 0
        self.population = None
        self.fitness_values = None
        self.best_solution = None
        self.best_fitness = float('-inf')
        
    def initialize_population(self):
        self.population = np.random.randint(0, 2, 
                                            size=(self.population_size, 
                                                 self.problem.n_variables))
        self.fitness_values = np.zeros(self.population_size)
        
        # Evaluate initial population
        for i in range(self.population_size):
            self.fitness_values[i] = self.problem.evaluate(self.population[i])
            self.evaluations += 1
            
            if self.fitness_values[i] > self.best_fitness:
                self.best_fitness = self.fitness_values[i]
                self.best_solution = self.population[i].copy()
    
              
              
