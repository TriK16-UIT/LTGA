from .base_problem import BaseProblem
import numpy as np

class DeceptiveTrap(BaseProblem):
    def __init__(self, trap_size: int = 5):
        self.trap_size = trap_size

    def evaluate(self, individual):
        fitness = 0.0
        num_traps = len(individual) // self.trap_size

        for i in range(0, len(individual), self.trap_size):
            sub_block = individual[i:i + self.trap_size]
            u = np.sum(sub_block)   

            if u == self.trap_size:
                sub_fitness = 1.0
            else:
                sub_fitness = (self.trap_size - 1 - u) / self.trap_size
            
            fitness += sub_fitness

        return fitness / num_traps