from abc import ABC, abstractmethod
from collections import Counter
import numpy as np

class BaseMeasure(ABC):
    def __init__(self):
        self.is_distance_measure = False
        
    @abstractmethod
    def calculate(self, population, var_i, var_j):
        pass
    
    @staticmethod
    def calculate_entropy(probabilities):
        valid_probs = probabilities[probabilities > 0]
        return -np.sum(valid_probs * np.log2(valid_probs))
    
    @staticmethod
    def get_marginal_probabilities(population, var_idx):
        n_individuals = population.shape[0]
        values, counts = np.unique(population[:, var_idx], return_counts=True)
        return {val: count / n_individuals for val, count in zip(values, counts)}
    
    @staticmethod
    def get_joint_probabilities(population, var_i, var_j):
        n_individuals = population.shape[0]
        
        data = population[:, [var_i, var_j]]
        
        tuples = [tuple(row) for row in data]
        
        unique_combinations = {}
        for t in tuples:
            if t in unique_combinations:
                unique_combinations[t] += 1
            else:
                unique_combinations[t] = 1
        
        return {comb: count / n_individuals for comb, count in unique_combinations.items()}
        