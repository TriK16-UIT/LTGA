import numpy as np
from .base_measure import BaseMeasure

class VariationInformation(BaseMeasure):
    def __init__(self):
        super().__init__()
        self.is_distance_measure = True
    
    def calculate(self, population, var_i, var_j):
        prob_i = self.get_marginal_probabilities(population, var_i)
        prob_j = self.get_marginal_probabilities(population, var_j)
        joint_probs = self.get_joint_probabilities(population, var_i, var_j)
        
        h_i = self.calculate_entropy(np.array(list(prob_i.values())))
        h_j = self.calculate_entropy(np.array(list(prob_j.values())))
        h_ij = self.calculate_entropy(np.array(list(joint_probs.values())))
        
        return 2 * h_ij - h_i - h_j