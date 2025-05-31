import numpy as np
from .base_measure import BaseMeasure

class JointEntropy(BaseMeasure):
    def __init__(self):
        super().__init__()
        self.is_distance_measure = True
    
    def calculate(self, population, var_i, var_j):
        joint_probs = self.get_joint_probabilities(population, var_i, var_j)
        h_ij = self.calculate_entropy(np.array(list(joint_probs.values())))
        return h_ij
    
    