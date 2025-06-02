from .base_problem import BaseProblem
import numpy as np

class OneMax(BaseProblem):
    def __init__(self):
        pass

    def evaluate(self, individual):
        return np.sum(individual)


