from .base_problem import BaseProblem
import numpy as np

class NKS1Landscape(BaseProblem):
    def __init__(self, n: int, k: int = 5, seed: int = 42):
        if n < k:
            raise ValueError(f"Problem size (n={n}) must be at least k={k}")

        self.n = n
        self.k = k
        
        self.num_subfunctions = n - k + 1
        
        self.generate_instance(seed)

    def generate_instance(self, seed: int = 42):
        rng = np.random.RandomState(seed)
        
        self.subfunction_tables = []
        
        for _ in range(self.num_subfunctions):
            table = rng.uniform(0, 1, 2**self.k)
            self.subfunction_tables.append(table)

    def evaluate(self, individual):
        if len(individual) != self.n:
            raise ValueError(f"Individual length ({len(individual)}) must match problem size (n={self.n})")

        fitness = 0.0

        for i in range(self.num_subfunctions):
            subfunction = individual[i:i+self.k]
            
            idx = 0
            for j in range(self.k):
                idx += subfunction[j] * (2 ** (self.k - j - 1))

            fitness += self.subfunction_tables[i][int(idx)]

        return round(fitness / self.num_subfunctions, 1)