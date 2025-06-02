from .base_problem import BaseProblem
import numpy as np

class NKLandscape(BaseProblem):
    def __init__(self, n: int, k: int = 5, s: int = 1, seed: int = 42):
        if n < k:
            raise ValueError(f"Problem size (n={n}) must be at least k={k}")

        self.n = n
        self.k = k
        self.s = s

        self.num_subfunctions = max(0, (n - k) // s + 1)

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
        evaluated_subfunctions = 0

        for i in range(self.num_subfunctions):
            start_idx = i * self.s
            end_idx = start_idx + self.k

            if end_idx > self.n:
                break

            subfunction = individual[start_idx:end_idx]
            
            idx = 0
            for j in range(self.k):
                idx += subfunction[j] * (2 ** (self.k - j - 1))

            fitness += self.subfunction_tables[i][int(idx)]
            evaluated_subfunctions += 1

        return fitness / evaluated_subfunctions if evaluated_subfunctions > 0 else 0.0