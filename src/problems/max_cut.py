from .base_problem import BaseProblem
import numpy as np

class WeightedMAXCUT(BaseProblem):
    def __init__(self, n, min_weight: int = 1, max_weight: int = 5, beta_a: int = 100, beta_b: int = 1, seed: int = 42):
        self.n = n
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.beta_a = beta_a
        self.beta_b = beta_b

        self.generate_instance(seed)

    def generate_instance(self, seed: int = 42):
        rng = np.random.RandomState(seed)

        num_edges = (self.n * (self.n - 1)) // 2

        raw_weights = rng.beta(self.beta_a, self.beta_b, size=num_edges)

        scale_range = self.max_weight - self.min_weight
        scaled_weights = self.min_weight + scale_range * raw_weights

        self.weight_matrix = np.zeros((self.n, self.n))
        
        edge_idx = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.weight_matrix[i, j] = scaled_weights[edge_idx]
                self.weight_matrix[j, i] = scaled_weights[edge_idx]
                edge_idx += 1

        self.total_weight = np.sum(self.weight_matrix) / 2

    def evaluate(self, individual):
        if len(individual) != self.n:
            raise ValueError(f"Individual length ({len(individual)}) must match number of nodes (n={self.n})")

        cut_weight = 0.0

        for i in range(self.n):
            for j in range(i + 1, self.n):
                if individual[i] != individual[j]:
                    cut_weight += self.weight_matrix[i, j]

        return cut_weight / self.total_weight