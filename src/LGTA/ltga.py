import numpy as np
from scipy.cluster.hierarchy import linkage

class LTGA:
    def __init__(self, problem, measure, population_size, problem_size, max_generations=None, seed=42):
        self.problem = problem
        self.measure = measure
        self.population_size = population_size
        self.problem_size = problem_size
        self.max_generations = max_generations

        self.rng = np.random.RandomState(seed)

        self.population = None
        self.fitness_values = None
        self.linkage_tree = None
        self.clusters = None
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.total_evaluations = 0  

    def initialize_population(self):
        self.population = self.rng.randint(0, 2, size=(self.population_size, self.problem_size))
        self.fitness_values = np.zeros(self.population_size)

        for i in range(self.population_size):
            self.fitness_values[i] = self.problem.evaluate(self.population[i])
            self.total_evaluations += 1  

    def tournament_selection(self, tournament_size=2):
        if self.population_size == 1:
            return
        
        selected_indices = np.zeros(self.population_size, dtype=int)
        
        for i in range(self.population_size):
            replace_flag = self.population_size < tournament_size
            tournament_indices = self.rng.choice(self.population_size, tournament_size, replace=replace_flag)
            tournament_fitness = self.fitness_values[tournament_indices]
            winner_in_tournament = np.argmax(tournament_fitness)     
            selected_indices[i] = tournament_indices[winner_in_tournament]
        
        self.population = self.population[selected_indices].copy()
        self.fitness_values = self.fitness_values[selected_indices].copy()

    def learn_linkage_tree(self):
        n_vars = self.problem_size

        condensed_distance = np.zeros((n_vars * (n_vars - 1)) // 2)
        idx = 0
        
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                distance = self.measure.calculate(self.population, i, j)
                if not self.measure.is_distance_measure:
                    distance = 1.0 - distance
                condensed_distance[idx] = distance
                idx += 1

        self.linkage_tree = linkage(condensed_distance, method='average')

        clusters = []
        current_clusters = {i: [i] for i in range(n_vars)}
        next_cluster_idx = n_vars
        
        for row in self.linkage_tree:
            i, j = int(row[0]), int(row[1])
            new_cluster = current_clusters[i] + current_clusters[j]
            clusters.append(new_cluster)
            current_clusters[next_cluster_idx] = new_cluster
            next_cluster_idx += 1

        self.clusters = clusters

    def genepool_optimal_mixing(self):
        new_population = []
        new_fitness = []

        for i in range(self.population_size):
            solution, fitness_value = self.fi_gom(self.population[i], self.fitness_values[i])
            new_population.append(solution)
            new_fitness.append(fitness_value)

        self.population = np.array(new_population)
        self.fitness_values = np.array(new_fitness)

    def fi_gom(self, solution, fitness_value):
        b = solution.copy()
        fitness_b = fitness_value
        improved = False

        shuffled_cluster_indices = self.rng.permutation(len(self.clusters))

        # Standard GOM phase
        for idx in shuffled_cluster_indices:
            cluster = self.clusters[idx]

            donor_idx = self.rng.choice(self.population_size)
            donor = self.population[donor_idx]
            o = b.copy()
            o[cluster] = donor[cluster]

            if not np.array_equal(o[cluster], b[cluster]):
                fitness_o = self.problem.evaluate(o)
                self.total_evaluations += 1  

                if fitness_o > fitness_b:
                    b[cluster] = o[cluster]
                    fitness_b = fitness_o
                    improved = True
                else:
                    o[cluster] = b[cluster]  

        # Forced Improvement phase if not improved in normal GOM
        if not improved:
            for idx in shuffled_cluster_indices:
                cluster = self.clusters[idx]
                o = b.copy()
                o[cluster] = self.best_solution[cluster]

                if not np.array_equal(o[cluster], b[cluster]):
                    fitness_o = self.problem.evaluate(o)
                    self.total_evaluations += 1  

                    if fitness_o > fitness_b:
                        b[cluster] = o[cluster]
                        fitness_b = fitness_o
                        break  
                    else:
                        o[cluster] = b[cluster]  
            
            if fitness_b == fitness_value:
                b = self.best_solution.copy()
                fitness_b = self.best_fitness

        return b, fitness_b

    def has_converged(self):
        if self.population is None or len(self.population) == 0:
            return False
        
        return np.all(np.all(self.population == self.population[0], axis=1))
    
    def run(self):
        if self.population is None:
            self.initialize_population()

        self.best_fitness = np.max(self.fitness_values)
        self.best_solution = self.population[np.argmax(self.fitness_values)].copy()
        
        generation = 0
        converged = False
        
        while (self.max_generations is None or generation < self.max_generations) and not converged:
            self.tournament_selection()
            self.learn_linkage_tree()
            self.genepool_optimal_mixing()

            self.best_fitness = np.max(self.fitness_values)
            self.best_solution = self.population[np.argmax(self.fitness_values)].copy()

            converged = self.has_converged()
            generation += 1
        
        return self.best_solution, self.best_fitness, self.total_evaluations  
