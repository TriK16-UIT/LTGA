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

    def initialize_population(self):
        self.population = self.rng.randint(0, 2, size=(self.population_size, self.problem_size))
        self.fitness_values = np.zeros(self.population_size)

        for i in range(self.population_size):
            self.fitness_values[i] = self.problem.evaluate(self.population[i])
            
        best_idx = np.argmax(self.fitness_values)
        self.best_fitness = self.fitness_values[best_idx]
        self.best_solution = self.population[best_idx].copy()

    def tournament_selection(self, tournament_size=2):
        selected_indices = np.zeros(self.population_size, dtype=int)
        
        for i in range(self.population_size):
            tournament_indices = self.rng.choice(self.population_size, tournament_size, replace=False)
            
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
        for i in range(self.population_size):
            solution = self.population[i].copy()
            original_fitness = self.fitness_values[i]
            improved = False

            shuffled_cluster_indices = self.rng.permutation(len(self.clusters))

            for idx in shuffled_cluster_indices:
                cluster = self.clusters[idx]
                
                if len(cluster) <= 1:
                    continue

                donor_candidates = np.arange(self.population_size)
                donor_candidates = donor_candidates[donor_candidates != i]  
                donor_idx = self.rng.choice(donor_candidates, 1)[0]

                new_solution = solution.copy()
                new_solution[cluster] = self.population[donor_idx, cluster]

                new_fitness = self.problem.evaluate(new_solution)
                
                if new_fitness >= self.fitness_values[i]:
                    solution = new_solution
                    self.fitness_values[i] = new_fitness
                    improved = True

                    if new_fitness > self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_solution = new_solution.copy()

            self.population[i] = solution

            if not improved:
                self.population[i] = self.forced_improvement(solution, i)

    def forced_improvement(self, solution, idx):
        fi_solution = solution.copy()
        original_fitness = self.problem.evaluate(fi_solution)
        
        shuffled_cluster_indices = self.rng.permutation(len(self.clusters))
        
        for c_idx in shuffled_cluster_indices:
            cluster = self.clusters[c_idx]
            
            if len(cluster) <= 1:
                continue
            
            new_solution = fi_solution.copy()
            new_solution[cluster] = self.best_solution[cluster]
            
            new_fitness = self.problem.evaluate(new_solution)
            
            if new_fitness > original_fitness:
                fi_solution = new_solution
                self.fitness_values[idx] = new_fitness
                original_fitness = new_fitness
                
                if new_fitness >= self.best_fitness:
                    break
        
        return fi_solution
    
    def has_converged(self):
        if self.population is None or len(self.population) == 0:
            return False
        
        return np.all(np.all(self.population == self.population[0], axis=1))
    
    def run(self):
        if self.population is None:
            self.initialize_population()
        
        generation = 0
        converged = False
        
        while (self.max_generations is None or generation < self.max_generations) and not converged:
            self.tournament_selection()
            
            self.learn_linkage_tree()
            
            self.genepool_optimal_mixing()
            
            converged = self.has_converged()
            
            generation += 1
        
        return self.best_solution, self.best_fitness                 