import numpy as np
from typing import List, Dict, Tuple, Optional, Type, Union
import matplotlib.pyplot as plt
import networkx as nx

from ..measures.base_measure import BaseMeasure


class LinkageTree:
    def __init__(self, measure: BaseMeasure):
        self.measure = measure
        self.clusters = []
        self.distances = {}
        self.merge_history = []
        self.tree = None
        
    def build(self, population: np.ndarray) -> List[List[int]]:
        """
        Build the linkage tree based on the population.
        
        Args:
            population (np.ndarray): The population matrix, where each row is an individual
                                    and each column is a variable/gene.
                                    
        Returns:
            List[List[int]]: The hierarchical clusters of variables.
        """
        n_vars = population.shape[1]
        
        # Initialize clusters with single variables
        self.clusters = [[i] for i in range(n_vars)]
        all_clusters = self.clusters.copy()
        
        # Compute initial pairwise distances
        self.distances = {}
        for i in range(len(self.clusters)):
            for j in range(i+1, len(self.clusters)):
                var_i, var_j = self.clusters[i][0], self.clusters[j][0]
                distance = self.measure.calculate(population, var_i, var_j)
                
                # Adjust distance based on the measure type
                if not self.measure.is_distance_measure:
                    # For similarity measures (e.g., mutual information),
                    # higher values indicate stronger relationships,
                    # so we need to invert for distance
                    distance = -distance
                    
                self.distances[(i, j)] = distance
        
        self.merge_history = []
        
        # Main loop for agglomerative clustering
        while len(self.clusters) > 1:
            # Find closest pair of clusters
            min_dist = float('inf')
            min_pair = None
            
            for i in range(len(self.clusters)):
                for j in range(i+1, len(self.clusters)):
                    if (i, j) in self.distances and self.distances[(i, j)] < min_dist:
                        min_dist = self.distances[(i, j)]
                        min_pair = (i, j)
            
            if min_pair is None:
                break
                
            i, j = min_pair
            
            # Merge clusters
            merged = self.clusters[i] + self.clusters[j]
            self.merge_history.append((self.clusters[i], self.clusters[j], merged))
            
            # Remove old clusters and add the merged one
            cluster_i = self.clusters.pop(j)  # Remove j first since j > i
            cluster_j = self.clusters.pop(i)
            self.clusters.append(merged)
            all_clusters.append(merged)
            
            # Update distances
            new_idx = len(self.clusters) - 1
            for k in range(new_idx):
                # Compute distance between new merged cluster and each existing cluster
                distance = self._compute_cluster_distance(population, self.clusters[k], merged)
                self.distances[(k, new_idx)] = distance
            
            # Remove old distances
            keys_to_remove = []
            for key in self.distances:
                if i in key or j in key:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                if key in self.distances:
                    del self.distances[key]
        
        # Return all clusters, including intermediate ones
        return all_clusters
    
    def _compute_cluster_distance(self, population: np.ndarray, 
                                 cluster_a: List[int], 
                                 cluster_b: List[int]) -> float:
        """
        Compute the distance between two clusters.
        
        Args:
            population (np.ndarray): The population matrix.
            cluster_a (List[int]): First cluster indices.
            cluster_b (List[int]): Second cluster indices.
            
        Returns:
            float: The distance between the clusters.
        """
        # Compute average pairwise distance between variables in the clusters
        total_distance = 0.0
        count = 0
        
        for var_i in cluster_a:
            for var_j in cluster_b:
                # Skip duplicate calculations
                if var_i == var_j:
                    continue
                
                distance = self.measure.calculate(population, var_i, var_j)
                
                # Adjust for similarity measures
                if not self.measure.is_distance_measure:
                    distance = -distance
                
                total_distance += distance
                count += 1
        
        if count == 0:
            return float('inf')
        
        return total_distance / count
    
    def visualize(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the linkage tree.
        
        Args:
            save_path (str, optional): Path to save the visualization.
        """
        G = nx.DiGraph()
        
        # Add nodes for all clusters
        for i, cluster in enumerate(self.merge_history):
            parent_cluster = cluster[2]
            left_cluster = cluster[0]
            right_cluster = cluster[1]
            
            # Convert lists to strings for node labels
            parent_label = f"Cluster {i}: {parent_cluster}"
            left_label = f"{left_cluster}"
            right_label = f"{right_cluster}"
            
            G.add_node(parent_label)
            G.add_node(left_label)
            G.add_node(right_label)
            
            G.add_edge(parent_label, left_label)
            G.add_edge(parent_label, right_label)
        
        plt.figure(figsize=(12, 8))
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        nx.draw(G, pos, with_labels=True, node_color="lightblue",
                node_size=2000, font_size=8, arrows=False)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
