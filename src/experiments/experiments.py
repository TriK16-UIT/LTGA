import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from LGTA.ltga import LTGA
import os
import pickle

def save_experiment_results(all_results, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(all_results, f)

def load_experiment_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# -----------------------------
# NEW: Pre-generate fixed instances
# -----------------------------

def generate_fixed_instances(problem_class, problem_size, num_instances=100, base_seed=12345, **problem_kwargs):
    """Generate a fixed set of problem instances for a given problem size"""
    instances = []
    for i in range(num_instances):
        if problem_class.__name__ == "NKS1Landscape":
            problem_kwargs['seed'] = base_seed + i  # Fixed seeds for instances
            problem_kwargs['n'] = problem_size
        instance = problem_class(**problem_kwargs)
        instances.append(instance)
    return instances

# -----------------------------
# Experiment core functions (MODIFIED)
# -----------------------------

def run_single_experiment(problem_instance, measure, population_size, problem_size, max_generations, rng_seed):
    """Run experiment on a pre-generated problem instance"""
    ltga = LTGA(problem_instance, measure, population_size, problem_size, max_generations, rng_seed)
    best_solution, best_fitness, total_evaluations = ltga.run()

    success = best_fitness >= 1.0
    return success, total_evaluations


def evaluate_population_size(problem_instances, measure, problem_size, population_size, max_generations, rng_seed):
    """Evaluate using the same set of problem instances"""
    success_count = 0
    total_evals_list = []

    for run_id in range(100):
        seed = rng_seed + run_id  # independent seeds for algorithm
        problem_instance = problem_instances[run_id]  # Use pre-generated instance
        success, total_evals = run_single_experiment(problem_instance, measure, population_size, problem_size, max_generations, seed)
        total_evals_list.append(total_evals)

        if success:
            success_count += 1

        if success_count >= 99:
            return True, total_evals_list

    return False, total_evals_list


def bisection_search(problem_instances, measure, problem_size, max_generations, rng_seed):
    """Bisection search using fixed problem instances"""
    lower = 1
    upper = 1

    while True:
        success, total_evals = evaluate_population_size(problem_instances, measure, problem_size, upper, max_generations, rng_seed)
        if success:
            break
        upper *= 2

    lower = upper // 2

    while lower < upper:
        mid = (lower + upper) // 2
        success, total_evals = evaluate_population_size(problem_instances, measure, problem_size, mid, max_generations, rng_seed)
        if success:
            upper = mid
        else:
            lower = mid + 1

    _, final_total_evals = evaluate_population_size(problem_instances, measure, problem_size, upper, max_generations, rng_seed)
    
    return upper, final_total_evals

# -----------------------------
# Full experiment function (MODIFIED)
# -----------------------------

def run_experiments(problem_class, measures_dict, problem_sizes, max_generations=None, base_rng_seed=42, **problem_kwargs):
    all_results = {}
    
    # Pre-generate fixed instances for each problem size
    print("Pre-generating fixed problem instances...")
    fixed_instances = {}
    for problem_size in problem_sizes:
        print(f"  Generating 100 instances for problem size {problem_size}...")
        fixed_instances[problem_size] = generate_fixed_instances(problem_class, problem_size, **problem_kwargs)

    for measure_name, measure in measures_dict.items():
        print(f"\nRunning experiments for measure: {measure_name}")
        population_results = {}
        evaluations_results = {}

        for problem_size in problem_sizes:
            print(f"  Problem size {problem_size}...")
            minimal_pops = []
            all_evals = []

            # Use the same fixed instances for all 10 trials
            problem_instances = fixed_instances[problem_size]

            for trial in tqdm(range(10)):
                trial_seed = base_rng_seed + trial * 1000  # independent seeds for each trial
                minimal_pop, evals_list = bisection_search(problem_instances, measure, problem_size, max_generations, trial_seed)
                minimal_pops.append(minimal_pop)
                all_evals.extend(evals_list)
                print(f"    Trial {trial+1}: minimal population size = {minimal_pop} - Seed: {trial_seed}")

            population_results[problem_size] = minimal_pops
            evaluations_results[problem_size] = all_evals

        all_results[measure_name] = {
            "population": population_results,
            "evaluations": evaluations_results
        }

    return all_results

# -----------------------------
# Plotting functions (UNCHANGED)
# -----------------------------

def plot_population_results(problem_name, all_results, save_dir = "plots"):
    plt.figure(figsize=(7, 5))

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

    markers = ['o', 'x', 's', 'D', '+', '|', '^', '*']
    linestyles = ['-', '--', '-.', ':']
    measure_names = list(all_results.keys())

    for i, measure_name in enumerate(measure_names):
        results = all_results[measure_name]
        population_results = results["population"]

        problem_sizes = sorted(population_results.keys())
        means = []

        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        color = colors[i % len(colors)]

        for problem_size in problem_sizes:
            pops = population_results[problem_size]
            means.append(np.mean(pops))
            plt.scatter([problem_size]*len(pops), pops, marker=marker, alpha=0.3, label=None, color=color)

        plt.plot(problem_sizes, means, linestyle=linestyle, marker=marker, label=measure_name)

    plt.xlabel("Problem Size")
    plt.ylabel("Minimal Population Size")
    plt.title(f"{problem_name}: Population Size Scaling")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.grid()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{problem_name}_population.png"), dpi=300, bbox_inches='tight')

    plt.show()

def plot_evaluation_results(problem_name, all_results, save_dir = "plots"):
    plt.figure(figsize=(7, 5))

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

    markers = ['o', 'x', 's', 'D', '+', '|', '^', '*']
    linestyles = ['-', '--', '-.', ':']
    measure_names = list(all_results.keys())

    for i, measure_name in enumerate(measure_names):
        results = all_results[measure_name]
        evaluations_results = results["evaluations"]

        problem_sizes = sorted(evaluations_results.keys())
        means = []

        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        color = colors[i % len(colors)]

        for problem_size in problem_sizes:
            evals = evaluations_results[problem_size]
            means.append(np.mean(evals))
            plt.scatter([problem_size]*len(evals), evals, marker=marker, alpha=0.3, label=None, color=color)

        plt.plot(problem_sizes, means, linestyle=linestyle, marker=marker, label=measure_name)

    plt.xlabel("Problem Size")
    plt.ylabel("Function Evaluations")
    plt.title(f"{problem_name}: Evaluations Scaling")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.grid()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{problem_name}_evaluations.png"), dpi=300, bbox_inches='tight')

    plt.show()