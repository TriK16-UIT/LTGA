from problems.one_max import OneMax
from problems.deceptive_trap import DeceptiveTrap
from measures.joint_entropy import JointEntropy
from measures.mutual_information import MutualInformation
from measures.mutual_normalized_information import MutualNormalizedInformation
from measures.normalized_variation_information import NormalizedVariationInformation
from measures.variation_information import VariationInformation
from experiments.experiments import run_experiments, plot_population_results, plot_evaluation_results, save_experiment_results

measures_dict = {
    "JointEntropy": JointEntropy(),
    "MutualInformation": MutualInformation(),
    "MutualNormalizedInformation": MutualNormalizedInformation(),
    "NormalizedVariationInformation": NormalizedVariationInformation(),
    "VariationInformation": VariationInformation()
}

problem_sizes = [2, 4, 6]

trap_results = run_experiments(
    problem_class=DeceptiveTrap,
    measures_dict=measures_dict,
    problem_sizes=problem_sizes,
    trap_size=2
)

save_experiment_results(trap_results, "results/trap2_results.pkl")

plot_population_results("Trap2", trap_results)
plot_evaluation_results("Trap2", trap_results)