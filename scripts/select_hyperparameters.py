import argparse
import itertools
import json
import math
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml


CLASSIFIER_EVAL_BUDGET = 10_000
POPULATION_GENERATION_PAIRS = [
    {"populacao_inicial": 50, "num_geracoes": 199},
    {"populacao_inicial": 100, "num_geracoes": 99},
    {"populacao_inicial": 200, "num_geracoes": 49},
]
EBIE_MUTATION_PROBS = [0.1, 0.2, 0.4]
EBIE_MUTATION_INTENSITIES = [0.05, 0.10, 0.20]
EBIE_CROSSOVER_PROBS = [0.5, 0.8, 0.95]
EBIE_ADD_TOKEN_PROBS = [0.1, 0.3, 0.5]
EBIE_REMOVE_TOKEN_PROBS = [0.1, 0.3, 0.5]
RCGA_MUTATION_PROBS = [0.1, 0.2, 0.4]
RCGA_CROSSOVER_PROBS = [0.5, 0.8, 0.95]
RCGA_SIGMAS = [0.05, 0.10, 0.20]
CMA_ES_POPULATION_SIZES = [16, 32, 64]
CMA_ES_SIGMAS = [0.1, 0.3, 0.5, 1.0]
RANDOM_SEARCH_SIGMAS = [0.05, 0.10, 0.20, 0.50]
HILL_SIGMAS = [0.05, 0.10, 0.20, 0.50]
HILL_NEIGHBORS = [1, 5, 10, 50]

METHOD_ORDER = ["ebie", "rcga", "cma_es", "random_search", "hill_climbing"]
ALGORITHM_BY_METHOD = {
    "ebie": "genetic",
    "rcga": "rcga",
    "cma_es": "cma_es",
    "random_search": "random_search",
    "hill_climbing": "hill_climbing",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run resumable hyperparameter selection for EBIE and baselines.")
    parser.add_argument("--config", default="config.yaml", help="Base YAML config.")
    parser.add_argument("--resume-run", help="Existing timestamp under generated_configs/hyperparameter_selection to resume.")
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=METHOD_ORDER, help="Methods to run, in the requested order unless omitted.")
    parser.add_argument("--num-runs", type=int, help="Override number of seeds/runs per configuration.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed when --num-runs is used.")
    return parser.parse_args()


def load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def slugify(value):
    return str(value).replace(".", "p")


def build_run_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def output_filename(algorithm, experiment_name, decoder_name="current_decoder"):
    return f"historico_completo_{algorithm}_{decoder_name}_{experiment_name}.json"


def build_sort_key(row):
    success_rate = row["success_rate"]
    evaluations_to_target_mean = row["evaluations_to_target_mean"]
    best_fitness_mean = row["best_fitness_mean"]
    best_fitness_std = row["best_fitness_std"]
    return (
        -1.0 if success_rate is None else -success_rate,
        math.inf if evaluations_to_target_mean is None else evaluations_to_target_mean,
        1.0 if best_fitness_mean is None else -best_fitness_mean,
        math.inf if best_fitness_std is None else best_fitness_std,
        row["experiment_name"],
    )


def is_completed(output_path):
    if not output_path.exists():
        return False
    try:
        payload = read_json(output_path)
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("progress", {}).get("status") == "completed"


def common_config(base_config, method, experiment_name):
    config = deepcopy(base_config)
    config["algorithms"] = [ALGORITHM_BY_METHOD[method]]
    config["experiment_name"] = experiment_name
    config["is_hyperparameter_selection"] = True
    config["save_run_history"] = False
    config["save_detailed_run_artifacts"] = False
    config["run_decoder_ablation"] = False
    config["decoder_config_name"] = "current_decoder"
    config["classifier_evaluation_budget"] = CLASSIFIER_EVAL_BUDGET
    config["total_classifier_evaluation_budget"] = CLASSIFIER_EVAL_BUDGET
    return config


def apply_seed_override(config, args):
    if args.num_runs is None:
        return
    config["experiment_seeds"] = list(range(args.seed_start, args.seed_start + args.num_runs))
    config["num_execucoes"] = args.num_runs


def build_manifest_entry(run_timestamp, method, config, config_path, output_file):
    return {
        "run_timestamp": run_timestamp,
        "method": method,
        "algorithm": config["algorithms"][0],
        "experiment_name": config["experiment_name"],
        "config_path": str(config_path),
        "output_file": str(output_file),
        "classifier_evaluation_budget": CLASSIFIER_EVAL_BUDGET,
        "parameters": {key: value for key, value in config.items() if not key.startswith("_")},
    }


def generate_ebie_stage1_configs(base_config, args):
    index = 1
    for schedule, mutation_prob, mutation_intensity in itertools.product(POPULATION_GENERATION_PAIRS, EBIE_MUTATION_PROBS, EBIE_MUTATION_INTENSITIES):
        experiment_name = (
            f"ebie_stage1_{index:03d}"
            f"_pop{schedule['populacao_inicial']}"
            f"_gen{schedule['num_geracoes']}"
            f"_pmut{slugify(mutation_prob)}"
            f"_mint{slugify(mutation_intensity)}"
            "_pcross0p8"
        )
        config = common_config(base_config, "ebie", experiment_name)
        config.update(schedule)
        config["prob_mutacao_embedding"] = mutation_prob
        config["mutation_intensity_percent"] = mutation_intensity
        config["prob_crossover_embedding"] = 0.8
        config["prob_add_random_token"] = 0.3
        config["prob_remover_token"] = 0.3
        config["classifier_evaluation_budget_kind"] = "total_including_initial_population"
        apply_seed_override(config, args)
        yield config
        index += 1


def generate_ebie_stage2_configs(base_config, args, best_stage1):
    index = 1
    for crossover_prob, add_prob, remove_prob in itertools.product(EBIE_CROSSOVER_PROBS, EBIE_ADD_TOKEN_PROBS, EBIE_REMOVE_TOKEN_PROBS):
        experiment_name = (
            f"ebie_stage2_{index:03d}"
            f"_pop{best_stage1['populacao_inicial']}"
            f"_gen{best_stage1['num_geracoes']}"
            f"_pmut{slugify(best_stage1['prob_mutacao_embedding'])}"
            f"_mint{slugify(best_stage1['mutation_intensity_percent'])}"
            f"_pcross{slugify(crossover_prob)}"
            f"_padd{slugify(add_prob)}"
            f"_prem{slugify(remove_prob)}"
        )
        config = common_config(base_config, "ebie", experiment_name)
        config["populacao_inicial"] = best_stage1["populacao_inicial"]
        config["num_geracoes"] = best_stage1["num_geracoes"]
        config["prob_mutacao_embedding"] = best_stage1["prob_mutacao_embedding"]
        config["mutation_intensity_percent"] = best_stage1["mutation_intensity_percent"]
        config["prob_crossover_embedding"] = crossover_prob
        config["prob_add_random_token"] = add_prob
        config["prob_remover_token"] = remove_prob
        config["classifier_evaluation_budget_kind"] = "total_including_initial_population"
        apply_seed_override(config, args)
        yield config
        index += 1


def generate_rcga_configs(base_config, args):
    index = 1
    for schedule, mutation_prob, crossover_prob, sigma in itertools.product(POPULATION_GENERATION_PAIRS, RCGA_MUTATION_PROBS, RCGA_CROSSOVER_PROBS, RCGA_SIGMAS):
        experiment_name = (
            f"rcga_grid_{index:03d}"
            f"_pop{schedule['populacao_inicial']}"
            f"_gen{schedule['num_geracoes']}"
            f"_pmut{slugify(mutation_prob)}"
            f"_pcross{slugify(crossover_prob)}"
            f"_sigma{slugify(sigma)}"
        )
        config = common_config(base_config, "rcga", experiment_name)
        config.update(schedule)
        config["rcga_mutation_prob"] = mutation_prob
        config["rcga_crossover_prob"] = crossover_prob
        config["rcga_embedding_mutation_std"] = sigma
        config["prob_mutacao_embedding"] = mutation_prob
        config["prob_crossover_embedding"] = crossover_prob
        config["mutation_intensity_percent"] = sigma
        config["classifier_evaluation_budget_kind"] = "total_including_initial_population"
        apply_seed_override(config, args)
        yield config
        index += 1


def generate_cma_es_configs(base_config, args):
    index = 1
    for population_size, sigma in itertools.product(CMA_ES_POPULATION_SIZES, CMA_ES_SIGMAS):
        generations = math.ceil((CLASSIFIER_EVAL_BUDGET - 1) / population_size)
        experiment_name = f"cma_es_grid_{index:03d}_pop{population_size}_gen{generations}_sigma{slugify(sigma)}"
        config = common_config(base_config, "cma_es", experiment_name)
        config["cma_es_population_size"] = population_size
        config["cma_es_sigma"] = sigma
        config["num_geracoes"] = generations
        config["cma_es_classifier_evals"] = CLASSIFIER_EVAL_BUDGET
        config["classifier_evaluation_budget_kind"] = "total_including_initial_solution"
        apply_seed_override(config, args)
        yield config
        index += 1


def generate_random_search_configs(base_config, args):
    index = 1
    for sigma in RANDOM_SEARCH_SIGMAS:
        experiment_name = f"random_search_grid_{index:03d}_evals{CLASSIFIER_EVAL_BUDGET}_sigma{slugify(sigma)}"
        config = common_config(base_config, "random_search", experiment_name)
        config["random_search_classifier_evals"] = CLASSIFIER_EVAL_BUDGET
        config["random_search_batch_size"] = config.get("populacao_inicial", 100)
        config["random_search_sigma"] = sigma
        config["mutation_intensity_percent"] = sigma
        config["random_search_sampling_mode"] = "gaussian_embedding_sampling_from_initial_population"
        config["classifier_evaluation_budget_kind"] = "total_random_samples_including_initial"
        apply_seed_override(config, args)
        yield config
        index += 1


def generate_hill_configs(base_config, args):
    index = 1
    for sigma, neighbors in itertools.product(HILL_SIGMAS, HILL_NEIGHBORS):
        generations = math.ceil((CLASSIFIER_EVAL_BUDGET - 1) / neighbors)
        experiment_name = f"hill_grid_{index:03d}_sigma{slugify(sigma)}_neighbors{neighbors}_gen{generations}"
        config = common_config(base_config, "hill_climbing", experiment_name)
        config["hill_climbing_sigma"] = sigma
        config["mutation_intensity_percent"] = sigma
        config["hill_climbing_neighbors"] = neighbors
        config["hill_climbing_restart"] = False
        config["num_geracoes"] = generations
        config["max_evaluations"] = CLASSIFIER_EVAL_BUDGET
        config["classifier_evaluation_budget_kind"] = "total_neighbor_samples_including_initial"
        apply_seed_override(config, args)
        yield config
        index += 1


def config_generator_for_method(method, base_config, args, previous_best=None):
    if method == "ebie_stage1":
        return generate_ebie_stage1_configs(base_config, args)
    if method == "ebie_stage2":
        return generate_ebie_stage2_configs(base_config, args, previous_best)
    if method == "rcga":
        return generate_rcga_configs(base_config, args)
    if method == "cma_es":
        return generate_cma_es_configs(base_config, args)
    if method == "random_search":
        return generate_random_search_configs(base_config, args)
    if method == "hill_climbing":
        return generate_hill_configs(base_config, args)
    raise ValueError(f"Unsupported method: {method}")


def run_grid(repo_root, run_timestamp, stage_name, configs):
    generated_configs_dir = repo_root / "generated_configs" / "hyperparameter_selection" / run_timestamp / stage_name
    outputs_dir = repo_root / "outputs" / "hyperparameter_selection" / run_timestamp / stage_name
    manifest_path = outputs_dir / f"manifest_{stage_name}.json"
    manifest = []

    for config in configs:
        algorithm = config["algorithms"][0]
        experiment_name = config["experiment_name"]
        config_path = generated_configs_dir / f"{experiment_name}.yaml"
        output_file = outputs_dir / output_filename(algorithm, experiment_name)
        config["output_file"] = str(outputs_dir / "historico_completo.json")
        write_yaml(config_path, config)
        manifest.append(build_manifest_entry(run_timestamp, stage_name, config, config_path, output_file))
        write_json(manifest_path, manifest)

        if is_completed(output_file):
            continue

        subprocess.run([sys.executable, str(repo_root / "run_experiments.py"), "--config", str(config_path)], check=True, cwd=repo_root)

    write_json(manifest_path, manifest)
    return outputs_dir, manifest


def rank_outputs(outputs_dir):
    rows = []
    for path in sorted(outputs_dir.glob("historico_completo_*.json")):
        payload = read_json(path)
        config = payload["config"]
        summary = payload["summary"]
        rows.append({
            "experiment_name": payload["experiment_name"],
            "algorithm": payload["algorithm"],
            "file": str(path),
            "success_rate": summary["success_rate"],
            "evaluations_to_target_mean": summary["evaluations_to_target_mean"],
            "best_fitness_mean": summary["best_fitness_mean"],
            "best_fitness_std": summary["best_fitness_std"],
            "average_length_final_mean": summary["average_length_final_mean"],
            "lexical_diversity_mean": summary["lexical_diversity_mean"],
            "seed_stability": summary["seed_stability"],
            "num_runs": summary["num_runs"],
            **{key: value for key, value in config.items() if key in {
                "populacao_inicial",
                "num_geracoes",
                "prob_mutacao_embedding",
                "mutation_intensity_percent",
                "prob_crossover_embedding",
                "prob_add_random_token",
                "prob_remover_token",
                "rcga_mutation_prob",
                "rcga_crossover_prob",
                "rcga_embedding_mutation_std",
                "cma_es_population_size",
                "cma_es_sigma",
                "random_search_sigma",
                "hill_climbing_sigma",
                "hill_climbing_neighbors",
                "max_evaluations",
            }},
        })
    return sorted(rows, key=build_sort_key)


def save_ranking(outputs_dir, stage_name):
    ranking = rank_outputs(outputs_dir)
    write_json(outputs_dir / f"ranking_{stage_name}.json", ranking)
    if ranking:
        write_json(outputs_dir / f"best_{stage_name}.json", ranking[0])
    return ranking


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_config = load_config(repo_root / args.config)
    run_timestamp = args.resume_run or build_run_timestamp()
    summary = {
        "run_timestamp": run_timestamp,
        "classifier_evaluation_budget_per_run": CLASSIFIER_EVAL_BUDGET,
        "methods_requested": args.methods,
        "stages": {},
    }

    for method in [item for item in METHOD_ORDER if item in args.methods]:
        if method == "ebie":
            stage1_outputs, _ = run_grid(repo_root, run_timestamp, "ebie_stage1", config_generator_for_method("ebie_stage1", base_config, args))
            stage1_ranking = save_ranking(stage1_outputs, "ebie_stage1")
            if not stage1_ranking:
                raise RuntimeError("EBIE stage 1 did not produce ranking rows.")

            stage2_outputs, _ = run_grid(repo_root, run_timestamp, "ebie_stage2", config_generator_for_method("ebie_stage2", base_config, args, stage1_ranking[0]))
            stage2_ranking = save_ranking(stage2_outputs, "ebie_stage2")
            summary["stages"]["ebie_stage1"] = {"outputs_dir": str(stage1_outputs), "best": stage1_ranking[0]}
            summary["stages"]["ebie_stage2"] = {"outputs_dir": str(stage2_outputs), "best": stage2_ranking[0] if stage2_ranking else None}
            continue

        outputs_dir, _ = run_grid(repo_root, run_timestamp, method, config_generator_for_method(method, base_config, args))
        ranking = save_ranking(outputs_dir, method)
        summary["stages"][method] = {"outputs_dir": str(outputs_dir), "best": ranking[0] if ranking else None}

    summary_path = repo_root / "outputs" / "hyperparameter_selection" / run_timestamp / "hyperparameter_selection_summary.json"
    write_json(summary_path, summary)
    print(f"Hyperparameter selection summary saved to {summary_path}")


if __name__ == "__main__":
    main()
