import itertools
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml


HILL_BUDGET = 10_000
STAGE1_STEP_SIZES = [0.05, 0.10, 0.20]
STAGE1_NEIGHBORS = [1, 5, 10]
STAGE1_FIXED_RESTART = False
STAGE2_RESTART_OPTIONS = [False, True]


def load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def build_experiment_name(stage_name, index, config):
    return (
        f"{stage_name}_{index:03d}"
        f"_step{int(config['mutation_intensity_percent'] * 100)}"
        f"_neighbors{config['hill_climbing_neighbors']}"
        f"_gen{config['num_geracoes']}"
        f"_restart{int(bool(config['hill_climbing_restart']))}"
    )


def generate_stage1_configs(base_config):
    combinations = itertools.product(STAGE1_STEP_SIZES, STAGE1_NEIGHBORS)

    for index, (step_size, neighbors) in enumerate(combinations, start=1):
        config = deepcopy(base_config)
        config["algorithms"] = ["hill_climbing"]
        config["hill_climbing_neighbors"] = neighbors
        config["hill_climbing_restart"] = STAGE1_FIXED_RESTART
        config["mutation_intensity_percent"] = step_size
        config["num_geracoes"] = HILL_BUDGET // neighbors
        config["classifier_evaluation_budget"] = HILL_BUDGET
        config["classifier_evaluation_budget_kind"] = "neighbor_steps"
        config["is_hyperparameter_selection"] = True
        config["experiment_name"] = build_experiment_name("hill_stage1", index, config)
        yield config


def generate_stage2_configs(base_config, best_stage1):
    for index, restart in enumerate(STAGE2_RESTART_OPTIONS, start=1):
        config = deepcopy(base_config)
        config["algorithms"] = ["hill_climbing"]
        config["hill_climbing_neighbors"] = best_stage1["hill_climbing_neighbors"]
        config["hill_climbing_restart"] = restart
        config["mutation_intensity_percent"] = best_stage1["mutation_intensity_percent"]
        config["num_geracoes"] = best_stage1["num_geracoes"]
        config["classifier_evaluation_budget"] = HILL_BUDGET
        config["classifier_evaluation_budget_kind"] = "neighbor_steps_with_optional_restart"
        config["is_hyperparameter_selection"] = True
        config["experiment_name"] = build_experiment_name("hill_stage2", index, config)
        yield config


def write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def build_run_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_configs(repo_root, stage_name, run_timestamp, configs):
    generated_configs_dir = repo_root / "generated_configs" / stage_name / run_timestamp
    outputs_dir = repo_root / "outputs" / stage_name / run_timestamp
    manifest_path = outputs_dir / f"manifest_{stage_name}.json"
    manifest = []

    for config in configs:
        experiment_name = config["experiment_name"]
        config_path = generated_configs_dir / f"{experiment_name}.yaml"
        config["output_file"] = str(outputs_dir / "historico_completo.json")
        write_yaml(config_path, config)

        expected_total_evals = 1 + config["hill_climbing_neighbors"] * config["num_geracoes"]
        if config["hill_climbing_restart"]:
            expected_total_evals += config["num_geracoes"]

        manifest.append({"run_timestamp":run_timestamp, "stage_name":stage_name, "experiment_name":experiment_name, "config_path":str(config_path), "output_file":config["output_file"], "parameters":{"algorithm":"hill_climbing", "mutation_intensity_percent":config["mutation_intensity_percent"], "hill_climbing_neighbors":config["hill_climbing_neighbors"], "num_geracoes":config["num_geracoes"], "hill_climbing_restart":config["hill_climbing_restart"], "classifier_evaluation_budget":config["classifier_evaluation_budget"], "expected_neighbor_evaluations":(config["hill_climbing_neighbors"] *config["num_geracoes"]), "expected_total_classifier_evaluations":expected_total_evals,},})

        subprocess.run([sys.executable, str(repo_root / "run_experiments.py"), "--config", str(config_path)], check=True, cwd=repo_root,)

    save_json(manifest_path, manifest)
    return outputs_dir


def build_sort_key(row):
    success_rate = row["success_rate"]
    evaluations_to_target_mean = row["evaluations_to_target_mean"]
    best_fitness_mean = row["best_fitness_mean"]
    best_fitness_std = row["best_fitness_std"]

    return (
        -1 if success_rate is None else -success_rate,
        float("inf") if evaluations_to_target_mean is None else evaluations_to_target_mean,
        1 if best_fitness_mean is None else -best_fitness_mean,
        float("inf") if best_fitness_std is None else best_fitness_std,
        row["experiment_name"],
    )


def rank_outputs(outputs_dir):
    rows = []
    for path in sorted(outputs_dir.glob("historico_completo_hill_climbing_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        config = data["config"]
        summary = data["summary"]
        rows.append({"experiment_name":data["experiment_name"], "file":str(path), "mutation_intensity_percent":config["mutation_intensity_percent"], "hill_climbing_neighbors":config["hill_climbing_neighbors"], "hill_climbing_restart":config["hill_climbing_restart"], "num_geracoes":config["num_geracoes"], "success_rate":summary["success_rate"], "evaluations_to_target_mean":summary["evaluations_to_target_mean"], "best_fitness_mean":summary["best_fitness_mean"], "best_fitness_std":summary["best_fitness_std"], "seed_stability":summary["seed_stability"], "num_runs":summary["num_runs"],})

    return sorted(rows, key=build_sort_key)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    base_config_path = repo_root / "config.yaml"
    base_config = load_config(base_config_path)
    run_timestamp = build_run_timestamp()

    stage1_outputs_dir = run_configs(repo_root, "hill_stage1", run_timestamp, generate_stage1_configs(base_config),)
    stage1_ranking = rank_outputs(stage1_outputs_dir)
    best_stage1 = stage1_ranking[0]
    save_json(stage1_outputs_dir / "ranking_hill_stage1.json", stage1_ranking)
    save_json(stage1_outputs_dir / "best_hill_stage1.json", best_stage1)

    stage2_outputs_dir = run_configs(repo_root, "hill_stage2", run_timestamp, generate_stage2_configs(base_config, best_stage1),)
    stage2_ranking = rank_outputs(stage2_outputs_dir)
    save_json(stage2_outputs_dir / "ranking_hill_stage2.json", stage2_ranking)
    save_json(stage2_outputs_dir / "hill_two_stage_selection_summary.json", {"run_timestamp":run_timestamp, "stage1_best":best_stage1, "stage2_ranking":stage2_ranking, "stage2_best":stage2_ranking[0] if stage2_ranking else None,},)


if __name__ == "__main__":
    main()
