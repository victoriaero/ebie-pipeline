import argparse
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml


VANILLA_GA_BUDGET = 10_000
POPULATION_GENERATION_PAIRS = [
    {"populacao_inicial": 50, "num_geracoes": 200},
    {"populacao_inicial": 100, "num_geracoes": 100},
    {"populacao_inicial": 200, "num_geracoes": 50},
]
PROB_MUTACAO = [0.1, 0.2, 0.4]
PROB_CROSSOVER = [0.5, 0.8, 0.95]


def load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Run or resume the Vanilla GA grid.")
    parser.add_argument(
        "--resume-run",
        help="Existing Vanilla GA run timestamp to resume, for example 20260430_120000.",
    )
    return parser.parse_args()


def slugify(value):
    return str(value).replace(".", "p")


def build_experiment_name(index, config):
    return (
        f"vanilla_ga_grid_{index:03d}"
        f"_pop{config['populacao_inicial']}"
        f"_gen{config['num_geracoes']}"
        f"_pmut{slugify(config['prob_mutacao_embedding'])}"
        f"_pcross{slugify(config['prob_crossover_embedding'])}"
    )


def generate_vanilla_ga_configs(base_config):
    index = 1
    for schedule in POPULATION_GENERATION_PAIRS:
        for prob_mutacao in PROB_MUTACAO:
            for prob_crossover in PROB_CROSSOVER:
                config = deepcopy(base_config)
                config["algorithms"] = ["vanilla_ga"]
                config["populacao_inicial"] = schedule["populacao_inicial"]
                config["num_geracoes"] = schedule["num_geracoes"]
                config["prob_mutacao_embedding"] = prob_mutacao
                config["prob_crossover_embedding"] = prob_crossover
                config["ga_mutation_prob"] = prob_mutacao
                config["ga_crossover_prob"] = prob_crossover
                config["ga_embedding_mutation_std"] = config.get(
                    "ga_embedding_mutation_std",
                    config.get("mutation_intensity_percent", 0.1),
                )
                config["classifier_evaluation_budget"] = VANILLA_GA_BUDGET
                config["classifier_evaluation_budget_kind"] = "descendants_only"
                config["is_hyperparameter_selection"] = True
                config["save_run_history"] = True
                config["experiment_name"] = build_experiment_name(index, config)
                yield config
                index += 1


def write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)


def save_manifest(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def build_run_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_output_filename(experiment_name):
    return f"historico_completo_vanilla_ga_current_decoder_{experiment_name}.json"


def load_existing_manifest(manifest_path):
    if not manifest_path.exists():
        return []

    with manifest_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def detect_experiment_status(outputs_dir, experiment_name):
    output_path = outputs_dir / build_output_filename(experiment_name)
    if not output_path.exists():
        return {"status": "missing", "completed_runs": 0, "total_runs": None}

    with output_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    progress = payload.get("progress", {})
    return {
        "status": progress.get("status", "unknown"),
        "completed_runs": progress.get("completed_runs", 0),
        "total_runs": progress.get("total_runs"),
    }


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_config_path = repo_root / "config.yaml"
    base_config = load_config(base_config_path)
    run_timestamp = args.resume_run or build_run_timestamp()

    generated_configs_dir = repo_root / "generated_configs" / "vanilla_ga_grid" / run_timestamp
    outputs_dir = repo_root / "outputs" / "vanilla_ga_grid" / run_timestamp
    manifest_path = outputs_dir / "manifest_vanilla_ga_grid.json"
    manifest = load_existing_manifest(manifest_path)
    manifest_by_experiment = {
        item["experiment_name"]: item for item in manifest if "experiment_name" in item
    }

    for config in generate_vanilla_ga_configs(base_config):
        experiment_name = config["experiment_name"]
        config_path = generated_configs_dir / f"{experiment_name}.yaml"
        config["output_file"] = str(outputs_dir / "historico_completo.json")
        write_yaml(config_path, config)

        manifest_by_experiment[experiment_name] = {
            "run_timestamp": run_timestamp,
            "experiment_name": experiment_name,
            "config_path": str(config_path),
            "output_file": config["output_file"],
            "parameters": {
                "algorithm": "vanilla_ga",
                "populacao_inicial": config["populacao_inicial"],
                "num_geracoes": config["num_geracoes"],
                "prob_mutacao_embedding": config["prob_mutacao_embedding"],
                "prob_crossover_embedding": config["prob_crossover_embedding"],
                "ga_mutation_prob": config["ga_mutation_prob"],
                "ga_crossover_prob": config["ga_crossover_prob"],
                "ga_embedding_mutation_std": config["ga_embedding_mutation_std"],
                "tournament_size": config["tournament_size"],
                "classifier_evaluation_budget": config["classifier_evaluation_budget"],
                "expected_descendant_evaluations": (
                    config["populacao_inicial"] * config["num_geracoes"]
                ),
                "expected_total_classifier_evaluations": (
                    config["populacao_inicial"] * (config["num_geracoes"] + 1)
                ),
            },
        }
        save_manifest(
            manifest_path,
            [manifest_by_experiment[key] for key in sorted(manifest_by_experiment)]
        )

        experiment_status = detect_experiment_status(outputs_dir, experiment_name)
        if experiment_status["status"] == "completed":
            continue

        subprocess.run(
            [sys.executable, str(repo_root / "run_experiments.py"), "--config", str(config_path)],
            check=True,
            cwd=repo_root,
        )

    save_manifest(
        manifest_path,
        [manifest_by_experiment[key] for key in sorted(manifest_by_experiment)]
    )


if __name__ == "__main__":
    main()
