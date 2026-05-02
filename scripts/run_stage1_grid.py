import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml


STAGE1_EBIE_BUDGET = 10_000
STAGE1_POPULATION_GENERATION_PAIRS = [
    {"populacao_inicial": 50, "num_geracoes": 200},
    {"populacao_inicial": 100, "num_geracoes": 100},
    {"populacao_inicial": 200, "num_geracoes": 50},
]
STAGE1_PROB_MUTACAO = [0.1, 0.2, 0.4]
STAGE1_STEP_SIZES = [0.05, 0.10, 0.20]
FIXED_PROB_CROSSOVER = 0.8

def load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def slugify(value):
    return str(value).replace(".", "p")


def build_experiment_name(index, config):
    return (
        f"ebie_stage1_{index:03d}"
        f"_pop{config['populacao_inicial']}"
        f"_gen{config['num_geracoes']}"
        f"_pmut{slugify(config['prob_mutacao_embedding'])}"
        f"_mint{int(config['mutation_intensity_percent'] * 100)}"
        f"_pcross{slugify(config['prob_crossover_embedding'])}"
    )


def generate_stage1_configs(base_config):
    index = 1
    for schedule in STAGE1_POPULATION_GENERATION_PAIRS:
        for prob_mutacao in STAGE1_PROB_MUTACAO:
            for step_size in STAGE1_STEP_SIZES:
                config = deepcopy(base_config)
                config["algorithms"] = ["genetic"]
                config["populacao_inicial"] = schedule["populacao_inicial"]
                config["num_geracoes"] = schedule["num_geracoes"]
                config["prob_mutacao_embedding"] = prob_mutacao
                config["mutation_intensity_percent"] = step_size
                config["prob_crossover_embedding"] = FIXED_PROB_CROSSOVER
                config["classifier_evaluation_budget"] = STAGE1_EBIE_BUDGET
                config["classifier_evaluation_budget_kind"] = "descendants_only"
                config["is_hyperparameter_selection"] = True
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


def main():
    repo_root = Path(__file__).resolve().parents[1]
    base_config_path = repo_root / "config.yaml"
    base_config = load_config(base_config_path)
    run_timestamp = build_run_timestamp()

    generated_configs_dir = repo_root / "generated_configs" / "ebie_stage1" / run_timestamp
    outputs_dir = repo_root / "outputs" / "ebie_stage1" / run_timestamp
    manifest_path = outputs_dir / "manifest_ebie_stage1.json"
    manifest = []

    for config in generate_stage1_configs(base_config):
        experiment_name = config["experiment_name"]
        config_path = generated_configs_dir / f"{experiment_name}.yaml"
        config["output_file"] = str(outputs_dir / "historico_completo.json")
        write_yaml(config_path, config)

        manifest.append(
            {
                "run_timestamp": run_timestamp,
                "experiment_name": experiment_name,
                "config_path": str(config_path),
                "output_file": config["output_file"],
                "parameters": {
                    "algorithm": "genetic",
                    "populacao_inicial": config["populacao_inicial"],
                    "num_geracoes": config["num_geracoes"],
                    "prob_mutacao_embedding": config["prob_mutacao_embedding"],
                    "mutation_intensity_percent": config["mutation_intensity_percent"],
                    "prob_crossover_embedding": config["prob_crossover_embedding"],
                    "classifier_evaluation_budget": config["classifier_evaluation_budget"],
                    "expected_descendant_evaluations": (
                        config["populacao_inicial"] * config["num_geracoes"]
                    ),
                    "expected_total_classifier_evaluations": (
                        config["populacao_inicial"] * (config["num_geracoes"] + 1)
                    ),
                },
            }
        )

        subprocess.run(
            [sys.executable, str(repo_root / "run_experiments.py"), "--config", str(config_path)],
            check=True,
            cwd=repo_root,
        )

    save_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
    
