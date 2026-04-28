import itertools
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml


STAGE2_EBIE_BUDGET = 10_000

# Fixed best Stage 1 configuration.
FIXED_POPULACAO_INICIAL = 50
FIXED_NUM_GERACOES = 200
FIXED_PROB_MUTACAO = 0.1
FIXED_MUTATION_INTENSITY = 0.10

STAGE2_PROB_CROSSOVER = [0.5, 0.8, 0.95]
STAGE2_PROB_ADD_TOKEN = [0.1, 0.3, 0.5]
STAGE2_PROB_REMOVE_TOKEN = [0.1, 0.3, 0.5]


def load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def slugify(value):
    return str(value).replace(".", "p")


def build_experiment_name(index, config):
    return (
        f"ebie_stage2_{index:03d}"
        f"_pop{config['populacao_inicial']}"
        f"_gen{config['num_geracoes']}"
        f"_pmut{slugify(config['prob_mutacao_embedding'])}"
        f"_mint{int(config['mutation_intensity_percent'] * 100)}"
        f"_pcross{slugify(config['prob_crossover_embedding'])}"
        f"_padd{slugify(config['prob_add_random_token'])}"
        f"_prem{slugify(config['prob_remover_token'])}"
    )


def generate_stage2_configs(base_config):
    combinations = itertools.product(
        STAGE2_PROB_CROSSOVER,
        STAGE2_PROB_ADD_TOKEN,
        STAGE2_PROB_REMOVE_TOKEN,
    )

    for index, (prob_crossover, prob_add, prob_remove) in enumerate(combinations, start=1):
        config = deepcopy(base_config)
        config["algorithms"] = ["genetic"]
        config["populacao_inicial"] = FIXED_POPULACAO_INICIAL
        config["num_geracoes"] = FIXED_NUM_GERACOES
        config["prob_mutacao_embedding"] = FIXED_PROB_MUTACAO
        config["mutation_intensity_percent"] = FIXED_MUTATION_INTENSITY
        config["prob_crossover_embedding"] = prob_crossover
        config["prob_add_random_token"] = prob_add
        config["prob_remover_token"] = prob_remove
        config["classifier_evaluation_budget"] = STAGE2_EBIE_BUDGET
        config["classifier_evaluation_budget_kind"] = "descendants_only"
        config["experiment_name"] = build_experiment_name(index, config)
        yield config


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

    generated_configs_dir = repo_root / "generated_configs" / "ebie_stage2" / run_timestamp
    outputs_dir = repo_root / "outputs" / "ebie_stage2" / run_timestamp
    manifest_path = outputs_dir / "manifest_ebie_stage2.json"
    manifest = []

    for config in generate_stage2_configs(base_config):
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
                    "prob_add_random_token": config["prob_add_random_token"],
                    "prob_remover_token": config["prob_remover_token"],
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
