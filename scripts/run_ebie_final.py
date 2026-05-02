import argparse
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml


FINAL_POPULACAO_INICIAL = 50
FINAL_NUM_GERACOES = 400
FINAL_PROB_MUTACAO = 0.1
FINAL_MUTATION_INTENSITY = 0.1
FINAL_PROB_CROSSOVER = 0.8
FINAL_PROB_ADD_TOKEN = 0.3
FINAL_PROB_REMOVE_TOKEN = 0.1
FINAL_NUM_RUNS = 30
FINAL_SEED_START = 0


def load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Run or resume the final EBIE experiment.")
    parser.add_argument(
        "--resume-run",
        help="Existing final EBIE run timestamp to resume, for example 20260501_120000.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=FINAL_SEED_START,
        help="First seed in the contiguous seed range.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=FINAL_NUM_RUNS,
        help="Number of seeds/runs to execute.",
    )
    return parser.parse_args()


def slugify(value):
    return str(value).replace(".", "p")


def build_experiment_name(config):
    return (
        "ebie_final"
        f"_pop{config['populacao_inicial']}"
        f"_gen{config['num_geracoes']}"
        f"_pmut{slugify(config['prob_mutacao_embedding'])}"
        f"_mint{int(config['mutation_intensity_percent'] * 100)}"
        f"_pcross{slugify(config['prob_crossover_embedding'])}"
        f"_padd{slugify(config['prob_add_random_token'])}"
        f"_prem{slugify(config['prob_remover_token'])}"
        f"_runs{len(config['experiment_seeds'])}"
    )


def build_final_config(base_config, seeds):
    config = deepcopy(base_config)
    config["algorithms"] = ["genetic"]
    config["populacao_inicial"] = FINAL_POPULACAO_INICIAL
    config["num_geracoes"] = FINAL_NUM_GERACOES
    config["prob_mutacao_embedding"] = FINAL_PROB_MUTACAO
    config["mutation_intensity_percent"] = FINAL_MUTATION_INTENSITY
    config["prob_crossover_embedding"] = FINAL_PROB_CROSSOVER
    config["prob_add_random_token"] = FINAL_PROB_ADD_TOKEN
    config["prob_remover_token"] = FINAL_PROB_REMOVE_TOKEN
    config["experiment_seeds"] = seeds
    config["num_execucoes"] = len(seeds)
    config["classifier_evaluation_budget"] = (
        FINAL_POPULACAO_INICIAL * FINAL_NUM_GERACOES
    )
    config["total_classifier_evaluation_budget"] = (
        FINAL_POPULACAO_INICIAL * (FINAL_NUM_GERACOES + 1)
    )
    config["classifier_evaluation_budget_kind"] = "descendants_only"
    config["save_run_history"] = False
    config["experiment_name"] = build_experiment_name(config)
    return config


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
    return f"historico_completo_genetic_current_decoder_{experiment_name}.json"


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
    seeds = list(range(args.seed_start, args.seed_start + args.num_runs))

    generated_configs_dir = repo_root / "generated_configs" / "ebie_final" / run_timestamp
    outputs_dir = repo_root / "outputs" / "ebie_final" / run_timestamp
    manifest_path = outputs_dir / "manifest_ebie_final.json"

    config = build_final_config(base_config, seeds)
    experiment_name = config["experiment_name"]
    config_path = generated_configs_dir / f"{experiment_name}.yaml"
    config["output_file"] = str(outputs_dir / "historico_completo.json")
    write_yaml(config_path, config)

    manifest = {
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
            "num_runs": len(seeds),
            "seeds": seeds,
            "seed_strategy": "contiguous_integer_seeds",
            "classifier_evaluation_budget": config["classifier_evaluation_budget"],
            "total_classifier_evaluation_budget": config["total_classifier_evaluation_budget"],
            "classifier_evaluation_budget_kind": config["classifier_evaluation_budget_kind"],
            "expected_descendant_evaluations_per_seed": (
                config["populacao_inicial"] * config["num_geracoes"]
            ),
            "expected_total_classifier_evaluations_per_seed": (
                config["populacao_inicial"] * (config["num_geracoes"] + 1)
            ),
            "expected_total_classifier_evaluations_all_seeds": (
                len(seeds) * config["populacao_inicial"] * (config["num_geracoes"] + 1)
            ),
        },
    }
    save_manifest(manifest_path, manifest)

    experiment_status = detect_experiment_status(outputs_dir, experiment_name)
    if experiment_status["status"] == "completed":
        return

    subprocess.run(
        [sys.executable, str(repo_root / "run_experiments.py"), "--config", str(config_path)],
        check=True,
        cwd=repo_root,
    )

    save_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
