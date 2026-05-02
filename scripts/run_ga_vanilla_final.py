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
FINAL_GA_PROB_MUTACAO = 0.4
FINAL_GA_PROB_CROSSOVER = 0.5
FINAL_GA_EMBEDDING_MUTATION_STD = 0.1
FINAL_TOURNAMENT_SIZE = 2
FINAL_NUM_RUNS = 30
FINAL_SEED_START = 0


def load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Run or resume the final Vanilla GA experiment.")
    parser.add_argument(
        "--resume-run",
        help="Existing final Vanilla GA run timestamp to resume, for example 20260501_120000.",
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
        "vanilla_ga_final"
        f"_pop{config['populacao_inicial']}"
        f"_gen{config['num_geracoes']}"
        f"_pmut{slugify(config['ga_mutation_prob'])}"
        f"_pcross{slugify(config['ga_crossover_prob'])}"
        f"_mstd{slugify(config['ga_embedding_mutation_std'])}"
        f"_tourn{config['tournament_size']}"
        f"_runs{len(config['experiment_seeds'])}"
    )


def build_final_config(base_config, seeds):
    config = deepcopy(base_config)

    config["algorithms"] = ["vanilla_ga"]

    config["populacao_inicial"] = FINAL_POPULACAO_INICIAL
    config["num_geracoes"] = FINAL_NUM_GERACOES

    config["ga_mutation_prob"] = FINAL_GA_PROB_MUTACAO
    config["ga_crossover_prob"] = FINAL_GA_PROB_CROSSOVER
    config["ga_embedding_mutation_std"] = FINAL_GA_EMBEDDING_MUTATION_STD
    config["tournament_size"] = FINAL_TOURNAMENT_SIZE

    # Mantém compatibilidade com scripts antigos e sumarizadores que ainda usam os nomes do EBIE.
    config["prob_mutacao_embedding"] = FINAL_GA_PROB_MUTACAO
    config["prob_crossover_embedding"] = FINAL_GA_PROB_CROSSOVER
    config["mutation_intensity_percent"] = FINAL_GA_EMBEDDING_MUTATION_STD

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
    return f"historico_completo_vanilla_ga_current_decoder_{experiment_name}.json"


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

    generated_configs_dir = (
        repo_root / "generated_configs" / "vanilla_ga_final" / run_timestamp
    )
    outputs_dir = repo_root / "outputs" / "vanilla_ga_final" / run_timestamp
    manifest_path = outputs_dir / "manifest_vanilla_ga_final.json"

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
            "algorithm": "vanilla_ga",
            "populacao_inicial": config["populacao_inicial"],
            "num_geracoes": config["num_geracoes"],
            "ga_mutation_prob": config["ga_mutation_prob"],
            "ga_crossover_prob": config["ga_crossover_prob"],
            "ga_embedding_mutation_std": config["ga_embedding_mutation_std"],
            "tournament_size": config["tournament_size"],
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
                len(seeds)
                * config["populacao_inicial"]
                * (config["num_geracoes"] + 1)
            ),
        },
    }

    save_manifest(manifest_path, manifest)

    experiment_status = detect_experiment_status(outputs_dir, experiment_name)
    if experiment_status["status"] == "completed":
        return

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_experiments.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        cwd=repo_root,
    )

    save_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()