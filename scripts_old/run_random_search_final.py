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
FINAL_RANDOM_SEARCH_CLASSIFIER_EVALS = FINAL_POPULACAO_INICIAL * (
    FINAL_NUM_GERACOES + 1
)
FINAL_RANDOM_SEARCH_BATCH_SIZE = FINAL_POPULACAO_INICIAL
FINAL_RANDOM_SEARCH_SIGMA = 0.1
FINAL_NUM_RUNS = 30
FINAL_SEED_START = 0


def load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Run or resume the final Random Search baseline.")
    parser.add_argument("--resume-run", help="Existing final Random Search run timestamp to resume, for example 20260501_120000.",)
    parser.add_argument("--seed-start", type=int, default=FINAL_SEED_START, help="First seed in the contiguous seed range.",)
    parser.add_argument("--num-runs", type=int, default=FINAL_NUM_RUNS, help="Number of seeds/runs to execute.",)
    parser.add_argument("--random-search-classifier-evals", type=int, default=FINAL_RANDOM_SEARCH_CLASSIFIER_EVALS, help="Total classifier evaluations per seed, including the initial baseline phrase.",)
    parser.add_argument("--random-search-batch-size", type=int, default=FINAL_RANDOM_SEARCH_BATCH_SIZE, help="Number of random candidates sampled per random-search generation/log batch.",)
    return parser.parse_args()


def build_experiment_name(config):
    return (
        "random_search_final"
        f"_evals{config['random_search_classifier_evals']}"
        f"_batch{config['random_search_batch_size']}"
        f"_sigma{str(config['random_search_sigma']).replace('.', 'p')}"
        f"_runs{len(config['experiment_seeds'])}"
    )


def build_final_config(base_config, seeds, random_search_classifier_evals, random_search_batch_size):
    config = deepcopy(base_config)
    config["algorithms"] = ["random_search"]
    config["populacao_inicial"] = FINAL_POPULACAO_INICIAL
    config["num_geracoes"] = FINAL_NUM_GERACOES
    for key in ("prob_mutacao_embedding", "prob_add_random_token", "prob_remover_token",):
        config.pop(key, None)
    config["random_search_sigma"] = FINAL_RANDOM_SEARCH_SIGMA
    config["mutation_intensity_percent"] = FINAL_RANDOM_SEARCH_SIGMA
    config["random_search_classifier_evals"] = random_search_classifier_evals
    config["random_search_batch_size"] = random_search_batch_size
    config["random_search_sampling_mode"] = "gaussian_embedding_sampling_from_initial_population"
    config["experiment_seeds"] = seeds
    config["num_execucoes"] = len(seeds)
    config["classifier_evaluation_budget"] = random_search_classifier_evals
    config["total_classifier_evaluation_budget"] = random_search_classifier_evals
    config["classifier_evaluation_budget_kind"] = "total_random_samples_including_initial"
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
    return f"historico_completo_random_search_current_decoder_{experiment_name}.json"


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

    generated_configs_dir = repo_root / "generated_configs" / "random_search_final" / run_timestamp
    outputs_dir = repo_root / "outputs" / "random_search_final" / run_timestamp
    manifest_path = outputs_dir / "manifest_random_search_final.json"

    config = build_final_config(base_config, seeds, args.random_search_classifier_evals, args.random_search_batch_size,)
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
            "algorithm": "random_search",
            "populacao_inicial": config["populacao_inicial"],
            "num_geracoes_reference": config["num_geracoes"],
            "random_search_sampling_mode": config["random_search_sampling_mode"],
            "random_search_sigma": config["random_search_sigma"],
            "random_search_classifier_evals": config["random_search_classifier_evals"],
            "random_search_batch_size": config["random_search_batch_size"],
            "num_runs": len(seeds),
            "seeds": seeds,
            "seed_strategy": "contiguous_integer_seeds",
            "classifier_evaluation_budget": config["classifier_evaluation_budget"],
            "total_classifier_evaluation_budget": config["total_classifier_evaluation_budget"],
            "classifier_evaluation_budget_kind": config["classifier_evaluation_budget_kind"],
            "expected_total_classifier_evaluations_per_seed": (
                config["random_search_classifier_evals"]
            ),
            "expected_random_candidates_per_seed": (
                config["random_search_classifier_evals"] - 1
            ),
            "expected_total_classifier_evaluations_all_seeds": (
                len(seeds) * config["random_search_classifier_evals"]
            ),
        },
    }
    save_manifest(manifest_path, manifest)

    experiment_status = detect_experiment_status(outputs_dir, experiment_name)
    if experiment_status["status"] == "completed":
        return

    subprocess.run([sys.executable, str(repo_root / "run_experiments.py"), "--config", str(config_path)], check=True, cwd=repo_root,)

    save_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
