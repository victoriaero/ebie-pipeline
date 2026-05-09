import argparse
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml


FINAL_POPULACAO_INICIAL = 50
FINAL_NUM_GERACOES_REFERENCE = 400
FINAL_HILL_NEIGHBORS = 1
FINAL_HILL_STEP_SIZE = 0.2
FINAL_HILL_RESTART = True
FINAL_MAX_EVALUATIONS = FINAL_POPULACAO_INICIAL * (
    FINAL_NUM_GERACOES_REFERENCE + 1
)
FINAL_NUM_RUNS = 10
FINAL_SEED_START = 10


def load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Run or resume the final Hill Climbing experiment.")
    parser.add_argument("--resume-run", help="Existing final Hill Climbing run timestamp to resume, for example 20260501_120000.",)
    parser.add_argument("--seed-start", type=int, default=FINAL_SEED_START, help="First seed in the contiguous seed range.",)
    parser.add_argument("--num-runs", type=int, default=FINAL_NUM_RUNS, help="Number of seeds/runs to execute.",)
    parser.add_argument("--max-evaluations", type=int, default=FINAL_MAX_EVALUATIONS, help="Total classifier evaluations per seed, including initial and restart evaluations.",)
    return parser.parse_args()


def build_experiment_name(config):
    return (
        "hill_final"
        f"_neighbors{config['hill_climbing_neighbors']}"
        f"_step{int(config['mutation_intensity_percent'] * 100)}"
        f"_restart{int(bool(config['hill_climbing_restart']))}"
        f"_evals{config['max_evaluations']}"
        f"_runs{len(config['experiment_seeds'])}"
    )


def build_final_config(base_config, seeds, max_evaluations):
    config = deepcopy(base_config)
    config["algorithms"] = ["hill_climbing"]
    config["populacao_inicial"] = FINAL_POPULACAO_INICIAL
    config["num_geracoes"] = max_evaluations - 1
    config["num_geracoes_reference"] = FINAL_NUM_GERACOES_REFERENCE
    config["hill_climbing_neighbors"] = FINAL_HILL_NEIGHBORS
    config["hill_climbing_restart"] = FINAL_HILL_RESTART
    config["mutation_intensity_percent"] = FINAL_HILL_STEP_SIZE
    config["max_evaluations"] = max_evaluations
    config["experiment_seeds"] = seeds
    config["num_execucoes"] = len(seeds)
    config["classifier_evaluation_budget"] = max_evaluations
    config["total_classifier_evaluation_budget"] = max_evaluations
    config["classifier_evaluation_budget_kind"] = "total_with_optional_restart"
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
    return f"historico_completo_hill_climbing_current_decoder_{experiment_name}.json"


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

    generated_configs_dir = repo_root / "generated_configs" / "hill_final" / run_timestamp
    outputs_dir = repo_root / "outputs" / "hill_final" / run_timestamp
    manifest_path = outputs_dir / "manifest_hill_final.json"

    config = build_final_config(base_config, seeds, args.max_evaluations)
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
            "algorithm": "hill_climbing",
            "populacao_inicial": config["populacao_inicial"],
            "num_geracoes": config["num_geracoes"],
            "num_geracoes_reference": config["num_geracoes_reference"],
            "hill_climbing_neighbors": config["hill_climbing_neighbors"],
            "mutation_intensity_percent": config["mutation_intensity_percent"],
            "hill_climbing_restart": config["hill_climbing_restart"],
            "max_evaluations": config["max_evaluations"],
            "num_runs": len(seeds),
            "seeds": seeds,
            "seed_strategy": "contiguous_integer_seeds",
            "classifier_evaluation_budget": config["classifier_evaluation_budget"],
            "total_classifier_evaluation_budget": config["total_classifier_evaluation_budget"],
            "classifier_evaluation_budget_kind": config["classifier_evaluation_budget_kind"],
            "expected_total_classifier_evaluations_per_seed": config["max_evaluations"],
            "expected_total_classifier_evaluations_all_seeds": (
                len(seeds) * config["max_evaluations"]
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
