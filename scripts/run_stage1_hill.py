import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml


STAGE1_STEP_SIZES = [0.05, 0.10, 0.20]
FIXED_NEIGHBORS = 5
FIXED_RESTART = False


def load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def build_experiment_name(index, config):
    return (
        f"hill_stage1_{index:03d}"
        f"_step{int(config['mutation_intensity_percent'] * 100)}"
        f"_neighbors{config['hill_climbing_neighbors']}"
        f"_restart{int(bool(config['hill_climbing_restart']))}"
    )


def generate_stage1_configs(base_config):
    for index, step_size in enumerate(STAGE1_STEP_SIZES, start=1):
        config = deepcopy(base_config)
        config["algorithms"] = ["hill_climbing"]
        config["hill_climbing_neighbors"] = FIXED_NEIGHBORS
        config["hill_climbing_restart"] = FIXED_RESTART
        config["mutation_intensity_percent"] = step_size
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

    generated_configs_dir = repo_root / "generated_configs" / "hill_stage1" / run_timestamp
    outputs_dir = repo_root / "outputs" / "hill_stage1" / run_timestamp
    manifest_path = outputs_dir / "manifest_hill_stage1.json"
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
                    "algorithm": "hill_climbing",
                    "mutation_intensity_percent": config["mutation_intensity_percent"],
                    "hill_climbing_neighbors": config["hill_climbing_neighbors"],
                    "hill_climbing_restart": config["hill_climbing_restart"],
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
