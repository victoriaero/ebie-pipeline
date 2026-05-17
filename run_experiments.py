import argparse
import json
from datetime import datetime
from pathlib import Path
import random
import yaml

from src.cma_es import cma_es
from src.ebie import algoritmo_genetico
from src.experiment_utils import configure_deterministic_backend, set_global_seed
from src.rcga import rcga
from src.hill import hill_climbing
from src.initialization import generate_initial_population
from src.metrics import summarize_evaluation_metrics, summarize_generation_metrics, summarize_run, summarize_runs
from src.random_search import random_search
from src.resources import load_resources


def load_config(config_path="config.yaml"):
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    config["_base_dir"] = config_file.parent.resolve()
    return config


def build_output_path(config, algorithm_name, decoder_name):
    output_path = (config["_base_dir"] / config["output_file"]).resolve()
    experiment_name = config.get("experiment_name")
    if experiment_name:
        suffix = (f"{output_path.stem}_{algorithm_name}_{decoder_name}_{experiment_name}"
            f"{output_path.suffix}")
    else:
        suffix = f"{output_path.stem}_{algorithm_name}_{decoder_name}{output_path.suffix}"
    return str(output_path.with_name(suffix))


def parse_args():
    parser = argparse.ArgumentParser(description="Run EBIE experiments from a YAML config.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML configuration file.")
    return parser.parse_args()


def load_json_payload(path):
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def load_existing_payload(output_file):
    output_path = Path(output_file)
    candidates = [load_json_payload(output_path), load_json_payload(output_path.with_name(f"{output_path.name}.tmp"))]
    candidates = [payload for payload in candidates if payload is not None]
    if not candidates:
        return None

    return max(candidates, key=lambda payload: payload.get("progress", {}).get("completed_runs", 0))


def serialize_run_config(run_config):
    return {
        key: value
        for key, value in run_config.items()
        if not key.startswith("_")
    }


def get_decoder_configs(config):
    if not config["run_decoder_ablation"]:
        return [{
                "name": config["decoder_config_name"],
                "decoder_family": config["decoder_family"],
                "decoder_strategy": config["decoder_strategy"],
                "decoder_top_k": config["decoder_top_k"],
                "decoder_similarity": config["decoder_similarity"],
                "filter_special_tokens": config["filter_special_tokens"],
            }]

    decoder_configs = []
    if config["decoder_include_baseline"]:
        decoder_configs.append({"name":"lm_head_weighted_sampling_top50", "decoder_family":"lm_head", "decoder_strategy":"weighted_sampling", "decoder_top_k":50, "decoder_similarity":config["decoder_similarity"], "filter_special_tokens":True,})

    for strategy in config["decoder_ablation_strategies"]:
        for top_k in config["decoder_ablation_top_ks"]:
            decoder_configs.append({"name":f"embedding_similarity_{strategy}_top{top_k}", "decoder_family":"embedding_similarity", "decoder_strategy":strategy, "decoder_top_k":top_k, "decoder_similarity":config["decoder_similarity"], "filter_special_tokens":True,})

    return decoder_configs


def get_experiment_seeds(config):
    if config.get("experiment_seeds") is not None:
        return config["experiment_seeds"]
    return list(range(config["num_execucoes"]))


def execute_single_run(algorithm_name, resources, config):
    if algorithm_name == "random_search":
        return random_search(resources, config, None)

    initial_population = generate_initial_population(resources, config, config["populacao_inicial"])
    if not initial_population:
        return {}

    if algorithm_name == "genetic":
        return algoritmo_genetico(resources, config, initial_population)

    if algorithm_name == "rcga":
        return rcga(resources, config, initial_population)

    if algorithm_name == "hill_climbing":
        return hill_climbing(resources, config, random.choice(initial_population))

    if algorithm_name == "cma_es":
        return cma_es(resources, config, random.choice(initial_population))

    raise ValueError(f"Unsupported algorithm: {algorithm_name}")


def save_payload(output_file, payload):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, separators=(",", ":"))
    temp_path.replace(output_path)


def build_payload(run_config, algorithm_name, decoder_config, runs_payload, run_summaries, seeds, status):
    return {
        "algorithm": algorithm_name,
        "decoder": decoder_config,
        "initialization_mode": run_config["initialization_mode"],
        "experiment_name": run_config.get("experiment_name"),
        "config": serialize_run_config(run_config),
        "success_target_score": run_config["success_target_score"],
        "progress": {
            "status": status,
            "completed_runs": len(runs_payload),
            "total_runs": len(seeds),
            "completed_seeds": [run["seed"] for run in runs_payload],
            "last_updated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "runs": runs_payload,
        "summary": summarize_runs(run_summaries),
    }


def restore_previous_progress(existing_payload, seeds):
    if not existing_payload:
        return [], []

    runs_payload = existing_payload.get("runs", [])
    completed_by_seed = {}
    for run in runs_payload:
        seed = run.get("seed")
        if seed in seeds and seed not in completed_by_seed:
            completed_by_seed[seed] = run

    ordered_runs = [completed_by_seed[seed] for seed in seeds if seed in completed_by_seed]
    run_summaries = [run["metrics"] for run in ordered_runs]
    return ordered_runs, run_summaries


def build_run_payload(run_config, seed, history, generation_metrics, evaluation_metrics, run_summary):
    run_payload = {
        "seed": seed,
        "config": serialize_run_config(run_config),
        "metrics": run_summary,
    }
    if generation_metrics is not None:
        run_payload["generation_metrics"] = generation_metrics
    if evaluation_metrics is not None:
        run_payload["evaluation_metrics"] = evaluation_metrics
    if run_config.get("save_run_history", True):
        run_payload["history"] = history
    return run_payload


def main():
    args = parse_args()
    config = load_config(args.config)
    configure_deterministic_backend()
    resources = load_resources(config)
    decoder_configs = get_decoder_configs(config)
    seeds = get_experiment_seeds(config)

    for algorithm_name in config["algorithms"]:
        for decoder_config in decoder_configs:
            run_config = dict(config)
            run_config.update(decoder_config)
            output_file = build_output_path(run_config, algorithm_name, decoder_config["name"])
            existing_payload = load_existing_payload(output_file)
            runs_payload, run_summaries = restore_previous_progress(existing_payload, seeds)
            completed_seeds = {run["seed"] for run in runs_payload}

            if len(completed_seeds) == len(seeds):
                payload = build_payload(run_config, algorithm_name, decoder_config, runs_payload, run_summaries, seeds, status="completed")
                save_payload(output_file, payload)
                continue

            for seed in seeds:
                if seed in completed_seeds:
                    continue

                set_global_seed(seed)
                seed_run_config = dict(run_config)
                seed_run_config["_current_seed"] = seed
                seed_run_config["_algorithm_name"] = algorithm_name
                seed_run_config["_resolved_output_file"] = output_file
                history = execute_single_run(algorithm_name, resources, seed_run_config)
                run_summary = summarize_run(history, seed_run_config["success_target_score"])
                is_hyperparameter_selection = bool(seed_run_config.get("is_hyperparameter_selection", False))
                generation_metrics = None
                evaluation_metrics = None
                if is_hyperparameter_selection:
                    generation_metrics = summarize_generation_metrics(history, seed_run_config["success_target_score"])
                    evaluation_metrics = summarize_evaluation_metrics(history, seed_run_config["success_target_score"])
                runs_payload.append(build_run_payload(seed_run_config, seed, history, generation_metrics, evaluation_metrics, run_summary))
                run_summaries.append(run_summary)
                completed_seeds.add(seed)
                save_payload(output_file, build_payload(run_config, algorithm_name, decoder_config, runs_payload, run_summaries, seeds, status="running"))

            payload = build_payload(run_config, algorithm_name, decoder_config, runs_payload, run_summaries, seeds, status="completed",)
            save_payload(output_file, payload)


if __name__ == "__main__":
    main()
