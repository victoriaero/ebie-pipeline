from pathlib import Path

import json
import yaml

from src.cma_es import cma_es
from src.ebie import algoritmo_genetico
from src.experiment_utils import set_global_seed
from src.hill import hill_climbing
from src.initialization import generate_initial_population
from src.metrics import summarize_run, summarize_runs
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
    suffix = f"{output_path.stem}_{algorithm_name}_{decoder_name}{output_path.suffix}"
    return str(output_path.with_name(suffix))


def get_decoder_configs(config):
    if not config["run_decoder_ablation"]:
        return [
            {
                "name": config["decoder_config_name"],
                "decoder_family": config["decoder_family"],
                "decoder_strategy": config["decoder_strategy"],
                "decoder_top_k": config["decoder_top_k"],
                "decoder_similarity": config["decoder_similarity"],
                "filter_special_tokens": config["filter_special_tokens"],
            }
        ]

    decoder_configs = []
    if config["decoder_include_baseline"]:
        decoder_configs.append(
            {
                "name": "lm_head_weighted_sampling_top50",
                "decoder_family": "lm_head",
                "decoder_strategy": "weighted_sampling",
                "decoder_top_k": 50,
                "decoder_similarity": config["decoder_similarity"],
                "filter_special_tokens": True,
            }
        )

    for strategy in config["decoder_ablation_strategies"]:
        for top_k in config["decoder_ablation_top_ks"]:
            decoder_configs.append(
                {
                    "name": f"embedding_similarity_{strategy}_top{top_k}",
                    "decoder_family": "embedding_similarity",
                    "decoder_strategy": strategy,
                    "decoder_top_k": top_k,
                    "decoder_similarity": config["decoder_similarity"],
                    "filter_special_tokens": True,
                }
            )

    return decoder_configs


def get_experiment_seeds(config):
    if config.get("experiment_seeds") is not None:
        return config["experiment_seeds"]
    return list(range(config["num_execucoes"]))


def execute_single_run(algorithm_name, resources, config):
    initial_population = generate_initial_population(resources, config, config["populacao_inicial"])
    if not initial_population:
        return {}

    if algorithm_name == "genetic":
        return algoritmo_genetico(resources, config, initial_population)

    if algorithm_name == "hill_climbing":
        return hill_climbing(resources, config, random.choice(initial_population))

    if algorithm_name == "cma_es":
        return cma_es(resources, config, random.choice(initial_population))

    if algorithm_name == "random_search":
        return random_search(resources, config, random.choice(initial_population))

    raise ValueError(f"Algoritmo não suportado: {algorithm_name}")


def save_payload(output_file, payload):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main():
    config = load_config()
    resources = load_resources(config)
    decoder_configs = get_decoder_configs(config)
    seeds = get_experiment_seeds(config)

    for algorithm_name in config["algorithms"]:
        for decoder_config in decoder_configs:
            run_config = dict(config)
            run_config.update(decoder_config)
            output_file = build_output_path(run_config, algorithm_name, decoder_config["name"])
            runs_payload = []
            run_summaries = []

            for seed in seeds:
                set_global_seed(seed)
                history = execute_single_run(algorithm_name, resources, run_config)
                run_summary = summarize_run(history, run_config["success_target_score"])
                runs_payload.append(
                    {
                        "seed": seed,
                        "history": history,
                        "metrics": run_summary,
                    }
                )
                run_summaries.append(run_summary)

            payload = {
                "algorithm": algorithm_name,
                "decoder": decoder_config,
                "initialization_mode": run_config["initialization_mode"],
                "success_target_score": run_config["success_target_score"],
                "runs": runs_payload,
                "summary": summarize_runs(run_summaries),
            }
            save_payload(output_file, payload)


if __name__ == "__main__":
    main()
