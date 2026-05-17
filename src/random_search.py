import random

import torch

from src.ebie import _sentence_embeddings
from src.initialization import generate_initial_population
from src.metrics import (
    RunLogger,
    build_initial_operator_records,
    evaluate_and_log_embeddings,
    evaluate_and_log_texts_with_embeddings,
)


def _get_random_search_sigma(config):
    return config.get(
        "random_search_sigma",
        config.get("random_search_embedding_std", config.get("mutation_intensity_percent", 0.1)),
    )


def _initial_population_embeddings(resources, initial_population):
    return [
        _sentence_embeddings(resources, text)[0][0]
        for text in initial_population
    ]


def build_random_sampling_operator_records(base_details):
    records = []
    for base_detail in base_details:
        records.append(
            {
                "parent_ids": [base_detail["candidate_id"]],
                "parent1_id": base_detail["candidate_id"],
                "parent2_id": None,
                "operator_used": "random_embedding_sampling",
                "mutation_type": "gaussian_noise",
                "crossover_type": None,
                "mutation_applied": True,
                "crossover_applied": False,
                "add_token_applied": False,
                "remove_token_applied": False,
                "num_tokens_added": 0,
                "num_tokens_removed": 0,
            }
        )
    return records


def _candidate_info(resources, candidate_detail, base_detail):
    return {
        "candidate_id": candidate_detail["candidate_id"],
        "descendente": candidate_detail["decoded_text"],
        "score_descendente": candidate_detail["target_class_score"],
        "objective_value": candidate_detail["objective_value"],
        "tokens_descendente": candidate_detail["tokens_descendente"],
        "pai1": base_detail["decoded_text"],
        "pai1_id": base_detail["candidate_id"],
        "score_pai1": base_detail["target_class_score"],
        "tokens_pai1": len(resources.tokenizer.tokenize(base_detail["decoded_text"])),
        "evaluation_index_pai1": base_detail["evaluation_id"],
        "evaluation_index_descendente": candidate_detail["evaluation_id"],
        "base_candidate_id": base_detail["candidate_id"],
        "base_objective_value": base_detail["objective_value"],
        "base_evaluation_index": base_detail["evaluation_id"],
    }


def random_search(resources, config, frase_base):
    del frase_base

    historico_geracoes = {}
    total_evaluations = config["random_search_classifier_evals"]
    if total_evaluations <= 0:
        return historico_geracoes

    initial_size = min(config.get("populacao_inicial", 1), total_evaluations)
    initial_population = generate_initial_population(resources, config, initial_size)
    initial_embeddings = _initial_population_embeddings(resources, initial_population)
    logger = RunLogger(resources, config, initial_size, initial_population)
    initial_details = evaluate_and_log_texts_with_embeddings(
        logger,
        generation=0,
        texts=initial_population,
        embeddings=initial_embeddings,
        operator_records=build_initial_operator_records(initial_size),
        embedding_source="initial_token_embedding",
        text_source="direct_initialization",
    )
    best_detail = max(initial_details, key=lambda candidate: candidate["objective_value"])
    sigma = _get_random_search_sigma(config)
    geracao = 1

    while logger.evaluation_counter < total_evaluations:
        remaining_evaluations = total_evaluations - logger.evaluation_counter
        batch_size = min(config["random_search_batch_size"], remaining_evaluations)
        selected_base_indices = [
            random.randrange(len(initial_embeddings))
            for _ in range(batch_size)
        ]
        base_details = [initial_details[index] for index in selected_base_indices]
        candidate_embeddings = [
            initial_embeddings[index] + sigma * torch.randn_like(initial_embeddings[index])
            for index in selected_base_indices
        ]
        candidate_details = evaluate_and_log_embeddings(logger, generation=geracao, embeddings=candidate_embeddings, operator_records=build_random_sampling_operator_records(base_details))

        candidates_info = []
        for candidate_detail, base_detail in zip(candidate_details, base_details, strict=True):
            if candidate_detail["objective_value"] > best_detail["objective_value"]:
                best_detail = candidate_detail
            candidates_info.append(_candidate_info(resources, candidate_detail, base_detail))

        top_5 = sorted(
            candidates_info,
            key=lambda x: x["score_descendente"],
            reverse=True,
        )[:5]
        historico_geracoes[f"geracao_{geracao}"] = {
            "top_5": top_5,
            "all_candidates": candidates_info,
            "evaluations_cumulative": logger.evaluation_counter,
        }

        geracao += 1

    logger.finalize([best_detail], geracao - 1)
    return historico_geracoes
