import random

import torch
from tqdm import tqdm

from src.metrics import (
    RunLogger,
    build_initial_operator_records,
    evaluate_and_log_embeddings,
    population_to_embeddings,
)


def _get_crossover_prob(config):
    return config.get("ga_crossover_prob", config.get("prob_crossover_embedding", 0.0))


def _get_mutation_prob(config):
    return config.get("ga_mutation_prob", config.get("prob_mutacao_embedding", 0.0))


def _get_embedding_mutation_std(config):
    return config.get(
        "ga_embedding_mutation_std",
        config.get("mutation_intensity_percent", 0.1),
    )


def torneio(population_size, fitness, tournament_size=2):
    tournament_size = min(tournament_size, population_size)
    selected = random.sample(range(population_size), tournament_size)
    return max(selected, key=lambda idx: fitness[idx])


def crossover_aritmetico(config, pai1_embedding, pai2_embedding):
    if random.random() >= _get_crossover_prob(config):
        parent = pai1_embedding if random.random() < 0.5 else pai2_embedding
        return parent.clone(), False, None

    alpha = random.random()
    filho = alpha * pai1_embedding + (1.0 - alpha) * pai2_embedding
    return filho, True, alpha


def mutacao_gaussiana(config, embedding):
    if random.random() >= _get_mutation_prob(config):
        return embedding, False

    sigma = _get_embedding_mutation_std(config)
    ruido = torch.randn_like(embedding) * sigma
    return embedding + ruido, True


def gerar_descendente_embedding(config, pai1_embedding, pai2_embedding):
    filho, crossover_aplicado, crossover_alpha = crossover_aritmetico(
        config,
        pai1_embedding,
        pai2_embedding,
    )
    filho, mutacao_aplicada = mutacao_gaussiana(config, filho)
    return filho, crossover_aplicado, crossover_alpha, mutacao_aplicada


def _candidate_info(
    resources,
    candidate_detail,
    pai1_detail,
    pai2_detail,
    crossover_aplicado,
    crossover_alpha,
    mutacao_aplicada,
):
    return {
        "candidate_id": candidate_detail["candidate_id"],
        "descendente": candidate_detail["decoded_text"],
        "score_descendente": candidate_detail["target_class_score"],
        "objective_value": candidate_detail["objective_value"],
        "tokens_descendente": candidate_detail["tokens_descendente"],
        "pai1": pai1_detail["decoded_text"],
        "pai1_id": pai1_detail["candidate_id"],
        "score_pai1": pai1_detail["target_class_score"],
        "tokens_pai1": len(resources.tokenizer.tokenize(pai1_detail["decoded_text"])),
        "evaluation_index_pai1": pai1_detail["evaluation_id"],
        "pai2": pai2_detail["decoded_text"],
        "pai2_id": pai2_detail["candidate_id"],
        "score_pai2": pai2_detail["target_class_score"],
        "tokens_pai2": len(resources.tokenizer.tokenize(pai2_detail["decoded_text"])),
        "evaluation_index_pai2": pai2_detail["evaluation_id"],
        "evaluation_index_descendente": candidate_detail["evaluation_id"],
        "crossover_applied": crossover_aplicado,
        "crossover_alpha": crossover_alpha,
        "mutation_applied": mutacao_aplicada,
        "elitism": False,
    }


def vanilla_ga(resources, config, populacao_inicial, vocabulary=None):
    del vocabulary

    if not populacao_inicial:
        return {}

    populacao_embeddings = population_to_embeddings(resources, populacao_inicial)
    population_size = len(populacao_embeddings)
    logger = RunLogger(resources, config, population_size, populacao_inicial)
    populacao_details = evaluate_and_log_embeddings(
        logger,
        generation=0,
        embeddings=populacao_embeddings,
        operator_records=build_initial_operator_records(population_size),
    )
    fitness = [candidate["objective_value"] for candidate in populacao_details]
    tournament_size = config.get("tournament_size", 2)
    historico_geracoes = {}

    for geracao in tqdm(range(config["num_geracoes"]), desc="Evoluindo Embedding Vanilla GA"):
        nova_populacao_embeddings = []
        parent_records = []

        while len(nova_populacao_embeddings) < population_size:
            pai1_idx = torneio(population_size, fitness, tournament_size)
            pai2_idx = torneio(population_size, fitness, tournament_size)
            pai1_detail = populacao_details[pai1_idx]
            pai2_detail = populacao_details[pai2_idx]
            logger.mark_selected_parent(pai1_detail["candidate_id"])
            logger.mark_selected_parent(pai2_detail["candidate_id"])
            filho, crossover_aplicado, crossover_alpha, mutacao_aplicada = (
                gerar_descendente_embedding(
                    config,
                    populacao_embeddings[pai1_idx],
                    populacao_embeddings[pai2_idx],
                )
            )

            nova_populacao_embeddings.append(filho)
            parent_records.append(
                {
                    "pai1_idx": pai1_idx,
                    "pai2_idx": pai2_idx,
                    "parent_ids": [pai1_detail["candidate_id"], pai2_detail["candidate_id"]],
                    "parent1_id": pai1_detail["candidate_id"],
                    "parent2_id": pai2_detail["candidate_id"],
                    "operator_used": "variation",
                    "mutation_type": "gaussian_embedding" if mutacao_aplicada else None,
                    "crossover_type": "arithmetic_convex" if crossover_aplicado else None,
                    "mutation_applied": mutacao_aplicada,
                    "crossover_applied": crossover_aplicado,
                    "crossover_alpha": crossover_alpha,
                }
            )

        nova_populacao_details = evaluate_and_log_embeddings(
            logger,
            generation=geracao + 1,
            embeddings=nova_populacao_embeddings,
            operator_records=parent_records,
        )

        descendentes_info = []
        for candidate_detail, parent_record in zip(
            nova_populacao_details,
            parent_records,
            strict=True,
        ):
            descendentes_info.append(
                _candidate_info(
                    resources,
                    candidate_detail,
                    populacao_details[parent_record["pai1_idx"]],
                    populacao_details[parent_record["pai2_idx"]],
                    parent_record["crossover_applied"],
                    parent_record["crossover_alpha"],
                    parent_record["mutation_applied"],
                )
            )

        top_5_descendentes = sorted(
            descendentes_info,
            key=lambda item: item["score_descendente"],
            reverse=True,
        )[:5]
        historico_geracoes[f"geracao_{geracao + 1}"] = {
            "top_5": top_5_descendentes,
            "all_candidates": descendentes_info,
            "evaluations_cumulative": logger.evaluation_counter,
            "elitism": False,
        }

        populacao_embeddings = nova_populacao_embeddings
        populacao_details = nova_populacao_details
        fitness = [candidate["objective_value"] for candidate in populacao_details]

    logger.finalize(populacao_details, config["num_geracoes"])
    return historico_geracoes
