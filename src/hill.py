import torch
from tqdm import tqdm

from src.ebie import _sentence_embeddings
from src.metrics import (
    RunLogger,
    build_initial_operator_records,
    evaluate_and_log_embeddings,
    evaluate_and_log_texts_with_embeddings,
)


def _text_embedding(resources, text):
    return _sentence_embeddings(resources, text)[0][0]


def _get_hill_sigma(config):
    return config.get(
        "hill_climbing_sigma",
        config.get("hill_climbing_step_size", config.get("mutation_intensity_percent", 0.1)),
    )


def _build_neighbor_record(candidate_id_atual):
    return {
        "parent_ids": [candidate_id_atual],
        "parent1_id": candidate_id_atual,
        "parent2_id": None,
        "operator_used": "gaussian_hill_climbing",
        "mutation_type": "gaussian_noise",
        "crossover_type": None,
        "mutation_applied": True,
        "crossover_applied": False,
        "add_token_applied": False,
        "remove_token_applied": False,
        "num_tokens_added": 0,
        "num_tokens_removed": 0,
    }


def _candidate_info(resources, vizinho_detail, solucao_atual, score_atual, evaluation_index_atual, candidate_id_atual):
    return {
        "candidate_id": vizinho_detail["candidate_id"],
        "descendente": vizinho_detail["decoded_text"],
        "score_descendente": vizinho_detail["target_class_score"],
        "objective_value": vizinho_detail["objective_value"],
        "tokens_descendente": vizinho_detail["tokens_descendente"],
        "pai1": solucao_atual,
        "pai1_id": candidate_id_atual,
        "score_pai1": score_atual,
        "tokens_pai1": len(resources.tokenizer.tokenize(solucao_atual)),
        "evaluation_index_pai1": evaluation_index_atual,
        "evaluation_index_descendente": vizinho_detail["evaluation_id"],
    }


def hill_climbing(resources, config, solucao_inicial):
    historico_geracoes = {}
    logger = RunLogger(resources, config, 1, [solucao_inicial])
    embedding_atual = _text_embedding(resources, solucao_inicial)
    solucao_details = evaluate_and_log_texts_with_embeddings(
        logger,
        generation=0,
        texts=[solucao_inicial],
        embeddings=[embedding_atual],
        operator_records=build_initial_operator_records(1),
        embedding_source="initial_token_embedding",
        text_source="direct_initialization",
    )[0]
    solucao_atual = solucao_inicial
    score_atual = solucao_details["target_class_score"]
    objective_atual = solucao_details["objective_value"]
    evaluation_index_atual = solucao_details["evaluation_id"]
    candidate_id_atual = solucao_details["candidate_id"]
    max_evaluations = config.get(
        "max_evaluations",
        1 + config["num_geracoes"] * config["hill_climbing_neighbors"],
    )
    sigma = _get_hill_sigma(config)
    geracao = 0

    progress_bar = tqdm(
        total=max_evaluations,
        initial=min(logger.evaluation_counter, max_evaluations),
        desc="Running Gaussian hill climbing",
    )

    while logger.evaluation_counter < max_evaluations:
        geracao += 1
        vizinhos_embeddings = []
        parent_records = []
        melhor_detail = solucao_details
        melhor_embedding = embedding_atual
        melhor_objective = objective_atual
        remaining_evaluations = max_evaluations - logger.evaluation_counter
        max_neighbors = min(config["hill_climbing_neighbors"], remaining_evaluations)

        for _ in range(max_neighbors):
            neighbor_embedding = embedding_atual + sigma * torch.randn_like(embedding_atual)
            vizinhos_embeddings.append(neighbor_embedding)
            parent_records.append(_build_neighbor_record(candidate_id_atual))

        vizinhos_details = evaluate_and_log_embeddings(logger, generation=geracao, embeddings=vizinhos_embeddings, operator_records=parent_records)
        progress_bar.update(len(vizinhos_details))
        vizinhos_info = []

        for vizinho_detail, vizinho_embedding in zip(vizinhos_details, vizinhos_embeddings, strict=True):
            vizinhos_info.append(_candidate_info(resources, vizinho_detail, solucao_atual, score_atual, evaluation_index_atual, candidate_id_atual))

            if vizinho_detail["objective_value"] > melhor_objective:
                melhor_detail = vizinho_detail
                melhor_embedding = vizinho_embedding
                melhor_objective = vizinho_detail["objective_value"]

        top_5_vizinhos = sorted(
            vizinhos_info,
            key=lambda x: x["score_descendente"],
            reverse=True,
        )[:5]
        historico_geracoes[f"geracao_{geracao}"] = {
            "top_5": top_5_vizinhos,
            "all_candidates": vizinhos_info,
            "evaluations_cumulative": logger.evaluation_counter,
        }

        if melhor_objective > objective_atual:
            solucao_atual = melhor_detail["decoded_text"]
            solucao_details = melhor_detail
            embedding_atual = melhor_embedding
            score_atual = melhor_detail["target_class_score"]
            objective_atual = melhor_objective
            evaluation_index_atual = melhor_detail["evaluation_id"]
            candidate_id_atual = melhor_detail["candidate_id"]

    progress_bar.close()
    logger.finalize([solucao_details], geracao)
    return historico_geracoes
