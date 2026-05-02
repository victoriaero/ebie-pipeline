import math

import numpy as np
import torch
from tqdm import tqdm

import src.decoder as decoder
from src.ga import GARunLogger, build_initial_operator_records, evaluate_and_log_embeddings
from src.resources import tokenize_for_roberta


def frase_para_embedding(resources, frase):
    inputs = tokenize_for_roberta(resources, frase)
    with torch.no_grad():
        outputs = resources.model.roberta(**inputs)
        embeddings = outputs.last_hidden_state
    return embeddings


def embedding_para_frase(resources, config, embedding):
    return decoder.decode_embeddings_to_text(resources, config, embedding)


def cma_es(resources, config, solucao_inicial):
    historico_geracoes = {}
    logger = GARunLogger(
        resources,
        config,
        config.get("cma_es_population_size", 1),
        [solucao_inicial],
    )

    embedding_inicial = frase_para_embedding(resources, solucao_inicial)
    initial_detail = evaluate_and_log_embeddings(
        logger,
        generation=0,
        embeddings=[embedding_inicial[0]],
        operator_records=build_initial_operator_records(1),
    )[0]
    shape = tuple(embedding_inicial.shape)
    mean = embedding_inicial.detach().cpu().numpy().reshape(-1)
    cov_diag = np.ones_like(mean, dtype=np.float64)

    score_atual = initial_detail["objective_value"]
    frase_atual = initial_detail["decoded_text"]
    candidate_id_atual = initial_detail["candidate_id"]
    evaluation_index_atual = initial_detail["evaluation_id"]

    for geracao in tqdm(range(config["num_geracoes"]), desc="Executando CMA-ES"):
        candidato_embeddings = []
        candidato_vectors = []
        operator_records = []

        for _ in range(config["cma_es_population_size"]):
            amostra = np.random.randn(mean.size)
            candidato = mean + config["cma_es_sigma"] * np.sqrt(cov_diag) * amostra
            embedding = torch.tensor(
                candidato.reshape(shape),
                dtype=torch.float32,
                device=resources.device,
            )
            candidato_embeddings.append(embedding[0])
            candidato_vectors.append(candidato)
            operator_records.append(
                {
                    "parent_ids": [candidate_id_atual],
                    "parent1_id": candidate_id_atual,
                    "parent2_id": None,
                    "operator_used": "sampling",
                    "mutation_type": "cma_es_gaussian_sampling",
                    "crossover_type": None,
                    "mutation_applied": True,
                    "crossover_applied": False,
                    "embedding_source": "search_embedding",
                }
            )

        candidatos_details = evaluate_and_log_embeddings(
            logger,
            generation=geracao + 1,
            embeddings=candidato_embeddings,
            operator_records=operator_records,
        )
        candidatos_info = []
        for detail, vector in zip(candidatos_details, candidato_vectors, strict=True):
            candidatos_info.append(
                {
                    "candidate_id": detail["candidate_id"],
                    "descendente": detail["decoded_text"],
                    "score_descendente": detail["target_class_score"],
                    "objective_value": detail["objective_value"],
                    "tokens_descendente": detail["tokens_descendente"],
                    "pai1": frase_atual,
                    "pai1_id": candidate_id_atual,
                    "score_pai1": score_atual,
                    "tokens_pai1": len(resources.tokenizer.tokenize(frase_atual)),
                    "evaluation_index_pai1": evaluation_index_atual,
                    "evaluation_index_descendente": detail["evaluation_id"],
                    "vector": vector,
                }
            )

        candidatos_ordenados = sorted(
            candidatos_info,
            key=lambda x: x["score_descendente"],
            reverse=True,
        )

        mu = max(1, math.ceil(config["cma_es_population_size"] * config["cma_es_elite_ratio"]))
        elite = candidatos_ordenados[:mu]
        elite_vectors = np.stack([item["vector"] for item in elite], axis=0)

        mean = np.mean(elite_vectors, axis=0)
        cov_diag = np.var(elite_vectors, axis=0) + config["cma_es_cov_eps"]

        melhor = elite[0]
        if melhor["score_descendente"] > score_atual:
            frase_atual = melhor["descendente"]
            score_atual = melhor["score_descendente"]
            candidate_id_atual = melhor["candidate_id"]
            evaluation_index_atual = melhor["evaluation_index_descendente"]

        top_5 = []
        for item in candidatos_ordenados[:5]:
            top_5.append(
                {
                    "descendente": item["descendente"],
                    "score_descendente": item["score_descendente"],
                    "tokens_descendente": item["tokens_descendente"],
                    "pai1": item["pai1"],
                    "score_pai1": item["score_pai1"],
                    "tokens_pai1": item["tokens_pai1"],
                }
            )

        historico_geracoes[f"geracao_{geracao + 1}"] = {
            "top_5": top_5,
            "all_candidates": [
                {
                    "descendente": item["descendente"],
                    "score_descendente": item["score_descendente"],
                    "tokens_descendente": item["tokens_descendente"],
                    "pai1": item["pai1"],
                    "score_pai1": item["score_pai1"],
                    "tokens_pai1": item["tokens_pai1"],
                    "evaluation_index_pai1": item["evaluation_index_pai1"],
                    "evaluation_index_descendente": item["evaluation_index_descendente"],
                }
                for item in candidatos_info
            ],
            "evaluations_cumulative": logger.evaluation_counter,
        }

    logger.finalize(
        [
            {
                "candidate_id": candidate_id_atual,
                "target_class_score": score_atual,
                "objective_value": score_atual,
            }
        ],
        config["num_geracoes"],
    )
    return historico_geracoes
