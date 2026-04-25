import math

import numpy as np
import torch
from tqdm import tqdm

import src.decoder as decoder
from src.ebie import avaliar_sentimento
from src.resources import tokenize_for_roberta


def frase_para_embedding(resources, frase):
    inputs = tokenize_for_roberta(resources, frase)
    with torch.no_grad():
        outputs = resources.model.roberta(**inputs)
        embeddings = outputs.last_hidden_state
    return embeddings


def embedding_para_frase(resources, config, embedding):
    return decoder.decode_embeddings_to_text(resources, config, embedding)


def avaliar_candidato(resources, config, vetor_candidato, shape, frase_referencia):
    embedding = torch.tensor(
        vetor_candidato.reshape(shape),
        dtype=torch.float32,
        device=resources.device,
    )
    nova_frase = embedding_para_frase(resources, config, embedding)
    score_pai = avaliar_sentimento(resources, config, [frase_referencia])[0]
    score_descendente = avaliar_sentimento(resources, config, [nova_frase])[0]

    return {
        "descendente": nova_frase,
        "score_descendente": score_descendente,
        "tokens_descendente": len(resources.tokenizer.tokenize(nova_frase)),
        "pai1": frase_referencia,
        "score_pai1": score_pai,
        "tokens_pai1": len(resources.tokenizer.tokenize(frase_referencia)),
        "vector": vetor_candidato,
    }


def cma_es(resources, config, solucao_inicial):
    historico_geracoes = {}

    embedding_inicial = frase_para_embedding(resources, solucao_inicial)
    shape = tuple(embedding_inicial.shape)
    mean = embedding_inicial.detach().cpu().numpy().reshape(-1)
    cov_diag = np.ones_like(mean, dtype=np.float64)

    score_atual = avaliar_sentimento(resources, config, [solucao_inicial])[0]
    frase_atual = solucao_inicial
    evaluations_count = 1

    for geracao in tqdm(range(config["num_geracoes"]), desc="Executando CMA-ES"):
        candidatos_info = []

        for _ in range(config["cma_es_population_size"]):
            amostra = np.random.randn(mean.size)
            candidato = mean + config["cma_es_sigma"] * np.sqrt(cov_diag) * amostra
            info = avaliar_candidato(resources, config, candidato, shape, frase_atual)
            info["evaluation_index_pai1"] = evaluations_count + 1
            info["evaluation_index_descendente"] = evaluations_count + 2
            evaluations_count += 2
            candidatos_info.append(info)

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
            "evaluations_cumulative": evaluations_count,
        }

    return historico_geracoes
