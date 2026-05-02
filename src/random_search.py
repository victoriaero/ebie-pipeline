import random

import torch

import src.decoder as decoder
from src.ebie import mutacao_embeddings, remover_token
from src.ga import (
    GARunLogger,
    build_initial_operator_records,
    evaluate_and_log_decoded_embeddings,
    evaluate_and_log_texts,
)
from src.resources import tokenize_for_roberta


def gerar_variacao_sem_avaliacao(
    resources,
    config,
    frase,
    return_embedding=False,
):
    inputs = tokenize_for_roberta(resources, frase)
    with torch.no_grad():
        outputs = resources.model.roberta(**inputs)
        embeddings = outputs.last_hidden_state

    if random.random() < config["prob_mutacao_embedding"]:
        novos_embeddings = mutacao_embeddings(resources, config, embeddings.clone())
    else:
        novos_embeddings = embeddings.clone()

    if random.random() < config["prob_add_random_token"]:
        random_embedding = torch.empty(novos_embeddings.shape[-1]).uniform_(-1, 1).to(resources.device)
        random_embedding = random_embedding.unsqueeze(0).unsqueeze(0)
        novos_embeddings = torch.cat([novos_embeddings, random_embedding], dim=1)
    elif random.random() < config["prob_remover_token"]:
        novos_embeddings = remover_token(novos_embeddings)
    else:
        idx = random.randint(0, novos_embeddings.shape[1] - 1)
        descendente_perturbacao_magnitude = random.uniform(
            0.5 * config["perturbacao_magnitude"],
            1.5 * config["perturbacao_magnitude"],
        )
        perturbacao = torch.randn(novos_embeddings[0, idx].shape).to(resources.device)
        perturbacao *= (
            torch.rand(novos_embeddings[0, idx].shape).to(resources.device)
            * 2
            * descendente_perturbacao_magnitude
            - descendente_perturbacao_magnitude
        )
        novos_embeddings[0, idx] += perturbacao

    nova_frase = decoder.decode_embeddings_to_text(resources, config, novos_embeddings)
    if return_embedding:
        return nova_frase, novos_embeddings[0]
    return nova_frase


def random_search(resources, config, frase_base):
    historico_geracoes = {}
    logger = GARunLogger(resources, config, 1, [frase_base])
    avaliacoes_restantes = config["random_search_classifier_evals"]

    if avaliacoes_restantes <= 0:
        return historico_geracoes

    base_detail = evaluate_and_log_texts(
        logger,
        generation=0,
        texts=[frase_base],
        operator_records=build_initial_operator_records(1),
    )[0]
    score_base = base_detail["objective_value"]
    avaliacoes_restantes -= 1
    geracao = 1

    while avaliacoes_restantes > 0:
        batch_atual = min(config["random_search_batch_size"], avaliacoes_restantes)
        candidatos_textos = []
        candidatos_embeddings = []
        operator_records = []

        for _ in range(batch_atual):
            nova_frase, nova_embedding = gerar_variacao_sem_avaliacao(
                resources,
                config,
                frase_base,
                return_embedding=True,
            )
            candidatos_textos.append(nova_frase)
            candidatos_embeddings.append(nova_embedding)
            operator_records.append(
                {
                    "parent_ids": [base_detail["candidate_id"]],
                    "parent1_id": base_detail["candidate_id"],
                    "parent2_id": None,
                    "operator_used": "sampling",
                    "mutation_type": "random_embedding_variation",
                    "crossover_type": None,
                    "mutation_applied": True,
                    "crossover_applied": False,
                }
            )

        candidatos_details = evaluate_and_log_decoded_embeddings(
            logger,
            generation=geracao,
            texts=candidatos_textos,
            embeddings=candidatos_embeddings,
            operator_records=operator_records,
        )
        candidatos_info = []
        for candidato_detail in candidatos_details:
            candidatos_info.append(
                {
                    "candidate_id": candidato_detail["candidate_id"],
                    "descendente": candidato_detail["decoded_text"],
                    "score_descendente": candidato_detail["target_class_score"],
                    "objective_value": candidato_detail["objective_value"],
                    "tokens_descendente": candidato_detail["tokens_descendente"],
                    "pai1": frase_base,
                    "pai1_id": base_detail["candidate_id"],
                    "score_pai1": score_base,
                    "tokens_pai1": len(resources.tokenizer.tokenize(frase_base)),
                    "evaluation_index_pai1": base_detail["evaluation_id"],
                    "evaluation_index_descendente": candidato_detail["evaluation_id"],
                }
            )

        top_5 = sorted(
            candidatos_info,
            key=lambda x: x["score_descendente"],
            reverse=True,
        )[:5]
        historico_geracoes[f"geracao_{geracao}"] = {
            "top_5": top_5,
            "all_candidates": candidatos_info,
            "evaluations_cumulative": logger.evaluation_counter,
        }

        avaliacoes_restantes -= batch_atual
        geracao += 1

    logger.finalize([base_detail], geracao - 1)
    return historico_geracoes
