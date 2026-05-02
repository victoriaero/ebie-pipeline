import copy
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.decoder as decoder
from src.ga import (
    GARunLogger,
    build_initial_operator_records,
    evaluate_and_log_decoded_embeddings,
    evaluate_and_log_texts,
)
from src.resources import tokenize_for_roberta


def _get_mutation_magnitude(config, token_embedding):
    if "mutation_intensity_percent" in config:
        rms_magnitude = token_embedding.pow(2).mean().sqrt().item()
        return max(rms_magnitude * config["mutation_intensity_percent"], 1.0e-8)

    return config["perturbacao_magnitude"]


def avaliar_sentimento(resources, config, frases):
    batch_size = config["speedup_factor"]
    classifier_target_label = config.get("classifier_target_label", 0)
    dataset = DataLoader(frases, batch_size=batch_size, shuffle=False)
    resultados = []

    for batch in dataset:
        inputs = resources.classifier_tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {key: value.to(resources.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = resources.classifier_model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            target_scores = scores[:, classifier_target_label].cpu().tolist()
        resultados.extend(target_scores)

    return resultados


def mutacao_embeddings(resources, config, embeddings):
    for i in range(embeddings.shape[1]):
        if random.random() < config["prob_mutacao_embedding"]:
            perturbacao_magnitude = _get_mutation_magnitude(config, embeddings[0, i])
            perturbacao = torch.randn(embeddings[0, i].shape).to(resources.device)
            perturbacao *= (
                torch.rand(embeddings[0, i].shape).to(resources.device)
                * 2
                * perturbacao_magnitude
                - perturbacao_magnitude
            )
            embeddings[0, i] += perturbacao
    return embeddings


def remover_token(embeddings):
    if embeddings.shape[1] > 1:
        idx = random.randint(0, embeddings.shape[1] - 1)
        embeddings = torch.cat([embeddings[:, :idx, :], embeddings[:, idx + 1 :, :]], dim=1)
    return embeddings


def gerar_variacao(resources, config, frase, return_embedding=False):
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
        base_perturbacao_magnitude = _get_mutation_magnitude(config, novos_embeddings[0, idx])
        descendente_perturbacao_magnitude = random.uniform(
            0.5 * base_perturbacao_magnitude,
            1.5 * base_perturbacao_magnitude,
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


def crossover_embeddings(config, pai1_embedding, pai2_embedding):
    min_len = min(pai1_embedding.shape[1], pai2_embedding.shape[1])
    pai1_embedding = pai1_embedding[:, :min_len]
    pai2_embedding = pai2_embedding[:, :min_len]

    descendente_embedding = pai1_embedding.clone()
    num_dimensoes = pai1_embedding.shape[2]
    max_dimensoes_trocadas = int(num_dimensoes * config["max_percent_dimensions_crossover"])
    num_trocas = random.randint(0, max_dimensoes_trocadas)
    indices_troca = random.sample(range(num_dimensoes), num_trocas)

    for i in indices_troca:
        descendente_embedding[:, :, i] = pai2_embedding[:, :, i]

    return descendente_embedding


def torneio(populacao, fitness, tamanho=2):
    selecionados = random.sample(range(len(populacao)), tamanho)
    melhor = max(selecionados, key=lambda idx: fitness[idx])
    return melhor


def crossover(resources, config, frase1, frase2, return_details=False):
    if random.random() < config["prob_crossover_embedding"]:
        inputs1 = tokenize_for_roberta(resources, frase1)
        inputs2 = tokenize_for_roberta(resources, frase2)
        with torch.no_grad():
            outputs1 = resources.model.roberta(**inputs1)
            outputs2 = resources.model.roberta(**inputs2)

        embeddings1 = outputs1.last_hidden_state
        embeddings2 = outputs2.last_hidden_state
        descendente_embedding = crossover_embeddings(config, embeddings1, embeddings2)
        descendente = decoder.decode_embeddings_to_text(resources, config, descendente_embedding)
        if return_details:
            return descendente, True
        return descendente

    palavras1 = frase1.split()
    palavras2 = frase2.split()
    if len(palavras1) > 1 and len(palavras2) > 1:
        ponto = random.randint(1, min(len(palavras1), len(palavras2)) - 1)
        nova_frase = palavras1[:ponto] + palavras2[ponto:]
    else:
        nova_frase = palavras1 if random.random() < 0.5 else palavras2
    descendente = " ".join(nova_frase)
    if return_details:
        return descendente, False
    return descendente


def algoritmo_genetico(resources, config, populacao):
    historico_geracoes = {}
    logger = GARunLogger(resources, config, len(populacao), populacao)
    populacao_details = evaluate_and_log_texts(
        logger,
        generation=0,
        texts=copy.deepcopy(populacao),
        operator_records=build_initial_operator_records(len(populacao)),
    )
    fitness = [candidate["objective_value"] for candidate in populacao_details]

    for geracao in tqdm(range(config["num_geracoes"]), desc="Evoluindo"):
        nova_populacao = []
        parent_records = []

        while len(nova_populacao) < len(populacao):
            pai1_idx = torneio(populacao, fitness, config["tournament_size"])
            pai2_idx = torneio(populacao, fitness, config["tournament_size"])
            pai1 = populacao[pai1_idx]
            pai2 = populacao[pai2_idx]
            pai1_detail = populacao_details[pai1_idx]
            pai2_detail = populacao_details[pai2_idx]
            logger.mark_selected_parent(pai1_detail["candidate_id"])
            logger.mark_selected_parent(pai2_detail["candidate_id"])
            descendente, crossover_aplicado = crossover(
                resources,
                config,
                pai1,
                pai2,
                return_details=True,
            )
            nova_frase, nova_embedding = gerar_variacao(
                resources,
                config,
                descendente,
                return_embedding=True,
            )

            parent_records.append(
                {
                    "pai1_idx": pai1_idx,
                    "pai2_idx": pai2_idx,
                    "parent_ids": [pai1_detail["candidate_id"], pai2_detail["candidate_id"]],
                    "parent1_id": pai1_detail["candidate_id"],
                    "parent2_id": pai2_detail["candidate_id"],
                    "operator_used": "variation",
                    "mutation_type": "ebie_embedding_variation",
                    "crossover_type": (
                        "ebie_dimension_crossover" if crossover_aplicado else "textual_fallback"
                    ),
                    "mutation_applied": True,
                    "crossover_applied": crossover_aplicado,
                }
            )
            nova_populacao.append(nova_frase)
            parent_records[-1]["embedding"] = nova_embedding

        nova_populacao_details = evaluate_and_log_decoded_embeddings(
            logger,
            generation=geracao + 1,
            texts=nova_populacao,
            embeddings=[record["embedding"] for record in parent_records],
            operator_records=parent_records,
        )
        descendentes_info = []
        for candidate_detail, parent_record in zip(
            nova_populacao_details,
            parent_records,
            strict=True,
        ):
            pai1_detail = populacao_details[parent_record["pai1_idx"]]
            pai2_detail = populacao_details[parent_record["pai2_idx"]]
            descendentes_info.append(
                {
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
                    "crossover_applied": parent_record["crossover_applied"],
                    "mutation_applied": parent_record["mutation_applied"],
                    "elitism": False,
                }
            )

        descendant_scores = avaliar_sentimento(resources, config, nova_populacao)
        descendant_evaluation_offset = evaluations_count
        evaluations_count += len(nova_populacao)

        for idx, (candidate_info, score_descendente) in enumerate(
            zip(descendentes_info, descendant_scores, strict=True)
        ):
            candidate_info["score_descendente"] = score_descendente
            candidate_info["evaluation_index_descendente"] = descendant_evaluation_offset + idx + 1

        top_5_descendentes = sorted(
            descendentes_info,
            key=lambda x: x["score_descendente"],
            reverse=True,
        )[:5]
        historico_geracoes[f"geracao_{geracao + 1}"] = {
            "top_5": top_5_descendentes,
            "all_candidates": descendentes_info,
            "evaluations_cumulative": logger.evaluation_counter,
        }
        populacao = nova_populacao
        populacao_details = nova_populacao_details
        fitness = [candidate["objective_value"] for candidate in populacao_details]

    logger.finalize(populacao_details, config["num_geracoes"])
    return historico_geracoes
