import copy
import random

import torch
from tqdm import tqdm

import src.decoder as decoder
from src.metrics import (RunLogger, build_initial_operator_records, evaluate_and_log_decoded_embeddings, evaluate_and_log_texts_with_embeddings,)


def _tokenize_for_operator(resources, text):
    return resources.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=resources.model_max_length,
        return_special_tokens_mask=True,
    ).to(resources.device)


def _content_token_indices(inputs, sequence_length):
    special_mask = inputs.get("special_tokens_mask")
    if special_mask is None:
        return list(range(sequence_length))

    indices = [
        index
        for index, is_special in enumerate(special_mask[0].tolist())
        if not is_special and index < sequence_length
    ]
    return indices or list(range(sequence_length))


def _sentence_embeddings(resources, text):
    inputs = _tokenize_for_operator(resources, text)
    with torch.no_grad():
        outputs = resources.model.roberta(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
    content_indices = _content_token_indices(inputs, outputs.last_hidden_state.shape[1])
    content_embeddings = outputs.last_hidden_state[:, content_indices, :]
    return content_embeddings, list(range(content_embeddings.shape[1]))


def mutacao_embeddings(resources, config, embeddings, candidate_indices=None):
    if random.random() >= config["prob_mutacao_embedding"]:
        return embeddings, False

    candidate_indices = candidate_indices or list(range(embeddings.shape[1]))
    idx = random.choice(candidate_indices)
    mutation_factor = config.get("mutation_intensity_percent", 0.1)
    directions = torch.where(
        torch.rand_like(embeddings[0, idx]) < 0.5,
        1.0 - mutation_factor,
        1.0 + mutation_factor,
    )
    embeddings[0, idx] *= directions
    return embeddings, True


def remover_token(embeddings, candidate_indices=None):
    candidate_indices = [
        index for index in (candidate_indices or list(range(embeddings.shape[1])))
        if index < embeddings.shape[1]
    ]
    if len(candidate_indices) <= 1:
        return embeddings, False

    idx = random.choice(candidate_indices)
    embeddings = torch.cat([embeddings[:, :idx, :], embeddings[:, idx + 1 :, :]], dim=1)
    return embeddings, True


def gerar_variacao(resources, config, frase=None, return_embedding=False, embeddings=None, content_indices=None, return_details=False):
    if embeddings is None:
        embeddings, content_indices = _sentence_embeddings(resources, frase)
    elif content_indices is None:
        content_indices = list(range(embeddings.shape[1]))

    novos_embeddings, mutation_applied = mutacao_embeddings(resources, config, embeddings.clone(), content_indices)
    num_tokens_added = 0
    num_tokens_removed = 0

    if random.random() < config["prob_add_random_token"]:
        random_embedding = torch.empty(novos_embeddings.shape[-1]).uniform_(-1, 1).to(resources.device)
        random_embedding = random_embedding.unsqueeze(0).unsqueeze(0)
        novos_embeddings = torch.cat([novos_embeddings, random_embedding], dim=1)
        content_indices = [*content_indices, novos_embeddings.shape[1] - 1]
        num_tokens_added = 1

    if random.random() < config["prob_remover_token"]:
        novos_embeddings, removed = remover_token(novos_embeddings, content_indices)
        num_tokens_removed = int(removed)

    nova_frase = decoder.decode_embeddings_to_text(resources, config, novos_embeddings)
    details = {
        "descendente": nova_frase,
        "embedding": novos_embeddings[0],
        "mutation_applied": mutation_applied,
        "num_tokens_added": num_tokens_added,
        "num_tokens_removed": num_tokens_removed,
    }
    if return_details:
        return details
    if return_embedding:
        return nova_frase, novos_embeddings[0]
    return nova_frase


def crossover_embeddings(config, pai1_embedding, pai2_embedding, pai1_indices=None, pai2_indices=None):
    descendente_embedding = pai1_embedding.clone()
    pai1_indices = pai1_indices or list(range(pai1_embedding.shape[1]))
    pai2_indices = pai2_indices or list(range(pai2_embedding.shape[1]))
    token_idx_pai1 = random.choice(pai1_indices)
    token_idx_pai2 = random.choice(pai2_indices)
    num_dimensoes = pai1_embedding.shape[2]
    swap_probability = config.get(
        "crossover_dimension_swap_probability",
        config.get("max_percent_dimensions_crossover", 0.02),
    )
    mascara_troca = torch.rand(num_dimensoes, device=pai1_embedding.device) < swap_probability

    descendente_embedding[0, token_idx_pai1, mascara_troca] = pai2_embedding[0, token_idx_pai2, mascara_troca]

    return descendente_embedding


def torneio(populacao, fitness, tamanho=2):
    selecionados = random.sample(range(len(populacao)), tamanho)
    melhor = max(selecionados, key=lambda idx: fitness[idx])
    return melhor


def crossover(resources, config, frase1, frase2, return_details=False):
    embeddings1, indices1 = _sentence_embeddings(resources, frase1)
    embeddings2, indices2 = _sentence_embeddings(resources, frase2)

    if random.random() < config["prob_crossover_embedding"]:
        descendente_embedding = crossover_embeddings(config, embeddings1, embeddings2, indices1, indices2)
        descendente = decoder.decode_embeddings_to_text(resources, config, descendente_embedding)
        if return_details:
            return descendente, descendente_embedding, indices1, True
        return descendente

    descendente = decoder.decode_embeddings_to_text(resources, config, embeddings1)
    if return_details:
        return descendente, embeddings1, indices1, False
    return descendente


def algoritmo_genetico(resources, config, populacao):
    historico_geracoes = {}
    logger = RunLogger(resources, config, len(populacao), populacao)
    initial_embeddings = [
        _sentence_embeddings(resources, text)[0][0]
        for text in populacao
    ]
    populacao_details = evaluate_and_log_texts_with_embeddings(logger, generation=0, texts=copy.deepcopy(populacao), embeddings=initial_embeddings, operator_records=build_initial_operator_records(len(populacao)), embedding_source="initial_token_embedding", text_source="direct_initialization",)
    fitness = [candidate["objective_value"] for candidate in populacao_details]

    for geracao in tqdm(range(config["num_geracoes"]), desc="Evolving"):
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
            _, descendente_embedding, content_indices, crossover_aplicado = crossover(resources, config, pai1, pai2, return_details=True,)
            variation = gerar_variacao(resources, config, embeddings=descendente_embedding, content_indices=content_indices, return_details=True,)
            nova_frase = variation["descendente"]
            nova_embedding = variation["embedding"]
            mutation_type = None
            if variation["mutation_applied"]:
                mutation_type = "embedding_scale_10_percent"

            parent_records.append({"pai1_idx":pai1_idx, "pai2_idx":pai2_idx, "parent_ids":[pai1_detail["candidate_id"], pai2_detail["candidate_id"]], "parent1_id":pai1_detail["candidate_id"], "parent2_id":pai2_detail["candidate_id"], "operator_used":"variation", "mutation_type":mutation_type, "crossover_type":("ebie_single_token_dimension_crossover" if crossover_aplicado else None), "mutation_applied":variation["mutation_applied"], "crossover_applied":crossover_aplicado, "add_token_applied":variation["num_tokens_added"] > 0, "remove_token_applied":variation["num_tokens_removed"] > 0, "num_tokens_added":variation["num_tokens_added"], "num_tokens_removed":variation["num_tokens_removed"],})
            nova_populacao.append(nova_frase)
            parent_records[-1]["embedding"] = nova_embedding

        nova_populacao_details = evaluate_and_log_decoded_embeddings(logger, generation=geracao + 1, texts=nova_populacao, embeddings=[record["embedding"] for record in parent_records], operator_records=parent_records,)
        descendentes_info = []
        for candidate_detail, parent_record in zip(nova_populacao_details, parent_records, strict=True,):
            pai1_detail = populacao_details[parent_record["pai1_idx"]]
            pai2_detail = populacao_details[parent_record["pai2_idx"]]
            descendentes_info.append({"candidate_id":candidate_detail["candidate_id"], "descendente":candidate_detail["decoded_text"], "score_descendente":candidate_detail["target_class_score"], "objective_value":candidate_detail["objective_value"], "tokens_descendente":candidate_detail["tokens_descendente"], "pai1":pai1_detail["decoded_text"], "pai1_id":pai1_detail["candidate_id"], "score_pai1":pai1_detail["target_class_score"], "tokens_pai1":len(resources.tokenizer.tokenize(pai1_detail["decoded_text"])), "evaluation_index_pai1":pai1_detail["evaluation_id"], "pai2":pai2_detail["decoded_text"], "pai2_id":pai2_detail["candidate_id"], "score_pai2":pai2_detail["target_class_score"], "tokens_pai2":len(resources.tokenizer.tokenize(pai2_detail["decoded_text"])), "evaluation_index_pai2":pai2_detail["evaluation_id"], "evaluation_index_descendente":candidate_detail["evaluation_id"], "crossover_applied":parent_record["crossover_applied"], "mutation_applied":parent_record["mutation_applied"], "elitism":False,})

        top_5_descendentes = sorted(descendentes_info, key=lambda x:x["score_descendente"], reverse=True,)[:5]
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
