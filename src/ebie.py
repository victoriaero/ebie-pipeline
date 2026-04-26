import copy
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.decoder as decoder
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


def gerar_variacao(resources, config, frase):
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


def crossover(resources, config, frase1, frase2):
    if random.random() < config["prob_crossover_embedding"]:
        inputs1 = tokenize_for_roberta(resources, frase1)
        inputs2 = tokenize_for_roberta(resources, frase2)
        with torch.no_grad():
            outputs1 = resources.model.roberta(**inputs1)
            outputs2 = resources.model.roberta(**inputs2)

        embeddings1 = outputs1.last_hidden_state
        embeddings2 = outputs2.last_hidden_state
        descendente_embedding = crossover_embeddings(config, embeddings1, embeddings2)
        return decoder.decode_embeddings_to_text(resources, config, descendente_embedding)

    palavras1 = frase1.split()
    palavras2 = frase2.split()
    if len(palavras1) > 1 and len(palavras2) > 1:
        ponto = random.randint(1, min(len(palavras1), len(palavras2)) - 1)
        nova_frase = palavras1[:ponto] + palavras2[ponto:]
    else:
        nova_frase = palavras1 if random.random() < 0.5 else palavras2
    return " ".join(nova_frase)


def algoritmo_genetico(resources, config, populacao):
    historico_geracoes = {}
    evaluations_count = 0
    for geracao in tqdm(range(config["num_geracoes"]), desc="Evoluindo"):
        populacao_copy = copy.deepcopy(populacao)
        fitness = avaliar_sentimento(resources, config, populacao_copy)
        parent_evaluation_offset = evaluations_count
        evaluations_count += len(populacao)
        nova_populacao = []
        descendentes_info = []

        while len(nova_populacao) < len(populacao):
            pai1_idx = torneio(populacao, fitness, config["tournament_size"])
            pai2_idx = torneio(populacao, fitness, config["tournament_size"])
            pai1 = populacao[pai1_idx]
            pai2 = populacao[pai2_idx]
            descendente = crossover(resources, config, pai1, pai2)
            nova_frase = gerar_variacao(resources, config, descendente)

            descendentes_info.append(
                {
                    "descendente": nova_frase,
                    "score_descendente": None,
                    "tokens_descendente": len(resources.tokenizer.tokenize(nova_frase)),
                    "pai1": pai1,
                    "score_pai1": fitness[pai1_idx],
                    "tokens_pai1": len(resources.tokenizer.tokenize(pai1)),
                    "evaluation_index_pai1": parent_evaluation_offset + pai1_idx + 1,
                    "evaluation_index_descendente": None,
                }
            )
            nova_populacao.append(nova_frase)

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
            "evaluations_cumulative": evaluations_count,
        }
        populacao = nova_populacao

    return historico_geracoes
