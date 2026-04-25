import random

import torch

import src.decoder as decoder
from src.ebie import avaliar_sentimento, mutacao_embeddings, remover_token


def gerar_variacao_sem_avaliacao(
    resources,
    config,
    frase,
):
    inputs = resources.tokenizer(frase, return_tensors="pt").to(resources.device)
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

    return decoder.decode_embeddings_to_text(resources, config, novos_embeddings)


def random_search(resources, config, frase_base):
    historico_geracoes = {}
    avaliacoes_restantes = config["random_search_classifier_evals"]

    if avaliacoes_restantes <= 0:
        return historico_geracoes

    score_base = avaliar_sentimento(resources, config, [frase_base])[0]
    avaliacoes_restantes -= 1
    evaluations_count = 1
    geracao = 1

    while avaliacoes_restantes > 0:
        candidatos_info = []
        batch_atual = min(config["random_search_batch_size"], avaliacoes_restantes)

        for _ in range(batch_atual):
            nova_frase = gerar_variacao_sem_avaliacao(resources, config, frase_base)
            score_descendente = avaliar_sentimento(resources, config, [nova_frase])[0]
            candidatos_info.append(
                {
                    "descendente": nova_frase,
                    "score_descendente": score_descendente,
                    "tokens_descendente": len(resources.tokenizer.tokenize(nova_frase)),
                    "pai1": frase_base,
                    "score_pai1": score_base,
                    "tokens_pai1": len(resources.tokenizer.tokenize(frase_base)),
                    "evaluation_index_descendente": evaluations_count + 1,
                }
            )
            evaluations_count += 1

        top_5 = sorted(
            candidatos_info,
            key=lambda x: x["score_descendente"],
            reverse=True,
        )[:5]
        historico_geracoes[f"geracao_{geracao}"] = {
            "top_5": top_5,
            "all_candidates": candidatos_info,
            "evaluations_cumulative": evaluations_count,
        }

        avaliacoes_restantes -= batch_atual
        geracao += 1

    return historico_geracoes
