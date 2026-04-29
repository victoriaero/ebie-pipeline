from tqdm import tqdm

from src.ebie import avaliar_sentimento, gerar_variacao
from src.initialization import generate_initial_population


def _extract_descendant_and_score(resources, config, variation_result):
    if isinstance(variation_result, tuple) and len(variation_result) == 2:
        descendente, score_descendente = variation_result
        if score_descendente is None:
            score_descendente = avaliar_sentimento(resources, config, [descendente])[0]
        return descendente, score_descendente

    if isinstance(variation_result, dict):
        descendente = variation_result["descendente"]
        score_descendente = variation_result.get("score_descendente")
        if score_descendente is None:
            score_descendente = avaliar_sentimento(resources, config, [descendente])[0]
        return descendente, score_descendente

    descendente = variation_result
    score_descendente = avaliar_sentimento(resources, config, [descendente])[0]
    return descendente, score_descendente


def hill_climbing(resources, config, solucao_inicial):
    historico_geracoes = {}
    solucao_atual = solucao_inicial
    score_atual = avaliar_sentimento(resources, config, [solucao_atual])[0]
    evaluation_index_atual = 1
    evaluations_count = 1
    max_evaluations = config.get(
        "max_evaluations",
        1 + config["num_geracoes"] * config["hill_climbing_neighbors"],
    )
    geracao = 0

    progress_bar = tqdm(
        total=max_evaluations,
        initial=min(evaluations_count, max_evaluations),
        desc="Subindo a colina",
    )

    while evaluations_count < max_evaluations:
        geracao += 1
        vizinhos_info = []
        melhor_vizinho = solucao_atual
        melhor_score = score_atual
        evaluation_index_melhor = evaluation_index_atual
        remaining_evaluations = max_evaluations - evaluations_count
        max_neighbors = min(config["hill_climbing_neighbors"], remaining_evaluations)

        for _ in range(max_neighbors):
            variation_result = gerar_variacao(resources, config, solucao_atual)
            nova_frase, score_descendente = _extract_descendant_and_score(
                resources,
                config,
                variation_result,
            )
            evaluations_count += 1
            progress_bar.update(1)
            evaluation_index_descendente = evaluations_count

            vizinhos_info.append(
                {
                    "descendente": nova_frase,
                    "score_descendente": score_descendente,
                    "tokens_descendente": len(resources.tokenizer.tokenize(nova_frase)),
                    "pai1": solucao_atual,
                    "score_pai1": score_atual,
                    "tokens_pai1": len(resources.tokenizer.tokenize(solucao_atual)),
                    "evaluation_index_pai1": evaluation_index_atual,
                    "evaluation_index_descendente": evaluation_index_descendente,
                }
            )

            if score_descendente > melhor_score:
                melhor_vizinho = nova_frase
                melhor_score = score_descendente
                evaluation_index_melhor = evaluation_index_descendente

        top_5_vizinhos = sorted(
            vizinhos_info,
            key=lambda x: x["score_descendente"],
            reverse=True,
        )[:5]
        historico_geracoes[f"geracao_{geracao}"] = {
            "top_5": top_5_vizinhos,
            "all_candidates": vizinhos_info,
            "evaluations_cumulative": evaluations_count,
        }

        if melhor_score > score_atual:
            solucao_atual = melhor_vizinho
            score_atual = melhor_score
            evaluation_index_atual = evaluation_index_melhor
        elif config.get("hill_climbing_restart") and evaluations_count < max_evaluations:
            restart_solution = generate_initial_population(resources, config, 1)[0]
            restart_score = avaliar_sentimento(resources, config, [restart_solution])[0]
            evaluations_count += 1
            progress_bar.update(1)
            restart_evaluation_index = evaluations_count

            restart_info = {
                "descendente": restart_solution,
                "score_descendente": restart_score,
                "tokens_descendente": len(resources.tokenizer.tokenize(restart_solution)),
                "pai1": solucao_atual,
                "score_pai1": score_atual,
                "tokens_pai1": len(resources.tokenizer.tokenize(solucao_atual)),
                "evaluation_index_pai1": evaluation_index_atual,
                "evaluation_index_descendente": restart_evaluation_index,
                "restart": True,
            }
            historico_geracoes[f"geracao_{geracao}"]["all_candidates"].append(restart_info)
            historico_geracoes[f"geracao_{geracao}"]["top_5"] = sorted(
                historico_geracoes[f"geracao_{geracao}"]["all_candidates"],
                key=lambda x: x["score_descendente"],
                reverse=True,
            )[:5]
            historico_geracoes[f"geracao_{geracao}"]["evaluations_cumulative"] = evaluations_count

            # Random restart jumps to a fresh point even if it is worse, so the search
            # can leave a plateau and continue exploring a new region.
            solucao_atual = restart_solution
            score_atual = restart_score
            evaluation_index_atual = restart_evaluation_index

    progress_bar.close()
    return historico_geracoes
