from tqdm import tqdm

from src.ebie import avaliar_sentimento, gerar_variacao
from src.initialization import generate_initial_population


def hill_climbing(resources, config, solucao_inicial):
    historico_geracoes = {}
    solucao_atual = solucao_inicial
    score_atual = avaliar_sentimento(resources, config, [solucao_atual])[0]
    evaluation_index_atual = 1
    evaluations_count = 1

    for geracao in tqdm(range(config["num_geracoes"]), desc="Subindo a colina"):
        vizinhos_info = []
        melhor_vizinho = solucao_atual
        melhor_score = score_atual
        vizinhos = []

        for _ in range(config["hill_climbing_neighbors"]):
            vizinhos.append(gerar_variacao(resources, config, solucao_atual))

        vizinhos_scores = avaliar_sentimento(resources, config, vizinhos)
        descendant_evaluation_offset = evaluations_count
        evaluations_count += len(vizinhos)

        for idx, (nova_frase, score_descendente) in enumerate(
            zip(vizinhos, vizinhos_scores, strict=True)
        ):
            evaluation_index_descendente = descendant_evaluation_offset + idx + 1

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
        historico_geracoes[f"geracao_{geracao + 1}"] = {
            "top_5": top_5_vizinhos,
            "all_candidates": vizinhos_info,
            "evaluations_cumulative": evaluations_count,
        }

        if melhor_score > score_atual:
            solucao_atual = melhor_vizinho
            score_atual = melhor_score
            evaluation_index_atual = evaluation_index_melhor
        elif config.get("hill_climbing_restart"):
            restart_solution = generate_initial_population(resources, config, 1)[0]
            restart_score = avaliar_sentimento(resources, config, [restart_solution])[0]
            evaluations_count += 1
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
            historico_geracoes[f"geracao_{geracao + 1}"]["all_candidates"].append(restart_info)
            historico_geracoes[f"geracao_{geracao + 1}"]["top_5"] = sorted(
                historico_geracoes[f"geracao_{geracao + 1}"]["all_candidates"],
                key=lambda x: x["score_descendente"],
                reverse=True,
            )[:5]
            historico_geracoes[f"geracao_{geracao + 1}"]["evaluations_cumulative"] = evaluations_count

            # Random restart jumps to a fresh point even if it is worse, so the search
            # can leave a plateau and continue exploring a new region.
            solucao_atual = restart_solution
            score_atual = restart_score
            evaluation_index_atual = restart_evaluation_index

    return historico_geracoes
