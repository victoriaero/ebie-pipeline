from tqdm import tqdm

from src.ebie import avaliar_sentimento, gerar_variacao


def hill_climbing(resources, config, solucao_inicial):
    historico_geracoes = {}
    solucao_atual = solucao_inicial
    score_atual = avaliar_sentimento(resources, config, [solucao_atual])[0]
    evaluations_count = 1

    for geracao in tqdm(range(config["num_geracoes"]), desc="Subindo a colina"):
        vizinhos_info = []
        melhor_vizinho = solucao_atual
        melhor_score = score_atual

        for _ in range(config["hill_climbing_neighbors"]):
            nova_frase, score_original, score_descendente = gerar_variacao(resources, config, solucao_atual)

            vizinhos_info.append(
                {
                    "descendente": nova_frase,
                    "score_descendente": score_descendente,
                    "tokens_descendente": len(resources.tokenizer.tokenize(nova_frase)),
                    "pai1": solucao_atual,
                    "score_pai1": score_original,
                    "tokens_pai1": len(resources.tokenizer.tokenize(solucao_atual)),
                    "evaluation_index_pai1": evaluations_count + 1,
                    "evaluation_index_descendente": evaluations_count + 2,
                }
            )
            evaluations_count += 2

            if score_descendente > melhor_score:
                melhor_vizinho = nova_frase
                melhor_score = score_descendente

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

    return historico_geracoes
