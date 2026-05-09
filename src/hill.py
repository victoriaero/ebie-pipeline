from tqdm import tqdm

from src.ebie import gerar_variacao
from src.metrics import (RunLogger, build_initial_operator_records, evaluate_and_log_decoded_embeddings, evaluate_and_log_texts,)
from src.initialization import generate_initial_population


def hill_climbing(resources, config, solucao_inicial):
    historico_geracoes = {}
    logger = RunLogger(resources, config, 1, [solucao_inicial])
    solucao_details = evaluate_and_log_texts(logger, generation=0, texts=[solucao_inicial], operator_records=build_initial_operator_records(1),)[0]
    solucao_atual = solucao_inicial
    score_atual = solucao_details["target_class_score"]
    objective_atual = solucao_details["objective_value"]
    evaluation_index_atual = solucao_details["evaluation_id"]
    candidate_id_atual = solucao_details["candidate_id"]
    max_evaluations = config.get("max_evaluations", 1 + config["num_geracoes"] *config["hill_climbing_neighbors"],)
    geracao = 0

    progress_bar = tqdm(total=max_evaluations, initial=min(logger.evaluation_counter, max_evaluations), desc="Running hill climbing",)

    while logger.evaluation_counter < max_evaluations:
        geracao += 1
        vizinhos_textos = []
        vizinhos_embeddings = []
        parent_records = []
        melhor_vizinho = solucao_atual
        melhor_detail = solucao_details
        melhor_objective = objective_atual
        evaluation_index_melhor = evaluation_index_atual
        candidate_id_melhor = candidate_id_atual
        remaining_evaluations = max_evaluations - logger.evaluation_counter
        max_neighbors = min(config["hill_climbing_neighbors"], remaining_evaluations)

        for _ in range(max_neighbors):
            variation_result = gerar_variacao(resources, config, solucao_atual, return_details=True,)
            nova_frase = variation_result["descendente"]
            nova_embedding = variation_result["embedding"]
            mutation_labels = []
            if variation_result["mutation_applied"]:
                mutation_labels.append("embedding_scale_10_percent")
            if variation_result["num_tokens_added"]:
                mutation_labels.append("add_random_token")
            if variation_result["num_tokens_removed"]:
                mutation_labels.append("remove_token")
            vizinhos_textos.append(nova_frase)
            vizinhos_embeddings.append(nova_embedding)
            parent_records.append({"parent_ids":[candidate_id_atual], "parent1_id":candidate_id_atual, "parent2_id":None, "operator_used":"variation", "mutation_type":"+".join(mutation_labels) if mutation_labels else None, "crossover_type":None, "mutation_applied":bool(mutation_labels), "crossover_applied":False, "num_tokens_added":variation_result["num_tokens_added"], "num_tokens_removed":variation_result["num_tokens_removed"],})

        vizinhos_details = evaluate_and_log_decoded_embeddings(logger, generation=geracao, texts=vizinhos_textos, embeddings=vizinhos_embeddings, operator_records=parent_records,)
        progress_bar.update(len(vizinhos_details))
        vizinhos_info = []

        for vizinho_detail in vizinhos_details:
            vizinhos_info.append({"candidate_id":vizinho_detail["candidate_id"], "descendente":vizinho_detail["decoded_text"], "score_descendente":vizinho_detail["target_class_score"], "objective_value":vizinho_detail["objective_value"], "tokens_descendente":vizinho_detail["tokens_descendente"], "pai1":solucao_atual, "pai1_id":candidate_id_atual, "score_pai1":score_atual, "tokens_pai1":len(resources.tokenizer.tokenize(solucao_atual)), "evaluation_index_pai1":evaluation_index_atual, "evaluation_index_descendente":vizinho_detail["evaluation_id"],})

            if vizinho_detail["objective_value"] > melhor_objective:
                melhor_vizinho = vizinho_detail["decoded_text"]
                melhor_detail = vizinho_detail
                melhor_objective = vizinho_detail["objective_value"]
                evaluation_index_melhor = vizinho_detail["evaluation_id"]
                candidate_id_melhor = vizinho_detail["candidate_id"]

        top_5_vizinhos = sorted(vizinhos_info, key=lambda x:x["score_descendente"], reverse=True,)[:5]
        historico_geracoes[f"geracao_{geracao}"] = {
            "top_5": top_5_vizinhos,
            "all_candidates": vizinhos_info,
            "evaluations_cumulative": logger.evaluation_counter,
        }

        if melhor_objective > objective_atual:
            solucao_atual = melhor_vizinho
            solucao_details = melhor_detail
            score_atual = melhor_detail["target_class_score"]
            objective_atual = melhor_objective
            evaluation_index_atual = evaluation_index_melhor
            candidate_id_atual = candidate_id_melhor
        elif config.get("hill_climbing_restart") and logger.evaluation_counter < max_evaluations:
            restart_solution = generate_initial_population(resources, config, 1)[0]
            restart_detail = evaluate_and_log_texts(logger, generation=geracao, texts=[restart_solution], operator_records=[{"parent_ids":[candidate_id_atual], "parent1_id":candidate_id_atual, "parent2_id":None, "operator_used":"restart", "mutation_type":None, "crossover_type":None, "mutation_applied":False, "crossover_applied":False,}],)[0]
            progress_bar.update(1)

            restart_info = {
                "candidate_id": restart_detail["candidate_id"],
                "descendente": restart_detail["decoded_text"],
                "score_descendente": restart_detail["target_class_score"],
                "objective_value": restart_detail["objective_value"],
                "tokens_descendente": len(resources.tokenizer.tokenize(restart_solution)),
                "pai1": solucao_atual,
                "pai1_id": candidate_id_atual,
                "score_pai1": score_atual,
                "tokens_pai1": len(resources.tokenizer.tokenize(solucao_atual)),
                "evaluation_index_pai1": evaluation_index_atual,
                "evaluation_index_descendente": restart_detail["evaluation_id"],
                "restart": True,
            }
            historico_geracoes[f"geracao_{geracao}"]["all_candidates"].append(restart_info)
            historico_geracoes[f"geracao_{geracao}"]["top_5"] = sorted(historico_geracoes[f"geracao_{geracao}"]["all_candidates"], key=lambda x:x["score_descendente"], reverse=True,)[:5]
            historico_geracoes[f"geracao_{geracao}"]["evaluations_cumulative"] = logger.evaluation_counter

            solucao_atual = restart_detail["decoded_text"]
            solucao_details = restart_detail
            score_atual = restart_detail["target_class_score"]
            objective_atual = restart_detail["objective_value"]
            evaluation_index_atual = restart_detail["evaluation_id"]
            candidate_id_atual = restart_detail["candidate_id"]

    progress_bar.close()
    logger.finalize([solucao_details], geracao)
    return historico_geracoes
