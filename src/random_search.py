from src.metrics import (
    RunLogger,
    build_initial_operator_records,
    evaluate_and_log_texts,
)
from src.initialization import generate_initial_population


def build_random_sampling_operator_records(count):
    return [
        {
            "parent_ids": [],
            "parent1_id": None,
            "parent2_id": None,
            "operator_used": "random_sampling",
            "mutation_type": None,
            "crossover_type": None,
            "mutation_applied": False,
            "crossover_applied": False,
        }
        for _ in range(count)
    ]


def random_search(resources, config, frase_base):
    historico_geracoes = {}
    logger = RunLogger(resources, config, 1, [frase_base])
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
    best_detail = base_detail
    avaliacoes_restantes -= 1
    geracao = 1

    while avaliacoes_restantes > 0:
        batch_atual = min(config["random_search_batch_size"], avaliacoes_restantes)
        candidatos_textos = generate_initial_population(resources, config, batch_atual)
        candidatos_details = evaluate_and_log_texts(
            logger,
            generation=geracao,
            texts=candidatos_textos,
            operator_records=build_random_sampling_operator_records(batch_atual),
        )
        candidatos_info = []
        for candidato_detail in candidatos_details:
            if candidato_detail["objective_value"] > best_detail["objective_value"]:
                best_detail = candidato_detail
            candidatos_info.append(
                {
                    "candidate_id": candidato_detail["candidate_id"],
                    "descendente": candidato_detail["decoded_text"],
                    "score_descendente": candidato_detail["target_class_score"],
                    "objective_value": candidato_detail["objective_value"],
                    "tokens_descendente": candidato_detail["tokens_descendente"],
                    "pai1": None,
                    "pai1_id": None,
                    "score_pai1": None,
                    "tokens_pai1": None,
                    "evaluation_index_pai1": None,
                    "evaluation_index_descendente": candidato_detail["evaluation_id"],
                    "base_candidate_id": base_detail["candidate_id"],
                    "base_objective_value": score_base,
                    "base_evaluation_index": base_detail["evaluation_id"],
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

    logger.finalize([best_detail], geracao - 1)
    return historico_geracoes
