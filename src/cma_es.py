import cma
import numpy as np
import torch
from tqdm import tqdm

from src.ebie import _sentence_embeddings
from src.metrics import (
    RunLogger,
    build_initial_operator_records,
    evaluate_and_log_embeddings,
    evaluate_and_log_texts_with_embeddings,
)


def _resolve_cma_seed(config):
    seed = config.get("cma_es_seed")
    if seed is None:
        seed = config.get("_current_seed", config.get("random_seed"))
    if seed is None:
        return 0
    return int(seed)


def _cma_options(config, max_cma_evals):
    options = dict(config.get("cma_es_options") or {})
    options["popsize"] = int(config["cma_es_population_size"])
    options["maxfevals"] = int(max_cma_evals)
    options.setdefault("CMA_diagonal", 0)
    options.setdefault("tolflatfitness", int(max_cma_evals) + 1)
    options.setdefault("tolfun", 0)
    options.setdefault("tolx", 0)
    options.setdefault("tolstagnation", int(max_cma_evals) + 1)
    options.setdefault("verb_disp", 0)
    options.setdefault("verbose", -9)
    options["seed"] = _resolve_cma_seed(config)
    return options


def _get_cma_classifier_budget(config, population_size):
    if config.get("cma_es_classifier_evals") is not None:
        return int(config["cma_es_classifier_evals"])
    if config.get("total_classifier_evaluation_budget") is not None:
        return int(config["total_classifier_evaluation_budget"])
    return 1 + int(config["num_geracoes"]) * int(population_size)


def _candidate_info_from_detail(resources, candidate_detail, reference_detail, generation, candidate_index, cma_sigma, cma_stop):
    return {
        "candidate_id": candidate_detail["candidate_id"],
        "descendente": candidate_detail["decoded_text"],
        "score_descendente": candidate_detail["target_class_score"],
        "objective_value": candidate_detail["objective_value"],
        "fitness_value": -candidate_detail["objective_value"],
        "tokens_descendente": candidate_detail["tokens_descendente"],
        "pai1": reference_detail["decoded_text"],
        "pai1_id": reference_detail["candidate_id"],
        "score_pai1": reference_detail["target_class_score"],
        "tokens_pai1": len(resources.tokenizer.tokenize(reference_detail["decoded_text"])),
        "evaluation_index_pai1": reference_detail["evaluation_id"],
        "evaluation_index_descendente": candidate_detail["evaluation_id"],
        "cma_generation": generation,
        "cma_candidate_index": candidate_index,
        "cma_sigma": cma_sigma,
        "cma_stop": cma_stop,
    }


def cma_es(resources, config, solucao_inicial):
    historico_geracoes = {}
    population_size = int(config["cma_es_population_size"])
    max_cma_classifier_evals = _get_cma_classifier_budget(config, population_size)

    if max_cma_classifier_evals <= 0:
        return historico_geracoes

    embedding_inicial = _sentence_embeddings(resources, solucao_inicial)[0][0]
    shape = tuple(embedding_inicial.shape)
    x0 = embedding_inicial.detach().cpu().numpy().reshape(-1).astype(np.float64)
    sigma0 = float(config["cma_es_sigma"])
    logger = RunLogger(resources, config, population_size, [solucao_inicial])

    initial_detail = evaluate_and_log_texts_with_embeddings(
        logger,
        generation=0,
        texts=[solucao_inicial],
        embeddings=[embedding_inicial],
        operator_records=build_initial_operator_records(1),
        embedding_source="initial_token_embedding",
        text_source="direct_initialization",
    )[0]
    best_detail = initial_detail
    last_generation_details = [initial_detail]

    try:
        options = _cma_options(config, max_cma_classifier_evals)
        strategy = cma.CMAEvolutionStrategy(x0, sigma0, options)
        cma_population_size = int(strategy.popsize)

        with tqdm(total=max_cma_classifier_evals, initial=min(logger.evaluation_counter, max_cma_classifier_evals), desc="Running CMA-ES", unit="eval",) as progress:
            geracao = 1
            while (
                logger.evaluation_counter < max_cma_classifier_evals
                and geracao <= int(config["num_geracoes"])
            ):
                reference_detail = best_detail
                cma_sigma = float(strategy.sigma)
                candidatos = strategy.ask()
                remaining_evaluations = max_cma_classifier_evals - logger.evaluation_counter
                candidatos_avaliados = candidatos[:remaining_evaluations]
                if not candidatos_avaliados:
                    break

                candidato_embeddings = [
                    torch.tensor(np.asarray(vetor_candidato).reshape(shape), dtype=torch.float32, device=resources.device,)
                    for vetor_candidato in candidatos_avaliados
                ]
                operator_records = [
                    {
                        "parent_ids": [reference_detail["candidate_id"]],
                        "parent1_id": reference_detail["candidate_id"],
                        "parent2_id": None,
                        "operator_used": "cma_es_sampling",
                        "mutation_type": "cma_es_gaussian_sampling",
                        "crossover_type": None,
                        "mutation_applied": True,
                        "crossover_applied": False,
                        "add_token_applied": False,
                        "remove_token_applied": False,
                        "num_tokens_added": 0,
                        "num_tokens_removed": 0,
                        "embedding_source": "cma_sample",
                    }
                    for _ in candidatos_avaliados
                ]
                candidate_details = evaluate_and_log_embeddings(logger, generation=geracao, embeddings=candidato_embeddings, operator_records=operator_records,)
                progress.update(len(candidate_details))

                fitness_values = [
                    -candidate_detail["objective_value"]
                    for candidate_detail in candidate_details
                ]
                if len(candidatos_avaliados) == cma_population_size:
                    strategy.tell(candidatos_avaliados, fitness_values)

                generation_best = max(candidate_details, key=lambda item:item["objective_value"],)
                if generation_best["objective_value"] > best_detail["objective_value"]:
                    best_detail = generation_best

                cma_stop = dict(strategy.stop())
                candidatos_info = [
                    _candidate_info_from_detail(resources, candidate_detail, reference_detail, geracao, candidate_index, cma_sigma, cma_stop,)
                    for candidate_index, candidate_detail in enumerate(candidate_details)
                ]
                candidatos_ordenados = sorted(candidatos_info, key=lambda item:item["score_descendente"], reverse=True,)
                top_5 = candidatos_ordenados[:5]
                best_global = {
                    "descendente": best_detail["decoded_text"],
                    "score_descendente": best_detail["target_class_score"],
                    "tokens_descendente": best_detail["tokens_descendente"],
                    "evaluation_index_descendente": best_detail["evaluation_id"],
                    "candidate_id": best_detail["candidate_id"],
                }
                best_global_payload = {
                    **best_global,
                    "evaluations_cumulative": logger.evaluation_counter,
                }

                for item in top_5:
                    item["is_best_global"] = (
                        item["evaluation_index_descendente"]
                        == best_global["evaluation_index_descendente"]
                    )
                for item in candidatos_info:
                    item["is_best_global"] = (
                        item["evaluation_index_descendente"]
                        == best_global["evaluation_index_descendente"]
                    )

                historico_geracoes[f"geracao_{geracao}"] = {
                    "top_5": top_5,
                    "all_candidates": candidatos_info,
                    "evaluations_cumulative": logger.evaluation_counter,
                    "best_global": best_global_payload,
                    "cma_stop": cma_stop,
                }
                last_generation_details = candidate_details
                geracao += 1
                if len(candidatos_avaliados) < cma_population_size or cma_stop:
                    break

        logger.finalize(last_generation_details, geracao - 1)
        return historico_geracoes
    except Exception:
        logger.finalize(last_generation_details, max(0, len(historico_geracoes)))
        raise
