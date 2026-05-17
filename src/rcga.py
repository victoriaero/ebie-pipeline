import random

import torch
from tqdm import tqdm

from src.ebie import _sentence_embeddings
from src.metrics import (
    RunLogger,
    build_initial_operator_records,
    evaluate_and_log_embeddings,
    evaluate_and_log_texts_with_embeddings,
)


def _get_crossover_prob(config):
    return config.get("rcga_crossover_prob", config.get("prob_crossover_embedding", 0.0))


def _get_mutation_prob(config):
    return config.get("rcga_mutation_prob", config.get("prob_mutacao_embedding", 0.0))


def _get_embedding_mutation_std(config):
    return config.get("rcga_embedding_mutation_std", config.get("mutation_intensity_percent", 0.1),)


def tournament(population_size, fitness, tournament_size=2):
    tournament_size = min(tournament_size, population_size)
    selected = random.sample(range(population_size), tournament_size)
    return max(selected, key=lambda idx: fitness[idx])


def arithmetic_crossover(config, parent1_embedding, parent2_embedding):
    if random.random() >= _get_crossover_prob(config):
        parent = parent1_embedding if random.random() < 0.5 else parent2_embedding
        return parent.clone(), False, None

    alpha = random.random()
    child = alpha * parent1_embedding + (1.0 - alpha) * parent2_embedding
    return child, True, alpha


def gaussian_mutation(config, embedding):
    if random.random() >= _get_mutation_prob(config):
        return embedding, False

    sigma = _get_embedding_mutation_std(config)
    noise = torch.randn_like(embedding) * sigma
    return embedding + noise, True


def generate_child_embedding(config, parent1_embedding, parent2_embedding):
    child, crossover_applied, crossover_alpha = arithmetic_crossover(config, parent1_embedding, parent2_embedding,)
    child, mutation_applied = gaussian_mutation(config, child)
    return child, crossover_applied, crossover_alpha, mutation_applied


def _initial_population_embeddings(resources, initial_population):
    embeddings = [
        _sentence_embeddings(resources, text)[0][0]
        for text in initial_population
    ]
    embedding_shapes = {tuple(embedding.shape) for embedding in embeddings}
    if len(embedding_shapes) != 1:
        raise ValueError(
            "RCGA requires a fixed-length initial representation. Use the same "
            "number of RoBERTa content tokens for every initial individual, e.g. "
            "initialization_mode: random_1_token."
        )
    return embeddings


def _candidate_info(resources, candidate_detail, parent1_detail, parent2_detail, crossover_applied, crossover_alpha, mutation_applied):
    return {
        "candidate_id": candidate_detail["candidate_id"],
        "descendente": candidate_detail["decoded_text"],
        "score_descendente": candidate_detail["target_class_score"],
        "objective_value": candidate_detail["objective_value"],
        "tokens_descendente": candidate_detail["tokens_descendente"],
        "pai1": parent1_detail["decoded_text"],
        "pai1_id": parent1_detail["candidate_id"],
        "score_pai1": parent1_detail["target_class_score"],
        "tokens_pai1": len(resources.tokenizer.tokenize(parent1_detail["decoded_text"])),
        "evaluation_index_pai1": parent1_detail["evaluation_id"],
        "pai2": parent2_detail["decoded_text"],
        "pai2_id": parent2_detail["candidate_id"],
        "score_pai2": parent2_detail["target_class_score"],
        "tokens_pai2": len(resources.tokenizer.tokenize(parent2_detail["decoded_text"])),
        "evaluation_index_pai2": parent2_detail["evaluation_id"],
        "evaluation_index_descendente": candidate_detail["evaluation_id"],
        "crossover_applied": crossover_applied,
        "crossover_alpha": crossover_alpha,
        "mutation_applied": mutation_applied,
        "elitism": False,
    }


def rcga(resources, config, initial_population, vocabulary=None):
    del vocabulary

    if not initial_population:
        return {}

    population_embeddings = _initial_population_embeddings(resources, initial_population)
    population_size = len(population_embeddings)
    logger = RunLogger(resources, config, population_size, initial_population)
    population_details = evaluate_and_log_texts_with_embeddings(
        logger,
        generation=0,
        texts=initial_population,
        embeddings=population_embeddings,
        operator_records=build_initial_operator_records(population_size),
        embedding_source="initial_token_embedding",
        text_source="direct_initialization",
    )
    fitness = [candidate["objective_value"] for candidate in population_details]
    tournament_size = config.get("tournament_size", 2)
    generation_history = {}

    for generation in tqdm(range(config["num_geracoes"]), desc="Evolving RCGA embeddings"):
        new_population_embeddings = []
        parent_records = []

        while len(new_population_embeddings) < population_size:
            parent1_idx = tournament(population_size, fitness, tournament_size)
            parent2_idx = tournament(population_size, fitness, tournament_size)
            parent1_detail = population_details[parent1_idx]
            parent2_detail = population_details[parent2_idx]
            logger.mark_selected_parent(parent1_detail["candidate_id"])
            logger.mark_selected_parent(parent2_detail["candidate_id"])
            child, crossover_applied, crossover_alpha, mutation_applied = (
                generate_child_embedding(config, population_embeddings[parent1_idx], population_embeddings[parent2_idx],)
            )

            new_population_embeddings.append(child)
            parent_records.append({"parent1_idx":parent1_idx, "parent2_idx":parent2_idx, "parent_ids":[parent1_detail["candidate_id"], parent2_detail["candidate_id"]], "parent1_id":parent1_detail["candidate_id"], "parent2_id":parent2_detail["candidate_id"], "operator_used":"variation", "mutation_type":"gaussian_embedding" if mutation_applied else None, "crossover_type":"arithmetic_convex" if crossover_applied else None, "mutation_applied":mutation_applied, "crossover_applied":crossover_applied, "crossover_alpha":crossover_alpha,})

        new_population_details = evaluate_and_log_embeddings(logger, generation=generation + 1, embeddings=new_population_embeddings, operator_records=parent_records,)

        descendants_info = []
        for candidate_detail, parent_record in zip(new_population_details, parent_records, strict=True,):
            descendants_info.append(_candidate_info(resources, candidate_detail, population_details[parent_record["parent1_idx"]], population_details[parent_record["parent2_idx"]], parent_record["crossover_applied"], parent_record["crossover_alpha"], parent_record["mutation_applied"],))

        top_5_descendants = sorted(descendants_info, key=lambda item:item["score_descendente"], reverse=True,)[:5]
        generation_history[f"geracao_{generation + 1}"] = {
            "top_5": top_5_descendants,
            "all_candidates": descendants_info,
            "evaluations_cumulative": logger.evaluation_counter,
            "elitism": False,
        }

        population_embeddings = new_population_embeddings
        population_details = new_population_details
        fitness = [candidate["objective_value"] for candidate in population_details]

    logger.finalize(population_details, config["num_geracoes"])
    return generation_history
