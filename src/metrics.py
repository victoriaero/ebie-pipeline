import math
from statistics import mean, pstdev


def _generation_sort_key(item):
    name = item[0]
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return math.inf


def flatten_candidates(history):
    candidates = []
    for _, generation_data in sorted(history.items(), key=_generation_sort_key):
        candidates.extend(generation_data.get("all_candidates", generation_data.get("top_5", [])))
    return candidates


def iter_candidates_with_generation(history):
    for generation_name, generation_data in sorted(history.items(), key=_generation_sort_key):
        candidates = generation_data.get("all_candidates", generation_data.get("top_5", []))
        for candidate in candidates:
            yield generation_name, candidate


def lexical_diversity(candidates):
    tokens = []
    for candidate in candidates:
        tokens.extend(candidate.get("descendente", "").split())

    if not tokens:
        return 0.0

    return len(set(tokens)) / len(tokens)


def average_length(candidates):
    lengths = [candidate.get("tokens_descendente", 0) for candidate in candidates]
    valid_lengths = [length for length in lengths if length is not None]

    if not valid_lengths:
        return 0.0

    return mean(valid_lengths)


def summarize_generation_metrics(history, target_score):
    generation_summaries = []
    best_fitness_so_far = None
    evaluations_to_target_so_far = None

    for generation_name, generation_data in sorted(history.items(), key=_generation_sort_key):
        candidates = generation_data.get("all_candidates", generation_data.get("top_5", []))
        scores = [candidate["score_descendente"] for candidate in candidates]
        generation_best = max(scores) if scores else None

        if generation_best is not None:
            if best_fitness_so_far is None:
                best_fitness_so_far = generation_best
            else:
                best_fitness_so_far = max(best_fitness_so_far, generation_best)

        if target_score is not None and evaluations_to_target_so_far is None:
            for candidate in candidates:
                if candidate["score_descendente"] >= target_score:
                    evaluations_to_target_so_far = candidate.get("evaluation_index_descendente")
                    break

        generation_summaries.append(
            {
                "generation": generation_name,
                "best_fitness_generation": generation_best,
                "best_fitness_so_far": best_fitness_so_far,
                "evaluations_cumulative": generation_data.get("evaluations_cumulative"),
                "success_generation": (
                    generation_best >= target_score
                    if target_score is not None and generation_best is not None
                    else None
                ),
                "success_so_far": (
                    best_fitness_so_far >= target_score
                    if target_score is not None and best_fitness_so_far is not None
                    else None
                ),
                "evaluations_to_target_so_far": evaluations_to_target_so_far,
                "average_length_generation": average_length(candidates),
                "lexical_diversity_generation": lexical_diversity(candidates),
                "num_candidates_generation": len(candidates),
            }
        )

    return generation_summaries


def summarize_evaluation_metrics(history, target_score):
    evaluation_summaries = []
    best_fitness_so_far = None
    evaluations_to_target_so_far = None

    candidates_with_generation = sorted(
        iter_candidates_with_generation(history),
        key=lambda item: (
            item[1].get("evaluation_index_descendente", math.inf),
            item[0],
        ),
    )

    for generation_name, candidate in candidates_with_generation:
        evaluation_index = candidate.get("evaluation_index_descendente")
        score = candidate.get("score_descendente")

        if score is not None:
            if best_fitness_so_far is None:
                best_fitness_so_far = score
            else:
                best_fitness_so_far = max(best_fitness_so_far, score)

        if (
            target_score is not None
            and evaluations_to_target_so_far is None
            and score is not None
            and score >= target_score
        ):
            evaluations_to_target_so_far = evaluation_index

        evaluation_summaries.append(
            {
                "evaluation": evaluation_index,
                "generation": generation_name,
                "score": score,
                "best_fitness_so_far": best_fitness_so_far,
                "success_evaluation": (
                    score >= target_score
                    if target_score is not None and score is not None
                    else None
                ),
                "success_so_far": (
                    best_fitness_so_far >= target_score
                    if target_score is not None and best_fitness_so_far is not None
                    else None
                ),
                "evaluations_to_target_so_far": evaluations_to_target_so_far,
                "tokens_descendente": candidate.get("tokens_descendente"),
            }
        )

    return evaluation_summaries


def summarize_run(history, target_score):
    candidates = flatten_candidates(history)
    scores = [candidate["score_descendente"] for candidate in candidates]

    if not scores:
        return {
            "best_fitness": None,
            "success": False,
            "evaluations_to_target": None,
            "average_length_final": 0.0,
            "lexical_diversity": 0.0,
            "num_candidates": 0,
        }

    best_fitness = max(scores)
    success = best_fitness >= target_score if target_score is not None else None
    evaluations_to_target = None

    if target_score is not None:
        for candidate in candidates:
            if candidate["score_descendente"] >= target_score:
                evaluations_to_target = candidate.get("evaluation_index_descendente")
                break

    final_generation_candidates = []
    sorted_generations = sorted(history.items(), key=_generation_sort_key)
    if sorted_generations:
        final_generation_candidates = sorted_generations[-1][1].get(
            "all_candidates",
            sorted_generations[-1][1].get("top_5", []),
        )

    return {
        "best_fitness": best_fitness,
        "success": success,
        "evaluations_to_target": evaluations_to_target,
        "average_length_final": average_length(final_generation_candidates),
        "lexical_diversity": lexical_diversity(final_generation_candidates),
        "num_candidates": len(candidates),
    }


def summarize_runs(run_summaries):
    valid_best = [run["best_fitness"] for run in run_summaries if run["best_fitness"] is not None]
    valid_avg_length = [run["average_length_final"] for run in run_summaries]
    valid_lexical = [run["lexical_diversity"] for run in run_summaries]
    valid_success = [run["success"] for run in run_summaries if run["success"] is not None]
    valid_evals = [run["evaluations_to_target"] for run in run_summaries if run["evaluations_to_target"] is not None]

    return {
        "best_fitness_mean": mean(valid_best) if valid_best else None,
        "best_fitness_std": pstdev(valid_best) if len(valid_best) > 1 else 0.0,
        "success_rate": mean(valid_success) if valid_success else None,
        "evaluations_to_target_mean": mean(valid_evals) if valid_evals else None,
        "average_length_final_mean": mean(valid_avg_length) if valid_avg_length else 0.0,
        "lexical_diversity_mean": mean(valid_lexical) if valid_lexical else 0.0,
        "seed_stability": pstdev(valid_best) if len(valid_best) > 1 else 0.0,
        "num_runs": len(run_summaries),
    }
