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


def lexical_diversity(candidates):
    tokens = []
    for candidate in candidates:
        tokens.extend(candidate.get("descendente", "").split())

    if not tokens:
        return 0.0

    return len(set(tokens)) / len(tokens)


def summarize_run(history, target_score):
    candidates = flatten_candidates(history)
    scores = [candidate["score_descendente"] for candidate in candidates]

    if not scores:
        return {
            "best_fitness": None,
            "success": False,
            "evaluations_to_target": None,
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

    return {
        "best_fitness": best_fitness,
        "success": success,
        "evaluations_to_target": evaluations_to_target,
        "lexical_diversity": lexical_diversity(candidates),
        "num_candidates": len(candidates),
    }


def summarize_runs(run_summaries):
    valid_best = [run["best_fitness"] for run in run_summaries if run["best_fitness"] is not None]
    valid_lexical = [run["lexical_diversity"] for run in run_summaries]
    valid_success = [run["success"] for run in run_summaries if run["success"] is not None]
    valid_evals = [run["evaluations_to_target"] for run in run_summaries if run["evaluations_to_target"] is not None]

    return {
        "best_fitness_mean": mean(valid_best) if valid_best else None,
        "best_fitness_std": pstdev(valid_best) if len(valid_best) > 1 else 0.0,
        "success_rate": mean(valid_success) if valid_success else None,
        "evaluations_to_target_mean": mean(valid_evals) if valid_evals else None,
        "lexical_diversity_mean": mean(valid_lexical) if valid_lexical else 0.0,
        "seed_stability": pstdev(valid_best) if len(valid_best) > 1 else 0.0,
        "num_runs": len(run_summaries),
    }
