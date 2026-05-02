import csv
import hashlib
import json
import random
import re
import subprocess
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import src.decoder as decoder


INDIVIDUAL_LOG_FIELDS = [
    "run_id",
    "experiment_id",
    "method",
    "seed",
    "target_class",
    "generation",
    "evaluation_id",
    "classifier_query_id",
    "candidate_id",
    "decoded_text",
    "normalized_text",
    "text_hash",
    "tokens",
    "token_ids",
    "num_tokens",
    "decode_success",
    "decode_error",
    "invalid_candidate",
    "invalid_reason",
    "target_class_score",
    "predicted_class",
    "predicted_class_score",
    "all_class_scores",
    "target_class_rank",
    "score_margin_target_vs_second_best",
    "objective_value",
    "embedding_path",
    "embedding_row_index",
    "embedding_source",
    "embedding_dim",
    "embedding_norm",
    "parent_ids",
    "parent1_id",
    "parent2_id",
    "operator_used",
    "mutation_type",
    "crossover_type",
    "mutation_applied",
    "crossover_applied",
    "num_tokens_added",
    "num_tokens_removed",
    "num_tokens_changed",
    "is_best_so_far",
    "best_so_far_score",
    "best_so_far_candidate_id",
    "best_so_far_text",
    "best_so_far_generation",
    "best_so_far_evaluation_id",
    "is_in_final_population",
    "is_elite",
    "rank_in_generation",
    "fitness_rank",
    "selected_as_parent",
    "survived_to_next_generation",
    "timestamp",
    "elapsed_time",
    "num_classifier_evaluations",
    "evaluation_budget",
]

GENERATION_SUMMARY_FIELDS = [
    "run_id",
    "method",
    "seed",
    "target_class",
    "generation",
    "start_evaluation_id",
    "end_evaluation_id",
    "num_evaluations_generation",
    "best_score_generation",
    "mean_score_generation",
    "std_score_generation",
    "best_objective_value_generation",
    "mean_objective_value_generation",
    "std_objective_value_generation",
    "best_so_far_score",
    "best_so_far_candidate_id",
    "num_unique_texts_generation",
    "num_invalid_candidates_generation",
    "mean_num_tokens_generation",
    "num_mutations",
    "num_crossovers",
    "num_elites",
]


def _get_ga_crossover_prob(config):
    return config.get("ga_crossover_prob", config.get("prob_crossover_embedding", 0.0))


def _get_ga_mutation_prob(config):
    return config.get("ga_mutation_prob", config.get("prob_mutacao_embedding", 0.0))


def _get_ga_embedding_mutation_std(config):
    return config.get(
        "ga_embedding_mutation_std",
        config.get("mutation_intensity_percent", 0.1),
    )


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_safe(value):
    try:
        json.dumps(value, default=_json_default)
        return value
    except TypeError:
        return str(value)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=_json_default)
        file.write("\n")


def _csv_safe_string(value):
    return value.replace("\x00", "\\u0000")


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict, tuple)):
        return _csv_safe_string(json.dumps(value, ensure_ascii=False, default=_json_default))
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return _csv_safe_string(value)
    return value


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
            quotechar='"',
            doublequote=True,
            escapechar="\\",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _normalize_text(text):
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.strip().lower()
    return re.sub(r"\s+", " ", normalized)


def _hash_text(normalized_text):
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _get_git_commit_hash(repo_root):
    if repo_root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _get_model_name(model, fallback):
    return getattr(getattr(model, "config", None), "_name_or_path", None) or fallback


def _get_label_name(resources, label_id):
    id2label = getattr(resources.classifier_model.config, "id2label", {}) or {}
    return id2label.get(label_id, str(label_id))


def _get_target_label_id(config):
    return int(config.get("classifier_target_label", 0))


def _get_target_class(resources, config):
    if config.get("classifier_target_class") is not None:
        return config["classifier_target_class"]
    if config.get("target_class") is not None:
        return config["target_class"]
    return _get_label_name(resources, _get_target_label_id(config))


def _get_evaluation_budget(config, population_size):
    method = config.get("_algorithm_name")
    if config.get("total_classifier_evaluation_budget") is not None:
        return config["total_classifier_evaluation_budget"]
    if method == "hill_climbing":
        return config.get(
            "max_evaluations",
            1 + config["num_geracoes"] * config["hill_climbing_neighbors"],
        )
    if method == "random_search":
        return config.get("random_search_classifier_evals")
    if method == "cma_es":
        return 1 + config["num_geracoes"] * config.get("cma_es_population_size", population_size)
    return population_size * (config["num_geracoes"] + 1)


def _build_run_dir(config, experiment_id, method, seed, timestamp_start):
    output_file = config.get("_resolved_output_file") or config.get("output_file", "historico.json")
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = (config.get("_base_dir", Path.cwd()) / output_path).resolve()
    timestamp_slug = timestamp_start.replace(":", "").replace("-", "").replace("T", "_")
    run_id = f"{experiment_id}_{method}_seed{seed}_{timestamp_slug}"
    run_dir = output_path.parent / "runs" / run_id
    return run_id, run_dir


def _method_hyperparameters(config, population_size):
    return {
        "population_size": population_size,
        "num_generations": config["num_geracoes"],
        "crossover_probability": _get_ga_crossover_prob(config),
        "mutation_probability": _get_ga_mutation_prob(config),
        "mutation_intensity": _get_ga_embedding_mutation_std(config),
        "add_token_probability": 0.0,
        "remove_token_probability": 0.0,
        "tournament_size": config.get("tournament_size", 2),
        "elitism_size": 0,
        "max_num_tokens": config.get("max_num_tokens"),
        "min_num_tokens": config.get("min_num_tokens"),
        "decoder_strategy": config.get("decoder_strategy"),
        "decoder_family": config.get("decoder_family"),
        "decoder_top_k": config.get("decoder_top_k"),
        "ga_embedding_mutation_std": _get_ga_embedding_mutation_std(config),
        "ga_crossover_prob": _get_ga_crossover_prob(config),
        "ga_mutation_prob": _get_ga_mutation_prob(config),
        "classifier_evaluation_budget": config.get("classifier_evaluation_budget"),
        "classifier_evaluation_budget_kind": config.get("classifier_evaluation_budget_kind"),
        "hill_climbing_neighbors": config.get("hill_climbing_neighbors"),
        "hill_climbing_restart": config.get("hill_climbing_restart"),
        "max_evaluations": config.get("max_evaluations"),
        "cma_es_population_size": config.get("cma_es_population_size"),
        "cma_es_sigma": config.get("cma_es_sigma"),
        "cma_es_elite_ratio": config.get("cma_es_elite_ratio"),
        "cma_es_cov_eps": config.get("cma_es_cov_eps"),
        "random_search_classifier_evals": config.get("random_search_classifier_evals"),
        "random_search_batch_size": config.get("random_search_batch_size"),
    }


def _should_persist_detailed_run_artifacts(config):
    if config.get("save_detailed_run_artifacts") is not None:
        return bool(config.get("save_detailed_run_artifacts"))
    return not bool(config.get("is_hyperparameter_selection", False))


def _tokenize_population_for_roberta(resources, population):
    return resources.tokenizer(
        population,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=resources.model_max_length,
    ).to(resources.device)


def _population_to_embeddings(resources, population):
    inputs = _tokenize_population_for_roberta(resources, population)
    with torch.no_grad():
        outputs = resources.model.roberta(**inputs)
    return [embedding.detach().clone() for embedding in outputs.last_hidden_state]


def _embedding_to_text(resources, config, embedding):
    return decoder.decode_embeddings_to_text(resources, config, embedding.unsqueeze(0))


def _classify_texts(resources, config, texts):
    target_label_id = _get_target_label_id(config)
    batch_size = config["speedup_factor"]
    details = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = resources.classifier_tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {key: value.to(resources.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = resources.classifier_model(**inputs)
            scores_tensor = torch.nn.functional.softmax(outputs.logits, dim=1)

        for scores in scores_tensor.cpu().tolist():
            predicted_label_id = max(range(len(scores)), key=lambda idx: scores[idx])
            target_score = scores[target_label_id]
            other_scores = [
                score for label_id, score in enumerate(scores) if label_id != target_label_id
            ]
            second_best_other = max(other_scores) if other_scores else 0.0
            ranked_label_ids = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
            details.append(
                {
                    "target_class_score": target_score,
                    "predicted_class": _get_label_name(resources, predicted_label_id),
                    "predicted_class_score": scores[predicted_label_id],
                    "all_class_scores": scores,
                    "target_class_rank": ranked_label_ids.index(target_label_id) + 1,
                    "score_margin_target_vs_second_best": target_score - second_best_other,
                    "objective_value": target_score,
                }
            )

    return details


class GARunLogger:
    def __init__(self, resources, config, population_size, initial_population):
        self.resources = resources
        self.config = config
        self.population_size = population_size
        self.initial_population = list(initial_population)
        algorithm_name = config.get("_algorithm_name", "vanilla_ga")
        self.method = "ebie" if algorithm_name == "genetic" else algorithm_name
        self.experiment_id = config.get("experiment_name", "unknown_experiment")
        self.seed = config.get("_current_seed", config.get("random_seed"))
        self.target_class = _get_target_class(resources, config)
        self.target_label_id = _get_target_label_id(config)
        self.evaluation_budget = _get_evaluation_budget(config, population_size)
        self.timestamp_start = _now_iso()
        self.start_time = time.time()
        self.persist_detailed_artifacts = _should_persist_detailed_run_artifacts(config)
        self.run_id, self.run_dir = _build_run_dir(
            config,
            self.experiment_id,
            self.method,
            self.seed,
            self.timestamp_start,
        )
        self.individual_log_path = self.run_dir / "individual_log.csv"
        self.generation_summary_path = self.run_dir / "generation_summary.csv"
        self.run_summary_path = self.run_dir / "run_summary.json"
        self.config_path = self.run_dir / "config.json"
        self.embeddings_path = self.run_dir / "embeddings.npz"
        self.rows = []
        self.generation_rows = []
        self.embedding_rows = []
        self.candidate_counter = 0
        self.evaluation_counter = 0
        self.best_score = None
        self.best_objective_value = None
        self.best_candidate_id = None
        self.best_text = None
        self.best_generation = None
        self.best_evaluation_id = None
        self.best_embedding_row_index = None
        self.selected_parent_ids = set()
        self.final_population_candidate_ids = []
        self.config_payload = self._build_config_payload(timestamp_end=None)
        if self.persist_detailed_artifacts:
            _write_json(self.config_path, self.config_payload)

    def _build_config_payload(self, timestamp_end):
        repo_root = self.config.get("_base_dir")
        classifier_name = self.config.get(
            "classifier_model_name",
            self.config.get("emotion_model_name"),
        )
        embedding_model = self.config.get("roberta_model_name")
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "method": self.method,
            "algorithm_name": self.config.get("_algorithm_name", self.method),
            "seed": self.seed,
            "target_class": self.target_class,
            "classifier_name": classifier_name,
            "classifier_version": _get_model_name(self.resources.classifier_model, classifier_name),
            "classifier_path": classifier_name,
            "decoder_name": self.config.get("decoder_config_name", self.config.get("decoder_family")),
            "decoder_version": self.config.get("decoder_family"),
            "embedding_model": embedding_model,
            "embedding_model_version": _get_model_name(self.resources.model, embedding_model),
            "tokenizer_name": embedding_model,
            "tokenizer_version": _get_model_name(self.resources.model, embedding_model),
            "evaluation_budget": self.evaluation_budget,
            "success_thresholds": [self.config.get("success_target_score")],
            "method_hyperparameters": _method_hyperparameters(self.config, self.population_size),
            "initialization_strategy": self.config.get("initialization_mode"),
            "timestamp_start": self.timestamp_start,
            "timestamp_end": timestamp_end,
            "git_commit_hash": _get_git_commit_hash(repo_root),
            "individual_log_path": str(self.individual_log_path),
            "embeddings_path": str(self.embeddings_path),
            "run_summary_path": str(self.run_summary_path),
            "generation_summary_path": str(self.generation_summary_path),
            "initial_text": self.initial_population[0] if self.initial_population else None,
            "initial_texts": self.initial_population,
            "initial_token_ids": (
                self.resources.tokenizer.encode(self.initial_population[0], add_special_tokens=False)
                if self.initial_population else []
            ),
            "raw_config": {
                key: _json_safe(value)
                for key, value in self.config.items()
                if not key.startswith("_")
            },
        }

    def next_candidate_id(self):
        self.candidate_counter += 1
        return f"{self.run_id}_cand_{self.candidate_counter:08d}"

    def append_embedding(self, embedding):
        flattened = embedding.detach().cpu().float().numpy().reshape(-1).astype(np.float32)
        row_index = len(self.embedding_rows)
        self.embedding_rows.append(flattened)
        return row_index, int(flattened.shape[0]), float(np.linalg.norm(flattened))

    def next_evaluation_id(self):
        self.evaluation_counter += 1
        return self.evaluation_counter

    def update_best(self, row):
        score = row["target_class_score"]
        if score is None:
            return False
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_objective_value = row["objective_value"]
            self.best_candidate_id = row["candidate_id"]
            self.best_text = row["decoded_text"]
            self.best_generation = row["generation"]
            self.best_evaluation_id = row["evaluation_id"]
            self.best_embedding_row_index = row["embedding_row_index"]
            return True
        return False

    def add_rows(self, rows):
        self.rows.extend(rows)
        self.flush_individual_log()

    def add_generation_summary(self, row):
        self.generation_rows.append(row)
        if self.persist_detailed_artifacts:
            _write_csv(self.generation_summary_path, self.generation_rows, GENERATION_SUMMARY_FIELDS)

    def flush_individual_log(self):
        if self.persist_detailed_artifacts:
            _write_csv(self.individual_log_path, self.rows, INDIVIDUAL_LOG_FIELDS)

    def flush_embeddings(self):
        if not self.persist_detailed_artifacts:
            return
        if not self.embedding_rows:
            embeddings = np.empty((0, 0), dtype=np.float32)
        else:
            max_dim = max(row.shape[0] for row in self.embedding_rows)
            padded_rows = []
            for row in self.embedding_rows:
                if row.shape[0] == max_dim:
                    padded_rows.append(row)
                else:
                    padded = np.zeros(max_dim, dtype=np.float32)
                    padded[: row.shape[0]] = row
                    padded_rows.append(padded)
            embeddings = np.stack(padded_rows, axis=0)
            for row in self.rows:
                row["embedding_dim"] = max_dim
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.embeddings_path, embeddings=embeddings)

    def mark_selected_parent(self, candidate_id):
        self.selected_parent_ids.add(candidate_id)

    def finalize_population_fields(self, final_population_candidate_ids):
        self.final_population_candidate_ids = list(final_population_candidate_ids)
        final_ids = set(final_population_candidate_ids)
        for row in self.rows:
            row["selected_as_parent"] = row["candidate_id"] in self.selected_parent_ids
            row["is_in_final_population"] = row["candidate_id"] in final_ids

    def finalize(self, final_candidate_details, num_generations_completed):
        self.finalize_population_fields([item["candidate_id"] for item in final_candidate_details])
        self.flush_embeddings()
        self.flush_individual_log()
        timestamp_end = _now_iso()
        wall_clock_time = time.time() - self.start_time
        final_scores = [item["target_class_score"] for item in final_candidate_details]
        final_objectives = [item["objective_value"] for item in final_candidate_details]
        best_embedding_path = str(self.embeddings_path) if self.best_candidate_id else None
        summary = {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "method": self.method,
            "seed": self.seed,
            "target_class": self.target_class,
            "evaluation_budget": self.evaluation_budget,
            "num_classifier_evaluations": self.evaluation_counter,
            "num_generations_completed": num_generations_completed,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": timestamp_end,
            "wall_clock_time_seconds": wall_clock_time,
            "best_candidate_id": self.best_candidate_id,
            "best_text": self.best_text,
            "best_score_seen": self.best_score,
            "best_objective_value_seen": self.best_objective_value,
            "evaluation_of_best": self.best_evaluation_id,
            "generation_of_best": self.best_generation,
            "best_embedding_path": best_embedding_path,
            "best_embedding_row_index": self.best_embedding_row_index,
            "final_population_candidate_ids": self.final_population_candidate_ids,
            "final_best_score": max(final_scores) if final_scores else None,
            "final_mean_score": float(np.mean(final_scores)) if final_scores else None,
            "final_std_score": float(np.std(final_scores)) if final_scores else None,
            "final_best_objective_value": max(final_objectives) if final_objectives else None,
            "cma_stop_reason": None,
            "hill_stop_reason": None,
            "cma_stop_reason": (
                "num_generations_completed" if self.method == "cma_es" else None
            ),
            "hill_stop_reason": (
                "evaluation_budget_exhausted" if self.method == "hill_climbing" else None
            ),
            "ga_stop_reason": (
                "num_generations_completed"
                if self.method in {"vanilla_ga", "ebie", "genetic"}
                else None
            ),
            "random_search_stop_reason": (
                "evaluation_budget_exhausted" if self.method == "random_search" else None
            ),
            "initial_embedding_path": str(self.embeddings_path),
            "initial_embedding_row_index": 0 if self.embedding_rows else None,
            "initial_target_class_score": self.rows[0]["target_class_score"] if self.rows else None,
            "initial_predicted_class": self.rows[0]["predicted_class"] if self.rows else None,
            "initial_all_class_scores": self.rows[0]["all_class_scores"] if self.rows else None,
            "validation": self.validate(),
        }
        if self.persist_detailed_artifacts:
            _write_json(self.run_summary_path, summary)
        self.config_payload = self._build_config_payload(timestamp_end=timestamp_end)
        self.config_payload.update(
            {
                "initial_embedding_path": str(self.embeddings_path),
                "initial_embedding_row_index": 0 if self.embedding_rows else None,
                "initial_target_class_score": summary["initial_target_class_score"],
                "initial_predicted_class": summary["initial_predicted_class"],
                "initial_all_class_scores": summary["initial_all_class_scores"],
                "best_candidate_id": self.best_candidate_id,
                "best_embedding_path": best_embedding_path,
                "best_embedding_row_index": self.best_embedding_row_index,
            }
        )
        if self.persist_detailed_artifacts:
            _write_json(self.config_path, self.config_payload)

    def validate(self):
        errors = []
        valid_rows = [row for row in self.rows if row["evaluation_id"] is not None]
        evaluation_ids = [row["evaluation_id"] for row in valid_rows]
        if evaluation_ids != sorted(evaluation_ids):
            errors.append("evaluation_id is not monotonic increasing")
        if any(row["classifier_query_id"] is None for row in valid_rows):
            errors.append("classifier_query_id missing for at least one classifier call")
        if len(valid_rows) != self.evaluation_counter:
            errors.append("individual_log row count does not match num_classifier_evaluations")
        scores = [row["target_class_score"] for row in valid_rows]
        if scores and self.best_score != max(scores):
            errors.append("best_score_seen does not match max target_class_score")
        if scores:
            first_best = next(row["evaluation_id"] for row in valid_rows if row["target_class_score"] == max(scores))
            if self.best_evaluation_id != first_best:
                errors.append("evaluation_of_best is not the first evaluation with best_score_seen")
        for row in valid_rows:
            all_scores = row["all_class_scores"]
            if not all_scores:
                errors.append(f"all_class_scores missing for {row['candidate_id']}")
                continue
            if row["target_class_score"] != all_scores[self.target_label_id]:
                errors.append(f"target_class_score mismatch for {row['candidate_id']}")
            predicted_idx = max(range(len(all_scores)), key=lambda idx: all_scores[idx])
            if row["predicted_class"] != _get_label_name(self.resources, predicted_idx):
                errors.append(f"predicted_class mismatch for {row['candidate_id']}")
            if row["objective_value"] is None:
                errors.append(f"objective_value missing for {row['candidate_id']}")
        embedding_dim = None
        for row in self.rows:
            row_index = row["embedding_row_index"]
            if row_index is None or row_index < 0 or row_index >= len(self.embedding_rows):
                errors.append(f"embedding_row_index invalid for {row['candidate_id']}")
            if embedding_dim is None:
                embedding_dim = row["embedding_dim"]
            elif row["embedding_dim"] != embedding_dim:
                errors.append("embedding_dim is not consistent within the run")
        return {"passed": not errors, "errors": errors}


def torneio(population_size, fitness, tournament_size=2):
    tournament_size = min(tournament_size, population_size)
    selected = random.sample(range(population_size), tournament_size)
    return max(selected, key=lambda idx: fitness[idx])


def crossover_aritmetico(config, pai1_embedding, pai2_embedding):
    if random.random() >= _get_ga_crossover_prob(config):
        parent = pai1_embedding if random.random() < 0.5 else pai2_embedding
        return parent.clone(), False, None

    alpha = random.random()
    filho = alpha * pai1_embedding + (1.0 - alpha) * pai2_embedding
    return filho, True, alpha


def mutacao_gaussiana(config, embedding):
    if random.random() >= _get_ga_mutation_prob(config):
        return embedding, False

    sigma = _get_ga_embedding_mutation_std(config)
    ruido = torch.randn_like(embedding) * sigma
    return embedding + ruido, True


def gerar_descendente_embedding(config, pai1_embedding, pai2_embedding):
    filho, crossover_aplicado, crossover_alpha = crossover_aritmetico(
        config,
        pai1_embedding,
        pai2_embedding,
    )
    filho, mutacao_aplicada = mutacao_gaussiana(config, filho)
    return filho, crossover_aplicado, crossover_alpha, mutacao_aplicada


def _decode_embeddings(resources, config, embeddings):
    decoded = []
    for embedding in embeddings:
        try:
            decoded.append(
                {
                    "decoded_text": _embedding_to_text(resources, config, embedding),
                    "decode_success": True,
                    "decode_error": None,
                }
            )
        except Exception as exc:
            decoded.append(
                {
                    "decoded_text": "",
                    "decode_success": False,
                    "decode_error": str(exc),
                }
            )
    return decoded


def _build_operator_used(record):
    if record.get("operator_used") == "initialization":
        return "initialization"
    if record.get("operator_used") and record.get("operator_used") != "variation":
        return record["operator_used"]
    operators = []
    if record.get("crossover_applied"):
        operators.append("crossover")
    if record.get("mutation_applied"):
        operators.append("mutation")
    if not operators:
        return "copy"
    return "+".join(operators)


def _evaluate_and_log_embeddings(logger, generation, embeddings, operator_records):
    decoded_records = _decode_embeddings(logger.resources, logger.config, embeddings)
    return _log_candidate_records(logger, generation, embeddings, decoded_records, operator_records)


def evaluate_and_log_embeddings(logger, generation, embeddings, operator_records):
    return _evaluate_and_log_embeddings(logger, generation, embeddings, operator_records)


def evaluate_and_log_texts(
    logger,
    generation,
    texts,
    operator_records,
    embedding_source="sentence_embedding",
):
    embeddings = _population_to_embeddings(logger.resources, texts)
    decoded_records = [
        {
            "decoded_text": text,
            "decode_success": True,
            "decode_error": None,
        }
        for text in texts
    ]
    normalized_records = []
    for record in operator_records:
        normalized_record = dict(record)
        normalized_record.setdefault("embedding_source", embedding_source)
        normalized_records.append(normalized_record)
    return _log_candidate_records(
        logger,
        generation,
        embeddings,
        decoded_records,
        normalized_records,
    )


def evaluate_and_log_decoded_embeddings(
    logger,
    generation,
    texts,
    embeddings,
    operator_records,
    embedding_source="decoder_embedding",
):
    decoded_records = [
        {
            "decoded_text": text,
            "decode_success": True,
            "decode_error": None,
        }
        for text in texts
    ]
    normalized_records = []
    for record in operator_records:
        normalized_record = dict(record)
        normalized_record.setdefault("embedding_source", embedding_source)
        normalized_records.append(normalized_record)
    return _log_candidate_records(
        logger,
        generation,
        embeddings,
        decoded_records,
        normalized_records,
    )


def build_initial_operator_records(count):
    return _initial_operator_records(count)


def _log_candidate_records(logger, generation, embeddings, decoded_records, operator_records):
    valid_indices = [
        index for index, decoded in enumerate(decoded_records)
        if decoded["decode_success"]
    ]
    classifier_details_by_index = {}
    if valid_indices:
        valid_texts = [decoded_records[index]["decoded_text"] for index in valid_indices]
        classifier_details = _classify_texts(logger.resources, logger.config, valid_texts)
        classifier_details_by_index = dict(zip(valid_indices, classifier_details, strict=True))

    rows = []
    candidate_details = []
    for index, (embedding, decoded, operator_record) in enumerate(
        zip(embeddings, decoded_records, operator_records, strict=True)
    ):
        candidate_id = logger.next_candidate_id()
        embedding_row_index, embedding_dim, embedding_norm = logger.append_embedding(embedding)
        decoded_text = decoded["decoded_text"]
        normalized_text = _normalize_text(decoded_text)
        token_ids = logger.resources.tokenizer.encode(decoded_text, add_special_tokens=False)
        tokens = logger.resources.tokenizer.convert_ids_to_tokens(token_ids)
        is_valid = decoded["decode_success"]

        if is_valid:
            evaluation_id = logger.next_evaluation_id()
            classifier_query_id = evaluation_id
            classifier_details = classifier_details_by_index[index]
        else:
            evaluation_id = None
            classifier_query_id = None
            classifier_details = {
                "target_class_score": 0.0,
                "predicted_class": None,
                "predicted_class_score": None,
                "all_class_scores": [],
                "target_class_rank": None,
                "score_margin_target_vs_second_best": None,
                "objective_value": 0.0,
            }

        row = {
            "run_id": logger.run_id,
            "experiment_id": logger.experiment_id,
            "method": logger.method,
            "seed": logger.seed,
            "target_class": logger.target_class,
            "generation": generation,
            "evaluation_id": evaluation_id,
            "classifier_query_id": classifier_query_id,
            "candidate_id": candidate_id,
            "decoded_text": decoded_text,
            "normalized_text": normalized_text,
            "text_hash": _hash_text(normalized_text),
            "tokens": tokens,
            "token_ids": token_ids,
            "num_tokens": len(token_ids),
            "decode_success": decoded["decode_success"],
            "decode_error": decoded["decode_error"],
            "invalid_candidate": not is_valid,
            "invalid_reason": None if is_valid else decoded["decode_error"],
            "embedding_path": str(logger.embeddings_path),
            "embedding_row_index": embedding_row_index,
            "embedding_source": operator_record.get("embedding_source", "search_embedding"),
            "embedding_dim": embedding_dim,
            "embedding_norm": embedding_norm,
            "parent_ids": operator_record.get("parent_ids", []),
            "parent1_id": operator_record.get("parent1_id"),
            "parent2_id": operator_record.get("parent2_id"),
            "operator_used": _build_operator_used(operator_record),
            "mutation_type": operator_record.get("mutation_type"),
            "crossover_type": operator_record.get("crossover_type"),
            "mutation_applied": operator_record.get("mutation_applied", False),
            "crossover_applied": operator_record.get("crossover_applied", False),
            "num_tokens_added": operator_record.get("num_tokens_added", 0),
            "num_tokens_removed": operator_record.get("num_tokens_removed", 0),
            "num_tokens_changed": operator_record.get("num_tokens_changed", 0),
            "is_best_so_far": False,
            "is_in_final_population": False,
            "is_elite": False,
            "rank_in_generation": None,
            "fitness_rank": None,
            "selected_as_parent": False,
            "survived_to_next_generation": True,
            "timestamp": _now_iso(),
            "elapsed_time": time.time() - logger.start_time,
            "num_classifier_evaluations": logger.evaluation_counter,
            "evaluation_budget": logger.evaluation_budget,
        }
        row.update(classifier_details)
        row["is_best_so_far"] = logger.update_best(row)
        row["best_so_far_score"] = logger.best_score
        row["best_so_far_candidate_id"] = logger.best_candidate_id
        row["best_so_far_text"] = logger.best_text
        row["best_so_far_generation"] = logger.best_generation
        row["best_so_far_evaluation_id"] = logger.best_evaluation_id
        rows.append(row)
        candidate_details.append(
            {
                "candidate_id": candidate_id,
                "embedding": embedding,
                "decoded_text": decoded_text,
                "target_class_score": row["target_class_score"],
                "objective_value": row["objective_value"],
                "evaluation_id": evaluation_id,
                "tokens_descendente": row["num_tokens"],
                "row": row,
            }
        )

    _assign_generation_ranks(candidate_details)
    logger.add_rows(rows)
    logger.add_generation_summary(
        _build_generation_summary(logger, generation, candidate_details)
    )
    return candidate_details


def _assign_generation_ranks(candidate_details):
    ranked = sorted(
        candidate_details,
        key=lambda item: item["objective_value"],
        reverse=True,
    )
    for rank, item in enumerate(ranked, start=1):
        item["row"]["rank_in_generation"] = rank
        item["row"]["fitness_rank"] = rank


def _build_generation_summary(logger, generation, candidate_details):
    valid_details = [
        item for item in candidate_details if item["evaluation_id"] is not None
    ]
    scores = [item["target_class_score"] for item in valid_details]
    objectives = [item["objective_value"] for item in valid_details]
    num_tokens = [item["tokens_descendente"] for item in candidate_details]
    evaluation_ids = [item["evaluation_id"] for item in valid_details]
    rows = [item["row"] for item in candidate_details]
    return {
        "run_id": logger.run_id,
        "method": logger.method,
        "seed": logger.seed,
        "target_class": logger.target_class,
        "generation": generation,
        "start_evaluation_id": min(evaluation_ids) if evaluation_ids else None,
        "end_evaluation_id": max(evaluation_ids) if evaluation_ids else None,
        "num_evaluations_generation": len(evaluation_ids),
        "best_score_generation": max(scores) if scores else None,
        "mean_score_generation": float(np.mean(scores)) if scores else None,
        "std_score_generation": float(np.std(scores)) if scores else None,
        "best_objective_value_generation": max(objectives) if objectives else None,
        "mean_objective_value_generation": float(np.mean(objectives)) if objectives else None,
        "std_objective_value_generation": float(np.std(objectives)) if objectives else None,
        "best_so_far_score": logger.best_score,
        "best_so_far_candidate_id": logger.best_candidate_id,
        "num_unique_texts_generation": len({item["row"]["text_hash"] for item in candidate_details}),
        "num_invalid_candidates_generation": sum(row["invalid_candidate"] for row in rows),
        "mean_num_tokens_generation": float(np.mean(num_tokens)) if num_tokens else None,
        "num_mutations": sum(row["mutation_applied"] for row in rows),
        "num_crossovers": sum(row["crossover_applied"] for row in rows),
        "num_elites": 0,
    }


def _candidate_info(
    resources,
    candidate_detail,
    pai1_detail,
    pai2_detail,
    crossover_aplicado,
    crossover_alpha,
    mutacao_aplicada,
):
    return {
        "candidate_id": candidate_detail["candidate_id"],
        "descendente": candidate_detail["decoded_text"],
        "score_descendente": candidate_detail["target_class_score"],
        "objective_value": candidate_detail["objective_value"],
        "tokens_descendente": candidate_detail["tokens_descendente"],
        "pai1": pai1_detail["decoded_text"],
        "pai1_id": pai1_detail["candidate_id"],
        "score_pai1": pai1_detail["target_class_score"],
        "tokens_pai1": len(resources.tokenizer.tokenize(pai1_detail["decoded_text"])),
        "evaluation_index_pai1": pai1_detail["evaluation_id"],
        "pai2": pai2_detail["decoded_text"],
        "pai2_id": pai2_detail["candidate_id"],
        "score_pai2": pai2_detail["target_class_score"],
        "tokens_pai2": len(resources.tokenizer.tokenize(pai2_detail["decoded_text"])),
        "evaluation_index_pai2": pai2_detail["evaluation_id"],
        "evaluation_index_descendente": candidate_detail["evaluation_id"],
        "crossover_applied": crossover_aplicado,
        "crossover_alpha": crossover_alpha,
        "mutation_applied": mutacao_aplicada,
        "elitism": False,
    }


def _initial_operator_records(count):
    return [
        {
            "parent_ids": [],
            "parent1_id": None,
            "parent2_id": None,
            "operator_used": "initialization",
            "mutation_type": None,
            "crossover_type": None,
            "mutation_applied": False,
            "crossover_applied": False,
        }
        for _ in range(count)
    ]


def vanilla_ga(resources, config, populacao_inicial, vocabulary=None):
    del vocabulary

    if not populacao_inicial:
        return {}

    populacao_embeddings = _population_to_embeddings(resources, populacao_inicial)
    population_size = len(populacao_embeddings)
    logger = GARunLogger(resources, config, population_size, populacao_inicial)
    populacao_details = _evaluate_and_log_embeddings(
        logger,
        generation=0,
        embeddings=populacao_embeddings,
        operator_records=_initial_operator_records(population_size),
    )
    fitness = [candidate["objective_value"] for candidate in populacao_details]
    tournament_size = config.get("tournament_size", 2)
    historico_geracoes = {}

    for geracao in tqdm(range(config["num_geracoes"]), desc="Evoluindo Embedding Vanilla GA"):
        nova_populacao_embeddings = []
        parent_records = []

        while len(nova_populacao_embeddings) < population_size:
            pai1_idx = torneio(population_size, fitness, tournament_size)
            pai2_idx = torneio(population_size, fitness, tournament_size)
            pai1_detail = populacao_details[pai1_idx]
            pai2_detail = populacao_details[pai2_idx]
            logger.mark_selected_parent(pai1_detail["candidate_id"])
            logger.mark_selected_parent(pai2_detail["candidate_id"])
            filho, crossover_aplicado, crossover_alpha, mutacao_aplicada = (
                gerar_descendente_embedding(
                    config,
                    populacao_embeddings[pai1_idx],
                    populacao_embeddings[pai2_idx],
                )
            )

            nova_populacao_embeddings.append(filho)
            parent_records.append(
                {
                    "pai1_idx": pai1_idx,
                    "pai2_idx": pai2_idx,
                    "parent_ids": [pai1_detail["candidate_id"], pai2_detail["candidate_id"]],
                    "parent1_id": pai1_detail["candidate_id"],
                    "parent2_id": pai2_detail["candidate_id"],
                    "operator_used": "variation",
                    "mutation_type": "gaussian_embedding" if mutacao_aplicada else None,
                    "crossover_type": "arithmetic_convex" if crossover_aplicado else None,
                    "mutation_applied": mutacao_aplicada,
                    "crossover_applied": crossover_aplicado,
                    "crossover_alpha": crossover_alpha,
                }
            )

        nova_populacao_details = _evaluate_and_log_embeddings(
            logger,
            generation=geracao + 1,
            embeddings=nova_populacao_embeddings,
            operator_records=parent_records,
        )

        descendentes_info = []
        for candidate_detail, parent_record in zip(
            nova_populacao_details,
            parent_records,
            strict=True,
        ):
            descendentes_info.append(
                _candidate_info(
                    resources,
                    candidate_detail,
                    populacao_details[parent_record["pai1_idx"]],
                    populacao_details[parent_record["pai2_idx"]],
                    parent_record["crossover_applied"],
                    parent_record["crossover_alpha"],
                    parent_record["mutation_applied"],
                )
            )

        top_5_descendentes = sorted(
            descendentes_info,
            key=lambda item: item["score_descendente"],
            reverse=True,
        )[:5]
        historico_geracoes[f"geracao_{geracao + 1}"] = {
            "top_5": top_5_descendentes,
            "all_candidates": descendentes_info,
            "evaluations_cumulative": logger.evaluation_counter,
            "elitism": False,
        }

        populacao_embeddings = nova_populacao_embeddings
        populacao_details = nova_populacao_details
        fitness = [candidate["objective_value"] for candidate in populacao_details]

    logger.finalize(populacao_details, config["num_geracoes"])
    return historico_geracoes
