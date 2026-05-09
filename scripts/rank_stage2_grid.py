import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Rank Stage 2 EBIE grid-search outputs.",)
    parser.add_argument("outputs_dir", help="Directory containing the Stage 2 output JSON files.",)
    parser.add_argument("--top", type=int, default=10, help="Number of ranked configurations to print.",)
    parser.add_argument("--save-json", action="store_true", help="Save ranking_stage2_grid.json and best_stage2_grid.json in outputs_dir.",)
    return parser.parse_args()


def build_sort_key(row):
    success_rate = row["success_rate"]
    evaluations_to_target_mean = row["evaluations_to_target_mean"]
    best_fitness_mean = row["best_fitness_mean"]
    best_fitness_std = row["best_fitness_std"]

    return (
        -1 if success_rate is None else -success_rate,
        float("inf") if evaluations_to_target_mean is None else evaluations_to_target_mean,
        1 if best_fitness_mean is None else -best_fitness_mean,
        float("inf") if best_fitness_std is None else best_fitness_std,
        row["experiment_name"],
    )


def load_rows(outputs_dir):
    rows = []
    pattern = "historico_completo_genetic_*_ebie_stage2_*.json"
    for path in sorted(outputs_dir.glob(pattern)):
        data = json.loads(path.read_text(encoding="utf-8"))
        config = data["config"]
        summary = data["summary"]
        rows.append({"experiment_name":data["experiment_name"], "file":str(path), "populacao_inicial":config["populacao_inicial"], "num_geracoes":config["num_geracoes"], "prob_mutacao_embedding":config["prob_mutacao_embedding"], "mutation_intensity_percent":config["mutation_intensity_percent"], "prob_crossover_embedding":config["prob_crossover_embedding"], "success_rate":summary["success_rate"], "evaluations_to_target_mean":summary["evaluations_to_target_mean"], "best_fitness_mean":summary["best_fitness_mean"], "best_fitness_std":summary["best_fitness_std"], "seed_stability":summary["seed_stability"], "num_runs":summary["num_runs"],})
    return rows


def format_metric(value, precision=9):
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def save_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",)


def main():
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    rows = load_rows(outputs_dir)

    if not rows:
        raise SystemExit(f"No result files found in {outputs_dir}")

    ranked = sorted(rows, key=build_sort_key)
    if args.save_json:
        save_json(outputs_dir / "ranking_stage2_grid.json", ranked)
        save_json(outputs_dir / "best_stage2_grid.json", ranked[0])

    print("Best configuration:")
    print(json.dumps(ranked[0], ensure_ascii=False, indent=2))

    print("\nTop configurations:")
    for index, row in enumerate(ranked[: args.top], start=1):
        print(f"{index:02d}. {row['experiment_name']} | " f"pop={row['populacao_inicial']} gen={row['num_geracoes']} " f"pmut={row['prob_mutacao_embedding']} " f"mint={row['mutation_intensity_percent']} " f"pcross={row['prob_crossover_embedding']} " f"padd={row['prob_add_random_token']} " f"prem={row['prob_remover_token']} " f"success_rate={row['success_rate']} " f"evals_to_target={row['evaluations_to_target_mean']} " f"best_fitness_mean={format_metric(row['best_fitness_mean'])} " f"std={format_metric(row['best_fitness_std'])} " f"avg_len={format_metric(row['average_length_final_mean'], 3)} " f"lexical_diversity={format_metric(row['lexical_diversity_mean'], 3)}")


if __name__ == "__main__":
    main()
