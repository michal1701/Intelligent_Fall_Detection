"""
Analyze and visualize Table 5 results from SAFE replication.

Reads CSV produced by main.py:
    results_table5/dl_results_spectrogram_cv.csv

Outputs:
- console summary tables (sorted, best configs, quick insights)
- plots saved to results_table5/analysis/
"""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import pandas as pd


METRICS = ["accuracy", "precision", "recall", "f1_score"]


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _pm(mean: float, std: float) -> str:
    return f"{mean * 100:.2f} +/- {std * 100:.2f}"


def print_ranked_table(df: pd.DataFrame) -> None:
    ranked = df.sort_values("accuracy_mean", ascending=False).copy()
    ranked["Accuracy"] = ranked.apply(lambda r: _pm(r["accuracy_mean"], r["accuracy_std"]), axis=1)
    ranked["Precision"] = ranked.apply(lambda r: _pm(r["precision_mean"], r["precision_std"]), axis=1)
    ranked["Recall"] = ranked.apply(lambda r: _pm(r["recall_mean"], r["recall_std"]), axis=1)
    ranked["F1"] = ranked.apply(lambda r: _pm(r["f1_score_mean"], r["f1_score_std"]), axis=1)

    print("\n" + "=" * 92)
    print("TABLE 5 RESULTS (sorted by Accuracy)")
    print("=" * 92)
    cols = ["model", "feature_type", "Accuracy", "Precision", "Recall", "F1", "cv_folds"]
    print(ranked[cols].to_string(index=False))


def print_insights(df: pd.DataFrame) -> None:
    best_acc = df.loc[df["accuracy_mean"].idxmax()]
    best_prec = df.loc[df["precision_mean"].idxmax()]
    best_rec = df.loc[df["recall_mean"].idxmax()]
    best_f1 = df.loc[df["f1_score_mean"].idxmax()]

    print("\n" + "=" * 92)
    print("KEY FINDINGS")
    print("=" * 92)
    print(f"- Best Accuracy : {best_acc['model']} on {best_acc['feature_type']} ({_pct(best_acc['accuracy_mean'])})")
    print(f"- Best Precision: {best_prec['model']} on {best_prec['feature_type']} ({_pct(best_prec['precision_mean'])})")
    print(f"- Best Recall   : {best_rec['model']} on {best_rec['feature_type']} ({_pct(best_rec['recall_mean'])})")
    print(f"- Best F1-score : {best_f1['model']} on {best_f1['feature_type']} ({_pct(best_f1['f1_score_mean'])})")

    # Quick interpretation for the currently available CSV
    feature_order = df.sort_values("accuracy_mean", ascending=False)["feature_type"].tolist()
    if len(df["model"].unique()) == 1:
        model_name = df["model"].iloc[0]
        best_feature = feature_order[0]
        worst_feature = feature_order[-1]
        delta = df["accuracy_mean"].max() - df["accuracy_mean"].min()
        paragraph = (
            f"For {model_name}, {best_feature} is currently the strongest representation while "
            f"{worst_feature} is the weakest in this run. The gap between best and worst "
            f"accuracy is {_pct(delta)}, which suggests feature choice has a large impact."
        )
        print("\n" + textwrap.fill(paragraph, width=92))


def plot_metric_bars(df: pd.DataFrame, out_dir: Path) -> None:
    for metric in METRICS:
        plt.figure(figsize=(10, 5))
        view = df.sort_values(f"{metric}_mean", ascending=False)
        x = range(len(view))
        plt.bar(
            x,
            view[f"{metric}_mean"].values * 100.0,
            yerr=view[f"{metric}_std"].values * 100.0,
            capsize=4,
            alpha=0.85,
        )
        labels = [f"{m}\n{f}" for m, f in zip(view["model"], view["feature_type"])]
        plt.xticks(list(x), labels, rotation=35, ha="right")
        plt.ylabel(f"{metric.replace('_', ' ').title()} (%)")
        plt.title(f"Table 5 - {metric.replace('_', ' ').title()} (mean +/- std)")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        save_path = out_dir / f"{metric}_barplot.png"
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close()


def plot_heatmap_like(df: pd.DataFrame, out_dir: Path) -> None:
    # Works nicely with one or many models; for many models it uses model|feature rows.
    row_label = df["model"] + " | " + df["feature_type"]
    hm = pd.DataFrame(
        {
            "Accuracy": df["accuracy_mean"].values * 100.0,
            "Precision": df["precision_mean"].values * 100.0,
            "Recall": df["recall_mean"].values * 100.0,
            "F1": df["f1_score_mean"].values * 100.0,
        },
        index=row_label,
    ).sort_values("Accuracy", ascending=False)

    plt.figure(figsize=(8.5, max(4, 0.38 * len(hm))))
    plt.imshow(hm.values, aspect="auto")
    plt.colorbar(label="Score (%)")
    plt.xticks(range(hm.shape[1]), hm.columns)
    plt.yticks(range(hm.shape[0]), hm.index)
    plt.title("Table 5 Metrics Heatmap")
    plt.tight_layout()
    save_path = out_dir / "metrics_heatmap.png"
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SAFE Table 5 CSV and create summary plots.")
    parser.add_argument(
        "--csv",
        default="results_table5/dl_results_spectrogram_cv.csv",
        help="Path to Table 5 CSV file.",
    )
    parser.add_argument(
        "--out-dir",
        default="results_table5/analysis",
        help="Directory where plots and summary CSV will be saved.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path.resolve()}")

    print_ranked_table(df)
    print_insights(df)

    # Save a compact ranked summary
    ranked = df.sort_values("accuracy_mean", ascending=False).copy()
    ranked.to_csv(out_dir / "ranked_summary.csv", index=False)

    plot_metric_bars(df, out_dir)
    plot_heatmap_like(df, out_dir)

    print("\nSaved outputs:")
    print(f"- {out_dir / 'ranked_summary.csv'}")
    print(f"- {out_dir / 'accuracy_barplot.png'}")
    print(f"- {out_dir / 'precision_barplot.png'}")
    print(f"- {out_dir / 'recall_barplot.png'}")
    print(f"- {out_dir / 'f1_score_barplot.png'}")
    print(f"- {out_dir / 'metrics_heatmap.png'}")


if __name__ == "__main__":
    main()

