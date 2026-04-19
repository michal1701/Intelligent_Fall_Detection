"""
Analyze Table 4 ML results and create readable tables + plots.

Input CSV (default):
    results_test_table4/ml_results_spectrogram.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def print_main_tables(df: pd.DataFrame) -> None:
    ranked = df.sort_values("accuracy", ascending=False).copy()
    for c in ["accuracy", "precision", "recall", "f1_score"]:
        ranked[c] = ranked[c].map(pct)

    print("\n" + "=" * 96)
    print("TABLE 4 RESULTS - RANKED BY ACCURACY")
    print("=" * 96)
    print(
        ranked[
            ["model", "feature_type", "accuracy", "precision", "recall", "f1_score"]
        ].to_string(index=False)
    )

    best_by_feature = (
        df.sort_values("accuracy", ascending=False)
        .groupby("feature_type", as_index=False)
        .first()
        .sort_values("accuracy", ascending=False)
    )
    for c in ["accuracy", "precision", "recall", "f1_score"]:
        best_by_feature[c] = best_by_feature[c].map(pct)

    print("\n" + "=" * 96)
    print("BEST MODEL PER FEATURE TYPE")
    print("=" * 96)
    print(
        best_by_feature[
            ["feature_type", "model", "accuracy", "precision", "recall", "f1_score"]
        ].to_string(index=False)
    )


def print_insights(df: pd.DataFrame) -> None:
    top = df.sort_values("accuracy", ascending=False).iloc[0]
    by_feature = df.groupby("feature_type", as_index=False)["accuracy"].mean().sort_values("accuracy", ascending=False)
    by_model = df.groupby("model", as_index=False)["accuracy"].mean().sort_values("accuracy", ascending=False)

    print("\n" + "=" * 96)
    print("KEY FINDINGS")
    print("=" * 96)
    print(f"- Best overall: {top['model']} + {top['feature_type']} with accuracy {pct(top['accuracy'])}")
    print(f"- Best average feature representation: {by_feature.iloc[0]['feature_type']} ({pct(by_feature.iloc[0]['accuracy'])})")
    print(f"- Best average model family: {by_model.iloc[0]['model']} ({pct(by_model.iloc[0]['accuracy'])})")
    print(f"- Weakest feature by average accuracy: {by_feature.iloc[-1]['feature_type']} ({pct(by_feature.iloc[-1]['accuracy'])})")


def plot_grouped_bars(df: pd.DataFrame, out_dir: Path) -> None:
    # model x feature for each metric
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        pivot = df.pivot(index="feature_type", columns="model", values=metric).sort_index()
        ax = pivot.plot(kind="bar", figsize=(12, 6))
        ax.set_title(f"Table 4 - {metric.replace('_', ' ').title()} by Feature and Model")
        ax.set_xlabel("Feature Type")
        ax.set_ylabel(f"{metric.replace('_', ' ').title()} (%)")
        ax.set_ylim(0.0, 1.02)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
        # show percentages
        yticks = ax.get_yticks()
        ax.set_yticklabels([f"{y*100:.0f}%" for y in yticks])
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_grouped_bar.png", dpi=180, bbox_inches="tight")
        plt.close()


def plot_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    # Heatmap-like image for accuracy (feature x model)
    pivot = df.pivot(index="feature_type", columns="model", values="accuracy").sort_index()
    plt.figure(figsize=(10, 4.8))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Accuracy")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Table 4 Accuracy Heatmap (Feature x Model)")
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Table 4 CSV and generate summary + plots.")
    parser.add_argument("--csv", default="results_test_table4/ml_results_spectrogram.csv")
    parser.add_argument("--out-dir", default="results_test_table4/analysis")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    print_main_tables(df)
    print_insights(df)

    # Save ranked + aggregated summaries
    df.sort_values("accuracy", ascending=False).to_csv(out_dir / "ranked_summary.csv", index=False)
    df.groupby("feature_type", as_index=False)[["accuracy", "precision", "recall", "f1_score"]].mean().sort_values(
        "accuracy", ascending=False
    ).to_csv(out_dir / "feature_averages.csv", index=False)
    df.groupby("model", as_index=False)[["accuracy", "precision", "recall", "f1_score"]].mean().sort_values(
        "accuracy", ascending=False
    ).to_csv(out_dir / "model_averages.csv", index=False)

    plot_grouped_bars(df, out_dir)
    plot_heatmap(df, out_dir)

    print("\nSaved files:")
    print(f"- {out_dir / 'ranked_summary.csv'}")
    print(f"- {out_dir / 'feature_averages.csv'}")
    print(f"- {out_dir / 'model_averages.csv'}")
    print(f"- {out_dir / 'accuracy_grouped_bar.png'}")
    print(f"- {out_dir / 'precision_grouped_bar.png'}")
    print(f"- {out_dir / 'recall_grouped_bar.png'}")
    print(f"- {out_dir / 'f1_score_grouped_bar.png'}")
    print(f"- {out_dir / 'accuracy_heatmap.png'}")


if __name__ == "__main__":
    main()

