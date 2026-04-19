"""
Main script to replicate SAFE paper results:
- Table 3: ML classifiers on raw waveform (sklearn all_estimators, 10-fold CV)
- Table 4: classical ML on spectrogram statistics (train/test split)
- Table 5: pretrained CNNs on spectrogram images (10-fold CV)
"""

import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from data_loader import SAFEDataLoader
from feature_extraction import SpectrogramFeatureExtractor
from ml_models import MLModelTrainer
from evaluation import ResultEvaluator

# Typing
from typing import List, Optional


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV atomically (same dir) so a crash mid-write does not leave a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _append_rate_to_combined_all_rates(output_dir: Path, rate_df: pd.DataFrame, sr: int) -> None:
    """
    Update ml_results_raw_audio_cv_all_rates.csv: replace rows for this sampling rate, keep others.
    Safe when Table 3 is run in separate processes (one rate per run).
    """
    combined_path = output_dir / "ml_results_raw_audio_cv_all_rates.csv"
    if combined_path.exists():
        old = pd.read_csv(combined_path)
        if "sampling_rate_hz" in old.columns:
            old = old[old["sampling_rate_hz"].astype(int) != int(sr)]
        else:
            old = pd.DataFrame()
        merged = pd.concat([old, rate_df], ignore_index=True)
    else:
        merged = rate_df
    _atomic_write_csv(merged, combined_path)


def run_table3_raw_audio(
    data_dir: str = "../data",
    output_dir: Path = Path("results_table3"),
    random_state: int = 42,
    cv_folds: int = 10,
    sample_rates: Optional[List[int]] = None,
    raw_duration_sec: float = 3.0,
    skip_gradient_boosting: bool = False,
) -> None:
    """
    SAFE paper Table 3: supervised sklearn classifiers on raw digital audio samples,
    10-fold stratified CV; mean ± std of accuracy, precision, recall, F1 over folds.
    """
    from table3_raw_audio_ml import run_table3_all_classifiers_cv

    print("=" * 80)
    print("SAFE Dataset Replication — Raw waveform + sklearn classifiers (Table 3)")
    print("10-fold stratified CV | all_estimators(type_filter='classifier')")
    if skip_gradient_boosting:
        print("(excluding GradientBoostingClassifier & HistGradientBoostingClassifier)")
    print("=" * 80)

    output_dir.mkdir(exist_ok=True)

    rates = sample_rates if sample_rates is not None else [48000, 32000, 16000, 8000, 4000]
    all_results = []

    for sr in rates:
        print(f"\n[Table 3] Loading SAFE dataset @ {sr} Hz ...")
        data_loader = SAFEDataLoader(data_dir=data_dir, target_sr=sr)

        try:
            audio_list, labels = data_loader.load_dataset()
        except ValueError as e:
            print(f"Error: {e}")
            print(f"Please ensure the SAFE dataset is in the {data_dir} directory.")
            return

        print(f"\nSamples: {len(audio_list)} | CV folds: {cv_folds} | raw window: {raw_duration_sec}s @ {sr} Hz")

        results_df = run_table3_all_classifiers_cv(
            audio_list=list(audio_list),
            y=labels,
            sample_rate=sr,
            raw_duration_sec=raw_duration_sec,
            n_splits=cv_folds,
            random_state=random_state,
            scale_features=True,
            skip_gradient_boosting=skip_gradient_boosting,
            progress_cb=print,
        )

        ok_df = results_df[results_df["status"] == "ok"].sort_values("accuracy_mean", ascending=False)
        print("\n" + "=" * 80)
        print(f"Table 3 — top 10 models (mean %, status=ok) @ {sr} Hz")
        print("=" * 80)
        print(f"{'Classifier algorithm':<34} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10}")
        print("-" * 80)
        for _, row in ok_df.head(10).iterrows():
            print(
                f"{row['model']:<34} "
                f"{row['accuracy_mean'] * 100:>7.2f}%   "
                f"{row['precision_mean'] * 100:>7.2f}%   "
                f"{row['recall_mean'] * 100:>7.2f}%   "
                f"{row['f1_score_mean'] * 100:>7.2f}%"
            )
        print("=" * 80)

        per_rate_csv = output_dir / f"ml_results_raw_audio_cv_{sr}hz.csv"
        out = results_df.assign(_ok=results_df["status"].eq("ok")).sort_values(
            ["_ok", "accuracy_mean"], ascending=[False, False]
        ).drop(columns=["_ok"])
        _atomic_write_csv(out, per_rate_csv)
        print(f"Saved (safe to stop before next rate): {per_rate_csv.resolve()}")

        top10_csv = output_dir / f"table3_top10_{sr}hz.csv"
        _atomic_write_csv(ok_df.head(10), top10_csv)
        print(f"Saved: {top10_csv.resolve()}")

        _append_rate_to_combined_all_rates(output_dir, results_df, sr)
        combined_path = output_dir / "ml_results_raw_audio_cv_all_rates.csv"
        print(f"Updated combined table (all finished rates): {combined_path.resolve()}")

        checkpoint = output_dir / "table3_checkpoint.log"
        with open(checkpoint, "a", encoding="utf-8") as lf:
            lf.write(
                f"{datetime.now().isoformat()}  Hz={sr}  "
                f"rows={len(results_df)}  files={per_rate_csv.name}, {top10_csv.name}\n"
            )
        print(f"Checkpoint log: {checkpoint.resolve()}")

        all_results.append(results_df)

    if all_results:
        print("\n" + "=" * 80)
        print(
            f"Table 3: combined CSV is updated after each rate: "
            f"{(output_dir / 'ml_results_raw_audio_cv_all_rates.csv').resolve()}"
        )
        print("=" * 80)


FEATURE_TYPES_TABLE4_5 = [
    "mel_spectrogram",
    "stft_spectrogram",
    "mfcc",
    "cqt_spectrogram",
    "cwt_spectrogram",
    "chroma",
]


def run_table4_ml(
    data_dir: str = "../data",
    output_dir: Path = Path("results_table4"),
    random_state: int = 42,
    test_size: float = 0.2,
) -> None:
    """SAFE paper Table 4: classical ML on statistical spectrogram features."""
    print("=" * 80)
    print("SAFE Dataset Replication - Machine Learning on Spectrogram Features (Table 4)")
    print("Section 4.2 & 5.2 Implementation")
    print("=" * 80)

    output_dir.mkdir(exist_ok=True)

    print("\n[Step 1] Loading SAFE dataset...")
    data_loader = SAFEDataLoader(data_dir=data_dir, target_sr=22050)

    try:
        audio_list, labels = data_loader.load_dataset()
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Please ensure the SAFE dataset is in the {data_dir} directory.")
        return

    print("\n[Step 2] Splitting dataset...")
    X_train_audio, X_test_audio, y_train, y_test = data_loader.get_train_test_split(
        test_size=test_size,
        random_state=random_state,
    )

    print(f"Training samples: {len(X_train_audio)}")
    print(f"Test samples: {len(X_test_audio)}")

    evaluator = ResultEvaluator()
    all_results = []

    for feature_type in FEATURE_TYPES_TABLE4_5:
        print(f"\n{'=' * 80}")
        print(f"[Step 3] Processing {feature_type} features...")
        print(f"{'=' * 80}")

        feature_extractor = SpectrogramFeatureExtractor(
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            n_mfcc=13,
            sr=22050,
        )

        print(f"Extracting {feature_type} features...")
        X_train_features = feature_extractor.extract_features_batch(
            X_train_audio,
            feature_type=feature_type,
            use_stats=True,
        )
        X_test_features = feature_extractor.extract_features_batch(
            X_test_audio,
            feature_type=feature_type,
            use_stats=True,
        )

        print(f"Feature shape: {X_train_features.shape}")

        feature_extractor.fit_scaler(X_train_features)
        X_train_scaled = feature_extractor.transform_features(X_train_features)
        X_test_scaled = feature_extractor.transform_features(X_test_features)

        trainer = MLModelTrainer(random_state=random_state)

        print(f"\nTraining models on {feature_type} features...")
        trained_models = trainer.train_all_models(
            X_train_scaled,
            y_train,
            use_grid_search=False,
        )

        print(f"\nEvaluating models on {feature_type} features...")
        for model_name, model in trained_models.items():
            y_pred = model.predict(X_test_scaled)

            metrics = evaluator.evaluate_model(
                model_name,
                y_test,
                y_pred,
                feature_type=feature_type,
            )

            all_results.append(metrics)

            print(
                f"  {model_name:<20} - Accuracy: {metrics['accuracy'] * 100:.2f}%, "
                f"F1: {metrics['f1_score'] * 100:.2f}%"
            )

            if metrics["accuracy"] > 0.90:
                cm_path = output_dir / f"confusion_matrix_{model_name}_{feature_type}.png"
                evaluator.plot_confusion_matrix(y_test, y_pred, model_name, str(cm_path))

    print(f"\n{'=' * 80}")
    print("[Step 4] Results Summary (Table 4)")
    print(f"{'=' * 80}")

    results_df = evaluator.create_results_table(all_results)
    evaluator.print_results_summary(results_df)

    results_path = output_dir / "ml_results_spectrogram.csv"
    evaluator.save_results(results_df, str(results_path))

    plot_path = output_dir / "results_comparison.png"
    evaluator.plot_results_comparison(results_df, metric="accuracy", save_path=str(plot_path))

    print("\n" + "=" * 80)
    print(f"Table 4 replication complete. Results saved under {output_dir.resolve()}/")
    print("=" * 80)


def run_table5_deep_learning(
    data_dir: str = "../data",
    output_dir: Path = Path("results_table5"),
    random_state: int = 42,
    cv_folds: int = 10,
    epochs: int = 25,
    batch_size: int = 16,
    device: str = "auto",
    table5_models: Optional[List[str]] = None,
    feature_types: Optional[List[str]] = None,
    image_size: int = 224,
    efficientnet_extra_epochs: int = 15,
) -> None:
    """
    SAFE paper Table 5: pretrained CNNs on spectrogram images with CV.
    Default: ResNet-18, EfficientNet-B0, ConvNeXt-Small; use table5_models / light preset to subset.
    """
    from table5_deep_learning import run_table5_cross_validation

    feats = feature_types if feature_types is not None else FEATURE_TYPES_TABLE4_5
    model_line = (
        ", ".join(table5_models)
        if table5_models
        else "ResNet-18, EfficientNet-B0, ConvNeXt-Small"
    )

    print("=" * 80)
    print("SAFE Dataset Replication - Deep Learning on Spectrogram Images (Table 5)")
    print(f"{model_line} | {cv_folds}-fold CV | image {image_size}px | {len(feats)} spectrogram type(s)")
    print("=" * 80)

    output_dir.mkdir(exist_ok=True)

    print("\n[Table 5] Loading SAFE dataset...")
    data_loader = SAFEDataLoader(data_dir=data_dir, target_sr=22050)

    try:
        audio_list, labels = data_loader.load_dataset()
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Please ensure the SAFE dataset is in the {data_dir} directory.")
        return

    fold_groups: Optional[np.ndarray] = data_loader.fold_ids
    if fold_groups is not None and (np.any(fold_groups < 1) or len(np.unique(fold_groups)) < cv_folds):
        print(
            "Note: Filename fold IDs missing or fewer than n_splits; "
            "using stratified K-fold instead of group (fold) CV."
        )
        fold_groups = None

    print(f"\nSamples: {len(audio_list)} | CV folds: {cv_folds} | epochs/fold: {epochs}")

    results_df = run_table5_cross_validation(
        audio_list=list(audio_list),
        y=labels,
        fold_groups=fold_groups,
        feature_types=list(feats),
        random_state=random_state,
        n_splits=cv_folds,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-4,
        image_size=image_size,
        extractor_kwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128, "n_mfcc": 13, "sr": 22050},
        progress_cb=print,
        device_preference=device,
        models=table5_models,
        efficientnet_extra_epochs=efficientnet_extra_epochs,
    )

    # Human-readable console table (mean ± std %)
    print("\n" + "=" * 80)
    print("Table 5 — mean ± std over folds (fractions shown as %)")
    print("=" * 80)
    print(
        f"{'Model':<18} {'Feature':<22} {'Acc %':<14} {'Prec %':<14} {'Rec %':<14} {'F1 %':<14}"
    )
    print("-" * 80)
    for _, row in results_df.sort_values(["feature_type", "model"]).iterrows():
        def fmt(m, s):
            return f"{m * 100:.2f}±{s * 100:.2f}"

        print(
            f"{row['model']:<18} {row['feature_type']:<22} "
            f"{fmt(row['accuracy_mean'], row['accuracy_std']):<14} "
            f"{fmt(row['precision_mean'], row['precision_std']):<14} "
            f"{fmt(row['recall_mean'], row['recall_std']):<14} "
            f"{fmt(row['f1_score_mean'], row['f1_score_std']):<14}"
        )
    print("=" * 80)

    csv_path = output_dir / "dl_results_spectrogram_cv.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nTable 5 results saved to {csv_path.resolve()}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="SAFE paper replication (Table 3 raw audio CV / Table 4 ML / Table 5 deep learning)."
    )
    parser.add_argument(
        "--table",
        choices=("3", "4", "5", "both", "all"),
        default="4",
        help="3=raw waveform + sklearn CV; 4=spectrogram ML; 5=CNN CV; both=4+5; all=3+4+5.",
    )
    parser.add_argument("--data-dir", default="../data", help="Directory containing SAFE .wav files.")
    parser.add_argument(
        "--raw-duration-sec",
        type=float,
        default=3.0,
        help="Table 3: clip length (seconds); waveforms are padded/truncated to sample_rate * this value.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Table 3: resampling rate when loading audio (Hz).",
    )
    parser.add_argument(
        "--table3-sample-rates",
        nargs="+",
        type=int,
        default=[48000, 32000, 16000, 8000, 4000],
        help="Table 3: sampling rates to evaluate (Hz). Default matches the paper: 48000 32000 16000 8000 4000.",
    )
    parser.add_argument(
        "--table3-no-gradient-boosting",
        action="store_true",
        help="Table 3: skip sklearn gradient boosting classifiers (GradientBoosting, HistGradientBoosting).",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs per fold (Table 5); EfficientNet-B0 adds 15.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for Table 5.")
    parser.add_argument("--cv-folds", type=int, default=10, help="Number of CV folds (Table 3 and Table 5).")
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu", "cuda"),
        default="auto",
        help="Device for Table 5 deep learning.",
    )
    parser.add_argument(
        "--table5-light",
        action="store_true",
        help=(
            "Table 5 quick run: EfficientNet-B0 only, mel spectrogram only, "
            "2-fold CV, 1 epoch, 128px images, no extra EfficientNet epochs."
        ),
    )
    parser.add_argument(
        "--table5-models",
        nargs="+",
        default=None,
        choices=["ResNet18", "EfficientNet_B0", "ConvNeXt_Small"],
        metavar="MODEL",
        help="Table 5: which CNN(s) to run (default: all three). Ignored if --table5-light.",
    )
    parser.add_argument(
        "--table5-features",
        nargs="+",
        default=None,
        choices=FEATURE_TYPES_TABLE4_5,
        metavar="FEATURE",
        help="Table 5: spectrogram types (default: all six). Ignored if --table5-light.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Table 5: spectrogram resize (square). Default 224; use 128 for faster runs.",
    )
    parser.add_argument(
        "--efficientnet-extra-epochs",
        type=int,
        default=15,
        help="Table 5: extra epochs for EfficientNet-B0 on top of --epochs (paper-style). Use 0 for fastest runs.",
    )
    args = parser.parse_args()

    if args.table in ("3", "all"):
        run_table3_raw_audio(
            data_dir=args.data_dir,
            cv_folds=args.cv_folds,
            sample_rates=args.table3_sample_rates,
            raw_duration_sec=args.raw_duration_sec,
            skip_gradient_boosting=args.table3_no_gradient_boosting,
        )
    if args.table in ("4", "both", "all"):
        run_table4_ml(data_dir=args.data_dir)
    if args.table in ("5", "both", "all"):
        if args.table5_light:
            run_table5_deep_learning(
                data_dir=args.data_dir,
                cv_folds=2,
                epochs=1,
                batch_size=args.batch_size,
                device=args.device,
                table5_models=["EfficientNet_B0"],
                feature_types=["mel_spectrogram"],
                image_size=128,
                efficientnet_extra_epochs=0,
            )
        else:
            run_table5_deep_learning(
                data_dir=args.data_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                cv_folds=args.cv_folds,
                device=args.device,
                table5_models=args.table5_models,
                feature_types=args.table5_features,
                image_size=args.image_size,
                efficientnet_extra_epochs=args.efficientnet_extra_epochs,
            )

    if args.table == "3":
        print("\nNote (SAFE paper / Table 3):")
        print("- Raw waveform features; many sklearn classifiers may skip or fail on high-dimensional input.")
        print("- See status column in results_table3/ml_results_raw_audio_cv.csv.")
    elif args.table == "4":
        print("\nNote (SAFE paper Section 5.2 / Table 4):")
        print("- Linear models can reach high accuracy on spectrogram statistics.")
        print("- Run Table 5 with: python main.py --table 5")
    elif args.table == "5":
        print("\nNote (SAFE paper Section 5.2 / Table 5):")
        print("- ImageNet transfer learning on resized spectrograms; metrics are CV mean ± std.")
        if not args.table5_light:
            print("- Fast demo: python main.py --table 5 --table5-light")
            print("- EfficientNet only: python main.py --table 5 --table5-models EfficientNet_B0 --cv-folds 2 --epochs 1 --efficientnet-extra-epochs 0")


if __name__ == "__main__":
    main()
