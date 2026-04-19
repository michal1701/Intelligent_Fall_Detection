"""
Table 5 (SAFE paper): pretrained CNNs on spectrogram images with 10-fold CV.
Models: ResNet-18, EfficientNet-B0, ConvNeXt-Small (torchvision, ImageNet weights).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from typing import Callable, Dict, List, Optional, Tuple

from feature_extraction import SpectrogramFeatureExtractor


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _resolve_device(device_preference: str = "auto") -> torch.device:
    pref = device_preference.lower()
    if pref == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("Requested device 'mps' but MPS is not available.")
    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Requested device 'cuda' but CUDA is not available.")
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _spectrogram_2d(extractor: SpectrogramFeatureExtractor, audio: np.ndarray, feature_type: str) -> np.ndarray:
    if feature_type == "mel_spectrogram":
        return extractor.extract_mel_spectrogram(audio)
    if feature_type == "stft_spectrogram":
        return extractor.extract_stft_spectrogram(audio)
    if feature_type == "mfcc":
        return extractor.extract_mfcc(audio)
    if feature_type == "cqt_spectrogram":
        return extractor.extract_cqt_spectrogram(audio)
    if feature_type == "cwt_spectrogram":
        return extractor.extract_cwt_spectrogram(audio)
    if feature_type == "chroma":
        return extractor.extract_chroma(audio)
    raise ValueError(f"Unknown feature_type: {feature_type}")


def audio_list_to_image_tensors(
    audio_list: List[np.ndarray],
    feature_type: str,
    extractor: SpectrogramFeatureExtractor,
    image_size: int = 224,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert waveforms to (N, 3, H, W) tensors, ImageNet-normalized."""
    if device is None:
        device = torch.device("cpu")
    tensors: List[torch.Tensor] = []
    for audio in audio_list:
        spec = _spectrogram_2d(extractor, audio, feature_type)
        spec = spec.astype(np.float32)
        smin, smax = float(spec.min()), float(spec.max())
        if smax > smin:
            spec = (spec - smin) / (smax - smin)
        else:
            spec = np.zeros_like(spec, dtype=np.float32)
        # (1,1,F,T) -> bilinear resize to (1,1,H,W)
        t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
        t = torch.nn.functional.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
        gray = t.squeeze(0)  # (1, H, W)
        rgb = gray.repeat(3, 1, 1)
        rgb = (rgb - IMAGENET_MEAN.to(rgb.device)) / IMAGENET_STD.to(rgb.device)
        tensors.append(rgb)
    return torch.stack(tensors, dim=0).to(device)


def _replace_classifier(model: nn.Module, arch: str, num_classes: int) -> nn.Module:
    arch_l = arch.lower()
    if arch_l == "resnet18":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch_l == "efficientnet_b0":
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif arch_l == "convnext_small":
        last = model.classifier[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError("Unexpected ConvNeXt head structure")
        in_features = last.in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model


def build_model(arch: str, num_classes: int = 2, weights: str = "DEFAULT") -> nn.Module:
    from torchvision import models

    def _load(factory):
        try:
            return factory(weights=weights)
        except TypeError:
            return factory(pretrained=True)

    if arch == "resnet18":
        m = _load(models.resnet18)
    elif arch == "efficientnet_b0":
        m = _load(models.efficientnet_b0)
    elif arch == "convnext_small":
        m = _load(models.convnext_small)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return _replace_classifier(m, arch, num_classes)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def _evaluate_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: List[int] = []
    trues: List[int] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        preds.extend(pred.tolist())
        trues.extend(yb.numpy().tolist())
    y_pred = np.array(preds, dtype=np.int64)
    y_true = np.array(trues, dtype=np.int64)
    return y_true, y_pred


def _fold_splitter(
    y: np.ndarray,
    groups: np.ndarray | None,
    n_splits: int,
    random_state: int,
):
    if groups is not None and len(np.unique(groups)) >= n_splits and np.all(groups >= 1):
        return GroupKFold(n_splits=n_splits), groups
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), None


TABLE5_ARCH_MAP = {
    "ResNet18": "resnet18",
    "EfficientNet_B0": "efficientnet_b0",
    "ConvNeXt_Small": "convnext_small",
}


def run_table5_cross_validation(
    audio_list: List[np.ndarray],
    y: np.ndarray,
    fold_groups: np.ndarray | None,
    feature_types: List[str],
    random_state: int = 42,
    n_splits: int = 10,
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-4,
    image_size: int = 224,
    extractor_kwargs: Dict | None = None,
    progress_cb: Callable[[str], None] | None = None,
    device_preference: str = "auto",
    models: Optional[List[str]] = None,
    efficientnet_extra_epochs: int = 15,
) -> pd.DataFrame:
    """
    10-fold CV (GroupKFold on filename fold when valid, else StratifiedKFold).
    Returns one row per (model, feature_type) with mean/std metrics across folds.
    """
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    device = _resolve_device(device_preference)
    log(f"Table 5 DL device: {device}")

    extractor_kw = dict(extractor_kwargs or {})
    sr = int(extractor_kw.pop("sr", 22050))
    extractor = SpectrogramFeatureExtractor(sr=sr, **extractor_kw)

    arch_map = dict(TABLE5_ARCH_MAP)
    if models is not None:
        unknown = set(models) - set(arch_map.keys())
        if unknown:
            raise ValueError(f"Unknown model name(s): {unknown}. Choose from {list(arch_map.keys())}.")
        arch_map = {k: v for k, v in arch_map.items() if k in models}

    rows: List[Dict] = []

    for feature_type in feature_types:
        log(f"Precomputing spectrogram images: {feature_type} ...")
        X_cpu = audio_list_to_image_tensors(
            audio_list, feature_type, extractor, image_size=image_size, device=torch.device("cpu")
        )
        y_t = torch.from_numpy(y.astype(np.int64))

        for display_name, arch in arch_map.items():
            fold_metrics: Dict[str, List[float]] = {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1_score": [],
            }

            cv, groups = _fold_splitter(y, fold_groups, n_splits, random_state)
            splitter = cv.split(np.arange(len(y)), y, groups) if groups is not None else cv.split(np.arange(len(y)), y)

            for fold_idx, (train_idx, val_idx) in enumerate(splitter):
                log(f"  {display_name} | {feature_type} | fold {fold_idx + 1}/{n_splits}")

                X_tr = X_cpu[train_idx]
                y_tr = y_t[train_idx]
                X_va = X_cpu[val_idx]
                y_va = y_t[val_idx]

                train_ds = TensorDataset(X_tr, y_tr)
                val_ds = TensorDataset(X_va, y_va)
                train_loader = DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=False,
                    pin_memory=(device.type == "cuda"),
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    pin_memory=(device.type == "cuda"),
                )

                model = build_model(arch, num_classes=2, weights="DEFAULT").to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
                criterion = nn.CrossEntropyLoss()

                extra = efficientnet_extra_epochs if arch == "efficientnet_b0" else 0
                eff_epochs = epochs + extra
                for _ in range(eff_epochs):
                    _train_one_epoch(model, train_loader, optimizer, criterion, device)

                y_true, y_pred = _evaluate_metrics(model, val_loader, device)
                fold_metrics["accuracy"].append(accuracy_score(y_true, y_pred))
                fold_metrics["precision"].append(
                    precision_score(y_true, y_pred, average="binary", zero_division=0)
                )
                fold_metrics["recall"].append(recall_score(y_true, y_pred, average="binary", zero_division=0))
                fold_metrics["f1_score"].append(f1_score(y_true, y_pred, average="binary", zero_division=0))

                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            rows.append(
                {
                    "model": display_name,
                    "feature_type": feature_type,
                    "accuracy_mean": float(np.mean(fold_metrics["accuracy"])),
                    "accuracy_std": float(np.std(fold_metrics["accuracy"])),
                    "precision_mean": float(np.mean(fold_metrics["precision"])),
                    "precision_std": float(np.std(fold_metrics["precision"])),
                    "recall_mean": float(np.mean(fold_metrics["recall"])),
                    "recall_std": float(np.std(fold_metrics["recall"])),
                    "f1_score_mean": float(np.mean(fold_metrics["f1_score"])),
                    "f1_score_std": float(np.std(fold_metrics["f1_score"])),
                    "cv_folds": n_splits,
                }
            )

    return pd.DataFrame(rows)
