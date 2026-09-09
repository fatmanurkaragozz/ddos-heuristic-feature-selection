"""Ortak yardımcılar — veri yükleme ve metrik hesaplama.

baseline_model.py, genetic_algorithm.py, pso.py ve evaluate.py bu modülü paylaşır.
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

# sklearn'ün joblib bağlamında ürettiği "delayed should be used with Parallel"
# uyarısı (davranışsal etkisi yok) log'ları boğuyordu — sustur.
warnings.filterwarnings("ignore", message=".*delayed.*should be used with.*Parallel.*")

# Windows konsolu (cp1254) Yunan harfi λ vb. karakterlerde çöküyor — UTF-8'e geç.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    features: list[str]
    class_names: list[str]        # indeks sırasına göre: [0]=Attack, [1]=Benign, [2]=Suspicious

    @property
    def X_trainval(self) -> pd.DataFrame:
        """train + validation birleşik — final 5-fold CV için."""
        return pd.concat([self.X_train, self.X_val], axis=0)

    @property
    def y_trainval(self) -> np.ndarray:
        return np.concatenate([self.y_train, self.y_val])


def load_processed(processed_dir: Path = PROCESSED) -> Dataset:
    """`src/preprocessing.py` çıktısını yükler."""
    meta = json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))
    class_names = [c for c, _ in sorted(meta["class_map"].items(), key=lambda kv: kv[1])]
    return Dataset(
        X_train=pd.read_parquet(processed_dir / "X_train.parquet"),
        X_val=pd.read_parquet(processed_dir / "X_val.parquet"),
        X_test=pd.read_parquet(processed_dir / "X_test.parquet"),
        y_train=np.load(processed_dir / "y_train.npy"),
        y_val=np.load(processed_dir / "y_val.npy"),
        y_test=np.load(processed_dir / "y_test.npy"),
        features=meta["kept_features"],
        class_names=class_names,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """accuracy + macro precision/recall/F1 + sınıf-bazlı skorlar.

    Dengesiz veride ana metrik **macro_f1**: her sınıfın F1'i eşit ağırlıkla ortalanır,
    böylece küçük `Suspicious` sınıfı da büyük `Benign` kadar sayılır.
    """
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": {},
    }
    p, r, f = (
        precision_score(y_true, y_pred, average=None, zero_division=0),
        recall_score(y_true, y_pred, average=None),
        f1_score(y_true, y_pred, average=None),
    )
    for i, name in enumerate(class_names):
        out["per_class"][name] = {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
    return out


def text_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> str:
    return classification_report(y_true, y_pred, target_names=class_names, digits=4)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)
