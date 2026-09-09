"""Adım 2 — Ön işleme.

Ham BCCC-cPacket-Cloud-DDoS-2024 parquet dosyasını alır ve modellemeye hazır
train / validation / test kümelerini üretir.

İş akışı
--------
1.  Ham veriyi oku, `activity` sütununu at (hedef = `label`).
2.  `label` sütununu sayıya çevir (LabelEncoder).
3.  Stratified 70 / 15 / 15 train / validation / test bölmesi (random_state=42).
4.  "Aşama 0" güvenli temizlik — YALNIZCA train istatistiğiyle hesaplanır,
    sonra üç kümeye de uygulanır:
      * sabit sütunlar (train'de tek değer)
      * neredeyse-sabit sütunlar (bir değer train satırlarının > %99.9'unu kaplıyor)
      * aşırı korele çiftler (|Pearson r| > 0.98) — çiftin ikinci sütunu atılır
5.  StandardScaler — train'e fit, validation + test'e transform.
6.  `data/processed/` altına kaydet:
      X_train/X_val/X_test.parquet, y_train/y_val/y_test.npy,
      scaler.joblib, label_encoder.joblib, metadata.json

Kullanım
--------
    python src/preprocessing.py
    python src/preprocessing.py --corr-threshold 0.95 --near-constant 0.995
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "archive" / "bccc-cpacket-cloud-ddos-2024-merged.parquet"
OUT = ROOT / "data" / "processed"
REPORT = ROOT / "results" / "metrics" / "preprocessing_report.json"

TARGET = "label"
DROP_COLS = ["activity"]          # ikinci etiket — bu deneyde kullanılmıyor
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Aşama 0 — güvenli temizlik (yalnızca train'den öğrenilir)
# --------------------------------------------------------------------------- #
def find_constant(df: pd.DataFrame) -> list[str]:
    """Train'de tek benzersiz değere sahip sütunlar — sıfır bilgi."""
    nunique = df.nunique()
    return nunique[nunique <= 1].index.tolist()


def find_near_constant(df: pd.DataFrame, threshold: float) -> list[str]:
    """Baskın bir değerin satırların > `threshold` oranını kapladığı sütunlar."""
    cols = []
    for c in df.columns:
        top_freq = df[c].value_counts(normalize=True).iloc[0]
        if top_freq > threshold:
            cols.append(c)
    return cols


def find_correlated(df: pd.DataFrame, threshold: float, sample: int = 100_000) -> list[str]:
    """|r| > `threshold` olan her çiftin İKİNCİ sütununu (sütun sırasına göre) döndürür.

    Korelasyon büyük bir train örnekleminden hesaplanır (Pearson r ölçekten
    bağımsız olduğu için ölçekleme öncesi/sonrası fark etmez).
    """
    if len(df) > sample:
        df = df.sample(n=sample, random_state=RANDOM_STATE)
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    return [c for c in upper.columns if (upper[c] > threshold).any()]


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--near-constant", type=float, default=0.999,
                    help="neredeyse-sabit eşiği (baskın değer oranı), varsayılan 0.999")
    ap.add_argument("--corr-threshold", type=float, default=0.98,
                    help="aşırı korelasyon eşiği |r|, varsayılan 0.98")
    ap.add_argument("--train-size", type=float, default=0.70)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    args = ap.parse_args()

    assert abs(args.train_size + args.val_size + args.test_size - 1.0) < 1e-9, \
        "train + val + test = 1.0 olmalı"

    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. yükle -------------------------------------------------------------
    print(f"[1/6] Ham veri okunuyor: {RAW.name}")
    df = pd.read_parquet(RAW)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    print(f"      {df.shape[0]:,} satır × {df.shape[1]} sütun ({df.shape[1] - 1} öznitelik + hedef)")

    # --- 2. hedefi encode et ----------------------------------------------------
    print("[2/6] Hedef (`label`) sayıya çevriliyor")
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    X = df.drop(columns=[TARGET]).astype("float32")
    class_map = {cls: int(i) for i, cls in enumerate(le.classes_)}
    print(f"      sınıf eşlemesi: {class_map}")

    # --- 3. stratified 70/15/15 bölme -----------------------------------------
    print(f"[3/6] Stratified bölme {args.train_size:.0%}/{args.val_size:.0%}/{args.test_size:.0%}")
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=args.val_size + args.test_size,
        stratify=y, random_state=RANDOM_STATE,
    )
    rel_test = args.test_size / (args.val_size + args.test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_test,
        stratify=y_tmp, random_state=RANDOM_STATE,
    )
    for name, yy in [("train", y_train), ("val", y_val), ("test", y_test)]:
        dist = {le.classes_[i]: int((yy == i).sum()) for i in range(len(le.classes_))}
        print(f"      {name:5s}: {len(yy):>7,} satır  {dist}")

    # --- 4. Aşama 0 temizlik (train'den öğren) -------------------------------
    print(f"[4/6] Aşama 0 temizlik (near-constant>{args.near_constant}, |r|>{args.corr_threshold})")
    constant = find_constant(X_train)
    near_const = [c for c in find_near_constant(X_train, args.near_constant) if c not in constant]
    remaining = [c for c in X_train.columns if c not in constant + near_const]
    correlated = find_correlated(X_train[remaining], args.corr_threshold)

    to_drop = sorted(set(constant + near_const + correlated))
    kept = [c for c in X_train.columns if c not in to_drop]
    print(f"      sabit           : {len(constant)}")
    print(f"      neredeyse-sabit : {len(near_const)}")
    print(f"      aşırı korele    : {len(correlated)}")
    print(f"      --> atılan {len(to_drop)}, kalan {len(kept)} öznitelik")

    X_train, X_val, X_test = X_train[kept], X_val[kept], X_test[kept]

    # --- 5. ölçekleme (train'e fit) -----------------------------------------
    print("[5/6] StandardScaler (train'e fit, val/test'e transform)")
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=kept, index=X_train.index).astype("float32")
    X_val = pd.DataFrame(scaler.transform(X_val), columns=kept, index=X_val.index).astype("float32")
    X_test = pd.DataFrame(scaler.transform(X_test), columns=kept, index=X_test.index).astype("float32")

    # --- 6. kaydet ---------------------------------------------------------------
    print(f"[6/6] Kaydediliyor -> {OUT}")
    X_train.to_parquet(OUT / "X_train.parquet")
    X_val.to_parquet(OUT / "X_val.parquet")
    X_test.to_parquet(OUT / "X_test.parquet")
    np.save(OUT / "y_train.npy", y_train)
    np.save(OUT / "y_val.npy", y_val)
    np.save(OUT / "y_test.npy", y_test)
    joblib.dump(scaler, OUT / "scaler.joblib")
    joblib.dump(le, OUT / "label_encoder.joblib")

    metadata = {
        "raw_file": RAW.name,
        "random_state": RANDOM_STATE,
        "split": {"train": args.train_size, "val": args.val_size, "test": args.test_size},
        "class_map": class_map,
        "n_features_raw": int(X.shape[1]),
        "n_features_kept": len(kept),
        "kept_features": kept,
        "dropped": {
            "constant": constant,
            "near_constant": near_const,
            "correlated": correlated,
        },
        "shapes": {
            "X_train": list(X_train.shape),
            "X_val": list(X_val.shape),
            "X_test": list(X_test.shape),
        },
        "params": {"near_constant": args.near_constant, "corr_threshold": args.corr_threshold},
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    REPORT.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nBitti. Sonraki adım: src/baseline_model.py")


if __name__ == "__main__":
    main()
