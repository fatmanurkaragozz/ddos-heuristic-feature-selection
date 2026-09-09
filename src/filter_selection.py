"""Aşama 1 — Filter ile ön eleme (183 -> ~60 aday öznitelik).

GA/PSO wrapper araması pahalıdır (binlerce model eğitimi). Arama uzayını
küçültmek için önce ucuz, model-hafif iki filtre uygulanır ve İKİ aday havuz
üretilir; hangisinin daha iyi olduğunu Aşama 2'de karşılaştırırız.

Yöntemler (ikisi de YALNIZCA train verisinden hesaplanır — test/val'e dokunulmaz)
------------------------------------------------------------------------------
A) Random Forest önem skoru — hızlı bir RF eğitilir, `feature_importances_`
   sıralamasından ilk `--top-k` alınır. Kombinasyon/etkileşimi kısmen görür.
B) Mutual Information — her özniteliğin `label` ile paylaştığı bilgi
   (doğrusal olmayan ilişkileri de yakalar), ilk `--top-k` alınır.

Çıktı: data/processed/filter_pools.json  +  results/figures/03b_filter_scores.png
       results/metrics/filter_selection.json

Kullanım
--------
    python src/filter_selection.py --top-k 60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from utils import ROOT, load_processed

PROCESSED = ROOT / "data" / "processed"
METRICS_DIR = ROOT / "results" / "metrics"
FIG_DIR = ROOT / "results" / "figures"
RANDOM_STATE = 42


def rf_importance(X: pd.DataFrame, y: np.ndarray, n_trees: int) -> pd.Series:
    rf = RandomForestClassifier(
        n_estimators=n_trees, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced",
    )
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)


def mi_scores(X: pd.DataFrame, y: np.ndarray, sample: int) -> pd.Series:
    if len(X) > sample:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X), sample, replace=False)
        X, y = X.iloc[idx], y[idx]
    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top-k", type=int, default=60, help="her havuzda tutulacak öznitelik sayısı")
    ap.add_argument("--rf-trees", type=int, default=200)
    ap.add_argument("--mi-sample", type=int, default=100_000, help="MI hesabı için train alt örneklemi")
    args = ap.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_processed()
    X, y = ds.X_train, ds.y_train          # <-- yalnızca train
    print(f"Filtre girdisi: {X.shape[1]} öznitelik, {len(y):,} train satırı")

    print("[A] Random Forest önem skoru hesaplanıyor...")
    rf_imp = rf_importance(X, y, args.rf_trees)
    print("[B] Mutual Information hesaplanıyor...")
    mi = mi_scores(X, y, args.mi_sample)

    k = args.top_k
    pool_rf = rf_imp.head(k).index.tolist()
    pool_mi = mi.head(k).index.tolist()
    overlap = sorted(set(pool_rf) & set(pool_mi))
    union = sorted(set(pool_rf) | set(pool_mi))
    print(f"\n  RF havuzu     : {len(pool_rf)} öznitelik")
    print(f"  MI havuzu     : {len(pool_mi)} öznitelik")
    print(f"  ortak         : {len(overlap)}")
    print(f"  birleşim      : {len(union)}")
    print(f"\n  RF ilk 10: {pool_rf[:10]}")
    print(f"  MI ilk 10: {pool_mi[:10]}")

    pools = {
        "top_k": k,
        "pool_rf_importance": pool_rf,
        "pool_mutual_info": pool_mi,
        "overlap": overlap,
        "union": union,
    }
    (PROCESSED / "filter_pools.json").write_text(json.dumps(pools, indent=2), encoding="utf-8")

    scores_out = {
        "params": vars(args),
        "rf_importance": {f: float(v) for f, v in rf_imp.items()},
        "mutual_info": {f: float(v) for f, v in mi.items()},
        "pool_sizes": {"rf": len(pool_rf), "mi": len(pool_mi),
                       "overlap": len(overlap), "union": len(union)},
    }
    (METRICS_DIR / "filter_selection.json").write_text(json.dumps(scores_out, indent=2), encoding="utf-8")

    # görsel: ilk 30 öznitelik her iki skorda
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    rf_imp.head(30).iloc[::-1].plot.barh(ax=ax[0], color="forestgreen")
    ax[0].set_title(f"Random Forest önem — ilk 30 (havuz: ilk {k})")
    mi.head(30).iloc[::-1].plot.barh(ax=ax[1], color="darkorange")
    ax[1].set_title(f"Mutual Information — ilk 30 (havuz: ilk {k})")
    plt.tight_layout(); plt.savefig(FIG_DIR / "03b_filter_scores.png", dpi=120); plt.close(fig)

    print(f"\nKaydedildi -> {PROCESSED / 'filter_pools.json'}")
    print("Sonraki adım: src/genetic_algorithm.py")


if __name__ == "__main__":
    main()
