"""Adım 5b — λ (sadelik cezası) taraması.

Fitness = macro_F1_val - λ·(seçilen / havuz)

λ büyüdükçe GA daha az öznitelik seçmeye zorlanır. Bu betik GA'yı birden çok
λ değerinde koşar ve "λ ↔ öznitelik sayısı ↔ macro-F1" denge eğrisini çıkarır.
Amaç: mentörün istediği 20-25 öznitelik bandına düşen λ'yı bulmak ve
performans-sadelik ödünleşimini görselleştirmek.

Her λ için:
  * GA (hızlı fitness: 60k/30k alt örneklem)
  * bulunan alt kümenin DOKUNULMAMIŞ test setinde tam-ölçek RF ile macro-F1'i

Çıktı
    results/metrics/lambda_sweep.json
    results/figures/06_lambda_sweep.png

Kullanım
--------
    python src/lambda_sweep.py
    python src/lambda_sweep.py --lambdas 0.01 0.02 0.05 0.1 --pop 30 --generations 18
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from fs_common import FitnessEvaluator
from genetic_algorithm import run_ga
from utils import ROOT, compute_metrics, load_processed

METRICS_DIR = ROOT / "results" / "metrics"
FIG_DIR = ROOT / "results" / "figures"
SEED = 42


def full_test_eval(feats: list[str], ds) -> dict:
    """Alt kümeyi (train+val)'e fit edilmiş tam-ölçek RF ile test setinde değerlendir."""
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED, class_weight="balanced")
    rf.fit(ds.X_trainval[feats], ds.y_trainval)
    pred = rf.predict(ds.X_test[feats])
    m = compute_metrics(ds.y_test, pred, ds.class_names)
    return {"test_macro_f1": m["macro_f1"], "test_accuracy": m["accuracy"],
            "test_suspicious_f1": m["per_class"]["Suspicious"]["f1"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.01, 0.02, 0.03, 0.05, 0.10, 0.20])
    ap.add_argument("--pool", default="union", choices=["rf", "mi", "union", "overlap"])
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--generations", type=int, default=18)
    args = ap.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_processed()

    rows = []
    for lam in args.lambdas:
        print(f"\n########## λ = {lam} ##########")
        random.seed(SEED)
        np.random.seed(SEED)
        ev = FitnessEvaluator(pool=args.pool, lam=lam, seed=SEED, ds=ds)
        t0 = time.perf_counter()
        res = run_ga(ev, pop=args.pop, generations=args.generations, cxpb=0.6, mutpb=0.2,
                     mut_indpb=0.05, elite=2, init_p=0.35, verbose=False)
        best = res["best"]
        test = full_test_eval(best["features"], ds)
        row = {
            "lambda": lam,
            "n_features": best["n_features"],
            "val_macro_f1": best["val_macro_f1"],
            "fitness": best["fitness"],
            "test_macro_f1": test["test_macro_f1"],
            "test_accuracy": test["test_accuracy"],
            "test_suspicious_f1": test["test_suspicious_f1"],
            "features": best["features"],
            "ga_runtime_s": res["runtime_s"],
        }
        rows.append(row)
        print(f"  -> {row['n_features']:2d} öznitelik | val macro-F1 {row['val_macro_f1']:.4f} | "
              f"TEST macro-F1 {row['test_macro_f1']:.4f} | acc {row['test_accuracy']:.4f} | "
              f"({time.perf_counter() - t0:.0f}s)")

    (METRICS_DIR / "lambda_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # tablo
    print("\n| λ | Öznitelik | Val macro-F1 | Test macro-F1 | Test acc | Suspicious F1 |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['lambda']} | {r['n_features']} | {r['val_macro_f1']:.4f} | "
              f"{r['test_macro_f1']:.4f} | {r['test_accuracy']:.4f} | {r['test_suspicious_f1']:.4f} |")

    # grafik: sol eksen macro-F1, sağ eksen öznitelik sayısı, x = λ
    lams = [r["lambda"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(lams, [r["test_macro_f1"] for r in rows], "g-o", label="test macro-F1")
    ax1.plot(lams, [r["val_macro_f1"] for r in rows], "g--^", alpha=0.5, label="val macro-F1 (GA içi)")
    ax1.plot(lams, [r["test_suspicious_f1"] for r in rows], "m-s", alpha=0.7, label="test Suspicious F1")
    ax1.set_xlabel("λ (sadelik cezası ağırlığı)"); ax1.set_ylabel("macro-F1", color="g")
    ax1.set_xscale("log"); ax1.legend(loc="center left")
    ax2 = ax1.twinx()
    ax2.plot(lams, [r["n_features"] for r in rows], "r-D", label="öznitelik sayısı")
    ax2.axhspan(20, 25, color="orange", alpha=0.15)
    ax2.set_ylabel("seçilen öznitelik sayısı", color="r")
    for r in rows:
        ax2.annotate(str(r["n_features"]), (r["lambda"], r["n_features"]),
                     textcoords="offset points", xytext=(0, 6), ha="center", color="r")
    plt.title("λ taraması — sadelik / performans ödünleşimi (turuncu bant: 20-25 hedefi)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "06_lambda_sweep.png", dpi=120); plt.close(fig)

    print(f"\nKaydedildi -> {METRICS_DIR / 'lambda_sweep.json'}")


if __name__ == "__main__":
    main()
