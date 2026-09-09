"""Adım 5 — Karşılaştırma: Baseline vs GA vs PSO vs basit yöntemler.

Aynı model ayarlarıyla (RandomForest, 200 ağaç, class_weight='balanced')
şu öznitelik alt kümelerini kıyaslar:

    all_183      : ön işleme sonrası tüm öznitelikler (referans)
    top_k_rf     : RF önem sıralamasında ilk K (K = GA'nın seçtiği sayı) — "naive"
    rfe          : Recursive Feature Elimination ile aynı sayıya inen — klasik wrapper
    ga           : Genetik Algoritma'nın bulduğu alt küme
    pso          : PSO'nun bulduğu alt küme

Her alt küme için:
    * (train+val) üzerinde 5-fold stratified CV  -> macro-F1, accuracy (ort ± std)
    * (train+val)'e fit, DOKUNULMAMIŞ test setinde tek seferlik değerlendirme
    * fit süresi, tahmin süresi
    * GA & PSO alt kümeleri ayrıca SVM-RBF ile de doğrulanır (alt örneklem)

Çıktı
    results/metrics/comparison.json
    results/comparison_table.md
    results/figures/05_comparison.png
    results/figures/05_confusion_<method>.png

Kullanım
--------
    python src/evaluate.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC

from fs_common import _stratified_subsample
from utils import ROOT, compute_metrics, confusion, load_processed, text_report

METRICS_DIR = ROOT / "results" / "metrics"
FIG_DIR = ROOT / "results" / "figures"
RESULTS = ROOT / "results"
SEED = 42
RF_TREES = 200

CV_SCORING = {"accuracy": "accuracy", "macro_f1": "f1_macro",
              "macro_precision": "precision_macro", "macro_recall": "recall_macro"}


def rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=RF_TREES, n_jobs=-1, random_state=SEED,
                                  class_weight="balanced")


def evaluate_subset(name: str, feats: list[str], ds, cv, run_svm: bool = False) -> dict:
    print(f"\n=== {name}  ({len(feats)} öznitelik) ===")
    Xtv = ds.X_trainval[feats].reset_index(drop=True)
    ytv = ds.y_trainval
    Xte = ds.X_test[feats]

    t0 = time.perf_counter()
    cv_res = cross_validate(rf(), Xtv, ytv, cv=cv, scoring=CV_SCORING, n_jobs=1)
    cv_time = time.perf_counter() - t0
    cv_summary = {k: {"mean": float(np.mean(cv_res[f"test_{k}"])),
                      "std": float(np.std(cv_res[f"test_{k}"]))} for k in CV_SCORING}
    print(f"  CV macro-F1 = {cv_summary['macro_f1']['mean']:.4f} ± {cv_summary['macro_f1']['std']:.4f}"
          f"  accuracy = {cv_summary['accuracy']['mean']:.4f}   ({cv_time:.0f}s)")

    model = rf()
    t0 = time.perf_counter(); model.fit(Xtv, ytv); fit_s = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = model.predict(Xte); pred_s = time.perf_counter() - t0
    test_metrics = compute_metrics(ds.y_test, pred, ds.class_names)
    print(f"  TEST macro-F1 = {test_metrics['macro_f1']:.4f}  accuracy = {test_metrics['accuracy']:.4f}"
          f"   (fit {fit_s:.1f}s, predict {pred_s:.2f}s)")
    print(text_report(ds.y_test, pred, ds.class_names))

    cm = confusion(ds.y_test, pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=ds.class_names, yticklabels=ds.class_names, ax=ax)
    ax.set_title(f"{name} — test (satır oranı)"); ax.set_xlabel("tahmin"); ax.set_ylabel("gerçek")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"05_confusion_{name}.png", dpi=120); plt.close(fig)

    out = {
        "n_features": len(feats), "features": feats,
        "cv": cv_summary,
        "test": test_metrics,
        "fit_time_s": fit_s, "predict_time_s": pred_s,
    }

    if run_svm:
        idx_tr = _stratified_subsample(ytv, 60_000, SEED)
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=SEED)
        t0 = time.perf_counter(); svm.fit(Xtv.iloc[idx_tr], ytv[idx_tr]); svm_fit = time.perf_counter() - t0
        svm_pred = svm.predict(Xte)
        svm_metrics = compute_metrics(ds.y_test, svm_pred, ds.class_names)
        out["svm_test"] = svm_metrics
        out["svm_fit_time_s"] = svm_fit
        print(f"  SVM-RBF TEST macro-F1 = {svm_metrics['macro_f1']:.4f}  "
              f"accuracy = {svm_metrics['accuracy']:.4f}  (60k alt örneklemde fit, {svm_fit:.0f}s)")
    return out


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_processed()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    ga = json.loads((METRICS_DIR / "ga_result.json").read_text(encoding="utf-8"))
    pso = json.loads((METRICS_DIR / "pso_result.json").read_text(encoding="utf-8"))
    filt = json.loads((METRICS_DIR / "filter_selection.json").read_text(encoding="utf-8"))

    ga_feats = ga["best"]["features"]
    pso_feats = pso["best"]["features"]
    k = len(ga_feats)
    top_k_rf = list(filt["rf_importance"].keys())[:k]     # zaten skora göre sıralı

    print(f"GA {len(ga_feats)} öznitelik | PSO {len(pso_feats)} öznitelik | Top-K RF K={k}")

    # RFE — RF tabanlı, hız için 60k alt örneklem
    print(f"\nRFE çalışıyor (hedef {k} öznitelik, 60k alt örneklem)...")
    idx = _stratified_subsample(ds.y_train, 60_000, SEED)
    rfe = RFE(RandomForestClassifier(n_estimators=60, n_jobs=-1, random_state=SEED,
                                     class_weight="balanced"),
              n_features_to_select=k, step=5)
    rfe.fit(ds.X_train.iloc[idx], ds.y_train[idx])
    rfe_feats = ds.X_train.columns[rfe.support_].tolist()
    print(f"RFE seçti: {rfe_feats}")

    subsets = {
        "all_183": ds.features,
        "top_k_rf": top_k_rf,
        "rfe": rfe_feats,
        "ga": ga_feats,
        "pso": pso_feats,
    }

    results = {}
    for name, feats in subsets.items():
        results[name] = evaluate_subset(name, feats, ds, cv,
                                        run_svm=name in ("all_183", "ga", "pso"))

    # öznitelik örtüşmesi
    results["_overlap"] = {
        "ga_pso_common": sorted(set(ga_feats) & set(pso_feats)),
        "ga_only": sorted(set(ga_feats) - set(pso_feats)),
        "pso_only": sorted(set(pso_feats) - set(ga_feats)),
    }
    (METRICS_DIR / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # --- markdown tablo ---
    baseline_f1 = results["all_183"]["test"]["macro_f1"]
    baseline_time = results["all_183"]["fit_time_s"]
    rows = ["| Yöntem | Öznitelik | CV macro-F1 | Test macro-F1 | Test accuracy | Suspicious F1 | Fit (s) | Fit hızlanma |",
            "|---|---|---|---|---|---|---|---|"]
    label = {"all_183": "Baseline (tüm)", "top_k_rf": f"Top-{k} (RF önem)",
             "rfe": f"RFE-{k}", "ga": "GA", "pso": "PSO"}
    for name in subsets:
        r = results[name]
        rows.append(
            f"| {label[name]} | {r['n_features']} | "
            f"{r['cv']['macro_f1']['mean']:.4f} ± {r['cv']['macro_f1']['std']:.3f} | "
            f"{r['test']['macro_f1']:.4f} | {r['test']['accuracy']:.4f} | "
            f"{r['test']['per_class']['Suspicious']['f1']:.4f} | "
            f"{r['fit_time_s']:.1f} | {baseline_time / r['fit_time_s']:.1f}× |"
        )
    table_md = "\n".join(rows)
    (RESULTS / "comparison_table.md").write_text(
        f"# Karşılaştırma tablosu\n\nBaseline test macro-F1 = {baseline_f1:.4f}\n\n{table_md}\n",
        encoding="utf-8")
    print("\n" + table_md)

    # --- karşılaştırma grafiği ---
    names = list(subsets)
    disp = [label[n] for n in names]
    f1s = [results[n]["test"]["macro_f1"] for n in names]
    nfeat = [results[n]["n_features"] for n in names]
    times = [results[n]["fit_time_s"] for n in names]

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    c = ["#888", "#f0a", "#fa0", "#2a7", "#27a"]
    ax[0].bar(disp, f1s, color=c); ax[0].set_title("Test macro-F1"); ax[0].set_ylim(min(f1s) - 0.02, max(f1s) + 0.01)
    ax[0].axhline(baseline_f1, ls="--", c="k", alpha=0.5)
    for i, v in enumerate(f1s): ax[0].text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax[1].bar(disp, nfeat, color=c); ax[1].set_title("Öznitelik sayısı")
    for i, v in enumerate(nfeat): ax[1].text(i, v, str(v), ha="center", va="bottom")
    ax[2].bar(disp, times, color=c); ax[2].set_title("Fit süresi (s, train+val)")
    for i, v in enumerate(times): ax[2].text(i, v, f"{v:.0f}", ha="center", va="bottom")
    for a in ax: a.tick_params(axis="x", rotation=20)
    plt.tight_layout(); plt.savefig(FIG_DIR / "05_comparison.png", dpi=120); plt.close(fig)

    print(f"\nKaydedildi -> {METRICS_DIR / 'comparison.json'}, {RESULTS / 'comparison_table.md'}")


if __name__ == "__main__":
    main()
