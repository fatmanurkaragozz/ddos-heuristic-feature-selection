"""Adım 3 — Baseline model (TÜM öznitelikler).

Ön işlemeden çıkan 183 özniteliğin HEPSİYLE model eğitir. Bu, GA/PSO ile
seçilmiş alt kümelerin kıyaslanacağı **referans nokta**.

Ne yapar
--------
1.  Random Forest (birincil) — (train+val) üzerinde 5-fold stratified CV.
2.  SVM-RBF (ikincil) — 378k satırda RBF-SVM pratik olmadığından stratified
    alt örneklem (varsayılan 60k) üzerinde 5-fold CV. "Yaklaşık" referans.
3.  Her iki model için (train+val)'e fit edip DOKUNULMAMIŞ test setinde tek
    seferlik değerlendirme.
4.  Kaydeder: results/metrics/baseline.json + karışıklık matrisi figürleri.

Metrikler: accuracy, macro-F1 (ana), macro precision/recall, sınıf-bazlı F1,
fit süresi. Dengesizlik nedeniyle her iki modelde `class_weight="balanced"`.

Kullanım
--------
    python src/baseline_model.py
    python src/baseline_model.py --rf-trees 300 --svm-subsample 40000 --skip-svm
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC

from utils import ROOT, compute_metrics, confusion, load_processed, text_report

METRICS_DIR = ROOT / "results" / "metrics"
FIG_DIR = ROOT / "results" / "figures"
RANDOM_STATE = 42

CV_SCORING = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "macro_precision": "precision_macro",
    "macro_recall": "recall_macro",
}


def summarize_cv(cv: dict) -> dict:
    """cross_validate çıktısını ortalama ± std sözlüğüne indirger."""
    out = {"fit_time_s": {"mean": float(np.mean(cv["fit_time"])), "std": float(np.std(cv["fit_time"]))}}
    for key in CV_SCORING:
        scores = cv[f"test_{key}"]
        out[key] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
    return out


def plot_confusion(y_true, y_pred, class_names, title, path: Path) -> None:
    cm = confusion(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax[0])
    ax[0].set_title(f"{title} — adet"); ax[0].set_xlabel("tahmin"); ax[0].set_ylabel("gerçek")
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax[1], vmin=0, vmax=1)
    ax[1].set_title(f"{title} — satır bazında oran"); ax[1].set_xlabel("tahmin"); ax[1].set_ylabel("gerçek")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def evaluate_model(name, model, ds, X_cv, y_cv, cv, do_test=True) -> dict:
    print(f"\n=== {name} ===")
    print(f"  5-fold CV ({len(y_cv):,} satır, {X_cv.shape[1]} öznitelik)...")
    t0 = time.perf_counter()
    # n_jobs=1: dıştan paralellik yok — model kendi içinde n_jobs=-1 kullandığından
    # çift paralellik (oversubscription) fit_time ölçümünü şişiriyordu.
    cv_res = cross_validate(model, X_cv, y_cv, cv=cv, scoring=CV_SCORING, n_jobs=1)
    cv_summary = summarize_cv(cv_res)
    print(f"  CV bitti ({time.perf_counter() - t0:.1f}s)  "
          f"macro-F1 = {cv_summary['macro_f1']['mean']:.4f} ± {cv_summary['macro_f1']['std']:.4f}  "
          f"accuracy = {cv_summary['accuracy']['mean']:.4f}")

    result = {"n_features": int(X_cv.shape[1]), "cv_rows": int(len(y_cv)), "cv": cv_summary}

    if do_test:
        print("  (train+val)'e fit ediliyor, test setinde değerlendiriliyor...")
        t0 = time.perf_counter()
        model.fit(ds.X_trainval, ds.y_trainval)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        y_pred = model.predict(ds.X_test)
        pred_s = time.perf_counter() - t0
        test_metrics = compute_metrics(ds.y_test, y_pred, ds.class_names)
        result["test"] = test_metrics
        result["full_fit_time_s"] = fit_s
        result["test_predict_time_s"] = pred_s
        print(f"  TEST  macro-F1 = {test_metrics['macro_f1']:.4f}  accuracy = {test_metrics['accuracy']:.4f}  "
              f"(fit {fit_s:.1f}s, predict {pred_s:.2f}s)")
        print(text_report(ds.y_test, y_pred, ds.class_names))
        plot_confusion(ds.y_test, y_pred, ds.class_names, name,
                       FIG_DIR / f"03_confusion_{name.lower().replace(' ', '_').replace('-', '')}.png")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rf-trees", type=int, default=200)
    ap.add_argument("--svm-subsample", type=int, default=60_000,
                    help="RBF-SVM CV'si için stratified alt örneklem boyutu")
    ap.add_argument("--skip-svm", action="store_true")
    args = ap.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_processed()
    print(f"Yüklendi: {ds.X_train.shape[1]} öznitelik | "
          f"train {len(ds.y_train):,}  val {len(ds.y_val):,}  test {len(ds.y_test):,}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {"dataset": {"n_features": ds.X_train.shape[1], "class_names": ds.class_names}}

    # --- Random Forest (birincil referans) --------------------------------
    rf = RandomForestClassifier(
        n_estimators=args.rf_trees, n_jobs=-1, random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    results["random_forest"] = evaluate_model(
        "Random Forest", rf, ds, ds.X_trainval, ds.y_trainval, cv,
    )

    # --- SVM-RBF (ikincil, alt örneklem) ---------------------------------
    if not args.skip_svm:
        rng = np.random.RandomState(RANDOM_STATE)
        Xtv, ytv = ds.X_trainval.reset_index(drop=True), ds.y_trainval
        n = min(args.svm_subsample, len(ytv))
        # stratified alt örneklem
        idx = np.hstack([
            rng.choice(np.where(ytv == c)[0], size=int(round(n * (ytv == c).mean())), replace=False)
            for c in np.unique(ytv)
        ])
        rng.shuffle(idx)
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE)
        res = evaluate_model("SVM-RBF", svm, ds, Xtv.iloc[idx], ytv[idx], cv, do_test=False)
        res["note"] = f"CV {n:,} satırlık stratified alt örneklem üzerinde; test değerlendirmesi atlandı (maliyet)."
        results["svm_rbf"] = res

    (METRICS_DIR / "baseline.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nKaydedildi -> {METRICS_DIR / 'baseline.json'}")
    print("Sonraki adım: src/genetic_algorithm.py")


if __name__ == "__main__":
    main()
