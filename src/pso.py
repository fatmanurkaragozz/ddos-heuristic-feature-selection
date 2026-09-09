"""Adım 4b — Particle Swarm Optimization ile öznitelik seçimi (pyswarms).

Temsil
------
Her parçacığın konumu = havuz boyutu uzunluğunda 0/1 vektörü (BinaryPSO).
    pos[i] = 1  ->  i. aday öznitelik seçili

Fitness  (GA ile BİREBİR aynı — fs_common.FitnessEvaluator)
    fitness = macro_F1_val(seçilenler) - λ·(seçilen / havuz)
pyswarms maliyeti (cost) MİNİMİZE ettiği için  cost = -fitness  kullanılır.

PSO mantığı
    Her parçacık, kendi bulduğu en iyi konuma (pbest, bilişsel terim c1) ve
    sürünün en iyisine (gbest, sosyal terim c2) doğru "çekilir"; w atalet.
    BinaryPSO'da hız bir sigmoid ile bit çevirme olasılığına dönüşür.

Çıktı
    results/metrics/pso_result.json
    results/figures/04_pso_convergence.png

Kullanım
--------
    python src/pso.py
    python src/pso.py --lam 0.1 --particles 40 --iters 30 --pool union
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyswarms.discrete import BinaryPSO

from fs_common import FitnessEvaluator
from utils import ROOT

METRICS_DIR = ROOT / "results" / "metrics"
FIG_DIR = ROOT / "results" / "figures"
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="union", choices=["rf", "mi", "union", "overlap"])
    ap.add_argument("--lam", type=float, default=0.05, help="sadelik cezası ağırlığı λ")
    ap.add_argument("--particles", type=int, default=30)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--c1", type=float, default=0.5, help="bilişsel katsayı (pbest'e çekim)")
    ap.add_argument("--c2", type=float, default=0.5, help="sosyal katsayı (gbest'e çekim)")
    ap.add_argument("--w", type=float, default=0.9, help="atalet ağırlığı")
    ap.add_argument("--n-train", type=int, default=60_000)
    ap.add_argument("--n-val", type=int, default=30_000)
    args = ap.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    ev = FitnessEvaluator(pool=args.pool, lam=args.lam, n_train=args.n_train, n_val=args.n_val, seed=SEED)
    print(f"Havuz '{args.pool}': {ev.pool_size} aday öznitelik | λ={args.lam} | "
          f"parçacık={args.particles} iterasyon={args.iters}")
    print(f"Fitness alt örneklem: train {len(ev.y_tr):,} / val {len(ev.y_va):,}\n")

    history: list[dict] = []
    running_best = {"fitness": -1.0}

    def objective(swarm: np.ndarray) -> np.ndarray:
        nonlocal running_best
        costs = np.empty(swarm.shape[0])
        iter_best = {"fitness": -1.0}
        for i, row in enumerate(swarm):
            r = ev.evaluate(row)
            costs[i] = -r["fitness"]
            if r["fitness"] > iter_best["fitness"]:
                iter_best = r
        if iter_best["fitness"] > running_best["fitness"]:
            running_best = iter_best
        history.append({
            "iter": len(history) + 1,
            "iter_best_fitness": float(iter_best["fitness"]),
            "best_fitness": float(running_best["fitness"]),
            "best_macro_f1": float(running_best["macro_f1"]),
            "best_n_features": int(running_best["n_selected"]),
            "mean_fitness": float(-costs.mean()),
            "unique_evals": ev.n_evals,
        })
        print(f"  iter {len(history):2d}: en iyi fitness={running_best['fitness']:.4f}  "
              f"macro-F1={running_best['macro_f1']:.4f}  öznitelik={running_best['n_selected']:2d}  "
              f"(benzersiz değerlendirme {ev.n_evals})")
        return costs

    options = {"c1": args.c1, "c2": args.c2, "w": args.w, "k": args.particles, "p": 2}
    optimizer = BinaryPSO(n_particles=args.particles, dimensions=ev.pool_size, options=options)

    t0 = time.perf_counter()
    cost, pos = optimizer.optimize(objective, iters=args.iters, verbose=False)
    runtime = time.perf_counter() - t0

    bi = ev.evaluate(pos)
    print(f"\nPSO bitti ({runtime:.0f}s, {ev.n_evals} benzersiz değerlendirme)")
    print(f"En iyi alt küme: {bi['n_selected']} öznitelik | val macro-F1 = {bi['macro_f1']:.4f} | "
          f"fitness = {bi['fitness']:.4f}")
    print("Öznitelikler:", bi["features"])

    result = {
        "method": "particle_swarm_optimization",
        "params": vars(args),
        "pool_size": ev.pool_size,
        "runtime_s": runtime,
        "unique_evaluations": ev.n_evals,
        "best": {
            "n_features": bi["n_selected"],
            "val_macro_f1": bi["macro_f1"],
            "fitness": bi["fitness"],
            "features": bi["features"],
        },
        "history": history,
    }
    (METRICS_DIR / "pso_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    its = [r["iter"] for r in history]
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(its, [r["best_fitness"] for r in history], "b-o", ms=3, label="en iyi fitness (gbest)")
    ax1.plot(its, [r["mean_fitness"] for r in history], "b--", alpha=0.5, label="ortalama fitness")
    ax1.plot(its, [r["best_macro_f1"] for r in history], "g-s", ms=3, label="en iyi macro-F1")
    ax1.set_xlabel("iterasyon"); ax1.set_ylabel("fitness / macro-F1", color="b"); ax1.legend(loc="lower right")
    ax2 = ax1.twinx()
    ax2.plot(its, [r["best_n_features"] for r in history], "r-^", ms=3, label="en iyi bireyin öznitelik sayısı")
    ax2.set_ylabel("öznitelik sayısı", color="r"); ax2.legend(loc="upper right")
    plt.title(f"PSO yakınsama (λ={args.lam}, havuz={args.pool})")
    plt.tight_layout(); plt.savefig(FIG_DIR / "04_pso_convergence.png", dpi=120); plt.close(fig)

    print(f"\nKaydedildi -> {METRICS_DIR / 'pso_result.json'}")
    print("Sonraki adım: src/evaluate.py")


if __name__ == "__main__":
    main()
