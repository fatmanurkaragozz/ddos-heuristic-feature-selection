"""Adım 4a — Genetik Algoritma ile öznitelik seçimi (DEAP).

Temsil
------
Birey (kromozom) = havuz boyutu uzunluğunda 0/1 dizisi.
    gen[i] = 1  ->  i. aday öznitelik seçili
    gen[i] = 0  ->  seçili değil

Fitness  (fs_common.FitnessEvaluator ile — PSO ile BİREBİR aynı)
    fitness = macro_F1_val(seçilenler) - λ·(seçilen / havuz)

Akış (elitizmli nesil döngüsü)
    1. Rastgele popülasyon (her gen ~%35 olasılıkla 1 -> ~20-25 öznitelikle başla)
    2. Her nesil: turnuva seçimi -> uniform çaprazlama -> bit-flip mutasyon
    3. En iyi `--elite` birey doğrudan sonraki nesle taşınır (elitizm)
    4. `--generations` nesil sonra en iyi bireyin öznitelik alt kümesi = sonuç

Çıktı
    results/metrics/ga_result.json      (en iyi alt küme + nesil-nesil log)
    results/figures/04_ga_convergence.png

Kullanım
--------
    python src/genetic_algorithm.py
    python src/genetic_algorithm.py --lam 0.1 --pop 50 --generations 30 --pool union
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools

from fs_common import FitnessEvaluator
from utils import ROOT

METRICS_DIR = ROOT / "results" / "metrics"
FIG_DIR = ROOT / "results" / "figures"
SEED = 42

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def build_toolbox(pool_size: int, evaluator: FitnessEvaluator, init_p: float, mut_indpb: float):
    tb = base.Toolbox()
    tb.register("attr_bit", lambda: int(random.random() < init_p))
    tb.register("individual", tools.initRepeat, creator.Individual, tb.attr_bit, n=pool_size)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("mate", tools.cxUniform, indpb=0.5)
    tb.register("mutate", tools.mutFlipBit, indpb=mut_indpb)
    tb.register("select", tools.selTournament, tournsize=3)

    def _eval(ind):
        return (evaluator.evaluate(ind)["fitness"],)

    tb.register("evaluate", _eval)
    return tb


def fix_empty(ind) -> None:
    """Tamamen sıfır bireyi geçersizlikten kurtar — rastgele bir bit aç."""
    if sum(ind) == 0:
        ind[random.randrange(len(ind))] = 1


def run_ga(ev: FitnessEvaluator, *, pop: int, generations: int, cxpb: float, mutpb: float,
           mut_indpb: float, elite: int, init_p: float, verbose: bool = True) -> dict:
    """GA çekirdeği. `ev` fitness değerlendirici. En iyi alt küme + nesil logu döndürür.

    lambda_sweep.py da bu fonksiyonu çağırır — GA mantığı tek yerde.
    """
    tb = build_toolbox(ev.pool_size, ev, init_p, mut_indpb)
    population = tb.population(n=pop)
    for ind in population:
        fix_empty(ind)
        ind.fitness.values = tb.evaluate(ind)

    log = []
    t0 = time.perf_counter()
    for gen in range(1, generations + 1):
        offspring = [tb.clone(o) for o in tb.select(population, len(population) - elite)]
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                tb.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for m in offspring:
            if random.random() < mutpb:
                tb.mutate(m)
                del m.fitness.values
        for ind in offspring:
            fix_empty(ind)
            if not ind.fitness.valid:
                ind.fitness.values = tb.evaluate(ind)
        population = tools.selBest(population, elite) + offspring

        bi = ev.evaluate(tools.selBest(population, 1)[0])
        sizes = [sum(i) for i in population]
        log.append({
            "gen": gen,
            "best_fitness": float(bi["fitness"]), "best_macro_f1": float(bi["macro_f1"]),
            "best_n_features": int(bi["n_selected"]),
            "mean_fitness": float(np.mean([i.fitness.values[0] for i in population])),
            "mean_n_features": float(np.mean(sizes)),
        })
        if verbose:
            print(f"  nesil {gen:2d}: en iyi fitness={bi['fitness']:.4f}  macro-F1={bi['macro_f1']:.4f}  "
                  f"öznitelik={bi['n_selected']:2d}  (ort {np.mean(sizes):.1f}, "
                  f"benzersiz değerlendirme {ev.n_evals})")

    bi = ev.evaluate(tools.selBest(population, 1)[0])
    return {
        "runtime_s": time.perf_counter() - t0,
        "unique_evaluations": ev.n_evals,
        "best": {"n_features": bi["n_selected"], "val_macro_f1": bi["macro_f1"],
                 "fitness": bi["fitness"], "features": bi["features"]},
        "history": log,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="union", choices=["rf", "mi", "union", "overlap"])
    ap.add_argument("--lam", type=float, default=0.05, help="sadelik cezası ağırlığı λ")
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--generations", type=int, default=25)
    ap.add_argument("--cxpb", type=float, default=0.6, help="çaprazlama olasılığı")
    ap.add_argument("--mutpb", type=float, default=0.2, help="birey başına mutasyon olasılığı")
    ap.add_argument("--mut-indpb", type=float, default=0.05, help="mutasyonda bit başına çevirme olasılığı")
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--init-p", type=float, default=0.35, help="başlangıçta bir genin 1 olma olasılığı")
    ap.add_argument("--n-train", type=int, default=60_000)
    ap.add_argument("--n-val", type=int, default=30_000)
    args = ap.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    ev = FitnessEvaluator(pool=args.pool, lam=args.lam, n_train=args.n_train, n_val=args.n_val, seed=SEED)
    print(f"Havuz '{args.pool}': {ev.pool_size} aday öznitelik | λ={args.lam} | "
          f"pop={args.pop} nesil={args.generations}")
    print(f"Fitness alt örneklem: train {len(ev.y_tr):,} / val {len(ev.y_va):,}\n")

    result = run_ga(ev, pop=args.pop, generations=args.generations, cxpb=args.cxpb,
                    mutpb=args.mutpb, mut_indpb=args.mut_indpb, elite=args.elite, init_p=args.init_p)
    result["method"] = "genetic_algorithm"
    result["params"] = vars(args)
    result["pool_size"] = ev.pool_size
    log = result["history"]
    bi = result["best"]
    print(f"\nGA bitti ({result['runtime_s']:.0f}s, {result['unique_evaluations']} benzersiz değerlendirme)")
    print(f"En iyi alt küme: {bi['n_features']} öznitelik | val macro-F1 = {bi['val_macro_f1']:.4f} | "
          f"fitness = {bi['fitness']:.4f}")
    print("Öznitelikler:", bi["features"])

    (METRICS_DIR / "ga_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # yakınsama grafiği
    gens = [r["gen"] for r in log]
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(gens, [r["best_fitness"] for r in log], "b-o", ms=3, label="en iyi fitness")
    ax1.plot(gens, [r["mean_fitness"] for r in log], "b--", alpha=0.5, label="ortalama fitness")
    ax1.plot(gens, [r["best_macro_f1"] for r in log], "g-s", ms=3, label="en iyi macro-F1")
    ax1.set_xlabel("nesil"); ax1.set_ylabel("fitness / macro-F1", color="b"); ax1.legend(loc="lower right")
    ax2 = ax1.twinx()
    ax2.plot(gens, [r["best_n_features"] for r in log], "r-^", ms=3, label="en iyi bireyin öznitelik sayısı")
    ax2.plot(gens, [r["mean_n_features"] for r in log], "r--", alpha=0.4)
    ax2.set_ylabel("öznitelik sayısı", color="r"); ax2.legend(loc="upper right")
    plt.title(f"GA yakınsama (λ={args.lam}, havuz={args.pool})")
    plt.tight_layout(); plt.savefig(FIG_DIR / "04_ga_convergence.png", dpi=120); plt.close(fig)

    print(f"\nKaydedildi -> {METRICS_DIR / 'ga_result.json'}")
    print("Sonraki adım: src/pso.py")


if __name__ == "__main__":
    main()
