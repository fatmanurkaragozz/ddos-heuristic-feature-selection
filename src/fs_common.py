"""GA ve PSO'nun PAYLAŞTIĞI fitness altyapısı.

Adil karşılaştırma için iki yöntem de birebir aynı fitness fonksiyonunu,
aynı alt örneklemi ve aynı model ayarlarını kullanmak zorunda.

Fitness
-------
    fitness(S) = macro_F1_val(S)  -  λ · ( |S| / |havuz| )

  * macro_F1_val(S): S öznitelik alt kümesiyle train alt örnekleminde eğitilen
    Random Forest'ın, validation alt örneklemindeki macro-F1'i.
  * λ (lambda): sadelik cezasının ağırlığı. Büyük λ -> daha az öznitelik.
  * |havuz|: Aşama 1 filtresinden gelen aday öznitelik sayısı (arama uzayı).

Hız
---
GA/PSO binlerce değerlendirme yapar. Her değerlendirmede tüm train'i (378k)
kullanmak saatler sürer; bu yüzden sabit, stratified bir alt örneklem
(varsayılan train 60k / val 30k) ve hafif bir RF (80 ağaç) kullanılır.
Sonuçlar ayrıca aynı bireyi tekrar görünce diye önbelleğe alınır.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from utils import ROOT, Dataset, load_processed

PROCESSED = ROOT / "data" / "processed"
RANDOM_STATE = 42


def load_pool(which: str = "union") -> list[str]:
    """Aşama 1 aday havuzunu döndürür: 'rf' | 'mi' | 'union' | 'overlap'."""
    pools = json.loads((PROCESSED / "filter_pools.json").read_text(encoding="utf-8"))
    key = {
        "rf": "pool_rf_importance",
        "mi": "pool_mutual_info",
        "union": "union",
        "overlap": "overlap",
    }[which]
    return list(pools[key])


def _stratified_subsample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n >= len(y):
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        take = int(round(n * len(c_idx) / len(y)))
        idx.append(rng.choice(c_idx, size=min(take, len(c_idx)), replace=False))
    out = np.concatenate(idx)
    rng.shuffle(out)
    return out


class FitnessEvaluator:
    """GA/PSO için paylaşılan fitness. `evaluate(mask)` -> dict döndürür."""

    def __init__(
        self,
        pool: str = "union",
        lam: float = 0.05,
        n_train: int = 60_000,
        n_val: int = 30_000,
        rf_trees: int = 80,
        seed: int = RANDOM_STATE,
        ds: Dataset | None = None,
    ):
        ds = ds or load_processed()
        self.features = load_pool(pool)
        self.pool_name = pool
        self.lam = lam
        self.rf_trees = rf_trees
        self.seed = seed
        self.class_names = ds.class_names

        tr = _stratified_subsample(ds.y_train, n_train, seed)
        va = _stratified_subsample(ds.y_val, n_val, seed + 1)
        self.X_tr = ds.X_train.iloc[tr][self.features].to_numpy(dtype=np.float32)
        self.y_tr = ds.y_train[tr]
        self.X_va = ds.X_val.iloc[va][self.features].to_numpy(dtype=np.float32)
        self.y_va = ds.y_val[va]

        self._cache: dict[tuple[int, ...], dict] = {}
        self.n_evals = 0

    # ------------------------------------------------------------------ #
    def evaluate(self, mask) -> dict:
        """mask: 0/1 dizisi (uzunluk = havuz boyutu).

        Döndürür: {fitness, macro_f1, n_selected, features}
        """
        mask = tuple(int(b) for b in mask)
        if mask in self._cache:
            return self._cache[mask]

        sel = np.array(mask, dtype=bool)
        n_sel = int(sel.sum())
        if n_sel == 0:                      # boş alt küme -> geçersiz
            res = {"fitness": -1.0, "macro_f1": 0.0, "n_selected": 0, "features": []}
            self._cache[mask] = res
            return res

        rf = RandomForestClassifier(
            n_estimators=self.rf_trees, n_jobs=-1, random_state=self.seed,
            class_weight="balanced",
        )
        rf.fit(self.X_tr[:, sel], self.y_tr)
        pred = rf.predict(self.X_va[:, sel])
        macro_f1 = float(f1_score(self.y_va, pred, average="macro"))
        penalty = self.lam * (n_sel / len(self.features))
        fitness = macro_f1 - penalty

        res = {
            "fitness": fitness,
            "macro_f1": macro_f1,
            "n_selected": n_sel,
            "features": [f for f, s in zip(self.features, sel) if s],
        }
        self._cache[mask] = res
        self.n_evals += 1
        return res

    @property
    def pool_size(self) -> int:
        return len(self.features)
