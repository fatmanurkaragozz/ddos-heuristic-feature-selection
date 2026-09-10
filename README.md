# DDoS Tespitinde Sezgisel/Metasezgisel Öznitelik Seçimi
# Heuristic / Metaheuristic Feature Selection for DDoS Detection

Fırat Üniversitesi – Yazılım Mühendisliği · staj/mentörlük projesi
Firat University – Software Engineering · internship/mentorship project

**🇹🇷 [Türkçe](#-türkçe)  ·  🇬🇧 [English](#-english)**

Ayrıntılı rapor / full report: [`results/REPORT.md`](results/REPORT.md)

---

# 🇹🇷 Türkçe

## Amaç

Bir bulut sunucusuna gelen ağ trafiğine bakıp **"normal kullanıcı mı, DDoS saldırısı mı?"**
sorusunu yanıtlayan bir model kuruyoruz. Veride her trafik kaydı **317 sayısal öznitelikle**
tanımlı — bu çok fazla. Asıl sorumuz:

> **317 öznitelikten hangi küçük alt kümesi (≈20) modeli neredeyse aynı, hatta daha iyi çalıştırır?**

Bu işe **öznitelik seçimi (feature selection)** denir. Az öznitelik → daha hızlı, daha az
gürültü, daha yorumlanabilir sistem. Alt kümeyi bulmak için **Genetik Algoritma (GA)** ve
**Particle Swarm Optimization (PSO)** kullanıyoruz; ikisini karşılaştırıyoruz.

### Öznitelik seçiminin 3 ailesi

| Aile | Nasıl çalışır | Hız | Örnek |
|---|---|---|---|
| **Filter** | Sadece istatistiğe bakar, model eğitmez | Çok hızlı | Korelasyon, Mutual Information |
| **Wrapper** | Her aday alt küme için model eğitir | Yavaş, isabetli | **GA, PSO**, RFE |
| **Embedded** | Model eğitilirken kendi eler | Orta | Lasso, ağaç önemi |

Biz **filter'ı ön eleme**, **wrapper'ı (GA/PSO) asıl seçim** için kullandık.

## Adım adım ne yaptık?

### 1. Veriyi tanıma — EDA (`notebooks/01_eda.ipynb`)

Modele geçmeden önce veriye "merhaba" deme aşaması; amaç sürprizleri önceden yakalamak.

- **540.494 satır** (her satır = bir ağ akışı), **317 öznitelik + 2 etiket**
- Hedef `label`: **Benign** (349k) / **Attack** (170k) / **Suspicious** (21k) — dengesizlik ~16.7×
- Eksik/bozuk değer yok; **50 öznitelik neredeyse sabit**; **568 çift öznitelik %95+ korele** (fazlalık!)

**Öğrenilen kavram — sınıf dengesizliği:** Model "her şeye Benign de" derse %65 doğruluk alır
ama işe yaramaz. Bu yüzden **accuracy tek başına yalancıdır.** Ana metriğimiz **macro-F1**:
her sınıfın F1'ini ayrı hesaplayıp düz ortalar → küçük sınıf da büyük kadar önemli.

### 2. Ön işleme (`src/preprocessing.py`)

**a) Veriyi üçe böldük — stratified 70 / 15 / 15:**

| Küme | Görevi |
|---|---|
| Train (%70) | Model bununla öğrenir |
| Validation (%15) | **GA/PSO buradaki skora bakarak öznitelik seçer** |
| Test (%15) | Sadece en sonda, bir kez — "gerçek" performans |

**Öğrenilen kavram — veri sızıntısı (data leakage):** GA/PSO binlerce kez model eğitip
skoruna bakacak. Bu skoru test'ten alsaydık, algoritma farkında olmadan test setine göre
optimize olur, sahte yüksek skor verirdi. Çözüm: GA/PSO **sadece validation'ı görür**, test
kilitli kalır. Aynı sebeple `StandardScaler` ve tüm filtreler **yalnızca train'den** öğrenilir.

**b) Aşama 0 — güvenli temizlik:** 3 sabit + 47 neredeyse-sabit + 84 aşırı korele
(`|r| > 0.98`) sütun atıldı → **317 → 183 öznitelik.** Bilgi kaybı yok, çöp temizliği.

**c) Ölçekleme:** Öznitelik ölçekleri çok farklı (byte oranları ~1e9, bazıları 0–1).
`StandardScaler` hepsini aynı ölçeğe çeker (train'e fit, val/test'e transform).

### 3. Baseline — referans nokta (`src/baseline_model.py`)

"Hiçbir seçim yapmasak ne olurdu?" ölçümü. 183 özniteliğin **hepsiyle** model eğittik.

**Öğrenilen kavram — çapraz doğrulama (5-fold CV):** Tek bir bölme şanslı/şansız olabilir.
5-fold: eğitim verisini 5 parçaya böl, 5 kez eğit, ortalamayı al → skor kararlı mı, tesadüf mü?

| Model | macro-F1 |
|---|---|
| Random Forest (tüm 183) | **0.869** (CV) / 0.869 (test) |
| SVM-RBF | 0.666 — çok kötü + çok yavaş |

- Sınıf bazında: Attack 0.95, Benign 0.99, **Suspicious 0.68** ← darboğaz burası
- CV skoru = test skoru → **overfitting (aşırı öğrenme) yok**, model sağlam
- **Karar:** GA/PSO döngüsünde Random Forest kullanılacak (SVM 60k satırda bile CV'si ~40 dk)

### 4. Öznitelik seçimi

**Aşama 1 — Filtre ile ön eleme (`src/filter_selection.py`): 183 → 65**

İki hızlı filtre (yalnızca train'den):
- **Random Forest önem skoru** → en iyi 60
- **Mutual Information** (öznitelik ↔ `label` bilgi paylaşımı) → en iyi 60

İki liste **55/60 örtüştü** → sinyal sağlam, tesadüf değil. Birleşim = **65 öznitelik**;
GA ve PSO bu havuzda arar.

**Fitness fonksiyonu (GA/PSO'nun pusulası) — `src/fs_common.py`:**

```
fitness = macro-F1(validation)  −  λ × (seçilen öznitelik sayısı / 65)
```

- 1. terim: az öznitelikle yüksek doğruluk = iyi
- 2. terim: çok öznitelik seçtin = ceza
- **λ (lambda):** iki hedef arasındaki denge ayarı — büyük λ → daha az öznitelik

> GA ve PSO **birebir aynı** fitness, aynı veri örneklemi (60k/30k), aynı model ayarını
> kullandı — yoksa karşılaştırma adil olmazdı.

**GA — doğal seçilim taklidi (`src/genetic_algorithm.py`, DEAP):**
Her aday = 65 bitlik dizi (1=seç, 0=seçme). 40 kromozomla başla → her nesilde iyileri
eşleştir (çaprazlama), biraz mutasyon, en iyi 2'yi koru (elitizm). 25 nesil.
→ **Sonuç: 15 öznitelik**, val macro-F1 0.887, 425 sn. Düzgün yakınsadı (~17. nesil).

**PSO — kuş sürüsü taklidi (`src/pso.py`, pyswarms BinaryPSO):**
Her aday = uzayda uçan parçacık; kendi en iyisine + sürünün en iyisine doğru çekilir.
30 parçacık, 25 iterasyon.
→ **Sonuç: 27 öznitelik**, val macro-F1 0.882, 743 sn. İlk 17 iterasyon takıldı, zayıf yakınsama.

### 5. Karşılaştırma (`src/evaluate.py`) — neyi kıyasladık?

**5 yöntemi** aynı model (Random Forest 200 ağaç), aynı 5-fold CV, aynı **dokunulmamış test seti** ile:

| Yöntem | Ne yapar | Öznitelik | Test macro-F1 | Test acc | Suspicious F1 |
|---|---|---:|---:|---:|---:|
| Baseline | seçim yok, 183'ün hepsi | 183 | 0.8693 | 0.9587 | 0.676 |
| Top-15 | "en önemli 15'i al" (basit) | 15 | 0.8970 | 0.9712 | 0.738 |
| RFE-15 | "her turda en zayıfı at" (klasik) | 15 | 0.8932 | 0.9697 | 0.729 |
| **GA** | genetik arama | **15** | **0.9065** | **0.9747** | **0.761** |
| PSO | sürü araması | 27 | 0.9004 | 0.9725 | 0.746 |

**Amaç iki soruyu cevaplamak:**
1. *Öznitelik seçimi işe yarıyor mu?* → **Evet.** Hepsi baseline'ı geçti (0.869 → 0.907).
2. *GA/PSO gibi "akıllı" yöntemler basit yöntemlerden iyi mi?* → **Evet.** GA (0.907), aynı
   15 öznitelik bütçesinde Top-15 (0.897) ve RFE-15'i (0.893) geçti.

**Neden GA basit yöntemleri geçti?** Top-15 her özniteliğe **tek tek** bakar; GA öznitelikleri
**birlikte** değerlendirir — "A ve B tek başına zayıf, birlikte güçlü" ilişkilerini yakalar.
Kazanç çoğunlukla zor sınıfta: **Suspicious F1 0.676 → 0.761.**

### 6. λ taraması (`src/lambda_sweep.py`)

Mentör 20-25 öznitelik istedi; GA λ=0.05'te 15 verdi. λ'yı 6 değerde denedik:

| λ | Öznitelik | Test macro-F1 |
|---:|---:|---:|
| 0.01 | 15 | 0.901 |
| 0.02 | **20** | 0.903 |
| 0.05 | 17 | 0.902 |
| 0.20 | 12 | 0.904 |

**Bulgu:** λ ne olursa olsun öznitelik sayısı 12-20, macro-F1 hep ~0.90. Bu veride sinyal
**~15 özniteliğe sıkışıyor**; 20-25'in ölçülebilir faydası yok. (20 gerekirse λ=0.02.)

**Öğrenilen kavram — trade-off (ödünleşim):** İki hedef çatışırsa (az öznitelik ↔ yüksek
doğruluk), bir parametre (λ) ile aralarında gezinip eğri çizersin. Tek bir sonuçtan çok
daha değerli bir içgörü verir.

## Bu projeden çıkması gereken 5 ders

1. **Veriyi modelden önce tanı (EDA)** — fazlalık, dengesizlik, sızıntı riskini önceden gör.
2. **Sızıntıyı önle** — train/validation/test ayrı; scaler ve filtreler sadece train'den; test en sona kilitli.
3. **Doğru metriği seç** — dengesiz veride accuracy değil, macro-F1.
4. **Adil karşılaştır** — tüm yöntemler aynı model, aynı CV, aynı test; GA ve PSO aynı fitness.
5. **Sonuç bir sayı değil, bir eğridir** — λ taraması "15 öznitelik yeter" içgörüsünü verdi.

**Ana sonuç:** 317 öznitelikten 15'ine indik, performans **arttı** (macro-F1 0.869 → 0.907),
model ~2× hızlandı; seçilen 15 özniteliğin hepsi DDoS literatüründe anlamlı (paketler arası
süre, paket boyutu tekdüzeliği, TCP bayrak oranları).

## Kurulum ve çalıştırma

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# Veriyi data/raw/archive/ altına indirin (Kaggle → Download → zip'i aç)

python src/preprocessing.py       # 317 -> 183; stratified 70/15/15
python src/baseline_model.py      # tüm özniteliklerle RF + SVM (referans)
python src/filter_selection.py    # Aşama 1 filtre: 183 -> 65
python src/genetic_algorithm.py   # GA (DEAP)
python src/pso.py                 # PSO (pyswarms)
python src/evaluate.py            # Baseline vs Top-K vs RFE vs GA vs PSO
python src/lambda_sweep.py        # λ ödünleşim eğrisi
```

Tüm betikler `random_state=42`. Çıktılar: `results/metrics/*.json`, `results/figures/*.png`.

## Klasör yapısı

```
data/raw/          ham veri (git'e girmez)
data/processed/    işlenmiş veri + scaler/encoder (git'e girmez)
notebooks/         01_eda.ipynb — keşifsel analiz
src/               pipeline betikleri
results/
  metrics/         *.json — tüm sayısal sonuçlar
  figures/         *.png — grafikler
  REPORT.md        mentör raporu
  comparison_table.md
```

| Betik | Açıklama |
|---|---|
| `src/preprocessing.py` | Temizlik, encode, stratified bölme, StandardScaler |
| `src/baseline_model.py` | Tüm özniteliklerle 5-fold CV + test |
| `src/filter_selection.py` | RF önem + Mutual Information ile ön eleme |
| `src/fs_common.py` | GA ve PSO'nun paylaştığı fitness fonksiyonu |
| `src/genetic_algorithm.py` | GA (elitizmli nesil döngüsü) |
| `src/pso.py` | BinaryPSO |
| `src/evaluate.py` | Karşılaştırma tablosu + figürler |
| `src/lambda_sweep.py` | λ taraması |
| `src/utils.py` | Ortak veri yükleme + metrik hesaplama |

## Veri seti

**BCCC-cPacket-Cloud-DDoS-2024** — [Kaggle (`dhoogla` temizlenmiş parquet)](https://www.kaggle.com/datasets/dhoogla/bccc-cpacket-cloud-ddos-2024)
York Üniversitesi (BCCC) + cPacket; NTLFlowLyzer ile çıkarılmış 317 akış özniteliği.
Referans: Shafi, Lashkari, Rodriguez, Nevo (2024). *Toward Generating a New Cloud-Based
Distributed Denial of Service (DDoS) Dataset...* Information 15(4).

---

# 🇬🇧 English

## Goal

We build a model that looks at network traffic hitting a cloud server and answers
**"legitimate user or DDoS attack?"** Each traffic record is described by **317 numeric
features** — too many. The real question:

> **Which small subset of the 317 features (≈20) makes the model run about as well, or better?**

This is **feature selection**. Fewer features → faster, less noise, more interpretable.
To find the subset we use a **Genetic Algorithm (GA)** and **Particle Swarm Optimization
(PSO)**, and we compare the two.

### Three families of feature selection

| Family | How it works | Speed | Example |
|---|---|---|---|
| **Filter** | Statistics only, trains no model | Very fast | Correlation, Mutual Information |
| **Wrapper** | Trains a model for each candidate subset | Slow, accurate | **GA, PSO**, RFE |
| **Embedded** | The model prunes during training | Medium | Lasso, tree importance |

We used a **filter as pre-screening** and a **wrapper (GA/PSO) as the main selector**.

## What we did, step by step

### 1. Get to know the data — EDA (`notebooks/01_eda.ipynb`)

Say "hello" to the data before modelling; catch surprises early.

- **540,494 rows** (each row = one network flow), **317 features + 2 labels**
- Target `label`: **Benign** (349k) / **Attack** (170k) / **Suspicious** (21k) — imbalance ~16.7×
- No missing/broken values; **50 near-constant features**; **568 feature pairs correlated >0.95** (redundancy!)

**Concept — class imbalance:** A model that "always says Benign" scores 65% accuracy but is
useless. So **accuracy alone is misleading.** Our main metric is **macro-F1**: compute each
class's F1 separately and take the plain average → the small class counts as much as the big one.

### 2. Preprocessing (`src/preprocessing.py`)

**a) Three-way stratified split — 70 / 15 / 15:**

| Set | Role |
|---|---|
| Train (70%) | The model learns from this |
| Validation (15%) | **GA/PSO pick features by looking at this score** |
| Test (15%) | Touched only once, at the very end — the "real" performance |

**Concept — data leakage:** GA/PSO will train a model thousands of times and read its score.
If that score came from the test set, the algorithm would quietly optimise against the test
set and produce a fake-high number. Fix: GA/PSO **only see validation**; the test set stays
locked. For the same reason `StandardScaler` and all filters are fit **on train only**.

**b) Stage 0 — safe cleanup:** dropped 3 constant + 47 near-constant + 84 highly-correlated
(`|r| > 0.98`) columns → **317 → 183 features.** No information lost, just junk removed.

**c) Scaling:** feature scales differ wildly (byte rates ~1e9, some ratios 0–1).
`StandardScaler` puts them on one scale (fit on train, transform val/test).

### 3. Baseline — the reference point (`src/baseline_model.py`)

"What if we selected nothing?" We trained a model on **all 183** features.

**Concept — cross-validation (5-fold CV):** a single split can be lucky or unlucky.
5-fold: split the training data into 5 parts, train 5 times, average → is the score stable
or a fluke?

| Model | macro-F1 |
|---|---|
| Random Forest (all 183) | **0.869** (CV) / 0.869 (test) |
| SVM-RBF | 0.666 — poor and very slow |

- Per class: Attack 0.95, Benign 0.99, **Suspicious 0.68** ← the bottleneck
- CV score = test score → **no overfitting**, the model is sound
- **Decision:** use Random Forest in the GA/PSO loop (SVM's CV takes ~40 min even on 60k rows)

### 4. Feature selection

**Stage 1 — filter pre-screening (`src/filter_selection.py`): 183 → 65**

Two fast filters (train only):
- **Random Forest importance** → top 60
- **Mutual Information** (feature ↔ `label` shared information) → top 60

The two lists **overlap 55/60** → the signal is robust, not a coincidence. Union = **65
features**; GA and PSO search inside this pool.

**Fitness function (the compass for GA/PSO) — `src/fs_common.py`:**

```
fitness = macro-F1(validation)  −  λ × (number of selected features / 65)
```

- term 1: high accuracy with few features = good
- term 2: you picked too many features = penalty
- **λ (lambda):** the balance knob between the two goals — larger λ → fewer features

> GA and PSO used the **exact same** fitness, the same data subsample (60k/30k) and the same
> model settings — otherwise the comparison wouldn't be fair.

**GA — mimics natural selection (`src/genetic_algorithm.py`, DEAP):**
Each candidate = a 65-bit string (1 = pick, 0 = drop). Start with 40 chromosomes → each
generation: mate the good ones (crossover), a little mutation, keep the best 2 (elitism).
25 generations. → **Result: 15 features**, validation macro-F1 0.887, 425 s. Converged cleanly (~gen 17).

**PSO — mimics a bird flock (`src/pso.py`, pyswarms BinaryPSO):**
Each candidate = a particle flying through the space; pulled toward its own best and the
swarm's best. 30 particles, 25 iterations.
→ **Result: 27 features**, validation macro-F1 0.882, 743 s. Stuck for the first 17 iterations, weak convergence.

### 5. Comparison (`src/evaluate.py`) — what did we compare?

**5 methods**, all with the same model (Random Forest, 200 trees), the same 5-fold CV, the
same **untouched test set**:

| Method | What it does | Features | Test macro-F1 | Test acc | Suspicious F1 |
|---|---|---:|---:|---:|---:|
| Baseline | no selection, all 183 | 183 | 0.8693 | 0.9587 | 0.676 |
| Top-15 | "take the 15 most important" (naive) | 15 | 0.8970 | 0.9712 | 0.738 |
| RFE-15 | "drop the weakest each round" (classic) | 15 | 0.8932 | 0.9697 | 0.729 |
| **GA** | genetic search | **15** | **0.9065** | **0.9747** | **0.761** |
| PSO | swarm search | 27 | 0.9004 | 0.9725 | 0.746 |

**The aim is to answer two questions:**
1. *Does feature selection help?* → **Yes.** Every method beat the baseline (0.869 → 0.907).
2. *Are "smart" methods (GA/PSO) better than naive ones?* → **Yes.** GA (0.907) beat Top-15
   (0.897) and RFE-15 (0.893) at the same 15-feature budget.

**Why did GA beat the naive methods?** Top-15 looks at each feature **individually**; GA
evaluates features **together** — it catches "A and B are weak alone but strong together"
relationships. The gain is mostly in the hard class: **Suspicious F1 0.676 → 0.761.**

### 6. λ sweep (`src/lambda_sweep.py`)

The mentor asked for 20-25 features; GA gave 15 at λ=0.05. We tried 6 values of λ:

| λ | Features | Test macro-F1 |
|---:|---:|---:|
| 0.01 | 15 | 0.901 |
| 0.02 | **20** | 0.903 |
| 0.05 | 17 | 0.902 |
| 0.20 | 12 | 0.904 |

**Finding:** whatever λ is, the feature count is 12-20 and macro-F1 stays ~0.90. In this
dataset the signal **compresses to ~15 features**; going to 20-25 brings no measurable
benefit. (If exactly 20 are required, use λ=0.02.)

**Concept — trade-off:** when two goals conflict (few features ↔ high accuracy), you move
between them with a parameter (λ) and draw a curve. That insight is far more valuable than
a single number.

## Five lessons from this project

1. **Know the data before modelling (EDA)** — spot redundancy, imbalance, leakage risk early.
2. **Prevent leakage** — separate train/validation/test; fit scaler and filters on train only; keep test locked until the end.
3. **Pick the right metric** — on imbalanced data, macro-F1, not accuracy.
4. **Compare fairly** — all methods on the same model, same CV, same test; GA and PSO on the same fitness.
5. **A result is a curve, not a number** — the λ sweep gave the insight "15 features is enough".

**Bottom line:** from 317 features down to 15, performance **improved** (macro-F1 0.869 →
0.907), the model got ~2× faster, and all 15 selected features are meaningful in the DDoS
literature (inter-arrival times, packet-size uniformity, TCP flag ratios).

## Setup and run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# Download the data into data/raw/archive/ (Kaggle -> Download -> unzip)

python src/preprocessing.py       # 317 -> 183; stratified 70/15/15
python src/baseline_model.py      # RF + SVM on all features (reference)
python src/filter_selection.py    # Stage 1 filter: 183 -> 65
python src/genetic_algorithm.py   # GA (DEAP)
python src/pso.py                 # PSO (pyswarms)
python src/evaluate.py            # Baseline vs Top-K vs RFE vs GA vs PSO
python src/lambda_sweep.py        # λ trade-off curve
```

All scripts use `random_state=42`. Outputs: `results/metrics/*.json`, `results/figures/*.png`.

## Folder layout

```
data/raw/          raw data (git-ignored)
data/processed/    processed data + scaler/encoder (git-ignored)
notebooks/         01_eda.ipynb — exploratory analysis
src/               pipeline scripts
results/
  metrics/         *.json — all numeric results
  figures/         *.png — plots
  REPORT.md        mentor report
  comparison_table.md
```

## Dataset

**BCCC-cPacket-Cloud-DDoS-2024** — [Kaggle (`dhoogla` cleaned parquet)](https://www.kaggle.com/datasets/dhoogla/bccc-cpacket-cloud-ddos-2024)
York University (BCCC) + cPacket; 317 flow features extracted with NTLFlowLyzer.
Reference: Shafi, Lashkari, Rodriguez, Nevo (2024). *Toward Generating a New Cloud-Based
Distributed Denial of Service (DDoS) Dataset...* Information 15(4).
