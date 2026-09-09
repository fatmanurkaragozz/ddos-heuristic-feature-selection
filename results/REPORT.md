# Sezgisel/Metasezgisel Öznitelik Seçimi ile Bulut DDoS Tespiti — Rapor

**Öğrenci:** Fatma Nur Karagöz — Fırat Üniversitesi, Yazılım Mühendisliği
**Veri seti:** BCCC-cPacket-Cloud-DDoS-2024 (`dhoogla` temizlenmiş parquet sürümü)
**Amaç:** Genetik Algoritma (GA) ve Particle Swarm Optimization (PSO) ile öznitelik
seçimi yapıp, seçilen alt kümelerin model performansına etkisini incelemek.

---

## 1. Özet (yönetici özeti)

| Yöntem | Öznitelik | Test macro-F1 | Test accuracy | Suspicious F1 | Fit süresi |
|---|---:|---:|---:|---:|---:|
| Baseline (tüm öznitelikler) | 183 | 0.8693 | 0.9587 | 0.676 | 42.3 sn |
| Top-15 (RF önem sıralaması) | 15 | 0.8970 | 0.9712 | 0.738 | 28.1 sn |
| RFE-15 (klasik wrapper) | 15 | 0.8932 | 0.9697 | 0.729 | 23.8 sn |
| **GA** | **15** | **0.9065** | **0.9747** | **0.761** | 22.7 sn |
| PSO | 27 | 0.9004 | 0.9725 | 0.746 | 24.7 sn |

**Ana çıkarımlar:**

1. **Öznitelik seçimi yapan her yöntem baseline'ı geçti.** 183 öznitelikten 15'e inince
   test macro-F1 **0.869 → 0.907** yükseldi. Fazla öznitelikler modele yardım etmiyor,
   gürültü ve fazlalık olarak zarar veriyordu.
2. **Kazanç zor sınıfta (`Suspicious`) yoğunlaştı:** F1 0.676 → 0.761.
3. **GA, hem baseline'ı hem de basit yöntemleri (Top-15, RFE-15) aynı öznitelik bütçesinde geçti.**
   Metasezgisel arama, öznitelikleri tek tek değil birlikte değerlendirdiği için üstün.
4. **GA > PSO:** GA 15 öznitelikle 0.9065; PSO 27 öznitelikle 0.9004. GA daha az öznitelik,
   daha yüksek skor, daha hızlı yakınsama (425 sn / 743 sn).
5. **λ taraması:** λ ∈ [0.01, 0.20] boyunca GA 12-20 öznitelikte kalıyor, test macro-F1
   ~0.90'da sabit. Bu veri setinde DDoS sinyali **~15 öznitelikte** yaşıyor; 20-25'e
   çıkmak ölçülebilir bir fayda getirmiyor. (20 öznitelik gerekirse λ=0.02 kullanılır.)

---

## 2. Veri seti ve ön işleme

### 2.1 Veri seti

- **540.494 akış kaydı**, her biri NTLFlowLyzer ile çıkarılmış **317 sayısal öznitelik**.
- Her satır bir **ağ akışı** (aynı 5'li: kaynak IP/port, hedef IP/port, protokol).
- İki etiket: `label` (3 sınıf) ve `activity` (26 ince taneli senaryo).
- **Hedef değişken: `label`** — Benign (349.178) / Attack (170.436) / Suspicious (20.880).
  Sınıf dengesizliği ~16.7x → ana metrik **macro-F1** (her sınıf eşit ağırlıkta).

### 2.2 Ön işleme (`src/preprocessing.py`)

1. `activity` sütunu atıldı; `label` sayıya çevrildi (Attack=0, Benign=1, Suspicious=2).
2. **Stratified 70/15/15 bölme** → train 378.345 / validation 81.074 / test 81.075.
   `random_state=42`. Test seti final rapora kadar hiç kullanılmadı.
3. **Aşama 0 güvenli temizlik** (yalnızca train istatistiğinden):
   - 3 sabit sütun (tek değerli)
   - 47 neredeyse-sabit sütun (bir değer satırların > %99.9'unu kaplıyor)
   - 84 aşırı korele sütun (`|Pearson r| > 0.98` olan çiftlerden biri)
   - **Sonuç: 317 → 183 öznitelik.** Bilgi kaybı yok, sadece fazlalık temizliği.
4. `StandardScaler` — train'e fit, validation + test'e transform (sızıntı önleme).

---

## 3. Yöntem: 3 aşamalı boru hattı

Metasezgisel arama pahalıdır (binlerce model eğitimi). Bu yüzden 3 aşamalı bir tasarım:

### Aşama 0 — Güvenli temizlik
Yukarıda: 317 → 183 (ön işlemenin parçası).

### Aşama 1 — Filter ile ön eleme (`src/filter_selection.py`)
İki hızlı, model-hafif filtre (yalnızca train'den):
- **Random Forest önem skoru** — en önemli 60 öznitelik
- **Mutual Information** — `label` ile en çok bilgi paylaşan 60 öznitelik

**Sonuç:** İki havuz **55/60 örtüşüyor** (birleşim = 65). İki bağımsız yöntem neredeyse
aynı öznitelikleri seçti → sinyal sağlam. Her ikisinin de en tepesinde: paketler arası
süre ortalaması (`fwd_packets_IAT_mean`, `packets_IAT_mean`), başlık boyutları, TCP başlangıç
pencere boyutu (`fwd_init_win_bytes`), portlar, paket hızı.

GA ve PSO, bu **65 özniteliklik birleşim havuzu** üzerinde arama yaptı.

### Aşama 2 — GA ve PSO (`src/genetic_algorithm.py`, `src/pso.py`)

**Ortak fitness fonksiyonu** (`src/fs_common.py` — adil kıyas için birebir aynı):

```
fitness(S) = macro_F1_val(S)  −  λ · ( |S| / 65 )
```

- `macro_F1_val(S)`: S alt kümesiyle train alt örnekleminde (60.000 satır, stratified)
  eğitilen Random Forest'ın validation alt örneklemindeki (30.000 satır) macro-F1'i.
- `λ` (lambda): sadelik cezasının ağırlığı. Varsayılan 0.05.
- Hız için hafif RF (80 ağaç) + sonuç önbelleği kullanıldı.

**GA parametreleri:** popülasyon 40, 25 nesil, turnuva seçimi (k=3), uniform çaprazlama
(p=0.6), bit-flip mutasyon (birey p=0.2, bit p=0.05), elitizm 2 birey.

**PSO parametreleri:** 30 parçacık, 25 iterasyon, `c1=c2=0.5`, `w=0.9`, BinaryPSO (sigmoid
hız → bit çevirme olasılığı).

---

## 4. Sonuçlar

### 4.1 Baseline (`src/baseline_model.py`)

| Model | Değerlendirme | Accuracy | macro-F1 |
|---|---|---:|---:|
| Random Forest | 5-fold CV | 0.9589 | 0.8695 ± 0.002 |
| Random Forest | Test (81k) | 0.9587 | 0.8693 |
| SVM-RBF | 5-fold CV (60k örneklem) | 0.8049 | 0.6660 |

- CV ve test skorları birebir → **overfitting yok**.
- Sınıf bazında (RF, test): Attack F1 0.945, Benign F1 0.987, **Suspicious F1 0.676**.
  `Suspicious` darboğaz: model bu sınıfı yakalıyor (recall 0.88) ama çok yanlış alarm
  veriyor (precision 0.55).
- **SVM-RBF her öznitelik setinde ~0.66 macro-F1** aldı (baseline, GA, PSO fark etmeksizin).
  Varsayılan RBF ayarı bu veriye uygun değil ve 540k satırda pratik dışı yavaş
  (60k örneklemde bile CV ~40 dk). → **Random Forest birincil model olarak seçildi;
  SVM GA/PSO döngüsünde kullanılmadı.**

### 4.2 GA ve PSO

| | Öznitelik | Fitness | Val macro-F1 | Süre | Benzersiz değerlendirme |
|---|---:|---:|---:|---:|---:|
| GA (λ=0.05) | 15 | 0.8750 | 0.8865 | 425 sn | 481 |
| PSO (λ=0.05) | 27 | 0.8612 | 0.8820 | 743 sn | 694 |

**Yakınsama davranışı** (bkz. `figures/04_ga_convergence.png`, `04_pso_convergence.png`):
- **GA:** Düzgün iniyor, ~17. nesilde oturuyor. Öznitelik sayısı 24 → 15'e düşerken
  macro-F1 düşmüyor.
- **PSO:** İlk 17 iterasyon 29 öznitelikte takılı kaldı, sonra küçük bir iyileşme (27).
  BinaryPSO'nun bilinen zayıflığı: `c1=c2=0.5, w=0.9` ayarıyla sürü, rastgele başlangıç
  yoğunluğunun (havuzun ~yarısı) etrafında dolanıyor.

### 4.3 Adil karşılaştırma (`src/evaluate.py`) — tümü aynı RF (200 ağaç), 5-fold CV + test

(Bkz. Bölüm 1 tablosu ve `figures/05_comparison.png`.)

- **Baseline (183) → GA (15):** macro-F1 +0.037, accuracy +0.016, Suspicious F1 +0.085,
  fit süresi ~1.9x hızlanma, tahmin süresi 0.58 sn → 0.33 sn.
- **GA (15) > Top-15 (0.897) > RFE-15 (0.893):** aynı öznitelik sayısında GA kazandı.
- **PSO (27) = 0.9004:** GA'nın 15 öznitelikle aldığından hem düşük hem ~2x fazla öznitelikle.

### 4.4 λ taraması (`src/lambda_sweep.py`)

| λ | Öznitelik | Test macro-F1 | Test acc | Suspicious F1 |
|---:|---:|---:|---:|---:|
| 0.01 | 15 | 0.9013 | 0.9729 | 0.748 |
| 0.02 | 20 | 0.9027 | 0.9734 | 0.751 |
| 0.03 | 17 | 0.9047 | 0.9743 | 0.756 |
| 0.05 | 17 | 0.9022 | 0.9734 | 0.750 |
| 0.10 | 14 | 0.9026 | 0.9735 | 0.751 |
| 0.20 | 12 | 0.9041 | 0.9742 | 0.741 |

**Test macro-F1 λ'dan bağımsız olarak ~0.90'da sabit; öznitelik sayısı 12-20.** En küçük
ceza (λ=0.01) bile 15 öznitelikte kalıyor. Bu veri setinde optimum bant **15-20 öznitelik**;
20-25'e çıkmanın ölçülebilir faydası yok. (Bkz. `figures/06_lambda_sweep.png`.)

---

## 5. Seçilen özniteliklerin yorumu

**GA'nın seçtiği 15 öznitelik** üç aileden geliyor ve hepsi DDoS literatüründe anlamlı:

| Aile | Öznitelikler | DDoS anlamı |
|---|---|---|
| Paketler arası süre (IAT) | `fwd_packets_IAT_mean`, `packets_IAT_mean`, `packets_IAT_cov` | Bot trafiği makine hızında ve düzenli aralıklı; insan trafiği düzensiz |
| Paket boyutu deltaları | `mean/max_bwd_packets_delta_len`, `mean_packets_delta_len` | Saldırıda paket boyutları tekdüze (tek şablon); normalde değişken |
| Zamanlama deltaları | `min_bwd_packets_delta_time`, `min_fwd_packets_delta_time` | Tarama/flood desenlerinin zaman imzası |
| TCP durumu | `fwd_init_win_bytes`, `bwd_init_win_bytes`, `fwd_rst_flag_counts`, `fwd_ack_flag_percentage_in_total` | SYN/RST/ACK oranları saldırı türünü ele verir; anormal pencere boyutları manipüle paketleri gösterir |
| Başlık | `fwd_max_header_bytes` | Anormal başlık boyutları |
| Bağlantı ucu | `src_port`, `dst_port` | Güçlü sinyal ama testbed'e özgü olabilir (bkz. Sınırlılıklar) |

**GA ∩ PSO — 6 çekirdek öznitelik** (iki bağımsız arama da seçti):
`src_port`, `dst_port`, `fwd_packets_IAT_mean`, `packets_IAT_mean`, `fwd_init_win_bytes`,
`fwd_rst_flag_counts`. Bunlar DDoS ayrımının bel kemiği.

---

## 6. Sınırlılıklar ve gelecek çalışma

1. **Port sızıntısı riski:** `src_port`/`dst_port` her yöntemin seçtiği güçlü öznitelikler.
   Bunlar testbed kurulumuna aşırı uyum sağlıyor olabilir; gerçek trafikte port dağılımı
   farklıdır. İleride portsuz bir koşu yapılıp fark ölçülmeli.
2. **SVM ayarlanmadı.** SVM-RBF varsayılan `C=1, gamma=scale` ile zayıf kaldı; `GridSearchCV`
   ile ayarlanırsa RF'e yaklaşabilir. Ancak veri boyutu SVM için elverişsiz.
3. **PSO ayarlanmadı.** `c2` artırılıp `w` azaltılırsa PSO'nun yakınsaması iyileşebilir;
   GA ile daha adil bir eşleşme için PSO parametre taraması yapılabilir.
4. **Fitness alt örneklemi** (60k/30k) hız için; tam veriyle koşulsa skorlar birkaç binde
   oynayabilir (yön değişmez).
5. **`activity` (26 sınıf)** ile ikinci bir deney: zor görev, yöntemler arası fark
   belirginleşir. Kod hazır, sadece hedef sütun değişir.

---

## 7. Tekrarlanabilirlik

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Veri: data/raw/archive/ altına Kaggle'dan indir
python src/preprocessing.py         # 317 -> 183, train/val/test
python src/baseline_model.py        # referans (RF + SVM)
python src/filter_selection.py      # Aşama 1: 183 -> 65 havuz
python src/genetic_algorithm.py     # GA
python src/pso.py                   # PSO
python src/evaluate.py              # karşılaştırma tablosu + figürler
python src/lambda_sweep.py          # λ ödünleşim eğrisi
```

Tüm betikler `random_state=42`. Çıktılar: `results/metrics/*.json`, `results/figures/*.png`,
`results/comparison_table.md`.
