# DDoS Tespitinde Sezgisel/Metasezgisel Öznitelik Seçimi

Fırat Üniversitesi Yazılım Mühendisliği – staj/mentörlük projesi.

**Amaç:** Bulut tabanlı DDoS tespiti için Genetik Algoritma (GA) ve Particle Swarm
Optimization (PSO) kullanarak öznitelik seçimi yapmak ve seçilen alt kümelerin
model performansına etkisini incelemek.

## Sonuç özeti

| Yöntem | Öznitelik | Test macro-F1 | Test accuracy | Suspicious F1 |
|---|---:|---:|---:|---:|
| Baseline (tümü) | 183 | 0.8693 | 0.9587 | 0.676 |
| Top-15 (RF önem) | 15 | 0.8970 | 0.9712 | 0.738 |
| RFE-15 | 15 | 0.8932 | 0.9697 | 0.729 |
| **GA** | **15** | **0.9065** | **0.9747** | **0.761** |
| PSO | 27 | 0.9004 | 0.9725 | 0.746 |

- **Öznitelik seçimi yapan her yöntem baseline'ı geçti** — 183 → 15 öznitelikte macro-F1
  0.869 → 0.907. Fazlalık modele zarar veriyordu.
- **GA en iyi:** aynı 15 öznitelik bütçesinde Top-15 ve RFE-15'i de geçti; PSO'dan hem daha
  az öznitelik hem daha yüksek skor.
- **λ taraması:** bu veride optimum bant 15-20 öznitelik; 20-25'in ek faydası yok.

Ayrıntılı rapor: [results/REPORT.md](results/REPORT.md)

## Veri Seti

**BCCC-cPacket-Cloud-DDoS-2024**
- Kaynak: https://www.kaggle.com/datasets/dhoogla/bccc-cpacket-cloud-ddos-2024
- York Üniversitesi (BCCC) + cPacket; NTLFlowLyzer ile çıkarılmış 317 ağ/akış özniteliği
- 540.494 akış; hedef `label`: Benign / Attack / Suspicious (dengesizlik ~16.7x)
- Referans: Shafi, Lashkari, Rodriguez, Nevo (2024). *Toward Generating a New Cloud-Based
  Distributed Denial of Service (DDoS) Dataset...* Information 15(4).

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Veriyi `data/raw/archive/` altına indirin (Kaggle → Download → zip'i aç).

## İş Akışı

```bash
python src/preprocessing.py       # 317 -> 183 öznitelik; stratified 70/15/15
python src/baseline_model.py      # tüm özniteliklerle RF + SVM (referans)
python src/filter_selection.py    # Aşama 1 filtre: 183 -> 65 aday havuz
python src/genetic_algorithm.py   # GA ile öznitelik seçimi (DEAP)
python src/pso.py                 # PSO ile öznitelik seçimi (pyswarms)
python src/evaluate.py            # Baseline vs Top-K vs RFE vs GA vs PSO
python src/lambda_sweep.py        # λ (sadelik/performans) ödünleşim eğrisi
```

Keşifsel analiz: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)

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

## Klasör Yapısı

```
data/raw/         # ham veri (git'e girmez)
data/processed/   # işlenmiş veri + scaler/encoder (git'e girmez)
notebooks/        # EDA
src/              # pipeline betikleri
results/
  metrics/        # *.json — tüm sayısal sonuçlar
  figures/        # *.png — grafikler
  REPORT.md       # mentör raporu
  comparison_table.md
```

## Metodolojik notlar

- **Test seti** final değerlendirmeye kadar hiç kullanılmadı. GA/PSO fitness'ı yalnızca
  validation setinden hesaplandı.
- Aşama 0 temizliği ve Aşama 1 filtresi **yalnızca train** istatistiğinden öğrenildi.
- Tüm betikler `random_state=42`.
- Dengesiz veri → ana metrik **macro-F1** (accuracy tek başına yanıltıcı).
