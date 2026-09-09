# DDoS Tespitinde Sezgisel/Metasezgisel Öznitelik Seçimi

Fırat Üniversitesi Yazılım Mühendisliği – staj/mentörlük projesi.

**Amaç:** Bulut tabanlı DDoS tespiti için Genetik Algoritma (GA) ve Particle Swarm
Optimization (PSO) kullanarak öznitelik seçimi yapmak ve seçilen alt kümelerin
model performansına etkisini incelemek.

## Veri Seti

**BCCC-cPacket-Cloud-DDoS-2024**
- Kaynak: https://www.kaggle.com/datasets/dhoogla/bccc-cpacket-cloud-ddos-2024
- York Üniversitesi (BCCC) + cPacket işbirliği
- NTLFlowLyzer ile çıkarılmış 300+ ağ/taşıma katmanı özniteliği
- 8+ normal aktivite + 17 DDoS saldırı senaryosu, 26 etiket
- Referans: Shafi, Lashkari, Rodriguez, Nevo (2024). *Toward Generating a New
  Cloud-Based Distributed Denial of Service (DDoS) Dataset...* Information 15(4).

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## İş Akışı

| Adım | Betik | Açıklama |
|------|-------|----------|
| 1 | `notebooks/01_eda.ipynb` | Keşifsel veri analizi |
| 2 | `src/preprocessing.py` | Temizlik, encode, normalize |
| 3 | `src/baseline_model.py` | Tüm özniteliklerle baseline (5-fold CV) |
| 4 | `src/genetic_algorithm.py` | GA ile öznitelik seçimi |
| 5 | `src/pso.py` | PSO ile öznitelik seçimi |
| 6 | `src/evaluate.py` | Baseline vs GA vs PSO karşılaştırması |

## Klasör Yapısı

```
data/raw/         # ham veri (git'e girmez)
data/processed/   # işlenmiş veri (git'e girmez)
notebooks/        # EDA
src/              # pipeline betikleri
results/          # metrikler, figürler, karşılaştırma tabloları
```

## Sonuçlar

_(Analiz tamamlandıkça doldurulacak)_
