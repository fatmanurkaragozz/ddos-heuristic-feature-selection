"""BCCC-cPacket-Cloud-DDoS-2024 veri setini Kaggle'dan indirir.

Ön koşul: Kaggle API token'i (~/.kaggle/kaggle.json veya %USERPROFILE%\\.kaggle\\kaggle.json).
    Kaggle > Account > "Create New API Token" ile kaggle.json indirilir.

Kullanım:
    python src/download_data.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DATASET = "dhoogla/bccc-cpacket-cloud-ddos-2024"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"'{DATASET}' -> {RAW_DIR}")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR), "--unzip"],
            check=True,
        )
    except FileNotFoundError:
        sys.exit("kaggle CLI bulunamadi. 'pip install -r requirements.txt' calistirin.")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"Indirme basarisiz (kod {exc.returncode}). kaggle.json'i kontrol edin.")

    files = sorted(p.name for p in RAW_DIR.iterdir())
    print(f"\n{len(files)} dosya indirildi:")
    for f in files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
