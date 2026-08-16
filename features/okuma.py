"""Gozlemlenebilirlik siniri - okuma tarafi.

core/io.py sinir YAZARKEN korunuyor (latent kolon adi gorurse hata firlatir).
Burada ayni sinir OKURKEN korunuyor:

    GozlemlenebilirKaynak yalnizca data/<kosu>/observable/ altini gorur.
    ground_truth/ dizinine giden bir yol YOK - yorumla degil, yapiyla.

Kural: `features/` altindaki hicbir modul ground_truth okumaz. Bunu
tests/test_features.py::test_feature_katmani_ground_truth_okumuyor hem statik
(kaynak taramasi) hem calisma zamani (dizini gizleyip panel kurma) olarak
sinar.

Oracle etiketleri yalnizca eval/oracle.py'de okunur ve YALNIZCA olcumde
kullanilir; egitim yoluna girmez.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


class GozlemlenebilirKaynak:
    """Tek bir kosunun observable katmani. Baska hicbir yeri okumaz."""

    def __init__(self, kosu_adi: str, kok: Path | None = None) -> None:
        self.kosu_adi = kosu_adi
        self.dizin = (kok or DATA_DIR) / kosu_adi / "observable"
        if not self.dizin.is_dir():
            raise FileNotFoundError(f"gozlemlenebilir katman yok: {self.dizin}")

    def tablo(self, ad: str) -> pl.DataFrame:
        if "/" in ad or ".." in ad:
            raise ValueError(f"tablo adi dizin gezinemez: {ad!r}")
        yol = self.dizin / f"{ad}.parquet"
        if not yol.is_file():
            raise FileNotFoundError(
                f"'{ad}' gozlemlenebilir katmanda yok. Mevcut: {self.tables()}"
            )
        return pl.read_parquet(yol)

    def tables(self) -> list[str]:
        return sorted(p.stem for p in self.dizin.glob("*.parquet"))
