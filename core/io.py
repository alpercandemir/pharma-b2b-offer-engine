"""Kosu dizini, parquet yazimi ve gozlemlenebilirlik siniri.

SPEC 3: "sim/ Sentetik dunya (ground truth burada, model asla goremez)".
Bu sinir yorum satiriyla degil, yazma anininda ZORLANARAK korunur:

    data/<kosu>/observable/    -> modelin gorebilecegi her sey
    data/<kosu>/ground_truth/  -> latent gercek; sadece olcum/oracle icin

`yaz_gozlemlenebilir()` bilinen bir latent kolon adi gorurse yazmaz, hata
firlatir. Kaza eseri sizinti bu yuzden kosuyu dusurur, sessizce gecmez.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

KOK = Path(__file__).resolve().parent.parent
VERI_DIZINI = KOK / "data"

# Gozlemlenebilir katmanda gorunmesi YASAK kolon adlari.
# Yeni bir latent buyukluk turetildiginde buraya eklenir; testi bu liste besler.
LATENT_KOLONLAR: frozenset[str] = frozenset(
    {
        "share_of_wallet",
        "sow",
        "stokculuk_egilimi",
        "miad_toleransi_gun",
        "latent_tuketim_hizi",
        "gercek_tuketim",
        "gercek_eczane_stogu",
        "rakip_siparis_adedi",
        "kapsama_hafta",
        "cesitte_var",
        "gercek_tukenme_haftasi",
        "antisipasyon_siddeti",
        "olay_antisipasyon_aktif",
    }
)


class Kosu:
    """Tek bir simulasyon kosusunun cikti dizini."""

    def __init__(self, ad: str, kok: Path | None = None) -> None:
        self.ad = ad
        self.kok = (kok or VERI_DIZINI) / ad
        self.gozlemlenebilir = self.kok / "observable"
        self.gercek = self.kok / "ground_truth"

    def hazirla(self, temizle: bool = True) -> "Kosu":
        if temizle and self.kok.exists():
            shutil.rmtree(self.kok)
        self.gozlemlenebilir.mkdir(parents=True, exist_ok=True)
        self.gercek.mkdir(parents=True, exist_ok=True)
        return self

    def yaz_gozlemlenebilir(self, ad: str, df: pl.DataFrame) -> Path:
        ihlal = sorted(set(df.columns) & LATENT_KOLONLAR)
        if ihlal:
            raise ValueError(
                f"Gozlemlenebilirlik ihlali: '{ad}' tablosu latent kolon tasiyor: {ihlal}. "
                f"Bu kolonlar ground_truth/ altina yazilir."
            )
        yol = self.gozlemlenebilir / f"{ad}.parquet"
        df.write_parquet(yol)
        return yol

    def yaz_gercek(self, ad: str, df: pl.DataFrame) -> Path:
        yol = self.gercek / f"{ad}.parquet"
        df.write_parquet(yol)
        return yol

    def oku_gozlemlenebilir(self, ad: str) -> pl.DataFrame:
        return pl.read_parquet(self.gozlemlenebilir / f"{ad}.parquet")

    def oku_gercek(self, ad: str) -> pl.DataFrame:
        return pl.read_parquet(self.gercek / f"{ad}.parquet")

    def manifest_yaz(self, icerik: dict) -> Path:
        icerik = dict(icerik)
        icerik["yazilma_zamani_utc"] = datetime.now(timezone.utc).isoformat()
        yol = self.kok / "manifest.json"
        yol.write_text(json.dumps(icerik, indent=2, ensure_ascii=False), encoding="utf-8")
        return yol

    def manifest_oku(self) -> dict:
        return json.loads((self.kok / "manifest.json").read_text(encoding="utf-8"))

    def tablolar(self) -> dict[str, list[str]]:
        return {
            "observable": sorted(p.stem for p in self.gozlemlenebilir.glob("*.parquet")),
            "ground_truth": sorted(p.stem for p in self.gercek.glob("*.parquet")),
        }
