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

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# Gozlemlenebilir katmanda gorunmesi YASAK kolon adlari.
# Yeni bir latent buyukluk turetildiginde buraya eklenir; testi bu liste besler.
LATENT_COLUMNS: frozenset[str] = frozenset(
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


class Run:
    """Tek bir simulasyon kosusunun cikti dizini."""

    def __init__(self, ad: str, kok: Path | None = None) -> None:
        self.ad = ad
        self.kok = (kok or DATA_DIR) / ad
        self.gozlemlenebilir = self.kok / "observable"
        self.gercek = self.kok / "ground_truth"

    def prepare(self, clean: bool = True) -> "Run":
        if clean and self.kok.exists():
            shutil.rmtree(self.kok)
        self.gozlemlenebilir.mkdir(parents=True, exist_ok=True)
        self.gercek.mkdir(parents=True, exist_ok=True)
        return self

    def write_observable(self, ad: str, df: pl.DataFrame) -> Path:
        violations = sorted(set(df.columns) & LATENT_COLUMNS)
        if violations:
            raise ValueError(
                f"Gozlemlenebilirlik ihlali: '{ad}' tablosu latent kolon tasiyor: {violations}. "
                f"Bu kolonlar ground_truth/ altina yazilir."
            )
        path = self.gozlemlenebilir / f"{ad}.parquet"
        df.write_parquet(path)
        return path

    def write_ground_truth(self, ad: str, df: pl.DataFrame) -> Path:
        path = self.gercek / f"{ad}.parquet"
        df.write_parquet(path)
        return path

    def read_observable(self, ad: str) -> pl.DataFrame:
        return pl.read_parquet(self.gozlemlenebilir / f"{ad}.parquet")

    def read_ground_truth(self, ad: str) -> pl.DataFrame:
        return pl.read_parquet(self.gercek / f"{ad}.parquet")

    def write_manifest(self, content: dict) -> Path:
        content = dict(content)
        content["yazilma_zamani_utc"] = datetime.now(timezone.utc).isoformat()
        path = self.kok / "manifest.json"
        path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def read_manifest(self) -> dict:
        return json.loads((self.kok / "manifest.json").read_text(encoding="utf-8"))

    def tables(self) -> dict[str, list[str]]:
        return {
            "observable": sorted(p.stem for p in self.gozlemlenebilir.glob("*.parquet")),
            "ground_truth": sorted(p.stem for p in self.gercek.glob("*.parquet")),
        }
