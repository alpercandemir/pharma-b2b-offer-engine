"""Tek eczane + tek karar haftasi icin teslim edilen teklif satirlarini yazar.

    uv run python -m scripts.teklif_bak --kosu fast
    uv run python -m scripts.teklif_bak --kosu fast --eczane ECZ0003 --politika propensity
    uv run python -m scripts.teklif_bak --kosu fast --origin-sayisi 6 --hafta 83

NOT: Bu bir MILESTONE ARTIFACT'I DEGIL, REHBER.md'nin okuma araci. Karar
uretmez, hicbir sey yazmaz; `experiments/run.py::m4_boru_hatti`in zaten
urettigi secimi bir eczane icin okunur hale getirir. Ayni sayilar
`harness.run --metin` brifinginde de var; bu script brifingin LLM/senaryo
katmani olmadan ham tablo halidir.
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from core.config import config_yukle
from core.io import VERI_DIZINI, Kosu
from experiments.run import m4_boru_hatti


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kosu", default="fast", help="data/<kosu> altindaki dunya")
    ap.add_argument("--eczane", default=None, help="ECZ0000; bos = ilk eczane")
    ap.add_argument("--hafta", type=int, default=None,
                    help="karar haftasi (origin); bos = son olcum origin'i")
    ap.add_argument("--politika", default="uplift_x",
                    help="uplift_x | uplift_t | propensity | m3_sabit_kampanya | ...")
    ap.add_argument("--origin-sayisi", type=int, default=None,
                    help="kac karar haftasi geriye gidilsin (varsayilan config'ten)")
    a = ap.parse_args()

    kosu = Kosu(a.kosu)
    ez = ({} if a.origin_sayisi is None
          else {"politika.aday.degerlendirme.origin_sayisi": a.origin_sayisi})
    cfg = config_yukle(profil=kosu.manifest_oku()["profil"], gecersiz_kilma=ez)
    m4 = m4_boru_hatti(cfg, a.kosu, VERI_DIZINI)

    t = a.hafta if a.hafta is not None else m4.olcum_originleri[-1]
    if t not in m4.olcum_originleri:
        raise SystemExit(f"hafta {t} olcum origin'lerinde yok: {m4.olcum_originleri} "
                         f"(--origin-sayisi ile geriye acabilirsin)")
    if a.politika not in m4.secimler:
        raise SystemExit(f"politika yok: {a.politika}. Secenekler: "
                         f"{sorted(m4.secimler)}")
    i = m4.olcum_originleri.index(t)
    blok, secim = m4.bloklar[i], m4.secimler[a.politika][i]

    ecz_id = m4.dunya.eczaneler["eczane_id"].to_numpy()
    sku_id = m4.dunya.urunler["sku_id"].to_numpy()
    hedef = a.eczane or ecz_id[0]
    bulunan = np.flatnonzero(ecz_id == hedef)
    if bulunan.size == 0:
        raise SystemExit(f"eczane yok: {hedef}")
    p_idx = int(bulunan[0])

    n = blok.teklifler.height
    satir = np.arange(n)
    df = blok.teklifler.with_columns(
        pl.Series("kol", [blok.mat.uzay.adlar[k] for k in secim.kol]),
        pl.Series("adet", blok.mat.adet[satir, secim.kol]),
        pl.Series("bedava", blok.mat.bedava[satir, secim.kol]),
        pl.Series("kabul_sartiyla_marj", blok.mat.marj[satir, secim.kol]),
        pl.Series("teklif_var", secim.teklif_maskesi),
    ).filter((pl.col("eczane_idx") == p_idx) & pl.col("teklif_var"))

    print(f"kosu={a.kosu} eczane={hedef} hafta={t} politika={a.politika} "
          f"| olcum origin'leri={m4.olcum_originleri}")
    print(f"aday satiri (veto sonrasi, bu eczane)="
          f"{blok.teklifler.filter(pl.col('eczane_idx') == p_idx).height} "
          f"-> teklife donen={df.height}")
    print(df.with_columns(pl.Series("sku", sku_id[df["sku_idx"].to_numpy()]))
            .select(["sku", "lot_id", "kol", "adet", "bedava",
                     "kabul_sartiyla_marj", "skor"]))


if __name__ == "__main__":
    main()
