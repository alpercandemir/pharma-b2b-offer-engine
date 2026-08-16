"""M1 giris noktasi: sentetik dunyayi uretir ve data/<kosu>/ altina yazar.

    python -m scripts.generate_world --profil full
    python -m scripts.generate_world --profil fast --kosu deneme --knob talep.dagilim.sifir_sisirme=0.5
"""

from __future__ import annotations

import argparse
import time

import polars as pl

from core.config import load_config
from core.io import Run
from core.rng import SeedBank
from sim.world import dunya_kos


def _knob_ayristir(ham: list[str]) -> dict[str, object]:
    cikti: dict[str, object] = {}
    for parca in ham:
        if "=" not in parca:
            raise SystemExit(f"--knob bicimi 'yol=deger' olmali: {parca}")
        yol, deger = parca.split("=", 1)
        try:
            cikti[yol] = int(deger)
        except ValueError:
            try:
                cikti[yol] = float(deger)
            except ValueError:
                cikti[yol] = {"true": True, "false": False}.get(deger.lower(), deger)
    return cikti


def dunya_yaz(cfg, kosu: Run, gecersiz: dict | None = None) -> dict:
    """Dunyayi uretir, kosu dizinine yazar, manifest'i dondurur.

    M2'den itibaren experiments/run.py de bu fonksiyonu cagirir; yazma yolu
    tek yerde kalsin diye ayrildi (gozlemlenebilirlik siniri core/io.py'de
    yazma aninda zorlanir).
    """
    seedler = SeedBank(cfg.profil.temel_seed)
    t0 = time.perf_counter()
    d = dunya_kos(cfg, seedler)
    sure = time.perf_counter() - t0

    kosu.prepare()
    gozlemlenebilir = {
        "urunler": d.urunler,
        "eczaneler": d.eczaneler,
        "takvim": d.takvim,
        "olaylar": d.olaylar_gozlemlenebilir,
        "siparisler": d.siparisler,
        "sevkiyat_satirlari": d.sevkiyat_satirlari,
        "stok_lotlari": d.stok_lotlari,
        "depo_stok_haftalik": d.depo_stok_haftalik,
        "imhalar": d.imhalar,
        "iadeler": d.iadeler,
        "urun_fiyat_haftalik": d.urun_fiyat_haftalik,
        "makro_haftalik": d.makro_haftalik,
    }
    gercek = {
        "latent_urun": d.latent_urun,
        "latent_eczane": d.latent_eczane,
        "hucre_haftalik": d.hucre_haftalik,
        "rakip_siparisleri": d.rakip_siparisleri,
        "sow_haftalik": d.sow_haftalik,
        "olaylar_gercek": d.olaylar_gercek,
        "tukenme_olaylari": d.tukenme_olaylari,
    }
    for ad, df in gozlemlenebilir.items():
        kosu.write_observable(ad, df)
    for ad, df in gercek.items():
        kosu.write_ground_truth(ad, df)

    manifest = {
        "profil": cfg.profil.ad,
        "config_hash": cfg.hash(),
        "dunya_hash": cfg.dunya_hash(),
        "temel_seed": cfg.profil.temel_seed,
        "knob_gecersiz_kilma": {k: str(v) for k, v in (gecersiz or {}).items()},
        "eczane_sayisi": cfg.profil.eczane_sayisi,
        "sku_sayisi": cfg.profil.sku_sayisi,
        "hafta_sayisi": cfg.profil.hafta_sayisi,
        "kosu_suresi_sn": round(sure, 2),
        "seed_tohumlari": seedler.issued,
        "satir_sayilari": {ad: df.height for ad, df in {**gozlemlenebilir, **gercek}.items()},
    }
    kosu.write_manifest(manifest)
    manifest["_tablolar"] = (list(gozlemlenebilir), list(gercek),
                             [df.height for df in gozlemlenebilir.values()],
                             [df.height for df in gercek.values()])
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profil", default="full")
    ap.add_argument("--kosu", default=None, help="cikti dizini adi (varsayilan: profil adi)")
    ap.add_argument("--seed", type=int, default=None, help="profildeki temel_seed'i ezer")
    ap.add_argument("--knob", action="append", default=[], help="yol=deger, tekrarlanabilir")
    args = ap.parse_args()

    gecersiz = _knob_ayristir(args.knob)
    if args.seed is not None:
        gecersiz["profil.temel_seed"] = args.seed
    cfg = load_config(args.profil, gecersiz_kilma=gecersiz)

    kosu = Run(args.kosu or cfg.profil.ad)
    manifest = dunya_yaz(cfg, kosu, gecersiz)
    gozlemlenebilir, gercek, g_satir, t_satir = manifest["_tablolar"]
    sure = manifest["kosu_suresi_sn"]

    print(f"kosu: {kosu.kok}")
    print(f"sure: {sure:.1f} sn | config_hash: {cfg.hash()}")
    with pl.Config(tbl_rows=40):
        print(
            pl.DataFrame(
                {
                    "tablo": gozlemlenebilir + gercek,
                    "katman": ["observable"] * len(gozlemlenebilir) + ["ground_truth"] * len(gercek),
                    "satir": g_satir + t_satir,
                }
            )
        )


if __name__ == "__main__":
    main()
