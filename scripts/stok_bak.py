"""Bir karar haftasinda depo stogunun ve imha muhasebesinin okunur ozeti.

    uv run python -m scripts.stok_bak --kosu fast
    uv run python -m scripts.stok_bak --kosu fast --hafta 51
    uv run python -m scripts.stok_bak --kosu full --hafta 91

NOT: Bu bir MILESTONE ARTIFACT'I DEGIL, REHBER.md'nin okuma araci. Karar
uretmez, hicbir sey yazmaz, model calistirmaz; yalnizca observable katmani
okur. `scripts/teklif_bak.py` "bu eczaneye ne teklif ediliyor" sorusunu
cevapliyor, bu script "o teklifin arkasindaki stok neye benziyor" sorusunu.

IKI STOK SAYISI AYNI SEY DEGIL
==============================
`depo_stok_haftalik` kaydi hafta SONUNDA, ikmalden sonra alinir ve
`giris_haftasi == t+1` olan partiyi de icerir. Politika o partiyi t
haftasinda tahsis EDEMEZ. Tahsis edilebilir stok bu yuzden lot
hareketlerinden yeniden kurulur (policy/candidates.py `_stok_gorunumu` ile
AYNI kural) ve kayitli stoktan hep kucuk ya da esittir:

    kalan = giris - (t'ye kadar sevk) - (t'ye kadar o lottan imha)

Ikisini yan yana basmak bu farkin gorunur kalmasi icin: kayitli stoga bakip
teklif planlamak olmayan mali soz vermektir.

IMHA MUHASEBESI
===============
`imhalar.kaynak` uc deger tasir ve ikisi depo defterinin DISINDADIR:
`eczane_iadesi:*` satirlarinin `lot_id`'si NULL'dur (sim/world.py
`_iade_isle`), cunku o mal zaten sevk edilmis, eczanede satilamamis, geri
donup imha edilmistir. Girisden bir kez daha dusulurse cift sayilir. Depo
tarafi kutle dengesi yalnizca `depo_miad` ile kapanir:

    giris - sevk - depo_miad_imhasi = kayitli stok        (kalinti sifir)

Bu ozdeslik tests/test_stok_bak.py'de sinaniyor.
"""

from __future__ import annotations

import argparse

import polars as pl

from core.config import Config, load_config
from core.io import Run
from features.okuma import GozlemlenebilirKaynak
from sim.calendar import GUN_HAFTA

# Imha kaynak etiketleri. sim/lots.py:54 sozlesmesi: depo tarafi imhasi lot
# referansi tasir, eczane iadesi tasimaz. Sabit cunku bunlar tuning knob'u
# degil, veri sozlugunun enum degerleri (DATA.md `imhalar`).
DEPO_IMHA_KAYNAGI = "depo_miad"


def tahsis_edilebilir_lotlar(
    kaynak: GozlemlenebilirKaynak, t: int
) -> pl.DataFrame:
    """t haftasinda tahsis edilebilir lotlar: lot basina kalan adet ve gun.

    policy/candidates.py `_stok_gorunumu` ile ayni kural. Lot referansi
    tasimayan imha satirlari (eczane iadesi) join'de dusuyor - depo
    defterinin disinda olduklari icin dusmeleri DOGRU davranis.
    """
    lotlar = kaynak.tablo("stok_lotlari")
    hareket = pl.concat([
        kaynak.tablo("sevkiyat_satirlari").select("lot_id", "hafta", "adet"),
        kaynak.tablo("imhalar").select("lot_id", "hafta", "adet"),
    ]).filter(pl.col("lot_id").is_not_null() & (pl.col("hafta") <= t))
    kullanilan = hareket.group_by("lot_id").agg(
        pl.col("adet").sum().alias("kullanilan"))

    return (
        lotlar.filter(pl.col("giris_haftasi") <= t)
        .join(kullanilan, on="lot_id", how="left")
        .with_columns(
            (pl.col("adet_giris") - pl.col("kullanilan").fill_null(0))
            .clip(lower_bound=0).alias("kalan_adet"),
            (pl.col("miad_gun_indeksi") - t * GUN_HAFTA).alias("kalan_gun"),
        )
        .filter((pl.col("kalan_adet") > 0) & (pl.col("kalan_gun") > 0))
    )


def raf_omru_bantlari(cfg: Config) -> list[tuple[float, float, str]]:
    """Bant sinirlari UYDURULMAZ, politikanin karar esiklerinden gelir.

    CLAUDE.md 2: kodda ciplak sayi olmaz. Her sinir o stoga NE OLABILECEGINI
    degistiren bir esiktir, bu yuzden bantlar da esiklerle birlikte kayar:

      temizlik tabani (tahsis.temizlik.asgari_kalan_raf_omru_gun)
          altinda hicbir rejimde teklif edilemez -> imhaya yazili
      normal veto (politika.kisit.asgari_kalan_raf_omru_gun)
          altinda YALNIZCA temizlik rejiminde teklif edilebilir
      miad baskisi (politika.aday.miad_baskisi_esik_gun)
          altinda aday skoru yukseltilir (SPEC 2.5 temizlik gudusu)

    Esikler esitse (varsayilanda veto == temizlik.tetik_gun == 120) bant
    sayisi kendiliginden azalir; tekrarli sinir uretmemek icin teklestirilir.
    """
    sinirlar = sorted({
        float(cfg.tahsis.temizlik.asgari_kalan_raf_omru_gun),
        float(cfg.politika.kisit.asgari_kalan_raf_omru_gun),
        float(cfg.politika.aday.miad_baskisi_esik_gun),
    })
    aciklama = {
        float(cfg.tahsis.temizlik.asgari_kalan_raf_omru_gun):
            "hicbir rejimde teklif edilemez",
        float(cfg.politika.kisit.asgari_kalan_raf_omru_gun):
            "yalnizca temizlik rejiminde",
        float(cfg.politika.aday.miad_baskisi_esik_gun):
            "teklif edilebilir, miad baskisi altinda",
    }
    bantlar: list[tuple[float, float, str]] = []
    alt = 0.0
    for ust in sinirlar:
        bantlar.append((alt, ust, aciklama[ust]))
        alt = ust
    bantlar.append((alt, float("inf"), "baski yok"))
    return bantlar


def ozet(kaynak: GozlemlenebilirKaynak, cfg: Config, t: int) -> dict:
    """t haftasindaki stok ve imha tablosu. Hicbir sey yazmaz."""
    depo = kaynak.tablo("depo_stok_haftalik").filter(pl.col("hafta") == t)
    if depo.height == 0:
        raise SystemExit(f"hafta {t} depo stok kaydinda yok")
    imha = kaynak.tablo("imhalar")
    iade = kaynak.tablo("iadeler")
    sevk = kaynak.tablo("sevkiyat_satirlari")
    lot = tahsis_edilebilir_lotlar(kaynak, t)

    tahsis_edilebilir = int(lot["kalan_adet"].sum())
    bantlar = []
    for alt, ust, etiket in raf_omru_bantlari(cfg):
        q = lot.filter((pl.col("kalan_gun") >= alt) & (pl.col("kalan_gun") < ust))
        bantlar.append({
            "alt": alt, "ust": ust, "etiket": etiket,
            "adet": int(q["kalan_adet"].sum()), "lot": q.height,
        })

    o_hafta = lambda df, sut: int(df.filter(pl.col("hafta") == t)[sut].sum())
    kumulatif = (imha.filter(pl.col("hafta") <= t)
                 .group_by("kaynak").agg(pl.col("adet").sum().alias("adet"))
                 .sort("adet", descending=True))
    giris = int(kaynak.tablo("stok_lotlari")
                .filter(pl.col("giris_haftasi") <= t)["adet_giris"].sum())
    depo_imhasi = int(imha.filter((pl.col("hafta") <= t)
                                  & (pl.col("kaynak") == DEPO_IMHA_KAYNAGI))
                      ["adet"].sum())
    return {
        "hafta": t,
        "kayitli_stok": int(depo["eldeki_adet"].sum()),
        "tahsis_edilebilir": tahsis_edilebilir,
        "lot_sayisi": lot.height,
        "sku_sayisi": lot["sku_id"].n_unique(),
        "bantlar": bantlar,
        "hafta_sevk": o_hafta(sevk, "adet"),
        "hafta_imha": o_hafta(imha, "adet"),
        "hafta_iade": o_hafta(iade, "iade_adet"),
        "hafta_iade_donen": o_hafta(iade, "depoya_donen_adet"),
        "kumulatif_imha": kumulatif,
        "kumulatif_giris": giris,
        "kumulatif_depo_imhasi": depo_imhasi,
    }


def bas(o: dict) -> None:
    t = o["hafta"]
    print(f"=== HAFTA {t} - DEPO ===")
    print(f"kayitli stok (hafta sonu) : {o['kayitli_stok']:>9,} adet")
    print(f"tahsis edilebilir         : {o['tahsis_edilebilir']:>9,} adet "
          f"| {o['lot_sayisi']} lot, {o['sku_sayisi']} SKU")
    fark = o["kayitli_stok"] - o["tahsis_edilebilir"]
    print(f"fark (gelecek haftanin partisi): {fark:>4,} adet")

    print(f"\n=== KALAN RAF OMRU BANTLARI (esikler config'ten) ===")
    toplam = max(o["tahsis_edilebilir"], 1)
    for b in o["bantlar"]:
        ust = "  +" if b["ust"] == float("inf") else f"{b['ust']:>3.0f}"
        print(f"  {b['alt']:>3.0f}-{ust} gun: {b['adet']:>8,} adet "
              f"({100 * b['adet'] / toplam:>5.1f}%) {b['lot']:>4} lot "
              f"| {b['etiket']}")

    print(f"\n=== HAFTA {t} AKISI ===")
    print(f"sevk : {o['hafta_sevk']:>7,} adet")
    print(f"imha : {o['hafta_imha']:>7,} adet")
    print(f"iade : {o['hafta_iade']:>7,} adet "
          f"(depoya donen {o['hafta_iade_donen']:,})")

    print(f"\n=== KUMULATIF IMHA (hafta 0-{t}) ===")
    print(o["kumulatif_imha"])
    giris, depo_imhasi = o["kumulatif_giris"], o["kumulatif_depo_imhasi"]
    print(f"giris {giris:,} -> depo imhasi {depo_imhasi:,} "
          f"= %{100 * depo_imhasi / max(giris, 1):.1f}")
    print(f"\nkutle dengesi: giris {giris:,} - sevk - depo_miad {depo_imhasi:,} "
          f"= kayitli stok (kalinti sifir, bkz. tests/test_stok_bak.py)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kosu", default="fast", help="data/<kosu> altindaki dunya")
    ap.add_argument("--hafta", type=int, default=None,
                    help="karar haftasi; bos = kosunun son haftasi")
    a = ap.parse_args()

    kaynak = GozlemlenebilirKaynak(a.kosu)
    cfg = load_config(profil=Run(a.kosu).read_manifest()["profil"])
    t = a.hafta if a.hafta is not None else cfg.profil.hafta_sayisi - 1
    bas(ozet(kaynak, cfg, t))


if __name__ == "__main__":
    main()
