"""Stok okuma aracinin degismezleri: muhasebe, sinir, config kuplaji.

Metrik degeri burada sinanmaz (dunyaya ve seed'e bagli); okumanin DOGRU
okuma oldugu sinanir. Uc iddia var ve ucu de bir kez yanlis yapilmaya
musait:

  1. Depo kutle dengesi yalnizca `depo_miad` ile kapanir. Eczane iadesi
     kaynakli imhalar depo defterinin disindadir (lot_id NULL) ve girisden
     dusulurse CIFT SAYILIR.
  2. Tahsis edilebilir stok kayitli stoktan buyuk olamaz - buyurse politika
     olmayan mali soz verir.
  3. Bant sinirlari config esikleridir; esik degisince bant kayar. Sabit
     kalirsa CLAUDE.md 2 ihlali (sihirli sayi) demektir.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from core.config import load_config
from core.io import Run
from features.okuma import GozlemlenebilirKaynak
from scripts.generate_world import dunya_yaz
from scripts.stok_bak import (DEPO_IMHA_KAYNAGI, ozet, raf_omru_bantlari,
                              tahsis_edilebilir_lotlar)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def kucuk_dunya(tmp_path_factory):
    cfg = load_config("fast")
    kok = tmp_path_factory.mktemp("dunya")
    dunya_yaz(cfg, Run("t", kok=kok))
    return cfg, kok


@pytest.fixture(scope="module")
def kaynak(kucuk_dunya):
    _, kok = kucuk_dunya
    return GozlemlenebilirKaynak("t", kok=kok)


def test_kutle_dengesi_yalnizca_depo_imhasiyla_kapanir(kucuk_dunya, kaynak):
    """giris - sevk - depo_miad = kayitli stok, KALINTI SIFIR.

    Eczane iadesi kaynakli imhalari da dusmek bu ozdesligi bozar; testin
    ikinci yarisi tam olarak o hatanin yakalandigini gosterir.
    """
    cfg, _ = kucuk_dunya
    t = cfg.profil.hafta_sayisi - 1
    imha = kaynak.tablo("imhalar")

    giris = kaynak.tablo("stok_lotlari")["adet_giris"].sum()
    sevk = kaynak.tablo("sevkiyat_satirlari")["adet"].sum()
    depo_imhasi = imha.filter(pl.col("kaynak") == DEPO_IMHA_KAYNAGI)["adet"].sum()
    kayitli = (kaynak.tablo("depo_stok_haftalik")
               .filter(pl.col("hafta") == t)["eldeki_adet"].sum())

    assert giris - sevk - depo_imhasi == kayitli

    iade_imhasi = imha.filter(pl.col("kaynak") != DEPO_IMHA_KAYNAGI)["adet"].sum()
    assert iade_imhasi > 0, "dunyada eczane iadesi yok - test bos donuyor"
    assert giris - sevk - depo_imhasi - iade_imhasi != kayitli


def test_iade_imhalari_lot_referansi_tasimaz(kaynak):
    """Depo defterinin disinda olduklarinin YAPISAL isareti (sim/world.py
    `_iade_isle` lot_id=None yazar). Bir gun lot_id yazilirsa kutle dengesi
    sessizce bozulur; bu test once patlar."""
    imha = kaynak.tablo("imhalar")
    iade = imha.filter(pl.col("kaynak") != DEPO_IMHA_KAYNAGI)
    depo = imha.filter(pl.col("kaynak") == DEPO_IMHA_KAYNAGI)
    assert iade.height > 0 and depo.height > 0
    assert iade["lot_id"].null_count() == iade.height
    assert depo["lot_id"].null_count() == 0


def test_tahsis_edilebilir_kayitli_stogu_asmaz(kucuk_dunya, kaynak):
    """Politika olmayan mali soz veremez: her haftada <= kayitli stok."""
    cfg, _ = kucuk_dunya
    kayitli = kaynak.tablo("depo_stok_haftalik")
    for t in (0, 1, 50, 51, cfg.profil.hafta_sayisi - 1):
        o = ozet(kaynak, cfg, t)
        beklenen = kayitli.filter(pl.col("hafta") == t)["eldeki_adet"].sum()
        assert o["kayitli_stok"] == beklenen
        assert o["tahsis_edilebilir"] <= o["kayitli_stok"], f"hafta {t}"


def test_gelecege_bakmiyor(kucuk_dunya, kaynak):
    """Point-in-time: t'den sonraki hareketler t'nin sonucunu degistirmemeli.

    Gelecegi kesip yeniden hesapla, karsilastir (features/panel.py'deki
    leakage guard'in ayni mantigi).
    """
    cfg, kok = kucuk_dunya
    t = 51
    tam = tahsis_edilebilir_lotlar(kaynak, t)

    class KesikKaynak:
        def tablo(self, ad: str) -> pl.DataFrame:
            df = kaynak.tablo(ad)
            if "hafta" in df.columns:
                df = df.filter(pl.col("hafta") <= t)
            if "giris_haftasi" in df.columns:
                df = df.filter(pl.col("giris_haftasi") <= t)
            return df

    kesik = tahsis_edilebilir_lotlar(KesikKaynak(), t)
    assert tam.height > 0
    assert (tam.sort("lot_id")["kalan_adet"].to_list()
            == kesik.sort("lot_id")["kalan_adet"].to_list())
    assert (tam.sort("lot_id")["lot_id"].to_list()
            == kesik.sort("lot_id")["lot_id"].to_list())


def test_bantlar_config_esiklerinden_turetiliyor(kucuk_dunya):
    """Sinirlar sabit degil: esigi oynatinca bant da oynamali."""
    cfg, _ = kucuk_dunya
    varsayilan = [b[1] for b in raf_omru_bantlari(cfg)]
    assert cfg.politika.aday.miad_baskisi_esik_gun in varsayilan
    assert cfg.politika.kisit.asgari_kalan_raf_omru_gun in varsayilan

    ezilmis = load_config(
        "fast", gecersiz_kilma={"politika.aday.miad_baskisi_esik_gun": 240})
    yeni = [b[1] for b in raf_omru_bantlari(ezilmis)]
    assert 240 in yeni and 180 not in yeni


def test_bantlar_stogu_tam_bolusturuyor(kucuk_dunya, kaynak):
    """Bantlar ortusmez ve bosluk birakmaz: toplam == tahsis edilebilir."""
    cfg, _ = kucuk_dunya
    o = ozet(kaynak, cfg, 51)
    assert sum(b["adet"] for b in o["bantlar"]) == o["tahsis_edilebilir"]
    assert sum(b["lot"] for b in o["bantlar"]) == o["lot_sayisi"]


def test_okuma_araci_ground_truth_okumuyor():
    """scripts/stok_bak.py observable sinirinin disina cikmamali."""
    metin = (REPO_ROOT / "scripts" / "stok_bak.py").read_text()
    assert "ground_truth" not in metin
    assert "GozlemlenebilirKaynak" in metin
