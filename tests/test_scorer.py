"""M4 aksiyon uzayi ve marj aritmetigi testleri (D1).

Bu dosya `policy/scorer.py`nin degismezlerini sinar: aksiyon uzayinin sekli,
koli yuvarlamasi, MF kanal kisiti ve aksiyon seciminin frekans tavani.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.config import load_config
from policy import scorer

PROFIL = "fast"


@pytest.fixture(scope="module")
def cfg():
    return load_config(PROFIL)


def _aday(n=6, adet=None, koli=None, mf_izinli=None, hiz=None):
    """Sentetik aday tablosu: kisit katmanindan cikmis gibi."""
    adet = np.array([1.0, 5.0, 10.0, 20.0, 40.0, 100.0]) if adet is None else adet
    return pl.DataFrame({
        "eczane_idx": np.zeros(n, dtype=np.int32),
        "sku_idx": np.arange(n, dtype=np.int32),
        "teklif_adedi": adet,
        "hiz_tahmini": np.full(n, 100.0) if hiz is None else hiz,
        "mf_izinli": np.ones(n, dtype=bool) if mf_izinli is None else mf_izinli,
        "skor": np.linspace(1.0, 0.5, n),
    })


class _SahteDunya:
    """AdayDunyasi'nin `teklif_matrisleri` icin gereken minimal yuzu."""

    def __init__(self, n, koli, dsf=100.0, marj=0.05):
        self.urunler = pl.DataFrame({
            "depo_kar_marji": np.full(n, marj),
            "koli_ici_adet": np.asarray(koli, dtype=np.int64),
        })
        self.eczaneler = pl.DataFrame({"vade_riski_skoru": np.zeros(1)})
        self.dsf = np.full(n, dsf)


def test_aksiyon_uzayi_yuzde_iskonto_icermiyor(cfg):
    """D1: aksiyon (MF, vade) ciftidir. Kol sayisi = 1 + |MF| x |vade|."""
    uzay = scorer.aksiyon_uzayi(cfg)
    a = cfg.politika.aksiyon
    assert uzay.A == 1 + len(a.mf_oranlari) * len(a.vade_gunleri)
    assert uzay.mf[scorer.TEKLIF_YOK] == 0.0
    assert uzay.vade[scorer.TEKLIF_YOK] == a.taban_vade_gun
    # Kollarin tamami config'teki iki listenin capraz carpimi; baska bir
    # aksiyon tipi (yuzde iskonto vb.) uretilmiyor.
    beklenen = {(m, float(v)) for m in a.mf_oranlari for v in a.vade_gunleri}
    assert {(float(m), float(v)) for m, v in zip(uzay.mf[1:], uzay.vade[1:])} == beklenen


def test_taban_vade_aksiyon_uzayinda_olmak_zorunda():
    """Karsi-olgusal marj taban vadeyle hesaplaniyor; kol yoksa kiyas anlamsiz.

    Config yuklemesinde reddedilir (ValidationError), kod incelemesiyle degil.
    """
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="taban_vade_gun"):
        load_config(PROFIL,
                     gecersiz_kilma={"politika.aksiyon.taban_vade_gun": 45})


def test_koli_yuvarlamasi_bedava_adedi_tam_sayi(cfg):
    """SPEC 2.1 + m3.md borcu: bedava adet tam sayi, sifirsa kanal kapali."""
    adet = np.array([3.0, 7.0, 10.0, 25.0])
    koli = np.array([10, 10, 10, 10])
    tavan = np.full(4, 1e9)
    efektif, bedava, yuvarlandi = scorer.koli_yuvarlamasi(adet, koli, 0.10, cfg, tavan)
    assert np.all(bedava == np.floor(bedava))
    assert np.all(efektif % 10 == 0)          # koli katina yuvarlandi
    assert np.all(yuvarlandi)
    # MF'siz kolda adet degismez, bedava yok.
    e0, b0, y0 = scorer.koli_yuvarlamasi(adet, koli, 0.0, cfg, tavan)
    assert np.array_equal(e0, adet) and not b0.any() and not y0.any()


def test_yuvarlama_emilim_tavanini_asmiyor(cfg):
    """Yuvarlama adedi buyutur; tavani asiyorsa ATLANIR, kanal kapanmaz."""
    adet = np.array([4.0])
    koli = np.array([30])
    tavan = np.array([10.0])
    efektif, bedava, yuvarlandi = scorer.koli_yuvarlamasi(adet, koli, 0.10, cfg, tavan)
    assert efektif[0] == 4.0 and not yuvarlandi[0]
    assert bedava[0] == 0.0                    # floor(4 * 0.10) = 0


def test_mf_kanali_sgk_urununde_kapali(cfg):
    """SPEC 2.5: SGK kapsaminda MF kanali kapali, vade kollari acik."""
    n = 6
    dunya = _SahteDunya(n, koli=[1] * n)
    teklifler = _aday(n, mf_izinli=np.array([False] * n))
    mat = scorer.teklif_matrisleri(dunya, cfg, teklifler)
    mf_kollari = np.flatnonzero(mat.uzay.mf > 0)
    vade_kollari = np.flatnonzero((mat.uzay.mf == 0) & (np.arange(mat.uzay.A) > 0))
    assert not mat.izinli[:, mf_kollari].any()
    assert mat.izinli[:, vade_kollari].all()


def test_mf_bedava_sifir_kolu_kapatiyor(cfg):
    """'7 adetlik satirda 10+1' anlamsiz: kol kapali (m3.md borcu)."""
    n = 6
    dunya = _SahteDunya(n, koli=[1] * n)
    teklifler = _aday(n, adet=np.array([1.0, 5.0, 9.0, 10.0, 40.0, 100.0]))
    mat = scorer.teklif_matrisleri(dunya, cfg, teklifler)
    mf10 = int(np.flatnonzero((mat.uzay.mf == 0.10)
                              & (mat.uzay.vade == cfg.politika.aksiyon.taban_vade_gun))[0])
    # adet < 10 -> floor(adet * 0.10) = 0 -> kapali
    assert not mat.izinli[:3, mf10].any()
    assert mat.izinli[3:, mf10].all()


def test_vade_uzadikca_marj_dusuyor(cfg):
    """Net isletme sermayesi: musteri vadesi uzadikca fonlama maliyeti artar."""
    n = 1
    dunya = _SahteDunya(n, koli=[1])
    mat = scorer.teklif_matrisleri(dunya, cfg, _aday(n, adet=np.array([50.0])))
    mfsiz = [a for a in range(mat.uzay.A) if mat.uzay.mf[a] == 0.0 and a > 0]
    sirali = sorted(mfsiz, key=lambda a: mat.uzay.vade[a])
    marjlar = [mat.marj[0, a] for a in sirali]
    assert marjlar == sorted(marjlar, reverse=True)


def test_mf_derinlestikce_marj_dusuyor(cfg):
    """MF bedava mal demektir; destek orani 1.0 olmadikca marj duser."""
    n = 1
    dunya = _SahteDunya(n, koli=[1])
    mat = scorer.teklif_matrisleri(dunya, cfg, _aday(n, adet=np.array([200.0])))
    taban = cfg.politika.aksiyon.taban_vade_gun
    kollar = sorted([a for a in range(1, mat.uzay.A) if mat.uzay.vade[a] == taban],
                    key=lambda a: mat.uzay.mf[a])
    marjlar = [mat.marj[0, a] for a in kollar]
    assert marjlar == sorted(marjlar, reverse=True)


def test_frekans_tavani_aksiyon_seciminde_de_bagliyor(cfg):
    """Eczane basina en cok `tavan` satir teklife donusur."""
    n = 12
    deger = np.zeros((n, 3))
    deger[:, 1] = np.linspace(10.0, 1.0, n)
    izinli = np.ones((n, 3), dtype=bool)
    ecz = np.zeros(n, dtype=np.int32)
    secim = scorer.sec("t", deger, np.zeros(n), izinli, ecz, tavan=4, esik=0.0)
    assert int(secim.teklif_maskesi.sum()) == 4
    # En yuksek degerli dort satir secilmis olmali.
    assert set(np.flatnonzero(secim.teklif_maskesi)) == {0, 1, 2, 3}


def test_esik_altindaki_kazanc_teklife_donusmuyor(cfg):
    """Uplift politikasinin 'teklif verme' karari: kazanc esigi gecmezse yok."""
    n = 4
    deger = np.zeros((n, 2))
    deger[:, 1] = np.array([5.0, 1.0, 0.5, 0.1])
    izinli = np.ones((n, 2), dtype=bool)
    ecz = np.zeros(n, dtype=np.int32)
    secim = scorer.sec("t", deger, np.full(n, 1.0), izinli, ecz, tavan=10, esik=0.0)
    # taban_deger = 1.0 -> yalnizca ilk satirin kazanci pozitif
    assert int(secim.teklif_maskesi.sum()) == 1


def test_izinsiz_kol_asla_secilmiyor(cfg):
    n = 5
    deger = np.tile(np.array([0.0, 100.0, 1.0]), (n, 1))
    izinli = np.ones((n, 3), dtype=bool)
    izinli[:, 1] = False                      # en yuksek degerli kol kapali
    ecz = np.zeros(n, dtype=np.int32)
    secim = scorer.sec("t", deger, np.zeros(n), izinli, ecz, tavan=n, esik=-np.inf)
    assert not (secim.kol == 1).any()
    assert (secim.kol[secim.teklif_maskesi] == 2).all()
