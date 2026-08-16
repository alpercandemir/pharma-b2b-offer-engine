"""M4 testleri: tepki fonksiyonu, kayit politikasi (D7), CATE ogreniciler,
politika ayrimi ve sizinti guard'lari.

En onemli iki test:

  `test_propensity_ve_uplift_ayni_kolu_seciyor`
      Ayni olasilik matrisiyle beslendiklerinde iki politikanin AKSIYON
      secimi ozdes olmak zorunda -- karsi-olgusal terim `p(0)*marj(0)`
      kola gore degismiyor, dolayisiyla argmax'i degistiremez. Fark
      yalnizca HANGI SATIRA teklif verildiginde ortaya cikabilir. M4
      raporunun ana iddiasi budur ve burada mekanik olarak sinaniyor.

  `test_teklif_ozellikleri_point_in_time`
      Gelecek silinince ozellik matrisi bit bazinda ayni kalmali.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from core.config import load_config
from core.io import Run
from core.rng import SeedBank
from eval import uplift as ev
from experiments.run import _origin_blogu, _politikalar, m4_boru_hatti
from features import teklif as ft
from features.okuma import GozlemlenebilirKaynak
from models import uplift as mu
from policy import bandit, scorer
from policy import candidates as pol
from scripts.generate_world import dunya_yaz
from sim import response as tepki_modulu
from sim.response import (GercekDurum, beklenen_miktar_carpani, sonuc_ornekle,
                          tepki_evreni_kur, tepki_hesapla)

PROFIL = "fast"
KOSU = "_test_m4"


@pytest.fixture(scope="module")
def cfg():
    return load_config(PROFIL)


@pytest.fixture(scope="module")
def dunya_dizini(tmp_path_factory, cfg):
    kok = tmp_path_factory.mktemp("m4")
    dunya_yaz(cfg, Run("dunya", kok=kok))
    return kok


@pytest.fixture(scope="module")
def ortam(cfg, dunya_dizini):
    kaynak = GozlemlenebilirKaynak("dunya", kok=dunya_dizini)
    dunya = pol.dunya_yukle(kaynak, cfg)
    td = ft.teklif_dunyasi_yukle(kaynak, cfg, dunya)
    durum = GercekDurum("dunya", kok=dunya_dizini)
    seedler = SeedBank(cfg.profil.temel_seed)
    evren = tepki_evreni_kur(cfg, seedler, dunya.eczaneler, dunya.urunler,
                             durum.latent_eczane)
    t = pol.origin_haftalari(cfg, dunya.W)[-1]
    blok = _origin_blogu(td, cfg, dunya, t)
    tp = tepki_hesapla(cfg, evren, durum, blok.mat.uzay, blok.teklifler, t,
                       blok.mat.adet)
    return dict(kok=dunya_dizini, dunya=dunya, td=td, durum=durum, evren=evren,
                seedler=seedler, t=t, blok=blok, tepki=tp,
                carpan=beklenen_miktar_carpani(cfg))


# --------------------------------------------------------------------------
# tepki fonksiyonu (ground truth)
# --------------------------------------------------------------------------
def test_kontrol_kolu_indeksi_iki_katmanda_ayni():
    """sim/ politikayi import etmiyor; kontrol kolu indeksi iki yerde yazili."""
    assert tepki_modulu.TEKLIF_YOK_KOLU == scorer.TEKLIF_YOK


def test_tepki_olasiliklari_gecerli(cfg, ortam):
    p = ortam["tepki"].olasilik
    assert p.shape == (ortam["blok"].teklifler.height, ortam["blok"].mat.uzay.A)
    assert np.isfinite(p).all()
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_bedelsiz_teklif_kabulu_yukseltiyor(cfg, ortam):
    """`taban_etki` > 0: gorunur olmak tek basina kabul olasiligini artirir.

    Miad direnci kapatilarak olculur: kisa miatli lotta teklif kolunun
    olasiligi kontrolun ALTINA inebilir ve bu dogru davranistir (SPEC 2.5),
    ama olculmek istenen sey o degil.
    """
    kapali = load_config(PROFIL,
                          gecersiz_kilma={"tepki.miad.direnc_katsayisi": 0.0})
    tp = tepki_hesapla(kapali, ortam["evren"], ortam["durum"],
                       ortam["blok"].mat.uzay, ortam["blok"].teklifler,
                       ortam["t"], ortam["blok"].mat.adet)
    uzay = ortam["blok"].mat.uzay
    taban_kol = int(np.flatnonzero(
        (uzay.mf == 0.0) & (uzay.vade == cfg.politika.aksiyon.taban_vade_gun)
        & (np.arange(uzay.A) > 0))[0])
    # Ayni adet, ayni sart; tek fark teklifin varligi.
    ayni_adet = ortam["blok"].mat.adet[:, taban_kol] == ortam["blok"].mat.adet[:, 0]
    assert ayni_adet.any()
    assert (tp.olasilik[ayni_adet, taban_kol] >= tp.olasilik[ayni_adet, 0]).all()


def test_ihtiyac_etkilesimi_upliftı_sonduruyor(cfg, ortam):
    """Acil ihtiyaci olan hucrede teklif esnekligi DUSUK olmali (D2)."""
    tp = ortam["tepki"]
    izin = ortam["blok"].mat.izinli.copy()
    izin[:, scorer.TEKLIF_YOK] = False
    en_iyi = np.where(izin, tp.uplift, -np.inf).max(axis=1)
    yuksek = tp.ihtiyac >= np.quantile(tp.ihtiyac, 0.75)
    dusuk = tp.ihtiyac <= np.quantile(tp.ihtiyac, 0.25)
    assert en_iyi[yuksek].mean() < en_iyi[dusuk].mean()


def test_heterojenlik_carpani_sifirda_duyarlilik_esitleniyor(cfg, ortam):
    """`heterojenlik_carpani = 0` -> butun eczaneler ayni duyarlilikta."""
    kapali = load_config(
        PROFIL, gecersiz_kilma={"tepki.duyarlilik.heterojenlik_carpani": 0.0})
    evren = tepki_evreni_kur(kapali, ortam["seedler"], ortam["dunya"].eczaneler,
                             ortam["dunya"].urunler, ortam["durum"].latent_eczane)
    assert np.allclose(evren.mf_duyarliligi, 1.0)
    assert np.allclose(evren.vade_duyarliligi, 1.0)


def test_tepki_tekrar_uretilebilir(cfg, ortam):
    a = tepki_hesapla(cfg, ortam["evren"], ortam["durum"], ortam["blok"].mat.uzay,
                      ortam["blok"].teklifler, ortam["t"], ortam["blok"].mat.adet)
    assert np.array_equal(a.olasilik, ortam["tepki"].olasilik)
    k1, m1 = sonuc_ornekle(cfg, SeedBank(cfg.profil.temel_seed), a,
                           np.zeros(a.olasilik.shape[0], dtype=int), ortam["t"])
    k2, m2 = sonuc_ornekle(cfg, SeedBank(cfg.profil.temel_seed), a,
                           np.zeros(a.olasilik.shape[0], dtype=int), ortam["t"])
    assert np.array_equal(k1, k2) and np.array_equal(m1, m2)


# --------------------------------------------------------------------------
# kayit politikasi (D7)
# --------------------------------------------------------------------------
def test_propensity_dagilimi_gecerli(cfg, ortam):
    """Satirlar 1.0'a toplanir, izinli kollar POZITIF, izinsizler tam sifir."""
    blok = ortam["blok"]
    pi = bandit.kayit_olasiliklari(ortam["dunya"], cfg, blok.mat.uzay,
                                   blok.teklifler, blok.mat.izinli)
    assert np.allclose(pi.sum(axis=1), 1.0)
    izinli = blok.mat.izinli.copy()
    izinli[:, scorer.TEKLIF_YOK] = True
    assert (pi[izinli] > 0).all()
    assert (pi[~izinli] == 0).all()


def test_kesif_orani_overlap_garantisi(cfg, ortam):
    """eps > 0 -> her izinli kolun propensity'si eps/|izinli| tabanindan buyuk."""
    blok = ortam["blok"]
    pi = bandit.kayit_olasiliklari(ortam["dunya"], cfg, blok.mat.uzay,
                                   blok.teklifler, blok.mat.izinli)
    izinli_teklif = blok.mat.izinli.copy()
    izinli_teklif[:, scorer.TEKLIF_YOK] = False
    n_izinli = izinli_teklif.sum(axis=1, keepdims=True)
    taban = cfg.uplift.kayit.kesif_orani / np.maximum(n_izinli, 1)
    # pi >= q * eps / |izinli|; q >= Q_ALT.
    assert (pi[izinli_teklif] >= (bandit.Q_ALT * taban * np.ones_like(pi))[izinli_teklif]
            - 1e-12).all()


def test_kesif_orani_sifir_config_reddediliyor():
    """Overlap kirilmasi kod incelemesine degil config yuklemesine bagli."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="kesif_orani"):
        load_config(PROFIL, gecersiz_kilma={"uplift.kayit.kesif_orani": 0.0})


def test_kayit_secimi_propensity_ile_tutarli(cfg, ortam):
    """Loglanan propensity, secilen kolun pi'si olmali."""
    blok = ortam["blok"]
    k = bandit.kayit_kosusu(ortam["dunya"], cfg, ortam["seedler"], blok.mat.uzay,
                            blok.teklifler, blok.mat.izinli, ortam["t"])
    n = k.kol.size
    assert np.allclose(k.propensity, k.pi[np.arange(n), k.kol])
    assert (k.propensity > 0).all()
    izinli = blok.mat.izinli.copy()
    izinli[:, scorer.TEKLIF_YOK] = True
    assert izinli[np.arange(n), k.kol].all()


# --------------------------------------------------------------------------
# politikalarin ayrimi -- M4'un ana iddiasi
# --------------------------------------------------------------------------
def test_propensity_ve_uplift_ayni_kolu_seciyor(cfg, ortam):
    """AYNI olasilik matrisiyle iki politikanin AKSIYON secimi ozdestir.

    Karsi-olgusal terim p(0)*marj(0) kola gore sabittir:
        argmax_a [p(a)m(a)]  ==  argmax_a [p(a)m(a) - p(0)m(0)]
    Yani propensity ile uplift arasindaki fark ASLA "hangi MF/vade" degil,
    yalnizca "hangi satira teklif" olabilir. M4 raporunun merkezindeki
    ayristirma (`farkli_kol_katki`) bu ozdeslikten cikiyor.
    """
    blok = ortam["blok"]
    n = blok.teklifler.height
    rng = np.random.default_rng(0)
    p = rng.random((n, blok.mat.uzay.A))
    secim = _politikalar(cfg, blok, p, p, ortam["tepki"].olasilik, ortam["carpan"])
    ikisi = secim["propensity"].teklif_maskesi & secim["uplift_t"].teklif_maskesi
    assert ikisi.any()
    assert np.array_equal(secim["propensity"].kol[ikisi], secim["uplift_t"].kol[ikisi])


def test_uplift_negatif_artimsal_satiri_eliyor(cfg, ortam):
    """Uplift politikasi kazanci esigin altinda kalan satira teklif vermez."""
    blok = ortam["blok"]
    n = blok.teklifler.height
    # Teklif kollari kontrolden cok KOTU ama sifir degil: propensity yine
    # teklif verir (tabani sifir sayar), uplift vermez.
    p = np.full((n, blok.mat.uzay.A), 0.01)
    p[:, scorer.TEKLIF_YOK] = 1.0
    secim = _politikalar(cfg, blok, p, p, ortam["tepki"].olasilik, ortam["carpan"])
    assert secim["uplift_t"].teklif_maskesi.sum() == 0
    # Propensity tabani sifir kabul ettigi icin yine teklif verir: korlugun ta kendisi.
    assert secim["propensity"].teklif_maskesi.sum() > 0


# --------------------------------------------------------------------------
# ogreniciler
# --------------------------------------------------------------------------
def test_destek_disi_kol_sifir_cate_donduruyor(cfg):
    """Orneklemi yetersiz kolda uydurma yok: tau = 0 (mu_a = mu_0)."""
    rng = np.random.default_rng(0)
    n, A = 2000, 4
    X = rng.normal(size=(n, 3)).astype(np.float32)
    kol = rng.choice([0, 1, 2], size=n, p=[0.5, 0.45, 0.05])
    y = (rng.random(n) < 0.3).astype(int)
    t = mu.TOgrenici(cfg, A, []).egit(X, kol, y)
    cate = t.cate(X)
    assert 3 not in t.destekli                     # hic gozlenmemis kol
    assert np.allclose(cate[:, 3], 0.0)
    assert np.allclose(cate[:, scorer.TEKLIF_YOK], 0.0)


def test_x_ogrenici_propensity_kullaniyor(cfg):
    """g(x) = pi_a / (pi_0 + pi_a); pi degisince CATE degisir (D7 bagimliligi)."""
    rng = np.random.default_rng(1)
    n, A = 4000, 3
    X = rng.normal(size=(n, 3)).astype(np.float32)
    kol = rng.choice([0, 1, 2], size=n, p=[0.6, 0.2, 0.2])
    y = ((X[:, 0] + (kol > 0) * 1.5 + rng.normal(size=n)) > 0).astype(int)
    t = mu.TOgrenici(cfg, A, []).egit(X, kol, y)
    x = mu.XOgrenici(cfg, t, []).egit(X, kol, y)
    pi_a = np.tile(np.array([0.6, 0.2, 0.2]), (n, 1))
    pi_b = np.tile(np.array([0.9, 0.05, 0.05]), (n, 1))
    assert not np.allclose(x.cate(X, pi_a), x.cate(X, pi_b))


def test_ogrenici_tekrar_uretilebilir(cfg):
    rng = np.random.default_rng(2)
    n, A = 3000, 3
    X = rng.normal(size=(n, 3)).astype(np.float32)
    kol = rng.choice([0, 1, 2], size=n)
    y = ((X[:, 0] + rng.normal(size=n)) > 0).astype(int)
    a = mu.TOgrenici(cfg, A, []).egit(X, kol, y).olasilik(X)
    b = mu.TOgrenici(cfg, A, []).egit(X, kol, y).olasilik(X)
    assert np.array_equal(a, b)


def test_egitim_originleri_olcume_tasmiyor(cfg):
    """Zaman bolmesi: egitim origin'leri olcum penceresine giremez."""
    W = cfg.profil.hafta_sayisi
    egitim = mu.egitim_originleri(cfg, W)
    d = cfg.politika.aday.degerlendirme
    olcum = pol.origin_haftalari(cfg, W)
    assert max(egitim) <= min(olcum) - cfg.uplift.egitim.sinir_tamponu_hafta
    assert len(egitim) <= cfg.uplift.egitim.azami_origin_sayisi


# --------------------------------------------------------------------------
# olcum
# --------------------------------------------------------------------------
def test_ayristirma_toplama_esit(cfg, ortam):
    """Bes sinifin katkisi toplam farka ESIT (ozdeslik)."""
    blok = ortam["blok"]
    n = blok.teklifler.height
    rng = np.random.default_rng(3)
    a = scorer.Secim("a", rng.integers(0, blok.mat.uzay.A, n).astype(np.int32),
                     np.zeros(n))
    b = scorer.Secim("b", rng.integers(0, blok.mat.uzay.A, n).astype(np.int32),
                     np.zeros(n))
    # Izinsiz kollar secilmis olabilir; ayristirma yine ozdes olmali.
    ayr = ev.marj_farki_ayristirmasi(ortam["tepki"], blok.mat, a, b, ortam["carpan"])
    toplam = sum(v for k, v in ayr.items() if k.endswith("_katki"))
    assert toplam == pytest.approx(ayr["toplam_fark"], abs=1e-6)


def test_artimsal_marj_kontrol_kolunda_sifir(cfg, ortam):
    n = ortam["blok"].teklifler.height
    kol = np.zeros(n, dtype=int)
    art = ev.artimsal_marj(ortam["tepki"], ortam["blok"].mat, kol, ortam["carpan"])
    assert np.array_equal(art, np.zeros(n))


# --------------------------------------------------------------------------
# sizinti guard'lari
# --------------------------------------------------------------------------
def test_teklif_ozellikleri_point_in_time(cfg, ortam, tmp_path):
    """Gelecek silinince ayni origin'in ozellik matrisi BIT BAZINDA ayni."""
    kaynak = GozlemlenebilirKaynak("dunya", kok=ortam["kok"])
    kesme = ortam["dunya"].W // 2
    hedef = Run("kesilmis", kok=tmp_path).prepare()
    for tablo in kaynak.tables():
        df = kaynak.tablo(tablo)
        if "hafta" in df.columns:
            df = df.filter(pl.col("hafta") <= kesme)
        hedef.write_observable(tablo, df)
    kesik_kaynak = GozlemlenebilirKaynak("kesilmis", kok=tmp_path)
    kesik = pol.dunya_yukle(kesik_kaynak, cfg)

    a = _origin_blogu(ortam["td"], cfg, ortam["dunya"], kesme)
    b = _origin_blogu(ft.teklif_dunyasi_yukle(kesik_kaynak, cfg, kesik), cfg,
                      kesik, kesme)
    assert a.X.shape == b.X.shape
    assert np.array_equal(np.nan_to_num(a.X), np.nan_to_num(b.X))


def test_feature_ve_policy_katmani_tepkiyi_gormuyor():
    """`sim/response.py` politika/feature/model katmaninda import EDILMEZ.

    Docstring'ler sinirin NEDEN boyle kuruldugunu anlatirken bu adlari
    kullaniyor; taramada yorum ve docstring'ler ayiklanir (verify_m2).
    """
    from scripts.verify_m2 import kod_metni
    kok = Path(__file__).resolve().parent.parent
    bulgular = []
    for dizin in ("policy", "features", "models"):
        for yol in sorted((kok / dizin).glob("*.py")):
            kod = kod_metni(yol)
            for yasak in ("sim.response", "sim/response", "tepki_hesapla",
                          "GercekDurum", "ground_truth"):
                if yasak in kod:
                    bulgular.append(f"{dizin}/{yol.name}:{yasak}")
    assert not bulgular, bulgular
