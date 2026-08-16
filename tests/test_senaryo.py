"""M7 senaryo katmani testleri (D3 + D4).

Uc grup:

  KILIT    : config yuklemesindeki mekanik kilitler. Taban rejimin notr
             olmasi ve en az bir rejimin notr OLMAMASI kod incelemesine
             degil yuklemeye bagli; gevsetme denemesi kosuyu dusurmeli.
  FORMUL   : erteleme kazancinin kapali formu. Buyuk kosuda "makul
             goruniyor" diye kabul edilemez; dogru cevabi burada kagit
             uzerinde biliyoruz (M5'in gölge fiyat testleriyle ayni
             disiplin).
  KANAL    : rejim carpanlarinin FIILEN dogru girdiyi oynattigi. Fonlama
             carpani vade maliyetini, antisipasyon carpani adedi
             degistirmeli; birbirinin yerine gecmemeli.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from agent import scenario as sc
from core.config import config_yukle
from sim.calendar import GUN_HAFTA

PROFIL = "fast"


@pytest.fixture(scope="module")
def cfg():
    return config_yukle(PROFIL)


# --------------------------------------------------------------------------
# KILIT
# --------------------------------------------------------------------------
def test_taban_rejim_notr_olmali():
    """Notr olmayan taban 'tabana gore fark' sutunlarini anlamsiz kilar."""
    with pytest.raises(ValidationError, match="notr degil"):
        config_yukle(PROFIL, gecersiz_kilma={
            "senaryo.rejimler": [
                {"ad": "baz", "aciklama": "x", "guncelleme_beklentisi_hafta": 26.0,
                 "referans_kur_artisi": 0.10, "fiyat_gecis_katsayisi": 0.5,
                 "antisipasyon_talep_carpani": 1.0, "fonlama_orani_carpani": 1.0},
                {"ad": "sok", "aciklama": "y", "guncelleme_beklentisi_hafta": 2.0,
                 "referans_kur_artisi": 0.30, "fiyat_gecis_katsayisi": 0.85,
                 "antisipasyon_talep_carpani": 2.2, "fonlama_orani_carpani": 1.6},
            ]})


def test_butun_rejimler_notrse_katman_olu():
    """Uc notr rejim ayni tabloyu uretir: senaryo katmani olu demektir."""
    notr = {"aciklama": "x", "guncelleme_beklentisi_hafta": 26.0,
            "referans_kur_artisi": 0.0, "fiyat_gecis_katsayisi": 0.0,
            "antisipasyon_talep_carpani": 1.0, "fonlama_orani_carpani": 1.0}
    with pytest.raises(ValidationError, match="senaryo katmani olu"):
        config_yukle(PROFIL, gecersiz_kilma={
            "senaryo.rejimler": [{"ad": "baz"} | notr, {"ad": "ikinci"} | notr]})


def test_taban_ad_rejimler_icinde_olmali():
    with pytest.raises(ValidationError, match="taban_ad"):
        config_yukle(PROFIL, gecersiz_kilma={"senaryo.taban_ad": "olmayan"})


def test_politika_adi_dogrulaniyor():
    """Kodun tanimadigi politika adi sessizce atlanmaz."""
    with pytest.raises(ValidationError, match="senaryo.politika tanimsiz"):
        config_yukle(PROFIL, gecersiz_kilma={"senaryo.politika": "yok_boyle"})


def test_teklif_yok_politikasi_reddediliyor():
    """Hic teklif vermeyen politikanin rejim duyarliligi tanimi geregi sifir."""
    with pytest.raises(ValidationError, match="senaryo.politika tanimsiz"):
        config_yukle(PROFIL, gecersiz_kilma={"senaryo.politika": "teklif_yok"})


# --------------------------------------------------------------------------
# FORMUL
# --------------------------------------------------------------------------
def test_beklenti_payi_ufkun_disinda_sifir(cfg):
    """Guncelleme ikame ufkunun otesindeyse bugunun karari etkilenmez."""
    uzak = cfg.senaryo.rejim("baz")
    assert uzak.guncelleme_beklentisi_hafta >= cfg.senaryo.ikame_ufku_hafta
    assert sc.beklenti_payi(cfg, uzak) == 0.0


def test_beklenti_payi_yaklastikca_buyuyor(cfg):
    paylar = [sc.beklenti_payi(cfg, r) for r in
              sorted(cfg.senaryo.rejimler,
                     key=lambda r: -r.guncelleme_beklentisi_hafta)]
    assert paylar == sorted(paylar)
    assert all(0.0 <= p <= 1.0 for p in paylar)


def test_taban_rejimde_erteleme_tam_sifir(cfg):
    """Taban bir mudahale degil, olcum sifiri: kalem TAM sifir olmali."""
    taban = cfg.senaryo.rejim(cfg.senaryo.taban_ad)
    dsf = np.array([100.0, 500.0, 1000.0])
    kalan = np.array([1000.0, 200.0, 130.0])
    soguk = np.zeros(3, dtype=bool)
    assert np.all(sc.erteleme_kazanci(cfg, taban, dsf, kalan, soguk) == 0.0)


def test_erteleme_kapali_form(cfg):
    """Formul dosya basligindakinin ta kendisi; carpanlar tek tek."""
    r = cfg.senaryo.rejim("sok")
    dsf = np.array([100.0])
    # Bekleyebilen lot: taban + bekleme suresinin uzerinde.
    kalan = np.array([cfg.politika.kisit.asgari_kalan_raf_omru_gun
                      + r.guncelleme_beklentisi_hafta * GUN_HAFTA + 1.0])
    soguk = np.zeros(1, dtype=bool)
    beklenen = (100.0 * r.referans_kur_artisi * r.fiyat_gecis_katsayisi
                * cfg.tahsis.temizlik.normal_realizasyon_orani
                * sc.beklenti_payi(cfg, r))
    assert sc.erteleme_kazanci(cfg, r, dsf, kalan, soguk)[0] == pytest.approx(beklenen)


def test_bekleyemeyen_lotta_erteleme_sifir(cfg):
    """D9 kesismesi: guncellemeyi tasimayan lot bekleyemez, kalem sifirdir."""
    r = cfg.senaryo.rejim("sok")
    sinir = (cfg.politika.kisit.asgari_kalan_raf_omru_gun
             + r.guncelleme_beklentisi_hafta * GUN_HAFTA)
    dsf = np.full(2, 100.0)
    kalan = np.array([sinir - 1.0, sinir + 1.0])
    soguk = np.zeros(2, dtype=bool)
    kazanc = sc.erteleme_kazanci(cfg, r, dsf, kalan, soguk)
    assert kazanc[0] == 0.0 and kazanc[1] > 0.0


def test_soguk_zincirde_bekleme_kapisi_daha_dar(cfg):
    """Soguk zincirde tolerans penceresi dar (SPEC 2.5); bekleme lüksü de az."""
    r = cfg.senaryo.rejim("sok")
    k = cfg.politika.kisit
    if k.soguk_zincir_raf_omru_carpani <= 1.0:
        pytest.skip("soguk zincir carpani gevsetici ayarlanmis")
    sinir = k.asgari_kalan_raf_omru_gun + r.guncelleme_beklentisi_hafta * GUN_HAFTA
    kalan = np.array([sinir + 1.0, sinir + 1.0])
    soguk = np.array([False, True])
    kapi = sc.bekleyebilir(cfg, r, kalan, soguk)
    assert kapi[0] and not kapi[1]


# --------------------------------------------------------------------------
# KANAL
# --------------------------------------------------------------------------
def test_fonlama_carpani_yalnizca_fonlamayi_oynatiyor(cfg):
    """Rejim config'i TEK alani degistirir; digerleri birebir korunur."""
    r = cfg.senaryo.rejim("sok")
    yeni = sc.rejim_config(cfg, r)
    assert yeni.politika.skor.yillik_fonlama_orani == pytest.approx(
        cfg.politika.skor.yillik_fonlama_orani * r.fonlama_orani_carpani)
    assert yeni.dunya_hash() == cfg.dunya_hash()
    assert (yeni.politika.aksiyon.mf_oranlari
            == cfg.politika.aksiyon.mf_oranlari)
    assert yeni.politika.kisit == cfg.politika.kisit


def test_rejim_config_diger_gecersiz_kilmalari_korur():
    """`config_yukle` ile yeniden okunsaydi sweep'in knob'i silinirdi."""
    cfg = config_yukle(PROFIL, gecersiz_kilma={
        "politika.kisit.eczane_haftalik_teklif_tavani": 3})
    yeni = sc.rejim_config(cfg, cfg.senaryo.rejim("sok"))
    assert yeni.politika.kisit.eczane_haftalik_teklif_tavani == 3


def test_antisipasyon_hiz_ve_adedi_birlikte_kaydiriyor(cfg):
    """Talep carpani hem hiz tahminini hem teklif adedini oynatmali."""
    r = cfg.senaryo.rejim("sok")
    havuz = pl.DataFrame({"hiz_tahmini": [1.0, 4.0, 10.0],
                          "teklif_adedi": [1.0, 1.0, 1.0]})
    yeni = sc.senaryo_talebi(havuz, cfg, r)
    assert np.allclose(yeni["hiz_tahmini"].to_numpy(),
                       havuz["hiz_tahmini"].to_numpy() * r.antisipasyon_talep_carpani)
    assert np.all(yeni["teklif_adedi"].to_numpy() >= havuz["teklif_adedi"].to_numpy())


def test_notr_rejimde_talep_dokunulmuyor(cfg):
    taban = cfg.senaryo.rejim(cfg.senaryo.taban_ad)
    havuz = pl.DataFrame({"hiz_tahmini": [1.0, 4.0], "teklif_adedi": [7.0, 9.0]})
    assert sc.senaryo_talebi(havuz, cfg, taban).equals(havuz)
