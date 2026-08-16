"""M5 tahsis katmani testleri (D5 + D9).

Bu dosya `policy/allocate.py`nin degismezlerini sinar. Uc grup:

  SALVAGE  : SPEC 2.5'in dinamik deger fonksiyonu ve isaret degisim noktasi.
  KUPLAJ   : SPEC 2.5'in `max_teklif_adedi` formulu; sabit tavan YOK.
  LP       : el ile kurulmus kucuk ornekler uzerinde tahsisin ve GÖLGE
             FIYATIN dogrulugu. Buyuk kosuda gölge fiyat "makul goruniyor"
             diye kabul edilemez; burada dogru cevabi kagit uzerinde
             biliyoruz.

Config kilitleri (temizlik rejiminin gevsetme olmasi, penceresinin bos
olmamasi) da burada sinaniyor: bunlar kod incelemesine degil config
yuklemesine bagli.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from core.config import config_yukle
from eval import allocation as ev
from policy import allocate as alloc
from policy import scorer
from sim.calendar import GUN_HAFTA

PROFIL = "fast"


@pytest.fixture(scope="module")
def cfg():
    return config_yukle(PROFIL)


# --------------------------------------------------------------------------
# SALVAGE (D9 / SPEC 2.5)
# --------------------------------------------------------------------------
def test_salvage_isaret_degistiriyor(cfg):
    """SPEC 2.5: deger uzun miatta pozitif, miad sonrasi -imha_maliyeti."""
    t = cfg.tahsis.temizlik
    dsf = np.full(5, 100.0)
    marj = np.full(5, 0.05)
    gun = np.array([t.tetik_gun * 3, t.tetik_gun, t.tetik_gun / 2, 1.0, -10.0])
    normal, v = alloc.salvage_degeri(cfg, dsf, marj, gun)

    assert np.allclose(normal, 100.0 * 0.05 * t.normal_realizasyon_orani)
    assert v[0] == pytest.approx(normal[0])          # pencere disinda = normal
    assert v[1] == pytest.approx(normal[1])          # tetigin tam ustunde
    assert v[-1] == pytest.approx(-100.0 * t.imha_birim_maliyeti_dsf_orani)
    # Miad sonrasi deger NEGATIF: stok varlik degil yukumluluk.
    assert v[-1] < 0 < v[0]
    assert np.all(np.diff(v) <= 1e-9)                # kalan gun azaldikca azalir


def test_salvage_isaret_esigi_kapali_form(cfg):
    """Isaret degisim noktasi bir iddia degil, kapali formdan gelen bir sayi."""
    esik = alloc.isaret_esigi_gun(cfg, dsf=100.0, depo_marji=0.05)
    alt, ust = np.array([esik - 1.0]), np.array([esik + 1.0])
    _, v_alt = alloc.salvage_degeri(cfg, np.full(1, 100.0), np.full(1, 0.05), alt)
    _, v_ust = alloc.salvage_degeri(cfg, np.full(1, 100.0), np.full(1, 0.05), ust)
    assert v_alt[0] < 0 < v_ust[0]


@pytest.mark.parametrize("egri", ["lineer", "eksponansiyel", "basamakli"])
def test_salvage_egri_tipleri_uc_noktalarda_ayni(cfg, egri):
    """Egri tipi ARADAKI yolu degistirir, uc noktalari degil."""
    alt_cfg = config_yukle(PROFIL, gecersiz_kilma={"tahsis.temizlik.deger_egrisi": egri})
    t = alt_cfg.tahsis.temizlik
    dsf, marj = np.full(2, 100.0), np.full(2, 0.05)
    normal, v = alloc.salvage_degeri(alt_cfg, dsf, marj,
                                     np.array([t.tetik_gun, 0.0]))
    assert v[0] == pytest.approx(normal[0])
    assert v[1] == pytest.approx(-100.0 * t.imha_birim_maliyeti_dsf_orani)


# --------------------------------------------------------------------------
# M2 KUPLAJI (SPEC 2.5)
# --------------------------------------------------------------------------
def test_azami_teklif_adedi_formulu(cfg):
    """max_teklif_adedi = tuketim_hizi x (kalan_gun - eczaci_marji) x guvenlik."""
    t = cfg.tahsis.temizlik
    hiz_haftalik = np.array([7.0, 7.0, 0.7])
    kalan_gun = np.array([t.eczaci_marji_gun + 100.0, t.eczaci_marji_gun, 400.0])
    tavan = alloc.azami_teklif_adedi(cfg, hiz_haftalik, kalan_gun)

    beklenen = 1.0 * 100.0 * t.guvenlik_katsayisi          # gunluk hiz = 7/7 = 1
    assert tavan[0] == pytest.approx(beklenen)
    # Kalan gun eczaci marjina esitse tavan SIFIR: o eczane o lot icin aday degil.
    assert tavan[1] == pytest.approx(0.0)
    # AYNI MIAD, FARKLI HIZ -> farkli tavan. Sabit gun esigi bunu yapamazdi.
    assert tavan[2] == pytest.approx(
        0.1 * (400.0 - t.eczaci_marji_gun) * t.guvenlik_katsayisi)


def test_azami_teklif_adedi_miad_gecmisse_sifir(cfg):
    tavan = alloc.azami_teklif_adedi(cfg, np.array([100.0]), np.array([-5.0]))
    assert tavan[0] == 0.0


def test_kuplaj_hizla_olcekleniyor(cfg):
    """Tavan tuketim hiziyla DOGRUSAL: sabit bir tavan degil (SPEC 2.5)."""
    gun = np.full(3, 200.0)
    tavan = alloc.azami_teklif_adedi(cfg, np.array([1.0, 2.0, 4.0]) * GUN_HAFTA, gun)
    assert tavan[1] == pytest.approx(2 * tavan[0])
    assert tavan[2] == pytest.approx(4 * tavan[0])


# --------------------------------------------------------------------------
# CONFIG KILITLERI
# --------------------------------------------------------------------------
def test_temizlik_rejimi_sikilastirma_olamaz():
    """D9: temizlik bir GEVSETMEDIR. Normal tabandan yuksek taban reddedilir."""
    normal = config_yukle(PROFIL).politika.kisit.asgari_kalan_raf_omru_gun
    with pytest.raises(ValidationError, match="gevsetme olmali"):
        config_yukle(PROFIL, gecersiz_kilma={
            "tahsis.temizlik.asgari_kalan_raf_omru_gun": normal + 1.0})


def test_olu_temizlik_penceresi_reddediliyor():
    """Tetik gunu temizlik tabaninin altina inerse rejim OLU olur."""
    with pytest.raises(ValidationError, match="rejim olu"):
        config_yukle(PROFIL, gecersiz_kilma={
            "tahsis.temizlik.tetik_gun": 40.0,
            "tahsis.temizlik.asgari_kalan_raf_omru_gun": 45.0})


def test_senaryo_dunyayi_degistirmiyor():
    """Senaryo kadranlari GORUNUMU degistirir, dunyayi degil (D3 disiplini)."""
    taban = config_yukle(PROFIL)
    for knob, deger in (("tahsis.senaryo.kit_stok_carpani", 0.3),
                        ("tahsis.senaryo.miad_hizlandirma_gun", 90)):
        alt = config_yukle(PROFIL, gecersiz_kilma={knob: deger})
        assert alt.dunya_hash() == taban.dunya_hash()
        assert alt.hash() != taban.hash()


# --------------------------------------------------------------------------
# LP: el ile kurulmus ornekler
# --------------------------------------------------------------------------
class _SahteDunya:
    """`lp_tahsisi` ve `acgozlu_tahsis` icin gereken minimal AdayDunyasi yuzu."""

    def __init__(self, P: int, S: int, dsf=100.0, marj=0.05):
        self.P, self.S = P, S
        self.dsf = np.full(S, dsf)
        self.eczaneler = pl.DataFrame({
            "dbs_limiti": np.full(P, 1e12),
            "vade_riski_skoru": np.zeros(P),
        })
        self.urunler = pl.DataFrame({"depo_kar_marji": np.full(S, marj)})


class _SahteGorunum:
    def __init__(self, t: int, P: int):
        self.t = t
        self.acik_bakiye = np.zeros(P)


def _lot_gorunumu(sku_idx, adet, kalan_gun, cfg, dsf=100.0, marj=0.05):
    sku_idx = np.asarray(sku_idx, dtype=np.int32)
    adet = np.asarray(adet, dtype=float)
    kalan_gun = np.asarray(kalan_gun, dtype=float)
    normal, salvage = alloc.salvage_degeri(
        cfg, np.full(sku_idx.size, dsf), np.full(sku_idx.size, marj), kalan_gun)
    sku_lotlari: dict[int, list[int]] = {}
    for i, s in enumerate(sku_idx):
        sku_lotlari.setdefault(int(s), []).append(i)
    return alloc.LotGorunumu(
        lot_id=np.array([f"L{i}" for i in range(sku_idx.size)], dtype=object),
        sku_idx=sku_idx, adet=adet.copy(), ham_adet=adet.copy(),
        taban_talebi=np.zeros_like(adet), taban_satir=np.zeros(0),
        taban_satir_lot=np.zeros(0, dtype=np.int32), kalan_gun=kalan_gun,
        birim_maliyet=np.full(sku_idx.size, dsf * (1 - marj)),
        birim_deger=normal, salvage=salvage,
        sku_lotlari={k: np.array(v, dtype=np.int32) for k, v in sku_lotlari.items()},
        lot_sirasi={f"L{i}": i for i in range(sku_idx.size)})


def _kolonlar(satir, lot, kol, kazanc, cekilen):
    n = len(satir)
    return alloc.Kolonlar(
        satir=np.array(satir, dtype=np.int32), lot=np.array(lot, dtype=np.int32),
        kol=np.array(kol, dtype=np.int32), kazanc=np.array(kazanc, dtype=float),
        cekilen=np.array(cekilen, dtype=float),
        nominal=np.array(cekilen, dtype=float), tutar=np.zeros(n),
        temizlik=np.zeros(n, dtype=bool), elenen={})


def _teklifler(eczane_idx, sku_idx):
    n = len(eczane_idx)
    return pl.DataFrame({
        "eczane_idx": np.array(eczane_idx, dtype=np.int32),
        "sku_idx": np.array(sku_idx, dtype=np.int32),
    })


def _coz(cfg, lotlar, teklifler, kolonlar, pol, P):
    return alloc.lp_tahsisi(_SahteDunya(P, int(lotlar.sku_idx.max()) + 1), cfg,
                            _SahteGorunum(0, P), lotlar, teklifler, kolonlar,
                            pol, teklifler.height)


def test_lp_kit_stogu_en_degerli_satira_veriyor(cfg):
    """Iki eczane, tek lot, tek adetlik kapasite: LP daha degerli olani secer.

    Ranking-only ayni lotu IKISINE birden soz verirdi; kit kaynak altinda
    ikisinden biri karsilanamazdi. D5'in tek cumlelik testi budur.
    """
    lotlar = _lot_gorunumu([0], [1.0], [500.0], cfg)
    teklifler = _teklifler([0, 1], [0, 0])
    kolonlar = _kolonlar(satir=[0, 1], lot=[0, 0], kol=[1, 1],
                         kazanc=[10.0, 3.0], cekilen=[1.0, 1.0])
    sonuc = _coz(cfg, lotlar, teklifler, kolonlar, alloc.POLITIKALAR["lp"], P=2)

    assert sonuc.kol[0] == 1 and sonuc.kol[1] == scorer.TEKLIF_YOK
    assert sonuc.lot[0] == 0


def test_golge_fiyat_kitlikta_devam_degerinin_ustune_cikiyor(cfg):
    """Lot tuketildiginde gölge fiyat = devam degeri + kitlik primi.

    Kagit uzerinde: kapasite 1 adet, iki talip var (10 TL ve 3 TL). Bir adet
    daha olsaydi 3 TL daha kazanirdik -> gölge fiyat 3 TL olmali (devam
    degeri 4.25 TL'nin altinda kaldigi icin burada baglayan taraf devam
    degeridir; kitlik primi ancak devam degerini asinca gorunur).
    """
    lotlar = _lot_gorunumu([0], [1.0], [500.0], cfg)
    teklifler = _teklifler([0, 1], [0, 0])
    kolonlar = _kolonlar([0, 1], [0, 0], [1, 1], [100.0, 30.0], [1.0, 1.0])
    sonuc = _coz(cfg, lotlar, teklifler, kolonlar, alloc.POLITIKALAR["lp"], P=2)

    # Lot tamamen tuketildi -> ikinci talibin degeri gölge fiyattir.
    assert sonuc.golge_fiyat[0] == pytest.approx(30.0, rel=1e-6)
    assert sonuc.golge_fiyat[0] > sonuc.lot_devam_degeri[0]


def test_golge_fiyat_artik_varken_devam_degerine_esit(cfg):
    """Lot tuketilmediginde dual TAM OLARAK devam degeridir (LP teorisi)."""
    lotlar = _lot_gorunumu([0], [1000.0], [500.0], cfg)
    teklifler = _teklifler([0], [0])
    kolonlar = _kolonlar([0], [0], [1], [100.0], [1.0])
    sonuc = _coz(cfg, lotlar, teklifler, kolonlar, alloc.POLITIKALAR["lp"], P=1)
    assert sonuc.golge_fiyat[0] == pytest.approx(sonuc.lot_devam_degeri[0], rel=1e-6)


def test_miad_rejiminde_golge_fiyat_negatife_donuyor(cfg):
    """D9'un tek cumlelik testi: AYNI LP, ayni kod yolu, isaret degisimi.

    Ayni lot (kalan_gun miad sonrasi) iki politikaya verilir. `lp` temizlik
    degerini kullanmaz ve gölge fiyat POZITIF kalir; `hedefli_temizlik`
    salvage egrisini kullanir ve gölge fiyat NEGATIFE doner.
    """
    lotlar = _lot_gorunumu([0], [1000.0], [-5.0], cfg)
    teklifler = _teklifler([0], [0])
    kolonlar = _kolonlar([0], [0], [1], [1.0], [1.0])

    normal = _coz(cfg, lotlar, teklifler, kolonlar, alloc.POLITIKALAR["lp"], P=1)
    temizlik = _coz(cfg, lotlar, teklifler, kolonlar,
                    alloc.POLITIKALAR["hedefli_temizlik"], P=1)

    assert normal.golge_fiyat[0] > 0
    assert temizlik.golge_fiyat[0] < 0
    assert temizlik.golge_fiyat[0] == pytest.approx(lotlar.salvage[0], rel=1e-6)


def test_negatif_golge_fiyat_zararina_teklifi_rasyonel_yapiyor(cfg):
    """"Normalde irrasyonel bir MF derinligi burada rasyoneldir" (SPEC 2.5).

    Kazanci NEGATIF olan bir kolon: normal rejimde secilmez. Miad rejiminde
    lotun devam degeri kolonun zararindan daha negatif oldugu icin AYNI kolon
    secilir. Amac fonksiyonu degismedi; lotun degeri degisti.
    """
    lotlar = _lot_gorunumu([0], [10.0], [-5.0], cfg)
    teklifler = _teklifler([0], [0])
    # Kazanc -1 TL; lotun 10 adedi kalirsa adet basina -8 TL (100 x 0.08).
    kolonlar = _kolonlar([0], [0], [1], [-1.0], [10.0])

    normal = _coz(cfg, lotlar, teklifler, kolonlar, alloc.POLITIKALAR["lp"], P=1)
    temizlik = _coz(cfg, lotlar, teklifler, kolonlar,
                    alloc.POLITIKALAR["hedefli_temizlik"], P=1)

    assert normal.kol[0] == scorer.TEKLIF_YOK
    assert temizlik.kol[0] == 1


def test_lp_satir_basina_tek_aksiyon(cfg):
    """Bir aday satiri en fazla bir (lot, kol) alir - SPEC 2.5 lot referansi."""
    lotlar = _lot_gorunumu([0, 0], [100.0, 100.0], [500.0, 400.0], cfg)
    teklifler = _teklifler([0], [0])
    kolonlar = _kolonlar([0, 0, 0], [0, 1, 0], [1, 1, 2], [10.0, 9.0, 8.0],
                         [1.0, 1.0, 1.0])
    sonuc = _coz(cfg, lotlar, teklifler, kolonlar, alloc.POLITIKALAR["lp"], P=1)
    assert int(sonuc.teklif_maskesi.sum()) == 1


def test_lp_frekans_tavanina_uyuyor(cfg):
    """Frekans tavani LP'nin KISITI (D6): optimizasyon onu asamaz."""
    tavan = cfg.politika.kisit.eczane_haftalik_teklif_tavani
    n = tavan + 4
    lotlar = _lot_gorunumu(list(range(n)), [1000.0] * n, [500.0] * n, cfg)
    teklifler = _teklifler([0] * n, list(range(n)))
    kolonlar = _kolonlar(list(range(n)), list(range(n)), [1] * n,
                         list(np.linspace(100.0, 10.0, n)), [1.0] * n)
    sonuc = _coz(cfg, lotlar, teklifler, kolonlar, alloc.POLITIKALAR["lp"], P=1)
    assert int(sonuc.teklif_maskesi.sum()) == tavan


def test_lp_kredi_limiti_bagliyor(cfg):
    """DBS limiti LP'nin icinde de veto yetkisini koruyor (D6)."""
    lotlar = _lot_gorunumu([0, 1], [1000.0, 1000.0], [500.0, 500.0], cfg)
    teklifler = _teklifler([0, 0], [0, 1])
    kolonlar = alloc.Kolonlar(
        satir=np.array([0, 1], dtype=np.int32), lot=np.array([0, 1], dtype=np.int32),
        kol=np.array([1, 1], dtype=np.int32), kazanc=np.array([10.0, 9.0]),
        cekilen=np.array([1.0, 1.0]), nominal=np.array([1.0, 1.0]),
        tutar=np.array([600.0, 600.0]), temizlik=np.zeros(2, dtype=bool), elenen={})

    dunya = _SahteDunya(1, 2)
    dunya.eczaneler = pl.DataFrame({
        # Etkin tavan = 1000 x kullanim_tavani; iki teklif (1200 TL) sigmaz.
        "dbs_limiti": np.array([1000.0 / cfg.politika.kisit.kredi_kullanim_tavani]),
        "vade_riski_skoru": np.zeros(1)})
    sonuc = alloc.lp_tahsisi(dunya, cfg, _SahteGorunum(0, 1), lotlar, teklifler,
                             kolonlar, alloc.POLITIKALAR["lp"], 2)
    assert int(sonuc.teklif_maskesi.sum()) == 1
    assert sonuc.kol[0] == 1                       # daha degerli olan kaldi


def test_lp_tekrar_uretilebilir(cfg):
    """CLAUDE.md 5: iki kez calistirinca ayni sonuc."""
    lotlar = _lot_gorunumu([0] * 3, [5.0, 5.0, 5.0], [500.0, 300.0, 200.0], cfg)
    teklifler = _teklifler([0, 1, 2, 3], [0, 0, 0, 0])
    kolonlar = _kolonlar([0, 1, 2, 3], [0, 1, 2, 0], [1, 1, 1, 2],
                         [10.0, 10.0, 10.0, 10.0], [5.0, 5.0, 5.0, 5.0])
    a = _coz(cfg, lotlar, teklifler, kolonlar, alloc.POLITIKALAR["lp"], P=4)
    b = _coz(cfg, lotlar, teklifler, kolonlar, alloc.POLITIKALAR["lp"], P=4)
    assert np.array_equal(a.kol, b.kol) and np.array_equal(a.lot, b.lot)
    assert np.allclose(a.golge_fiyat, b.golge_fiyat)


def test_acgozlu_stok_ayirmiyor_lp_ayiriyor(cfg):
    """(a) karsilastirmasinin ta kendisi: ayni girdi, iki farkli davranis."""
    lotlar = _lot_gorunumu([0], [1.0], [500.0], cfg)
    teklifler = _teklifler([0, 1], [0, 0])
    kolonlar = _kolonlar([0, 1], [0, 0], [1, 1], [10.0, 9.0], [1.0, 1.0])
    dunya, gor = _SahteDunya(2, 1), _SahteGorunum(0, 2)

    acgozlu = alloc.acgozlu_tahsis(dunya, cfg, gor, lotlar, teklifler, kolonlar,
                                   kolonlar.kazanc,
                                   alloc.POLITIKALAR["ranking_only"], 2)
    lp = _coz(cfg, lotlar, teklifler, kolonlar, alloc.POLITIKALAR["lp"], P=2)

    # Ranking-only 1 adetlik lotu IKI eczaneye birden soz veriyor.
    assert int(acgozlu.teklif_maskesi.sum()) == 2
    assert int(lp.teklif_maskesi.sum()) == 1


# --------------------------------------------------------------------------
# DEPO IMHASI (olcum tarafi)
# --------------------------------------------------------------------------
def test_depo_imhasi_fefo_kumulatifi(cfg):
    """Onde duran (erken miatli) lot once satilir; kapasite ondan artan kadar.

    Haftalik hiz 10, iki lot: 70 gun (=10 hafta -> 100 adet kapasite) 60 adet,
    140 gun (=20 hafta -> 200 adet kapasite) 200 adet. Ilk lot tamamen satilir;
    ikinci lot icin kalan kapasite 200-60 = 140 -> 60 adet imha.
    """
    lotlar = _lot_gorunumu([0, 0], [60.0, 200.0], [70.0, 140.0], cfg)
    imha = ev.depo_imhasi(lotlar, np.array([60.0, 200.0]), np.array([10.0]))
    assert imha[0] == pytest.approx(0.0)
    assert imha[1] == pytest.approx(60.0)


def test_depo_imhasi_miadi_gecmis_lotu_tamamen_yaziyor(cfg):
    lotlar = _lot_gorunumu([0], [40.0], [-3.0], cfg)
    imha = ev.depo_imhasi(lotlar, np.array([40.0]), np.array([100.0]))
    assert imha[0] == pytest.approx(40.0)


def test_organik_cikis_hizi_gelecegi_gormuyor(cfg):
    """Point-in-time: t sonrasi sevkiyat projeksiyona giremez."""
    class _D:
        S = 2
        sevk_w = np.array([1, 5, 9, 12])
        sevk_s = np.array([0, 0, 1, 0])
        sevk_adet = np.array([10.0, 10.0, 10.0, 999.0])
    hiz = ev.organik_sevk_hizi(_D(), cfg, t=10)
    pencere = min(cfg.tahsis.degerlendirme.organik_cikis_pencere_hafta, 11)
    assert hiz[0] == pytest.approx(20.0 / pencere)     # 12. hafta DAHIL DEGIL
