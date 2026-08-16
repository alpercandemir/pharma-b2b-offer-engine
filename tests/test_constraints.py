"""Kisit katmaninin degismezleri (D6). PROMPTS.md M3'un istedigi dort test:

  1. Kirmizi ve yesil receteli urun hicbir kosulda oneri listesinde cikmiyor
  2. Soguk zincir min siparis kurali ihlal edilmiyor
  3. Kredi limiti asan teklif uretilmiyor
  4. Miad baskisi promosyon vetosunu ASMIYOR (SPEC 2.5)

VAKUM TESTI TEHLIKESI. Bu dunyada kirmizi/yesil urun katalogun %1'inden azi
(fast: 100 SKU'da 2, full: 300 SKU'da 1) ve varsayilan ayarda kredi limiti hic
baglamiyor. Bu ayarlarla "listede kirmizi yok" demek hicbir sey kanitlamaz.
Bu yuzden her test iki parcalidir:

  (a) STRES: kisit BAGLAYACAK sekilde kurulur (en yuksek skorlu urunler
      kirmizi ilan edilir, kredi tavani daraltilir) ve veto'nun fiilen
      ATESLENDIGI ayrica dogrulanir;
  (b) DEGISMEZ: kisit ihlali sifir.

(a) olmadan (b) her zaman gecer ve testin degeri olmaz.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.config import load_config
from core.io import Run
from features.okuma import GozlemlenebilirKaynak
from policy import candidates as ad
from policy.constraints import VETO_SEBEPLERI, kisit_uygula, oneri_listesi
from pydantic import ValidationError
from scripts.generate_world import dunya_yaz

VETO_RENKLERI = ("KIRMIZI", "YESIL")


@pytest.fixture(scope="module")
def dunya_kok(tmp_path_factory):
    cfg = load_config("fast")
    kok = tmp_path_factory.mktemp("dunya")
    dunya_yaz(cfg, Run("t", kok=kok))
    return kok


def _kur(kok, gecersiz: dict | None = None):
    cfg = load_config("fast", gecersiz_kilma=gecersiz or {})
    dunya = ad.dunya_yukle(GozlemlenebilirKaynak("t", kok=kok), cfg)
    t = ad.origin_haftalari(cfg, dunya.W)[-1]
    gor = ad.gorunum_kur(dunya, cfg, t)
    havuz, _ = ad.aday_havuzu(dunya, cfg, gor)
    return cfg, dunya, gor, havuz


def _renklendir(dunya: ad.AdayDunyasi, sku_idler, renk: str) -> None:
    """Verilen SKU'lari kontrollu dagitim urunu ilan eder (urun master'i).

    Dunyayi yeniden uretmeden kisit katmanini strese sokmanin yolu bu:
    aday uretimi renkten habersiz oldugu icin havuz AYNI kalir, degisen tek
    sey kisit katmaninin gordugu regulasyon bayragidir.
    """
    hedef = pl.col("sku_id").is_in(list(sku_idler))
    dunya.urunler = dunya.urunler.with_columns([
        pl.when(hedef).then(pl.lit(renk)).otherwise(pl.col("recete_rengi"))
          .alias("recete_rengi"),
        pl.when(hedef).then(pl.lit(False)).otherwise(pl.col("promosyon_serbest"))
          .alias("promosyon_serbest"),
    ])


def _renkli_liste(dunya, teklifler) -> pl.DataFrame:
    liste = oneri_listesi(teklifler)
    return liste.join(dunya.urunler.select(["sku_id", "recete_rengi"]),
                      on="sku_id", how="left").filter(
        pl.col("recete_rengi").is_in(list(VETO_RENKLERI)))


# --------------------------------------------------------------------------
# 1. kirmizi / yesil recete
# --------------------------------------------------------------------------
def test_kirmizi_yesil_hicbir_kosulda_listede_yok(dunya_kok):
    """En yuksek skorlu urunler kirmizi/yesil ilan edilir; hicbiri listeye
    girmemeli ve veto fiilen atesle(n)meli."""
    cfg, dunya, gor, havuz = _kur(dunya_kok)
    tepe = (havuz.sort("skor", descending=True)["sku_id"].unique(maintain_order=True)
            .head(10).to_list())
    _renklendir(dunya, tepe[:5], "KIRMIZI")
    _renklendir(dunya, tepe[5:], "YESIL")

    teklifler = kisit_uygula(dunya, cfg, gor, havuz)
    vetolanan = teklifler.filter(pl.col("veto_sebebi") == "recete_rengi")
    assert vetolanan.height > 0, "stres kurulumu baglamadi - test vakum"
    assert _renkli_liste(dunya, teklifler).height == 0
    # MF kanali da kapali olmali (kanal kisiti veto ile tutarli).
    assert not vetolanan["mf_izinli"].any()


def test_regulasyonu_gevseten_config_reddediliyor():
    """D6 mekanik kilidi: veto listesinden kirmizi cikarilirsa config yuklenmez.

    "Hicbir kosulda" iddiasi boylece kod incelemesine degil, sema
    dogrulamasina baglanir (core/config.py Config._capraz_kontrol).
    """
    with pytest.raises(ValidationError):
        load_config("fast", gecersiz_kilma={
            "politika.kisit.recete_rengi_vetosu": ["YESIL"]})
    with pytest.raises(ValidationError):
        load_config("fast", gecersiz_kilma={"politika.kisit.recete_rengi_vetosu": []})


# --------------------------------------------------------------------------
# 2. soguk zincir minimum siparis
# --------------------------------------------------------------------------
def test_soguk_zincir_min_siparis_ihlal_edilmiyor(dunya_kok):
    """Frekans tavani gevsetilir: kurali sinamak istiyoruz, budamayi degil.

    Varsayilan tavan (5) soguk zincir satirlarinin cogunu listeden zaten
    cikariyor ve test "listede soguk zincir yok" diye vakuma dusuyor.
    """
    cfg, dunya, gor, havuz = _kur(
        dunya_kok, {"politika.kisit.eczane_haftalik_teklif_tavani": 50})
    teklifler = kisit_uygula(dunya, cfg, gor, havuz)
    liste = oneri_listesi(teklifler)
    min_adet = cfg.politika.kisit.soguk_zincir_min_siparis_adedi

    soguk = liste.filter(pl.col("soguk_zincir"))
    assert soguk.height > 0, "listede hic soguk zincir urunu yok - test vakum"
    assert soguk["teklif_adedi"].min() >= min_adet
    assert teklifler["soguk_zincir_yukseltildi"].sum() > 0, "yukseltme hic calismadi"
    # Yukseltme emilim tavanini delmiyor: adet minimuma cikarilinca eczanenin
    # emebileceginden fazlasi olan satirlar VETOLANIYOR, gizlice gecmiyor.
    yukseltilip_vetolanan = teklifler.filter(
        pl.col("soguk_zincir_yukseltildi") & (pl.col("veto_sebebi") == "emilim_tavani"))
    assert yukseltilip_vetolanan.height > 0


def test_soguk_zincir_min_altinda_veto_secenegi(dunya_kok):
    """`veto` modunda minimumun altindaki soguk zincir satiri LISTEDEN CIKAR,
    adet yukseltilerek gizlenmez."""
    cfg, dunya, gor, havuz = _kur(
        dunya_kok, {"politika.kisit.soguk_zincir_min_altinda": "veto"})
    teklifler = kisit_uygula(dunya, cfg, gor, havuz)
    assert (teklifler["veto_sebebi"] == "soguk_zincir_min").sum() > 0
    liste = oneri_listesi(teklifler)
    min_adet = cfg.politika.kisit.soguk_zincir_min_siparis_adedi
    assert liste.filter(pl.col("soguk_zincir")
                        & (pl.col("teklif_adedi") < min_adet)).height == 0


# --------------------------------------------------------------------------
# 3. kredi limiti
# --------------------------------------------------------------------------
def test_kredi_limitini_asan_teklif_uretilmiyor(dunya_kok):
    """Kisit PORTFOY duzeyinde: tek tek limitin altinda kalan teklifler
    toplamda limiti asamaz.

    Varsayilan ayarda DBS limiti hic baglamiyor (olculdu: veto orani 0), bu
    yuzden tavan daraltilarak baglayici hale getirilir.
    """
    cfg, dunya, gor, havuz = _kur(dunya_kok, {
        "politika.kisit.kredi_kullanim_tavani": 0.02,
        "politika.kisit.eczane_haftalik_teklif_tavani": 50,
    })
    teklifler = kisit_uygula(dunya, cfg, gor, havuz)
    assert (teklifler["veto_sebebi"] == "kredi_limiti").sum() > 0, "test vakum"

    k = cfg.politika.kisit
    dbs = dunya.eczaneler["dbs_limiti"].to_numpy().astype(float)
    risk = dunya.eczaneler["vade_riski_skoru"].to_numpy().astype(float)
    kalan = np.maximum(
        dbs * k.kredi_kullanim_tavani * (1.0 - risk * k.vade_riski_cezasi)
        - gor.acik_bakiye, 0.0)

    liste = oneri_listesi(teklifler)
    yuk = np.zeros(dunya.P)
    np.add.at(yuk, liste["eczane_idx"].to_numpy(), liste["teklif_tutari"].to_numpy())
    assert (yuk <= kalan + 1e-6).all(), "eczane bazinda kredi limiti asildi"


def test_kredi_vetosu_atlar_durmaz(dunya_kok):
    """Limite sigmayan buyuk teklif vetolanir ama arkasindaki kucuk teklifler
    degerlendirilmeye devam eder (greedy 'atla', 'dur' degil)."""
    cfg, dunya, gor, havuz = _kur(dunya_kok, {
        "politika.kisit.kredi_kullanim_tavani": 0.02,
        "politika.kisit.eczane_haftalik_teklif_tavani": 50,
    })
    teklifler = kisit_uygula(dunya, cfg, gor, havuz)
    atlanan = 0
    for _, grup in teklifler.filter(~pl.col("vetolu") | (pl.col("veto_sebebi")
                                                         == "kredi_limiti")
                                    ).sort("skor", descending=True).group_by(
            "eczane_idx", maintain_order=True):
        vetolu = grup["veto_sebebi"].to_numpy() == "kredi_limiti"
        if vetolu.any() and (~vetolu[np.argmax(vetolu):]).any():
            atlanan += 1
    assert atlanan > 0, "kredi vetosu sonrasi hic teklif gecmemis - 'dur' davranisi"


# --------------------------------------------------------------------------
# 4. miad baskisi vetoyu asmiyor (SPEC 2.5)
# --------------------------------------------------------------------------
def test_miad_baskisi_promosyon_vetosunu_asmiyor(dunya_kok):
    """SPEC 2.5: "promosyon_serbest = false olan urunlerde temizlik kampanyasi
    yapilamaz - miad baskisi bu vetoyu asmaz."

    Kurulum kasitli olarak en agir hali: her SKU'nun depo stogu tamamen kisa
    miatli ilan edilir (baski = 1.0) ve baski agirligi 1000'e cikarilir. Bu
    kombinasyonda temizlik guduzu ne kadar buyurse buyusun kirmizi/yesil urun
    listeye giremez.
    """
    cfg, dunya, gor, havuz = _kur(dunya_kok, {
        "politika.aday.miad_baskisi_agirligi": 1000.0,
        "politika.aday.miad_baskisi_esik_gun": 100000.0,     # her lot "baski altinda"
    })
    # Stogu olan her SKU tamamen kisa miatli sayiliyor. (Stogu olmayan SKU'da
    # baski tanimsiz; oran sifir kalir.)
    assert float(gor.miad_baskisi[gor.depo_stok > 0].min()) == 1.0, "kurulum tutmadi"

    tepe = (havuz.sort("skor", descending=True)["sku_id"].unique(maintain_order=True)
            .head(8).to_list())
    _renklendir(dunya, tepe, "KIRMIZI")
    teklifler = kisit_uygula(dunya, cfg, gor, havuz)

    assert (teklifler["veto_sebebi"] == "recete_rengi").sum() > 0, "test vakum"
    assert _renkli_liste(dunya, teklifler).height == 0
    # Baski SIRALAMAYI gercekten oynatiyor olmali; oynatmiyorsa test, gucsuz
    # bir carpani sinamis olurdu.
    _, _, sakin_gor, sakin_havuz = _kur(dunya_kok)
    assert not sakin_havuz["sku_id"].to_list() == havuz["sku_id"].to_list()


def test_miad_baskisi_raf_omru_vetosunu_da_asmiyor(dunya_kok):
    """Baski ne kadar buyurse buyusun, asgari raf omrunun altindaki lot
    teklif edilemez: temizlik zarari eczaneye TRANSFER etmek degildir."""
    cfg, dunya, gor, havuz = _kur(dunya_kok, {
        "politika.aday.miad_baskisi_agirligi": 1000.0,
        "politika.kisit.asgari_kalan_raf_omru_gun": 400.0,
    })
    teklifler = kisit_uygula(dunya, cfg, gor, havuz)
    liste = oneri_listesi(teklifler)
    assert (teklifler["veto_sebebi"] == "raf_omru").sum() > 0, "test vakum"
    assert liste.height > 0
    assert liste["lot_kalan_gun"].min() >= 400.0
    assert liste["lot_id"].null_count() == 0


# --------------------------------------------------------------------------
# butunluk
# --------------------------------------------------------------------------
def test_veto_maskesi_ve_sebep_tutarli(dunya_kok):
    """`veto_sebebi` ilk baglayan sebeptir; `veto_maskesi` hepsini tasir."""
    cfg, dunya, gor, havuz = _kur(dunya_kok)
    teklifler = kisit_uygula(dunya, cfg, gor, havuz)
    maske = teklifler["veto_maskesi"].to_numpy()
    vetolu = teklifler["vetolu"].to_numpy()
    sebep = teklifler["veto_sebebi"].to_numpy()
    assert ((maske > 0) == vetolu).all()
    assert (sebep[~vetolu] == "").all()
    for i in np.flatnonzero(vetolu):
        bitler = [ad_ for bit, ad_ in enumerate(VETO_SEBEPLERI)
                  if (maske[i] >> bit) & 1]
        assert sebep[i] == bitler[0]


def test_liste_veto_edilmis_satir_icermiyor(dunya_kok):
    cfg, dunya, gor, havuz = _kur(dunya_kok)
    teklifler = kisit_uygula(dunya, cfg, gor, havuz)
    assert not oneri_listesi(teklifler)["vetolu"].any()
    tavan = cfg.politika.kisit.eczane_haftalik_teklif_tavani
    assert oneri_listesi(teklifler).group_by("eczane_idx").len()["len"].max() <= tavan


def test_havuz_kisittan_habersiz(dunya_kok):
    """Aday uretimi kisit knob'larindan etkilenmemeli (D6 ayrimi).

    Etkilenseydi vetonun bedeli olculemezdi: havuz zaten kisiti icsellestirmis
    olurdu ve "kisit katmani ne kadar recall'a mal oluyor" sorusu anlamsizlasirdi.
    """
    _, _, _, temel = _kur(dunya_kok)
    _, _, _, degisik = _kur(dunya_kok, {
        "politika.kisit.asgari_kalan_raf_omru_gun": 500.0,
        "politika.kisit.kredi_kullanim_tavani": 0.01,
        "politika.kisit.eczane_haftalik_teklif_tavani": 1,
    })
    assert temel.equals(degisik)
