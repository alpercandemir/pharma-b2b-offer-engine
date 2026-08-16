"""M7 ajan katmani testleri (D8): araclar, olgu paketi, kayit/oynatma.

Bu dosyanin baglami ELLE kurulur (`baglam()`). Sebep M5'in LP testleriyle
ayni: dogru cevabi kagit uzerinde bilmek. Gercek dunyadan gelen bir baglamda
"arac dogru sayiyi dondurdu mu" sorusu ancak "makul goruniyor" duzeyinde
cevaplanabilirdi.

Uc grup:

  ARAC     : salt okur olma sozu, var olmayan kimlikte ACIK hata, sayi
             defterinin rejim etiketlemesi.
  BRIFING  : olgu paketinin kirpilmasi, bicim sozlesmesi, ve sablonun
             yazdigi HER sayinin deftere dayanmasi.
  KAYIT    : oynatmanin determinizmi ve bayat kaydin sessiz gecmemesi.
"""

from __future__ import annotations

import json

import pytest

from agent import client as ac
from agent import scenario as sc
from agent import narrative as nv
from agent import tools as at
from core.config import config_yukle

PROFIL = "fast"


@pytest.fixture(scope="module")
def cfg():
    return config_yukle(PROFIL)


def _teklif(rejim: str, sku: str, lot: str, **ustune) -> dict:
    satir = {
        "rejim": rejim, "eczane_id": "ECZ0000", "sku_id": sku, "lot_id": lot,
        "mf_orani": 0.1, "mf_ifadesi": "20+2", "vade_gun": 90, "adet": 20,
        "bedava_adet": 2, "kabul_olasiligi": 0.42,
        "teklifsiz_kabul_olasiligi": 0.31, "kabul_sartiyla_brut_marj_tl": 512.5,
        "artimsal_beklenen_marj_tl": 44.25, "lot_kalan_gun": 480.0,
        "lot_bekleyebilir": True, "erteleme_tl_adet": 0.0,
        "haftalik_hiz_tahmini": 3.5, "mf_kanali_acik": True,
        "soguk_zincir": False, "sgk_geri_odeme": False,
    }
    return satir | ustune


@pytest.fixture()
def baglam() -> at.AjanBaglami:
    """Iki rejimli, tek eczaneli, elle kurulmus baglam."""
    rejimler = ["baz", "sok"]
    par = {ad: {"rejim": ad, "aciklama": f"{ad} rejimi",
                "guncelleme_beklentisi_hafta": 26.0 if ad == "baz" else 2.0,
                "referans_kur_artisi": 0.0 if ad == "baz" else 0.3,
                "fiyat_gecis_katsayisi": 0.0 if ad == "baz" else 0.85,
                "antisipasyon_talep_carpani": 1.0 if ad == "baz" else 2.2,
                "fonlama_orani_carpani": 1.0 if ad == "baz" else 1.6,
                "taban_mi": ad == "baz"} for ad in rejimler}
    # Ozet sozlugu URETIMDEKI ile ayni semadan kurulur (`baglam_kur`).
    # Elle yazilsaydi `RejimOzeti`ye yeni bir alan eklendiginde test sessizce
    # eski semayi sinamaya devam ederdi.
    ozet = {}
    for i, ad in enumerate(rejimler):
        o = sc.RejimOzeti(
            ad=ad, aday_satiri=40, teklif_sayisi=11 if i == 0 else 7,
            vetolu_satir=2,
            beklenen_artimsal_marj=1234.5 if i == 0 else 789.25,
            beklenen_brut_marj=9000.0 + i, teklif_adedi=220 + i,
            bedava_adet=22 + i, ortalama_mf=0.1 - 0.02 * i,
            ortalama_vade=90.0 - i, mf_teklif_sayisi=6 - i,
            bekleyemeyen_pay=0.2 + 0.05 * i,
            bekleyemeyen_teklif_pay=0.15 + 0.05 * i,
            ortalama_erteleme_tl=0.0 if i == 0 else 30.6,
            negatif_taban_marj_orani=0.0 if i == 0 else 0.9,
            talep_baskilayan_teklif_orani=0.0 if i == 0 else 0.8,
            veto_dagilimi={"recete_rengi": 2})
        ozet[ad] = {"rejim": ad} | {k: v for k, v in vars(o).items()
                                    if k not in ("ad", "veto_dagilimi")}
        ozet[ad]["veto_dagilimi"] = dict(o.veto_dagilimi)
    return at.AjanBaglami(
        origin=99, politika="uplift_x", taban_rejim="baz",
        rejim_parametreleri=par, rejim_ozetleri=ozet,
        rejim_farklari={"sok": {
            k: v for k, v in vars(sc.RejimFarki(
                ad="sok", kol_degisen_satir=5, teklife_giren=2,
                teklifden_cikan=6, teklif_sayisi_farki=-4,
                artimsal_marj_farki=-445.25, ortalama_mf_farki=-0.02,
                ortalama_vade_farki=-1.0, bedava_adet_farki=-1,
                bekleyemeyen_teklif_pay_farki=0.05, veto_farki=-3)).items()
            if k != "ad"}},
        eczaneler={"ECZ0000": {"eczane_id": "ECZ0000", "il": "Konya",
                               "ilce": "Konya-ILCE1", "aylik_recete_adedi": 2629,
                               "hastane_yakinligi_km": 2.03,
                               "semt_sosyoekonomik_index": 0.51,
                               "turizm_bolgesi": False, "aylik_ciro_bandi": "XL",
                               "vade_riski_skoru": 0.4485,
                               "dbs_limiti_tl": 2113390.27,
                               "acik_bakiye_tl": 97366.23,
                               "sgk_recete_orani": 0.9217,
                               "haftalik_teklif_tavani": 5}},
        urunler={"SKU0001": {"sku_id": "SKU0001", "kategori_kod": "J01",
                             "urun_tipi": "RX", "recete_rengi": "NORMAL",
                             "sgk_geri_odeme": False, "dsf_tl": 120.0,
                             "koli_ici_adet": 10, "depo_kar_marji": 0.06,
                             "soguk_zincir": False, "promosyon_serbest": True,
                             "titck_tedarik_guclugu": False, "atc_kodu": "J01CA04",
                             "etken_madde": "J01-INN-01"},
                 "SKU0002": {"sku_id": "SKU0002", "kategori_kod": "N02",
                             "urun_tipi": "RX", "recete_rengi": "KIRMIZI",
                             "sgk_geri_odeme": True, "dsf_tl": 80.0,
                             "koli_ici_adet": 5, "depo_kar_marji": 0.08,
                             "soguk_zincir": False, "promosyon_serbest": False,
                             "titck_tedarik_guclugu": False, "atc_kodu": "N02AA01",
                             "etken_madde": "N02-INN-01"},
                 # Ne onerilen ne vetolanan urun: "politikanin uretmedigi
                 # oneri" mutasyonunun hedefi (tests/test_harness.py).
                 "SKU0003": {"sku_id": "SKU0003", "kategori_kod": "R05",
                             "urun_tipi": "OTC", "recete_rengi": "NORMAL",
                             "sgk_geri_odeme": False, "dsf_tl": 45.0,
                             "koli_ici_adet": 12, "depo_kar_marji": 0.09,
                             "soguk_zincir": False, "promosyon_serbest": True,
                             "titck_tedarik_guclugu": False, "atc_kodu": "R05CB02",
                             "etken_madde": "R05-INN-01"}},
        lotlar={"LOT000001": {"lot_id": "LOT000001", "sku_id": "SKU0001",
                              "kalan_adet": 900, "kalan_gun": 480.0},
                "LOT000002": {"lot_id": "LOT000002", "sku_id": "SKU0001",
                              "kalan_adet": 300, "kalan_gun": 150.0}},
        teklifler={
            ("baz", "ECZ0000"): [_teklif("baz", "SKU0001", "LOT000001")],
            ("sok", "ECZ0000"): [_teklif("sok", "SKU0001", "LOT000001",
                                         adet=44, bedava_adet=4,
                                         mf_ifadesi="44+4",
                                         artimsal_beklenen_marj_tl=12.75,
                                         erteleme_tl_adet=30.6)],
        },
        vetolar={
            ("baz", "ECZ0000"): [{"rejim": "baz", "eczane_id": "ECZ0000",
                                  "sku_id": "SKU0002",
                                  "veto_sebebi": "recete_rengi",
                                  "dayanak": at.VETO_ACIKLAMASI["recete_rengi"],
                                  "skor": 0.77, "istenen_adet": 12}],
        },
        kol_ekonomisi={
            ("baz", "ECZ0000", "SKU0001"): [
                {"kol": "teklif_yok", "mf_orani": 0.0, "vade_gun": 60,
                 "adet": 20, "bedava_adet": 0, "kabul_olasiligi": 0.31,
                 "kabul_sartiyla_brut_marj_tl": 512.5,
                 "artimsal_beklenen_marj_tl": 0.0, "izinli": True,
                 "secilen": False}],
        })


# --------------------------------------------------------------------------
# ARAC
# --------------------------------------------------------------------------
def test_arac_semalari_govdelerle_ortusuyor():
    """Semada olup govdesi olmayan (ya da tersi) arac sessizce olmesin."""
    assert set(at.ARAC_ADLARI) == set(at.GOVDELER)


def test_teklif_listesi_hazir_tabloyu_okuyor(baglam):
    sonuc = at.teklif_listesi(baglam, "ECZ0000", "sok")
    assert sonuc["teklif_sayisi"] == 1
    assert sonuc["teklifler"][0]["adet"] == 44
    assert sonuc["teklifler"][0]["rejim"] == "sok"


def test_var_olmayan_kimlik_acik_hata_veriyor(baglam):
    """Sessiz bos liste, modeli "demek ki yok" diye uydurmaya davet ederdi."""
    assert "hata" in at.eczane_profili(baglam, "ECZ9999")
    assert "hata" in at.lot_bilgisi(baglam, "LOT999999")
    assert "hata" in at.teklif_listesi(baglam, "ECZ0000", "olmayan_rejim")


def test_kol_ekonomisi_teklifsiz_satirda_tanimsiz(baglam):
    sonuc = at.kol_ekonomisi(baglam, "ECZ0000", "SKU0002", "baz")
    assert "hata" in sonuc


def test_defter_rejim_etiketliyor(baglam):
    """`sok` cagrisinin sayilari `baz` defterine yazilmamali."""
    defter = at.SayiDefteri()
    at.cagir(baglam, "teklif_listesi", {"eczane_id": "ECZ0000", "rejim": "sok"},
             defter)
    assert 44.0 in defter.kume("sok")
    assert 44.0 not in defter.kume("baz")
    assert 44.0 not in defter.kume(at.GENEL)


def test_defter_metin_icindeki_sayilari_da_topluyor(baglam):
    """"44+4" gibi saha ifadeleri brifingde aynen kullanilabilmeli."""
    defter = at.SayiDefteri()
    defter.ekle("sok", {"mf_ifadesi": "44+4"})
    assert {44.0, 4.0} <= defter.kume("sok")


def test_defter_izinli_kumesi_genel_ile_birlesiyor(baglam):
    defter = at.SayiDefteri()
    defter.ekle(at.GENEL, 99)
    defter.ekle("sok", 44)
    assert {99.0, 44.0} <= defter.izinli("sok")
    assert 44.0 not in defter.izinli("baz")


def test_bilinmeyen_arac_hata_donduruyor(baglam):
    defter = at.SayiDefteri()
    assert "hata" in at.cagir(baglam, "olmayan_arac", {}, defter)


def test_sema_disi_parametre_yoksayiliyor(baglam):
    """Model uydurma parametre gonderirse arac cokmemeli."""
    defter = at.SayiDefteri()
    sonuc = at.cagir(baglam, "eczane_profili",
                     {"eczane_id": "ECZ0000", "uydurma": 1}, defter)
    assert sonuc["eczane_id"] == "ECZ0000"


# --------------------------------------------------------------------------
# BRIFING
# --------------------------------------------------------------------------
def test_brifing_kirpma_knoblarina_uyuyor(cfg, baglam):
    brifing = nv.brifing_kur(cfg, baglam, "ECZ0000")
    for ad in brifing.rejim_adlari:
        assert len(brifing.teklifler[ad]) <= cfg.ajan.brifing_teklif_sayisi
        assert len(brifing.vetolar[ad]) <= cfg.ajan.brifing_veto_sayisi


def test_brifing_butun_rejimleri_tasiyor(cfg, baglam):
    """D3: tek rejim one cikarilamaz."""
    metin = nv.sablon_metni(nv.brifing_kur(cfg, baglam, "ECZ0000"))
    for ad in baglam.rejimler:
        assert f"{nv.REJIM_BASLIGI}{ad}" in metin


def test_sablon_bicim_sozlesmesine_uyuyor(cfg, baglam):
    metin = nv.sablon_metni(nv.brifing_kur(cfg, baglam, "ECZ0000"))
    assert metin.startswith(nv.BASLIK_ONEKI)
    for baslik in (nv.ECZANE_BASLIGI, nv.KISIT_BASLIGI, nv.ONERI_BASLIGI):
        assert baslik in metin
    assert f"```{nv.ONERI_BLOGU}" in metin


def test_olmayan_eczane_hata_veriyor(cfg, baglam):
    with pytest.raises(KeyError):
        nv.brifing_kur(cfg, baglam, "ECZ9999")


def test_sablon_deterministik(cfg, baglam):
    brifing = nv.brifing_kur(cfg, baglam, "ECZ0000")
    assert nv.sablon_metni(brifing) == nv.sablon_metni(brifing)


def test_sayi_bicimi_binlik_ayraci_kullanmiyor():
    """Binlik ayraci belirsizlik yaratir ve denetciyi gevsetirdi."""
    assert nv.sayi_bicimle(2113390.27) == "2113390,27"
    assert nv.sayi_bicimle(2629) == "2629"
    assert nv.sayi_bicimle(-1453.008) == "-1453,01"


# --------------------------------------------------------------------------
# KAYIT
# --------------------------------------------------------------------------
def _kayit_yaz(yol, sistem, kullanici, metin):
    return ac.kayit_yaz(yol, istem={"sistem": sistem, "kullanici": kullanici},
                        turlar=[{"metin": metin, "araclar": [],
                                 "durdurma_sebebi": "end_turn"}])


def test_kayitli_oynatma_metni_geri_veriyor(cfg, baglam, tmp_path):
    brifing = nv.brifing_kur(cfg, baglam, "ECZ0000")
    metin = nv.sablon_metni(brifing)
    yol = _kayit_yaz(tmp_path / "k.json", nv.SISTEM_ISTEMI,
                     nv.kullanici_istemi(brifing), metin)
    cikti = nv.brifing_uret(cfg, baglam, "ECZ0000", ac.KayitliIstemci(yol))
    assert cikti.metin == metin and cikti.kaynak == "llm"


def test_bayat_kayit_sessiz_gecmiyor(cfg, baglam, tmp_path):
    """Istem degistiyse eski cevabi yeni soruya karsi test etmek anlamsiz."""
    yol = _kayit_yaz(tmp_path / "k.json", "eski sistem istemi", "eski soru", "x")
    with pytest.raises(ac.IstemciHatasi, match="ayni degil"):
        nv.brifing_uret(cfg, baglam, "ECZ0000", ac.KayitliIstemci(yol))


def test_kayit_surumu_denetleniyor(tmp_path):
    yol = tmp_path / "k.json"
    yol.write_text(json.dumps({"surum": 999, "turlar": []}), encoding="utf-8")
    with pytest.raises(ac.IstemciHatasi, match="surum"):
        ac.KayitliIstemci(yol)


def test_arac_cagrilari_oynatmada_YENIDEN_hesaplaniyor(cfg, baglam, tmp_path):
    """Arac sonucu kayittan okunsaydi bozuk kayit kendini onaylardi.

    Kayitta yalnizca "hangi arac hangi girdiyle" duruyor; sonuc o an
    uretiliyor ve defter GERCEK arac ciktisindan kuruluyor.
    """
    brifing = nv.brifing_kur(cfg, baglam, "ECZ0000")
    yol = ac.kayit_yaz(
        tmp_path / "k.json",
        istem={"sistem": nv.SISTEM_ISTEMI,
               "kullanici": nv.kullanici_istemi(brifing)},
        turlar=[{"metin": "", "durdurma_sebebi": "tool_use",
                 "araclar": [{"kimlik": "a1", "ad": "teklif_listesi",
                              "girdi": {"eczane_id": "ECZ0000", "rejim": "sok"}}]},
                {"metin": "son", "durdurma_sebebi": "end_turn", "araclar": []}])
    cikti = nv.brifing_uret(cfg, baglam, "ECZ0000", ac.KayitliIstemci(yol))
    assert cikti.tur_sayisi == 2
    assert cikti.arac_cagrilari == [{"ad": "teklif_listesi",
                                     "girdi": {"eczane_id": "ECZ0000",
                                               "rejim": "sok"}}]
    # 44 yalnizca `sok` teklif satirinda var: defter araclardan beslenmis.
    assert 44.0 in cikti.defter.kume("sok")


def test_kayit_tukenirse_hata(cfg, baglam, tmp_path):
    brifing = nv.brifing_kur(cfg, baglam, "ECZ0000")
    yol = ac.kayit_yaz(
        tmp_path / "k.json",
        istem={"sistem": nv.SISTEM_ISTEMI,
               "kullanici": nv.kullanici_istemi(brifing)},
        turlar=[{"metin": "", "durdurma_sebebi": "tool_use",
                 "araclar": [{"kimlik": "a1", "ad": "eczane_profili",
                              "girdi": {"eczane_id": "ECZ0000"}}]}])
    with pytest.raises(ac.IstemciHatasi, match="kayit tukendi"):
        nv.brifing_uret(cfg, baglam, "ECZ0000", ac.KayitliIstemci(yol))


def test_kayit_yoksa_acik_hata(tmp_path):
    with pytest.raises(ac.IstemciHatasi, match="kayit bulunamadi"):
        ac.KayitliIstemci(tmp_path / "yok.json")
