"""M7 harness testleri: denetciler ve mutasyonlar.

SPEC M7'nin cikis kriteri bu dosyada iki ayri duzeyde sinaniyor:

  BIRIM  : denetcilerin kendisi. Bozuk metin ELLE yaziliyor ve hangi
           bulguyu uretmesi gerektigi kagit uzerinde biliniyor.
  UCTAN UCA : gercek dunya, gercek senaryo kosusu, gercek brifing. Temiz
           cikti SIFIR bulgu vermeli, mutantlarin HEPSI yakalanmali.

Ikisi de gerekli. Birim testleri denetcinin mantigini, uctan uca test
gercek olgu paketiyle calistigini gosterir; birincisi ikincisinin
regresyonunu okunur kilar, ikincisi birincisinin gercekle bagini kurar.

SAYI OKUMA kurallari ayrica sinaniyor: "0,38" belirtecinin Ingilizce
okunusla 38 sayilmasi ilk uygulamada gercek bir yanlis siniflandirmaya yol
acti (reports/m7.md 6.2) ve regresyonu burada tutuluyor.
"""

from __future__ import annotations

import pytest

from agent import narrative as nv
from agent import tools as at
from core.config import load_config
from harness import denetim as dn
from harness import mutasyon as mt
from harness import run as hr
from tests.test_ajan import baglam  # noqa: F401  (paylasilan elle kurulmus baglam)

PROFIL = "fast"
KOSU = "dunya"


@pytest.fixture(scope="module")
def cfg():
    return load_config(PROFIL)


@pytest.fixture()
def temiz(cfg, baglam):  # noqa: F811
    """Sablonla uretilmis temiz brifing + defteri."""
    return nv.brifing_uret(cfg, baglam, "ECZ0000", istemci=None)


# --------------------------------------------------------------------------
# SAYI OKUMA
# --------------------------------------------------------------------------
@pytest.mark.parametrize("belirtec,beklenen", [
    ("0,38", [0.38]),              # TR ondalik; "038" = 38 OKUNMAZ
    ("0.38", [0.38]),              # EN ondalik
    ("-1453,01", [-1453.01]),      # isaret belirtece dahil
    ("1.234,56", [1234.56]),       # TR binlik + ondalik
    ("1,234.56", [1234.56]),       # EN binlik + ondalik
    ("60", [60.0, 60.0]),          # tek okuma, iki bicimde de ayni
])
def test_sayi_adaylari(belirtec, beklenen):
    assert dn.sayi_adaylari(belirtec) == pytest.approx(beklenen)


def test_2629_gercekten_belirsiz():
    """"2.629" hem 2629 hem 2,629 olabilir; belirsizlik korunmali."""
    assert dn.sayi_adaylari("2.629") == pytest.approx([2629.0, 2.629])


def test_gecersiz_gruplama_okunmuyor():
    """Binlik grubu ucer basamak degilse o okuma gecersizdir."""
    assert dn.sayi_adaylari("1.23") == pytest.approx([1.23])


# --------------------------------------------------------------------------
# BIRIM: bolum atfi
# --------------------------------------------------------------------------
def test_bolum_atfi_rejim_basligini_izliyor():
    metin = "\n".join(["# Teklif brifingi", "## Eczane", "genel satir",
                       "## Rejim: sok", "sok satiri", "## Kisit notlari",
                       "- [sok] satir icinde rejim"])
    atif = dict((s.strip(), e) for e, s in dn.bolumlere_ayir(metin, ["sok"]) if s.strip())
    assert atif["genel satir"] == at.GENEL
    assert atif["sok satiri"] == "sok"
    assert atif["- [sok] satir icinde rejim"] == "sok"


def test_kimlikler_sayi_taramasindan_cikariliyor():
    """"ECZ0007" icindeki 0007 bir olgu degil, kimligin parcasi."""
    assert "0007" not in dn.satiri_temizle("ECZ0007 icin brifing")


# --------------------------------------------------------------------------
# BIRIM: denetciler
# --------------------------------------------------------------------------
def test_temiz_brifing_sifir_bulgu(cfg, baglam, temiz):  # noqa: F811
    assert dn.denetle(cfg, temiz.metin, baglam, temiz.defter) == []


def test_uydurulmus_sayi_yakalaniyor(cfg, baglam, temiz):  # noqa: F811
    bozuk = temiz.metin.replace("## Rejim: sok", "## Rejim: sok\nBu rejimde 7777,77 TL bekleniyor.")
    tipler = [b.tip for b in dn.denetle(cfg, bozuk, baglam, temiz.defter)]
    assert "sayi_uydurma" in tipler


def test_baska_rejimin_sayisi_karisma_olarak_isaretleniyor(cfg, baglam, temiz):  # noqa: F811
    """Sayi GERCEK ama yanlis bolumde: bagil tolerans bunu goremez."""
    baz_marj = baglam.rejim_ozetleri["baz"]["beklenen_artimsal_marj"]
    bozuk = temiz.metin.replace(
        "## Rejim: sok",
        f"## Rejim: sok\nArtimsal marj {nv.sayi_bicimle(baz_marj)} TL.")
    bulgular = dn.denetle(cfg, bozuk, baglam, temiz.defter)
    assert any(b.tip == "senaryo_karismasi" and b.bolum == "sok" for b in bulgular)


def test_olmayan_kimlik_hallusinasyon(cfg, baglam, temiz):  # noqa: F811
    bozuk = temiz.metin.replace("SKU0001", "SKU9999", 1)
    assert any(b.tip == "hallusinasyon"
               for b in dn.denetle(cfg, bozuk, baglam, temiz.defter))


def test_eksik_rejim_bolumu_bicim_ihlali(cfg, baglam, temiz):  # noqa: F811
    """D3: butun rejimler yazilmak zorunda."""
    bozuk = temiz.metin.replace("## Rejim: sok", "## Baska baslik")
    assert any(b.tip == "bicim_ihlali"
               for b in dn.denetle(cfg, bozuk, baglam, temiz.defter))


def test_oneri_blogu_yoksa_bicim_ihlali(cfg, baglam, temiz):  # noqa: F811
    bozuk = temiz.metin.split(nv.ONERI_BASLIGI)[0]
    assert any(b.tip == "bicim_ihlali"
               for b in dn.denetle(cfg, bozuk, baglam, temiz.defter))


def test_vetolu_urun_onerisi_kisit_ihlali(cfg, baglam, temiz):  # noqa: F811
    """SKU0002 kirmizi receteli: hicbir kosulda onerilemez (D6)."""
    bozuk = mt.uygula("veto_onerisi", temiz.metin, baglam, "ECZ0000", 0.25)
    bulgular = dn.denetle(cfg, bozuk, baglam, temiz.defter)
    assert any(b.tip == "kisit_ihlali" and "SKU0002" in b.kanit for b in bulgular)


def test_yanlis_lot_referansi_kisit_ihlali(cfg, baglam, temiz):  # noqa: F811
    bozuk = mt.uygula("lot_karistirma", temiz.metin, baglam, "ECZ0000", 0.25)
    assert any(b.tip == "kisit_ihlali" and "lot" in b.kanit.lower()
               for b in dn.denetle(cfg, bozuk, baglam, temiz.defter))


def test_uretilmemis_oneri_ayri_tip(cfg, baglam, temiz):  # noqa: F811
    """Kimlikler gercek, kisit delinmemis, ama politika bunu onermedi."""
    bozuk = mt.uygula("uydurma_oneri", temiz.metin, baglam, "ECZ0000", 0.25)
    assert any(b.tip == "uydurma_oneri"
               for b in dn.denetle(cfg, bozuk, baglam, temiz.defter))


def test_oneri_alanlari_gercek_satirla_karsilastiriliyor(cfg, baglam, temiz):  # noqa: F811
    """Adet degistirilirse (ve baska rejimde de yoksa) uydurma sayilir."""
    bozuk = temiz.metin.replace("  adet: 20", "  adet: 21")
    bulgular = dn.denetle(cfg, bozuk, baglam, temiz.defter)
    assert any(b.tip in ("sayi_uydurma", "senaryo_karismasi") for b in bulgular)


# --------------------------------------------------------------------------
# BIRIM: mutasyon ureteci
# --------------------------------------------------------------------------
def test_her_mutasyonun_beklenen_tipi_taniniyor():
    for m in mt.MUTASYONLAR.values():
        assert m.beklenen_tip in dn.BULGU_TIPLERI


def test_uygulanamayan_mutasyon_sessiz_atlanmiyor(cfg, baglam, temiz):  # noqa: F811
    """Bu baglamda MF kanali kapali satir yok; mutasyon HATA vermeli."""
    assert not mt.MUTASYONLAR["mf_kanali_ihlali"].uygun(baglam, "ECZ0000")
    with pytest.raises(mt.MutasyonUygulanamaz):
        mt.uygula("mf_kanali_ihlali", temiz.metin, baglam, "ECZ0000", 0.25)


def test_uygun_eczane_bulunamazsa_hata(baglam):
    with pytest.raises(mt.MutasyonUygulanamaz):
        mt.uygun_eczane("mf_kanali_ihlali", baglam, ["ECZ0000"])


def test_sayi_bozma_temiz_ornek_uretiyor(cfg, baglam, temiz):  # noqa: F811
    """Bozulmus deger baska bir olguya carpiyorsa o aday secilmez."""
    bozuk = mt.uygula("sayi_bozma", temiz.metin, baglam, "ECZ0000",
                      cfg.harness.mutasyon_sapmasi, temiz.defter,
                      cfg.harness.sayi_toleransi_bagil,
                      tuple(cfg.harness.yuvarlama_basamaklari))
    tipler = [b.tip for b in dn.denetle(cfg, bozuk, baglam, temiz.defter)]
    assert tipler == ["sayi_uydurma"]


# --------------------------------------------------------------------------
# KONFIG KILIDI
# --------------------------------------------------------------------------
def test_tolerans_mutasyon_sapmasini_gecemez():
    """Tolerans genisse bozulmus sayi da 'eslesti' sayilir: denetci olur."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="denetcisi olu"):
        load_config(PROFIL, gecersiz_kilma={
            "harness.sayi_toleransi_bagil": 0.30,
            "harness.mutasyon_sapmasi": 0.25})


# --------------------------------------------------------------------------
# UCTAN UCA (gercek dunya)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gercek(cfg, tmp_path_factory):
    """Gercek dunya -> M4 hatti -> senaryo kosusu -> ajan baglami.

    Pahali (M4 ogreticisi egitiliyor) ama SART: elle kurulmus baglam
    denetcinin mantigini gosterir, gercek olgu paketiyle calistigini degil.
    """
    from core.io import Run
    from scripts.generate_world import dunya_yaz

    kok = tmp_path_factory.mktemp("m7")
    dunya_yaz(cfg, Run(KOSU, kok=kok))
    return hr.baglam_hazirla(cfg, KOSU, kok)


def test_gercek_dunyada_temiz_brifing_sifir_bulgu(cfg, gercek):
    eczane_id = gercek.adaylar[0]
    cikti = nv.brifing_uret(cfg, gercek.baglam, eczane_id, istemci=None)
    bulgular = dn.denetle(cfg, cikti.metin, gercek.baglam, cikti.defter)
    assert bulgular == [], f"yanlis alarm: {[str(b) for b in bulgular][:5]}"


def test_gercek_dunyada_butun_mutantlar_yakalaniyor(cfg, gercek):
    """M7'nin merkezi iddiasi: denetciler olu degil."""
    vakalar = [v for v in hr.vakalari_yukle() if v.kaynak == "sablon"]
    sonuc = hr.harness_kos(gercek, vakalar)
    kalan = [(s.vaka.ad, s.olcum) for s in sonuc.kalan]
    assert not kalan, f"yakalanmayan mutasyon: {kalan}"
    assert sum(1 for s in sonuc.sonuclar if s.vaka.tip == "mutant") >= 5


def test_gercek_dunyada_brifing_butun_rejimleri_tasiyor(cfg, gercek):
    """D3: kur tahmin edilmiyor, uc rejim de kosullu okuma olarak yaziliyor."""
    metin = nv.brifing_uret(cfg, gercek.baglam, gercek.adaylar[0],
                            istemci=None).metin
    for ad in gercek.baglam.rejimler:
        assert f"{nv.REJIM_BASLIGI}{ad}" in metin
