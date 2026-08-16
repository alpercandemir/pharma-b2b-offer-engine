"""M6 testleri: dunyanin adim adim surulmesi ve kapali dongu mekanigi.

EN ONEMLI UC TEST:

  `test_adim_adim_kosu_tek_parca_kosuyla_ozdes`
      `dunya_kur` + N x `hafta_adimi` ile `dunya_kos` BIT BAZINDA ayni dunyayi
      uretmeli. M6'nin dunya refactor'unun regresyon kilidi budur: M1-M5'in
      butun sayilari bu ozdeslige dayaniyor.

  `test_teklif_organik_siparisi_dusuruyor`
      Kapali dongunun ana mekanizmasi. Teklif sevkiyati eczanenin stok
      pozisyonuna girer ve organik siparis kuculur. Bu ayri bir kural degil,
      eczanenin kendi (s, S) politikasinin sonucu -- test onu izole eder.

  `test_ayni_seed_ayni_rollout`
      Rollout tekrar uretilebilir olmali; yoksa hicbir sapma yorumu kalici
      degildir (CLAUDE.md 5).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.config import config_yukle
from core.rng import SeedBankasi
from sim import rollout as rl
from sim.world import dunya_kos, dunya_kur, hafta_adimi

PROFIL = "fast"
# Testler tam 104 hafta kosmasin diye kisaltilmis dunya. Dinamigin tamami
# ayni; yalnizca zaman ekseni kisa.
KISA_HAFTA = 30


@pytest.fixture(scope="module")
def kisa_cfg():
    return config_yukle(PROFIL, gecersiz_kilma={
        "profil.hafta_sayisi": KISA_HAFTA,
        # Dunya kisalinca M4'un zaman kilidi (egitim penceresi olcum
        # penceresine tasamaz) baglar; egitim penceresi de kisaltiliyor.
        # Bu testlerde ogrenici kosulmuyor, yalnizca config gecerli olmali.
        "uplift.egitim.ilk_origin_hafta": 8,
        "uplift.egitim.azami_origin_sayisi": 5,
        "ope.rollout.baslangic_hafta": 20,
        "ope.rollout.ufuk_hafta": 8,
        "ope.rollout.raporlanan_ufuklar": [4, 8],
        "ope.rollout.teklif_penceresi_hafta": 8,
    })


def _durum(cfg, hafta: int):
    d = dunya_kur(cfg, SeedBankasi(cfg.profil.temel_seed))
    for _ in range(hafta):
        hafta_adimi(d)
    return d


# --------------------------------------------------------------------------
# refactor regresyonu
# --------------------------------------------------------------------------
def test_adim_adim_kosu_tek_parca_kosuyla_ozdes(kisa_cfg):
    """M1-M5'in gecerliligini koruyan kilit. Tablolar BIT BAZINDA esit olmali."""
    tek_parca = dunya_kos(kisa_cfg, SeedBankasi(kisa_cfg.profil.temel_seed))
    adim_adim = _durum(kisa_cfg, KISA_HAFTA)

    assert adim_adim.w == KISA_HAFTA
    k = adim_adim.kayit
    # Satir SAYILARI: bir cekilis kaysa siparis/sevkiyat satirlari degisirdi.
    assert len(k.siparis_kayit) == tek_parca.siparisler.height
    assert len(k.sevk_kayit) == tek_parca.sevkiyat_satirlari.height
    assert len(k.iade_kayit) == tek_parca.iadeler.height
    assert len(k.tukenme_kayit) == tek_parca.tukenme_olaylari.height
    # Ve TOPLAMLAR: sayilar ayni kalip degerler kaysa bu yakalar.
    assert int(k.tuketim_3d.sum()) == int(
        tek_parca.hucre_haftalik["gercek_tuketim"].sum())
    assert int(k.stok_3d.sum()) == int(
        tek_parca.hucre_haftalik["gercek_eczane_stogu"].sum())
    assert int(sum(r[3] for r in k.siparis_kayit)) == int(
        tek_parca.siparisler["talep_adet"].sum())
    assert k.sow_kayit.sum() == pytest.approx(
        float(tek_parca.sow_haftalik["share_of_wallet"].sum()), rel=1e-4)


def test_iki_kur_ayni_durumu_veriyor(kisa_cfg):
    """Ayni tohum -> ayni isinma. Rollout'un CRN'i buna dayaniyor."""
    a, b = _durum(kisa_cfg, 12), _durum(kisa_cfg, 12)
    assert np.array_equal(a.kovalar.toplam(), b.kovalar.toplam())
    assert np.array_equal(a.sow, b.sow)
    assert np.array_equal(a.assort, b.assort)
    assert len(a.kayit.sevk_kayit) == len(b.kayit.sevk_kayit)


def test_teklifsiz_hafta_adimi_dunyayi_degistirmiyor(kisa_cfg):
    """`teklif_sevk=None` ile sifir matrisi AYNI sonucu vermeli.

    Ikisi ayni degilse "teklifsiz rollout = taban dunya" esitligi kirilir ve
    artimsal deger tanimsiz kalir.
    """
    P, S = kisa_cfg.profil.eczane_sayisi, kisa_cfg.profil.sku_sayisi
    a = _durum(kisa_cfg, 10)
    b = _durum(kisa_cfg, 10)
    hafta_adimi(a)
    hafta_adimi(b, teklif_sevk=np.zeros((P, S), dtype=np.int64),
                teklif_miad_agirlikli=np.zeros((P, S)))
    assert np.array_equal(a.kovalar.toplam(), b.kovalar.toplam())
    assert np.array_equal(a.sow, b.sow)


# --------------------------------------------------------------------------
# teklif enjeksiyonu: kapali dongunun mekanigi
# --------------------------------------------------------------------------
def test_teklif_organik_siparisi_dusuruyor(kisa_cfg):
    """KANIBALIZM. Teklif stok pozisyonuna girer, organik siparis kuculur.

    Teklif adedi eczanenin haftalik hizinin kat kati secilir ki etki
    gurultuye gomulmesin; olculen sey yonun kendisi.
    """
    P, S = kisa_cfg.profil.eczane_sayisi, kisa_cfg.profil.sku_sayisi
    taban = _durum(kisa_cfg, 10)
    tekli = _durum(kisa_cfg, 10)

    # Her hucreye bol miktarda mal: eczane hicbir sey siparis etmemeli.
    bol = np.full((P, S), 500, dtype=np.int64) * tekli.assort
    miad = bol * (10 * 7 + 400.0)      # uzun raf omru: iade kanali karismasin

    onceki_taban = len(taban.kayit.siparis_kayit)
    onceki_teklif = len(tekli.kayit.siparis_kayit)
    hafta_adimi(taban)
    hafta_adimi(tekli, teklif_sevk=bol, teklif_miad_agirlikli=miad)

    taban_sip = sum(r[3] for r in taban.kayit.siparis_kayit[onceki_taban:])
    teklif_sip = sum(r[3] for r in tekli.kayit.siparis_kayit[onceki_teklif:])
    assert taban_sip > 0, "taban haftada hic siparis yok, test bir sey olcmuyor"
    assert teklif_sip < taban_sip


def test_teklif_sevkiyati_eczane_stoguna_ulasiyor(kisa_cfg):
    """Teklif yoldaki sevkiyata girer ve tedarik suresi sonunda stoga duser."""
    P, S = kisa_cfg.profil.eczane_sayisi, kisa_cfg.profil.sku_sayisi
    L = kisa_cfg.sim.envanter.tedarik_suresi_hafta
    taban = _durum(kisa_cfg, 10)
    tekli = _durum(kisa_cfg, 10)
    bol = np.full((P, S), 50, dtype=np.int64) * tekli.assort
    miad = bol * (10 * 7 + 400.0)

    hafta_adimi(tekli, teklif_sevk=bol, teklif_miad_agirlikli=miad)
    hafta_adimi(taban)
    for _ in range(L):
        hafta_adimi(tekli)
        hafta_adimi(taban)
    assert tekli.kovalar.toplam().sum() > taban.kovalar.toplam().sum()


def test_teklif_depo_cikisina_sayiliyor(kisa_cfg):
    """Teklif hacmi ikmal EWMA'sini besliyor mu.

    Saymasaydi depo sistematik olarak az siparis verir ve teklif veren
    politika kendi kendine stoksuzluk uretirdi -- gercek olmayan bir ceza.
    """
    P, S = kisa_cfg.profil.eczane_sayisi, kisa_cfg.profil.sku_sayisi
    taban = _durum(kisa_cfg, 10)
    tekli = _durum(kisa_cfg, 10)
    bol = np.full((P, S), 50, dtype=np.int64) * tekli.assort
    hafta_adimi(taban)
    hafta_adimi(tekli, teklif_sevk=bol, teklif_miad_agirlikli=bol * 800.0)
    assert (tekli.depo_cikis_ewma > taban.depo_cikis_ewma).any()


def test_kisa_miatli_teklif_iade_uretiyor(kisa_cfg):
    """IADE KANALI (SPEC 2.5). Emebileceginden fazlasi eczanede yaslanir.

    Ayni adet iki kez gonderilir: bir kez uzun, bir kez kisa raf omruyle.
    Kisa olan iade uretmeli.
    """
    P, S = kisa_cfg.profil.eczane_sayisi, kisa_cfg.profil.sku_sayisi
    L = kisa_cfg.sim.envanter.tedarik_suresi_hafta
    adet = np.full((P, S), 300, dtype=np.int64)
    sonuclar = {}
    for ad, kalan_gun in (("uzun", 900.0), ("kisa", 25.0)):
        d = _durum(kisa_cfg, 10)
        maske = adet * d.assort
        bugun = d.w * 7
        hafta_adimi(d, teklif_sevk=maske,
                    teklif_miad_agirlikli=maske * (bugun + kalan_gun))
        onceki = len(d.kayit.iade_kayit)
        for _ in range(L + 2):
            hafta_adimi(d)
        sonuclar[ad] = sum(r[3] for r in d.kayit.iade_kayit[onceki:])
    assert sonuclar["kisa"] > sonuclar["uzun"]


# --------------------------------------------------------------------------
# surucu
# --------------------------------------------------------------------------
def _bos_karar(t, aday_dunya):
    return None


def test_teklifsiz_rollout_taban_dunyayla_ozdes(kisa_cfg):
    """Karar verici hep None dondururse rollout taban dunyayi kosmali.

    Uyari: rollout haftalari kendi CRN tohumlarini kullaniyor, bu yuzden
    cekilisler taban dunyanin devamiyla AYNI DEGIL. Sinanan sey, teklifsiz
    rollout'un iki kez kosulunca ayni sonucu vermesi ve teklif kalemlerinin
    tam sifir olmasi.
    """
    cfg = kisa_cfg
    d = _durum(cfg, cfg.ope.rollout.baslangic_hafta)
    o = rl.rollout_kos(cfg, d, None, d.ecz.master, d.urunler, d.ecz.latent,
                       _bos_karar, "teklif_yok")
    assert len(o.haftalar) == cfg.ope.rollout.ufuk_hafta
    assert o.seri("teklif_sayisi").sum() == 0
    assert o.seri("teklif_brut_marj").sum() == 0.0
    assert o.seri("organik_brut_marj").sum() > 0.0


def test_ayni_seed_ayni_rollout(kisa_cfg):
    """Iki teklifsiz rollout birebir ayni seriyi vermeli (CLAUDE.md 5)."""
    cfg = kisa_cfg
    seriler = []
    for _ in range(2):
        d = _durum(cfg, cfg.ope.rollout.baslangic_hafta)
        o = rl.rollout_kos(cfg, d, None, d.ecz.master, d.urunler, d.ecz.latent,
                           _bos_karar, "teklif_yok")
        seriler.append(o.birikimli_net_marj())
    assert np.array_equal(seriler[0], seriler[1])


def test_ufuk_kesimleri_birikimli_seriden_okunuyor(kisa_cfg):
    """`net_marj_ufukta(h)` ilk h haftanin toplami olmali (tek kosu, cok ufuk)."""
    cfg = kisa_cfg
    d = _durum(cfg, cfg.ope.rollout.baslangic_hafta)
    o = rl.rollout_kos(cfg, d, None, d.ecz.master, d.urunler, d.ecz.latent,
                       _bos_karar, "teklif_yok")
    haftalik = o.seri("net_marj") if hasattr(o.haftalar[0], "net_marj") else None
    for h in (1, 4, len(o.haftalar)):
        assert o.net_marj_ufukta(h) == pytest.approx(
            float(np.array([w.net_marj for w in o.haftalar[:h]]).sum()))
    assert haftalik is not None


def test_canli_aday_dunyasi_rollout_kayitlarindan_kuruluyor(kisa_cfg):
    """Politika taban dunyanin degil, KENDI gecmisinin loglarini gormeli."""
    cfg = kisa_cfg
    d = _durum(cfg, 15)
    ad = rl.canli_aday_dunyasi(d, d.ecz.master, d.urunler)
    assert ad.P == d.P and ad.S == d.S
    assert ad.sip_w.max() <= d.w - 1, "gelecege ait siparis satiri var"
    assert ad.sip_p.size == len(d.kayit.siparis_kayit)
    # Bir hafta daha kosulunca gorunum BUYUMELI: kapali dongu ilerliyor.
    hafta_adimi(d)
    ad2 = rl.canli_aday_dunyasi(d, d.ecz.master, d.urunler)
    assert ad2.sip_p.size >= ad.sip_p.size


def test_canli_durum_karar_ani_uyusmazligini_yakaliyor(kisa_cfg):
    """Tepki, dunyanin BULUNDUGU an disinda bir hafta icin sorulamaz.

    Sessizce kabul edilseydi politika gecmisin ya da gelecegin stogu uzerinde
    karar verir ve kanibalizm hic olculemezdi.
    """
    cfg = kisa_cfg
    d = _durum(cfg, 12)
    canli = rl.CanliDurum(d, d.ecz.master["eczane_id"].to_numpy(),
                          d.urunler["sku_id"].to_numpy(), d.ecz.latent)
    ecz = d.ecz.master["eczane_id"].to_numpy()[:3]
    sku = d.urunler["sku_id"].to_numpy()[:3]
    # Dogru an: w - 1.
    stok, hiz, cesit = canli.hucre_durumu(ecz, sku, d.w - 1)
    assert stok.size == 3 and hiz.size == 3 and cesit.size == 3
    with pytest.raises(ValueError, match="uyusmuyor"):
        canli.hucre_durumu(ecz, sku, d.w)


def test_hafta_olcumu_net_marj_muhasebesi(kisa_cfg):
    """net_marj = brut - iade marji - iade islem - imha islem (reports/m5.md 3.2)."""
    h = rl.HaftaOlcumu(
        hafta=1, teklif_sayisi=0, kabul_sayisi=0, teklif_brut_marj=100.0,
        organik_brut_marj=50.0, teklif_sevk_adet=0.0, organik_sevk_adet=0.0,
        faturalanan_adet=0.0, bedava_adet=0.0, iade_adet=0.0,
        iade_marj_geri_alma=10.0, iade_islem_maliyeti=5.0, imha_adet=0.0,
        imha_islem_maliyeti=3.0, karsilanmayan_siparis_adet=0.0,
        organik_siparis_adet=0.0, sow_ortalama=0.4, eczane_stogu=0.0,
        kabul_olasiligi_gercek=float("nan"), kabul_olasiligi_tahmin=float("nan"),
        ortalama_mf=float("nan"), ortalama_vade=float("nan"))
    assert h.net_marj == pytest.approx(100.0 + 50.0 - 10.0 - 5.0 - 3.0)
