"""Dunyanin degismezleri: determinizm, FEFO, gozlemlenebilirlik siniri, seyreklik.

Bunlar M1'in cikis kriterinin makineyle kontrol edilebilir kismidir.
Grafikli tam dogrulama scripts/verify_m1.py'de.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.config import load_config
from core.io import LATENT_COLUMNS
from core.rng import SeedBank
from sim.calendar import GUN_HAFTA
from sim.world import dunya_kos


@pytest.fixture(scope="module")
def dunya():
    cfg = load_config("fast")
    return cfg, dunya_kos(cfg, SeedBank(cfg.profil.temel_seed))


def test_ayni_seed_ayni_dunya():
    cfg = load_config("fast")
    a = dunya_kos(cfg, SeedBank(cfg.profil.temel_seed))
    b = dunya_kos(cfg, SeedBank(cfg.profil.temel_seed))
    assert a.siparisler.equals(b.siparisler)
    assert a.sevkiyat_satirlari.equals(b.sevkiyat_satirlari)
    assert a.hucre_haftalik.equals(b.hucre_haftalik)


def test_farkli_seed_farkli_dunya():
    cfg = load_config("fast")
    a = dunya_kos(cfg, SeedBank(cfg.profil.temel_seed))
    b = dunya_kos(cfg, SeedBank(cfg.profil.temel_seed + 1))
    assert not a.siparisler.equals(b.siparisler)


def test_seed_asamalari_bagimsiz():
    """Bir asamanin cekilis sayisi degisse bile digerinin akisi kaymamali."""
    a = SeedBank(7)
    assert a.seed_for("urun_evreni") != a.seed_for("eczane_evreni")
    assert SeedBank(7).seed_for("olaylar") == SeedBank(7).seed_for("olaylar")


def test_gozlemlenebilir_tablolarda_latent_kolon_yok(dunya):
    _, d = dunya
    gozlemlenebilir = [
        d.urunler, d.eczaneler, d.takvim, d.olaylar_gozlemlenebilir, d.siparisler,
        d.sevkiyat_satirlari, d.stok_lotlari, d.depo_stok_haftalik, d.imhalar,
        d.urun_fiyat_haftalik, d.makro_haftalik,
    ]
    for df in gozlemlenebilir:
        assert not (set(df.columns) & LATENT_COLUMNS)


def test_olay_tablosu_gelecegi_sizdirmiyor(dunya):
    """D4: gozlemlenebilir olay tablosunda antisipasyon bilgisi olmamali."""
    _, d = dunya
    assert "antisipasyon_baslangic_hafta" not in d.olaylar_gozlemlenebilir.columns
    assert "antisipasyon_siddeti" not in d.olaylar_gozlemlenebilir.columns
    # Olay ancak yururluge girdigi hafta gorunur olur.
    assert (d.olaylar_gercek["gorunur_hafta"]
            == d.olaylar_gercek["yururluk_hafta"]).all()


def test_talep_intermittent(dunya):
    """CLAUDE.md 7: cogu (eczane, SKU) hucresi cogu hafta sifir."""
    cfg, d = dunya
    toplam = cfg.profil.eczane_sayisi * cfg.profil.sku_sayisi * cfg.profil.hafta_sayisi
    dolu = (d.hucre_haftalik["gercek_tuketim"] > 0).sum()
    assert dolu / toplam < 0.20


def test_kronik_kategoriler_mevsimsel_degil(dunya):
    """Kronik ve mevsimsel kategori kontrasti olculebilir olmali."""
    cfg, d = dunya
    urun = d.urunler.select(["sku_id", "kategori_kod"])
    ay = d.takvim.select(["hafta", "ay"])
    aylik = (d.hucre_haftalik.join(urun, on="sku_id").join(ay, on="hafta")
             .group_by(["kategori_kod", "ay"])
             .agg(pl.col("gercek_tuketim").mean().alias("ort")))

    def cv(kod: str) -> float:
        x = aylik.filter(pl.col("kategori_kod") == kod)["ort"].to_numpy()
        return float(x.std() / max(x.mean(), 1e-9))

    kronik = [k.kod for k in cfg.urun.kategoriler if k.kronik]
    assert np.mean([cv(k) for k in kronik]) < cv("J07")


def test_fefo_ve_miad_toleransi(dunya):
    """Sevk edilen her lot, o eczanenin kabul esigini asmali."""
    cfg, d = dunya
    carpan = {k.kod: k.miad_toleransi_carpani for k in cfg.urun.kategoriler}
    s = (d.sevkiyat_satirlari
         .join(d.latent_eczane.select(["eczane_id", "miad_toleransi_gun"]), on="eczane_id")
         .join(d.urunler.select(["sku_id", "kategori_kod"]), on="sku_id")
         .with_columns((pl.col("miad_toleransi_gun")
                        * pl.col("kategori_kod").replace_strict(carpan, return_dtype=pl.Float64)
                        ).alias("gerekli")))
    assert s.height > 0
    assert (s["kalan_raf_omru_gun"] >= s["gerekli"]).all()


def test_lot_bakiyesi_asilmiyor(dunya):
    """Bir lottan giris adedinden fazla sevk + imha yapilamaz."""
    _, d = dunya
    sevk = d.sevkiyat_satirlari.group_by("lot_id").agg(pl.col("adet").sum().alias("sevk"))
    imha = d.imhalar.group_by("lot_id").agg(pl.col("adet").sum().alias("imha"))
    b = (d.stok_lotlari.join(sevk, on="lot_id", how="left")
         .join(imha, on="lot_id", how="left")
         .with_columns(pl.col("sevk").fill_null(0), pl.col("imha").fill_null(0)))
    assert (b["sevk"] + b["imha"] <= b["adet_giris"]).all()


def test_imha_sadece_miadi_gecmis_lotta(dunya):
    _, d = dunya
    b = d.imhalar.join(d.stok_lotlari.select(["lot_id", "miad_gun_indeksi"]), on="lot_id")
    assert (b["miad_gun_indeksi"] <= b["hafta"] * GUN_HAFTA).all()


def test_siparis_karsilanani_asamaz(dunya):
    _, d = dunya
    assert (d.siparisler["karsilanan_adet"] <= d.siparisler["talep_adet"]).all()
    assert (d.siparisler["miad_kisiti_nedeniyle_verilemeyen"]
            <= d.siparisler["talep_adet"] - d.siparisler["karsilanan_adet"]).all()


def test_share_of_wallet_latent_kaliyor(dunya):
    """Modelin gorebildigi hicbir tabloda SOW turevi olmamali."""
    _, d = dunya
    assert "share_of_wallet" in d.sow_haftalik.columns          # ground truth'ta var
    assert "share_of_wallet" not in d.eczaneler.columns         # observable'da yok


def test_cesit_orani_durgun(dunya):
    """Cesit SEVIYESI ufuk boyunca kaymamali, sadece kompozisyonu degismeli.

    Bu bir regresyon testi: ekleme ve cikarma ayri olasiliklar oldugunda
    tabanlari farkli oldugu icin cesit orani 104 haftada %15.8'den %32.8'e
    kaymisti (reports/m1.md 3.6).
    """
    cfg, d = dunya
    g = (d.hucre_haftalik.group_by("hafta")
         .agg(pl.col("cesitte_var").sum().alias("n")).sort("hafta"))
    seri = g["n"].to_numpy().astype(float)
    assert abs(seri[-1] - seri[0]) / seri[0] < 0.10
    # ...ama kompozisyon gercekten degismeli, yoksa churn olu demektir.
    ilk = set(d.hucre_haftalik.filter((pl.col("hafta") == 0) & pl.col("cesitte_var"))
              .select(pl.concat_str(["eczane_id", "sku_id"])).to_series().to_list())
    son = set(d.hucre_haftalik.filter((pl.col("hafta") == cfg.profil.hafta_sayisi - 1)
                                      & pl.col("cesitte_var"))
              .select(pl.concat_str(["eczane_id", "sku_id"])).to_series().to_list())
    assert 0 < len(ilk - son) < len(ilk)


def test_eczane_stogu_makul_kapsamada(dunya):
    """Eczane stogu hedef kapsamanin katbekat uzerine cikmamali.

    Regresyon: varyans gudumlu emniyet stogu tavansizken hucreler 50+ haftalik
    stoga cikiyordu (reports/m1.md 3.7).
    """
    cfg, d = dunya
    a = d.hucre_haftalik.filter(pl.col("cesitte_var"))
    haftalik_tuketim = a["gercek_tuketim"].sum() / cfg.profil.hafta_sayisi
    haftalik_stok = (a.group_by("hafta").agg(pl.col("gercek_eczane_stogu").sum())
                     ["gercek_eczane_stogu"].mean())
    assert haftalik_stok / haftalik_tuketim < 12.0


def test_iade_tuketim_hiziyla_kupleli(dunya):
    """SPEC 2.5: ayni kalan raf omru yavas eczanede zayi, hizli eczanede degil.

    Duz bir gun esigi kullanilsaydi iade orani hucre hizindan bagimsiz olurdu.
    """
    cfg, d = dunya
    assert d.iadeler.height > 0
    a = d.hucre_haftalik.filter(pl.col("cesitte_var"))
    hiz = a.group_by(["eczane_id", "sku_id"]).agg(
        (pl.col("gercek_tuketim").sum() / cfg.profil.hafta_sayisi).alias("hz"))
    sevk = d.sevkiyat_satirlari.group_by(["eczane_id", "sku_id"]).agg(
        pl.col("adet").sum().alias("sevk"))
    b = (sevk.join(d.iadeler.group_by(["eczane_id", "sku_id"]).agg(pl.col("iade_adet").sum()),
                   on=["eczane_id", "sku_id"], how="left")
         .join(hiz, on=["eczane_id", "sku_id"], how="left")
         .with_columns(pl.col("iade_adet").fill_null(0), pl.col("hz").fill_null(0.0)))
    yavas = b.filter(pl.col("hz") < 0.5)
    hizli = b.filter(pl.col("hz") >= 2.0)
    assert (yavas["iade_adet"].sum() / yavas["sevk"].sum()
            > hizli["iade_adet"].sum() / hizli["sevk"].sum())


def test_eczane_stogu_miad_tasiyor(dunya):
    """Iade edilen malin kalan raf omru degerlendirme penceresi icinde olmali."""
    cfg, d = dunya
    assert (d.iadeler["kalan_raf_omru_gun"] <= cfg.sim.iade.degerlendirme_esigi_gun).all()
    assert (d.iadeler["kalan_raf_omru_gun"] >= 0).all()
    assert (d.iadeler["depoya_donen_adet"] <= d.iadeler["iade_adet"]).all()


def test_iade_imhaya_kaynagiyla_yansiyor(dunya):
    """Depoya donen iade satilamaz: imha kayitlarinda kaynagiyla gorunmeli."""
    _, d = dunya
    kaynaklar = set(d.imhalar["kaynak"].unique().to_list())
    assert "depo_miad" in kaynaklar
    assert any(k.startswith("eczane_iadesi") for k in kaynaklar)
    iade_imha = d.imhalar.filter(pl.col("kaynak").str.starts_with("eczane_iadesi"))
    assert iade_imha["adet"].sum() == d.iadeler["depoya_donen_adet"].sum()


def test_referans_kur_antisipasyonu_tuketimi_degistirmiyor(dunya):
    """D4: kur beklentisi SIPARISI one ceker, TUKETIMI degistirmez."""
    cfg, d = dunya
    kur = d.olaylar_gercek.filter(pl.col("tip") == "REFERANS_KUR_GUNCELLEME")
    if kur.height == 0:
        pytest.skip("bu profilde kur guncelleme olayi olusmadi")
    W = cfg.profil.hafta_sayisi
    tuketim = np.zeros(W)
    g = d.hucre_haftalik.group_by("hafta").agg(pl.col("gercek_tuketim").sum()).sort("hafta")
    tuketim[g["hafta"].to_numpy()] = g["gercek_tuketim"].to_numpy()
    oranlar = []
    for satir in kur.iter_rows(named=True):
        t0, bas = satir["yururluk_hafta"], satir["antisipasyon_baslangic_hafta"]
        if t0 <= bas:
            continue
        dis = np.ones(W, dtype=bool)
        dis[max(0, bas - 4): min(W, t0 + 4)] = False
        if tuketim[dis].mean() > 0:
            oranlar.append(tuketim[bas:t0].mean() / tuketim[dis].mean())
    if oranlar:
        assert abs(float(np.mean(oranlar)) - 1.0) < 0.25
