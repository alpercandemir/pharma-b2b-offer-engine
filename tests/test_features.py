"""Feature katmaninin degismezleri: sizinti sinirı, point-in-time, etiket.

Bunlar M2'nin "durustluk" testleridir. Metrikler ancak bunlar geciyorsa
anlamlidir; bu yuzden metrik testlerinden ayri dosyada duruyorlar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from core.config import config_yukle
from core.io import Kosu
from features.okuma import GozlemlenebilirKaynak
from features.panel import izgara_kur, panel_kur
from scripts.generate_world import dunya_yaz
from scripts.verify_m2 import YASAK_ADLAR, kod_metni

KOK = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def kucuk_dunya(tmp_path_factory):
    """Test icin kucuk bir dunya. Kosu dizini gecici; repo veri dizini kirlenmez."""
    cfg = config_yukle("fast")
    kok = tmp_path_factory.mktemp("dunya")
    dunya_yaz(cfg, Kosu("t", kok=kok))
    return cfg, kok


def test_feature_katmani_ground_truth_okumuyor():
    """features/ ve models/ altinda ground_truth'a giden kod yolu olmamali.

    Docstring'lerde gecmesi serbest (sinirin neden boyle kuruldugunu
    anlatiyorlar); kodda gecmesi degil.
    """
    bulgular = []
    for dizin in ("features", "models"):
        for yol in sorted((KOK / dizin).glob("*.py")):
            kod = kod_metni(yol)
            bulgular += [f"{yol.name}:{k}" for k in YASAK_ADLAR if k in kod]
    assert not bulgular, f"feature/model katmani ground_truth'a dokunuyor: {bulgular}"


def test_kaynak_ground_truth_tablosunu_acmiyor(kucuk_dunya):
    _, kok = kucuk_dunya
    kaynak = GozlemlenebilirKaynak("t", kok=kok)
    assert "siparisler" in kaynak.tablolar()
    assert "hucre_haftalik" not in kaynak.tablolar()
    with pytest.raises(FileNotFoundError):
        kaynak.tablo("hucre_haftalik")
    with pytest.raises(ValueError):
        kaynak.tablo("../ground_truth/hucre_haftalik")


def test_point_in_time_gelecek_silinince_ozellikler_degismiyor(kucuk_dunya, tmp_path):
    """Leakage guard'in asil testi: gelecegi kesip yeniden hesapla, karsilastir."""
    cfg, kok = kucuk_dunya
    kaynak = GozlemlenebilirKaynak("t", kok=kok)
    izg = izgara_kur(kaynak, cfg)
    kesme = izg.W // 2
    originler = np.array([kesme - 6, kesme - 2, kesme])
    tam = panel_kur(izg, cfg, originler)

    kesik_kosu = Kosu("kesik", kok=tmp_path).hazirla()
    for ad in kaynak.tablolar():
        df = kaynak.tablo(ad)
        if "hafta" in df.columns:
            df = df.filter(pl.col("hafta") <= kesme)
        kesik_kosu.yaz_gozlemlenebilir(ad, df)
    kesik = panel_kur(izgara_kur(GozlemlenebilirKaynak("kesik", kok=tmp_path), cfg),
                      cfg, originler)

    a = tam.anahtar.with_row_index("i")
    b = kesik.anahtar.with_row_index("j")
    ortak = a.join(b, on=["eczane_id", "sku_id", "origin"], how="inner")
    assert ortak.height > 100
    fark = np.abs(tam.X[ortak["i"].to_numpy()] - kesik.X[ortak["j"].to_numpy()])
    en_kotu = int(np.nanargmax(fark.max(axis=0)))
    assert np.nanmax(fark) == 0.0, (
        f"gelecek ozellikleri etkiliyor; en kotu sutun: {tam.ozellik_adlari[en_kotu]}")


def test_etiket_gelecekten_geliyor(kucuk_dunya):
    """etiket_k = origin sonrasi ilk siparis; siparis tablosuyla dogrulanir."""
    cfg, kok = kucuk_dunya
    izg = izgara_kur(GozlemlenebilirKaynak("t", kok=kok), cfg)
    panel = panel_kur(izg, cfg)
    olayli = np.flatnonzero(panel.etiket_k > 0)[:500]
    for i in olayli:
        t, k = int(panel.origin[i]), int(panel.etiket_k[i])
        seri = izg.talep[panel.hucre_idx[i]]
        assert seri[t + k] > 0, "etiket haftasinda siparis yok"
        assert seri[t + 1: t + k].sum() == 0, "daha erken bir siparis atlanmis"


def test_etiket_ufku_asmiyor(kucuk_dunya):
    cfg, kok = kucuk_dunya
    panel = panel_kur(izgara_kur(GozlemlenebilirKaynak("t", kok=kok), cfg), cfg)
    ufuk = cfg.tukenme.hedef.ufuk_hafta
    assert panel.etiket_k.max() <= ufuk
    assert (panel.izlenen_k <= ufuk).all()
    assert (panel.etiket_k[panel.etiket_k > 0]
            <= panel.izlenen_k[panel.etiket_k > 0]).all()


def test_hiz_tahmini_gozlenen_payla_olcekli(kucuk_dunya):
    """Akis tahmini gercek tuketimin share_of_wallet kadarlik parcasi olmali.

    Bu bir MODEL testi degil, DUNYA testi: M2'nin butun cikarim yapisi bu
    olceklemeye dayaniyor. Bozulursa (orn. simulator degisirse) once burasi
    kirmizi yanar.
    """
    cfg, kok = kucuk_dunya
    izg = izgara_kur(GozlemlenebilirKaynak("t", kok=kok), cfg)
    panel = panel_kur(izg, cfg)
    hucre = pl.read_parquet(kok / "t" / "ground_truth" / "hucre_haftalik.parquet")
    sow = pl.read_parquet(kok / "t" / "ground_truth" / "sow_haftalik.parquet")

    uzun = max(cfg.feature.hiz.pencereler_hafta)
    sutun = panel.ozellik_adlari.index(f"hiz_akis_{uzun}h")
    son_origin = int(panel.origin.max())
    sec = np.flatnonzero(panel.origin == son_origin)
    tahmin = panel.X[sec, sutun]

    gercek = (hucre.filter((pl.col("hafta") > son_origin - uzun)
                           & (pl.col("hafta") <= son_origin))
              .group_by(["eczane_id", "sku_id"])
              .agg((pl.col("gercek_tuketim").sum() / uzun).alias("hiz")))
    esle = dict(zip(zip(gercek["eczane_id"], gercek["sku_id"]), gercek["hiz"]))
    g = np.array([esle.get((e, s), 0.0) for e, s
                  in zip(panel.anahtar["eczane_id"].to_numpy()[sec],
                         panel.anahtar["sku_id"].to_numpy()[sec])])
    gecerli = (g > 0) & (tahmin > 0)
    oran = float(np.median(tahmin[gecerli] / g[gecerli]))
    ortalama_sow = float(sow["share_of_wallet"].mean())
    assert 0.3 * ortalama_sow < oran < 2.0 * ortalama_sow, (
        f"akis/gercek orani {oran:.2f}, ortalama share_of_wallet {ortalama_sow:.2f}")
    assert oran < 1.0, "akis tahmini gercek tuketimi asamaz (multi-homing var)"


def test_defter_orani_gozlenen_pay_telafisinden_bagimsiz(kucuk_dunya):
    """Oran sadelesmesi: telafi katsayisi seviyeleri kaydirir, SUREYI degil.

    features/stok.py'nin iddiasi bu. Iddia degil, test.
    """
    _, kok = kucuk_dunya
    ozellikler = {}
    for pay in (1.0, 0.4):
        cfg = config_yukle("fast",
                           gecersiz_kilma={"feature.stok.varsayilan_gozlenen_pay": pay})
        panel = panel_kur(izgara_kur(GozlemlenebilirKaynak("t", kok=kok), cfg), cfg)
        ad = panel.ozellik_adlari
        ozellikler[pay] = {
            "stok": panel.X[:, ad.index("defter_stok")],
            "sure": panel.X[:, ad.index("defter_tukenme_hafta")],
            "son_siparis_sure": panel.X[:, ad.index("son_siparis_tukenme_hafta")],
        }
    a, b = ozellikler[1.0], ozellikler[0.4]
    # Seviye: telafi ile olceklenmeli (kirpilmamis hucrelerde)
    buyuk = a["stok"] > 0
    assert np.median(b["stok"][buyuk] / a["stok"][buyuk]) > 1.5
    # Sure: sadelesmeli. min_hiz kirpmasi birkac hucreyi bozabilir, medyan sifir.
    assert float(np.median(np.abs(b["sure"] - a["sure"]))) == 0.0
    # Son-siparis tahmincisi sadelesmez: telafi arttikca sure KISALIR.
    assert float(np.mean(b["son_siparis_sure"])) < float(np.mean(a["son_siparis_sure"]))
