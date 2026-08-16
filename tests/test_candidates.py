"""Aday uretiminin degismezleri: sinir, point-in-time, muhasebe, determinizm.

Metrik degeri burada sinanmaz (o dunyaya ve seed'e bagli, isi
scripts/verify_m3.py'nin); yapisal dogruluk sinanir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from core.config import load_config
from core.io import Run
from features.okuma import GozlemlenebilirKaynak
from policy import candidates as ad
from scripts.generate_world import dunya_yaz
from scripts.verify_m2 import YASAK_ADLAR, kod_metni

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def kucuk_dunya(tmp_path_factory):
    cfg = load_config("fast")
    kok = tmp_path_factory.mktemp("dunya")
    dunya_yaz(cfg, Run("t", kok=kok))
    return cfg, kok


@pytest.fixture(scope="module")
def hazirlik(kucuk_dunya):
    cfg, kok = kucuk_dunya
    dunya = ad.dunya_yukle(GozlemlenebilirKaynak("t", kok=kok), cfg)
    t = ad.origin_haftalari(cfg, dunya.W)[-1]
    return cfg, kok, dunya, t, ad.gorunum_kur(dunya, cfg, t)


def test_politika_katmani_ground_truth_okumuyor():
    """policy/ altinda ground_truth'a giden bir kod yolu olmamali.

    features/ ve models/ icin ayni tarama tests/test_features.py'de. Aday
    uretimi de bir MODEL katmanidir; hedef kumesi ve oracle tavani yalnizca
    eval/aday.py'de kurulur.
    """
    bulgular = []
    for yol in sorted((REPO_ROOT / "policy").glob("*.py")):
        kod = kod_metni(yol)
        bulgular += [f"{yol.name}:{k}" for k in YASAK_ADLAR if k in kod]
    assert not bulgular, f"politika katmani ground_truth'a dokunuyor: {bulgular}"


def test_point_in_time_gelecek_silinince_skorlar_degismiyor(kucuk_dunya, tmp_path):
    """Aday skorlari yalnizca hafta <= origin verisinden kurulmali.

    Yorumla degil: dunyanin gelecegi kesilir, ayni origin yeniden kurulur ve
    BUTUN uretici matrisleri bit bazinda karsilastirilir.
    """
    cfg, kok = kucuk_dunya
    kaynak = GozlemlenebilirKaynak("t", kok=kok)
    dunya = ad.dunya_yukle(kaynak, cfg)
    kesme = dunya.W // 2

    kesik_kosu = Run("kesik", kok=tmp_path).prepare()
    for tablo in kaynak.tables():
        df = kaynak.tablo(tablo)
        if "hafta" in df.columns:
            df = df.filter(pl.col("hafta") <= kesme)
        kesik_kosu.write_observable(tablo, df)
    kesik = ad.dunya_yukle(GozlemlenebilirKaynak("kesik", kok=tmp_path), cfg)

    tam_gor = ad.gorunum_kur(dunya, cfg, kesme)
    kesik_gor = ad.gorunum_kur(kesik, cfg, kesme)
    tam = ad.uretici_skorlari(dunya, cfg, tam_gor)
    kes = ad.uretici_skorlari(kesik, cfg, kesik_gor)
    for uretici in tam:
        en_buyuk = float(np.max(np.abs(tam[uretici] - kes[uretici])))
        assert en_buyuk == 0.0, f"{uretici} gelecege bakiyor (fark={en_buyuk:.3g})"
    assert float(np.max(np.abs(tam_gor.depo_stok - kesik_gor.depo_stok))) == 0.0
    assert float(np.max(np.abs(tam_gor.miad_baskisi - kesik_gor.miad_baskisi))) == 0.0


def test_lot_kalanlari_kayitli_stokla_ozdes(kucuk_dunya):
    """Yeniden kurulan tahsis edilebilir stok, kayitli stoktan tam olarak
    gelecek haftanin partisi kadar kucuk olmali.

    Muhasebe kontrolu: politikanin gordugu stok uydurma degil, gozlemlenebilir
    tablolardan turetilmis ve kayitla ozdes. Fark tesadufi degil, tanimli.
    """
    cfg, kok = kucuk_dunya
    kaynak = GozlemlenebilirKaynak("t", kok=kok)
    dunya = ad.dunya_yukle(kaynak, cfg)
    kayit = kaynak.tablo("depo_stok_haftalik")
    sira = {s: i for i, s in enumerate(dunya.urunler["sku_id"].to_list())}

    for t in (20, 55, dunya.W - 1):
        gor = ad.gorunum_kur(dunya, cfg, t)
        ref = kayit.filter(pl.col("hafta") == t)
        kayitli = np.zeros(dunya.S)
        kayitli[[sira[s] for s in ref["sku_id"]]] = ref["eldeki_adet"].to_numpy()
        ileri = np.zeros(dunya.S)
        m = dunya.lot_giris == t + 1
        np.add.at(ileri, dunya.lot_s[m], dunya.lot_adet[m])
        assert np.abs((kayitli - gor.depo_stok) - ileri).max() == 0.0
        assert (gor.depo_stok >= 0).all()
        assert (gor.depo_stok <= kayitli).all(), "olmayan mal soz verilemez"


def test_tekrar_uretici_yeni_hucrede_sessiz(hazirlik):
    """`tekrar` tanimi geregi kordur: gecmisi olmayan hucrede sinyal uretemez.

    Cold start probleminin varligi budur; soguk_start ureticisi bu bosluk icin
    var (SPEC M3 "cold start icin eczane attribute'lari").
    """
    cfg, _, dunya, _, gor = hazirlik
    tekrar = ad.uretici_tekrar(dunya, cfg, gor)
    assert (tekrar[~gor.ikili] == 0).all()
    assert (tekrar[gor.ikili] > 0).all()
    soguk = ad.uretici_soguk_start(dunya, cfg, gor)
    assert (soguk[~gor.ikili] > 0).any(), "soguk start yeni hucrede de sessiz"


def test_soguk_start_eczanenin_kendi_gecmisini_kullanmiyor(hazirlik):
    """Cold start yalnizca KOMSULARIN alimina bakmali.

    Bir eczanenin kendi alim satiri silinince o eczanenin cold start skoru
    degismemeli; degisiyorsa uretici gecmisi olmayan hucrede calismaz.
    """
    cfg, _, dunya, _, gor = hazirlik
    onceki = ad.uretici_soguk_start(dunya, cfg, gor)
    hedef = 0
    bozuk = ad.OriginGorunumu(**{**gor.__dict__, "ikili": gor.ikili.copy()})
    bozuk.ikili[hedef, :] = False
    sonraki = ad.uretici_soguk_start(dunya, cfg, bozuk)
    assert np.abs(onceki[hedef] - sonraki[hedef]).max() == 0.0
    assert np.abs(onceki - sonraki).max() > 0.0, "komsulari da etkilememis - suphe"


def test_havuz_tekrar_uretilebilir_ve_tavani_asmiyor(hazirlik):
    cfg, _, dunya, _, gor = hazirlik
    a, _ = ad.aday_havuzu(dunya, cfg, gor)
    b, _ = ad.aday_havuzu(dunya, cfg, gor)
    assert a.equals(b)
    sayim = a.group_by("eczane_idx").len()["len"].max()
    assert sayim <= cfg.politika.aday.havuz_boyutu_k
    assert (a["skor"].to_numpy() > 0).all()
    assert a["teklif_adedi"].min() >= 1


def test_olcum_originleri_ortusmuyor(kucuk_dunya):
    """Degerlendirme pencereleri ortusurse ayni siparis birden fazla origin'de
    hedef sayilir ve recall'un guven araligi sahte bicimde daralir."""
    cfg, _ = kucuk_dunya
    originler = ad.origin_haftalari(cfg, 104)
    ufuk = cfg.politika.aday.degerlendirme.ufuk_hafta
    assert all(b - a >= ufuk for a, b in zip(originler, originler[1:]))
    assert max(originler) + ufuk <= 103


def test_hibrit_karisim_agirligi_sirayi_degistiriyor(hazirlik):
    """Karisim agirliklari olu knob olmamali: tek uretici acilinca hibrit o
    ureticinin sirasina donmeli."""
    cfg, kok, dunya, t, gor = hazirlik
    skorlar = ad.uretici_skorlari(dunya, cfg, gor)
    # Miad carpani SKU bazli bir carpan; yalnizca karisimin sirasini sinamak
    # icin kapatilir (carpanin kendi testi tests/test_constraints.py'de).
    carpansiz = load_config("fast", gecersiz_kilma={
        **{f"politika.aday.karisim_agirliklari.{u}": (1.0 if u == "cf" else 0.0)
           for u in cfg.politika.aday.URETICILER},
        "politika.aday.miad_baskisi_agirligi": 0.0})
    hibrit = ad.hibrit_skor(skorlar, carpansiz, gor)
    assert np.array_equal(np.argsort(-hibrit, axis=1, kind="stable"),
                          np.argsort(-ad.sira_normalize(skorlar["cf"]), axis=1,
                                     kind="stable"))
