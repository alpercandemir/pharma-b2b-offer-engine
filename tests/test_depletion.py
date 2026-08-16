"""Tukenme modeli ve olcum katmaninin degismezleri.

Buradaki testler METRIK degeri sinamaz (o dunyaya ve seed'e bagli, isi
scripts/verify_m2.py'nin); yapisal dogrulugu sinar: kisi-periyot acilimi,
hayatta kalma egrisinin tutarliligi, oracle'in rakip riski ayirmasi ve
kosunun tekrar uretilebilirligi.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import config_yukle
from core.io import Kosu
from eval import metrics as mt
from eval.oracle import Oracle
from experiments.run import kosu_yap, zaman_bolmesi
from features.okuma import GozlemlenebilirKaynak
from features.panel import izgara_kur, panel_kur
from models.depletion import (HazardTahmincisi, KovaKalibratoru,
                              KuralTahmincisi, gozlemlenebilir_olay,
                              kisi_periyot)
from scripts.generate_world import dunya_yaz

# Testler hizli kossun diye kucultulmus model. Determinizm ve yapi agac
# sayisindan bagimsiz; metrik seviyesi bu testlerde zaten sinanmiyor.
TEST_KNOBLARI = {
    "tukenme.model.azami_agac": 40,
    "tukenme.degerlendirme.oracle_teshisi": False,
}


@pytest.fixture(scope="module")
def hazirlik(tmp_path_factory):
    cfg = config_yukle("fast", gecersiz_kilma=TEST_KNOBLARI)
    kok = tmp_path_factory.mktemp("dunya")
    dunya_yaz(cfg, Kosu("t", kok=kok))
    izg = izgara_kur(GozlemlenebilirKaynak("t", kok=kok), cfg)
    panel = panel_kur(izg, cfg)
    return cfg, kok, izg, panel, zaman_bolmesi(panel, cfg, izg.W)


def test_kisi_periyot_acilimi(hazirlik):
    cfg, _, _, panel, bolme = hazirlik
    ufuk = cfg.tukenme.hedef.ufuk_hafta
    idx = bolme.egitim
    satir, ky = kisi_periyot(panel, idx, panel.etiket_k[idx], panel.izlenen_k[idx], ufuk)
    k, y = ky[:, 0], ky[:, 1]

    # Her satir 1'den baslar ve olay/sansur haftasinda biter.
    ilk = np.zeros(idx.size, dtype=int)
    np.maximum.at(ilk, satir, k)
    olay_k = panel.etiket_k[idx]
    beklenen = np.where((olay_k > 0) & (olay_k <= ufuk), olay_k,
                        np.minimum(panel.izlenen_k[idx], ufuk))
    assert (ilk == beklenen).all()
    # Olay yalnizca son periyotta ve yalnizca olayli satirlarda 1 olur.
    olayli = np.zeros(idx.size, dtype=int)
    np.add.at(olayli, satir, y)
    assert set(np.unique(olayli)) <= {0, 1}
    assert (olayli == ((olay_k > 0) & (olay_k <= ufuk)).astype(int)).all()
    assert k.min() >= 1 and k.max() <= ufuk


def test_hayatta_kalma_egrisi_tutarli(hazirlik):
    cfg, _, _, panel, bolme = hazirlik
    hz = HazardTahmincisi(cfg, panel).egit(panel, bolme.egitim,
                                           panel.etiket_k[bolme.egitim],
                                           panel.izlenen_k[bolme.egitim])
    ornek = bolme.test[:2000]
    h = hz.hazard_egrisi(panel, ornek)
    assert ((h >= 0) & (h <= 1)).all()
    S = np.cumprod(1.0 - h, axis=1)
    assert (np.diff(S, axis=1) <= 1e-12).all(), "hayatta kalma egrisi artamaz"
    t = hz.tahmin(panel, ornek)
    ufuk = cfg.tukenme.hedef.ufuk_hafta
    assert ((t.tukenme_hafta >= 0) & (t.tukenme_hafta <= ufuk)).all()
    assert ((t.olasilik >= 0) & (t.olasilik <= 1)).all()


def test_kural_literal_n_config_ten_gelir(hazirlik):
    cfg, _, _, panel, bolme = hazirlik
    y = gozlemlenebilir_olay(panel, bolme.egitim, cfg.tukenme.hedef.karar_ufku_hafta)
    literal = KuralTahmincisi(cfg, panel, ikili=True).egit(panel, bolme.egitim, y)
    ayarli = KuralTahmincisi(cfg, panel, ikili=False).egit(panel, bolme.egitim, y)
    assert literal.n_gun == cfg.tukenme.taban_kural.son_n_gun
    assert ayarli.n_gun in cfg.tukenme.taban_kural.n_gun_adaylari


def test_kalibrator_kova_frekansini_ogrenir():
    rng = np.random.default_rng(0)
    skor = rng.random(20000)
    y = (rng.random(20000) < skor).astype(int)      # P(y=1) = skor
    kal = KovaKalibratoru(10).egit(skor, y)
    p = kal.uygula(np.array([0.05, 0.5, 0.95]))
    assert p[0] < p[1] < p[2]
    assert abs(p[1] - 0.5) < 0.05
    assert mt.beklenen_kalibrasyon_hatasi(y, kal.uygula(skor), 10) < 0.02


def test_sabit_skor_kazanc_uretmez():
    """Beraberlikte ust dilim kazanci 1'e yakin olmali (sahte kazanc olmasin)."""
    rng = np.random.default_rng(1)
    y = (rng.random(50000) < 0.1).astype(int)
    y[:5000] = 1                                    # basa yigilmis olaylar
    assert abs(mt.ust_dilim_kazanci(y, np.zeros_like(y, dtype=float)) - 1.0) < 0.25
    assert mt.guvenli_auc(y, np.zeros_like(y, dtype=float)) == 0.5


def test_oracle_listeden_dusmeyi_tukenme_saymiyor(hazirlik):
    cfg, kok, _, panel, bolme = hazirlik
    o = Oracle("t", kok=kok).etiketle(
        panel.anahtar["eczane_id"].to_numpy()[bolme.test],
        panel.anahtar["sku_id"].to_numpy()[bolme.test],
        panel.origin[bolme.test], cfg.tukenme.hedef.ufuk_hafta)
    assert not (o.olay & o.rakip_sansur).any()
    assert o.rakip_sansur.any(), "cesit churn'u varken hic rakip sansur yok - suphe"
    assert (o.tukenme_k[o.olay] > 0).all()
    assert (o.izlenen_k <= cfg.tukenme.hedef.ufuk_hafta).all()


def test_zaman_bolmesi_tamponlu(hazirlik):
    cfg, _, _, panel, bolme = hazirlik
    egitim_son = panel.origin[bolme.egitim].max()
    test_ilk = panel.origin[bolme.test].min()
    assert test_ilk - egitim_son >= cfg.tukenme.hedef.ufuk_hafta
    assert set(np.unique(panel.origin[bolme.egitim])).isdisjoint(
        set(np.unique(panel.origin[bolme.test])))


def test_kosu_tekrar_uretilebilir(tmp_path):
    """Ayni config + ayni seed -> ayni metrikler (CLAUDE.md 5)."""
    cfg = config_yukle("fast", gecersiz_kilma=TEST_KNOBLARI)
    a = kosu_yap(cfg, "tekrar_a", {}, veri_tut=False, tahmin_yaz=False, kok=tmp_path)
    b = kosu_yap(cfg, "tekrar_b", {}, veri_tut=False, tahmin_yaz=False, kok=tmp_path)
    # NaN == NaN yanlistir; tanimsiz metrik (orn. sifir teklifli politikanin
    # teklif basina marji) iki kosuda da tanimsiz olmali. Karsilastirma bu
    # yuzden NaN-duyarli yapilir, metrik listesi degil.
    assert set(a["duz"]) == set(b["duz"])
    farkli = [k for k, v in a["duz"].items()
              if not (v == b["duz"][k]
                      or (isinstance(v, float) and np.isnan(v)
                          and isinstance(b["duz"][k], float) and np.isnan(b["duz"][k])))]
    assert not farkli, farkli
    assert a["dunya_hash"] == b["dunya_hash"]


def test_blok_bootstrap_araligi_satir_bootstrapindan_genis():
    """Ortusen satirlar bagimsiz sayilirsa aralik OLDUGUNDAN DAR cikar.

    Sentetik kurulum: her hucrenin 10 satiri birebir ayni. Satir bazli
    bootstrap bunlari 10 bagimsiz gozlem sanir; blok bootstrap saymaz.
    """
    rng = np.random.default_rng(0)
    n_hucre, tekrar_satir = 200, 10
    hucre = np.repeat(np.arange(n_hucre), tekrar_satir)
    y = np.repeat((rng.random(n_hucre) < 0.3).astype(int), tekrar_satir)
    a = np.repeat(rng.random(n_hucre), tekrar_satir)
    b = np.repeat(rng.random(n_hucre), tekrar_satir)

    _, s_alt, s_ust = mt.bootstrap_farki(y, a, b, mt.guvenli_auc, 200, 1)
    _, b_alt, b_ust = mt.bootstrap_farki(y, a, b, mt.guvenli_auc, 200, 1, grup=hucre)
    assert (b_ust - b_alt) > 1.5 * (s_ust - s_alt)
