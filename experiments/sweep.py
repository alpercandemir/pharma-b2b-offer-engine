"""Knob taramasi: deger listesi x seed -> paralel kosu -> karsilastirma tablosu.

SPEC 5b.2'nin zorunlu komutu:

    python -m experiments.sweep --knob <knob> --values a,b,c --seeds 5

Bir degeri N seed'le kosmanin sebebi: tek seed'de metrik farklari gurultuden
ayirt edilemiyor. Tablodaki her hucre ortalama +- standart sapmadir ve
`--seeds 1` verilirse sapma sutunu yazilmaz (yaniltmasin diye).

Seed disiplini: seed j, profildeki temel_seed + j'dir. Ayni j farkli knob
degerlerinde AYNI dunyayi uretmez (knob dunyayi da degistirebilir) ama ayni
knob degerinde her zaman ayni dunyayi uretir - tekrar uretilebilirlik burada.

Cikti: experiments/runs/_sweep_<knob>_<zaman>/
    tablo.csv     her (deger, seed) icin duz metrikler
    ozet.csv      deger basina ortalama / sapma
    sweep.png     secili metriklerin knob boyunca hareketi
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from core.config import load_config  # noqa: E402
from experiments.run import (ASAMALAR, KOSU_DIZINI, deger_coz,  # noqa: E402
                             knob_ayristir, kosu_yap)

# Tabloda one cikarilan metrikler. Tam metrik seti tablo.csv'de duruyor;
# bunlar M2 ve M3'un cikis kriterini tasiyan olculer. Asamaya gore hangileri
# uretilirse o gosterilir (ozet tablosunda olmayan metrik atlanir).
ONE_CIKAN = [
    # M2
    "hazard.auc", "kural_ikili.auc", "defter.auc", "teshis_oracle_ozellik.auc",
    "hazard.pr_auc", "hazard.ust_dilim_kazanci", "hazard.mae_gun",
    "sabit.mae_gun", "hazard.kalibrasyon_hatasi", "hazard.auc_gozlemlenebilir",
    "panel.gercek_tukenme_taban_orani", "panel.olcum_satiri",
    # M3
    "aday.hibrit.recall", "aday.hibrit.yeni_recall", "aday.hibrit.oracle_recall",
    "aday.tekrar.recall", "aday.cf.recall", "aday.populerlik.recall",
    "aday.hibrit.kapsama", "aday.hibrit.soguk_eczane_recall",
    "kisit.havuz_recall", "kisit.veto_sonrasi_recall", "kisit.liste_recall",
    "kisit.veto_orani", "kisit.ust_dilim_veto_orani", "kisit.liste_satiri",
    # M4 - cikis kriteri: marj farki ve nerede yandigi
    "m4.marj_farki_tl", "m4.marj_farki_yuzde", "m4.oracle_marj_farki_tl",
    "m4.tahmin_hatasinin_bedeli_tl",
    "m4.propensity.artimsal_marj", "m4.uplift_x.artimsal_marj",
    "m4.uplift_t.artimsal_marj", "m4.oracle_uplift.artimsal_marj",
    "m4.oracle_propensity.artimsal_marj", "m4.propensity_ham.artimsal_marj",
    "m4.m3_sabit_kampanya.artimsal_marj",
    "m4.propensity.yakilan_marj", "m4.uplift_x.yakilan_marj",
    "m4.propensity_ham.yakilan_marj",
    "m4.propensity.negatif_teklif_orani", "m4.uplift_x.negatif_teklif_orani",
    "m4.uplift_x.teklif_sayisi", "m4.uplift_x.ortalama_mf",
    "m4.cate.pehe_t", "m4.cate.pehe_x", "m4.cate.sira_kor_t", "m4.cate.sira_kor_x",
    "m4.heterojenlik.cate_sapmasi", "m4.heterojenlik.farkli_karar_orani",
    "m4.destek.propensity_min", "m4.destek.kol_orneklemi_min",
    # M5 - cikis kriteri (a): kit stok altinda tahsis
    "m5.ranking_only.karsilanmayan_adet", "m5.lp.karsilanmayan_adet",
    "m5.ranking_only.stockout_sayisi", "m5.lp.stockout_sayisi",
    "m5.a.karsilanmayan_farki", "m5.a.stockout_farki", "m5.a.net_marj_farki",
    "m5.ranking_only.net_marj", "m5.lp.net_marj", "m5.lp.brut_marj",
    # M5 - cikis kriteri (b): miad rejimi
    "m5.lp.imha_adet_temizlik", "m5.kor_iskonto.imha_adet_temizlik",
    "m5.hedefli_temizlik.imha_adet_temizlik",
    "m5.lp.iade_adet", "m5.kor_iskonto.iade_adet", "m5.hedefli_temizlik.iade_adet",
    "m5.kor_iskonto.net_marj", "m5.hedefli_temizlik.net_marj",
    "m5.b.imha_temizlik_kor_farki", "m5.b.imha_temizlik_hedefli_farki",
    "m5.b.iade_kor_farki", "m5.b.iade_hedefli_farki",
    "m5.b.net_marj_kor_farki", "m5.b.net_marj_hedefli_farki",
    "m5.lp.memnuniyet_proxy", "m5.kor_iskonto.memnuniyet_proxy",
    "m5.hedefli_temizlik.memnuniyet_proxy",
    "m5.b.memnuniyet_kor_farki", "m5.b.memnuniyet_hedefli_farki",
    # M5 - D9 gölge fiyatlari
    "m5.golge.hedefli_temizlik.negatif_golge_lot_orani",
    "m5.golge.hedefli_temizlik.golge_asgari",
    "m5.golge.hedefli_temizlik.negatif_golge_adet",
    "m5.golge.lp.golge_ortalama", "m5.golge.lp.negatif_golge_lot_orani",
    "m5.hedefli_temizlik.ortalama_mf_temizlik",
    "m5.hedefli_temizlik.teklif_sayisi_temizlik",
    "m5.lp.butunluk_acigi", "m5.hedefli_temizlik.butunluk_acigi",
    # M6 - cikis kriteri: offline tahmin vs kapali dongu, ve sapmanin kaynagi
    "m6.ozdeslik.ozdeslik_sapma_yuzde",
    "m6.offline.uplift_x.oracle", "m6.offline.uplift_x.ips",
    "m6.offline.uplift_x.snips", "m6.offline.uplift_x.dr",
    "m6.denetim.uplift_x.ips.sapma_yuzde", "m6.denetim.uplift_x.dr.sapma_yuzde",
    "m6.denetim.propensity.ips.sapma_yuzde", "m6.denetim.agresif.ips.sapma_yuzde",
    "m6.ayristirma.uplift_x.varyans", "m6.ayristirma.uplift_x.kirpma",
    "m6.ayristirma.uplift_x.propensity", "m6.ayristirma.uplift_x.artik",
    "m6.ayristirma.uplift_x.ortusme_kor_deger_payi",
    "m6.ayristirma.uplift_x.ekstrap_zayif", "m6.ayristirma.uplift_x.ekstrap_guclu",
    "m6.teshis.uplift_x.ess_orani", "m6.teshis.uplift_x.eslesme_orani",
    "m6.teshis.uplift_x.agirlik_azami", "m6.teshis.uplift_x.kirpilan_kutle_orani",
    "m6.teshis.uplift_x.ortusme_ihlali_orani",
    "m6.sapma_sd.uplift_x.ips_sapma", "m6.sapma_sd.uplift_x.dr_sapma",
    "m6.sapma_sd.uplift_x.snips_sapma",
    "m6.propensity.kalibrasyon_hatasi", "m6.propensity.ortalama_mutlak_hata",
    # M6 - kapali dongu (ufka gore isaret donmesi)
    "m6.online.agresif.artimsal_yuzde_son", "m6.online.agresif_vade.artimsal_yuzde_son",
    "m6.online.uplift_x.artimsal_yuzde_son", "m6.online.propensity.artimsal_yuzde_son",
    "m6.online.lp.artimsal_yuzde_son",
    "m6.online.agresif_vade.artimsal_son", "m6.online.uplift_x.artimsal_son",
    "m6.online.uplift_x.iade_adet", "m6.online.agresif_vade.iade_adet",
    "m6.gecikmeli.uplift_x.terminal_riskli_pay@4",
    "m6.gecikmeli.agresif.terminal_riskli_pay@4",
    "m6.online.uplift_x.hafta_sayisi",
    "m6.gecikmeli.uplift_x.kanibalizm_organik_siparis_farki",
    "m6.gecikmeli.agresif_vade.kanibalizm_organik_siparis_farki",
    "m6.gecikmeli.uplift_x.sow_son_fark",
    "m6.rollout.egitim_ortusmesi_hafta",
    # M7 - senaryo katmani (D3): rejim basina politikanin ne onerdigi
    "m7.senaryo.baz.teklif_sayisi", "m7.senaryo.yuksek.teklif_sayisi",
    "m7.senaryo.sok.teklif_sayisi",
    "m7.senaryo.baz.artimsal_marj", "m7.senaryo.yuksek.artimsal_marj",
    "m7.senaryo.sok.artimsal_marj",
    "m7.senaryo.baz.ortalama_mf", "m7.senaryo.sok.ortalama_mf",
    "m7.senaryo.baz.ortalama_vade", "m7.senaryo.sok.ortalama_vade",
    "m7.senaryo.sok.bedava_adet", "m7.senaryo.baz.bedava_adet",
    "m7.senaryo.sok.erteleme_tl_adet", "m7.senaryo.yuksek.erteleme_tl_adet",
    "m7.senaryo.sok.bekleyemeyen_teklif_pay",
    "m7.senaryo.yuksek.bekleyemeyen_teklif_pay",
    "m7.fark.sok.kol_degisen", "m7.fark.yuksek.kol_degisen",
    "m7.fark.sok.teklif_farki", "m7.fark.yuksek.teklif_farki",
    "m7.fark.sok.artimsal_marj_farki", "m7.fark.yuksek.artimsal_marj_farki",
    "m7.fark.sok.teklifden_cikan", "m7.fark.sok.teklife_giren",
    # M7 - harness (cikis kriteri): temiz vaka temiz mi, mutant yakalandi mi
    "m7.harness.temiz_bulgu", "m7.harness.mutant_yakalanan",
    "m7.harness.mutant_sayisi", "m7.harness.kalan",
]
# Grafige giren metrikler.
GRAFIK_METRIKLERI = ["hazard.auc", "kural_ikili.auc", "defter.auc",
                     "teshis_oracle_ozellik.auc", "hazard.mae_gun"]
GRAFIK_METRIKLERI_M3 = ["aday.hibrit.recall", "aday.hibrit.yeni_recall",
                        "kisit.havuz_recall", "kisit.veto_sonrasi_recall",
                        "kisit.liste_recall", "kisit.veto_orani"]
# M4 grafigi: solda politika degerleri, sagda marj farki + yakilan marj.
GRAFIK_POLITIKALARI_M4 = ["m4.propensity.artimsal_marj", "m4.uplift_x.artimsal_marj",
                          "m4.oracle_propensity.artimsal_marj",
                          "m4.oracle_uplift.artimsal_marj"]
GRAFIK_FARK_M4 = ["m4.marj_farki_tl", "m4.oracle_marj_farki_tl"]
# M5 grafigi: solda (a) kit stok karsilastirmasi, sagda (b) miad rejimi.
GRAFIK_A_M5 = ["m5.ranking_only.karsilanmayan_adet", "m5.lp.karsilanmayan_adet",
               "m5.ranking_only.stockout_sayisi", "m5.lp.stockout_sayisi"]
GRAFIK_B_M5 = ["m5.b.imha_temizlik_kor_farki", "m5.b.imha_temizlik_hedefli_farki",
               "m5.b.iade_kor_farki", "m5.b.iade_hedefli_farki"]
# M6 grafigi: solda offline tahmin vs oracle (tahminci sapmasi), sagda kapali
# dongu artimsal degeri (ufka gore isaret donmesi burada gorunur).
GRAFIK_OFFLINE_M6 = ["m6.offline.uplift_x.oracle", "m6.offline.uplift_x.ips",
                     "m6.offline.uplift_x.snips", "m6.offline.uplift_x.dr"]
GRAFIK_ONLINE_M6 = ["m6.online.uplift_x.artimsal_yuzde_son",
                    "m6.online.propensity.artimsal_yuzde_son",
                    "m6.online.agresif.artimsal_yuzde_son",
                    "m6.online.agresif_vade.artimsal_yuzde_son",
                    "m6.online.lp.artimsal_yuzde_son"]
# M7 grafigi: solda rejim basina teklif hacmi ve artimsal marj (senaryo
# katmaninin kendisi), sagda harness'in canliligi (mutant yakalama).
GRAFIK_SENARYO_M7 = ["m7.senaryo.baz.artimsal_marj", "m7.senaryo.yuksek.artimsal_marj",
                     "m7.senaryo.sok.artimsal_marj"]
GRAFIK_HACIM_M7 = ["m7.senaryo.baz.teklif_sayisi", "m7.senaryo.yuksek.teklif_sayisi",
                   "m7.senaryo.sok.teklif_sayisi"]
GRAFIK_HARNESS_M7 = ["m7.harness.mutant_yakalanan", "m7.harness.mutant_sayisi",
                     "m7.harness.temiz_bulgu"]
# Sifir olmasi gereken kisit ihlalleri. Sweep bunlari HER kosuda kontrol eder:
# bir knob kombinasyonu kisiti deliyorsa tarama tablosu degil, uyari cikar.
IHLAL_ONEKI = "ihlal."


def _tek_kosu(gorev: tuple) -> dict:
    profil, knob, deger, seed, sabit, kok, asamalar = gorev
    gecersiz = dict(sabit)
    if knob is not None:
        gecersiz[knob] = deger_coz(str(deger))
    gecersiz["profil.temel_seed"] = seed
    cfg = load_config(profil, gecersiz_kilma=gecersiz)
    ad = f"_sweep/{_temiz(str(deger))}_s{seed}"
    # Her surec tek is parcacigi kullanir. Aksi halde N surec x M is parcacigi
    # cekirdek sayisini asiyor ve paralellik kosuyu HIZLANDIRMAK yerine
    # yavaslatiyor (olculdu: 6 kosu 147 sn -> 25 sn).
    with threadpool_limits(limits=1):
        icerik = kosu_yap(cfg, ad, gecersiz, veri_tut=False, tahmin_yaz=False,
                          kok=kok, asamalar=asamalar)
    return {"deger": str(deger), "seed": seed, **icerik["duz"]}


def _temiz(metin: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]", "_", metin)


def tarama(profil: str, knob: str, degerler: list[str], seedler: int,
           sabit: dict, isci: int, cikti: Path,
           asamalar: tuple[str, ...]) -> pl.DataFrame:
    temel = load_config(profil).profil.temel_seed
    gorevler = [(profil, knob, d, temel + j, sabit, cikti / "_ham", asamalar)
                for d in degerler for j in range(seedler)]
    satirlar: list[dict] = []
    # Ilerleme satirinda gosterilen metrik: kosulan EN ILERI asamanin
    # basligi. Zincirin sonunda m6 varsayilani duruyordu ve `--asama m7`
    # o metrigi hic hesaplamadigi icin her satir `nan` basiyordu -- kosu
    # ilerledigini gosteren tek gosterge okunamaz haldeydi.
    izlenen = ("hazard.auc" if "m2" in asamalar else
               "aday.hibrit.recall" if "m3" in asamalar else
               "m4.marj_farki_tl" if "m4" in asamalar else
               "m5.a.karsilanmayan_farki" if "m5" in asamalar else
               "m6.denetim.uplift_x.dr.sapma_yuzde" if "m6" in asamalar else
               "m7.fark.sok.kol_degisen")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=isci) as havuz:
        for i, satir in enumerate(havuz.map(_tek_kosu, gorevler), start=1):
            satirlar.append(satir)
            print(f"  [{i}/{len(gorevler)}] deger={satir['deger']} seed={satir['seed']} "
                  f"{izlenen}={satir.get(izlenen, float('nan')):.3f}", flush=True)
    print(f"toplam {len(gorevler)} kosu, {time.perf_counter() - t0:.1f} sn")
    return pl.DataFrame(satirlar)


def ihlal_denetimi(tablo: pl.DataFrame) -> list[str]:
    """Sifir olmasi gereken kisit ihlalleri her kosuda sifir mi.

    Bir sweep degeri kisiti deliyorsa bunu tabloyu okuyan insanin fark
    etmesini beklemek yanlis: burada acikca bagirilir.
    """
    bulgular = []
    for kolon in tablo.columns:
        if not kolon.startswith(IHLAL_ONEKI):
            continue
        v = tablo[kolon].to_numpy().astype(float)
        if np.nansum(v) > 0:
            kotu = tablo.filter(pl.col(kolon) > 0)["deger"].unique().to_list()
            bulgular.append(f"{kolon}: toplam {int(np.nansum(v))} ihlal, "
                            f"deger(ler)={sorted(kotu)}")
    return bulgular


def ozetle(tablo: pl.DataFrame, degerler: list[str], seedler: int) -> pl.DataFrame:
    metrikler = [k for k in tablo.columns if k not in ("deger", "seed")]
    satirlar = []
    for d in degerler:
        alt = tablo.filter(pl.col("deger") == str(d))
        satir: dict[str, object] = {"deger": str(d), "kosu": alt.height}
        for m in metrikler:
            v = alt[m].to_numpy().astype(float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            satir[m] = round(float(v.mean()), 4)
            if seedler > 1 and v.size > 1:
                satir[f"{m}_sd"] = round(float(v.std(ddof=1)), 4)
        satirlar.append(satir)
    return pl.DataFrame(satirlar)


def grafik(ozet: pl.DataFrame, knob: str, yol: Path, seedler: int) -> None:
    if any(m in ozet.columns for m in GRAFIK_SENARYO_M7 + GRAFIK_HARNESS_M7):
        _grafik_m7(ozet, knob, yol.with_name("sweep_m7.png"), seedler)
    if any(m in ozet.columns for m in GRAFIK_OFFLINE_M6 + GRAFIK_ONLINE_M6):
        _grafik_m6(ozet, knob, yol.with_name("sweep_m6.png"), seedler)
    if any(m in ozet.columns for m in GRAFIK_A_M5 + GRAFIK_B_M5):
        _grafik_m5(ozet, knob, yol.with_name("sweep_m5.png"), seedler)
    if any(m in ozet.columns for m in GRAFIK_FARK_M4):
        _grafik_m4(ozet, knob, yol.with_name("sweep_m4.png"), seedler)
    mevcut = [m for m in GRAFIK_METRIKLERI if m in ozet.columns]
    if not mevcut:
        return _grafik_m3(ozet, knob, yol, seedler)
    x = np.arange(ozet.height)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in mevcut:
        if m.endswith(".auc"):
            sd = ozet[f"{m}_sd"].to_numpy() if f"{m}_sd" in ozet.columns else None
            ax[0].errorbar(x, ozet[m].to_numpy(), yerr=sd, marker="o", ms=4,
                           capsize=3, label=m)
    ax[0].axhline(0.5, c="k", ls="--", lw=1)
    ax[0].set_ylabel("AUC (gercek tukenme, karar ufku)")
    ax[0].set_title(f"{knob} boyunca ayirt etme gucu")
    ax[0].legend(fontsize=8)
    for m in [k for k in mevcut if k.endswith("mae_gun")]:
        sd = ozet[f"{m}_sd"].to_numpy() if f"{m}_sd" in ozet.columns else None
        ax[1].errorbar(x, ozet[m].to_numpy(), yerr=sd, marker="s", ms=4,
                       capsize=3, label=m)
    if "sabit.mae_gun" in ozet.columns:
        ax[1].plot(x, ozet["sabit.mae_gun"].to_numpy(), "k:", label="sabit.mae_gun")
    ax[1].set_ylabel("MAE (gun)")
    ax[1].set_title("Kirpilmis tukenme suresi hatasi")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(ozet["deger"].to_list())
        a.set_xlabel(knob)
    fig.suptitle(f"sweep: {knob}  ({seedler} seed)", fontsize=10)
    plt.tight_layout()
    plt.savefig(yol, dpi=110)
    plt.close()


def _grafik_m3(ozet: pl.DataFrame, knob: str, yol: Path, seedler: int) -> None:
    """M3 taramasi: solda recall ailesi, sagda kisit katmaninin bedeli."""
    mevcut = [m for m in GRAFIK_METRIKLERI_M3 if m in ozet.columns]
    if not mevcut:
        return
    x = np.arange(ozet.height)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in mevcut:
        hedef = ax[1] if m.startswith("kisit.") else ax[0]
        sd = ozet[f"{m}_sd"].to_numpy() if f"{m}_sd" in ozet.columns else None
        hedef.errorbar(x, ozet[m].to_numpy(), yerr=sd, marker="o", ms=4,
                       capsize=3, label=m)
    ax[0].set_ylabel("recall@K (gozlemlenebilir hedef)")
    ax[0].set_title(f"{knob} boyunca aday havuzu")
    ax[1].set_ylabel("oran")
    ax[1].set_title("Kisit katmani: havuz -> veto -> liste")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(ozet["deger"].to_list())
        a.set_xlabel(knob)
        a.legend(fontsize=8)
    fig.suptitle(f"sweep: {knob}  ({seedler} seed)", fontsize=10)
    plt.tight_layout()
    plt.savefig(yol, dpi=110)
    plt.close()


def _grafik_m4(ozet: pl.DataFrame, knob: str, yol: Path, seedler: int) -> None:
    """M4 taramasi: solda politika degerleri, sagda marj farki.

    Sag panelde IKI cizgi var ve ikisinin ayrimi M4'un ana okumasi:
      `oracle_marj_farki_tl`  amac fonksiyonunun TEK BASINA yarattigi fark
                              (model hatasi sifir)
      `marj_farki_tl`         tahmin edilen CATE ile FIILEN elde edilen fark
    Ikisi arasindaki bosluk tahmin hatasinin bedelidir.
    """
    x = np.arange(ozet.height)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in [k for k in GRAFIK_POLITIKALARI_M4 if k in ozet.columns]:
        sd = ozet[f"{m}_sd"].to_numpy() if f"{m}_sd" in ozet.columns else None
        ax[0].errorbar(x, ozet[m].to_numpy(), yerr=sd, marker="o", ms=4,
                       capsize=3, label=m.replace("m4.", "").replace(".artimsal_marj", ""))
    ax[0].axhline(0, c="k", ls="--", lw=1)
    ax[0].set_ylabel("artimsal marj (TL)")
    ax[0].set_title(f"{knob} boyunca politika degeri", fontsize=9)
    ax[0].legend(fontsize=8)

    for m in [k for k in GRAFIK_FARK_M4 if k in ozet.columns]:
        sd = ozet[f"{m}_sd"].to_numpy() if f"{m}_sd" in ozet.columns else None
        ax[1].errorbar(x, ozet[m].to_numpy(), yerr=sd, marker="s", ms=4,
                       capsize=3, label=m.replace("m4.", ""))
    ax[1].axhline(0, c="k", ls="--", lw=1)
    ax[1].set_ylabel("uplift - propensity (TL)")
    ax[1].set_title("Marj farki: amac fonksiyonu vs fiilen elde edilen", fontsize=9)
    ax[1].legend(fontsize=8)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(ozet["deger"].to_list())
        a.set_xlabel(knob)
    fig.suptitle(f"sweep (M4): {knob}  ({seedler} seed)", fontsize=10)
    plt.tight_layout()
    plt.savefig(yol, dpi=110)
    plt.close()


def _grafik_m5(ozet: pl.DataFrame, knob: str, yol: Path, seedler: int) -> None:
    """M5 taramasi: (a) karsilanmayan talep / stockout, (b) imha ve iade farki.

    (b) panelinde FARK cizilir, seviye degil: imha seviyesi yapisal fazla
    stoktan besleniyor ve politikalar arasinda neredeyse sabit; okunabilir
    olan, politikanin FIILEN oynattigi miktar.
    """
    x = np.arange(ozet.height)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in [k for k in GRAFIK_A_M5 if k in ozet.columns]:
        sd = ozet[f"{m}_sd"].to_numpy() if f"{m}_sd" in ozet.columns else None
        stil = "o-" if "karsilanmayan" in m else "s--"
        ax[0].errorbar(x, ozet[m].to_numpy(), yerr=sd, fmt=stil, ms=4, capsize=3,
                       label=m.replace("m5.", ""))
    ax[0].set_ylabel("adet / teklif")
    ax[0].set_title(f"(a) {knob} boyunca kit stok altinda karsilama", fontsize=9)
    ax[0].legend(fontsize=7)

    for m in [k for k in GRAFIK_B_M5 if k in ozet.columns]:
        sd = ozet[f"{m}_sd"].to_numpy() if f"{m}_sd" in ozet.columns else None
        stil = "o-" if "imha" in m else "s--"
        ax[1].errorbar(x, ozet[m].to_numpy(), yerr=sd, fmt=stil, ms=4, capsize=3,
                       label=m.replace("m5.b.", ""))
    ax[1].axhline(0, c="k", ls="--", lw=1)
    ax[1].set_ylabel("temizlik yok'a gore fark (adet)")
    ax[1].set_title("(b) miad rejimi: imha kazanci vs iade bedeli", fontsize=9)
    ax[1].legend(fontsize=7)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(ozet["deger"].to_list())
        a.set_xlabel(knob)
    fig.suptitle(f"sweep (M5): {knob}  ({seedler} seed)", fontsize=10)
    plt.tight_layout()
    plt.savefig(yol, dpi=110)
    plt.close()


def _grafik_m6(ozet: pl.DataFrame, knob: str, yol: Path, seedler: int) -> None:
    """M6 taramasi: solda tahminciler ORACLE ile birlikte, sagda kapali dongu.

    Sol panelde oracle cizgisi KALIN ve siyah: tahmincilerin ondan sapmasi
    gozle olculebilsin diye. Bir tahminci oracle'a yapisik gidiyorsa o knob
    degerinde offline degerlendirme guvenilirdir; ayrisiyorsa raporun
    ayristirma tablosuna bakilir.

    Sag panelde sifir cizgisi var cunku okunacak sey SEVIYE degil ISARET:
    ufuk boyunca artimsal degerin isaret degistirdigi nokta SPEC 5'in
    "kisa ufukta kazanir, uzun ufukta kaybeder" cumlesinin ta kendisi.
    """
    x = np.arange(ozet.height)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in [k for k in GRAFIK_OFFLINE_M6 if k in ozet.columns]:
        oracle_mi = m.endswith(".oracle")
        sd = ozet[f"{m}_sd"].to_numpy() if f"{m}_sd" in ozet.columns else None
        ax[0].errorbar(x, ozet[m].to_numpy(), yerr=sd, marker="o", ms=4, capsize=3,
                       lw=2.5 if oracle_mi else 1.2, color="k" if oracle_mi else None,
                       zorder=3 if oracle_mi else 2,
                       label=m.replace("m6.offline.uplift_x.", ""))
    ax[0].set_ylabel("V(uplift_x)  TL/satir")
    ax[0].set_title(f"{knob} boyunca offline tahmin vs oracle", fontsize=9)
    ax[0].legend(fontsize=8)

    for m in [k for k in GRAFIK_ONLINE_M6 if k in ozet.columns]:
        sd = ozet[f"{m}_sd"].to_numpy() if f"{m}_sd" in ozet.columns else None
        ax[1].errorbar(x, ozet[m].to_numpy(), yerr=sd, marker="s", ms=4, capsize=3,
                       label=m.replace("m6.online.", "").replace(".artimsal_yuzde_son", ""))
    ax[1].axhline(0, c="k", ls="--", lw=1)
    ax[1].set_ylabel("kapali dongu artimsal (%, tabana gore)")
    ax[1].set_title("Ufuk sonundaki gerceklesen deger", fontsize=9)
    ax[1].legend(fontsize=8)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(ozet["deger"].to_list())
        a.set_xlabel(knob)
    fig.suptitle(f"sweep (M6): {knob}  ({seedler} seed)", fontsize=10)
    plt.tight_layout()
    plt.savefig(yol, dpi=110)
    plt.close()


def _grafik_m7(ozet: pl.DataFrame, knob: str, yol: Path, seedler: int) -> None:
    """M7 taramasi: solda rejimlerin AYRISMASI, sagda denetcinin canliligi.

    Sol panelde okunacak sey seviye degil ARALIK: uc rejim cizgisi
    birbirine yapisiyorsa senaryo katmani o knob degerinde olu demektir
    (config yuklemesindeki notrluk kilidi yalnizca TANIMIN olu olmadigini
    garanti eder, SONUCUN ayristigini degil).

    Sag panelde `mutant_yakalanan` ile `mutant_sayisi` ust uste binmek
    ZORUNDA. Ayrilan her nokta bir denetcinin o ayarda kor kaldigi
    anlamina gelir ve tarama tablosu degil, o kontrol okunmalidir.
    """
    x = np.arange(ozet.height)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in [k for k in GRAFIK_SENARYO_M7 if k in ozet.columns]:
        sd = ozet[f"{m}_sd"].to_numpy() if f"{m}_sd" in ozet.columns else None
        ax[0].errorbar(x, ozet[m].to_numpy(), yerr=sd, marker="o", ms=4, capsize=3,
                       label=m.replace("m7.senaryo.", "").replace(".artimsal_marj", ""))
    ikiz = ax[0].twinx()
    for m in [k for k in GRAFIK_HACIM_M7 if k in ozet.columns]:
        ikiz.plot(x, ozet[m].to_numpy(), ls=":", marker="x", ms=4, alpha=0.6,
                  label=m.replace("m7.senaryo.", "").replace(".teklif_sayisi", ""))
    ax[0].set_ylabel("beklenen artimsal marj (TL)")
    ikiz.set_ylabel("teklif sayisi (kesikli)")
    ax[0].set_title(f"{knob} boyunca rejimlerin ayrismasi", fontsize=9)
    ax[0].legend(fontsize=8, loc="upper left")

    for m in [k for k in GRAFIK_HARNESS_M7 if k in ozet.columns]:
        ax[1].plot(x, ozet[m].to_numpy(), marker="s", ms=4,
                   label=m.replace("m7.harness.", ""))
    ax[1].set_ylabel("vaka / bulgu sayisi")
    ax[1].set_title("Denetci canli mi (yakalanan = mutant sayisi olmali)", fontsize=9)
    ax[1].legend(fontsize=8)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(ozet["deger"].to_list())
        a.set_xlabel(knob)
    fig.suptitle(f"sweep (M7): {knob}  ({seedler} seed)", fontsize=10)
    plt.tight_layout()
    plt.savefig(yol, dpi=110)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", required=True, help="config yolu, orn. tukenme.hedef.karar_ufku_hafta")
    ap.add_argument("--values", required=True, help="virgulle ayrilmis deger listesi")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--profil", default="fast")
    ap.add_argument("--sabit", action="append", default=[],
                    help="taramada sabit tutulan knob: yol=deger, tekrarlanabilir")
    ap.add_argument("--isci", type=int, default=4, help="paralel surec sayisi")
    ap.add_argument("--ad", default=None)
    ap.add_argument("--asama", default="m2,m3,m4,m5",
                    help=f"kosulacak asamalar, virgulle: {','.join(ASAMALAR)}")
    args = ap.parse_args()

    asamalar = tuple(a.strip() for a in args.asama.split(",") if a.strip())
    bilinmeyen = set(asamalar) - set(ASAMALAR)
    if bilinmeyen:
        raise SystemExit(f"bilinmeyen asama: {sorted(bilinmeyen)} (gecerli: {ASAMALAR})")

    degerler = args.values.split(",")
    sabit = knob_ayristir(args.sabit)
    ad = args.ad or f"_sweep_{_temiz(args.knob)}_{time.strftime('%Y%m%d-%H%M%S')}"
    cikti = KOSU_DIZINI / ad
    cikti.mkdir(parents=True, exist_ok=True)

    print(f"sweep: {args.knob} = {degerler} | profil={args.profil} | seed={args.seeds} "
          f"| asama={asamalar} | {len(degerler) * args.seeds} kosu")
    tablo = tarama(args.profil, args.knob, degerler, args.seeds, sabit, args.isci,
                   cikti, asamalar)
    ozet = ozetle(tablo, degerler, args.seeds)
    tablo.write_csv(cikti / "tablo.csv")
    ozet.write_csv(cikti / "ozet.csv")
    grafik(ozet, args.knob, cikti / "sweep.png", args.seeds)
    (cikti / "manifest.json").write_text(json.dumps({
        "knob": args.knob, "degerler": degerler, "seeds": args.seeds,
        "profil": args.profil, "asamalar": list(asamalar),
        "sabit": {k: str(v) for k, v in sabit.items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    gosterilecek = ["deger"] + [m for m in ONE_CIKAN if m in ozet.columns]
    if args.seeds > 1:
        gosterilecek += [f"{m}_sd" for m in ("hazard.auc", "hazard.mae_gun",
                                             "aday.hibrit.recall",
                                             "m4.marj_farki_tl",
                                             "m4.oracle_marj_farki_tl",
                                             "m5.a.karsilanmayan_farki",
                                             "m5.b.net_marj_hedefli_farki",
                                             "m6.denetim.uplift_x.dr.sapma_yuzde",
                                             "m6.online.agresif_vade.artimsal_yuzde_son",
                                             # M7'nin cikis kriteri metrigi:
                                             # rejim ayrismasi. SD'siz
                                             # basilirsa seed gurultusu ile
                                             # rejim etkisi ayirt edilemez.
                                             "m7.fark.sok.kol_degisen",
                                             "m7.senaryo.sok.erteleme_tl_adet")
                         if f"{m}_sd" in ozet.columns]
    with pl.Config(tbl_rows=40, tbl_cols=40, tbl_width_chars=260, float_precision=4):
        print(ozet.select(gosterilecek))

    bulgular = ihlal_denetimi(tablo)
    if bulgular:
        print("\n!!! KISIT IHLALI (bu tarama gecerli bir politika uretmiyor):")
        for b in bulgular:
            print(f"    {b}")
    print(f"\ncikti: {cikti}")


if __name__ == "__main__":
    main()
