"""Tek config -> tek kosu -> metrik seti. SPEC 5b.2.

    python -m experiments.run --profil fast --seed 1
    python -m experiments.run --profil full --knob tukenme.hedef.karar_ufku_hafta=8

Bir kosunun tamami:
    config (+ knob gecersiz kilma)
      -> dunya uretimi (seed'li)            sim/
      -> gozlemlenebilir panel              features/
      -> zaman bazli egitim/test bolmesi
      -> tahminciler (sabit, kural, defter, hazard)   models/
      -> ORACLE ile olcum                   eval/
      -> experiments/runs/<kosu_id>/

Cikti dosyalari:
    manifest.json    profil, config hash, seed, sureler, panel ozeti
    metrikler.json   tahminci basina tum metrikler + duz (flat) sozluk
    tahminler.parquet  satir bazli tahminler (compare.py'nin eslesmis
                       bootstrap'i icin; --tahmin-yazma ile kapatilir)

Kosu KENDI dunyasini uretir ve varsayilan olarak silmez; ayni kosu_id iki kez
calistirilinca ayni sonucu vermeli (tests/test_depletion.py::test_kosu_tekrar
_uretilebilir).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl

from agent import scenario as ag_senaryo
from agent import tools as ag_arac
from core.config import Config, load_config
from core.io import Run
from core.rng import SeedBank
from eval import aday as ev_aday
from eval import allocation as ev_tahsis
from eval import metrics as mt
from eval import ope as ev_ope
from eval import oracle as ev_oracle
from eval import report as ev_rapor
from eval import uplift as ev_uplift
from eval.oracle import KarsiOlgusalOracle, Oracle, OracleEtiketleri
from features import teklif as ft
from features.okuma import GozlemlenebilirKaynak
from harness import run as hr
from features.panel import Panel, izgara_kur, panel_kur
from models import uplift as mu
from models.depletion import (DefterTahmincisi, HazardTahmincisi,
                              KovaKalibratoru, KuralTahmincisi, SabitTahminci,
                              Tahmin, gozlemlenebilir_olay)
from policy import allocate as alloc
from policy import bandit
from policy import candidates as pol_aday
from policy import scorer
from policy.constraints import VETO_SEBEPLERI, kisit_uygula
from scripts.generate_world import dunya_yaz
from sim import rollout as rl
from sim.calendar import GUN_HAFTA
from sim.response import (GercekDurum, beklenen_miktar_carpani, sonuc_ornekle,
                          tepki_evreni_kur, tepki_hesapla)
from sim.world import dunya_kur, hafta_adimi

REPO_ROOT = Path(__file__).resolve().parent.parent
KOSU_DIZINI = REPO_ROOT / "experiments" / "runs"
ASAMALAR = ("m2", "m3", "m4", "m5", "m6", "m7")


def knob_ayristir(ham: list[str]) -> dict[str, object]:
    cikti: dict[str, object] = {}
    for parca in ham:
        if "=" not in parca:
            raise SystemExit(f"--knob bicimi 'yol=deger' olmali: {parca}")
        yol, deger = parca.split("=", 1)
        cikti[yol] = deger_coz(deger)
    return cikti


def deger_coz(deger: str) -> object:
    """'3' -> 3, '0.5' -> 0.5, 'true' -> True, '[1,2]' -> [1, 2]."""
    metin = deger.strip()
    if metin.lower() in ("true", "false"):
        return metin.lower() == "true"
    if metin.startswith("["):
        return json.loads(metin)
    try:
        return int(metin)
    except ValueError:
        pass
    try:
        return float(metin)
    except ValueError:
        return metin


@dataclass
class Bolme:
    egitim: np.ndarray
    test: np.ndarray
    sinir_hafta: int


def zaman_bolmesi(panel: Panel, cfg: Config, W: int) -> Bolme:
    """Zaman bazli bolme. Rassal bolme yapilmaz: ayni hucrenin gelecegi
    egitime sizardi.

    Ayrica egitim ile test arasina `sinir_tamponu_hafta` bosluk konur; egitim
    etiketi ufuk kadar ileri baktigi icin bosluk olmazsa egitim satirlari test
    penceresini gormus olur.
    """
    hedef = cfg.tukenme.hedef
    originler = np.unique(panel.origin)
    # Etiketin TAM gozlenebildigi origin'ler: ufuk kadar ileri yer kalmali.
    uygun = panel.origin <= W - 1 - hedef.ufuk_hafta
    sinir = int(np.quantile(originler, cfg.feature.panel.egitim_orani))
    egitim = np.flatnonzero(uygun & (panel.origin <= sinir - hedef.sinir_tamponu_hafta))
    test = np.flatnonzero(uygun & (panel.origin > sinir))
    if egitim.size == 0 or test.size == 0:
        raise ValueError(
            f"bolme bos (egitim={egitim.size}, test={test.size}). Profil cok kisa "
            f"ya da ilk_origin_hafta / sinir_tamponu_hafta cok buyuk."
        )
    return Bolme(egitim=egitim, test=test, sinir_hafta=sinir)


def _oracle_hedefleri(o: OracleEtiketleri, ufuk: int, karar_ufku: int):
    """Olcum maskesi ve gercek hedefler.

    gecerli: origin'de gercekten stogu olan VE tukenmeden once listeden
    dusmemis satirlar. Listeden dusme rakip risktir (eval/oracle.py).
    """
    gecerli = o.canli & ~o.rakip_sansur
    y = ((o.tukenme_k > 0) & (o.tukenme_k <= karar_ufku)).astype(int)
    T = np.where(o.olay, np.minimum(o.tukenme_k, ufuk), ufuk).astype(float)
    return gecerli, y, T


def degerlendir(ad: str, tahmin, o: OracleEtiketleri, gozlem_y: np.ndarray,
                cfg: Config) -> dict:
    """Bir tahminciyi ORACLE'a karsi olcer + gozlemlenebilir etiketi de raporlar."""
    hedef = cfg.tukenme.hedef
    d = cfg.tukenme.degerlendirme
    gecerli, y, T = _oracle_hedefleri(o, hedef.ufuk_hafta, hedef.karar_ufku_hafta)
    p = tahmin.olasilik[gecerli]
    s = tahmin.skor[gecerli]
    yg = y[gecerli]
    tahmini_sure = tahmin.tukenme_hafta[gecerli]
    olayli = o.olay[gecerli]

    return {
        "ad": ad,
        "auc": mt.guvenli_auc(yg, s),
        "pr_auc": mt.guvenli_pr_auc(yg, s),
        "brier": mt.brier(yg, p),
        "kalibrasyon_hatasi": mt.beklenen_kalibrasyon_hatasi(yg, p, d.kalibrasyon_kova_sayisi),
        "ust_dilim_kazanci": mt.ust_dilim_kazanci(yg, s),
        "mae_gun": mt.mae(tahmini_sure, T[gecerli]) * GUN_HAFTA,
        "mae_gun_olayli": mt.mae(tahmini_sure[olayli], T[gecerli][olayli]) * GUN_HAFTA,
        "yanlilik_gun": mt.yanlilik(tahmini_sure, T[gecerli]) * GUN_HAFTA,
        # Modelin EGITILDIGI soru: "bize siparis gelir mi". Gercek hedefle
        # arasindaki fark M2'nin asil bulgusu (reports/m2.md 3.3).
        "auc_gozlemlenebilir": mt.guvenli_auc(gozlem_y, tahmin.skor),
    }


@dataclass
class BoruCiktisi:
    """Bir dunya uzerinde kurulan tam M2 boru hatti (olcume hazir)."""

    izgara: object
    panel: Panel
    bolme: Bolme
    tahminciler: list
    tahminler: dict
    oracle: Oracle
    o_test: OracleEtiketleri
    y_egitim: np.ndarray
    y_test: np.ndarray
    zaman: dict


def boru_hatti(cfg: Config, kosu_adi: str, kok: Path) -> BoruCiktisi:
    """Var olan bir dunya dizininden panel -> egitim -> tahmin.

    Dunyayi URETMEZ; hem experiments/run.py hem scripts/verify_m2.py bunu
    kullanir ki olculen sey ile raporlanan sey ayni kod yolundan gecsin.
    """
    zaman = {}
    t0 = time.perf_counter()
    izg = izgara_kur(GozlemlenebilirKaynak(kosu_adi, kok=kok), cfg)
    panel = panel_kur(izg, cfg)
    bolme = zaman_bolmesi(panel, cfg, izg.W)
    zaman["panel_sn"] = round(time.perf_counter() - t0, 2)

    ufuk, H = cfg.tukenme.hedef.ufuk_hafta, cfg.tukenme.hedef.karar_ufku_hafta
    y_egitim = gozlemlenebilir_olay(panel, bolme.egitim, H)
    y_test = gozlemlenebilir_olay(panel, bolme.test, H)

    t0 = time.perf_counter()
    tahminciler = [
        SabitTahminci(cfg).egit(panel, bolme.egitim, y_egitim),
        KuralTahmincisi(cfg, panel, ikili=True).egit(panel, bolme.egitim, y_egitim),
        KuralTahmincisi(cfg, panel, ikili=False).egit(panel, bolme.egitim, y_egitim),
        DefterTahmincisi(cfg, panel).egit(panel, bolme.egitim, y_egitim),
        HazardTahmincisi(cfg, panel).egit(panel, bolme.egitim,
                                          panel.etiket_k[bolme.egitim],
                                          panel.izlenen_k[bolme.egitim]),
    ]
    zaman["egitim_sn"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    oracle = Oracle(kosu_adi, kok=kok)
    o_test = oracle.etiketle(
        panel.anahtar["eczane_id"].to_numpy()[bolme.test],
        panel.anahtar["sku_id"].to_numpy()[bolme.test],
        panel.origin[bolme.test], ufuk,
    )
    tahminler = {t.ad: t.tahmin(panel, bolme.test) for t in tahminciler}
    if cfg.tukenme.degerlendirme.oracle_teshisi:
        tahminler.update(_teshisler(cfg, panel, bolme, oracle, y_egitim))
    zaman["olcum_sn"] = round(time.perf_counter() - t0, 2)

    return BoruCiktisi(izgara=izg, panel=panel, bolme=bolme, tahminciler=tahminciler,
                       tahminler=tahminler, oracle=oracle, o_test=o_test,
                       y_egitim=y_egitim, y_test=y_test, zaman=zaman)


# --------------------------------------------------------------------------
# M3: aday uretimi + kisit katmani
# --------------------------------------------------------------------------
@dataclass
class M3Ciktisi:
    dunya: pol_aday.AdayDunyasi
    originler: list[int]
    gorunumler: dict[int, pol_aday.OriginGorunumu]
    teklifler: pl.DataFrame        # teslim edilen havuz, kisit uygulanmis
    olcum: pl.DataFrame            # uretici x K x hedef recall satirlari
    liste: pl.DataFrame            # havuz -> veto -> liste recall satirlari
    veto: pl.DataFrame             # origin basina veto ozeti
    zaman: dict


def m3_boru_hatti(cfg: Config, kosu_adi: str, kok: Path,
                  oracle_hedefi: bool = True) -> M3Ciktisi:
    """Origin basina: aday havuzu -> kisit -> olcum.

    Hem experiments/run.py hem scripts/verify_m3.py bunu cagirir; raporlanan
    sayi ile olculen sayi ayni kod yolundan gecsin diye (M2 ile ayni disiplin).
    """
    a = cfg.politika.aday
    t0 = time.perf_counter()
    dunya = pol_aday.dunya_yukle(GozlemlenebilirKaynak(kosu_adi, kok=kok), cfg)
    originler = pol_aday.origin_haftalari(cfg, dunya.W)
    k_degerleri = sorted(set(a.degerlendirme.k_degerleri) | {a.havuz_boyutu_k})

    gorunumler, teklif_bloklari, olcum, liste, veto = {}, [], [], [], []
    for t in originler:
        gor = pol_aday.gorunum_kur(dunya, cfg, t)
        gorunumler[t] = gor
        havuz, skorlar = pol_aday.aday_havuzu(dunya, cfg, gor)
        teklifler = kisit_uygula(dunya, cfg, gor, havuz)
        teklif_bloklari.append(teklifler)

        hedefler = [ev_aday.gozlemlenebilir_hedef(dunya, gor, a.degerlendirme.ufuk_hafta)]
        if oracle_hedefi:
            hedefler.append(ev_aday.oracle_hedef(kosu_adi, dunya, gor,
                                                 a.degerlendirme.ufuk_hafta, kok=kok))
        olcum += ev_aday.origin_olcumu(dunya, gor, skorlar, hedefler, k_degerleri,
                                       a.soguk_start.soguk_dilim)
        liste += ev_aday.liste_olcumu(dunya, gor, teklifler, hedefler)
        veto.append({"origin": t, **ev_aday.veto_ozeti(teklifler, VETO_SEBEPLERI)})

    return M3Ciktisi(
        dunya=dunya, originler=originler, gorunumler=gorunumler,
        teklifler=pl.concat(teklif_bloklari), olcum=pl.DataFrame(olcum),
        liste=pl.DataFrame(liste), veto=pl.DataFrame(veto),
        zaman={"m3_sn": round(time.perf_counter() - t0, 2)},
    )


def m3_ihlaller(c: M3Ciktisi, cfg: Config) -> dict:
    """Cikis kriterinin degismezleri. Hepsi SIFIR olmak zorunda.

    Metrik olarak uretiliyorlar ki her sweep kosusu bunlari da raporlasin:
    bir knob kombinasyonu kisiti delerse tabloda gorunur, testin kosulmasini
    beklemez.
    """
    k = cfg.politika.kisit
    liste = c.teklifler.filter(pl.col("listede"))
    renk = liste.join(c.dunya.urunler.select(["sku_id", "recete_rengi"]),
                      on="sku_id", how="left")["recete_rengi"].to_numpy()
    ihlal = {
        "kirmizi_yesil_listede": int(np.isin(renk, list(k.recete_rengi_vetosu)).sum()),
        "soguk_zincir_min_ihlali": int(liste.filter(
            pl.col("soguk_zincir")
            & (pl.col("teklif_adedi") < k.soguk_zincir_min_siparis_adedi)).height),
        "lotsuz_teklif": int(liste.filter(pl.col("lot_id").is_null()).height),
        "raf_omru_ihlali": int(liste.filter(
            pl.col("lot_kalan_gun") < k.asgari_kalan_raf_omru_gun).height),
        "tedarik_guclugu_listede": int(liste.join(
            c.dunya.urunler.select(["sku_id", "titck_tedarik_guclugu"]),
            on="sku_id", how="left").filter(pl.col("titck_tedarik_guclugu")).height)
        if k.tedarik_guclugu_veto else 0,
    }
    ihlal["kredi_asimi"] = _kredi_asimi(c, cfg)
    return ihlal


def _kredi_asimi(c: M3Ciktisi, cfg: Config) -> int:
    """Listedeki tekliflerin toplami DBS limitini asan eczane sayisi."""
    k = cfg.politika.kisit
    dbs = c.dunya.eczaneler["dbs_limiti"].to_numpy().astype(float)
    risk = c.dunya.eczaneler["vade_riski_skoru"].to_numpy().astype(float)
    tavan = dbs * k.kredi_kullanim_tavani * (1.0 - risk * k.vade_riski_cezasi)
    asan = 0
    for t, gor in c.gorunumler.items():
        liste = c.teklifler.filter(pl.col("listede") & (pl.col("origin") == t))
        if liste.height == 0:
            continue
        toplam = np.zeros(c.dunya.P)
        np.add.at(toplam, liste["eczane_idx"].to_numpy(),
                  liste["teklif_tutari"].to_numpy())
        # Kalan limit negatife dusemez: acik bakiyesi tavanı asmis eczaneye
        # hic teklif cikmamali.
        asan += int((toplam > np.maximum(tavan - gor.acik_bakiye, 0.0) + 1e-6).sum())
    return asan


def m3_duz_metrikler(c: M3Ciktisi, cfg: Config) -> dict:
    """Sweep tablosuna giren duz M3 metrikleri (origin ortalamalari)."""
    a = cfg.politika.aday
    K = a.havuz_boyutu_k
    duz: dict[str, float] = {}

    goz = c.olcum.filter((pl.col("k") == K) & (pl.col("hedef") == "gozlemlenebilir"))
    for satir in goz.group_by("uretici").agg(
            pl.col("recall").mean(), pl.col("yeni_hucre_recall").mean(),
            pl.col("kapsama").mean()).iter_rows(named=True):
        u = satir["uretici"]
        duz[f"aday.{u}.recall"] = satir["recall"]
        duz[f"aday.{u}.yeni_recall"] = satir["yeni_hucre_recall"]
        duz[f"aday.{u}.kapsama"] = satir["kapsama"]

    hibrit = goz.filter(pl.col("uretici") == "hibrit")
    duz["aday.hibrit.soguk_eczane_recall"] = float(hibrit["soguk_eczane_recall"].mean())
    duz["aday.hibrit.sicak_eczane_recall"] = float(hibrit["sicak_eczane_recall"].mean())
    duz["aday.hibrit.precision"] = float(hibrit["precision"].mean())
    duz["aday.hedef_sayisi"] = float(hibrit["hedef_sayisi"].mean())
    duz["aday.yeni_hedef_orani"] = float(
        (hibrit["yeni_hedef_sayisi"] / hibrit["hedef_sayisi"]).mean())

    orc = c.olcum.filter((pl.col("k") == K) & (pl.col("hedef") == "oracle")
                         & (pl.col("uretici") == "hibrit"))
    if orc.height:
        duz["aday.hibrit.oracle_recall"] = float(orc["recall"].mean())
        duz["aday.hibrit.oracle_yeni_recall"] = float(orc["yeni_hucre_recall"].mean())
        duz["aday.oracle_hedef_sayisi"] = float(orc["hedef_sayisi"].mean())

    lst = c.liste.filter(pl.col("hedef") == "gozlemlenebilir")
    duz["kisit.havuz_recall"] = float(lst["havuz_recall"].mean())
    duz["kisit.veto_sonrasi_recall"] = float(lst["veto_sonrasi_recall"].mean())
    duz["kisit.liste_recall"] = float(lst["liste_recall"].mean())
    duz["kisit.veto_recall_bedeli"] = float(
        (lst["havuz_recall"] - lst["veto_sonrasi_recall"]).mean())
    duz["kisit.liste_precision"] = float(lst["liste_precision"].mean())

    for kolon in c.veto.columns:
        if kolon != "origin":
            duz[f"kisit.{kolon}"] = float(c.veto[kolon].mean())
    duz.update({f"ihlal.{ad}": float(v) for ad, v in m3_ihlaller(c, cfg).items()})
    return duz


# --------------------------------------------------------------------------
# M4: uplift (CATE) + aksiyon secimi
# --------------------------------------------------------------------------
@dataclass
class OriginBlogu:
    """Bir origin'in aday satirlari + kol matrisleri + ozellikler.

    Aday kumesi VETO SONRASIDIR: kisit katmani aksiyon seciminden once
    calisir ve veto yetkisi M4'te de degismedi (D6). M3'un frekans tavani
    ise burada UYGULANMAZ - onu aksiyon secimi kendi amac fonksiyonuyla
    yeniden yapar; M4'un ekledigi sey tam olarak budur.
    """

    t: int
    gor: pol_aday.OriginGorunumu
    teklifler: pl.DataFrame
    mat: object
    X: np.ndarray
    pi: np.ndarray
    ozellik_adlari: list[str]
    kategorik: list[int]
    # Olcum origin'lerinde doldurulur; egitim origin'lerinde bos kalir.
    p_t: np.ndarray | None = None
    p_x: np.ndarray | None = None


@dataclass
class M4Ciktisi:
    dunya: pol_aday.AdayDunyasi
    egitim_originleri: list[int]
    olcum_originleri: list[int]
    ozellik_adlari: list[str]
    egitim_satiri: int
    bloklar: list[OriginBlogu]
    tepkiler: list[object]
    secimler: dict[str, list]         # politika -> origin basina Secim
    olcumler: dict[str, ev_uplift.PolitikaOlcumu]
    ayristirma: dict
    teshis: ev_uplift.HeterojenlikTeshisi
    cate: dict                        # PEHE / sira korelasyonu / AUUC
    destek: dict
    guven: dict
    kredi_vetosu: int
    zaman: dict
    # --- M6'nin kapali dongusune tasinan artifact'ler ---
    # Rollout MODELI YENIDEN EGITMEZ: M4'te egitilmis T/X ogrenicisini her
    # karar haftasinda yeniden kullanir. Gercek sistemde de model haftalik
    # egitilmez; ayrica yeniden egitim olsaydi olculen fark politika farki
    # degil model tazeligi farki olurdu.
    t_ogr: object = None
    x_ogr: object = None
    evren: object = None      # sim.response.TepkiEvreni (dunyanin tepki latentleri)
    td: object = None         # features.teklif.TeklifDunyasi (statik tablolar)
    carpan: float = 1.0


def _origin_blogu(td, cfg: Config, dunya, t: int) -> OriginBlogu:
    """Tek origin: aday havuzu -> veto -> kol matrisleri -> ozellikler."""
    gor = pol_aday.gorunum_kur(dunya, cfg, t)
    havuz, skorlar = pol_aday.aday_havuzu(dunya, cfg, gor)
    tumu = kisit_uygula(dunya, cfg, gor, havuz)
    aday = tumu.filter(~pl.col("vetolu"))
    mat = scorer.teklif_matrisleri(dunya, cfg, aday)
    X, adlar, kategorik = ft.ozellik_matrisi(td, cfg, gor, aday, skorlar)
    pi = bandit.kayit_olasiliklari(dunya, cfg, mat.uzay, aday, mat.izinli)
    return OriginBlogu(t=t, gor=gor, teklifler=aday, mat=mat, X=X, pi=pi,
                       ozellik_adlari=adlar, kategorik=kategorik)


def _gozlemlenebilir_politikalar(cfg: Config, mat, teklifler: pl.DataFrame,
                                 p_t: np.ndarray, p_x: np.ndarray,
                                 carpan: float) -> dict[str, scorer.Secim]:
    """Yalnizca GOZLEMLENEBILIR buyukluklerle kurulan politikalar.

    M6'nin kapali dongusu (sim/rollout.py) bunlari her karar haftasinda
    yeniden kosar ve orada gercek tepki olasiligi ELDE YOKTUR. Oracle
    politikalari bu yuzden ayri tutuldu: ayni fonksiyonda dursalardi rollout
    ya onlari kosamaz ya da ground_truth okurdu.

    KRITIK: `propensity` ve `uplift_t` AYNI olasilik matrisini (p_t)
    kullanir. Aralarindaki tek fark amac fonksiyonudur:

        propensity : deger = p(a) * marj(a),        taban = 0
        uplift     : deger = p(a) * marj(a),        taban = p(0) * marj(0)

    Boylece olculen marj farki model kalitesine degil, YALNIZCA "seviyeyi mi
    artimi mi optimize ediyorsun" sorusuna atfedilebilir.
    """
    k = cfg.politika.kisit
    s = cfg.politika.skor
    ecz = teklifler["eczane_idx"].to_numpy()
    n = teklifler.height
    tavan = k.eczane_haftalik_teklif_tavani
    esik = s.asgari_teklif_marji
    sifir = np.zeros(n)

    def _deger(p: np.ndarray) -> np.ndarray:
        return p * mat.marj * carpan

    d_t, d_x = _deger(p_t), _deger(p_x)
    # "Herkese ayni kampanya": en derin MF + taban vade. Saha kuralinin
    # karsiligi; aksiyon secimi yok, yalnizca M3 skor sirasi var.
    sabit = int(np.argmax((mat.uzay.mf == mat.uzay.mf.max())
                          & (mat.uzay.vade == cfg.politika.aksiyon.taban_vade_gun)))
    sabit_deger = np.full((n, mat.uzay.A), -np.inf)
    sabit_deger[:, sabit] = teklifler["skor"].to_numpy()

    return {
        "teklif_yok": scorer.Secim("teklif_yok", np.zeros(n, dtype=np.int32), sifir),
        "m3_sabit_kampanya": scorer.sec("m3_sabit_kampanya", sabit_deger, sifir,
                                        mat.izinli, ecz, tavan, -np.inf),
        "propensity_ham": scorer.sec("propensity_ham", p_t * carpan, sifir,
                                     mat.izinli, ecz, tavan, -np.inf),
        "propensity": scorer.sec("propensity", d_t, sifir, mat.izinli, ecz, tavan, esik),
        "uplift_t": scorer.sec("uplift_t", d_t, d_t[:, scorer.TEKLIF_YOK],
                               mat.izinli, ecz, tavan, esik),
        "uplift_x": scorer.sec("uplift_x", d_x, d_x[:, scorer.TEKLIF_YOK],
                               mat.izinli, ecz, tavan, esik),
    }


AGRESIF_POLITIKALARI = ("agresif", "agresif_vade")


def agresif_politika(cfg: Config, mat, teklifler: pl.DataFrame,
                     ad: str = "agresif") -> scorer.Secim:
    """SPEC 5'in "agresif iskonto"su. Iki varyant, cunku D1'in iki ekseni var.

        agresif       : EN DERIN MF + EN UZUN VADE. Sahanin refleksi;
                        "ne verirsen ver, satsin".
        agresif_vade  : MF YOK + EN UZUN VADE. Yalnizca sartlar agresif.

    IKISI AYRI OLMAK ZORUNDA cunku iki kalemin zaman olcegi farkli. MF bir
    MAL maliyetidir ve teklif aninda pesin odenir; vade bir FONLAMA
    maliyetidir, gunluk ve kucuktur, ama kabul olasiligini ayni yonde iter.
    Tek bir "agresif" tanimi kullansaydik ve o tanim MF'i iceriyorsa, MF'in
    ani bedeli gecikmeli kanibalizmi tamamen ortbas ederdi -- SPEC'in
    "kisa ufukta kazanir" yarisi hic sinanamazdi.

    `m3_sabit_kampanya`dan farki vade boyutu: o taban vadede kalir.

    Aksiyon secimi YOK: satirlar M3 skoruna gore siralanir, frekans tavanina
    kadar herkese ayni kol verilir. Yeni knob getirmez -- kollar aksiyon
    uzayinin ucundan okunur.
    """
    if ad not in AGRESIF_POLITIKALARI:
        raise KeyError(f"bilinmeyen agresif varyant: {ad}")
    n = teklifler.height
    mf_hedef = 0.0 if ad == "agresif_vade" else mat.uzay.mf.max()
    kol = int(np.argmax((mat.uzay.mf == mf_hedef)
                        & (mat.uzay.vade == mat.uzay.vade.max())))
    deger = np.full((n, mat.uzay.A), -np.inf)
    deger[:, kol] = teklifler["skor"].to_numpy()
    return scorer.sec(ad, deger, np.zeros(n), mat.izinli,
                      teklifler["eczane_idx"].to_numpy(),
                      cfg.politika.kisit.eczane_haftalik_teklif_tavani, -np.inf)


def _politikalar(cfg: Config, blok: OriginBlogu, p_t: np.ndarray, p_x: np.ndarray,
                 p_gercek: np.ndarray, carpan: float) -> dict[str, scorer.Secim]:
    """M4'un sekiz politikasi: alti gozlemlenebilir + iki oracle."""
    k = cfg.politika.kisit
    s = cfg.politika.skor
    mat = blok.mat
    ecz = blok.teklifler["eczane_idx"].to_numpy()
    n = blok.teklifler.height
    tavan = k.eczane_haftalik_teklif_tavani
    esik = s.asgari_teklif_marji
    sifir = np.zeros(n)
    d_g = p_gercek * mat.marj * carpan

    politikalar = _gozlemlenebilir_politikalar(cfg, mat, blok.teklifler, p_t,
                                               p_x, carpan)
    politikalar.update({
        # Iki ORACLE politikasi: model hatasi SIFIRKEN amac fonksiyonunun
        # tek basina ne kadar fark yarattigini olcer. Aradaki fark "uplift
        # modellemenin tavani", `uplift_x - propensity` ise "bugun elde
        # edilen". Ikisinin farki tahmin hatasinin bedelidir (reports/m4.md 6).
        "oracle_propensity": scorer.sec("oracle_propensity", d_g, sifir,
                                        mat.izinli, ecz, tavan, esik),
        "oracle_uplift": scorer.sec("oracle_uplift", d_g, d_g[:, scorer.TEKLIF_YOK],
                                    mat.izinli, ecz, tavan, esik),
    })
    return politikalar


def m4_boru_hatti(cfg: Config, kosu_adi: str, kok: Path) -> M4Ciktisi:
    """Kayit kosusu -> T/X ogrenici -> aksiyon secimi -> gercek marj olcumu."""
    t0 = time.perf_counter()
    kaynak = GozlemlenebilirKaynak(kosu_adi, kok=kok)
    dunya = pol_aday.dunya_yukle(kaynak, cfg)
    td = ft.teklif_dunyasi_yukle(kaynak, cfg, dunya)
    seedler = SeedBank(cfg.profil.temel_seed)
    durum = GercekDurum(kosu_adi, kok=kok)
    evren = tepki_evreni_kur(cfg, seedler, dunya.eczaneler, dunya.urunler,
                             durum.latent_eczane)
    carpan = beklenen_miktar_carpani(cfg)

    egitim_t = mu.egitim_originleri(cfg, dunya.W)
    olcum_t = pol_aday.origin_haftalari(cfg, dunya.W)
    zaman = {}

    # --- 1) kayit kosusu: loglanmis teklifler (D7) ---
    X_bloklari, kol_bloklari, y_bloklari, prop_bloklari = [], [], [], []
    adlar: list[str] = []
    kategorik: list[int] = []
    for t in egitim_t:
        blok = _origin_blogu(td, cfg, dunya, t)
        adlar, kategorik = blok.ozellik_adlari, blok.kategorik
        kayit = bandit.kayit_kosusu(dunya, cfg, seedler, blok.mat.uzay,
                                    blok.teklifler, blok.mat.izinli, t)
        tepki = tepki_hesapla(cfg, evren, durum, blok.mat.uzay, blok.teklifler,
                              t, blok.mat.adet)
        kabul, _ = sonuc_ornekle(cfg, seedler, tepki, kayit.kol, t)
        X_bloklari.append(blok.X)
        kol_bloklari.append(kayit.kol)
        y_bloklari.append(kabul)
        prop_bloklari.append(kayit.propensity)
    X_egitim = np.vstack(X_bloklari)
    kol_egitim = np.concatenate(kol_bloklari)
    y_egitim = np.concatenate(y_bloklari).astype(int)
    prop_egitim = np.concatenate(prop_bloklari)
    zaman["kayit_sn"] = round(time.perf_counter() - t0, 2)

    # --- 2) ogreniciler ---
    t0 = time.perf_counter()
    A = len(scorer.aksiyon_uzayi(cfg).adlar)
    t_ogr = mu.TOgrenici(cfg, A, kategorik).egit(X_egitim, kol_egitim, y_egitim)
    x_ogr = mu.XOgrenici(cfg, t_ogr, kategorik).egit(X_egitim, kol_egitim, y_egitim)
    zaman["egitim_sn"] = round(time.perf_counter() - t0, 2)

    # --- 3) olcum origin'leri: aksiyon secimi + gercek marj ---
    t0 = time.perf_counter()
    bloklar, tepkiler = [], []
    secimler: dict[str, list] = {}
    kredi_vetolanan = 0
    cate_yigin = {"gercek": [], "t": [], "x": [], "izinli": []}
    for t in olcum_t:
        blok = _origin_blogu(td, cfg, dunya, t)
        tepki = tepki_hesapla(cfg, evren, durum, blok.mat.uzay, blok.teklifler,
                              t, blok.mat.adet)
        blok.p_t = t_ogr.olasilik(blok.X)
        blok.p_x = x_ogr.olasilik(blok.X, blok.pi)
        p_t, p_x = blok.p_t, blok.p_x
        secim = _politikalar(cfg, blok, p_t, p_x, tepki.olasilik, carpan)
        # D6: kisit katmani aksiyon seciminden SONRA da veto yetkisini korur.
        # Koli yuvarlamasi tutari buyutuyor; portfoy kredi limiti yeniden bakilir.
        for s in secim.values():
            veto = scorer.kredi_son_kontrolu(dunya, cfg, blok.gor, blok.teklifler,
                                             blok.mat, s)
            kredi_vetolanan += int(veto.sum())
            s.kol = np.where(veto, scorer.TEKLIF_YOK, s.kol).astype(np.int32)
        bloklar.append(blok)
        tepkiler.append(tepki)
        for ad, s in secim.items():
            secimler.setdefault(ad, []).append(s)
        cate_yigin["gercek"].append(tepki.uplift)
        cate_yigin["t"].append(t_ogr.cate(blok.X))
        cate_yigin["x"].append(x_ogr.cate(blok.X, blok.pi))
        cate_yigin["izinli"].append(blok.mat.izinli)
    zaman["olcum_sn"] = round(time.perf_counter() - t0, 2)

    cikti = _m4_topla(cfg, egitim_t, olcum_t, adlar, X_egitim.shape[0],
                      bloklar, tepkiler, secimler, cate_yigin, carpan,
                      kol_egitim, prop_egitim, kredi_vetolanan, zaman, dunya)
    cikti.t_ogr, cikti.x_ogr = t_ogr, x_ogr
    cikti.evren, cikti.td, cikti.carpan = evren, td, carpan
    return cikti


def _m4_topla(cfg: Config, egitim_t, olcum_t, adlar, egitim_satiri,
              bloklar, tepkiler, secimler, cate_yigin, carpan, kol_egitim,
              prop_egitim, kredi_vetolanan, zaman, dunya) -> M4Ciktisi:
    """Origin'ler boyunca toplanmis olcum. Marjlar TOPLANIR (ortalanmaz):
    politika degeri bir toplam buyuklugudur, origin sayisi tabloda yazili."""
    d = cfg.uplift.degerlendirme
    olcumler = {}
    for ad, liste in secimler.items():
        parcalar = [ev_uplift.politika_olcumu(ad, tp, b.mat, s, carpan)
                    for tp, b, s in zip(tepkiler, bloklar, liste)]
        olcumler[ad] = ev_uplift.PolitikaOlcumu(
            ad=ad,
            toplam_marj=sum(p.toplam_marj for p in parcalar),
            artimsal_marj=sum(p.artimsal_marj for p in parcalar),
            teklif_sayisi=sum(p.teklif_sayisi for p in parcalar),
            satir_sayisi=sum(p.satir_sayisi for p in parcalar),
            negatif_teklif_sayisi=sum(p.negatif_teklif_sayisi for p in parcalar),
            negatif_marj=sum(p.negatif_marj for p in parcalar),
            ortalama_mf=float(np.mean([p.ortalama_mf for p in parcalar])),
            ortalama_vade=float(np.mean([p.ortalama_vade for p in parcalar])),
            beklenen_kabul=float(np.mean([p.beklenen_kabul for p in parcalar])),
        )

    ayristirma: dict = {}
    for tp, b, u, p in zip(tepkiler, bloklar, secimler["uplift_x"],
                           secimler["propensity"]):
        for anahtar, deger in ev_uplift.marj_farki_ayristirmasi(
                tp, b.mat, u, p, carpan).items():
            ayristirma[anahtar] = ayristirma.get(anahtar, 0) + deger

    teshis_parca = [ev_uplift.heterojenlik_teshisi(tp, b.mat, u, p, carpan)
                    for tp, b, u, p in zip(tepkiler, bloklar, secimler["uplift_x"],
                                           secimler["propensity"])]
    teshis = ev_uplift.HeterojenlikTeshisi(
        cate_sapmasi=float(np.mean([x.cate_sapmasi for x in teshis_parca])),
        cate_dilim_orani=float(np.mean([x.cate_dilim_orani for x in teshis_parca])),
        artimsal_sapma=float(np.mean([x.artimsal_sapma for x in teshis_parca])),
        farkli_karar_orani=float(np.mean([x.farkli_karar_orani for x in teshis_parca])),
        marj_farki=float(np.sum([x.marj_farki for x in teshis_parca])),
    )

    gercek = np.vstack(cate_yigin["gercek"])
    izinli = np.vstack(cate_yigin["izinli"])
    cate = {}
    for ad, tahmin in (("t", np.vstack(cate_yigin["t"])), ("x", np.vstack(cate_yigin["x"]))):
        maske = izinli.copy()
        maske[:, scorer.TEKLIF_YOK] = False
        cate[f"pehe_{ad}"] = ev_uplift.pehe(tahmin, gercek, izinli)
        cate[f"sira_kor_{ad}"] = ev_uplift.sira_korelasyonu(tahmin[maske], gercek[maske])

    cate["kazanc_egrisi"] = _kazanc_egrileri(cfg, tepkiler, bloklar, carpan)

    eczane = np.concatenate([b.teklifler["eczane_idx"].to_numpy() for b in bloklar])
    fark = np.concatenate([
        ev_uplift.artimsal_marj(tp, b.mat, u.kol, carpan)
        - ev_uplift.artimsal_marj(tp, b.mat, p.kol, carpan)
        for tp, b, u, p in zip(tepkiler, bloklar, secimler["uplift_x"],
                               secimler["propensity"])])
    temel, alt, ust = ev_uplift.eczane_bootstrap(fark, eczane, d.bootstrap_orneklem,
                                                 d.bootstrap_seed)
    destek = ev_uplift.kol_destegi(kol_egitim, prop_egitim,
                                   len(bloklar[0].mat.uzay.adlar))

    return M4Ciktisi(
        dunya=dunya, egitim_originleri=egitim_t, olcum_originleri=olcum_t,
        ozellik_adlari=adlar, egitim_satiri=egitim_satiri, bloklar=bloklar,
        tepkiler=tepkiler, secimler=secimler, olcumler=olcumler,
        ayristirma=ayristirma, teshis=teshis, cate=cate, destek=destek,
        guven={"marj_farki": temel, "alt": alt, "ust": ust},
        kredi_vetosu=kredi_vetolanan, zaman=zaman,
    )


def _kazanc_egrileri(cfg: Config, tepkiler, bloklar, carpan: float) -> dict:
    """Siralamanin degeri: Qini egrisinin marj karsiligi.

    Satirlar politikanin TAHMIN ETTIGI kazanca gore siralanir, egri o sirada
    birikimli GERCEK artimsal marji verir. Frekans tavani BURADA YOK - bu bir
    siralama teshisidir, teslim edilen politika degil: "sadece N teklif
    yapabilseydim, hangi siralama daha cok marj toplardi".
    """
    d = cfg.uplift.degerlendirme
    cikti: dict[str, dict] = {}
    for ad, alan, artimsal_mi in (("uplift_t", "p_t", True), ("uplift_x", "p_x", True),
                                  ("propensity", "p_t", False)):
        skorlar, gercekler = [], []
        for tp, b in zip(tepkiler, bloklar):
            deger = getattr(b, alan) * b.mat.marj * carpan
            taban = deger[:, [scorer.TEKLIF_YOK]] if artimsal_mi else 0.0
            izinli = b.mat.izinli.copy()
            izinli[:, scorer.TEKLIF_YOK] = False
            kazanc = np.where(izinli, deger - taban, -np.inf)
            kol = np.argmax(kazanc, axis=1)
            skorlar.append(kazanc[np.arange(kol.size), kol])
            gercekler.append(ev_uplift.artimsal_marj(tp, b.mat, kol, carpan))
        skor = np.concatenate(skorlar)
        gercek = np.concatenate(gercekler)
        sonlu = np.isfinite(skor)
        x, y, auuc = ev_uplift.kazanc_egrisi(skor[sonlu], gercek[sonlu],
                                             d.qini_dilim_sayisi)
        cikti[ad] = {"x": x.tolist(), "y": y.tolist(), "auuc": auuc}
    return cikti


def m4_ihlaller(c: M4Ciktisi, cfg: Config) -> dict:
    """M4'un degismezleri. Hepsi SIFIR olmak zorunda; her kosuda uretilir.

    D1 ve D6'nin aksiyon secimi tarafindaki karsiligi: vetolanmis satira
    teklif cikamaz, MF kanali kapali satirda MF orani sifirdan buyuk olamaz,
    MF'li teklifin adedi koli katina yuvarlanmis olmali, ve secim sonrasi
    portfoy kredi limiti asilamaz.
    """
    ihlal = {"vetolu_satira_teklif": 0, "mf_kanali_ihlali": 0,
             "koli_yuvarlama_ihlali": 0, "kredi_asimi": 0,
             "izinsiz_kol_secildi": 0}
    koli_tablosu = c.dunya.urunler["koli_ici_adet"].to_numpy().astype(float)
    yuvarla = cfg.politika.aksiyon.koli_katina_yuvarla
    for liste in c.secimler.values():
        for b, s in zip(c.bloklar, liste):
            teklif = s.teklif_maskesi
            if not teklif.any():
                continue
            idx = np.arange(s.kol.size)
            mf = b.mat.uzay.mf[s.kol]
            mf_izinli = b.teklifler["mf_izinli"].to_numpy().astype(bool)
            koli = koli_tablosu[b.teklifler["sku_idx"].to_numpy()]
            ihlal["mf_kanali_ihlali"] += int((teklif & (mf > 0) & ~mf_izinli).sum())
            ihlal["izinsiz_kol_secildi"] += int(
                (~b.mat.izinli[idx, s.kol] & teklif).sum())
            if yuvarla:
                # Yuvarlama yalnizca emilim tavanina sigdiginda uygulanir
                # (policy/scorer.py); atlanan satirlar ihlal degildir.
                adet = b.mat.adet[idx, s.kol]
                uygulandi = b.mat.yuvarlandi[idx, s.kol]
                ihlal["koli_yuvarlama_ihlali"] += int(
                    (teklif & (mf > 0) & uygulandi & (np.mod(adet, koli) != 0)).sum())
    # Vetolanmis satir aday kumesine hic girmiyor (aday = ~vetolu); yine de
    # bir degismez olarak kontrol edilir: kume tanimi degisirse bagirir.
    for b in c.bloklar:
        if "vetolu" in b.teklifler.columns:
            ihlal["vetolu_satira_teklif"] += int(b.teklifler["vetolu"].sum())
    ihlal["kredi_asimi"] = _m4_kredi_asimi(c, cfg)
    return ihlal


def _m4_kredi_asimi(c: M4Ciktisi, cfg: Config) -> int:
    """Secilen tekliflerin toplami DBS limitini asan (eczane, origin, politika)."""
    k = cfg.politika.kisit
    dbs = c.dunya.eczaneler["dbs_limiti"].to_numpy().astype(float)
    risk = c.dunya.eczaneler["vade_riski_skoru"].to_numpy().astype(float)
    tavan = dbs * k.kredi_kullanim_tavani * (1.0 - risk * k.vade_riski_cezasi)
    asan = 0
    for liste in c.secimler.values():
        for b, s in zip(c.bloklar, liste):
            n = s.kol.size
            if n == 0:
                continue
            dsf = c.dunya.dsf[b.teklifler["sku_idx"].to_numpy()]
            tutar = np.where(s.teklif_maskesi,
                             b.mat.adet[np.arange(n), s.kol] * dsf, 0.0)
            toplam = np.zeros(c.dunya.P)
            np.add.at(toplam, b.teklifler["eczane_idx"].to_numpy(), tutar)
            asan += int((toplam > np.maximum(tavan - b.gor.acik_bakiye, 0.0) + 1e-6).sum())
    return asan


def m4_duz_metrikler(c: M4Ciktisi, cfg: Config) -> dict:
    """Sweep tablosuna giren duz M4 metrikleri."""
    duz: dict[str, float] = {}
    taban = c.olcumler["teklif_yok"].toplam_marj
    for ad, o in c.olcumler.items():
        duz[f"m4.{ad}.toplam_marj"] = o.toplam_marj
        duz[f"m4.{ad}.artimsal_marj"] = o.artimsal_marj
        duz[f"m4.{ad}.teklif_sayisi"] = float(o.teklif_sayisi)
        duz[f"m4.{ad}.teklif_basina_artimsal"] = o.teklif_basina_artimsal
        duz[f"m4.{ad}.negatif_teklif_orani"] = (
            o.negatif_teklif_sayisi / o.teklif_sayisi if o.teklif_sayisi else 0.0)
        duz[f"m4.{ad}.yakilan_marj"] = o.negatif_marj
        duz[f"m4.{ad}.ortalama_mf"] = o.ortalama_mf
        duz[f"m4.{ad}.ortalama_vade"] = o.ortalama_vade
        duz[f"m4.{ad}.beklenen_kabul"] = o.beklenen_kabul

    p = c.olcumler["propensity"].artimsal_marj
    duz["m4.marj_farki_tl"] = c.olcumler["uplift_x"].artimsal_marj - p
    duz["m4.marj_farki_yuzde"] = (duz["m4.marj_farki_tl"] / abs(p) * 100.0
                                  if p else float("nan"))
    duz["m4.marj_farki_t_tl"] = c.olcumler["uplift_t"].artimsal_marj - p
    duz["m4.marj_farki_alt"] = c.guven["alt"]
    duz["m4.marj_farki_ust"] = c.guven["ust"]
    duz["m4.taban_marj"] = taban
    duz["m4.oracle_acigi_tl"] = (c.olcumler["oracle_uplift"].artimsal_marj
                                 - c.olcumler["uplift_x"].artimsal_marj)
    # Amac fonksiyonunun tek basina degeri (model hatasi sifirken).
    duz["m4.oracle_marj_farki_tl"] = (c.olcumler["oracle_uplift"].artimsal_marj
                                      - c.olcumler["oracle_propensity"].artimsal_marj)
    duz["m4.tahmin_hatasinin_bedeli_tl"] = (duz["m4.oracle_marj_farki_tl"]
                                            - duz["m4.marj_farki_tl"])

    for ad, v in c.ayristirma.items():
        duz[f"m4.ayristirma.{ad}"] = float(v)
    duz.update({f"m4.cate.{k}": float(v) for k, v in c.cate.items()
                if isinstance(v, (int, float))})
    for ad, egri in c.cate["kazanc_egrisi"].items():
        duz[f"m4.auuc.{ad}"] = egri["auuc"]
    duz.update({
        "m4.heterojenlik.cate_sapmasi": c.teshis.cate_sapmasi,
        "m4.heterojenlik.cate_dilim_orani": c.teshis.cate_dilim_orani,
        "m4.heterojenlik.artimsal_sapma": c.teshis.artimsal_sapma,
        "m4.heterojenlik.farkli_karar_orani": c.teshis.farkli_karar_orani,
        "m4.egitim_satiri": float(c.egitim_satiri),
        "m4.kredi_son_vetosu": float(c.kredi_vetosu),
    })
    duz.update({f"m4.destek.{k}": float(v) for k, v in c.destek.items()})
    duz.update({f"ihlal.m4_{ad}": float(v) for ad, v in m4_ihlaller(c, cfg).items()})
    return duz


# --------------------------------------------------------------------------
# M5: kit stok altinda tahsis (D5) + miad rejimi (D9)
# --------------------------------------------------------------------------
@dataclass
class M5Ciktisi:
    dunya: pol_aday.AdayDunyasi
    originler: list[int]
    lotlar: dict[int, alloc.LotGorunumu]
    sonuclar: dict[str, list]          # politika -> origin basina TahsisSonucu
    olcumler: dict[str, ev_tahsis.TahsisOlcumu]
    golge: pl.DataFrame                # lot x politika gölge fiyat tablosu
    carpan: float
    zaman: dict


def m5_boru_hatti(cfg: Config, m4: M4Ciktisi, kosu_adi: str,
                  kok: Path) -> M5Ciktisi:
    """M4'un olcum origin'leri uzerinde dort tahsis politikasi.

    M4'u YENIDEN KOSMAZ: ayni aday satirlari, ayni kol matrisleri, ayni CATE
    tahminleri kullanilir. Boylece M5 ile M4 arasindaki fark tahsis
    katmanindan gelir, model farkindan degil.

    Politikalar:
      ranking_only     (a) taban cizgisi - stok paylastirilmaz
      lp               (a) LP  /  (b) "temizlik yok" - AYNI politika (D9)
      kor_iskonto      (b) kisa miatli lotta en derin MF, adet kuplaji yok
      hedefli_temizlik (b) ayni LP + negatif gölge fiyat + M2 kuplaji
    """
    t0 = time.perf_counter()
    durum = GercekDurum(kosu_adi, kok=kok)
    carpan = beklenen_miktar_carpani(cfg)
    dunya = m4.dunya

    lotlar: dict[int, alloc.LotGorunumu] = {}
    sonuclar: dict[str, list] = {ad: [] for ad in alloc.POLITIKALAR}
    parcalar: dict[str, list] = {ad: [] for ad in alloc.POLITIKALAR}
    golge_bloklari = []

    for blok, tepki in zip(m4.bloklar, m4.tepkiler):
        gor = blok.gor
        # Taban (teklifsiz) organik cekilis butun politikalar icin AYNI ve
        # bir kez dusulur; LP ile acgozlu ayni stok tabanini gorsun diye
        # politika dongusunun DISINDA.
        lot_gor = alloc.taban_talebini_dus(
            dunya, cfg, alloc.lot_gorunumu(dunya, cfg, gor), blok.teklifler,
            blok.mat, blok.p_x, carpan)
        lotlar[gor.t] = lot_gor
        sevk_hizi = ev_tahsis.organik_sevk_hizi(dunya, cfg, gor.t)
        for ad, pol in alloc.POLITIKALAR.items():
            sonuc, kolonlar = alloc.tahsis_et(dunya, cfg, gor, lot_gor,
                                              blok.teklifler, blok.mat, blok.p_x,
                                              carpan, pol)
            sonuclar[ad].append(sonuc)
            parcalar[ad].append(ev_tahsis.politika_olcumu(
                cfg, dunya, gor, lot_gor, blok.teklifler, blok.mat, tepki,
                sonuc, kolonlar, durum, sevk_hizi, carpan))
            golge_bloklari.append(
                ev_tahsis.golge_fiyat_tablosu(cfg, dunya, lot_gor, sonuc))

    olcumler = {ad: ev_tahsis.olcum_birlestir(ad, parcalar[ad])
                for ad in alloc.POLITIKALAR}
    golge = pl.concat([g for g in golge_bloklari if g.height]) if golge_bloklari \
        else pl.DataFrame()
    return M5Ciktisi(dunya=dunya, originler=m4.olcum_originleri, lotlar=lotlar,
                     sonuclar=sonuclar, olcumler=olcumler, golge=golge,
                     carpan=carpan,
                     zaman={"m5_sn": round(time.perf_counter() - t0, 2)})


def m5_ihlaller(c: M5Ciktisi, cfg: Config, m4: M4Ciktisi) -> dict:
    """M5'in degismezleri. Hepsi SIFIR olmak zorunda; her kosuda uretilir.

    `stok_asimi`     : stok ayiran bir politika lotun kalanindan fazlasini
                       PLANLAYAMAZ. `ranking_only` bu kontrole tabi degil -
                       asmasi zaten olculen sey (SPEC M5 (a)).
    `raf_omru_ihlali`: hicbir politika, hicbir rejimde temizlik tabaninin
                       altindaki lottan teklif cikaramaz (D9 gevsetmedir,
                       kaldirma degil).
    `temizlik_disi_kisa_lot`: temizlik rejimi KAPALI politikalar normal raf
                       omru tabaninin altina inemez.
    `hiz_kuplaji_ihlali`: SPEC 2.5 adet tavani asilmis mi.
    `kredi_asimi`    : LP kisiti aciksa DBS limiti asilamaz (D6).
    """
    k = cfg.politika.kisit
    tk = cfg.tahsis.temizlik
    ihlal = {ad: 0 for ad in ("stok_asimi", "raf_omru_ihlali",
                              "temizlik_disi_kisa_lot", "hiz_kuplaji_ihlali",
                              "kredi_asimi", "izinsiz_kol_secildi")}
    for ad, pol in alloc.POLITIKALAR.items():
        for blok, sonuc in zip(m4.bloklar, c.sonuclar[ad]):
            lot_gor = c.lotlar[sonuc.t]
            secilen = sonuc.teklif_maskesi
            if not secilen.any():
                continue
            i = np.flatnonzero(secilen)
            l = sonuc.lot[i]
            soguk = blok.teklifler["soguk_zincir"].to_numpy().astype(bool)[i]
            carpan_soguk = np.where(soguk, k.soguk_zincir_raf_omru_carpani, 1.0)
            kalan_gun = lot_gor.kalan_gun[l]
            taban = (tk.asgari_kalan_raf_omru_gun if pol.temizlik_rejimi
                     else k.asgari_kalan_raf_omru_gun) * carpan_soguk
            ihlal["raf_omru_ihlali"] += int((kalan_gun < taban - 1e-9).sum())
            if not pol.temizlik_rejimi:
                ihlal["temizlik_disi_kisa_lot"] += int(
                    (kalan_gun < k.asgari_kalan_raf_omru_gun * carpan_soguk - 1e-9).sum())
            ihlal["izinsiz_kol_secildi"] += int(
                (~blok.mat.izinli[i, sonuc.kol[i]]).sum())

            nominal = blok.mat.adet[i, sonuc.kol[i]] + blok.mat.bedava[i, sonuc.kol[i]]
            if pol.temizlik_hiz_kuplaji:
                hiz = (blok.teklifler["hiz_tahmini"].to_numpy()[i]
                       * cfg.politika.aday.hiz_telafi_katsayisi)
                tavan = alloc.azami_teklif_adedi(cfg, hiz, kalan_gun)
                ihlal["hiz_kuplaji_ihlali"] += int((nominal > tavan + 1e-6).sum())

            if pol.stok_ayirma:
                cekilis = np.zeros(lot_gor.L)
                np.add.at(cekilis, l, _planlanan_cekilis(blok, sonuc, i, c.carpan))
                ihlal["stok_asimi"] += int((cekilis > lot_gor.adet + 1e-6).sum())
            if pol.lp and cfg.tahsis.lp.kredi_kisiti:
                ihlal["kredi_asimi"] += _m5_kredi_asimi(c, cfg, blok, sonuc)
    return ihlal


def _planlanan_cekilis(blok: OriginBlogu, sonuc, i: np.ndarray,
                       carpan: float) -> np.ndarray:
    """Secilen kollarin lottan ARTIMSAL adet rezervasyonu.

    Tahsis katmani artimsal rezerve ediyor (policy/allocate.py
    `taban_talebini_dus`), dolayisiyla degismez kontrolu de artimsal olmak
    zorunda. Brut karsilastirilirsa her stok ayiran politika sahte ihlal verir.
    """
    kol = sonuc.kol[i]
    nominal = blok.mat.adet[i, kol] + blok.mat.bedava[i, kol]
    taban = blok.p_x[i, scorer.TEKLIF_YOK] * blok.mat.adet[i, scorer.TEKLIF_YOK]
    return np.maximum(carpan * (blok.p_x[i, kol] * nominal - taban), 0.0)


def _m5_kredi_asimi(c: M5Ciktisi, cfg: Config, blok: OriginBlogu, sonuc) -> int:
    k = cfg.politika.kisit
    dbs = c.dunya.eczaneler["dbs_limiti"].to_numpy().astype(float)
    risk = c.dunya.eczaneler["vade_riski_skoru"].to_numpy().astype(float)
    tavan = dbs * k.kredi_kullanim_tavani * (1.0 - risk * k.vade_riski_cezasi)
    n = sonuc.kol.size
    dsf = c.dunya.dsf[blok.teklifler["sku_idx"].to_numpy()]
    tutar = np.where(sonuc.teklif_maskesi,
                     blok.p_x[np.arange(n), sonuc.kol] * c.carpan
                     * blok.mat.adet[np.arange(n), sonuc.kol] * dsf, 0.0)
    toplam = np.zeros(c.dunya.P)
    np.add.at(toplam, blok.teklifler["eczane_idx"].to_numpy(), tutar)
    return int((toplam > np.maximum(tavan - blok.gor.acik_bakiye, 0.0) + 1e-6).sum())


def m5_duz_metrikler(c: M5Ciktisi, cfg: Config, m4: M4Ciktisi) -> dict:
    """Sweep tablosuna giren duz M5 metrikleri."""
    duz: dict[str, float] = {}
    for ad, o in c.olcumler.items():
        duz.update({
            f"m5.{ad}.teklif_sayisi": o.teklif_sayisi,
            f"m5.{ad}.teklif_sayisi_temizlik": o.teklif_sayisi_temizlik,
            f"m5.{ad}.ortalama_mf_temizlik": o.ortalama_mf_temizlik,
            f"m5.{ad}.ortalama_mf_normal": o.ortalama_mf_normal,
            f"m5.{ad}.beklenen_artimsal_marj": o.beklenen_artimsal_marj,
            f"m5.{ad}.brut_marj": o.brut_marj,
            f"m5.{ad}.net_marj": o.net_marj,
            f"m5.{ad}.talep_adet": o.talep_adet,
            f"m5.{ad}.karsilanan_adet": o.karsilanan_adet,
            f"m5.{ad}.karsilanmayan_adet": o.karsilanmayan_adet,
            f"m5.{ad}.karsilama_orani": o.karsilama_orani,
            f"m5.{ad}.stockout_sayisi": o.stockout_sayisi,
            f"m5.{ad}.stockout_eczane_sayisi": o.stockout_eczane_sayisi,
            f"m5.{ad}.teslim_adet": o.teslim_adet,
            f"m5.{ad}.iade_adet": o.iade_adet,
            f"m5.{ad}.imha_adet": o.imha_adet,
            f"m5.{ad}.imha_adet_temizlik": o.imha_adet_temizlik,
            f"m5.{ad}.teslim_adet_temizlik": o.teslim_adet_temizlik,
            f"m5.{ad}.iade_imhasi_adet": o.iade_imhasi_adet,
            f"m5.{ad}.imha_islem_maliyeti": o.imha_islem_maliyeti,
            f"m5.{ad}.iade_islem_maliyeti": o.iade_islem_maliyeti,
            f"m5.{ad}.iade_kredi_tutari": o.iade_kredi_tutari,
            f"m5.{ad}.imha_batik_tutari": o.imha_batik_tutari,
            f"m5.{ad}.memnuniyet_proxy": o.memnuniyet_proxy,
            f"m5.{ad}.sow_kaybi_iade": o.sow_kaybi_iade,
            f"m5.{ad}.sow_kaybi_stoksuzluk": o.sow_kaybi_stoksuzluk,
            f"m5.{ad}.ortalama_teslim_raf_omru": o.ortalama_teslim_raf_omru,
            f"m5.{ad}.butunluk_acigi": o.butunluk_acigi,
            f"m5.{ad}.kesirli_sutun": o.kesirli_sutun,
            f"m5.{ad}.kolon_sayisi": o.kolon_sayisi,
        })
    # --- (a) cikis kriteri: ranking-only vs LP ---
    r, l = c.olcumler["ranking_only"], c.olcumler["lp"]
    duz["m5.a.stockout_farki"] = l.stockout_sayisi - r.stockout_sayisi
    duz["m5.a.karsilanmayan_farki"] = l.karsilanmayan_adet - r.karsilanmayan_adet
    duz["m5.a.net_marj_farki"] = l.net_marj - r.net_marj
    duz["m5.a.brut_marj_farki"] = l.brut_marj - r.brut_marj
    # --- (b) cikis kriteri: temizlik yok / kor iskonto / hedefli ---
    y, ko, h = c.olcumler["lp"], c.olcumler["kor_iskonto"], c.olcumler["hedefli_temizlik"]
    duz["m5.b.imha_kor_farki"] = ko.imha_adet - y.imha_adet
    duz["m5.b.imha_hedefli_farki"] = h.imha_adet - y.imha_adet
    duz["m5.b.imha_temizlik_kor_farki"] = ko.imha_adet_temizlik - y.imha_adet_temizlik
    duz["m5.b.imha_temizlik_hedefli_farki"] = h.imha_adet_temizlik - y.imha_adet_temizlik
    duz["m5.b.iade_kor_farki"] = ko.iade_adet - y.iade_adet
    duz["m5.b.iade_hedefli_farki"] = h.iade_adet - y.iade_adet
    duz["m5.b.net_marj_kor_farki"] = ko.net_marj - y.net_marj
    duz["m5.b.net_marj_hedefli_farki"] = h.net_marj - y.net_marj
    duz["m5.b.memnuniyet_kor_farki"] = ko.memnuniyet_proxy - y.memnuniyet_proxy
    duz["m5.b.memnuniyet_hedefli_farki"] = h.memnuniyet_proxy - y.memnuniyet_proxy
    # --- gölge fiyatlar (D9) ---
    for ad in ("lp", "hedefli_temizlik"):
        alt = c.golge.filter(pl.col("politika") == ad) if c.golge.height else c.golge
        duz.update({f"m5.golge.{ad}.{k}": v
                    for k, v in ev_tahsis.golge_fiyat_ozeti(alt).items()})
    duz.update({f"ihlal.m5_{ad}": float(v)
                for ad, v in m5_ihlaller(c, cfg, m4).items()})
    return duz


# --------------------------------------------------------------------------
# M6: off-policy degerlendirme + kapali dongu rollout
# --------------------------------------------------------------------------
@dataclass
class M6Ciktisi:
    olcum_originleri: list[int]
    veri: ev_ope.LoglanmisVeri
    prop: ev_ope.PropensityCiktisi
    oracle: KarsiOlgusalOracle
    hedefler: dict[str, np.ndarray]
    ope_sonuclari: dict[str, ev_ope.OPESonucu]
    tekrar_sapmalari: dict[str, dict]
    denetimler: list
    ayristirmalar: dict
    ozdeslik: dict
    rollout: dict[str, rl.RolloutOlcumu]
    kopruler: list
    gecikmeli: dict[str, dict]
    satir_sayisi: float
    egitim_ortusmesi: int
    zaman: dict


def m6_kayit_verisi(cfg: Config, m4: M4Ciktisi) -> tuple:
    """Olcum origin'lerinde kayit politikasini `tekrar_sayisi` kez kosar.

    M4 kayit kosusunu EGITIM origin'lerinde yapiyordu (ogreticiyi beslemek
    icin). M6 ayni politikayi OLCUM origin'lerinde kosar: degerlendirilecek
    hedef politikalar orada seciliyor ve OPE ancak hedefin degerlendirildigi
    satirlarda tanimli.

    TEKRAR NEDEN VAR. Tek bir kayit kosusu tahmincinin BIR cekilisidir; onun
    ne kadar oynak oldugu gorulemez. `tekrar_sayisi` bagimsiz kosu, sapma
    ayristirmasinin "varyans" kalemini bootstrap'a degil GERCEK tekrara
    baglar (eval/ope.py::tekrar_sapmasi).
    """
    carpan = m4.carpan
    parcalar, oracle_parcalari = [], []
    q_parcalari = []
    for r in range(cfg.ope.kayit.tekrar_sayisi):
        # Her tekrar KENDI seed bankasi: ayni satirlar, farkli aksiyon ve
        # farkli kabul zarlari.
        seedler = SeedBank(cfg.ope.kayit.seed + r)
        for blok, tepki in zip(m4.bloklar, m4.tepkiler):
            n = blok.teklifler.height
            if n == 0:
                continue
            idx = np.arange(n)
            kayit = bandit.kayit_kosusu(m4.dunya, cfg, seedler, blok.mat.uzay,
                                        blok.teklifler, blok.mat.izinli, blok.t)
            _, miktar = sonuc_ornekle(cfg, seedler, tepki, kayit.kol, blok.t)
            ecz = blok.teklifler["eczane_idx"].to_numpy()
            parcalar.append(ev_ope.LoglanmisVeri(
                X=blok.X, kol=kayit.kol, propensity=kayit.propensity,
                pi_log=kayit.pi, odul=miktar * blok.mat.marj[idx, kayit.kol],
                izinli=blok.mat.izinli, eczane_idx=ecz,
                origin=np.full(n, blok.t, dtype=np.int32),
                tekrar=np.full(n, r, dtype=np.int32)))
            if r == 0:
                oracle_parcalari.append(ev_oracle.karsi_olgusal_oracle(
                    tepki, blok.mat, carpan, ecz))
                q_parcalari.append(blok.p_t * blok.mat.marj * carpan)
    veri = ev_ope.birlestir(parcalar)
    # Oracle ve sonuc modeli satira bagli, tekrara degil: bir kez kurulup
    # tekrar sayisi kadar tekrarlanir.
    tek = ev_oracle.oracle_birlestir(oracle_parcalari)
    R = cfg.ope.kayit.tekrar_sayisi
    oracle = KarsiOlgusalOracle(
        deger_matrisi=np.tile(tek.deger_matrisi, (R, 1)),
        izinli=np.tile(tek.izinli, (R, 1)),
        eczane_idx=np.tile(tek.eczane_idx, R))
    q = np.tile(np.vstack(q_parcalari), (R, 1))
    return veri, oracle, q, tek.n


def m6_hedefler(cfg: Config, m4: M4Ciktisi, R: int) -> dict[str, np.ndarray]:
    """Politika -> [n] hedef kol vektoru (tekrarlar boyunca tekrarlanmis)."""
    hedefler: dict[str, np.ndarray] = {}
    for ad, liste in m4.secimler.items():
        if ad.startswith("oracle_"):
            continue          # oracle politikalari OPE'nin hedefi olamaz
        hedefler[ad] = np.tile(np.concatenate([s.kol for s in liste]), R)
    for ad in AGRESIF_POLITIKALARI:
        secimler = [agresif_politika(cfg, b.mat, b.teklifler, ad) for b in m4.bloklar]
        hedefler[ad] = np.tile(np.concatenate([s.kol for s in secimler]), R)
    return hedefler


def m6_ozdeslik_testi(cfg: Config, veri: ev_ope.LoglanmisVeri,
                      prop: ev_ope.PropensityCiktisi, q: np.ndarray,
                      oracle: KarsiOlgusalOracle) -> dict:
    """OPE'nin en temel ozdesligi: hedef = KAYIT politikasi.

    Kayit politikasi stokastiktir, dolayisiyla deterministik hedef kurali
    kurulamaz; bunun yerine gozlenen aksiyonun kendisi hedef alinir. O zaman
    w_i = 1 / pi(a_i) x 1[a_i = a_i] degil, tanim geregi 1 olmalidir --
    yani IPS dogrudan gozlenen odulun ortalamasidir ve kayit politikasinin
    GERCEK degerine yakinsamalidir.

    Tutmuyorsa hata tahmincide degil loglamadadir (yanlis propensity, yanlis
    odul olcegi) ve butun M6 tablosu cope gider. Bu yuzden ayri bir kalem.
    """
    gercek = oracle.karisim_degeri(veri.pi_log)
    gozlenen = float(veri.odul.mean()) if veri.n else float("nan")
    ips_kendi = ev_ope.ips(veri.odul, np.where(
        veri.kol == veri.kol, 1.0, 0.0))     # w = 1 (hedef = gozlenen aksiyon)
    return {
        "kayit_politikasi_oracle": gercek,
        "kayit_politikasi_gozlenen": gozlenen,
        "kayit_politikasi_ips": ips_kendi,
        "ozdeslik_sapmasi": gozlenen - gercek,
        "ozdeslik_sapma_yuzde": (abs(gozlenen - gercek) / abs(gercek) * 100.0
                                 if abs(gercek) > 1e-12 else float("nan")),
    }


def _rollout_karar_verici(cfg: Config, m4: M4Ciktisi, politika_adi: str):
    """Karar haftasinda tam politika hattini kosan geri cagirma.

    Hat M4/M5 ile AYNI: aday havuzu -> kisit vetosu -> kol matrisleri ->
    ozellikler -> CATE -> aksiyon secimi -> (lp'de) tahsis -> kredi son
    kontrolu. Rollout icin kisaltilmis bir kopya YOK; kisayol olsaydi
    "kapali dongude olculen politika" ile "M4'te olculen politika" ayni sey
    olmazdi ve iki sayi karsilastirilamazdi.

    Fark tek yerde: gorunum, taban dunyanin parquet'inden degil rollout'un
    KENDI kayitlarindan kuruluyor (sim/rollout.py::canli_aday_dunyasi).
    """
    def karar(t: int, aday_dunya) -> "rl.TeklifPlani | None":
        td = replace(m4.td, aday=aday_dunya)
        gor = pol_aday.gorunum_kur(aday_dunya, cfg, t)
        havuz, skorlar = pol_aday.aday_havuzu(aday_dunya, cfg, gor)
        tumu = kisit_uygula(aday_dunya, cfg, gor, havuz)
        aday = tumu.filter(~pl.col("vetolu"))
        if aday.height == 0:
            return None
        mat = scorer.teklif_matrisleri(aday_dunya, cfg, aday)
        X, _, _ = ft.ozellik_matrisi(td, cfg, gor, aday, skorlar)
        pi = bandit.kayit_olasiliklari(aday_dunya, cfg, mat.uzay, aday, mat.izinli)
        p_t = m4.t_ogr.olasilik(X)
        p_x = m4.x_ogr.olasilik(X, pi)

        lot_id = None
        if politika_adi == "lp":
            lot_gor = alloc.taban_talebini_dus(
                aday_dunya, cfg, alloc.lot_gorunumu(aday_dunya, cfg, gor),
                aday, mat, p_x, m4.carpan)
            sonuc, _ = alloc.tahsis_et(aday_dunya, cfg, gor, lot_gor, aday, mat,
                                       p_x, m4.carpan, alloc.POLITIKALAR["lp"])
            secim = scorer.Secim("lp", sonuc.kol.astype(np.int32),
                                 np.zeros(aday.height))
            lot_id = np.array([lot_gor.lot_id[l] if l >= 0 else None
                               for l in sonuc.lot], dtype=object)
        elif politika_adi in AGRESIF_POLITIKALARI:
            secim = agresif_politika(cfg, mat, aday, politika_adi)
        else:
            secim = _gozlemlenebilir_politikalar(
                cfg, mat, aday, p_t, p_x, m4.carpan)[politika_adi]

        # D6: kisit katmani aksiyon seciminden SONRA da veto yetkisini korur.
        veto = scorer.kredi_son_kontrolu(aday_dunya, cfg, gor, aday, mat, secim)
        kol = np.where(veto, scorer.TEKLIF_YOK, secim.kol).astype(np.int32)
        if lot_id is not None:
            lot_id = np.where(veto, None, lot_id)
        return rl.TeklifPlani(teklifler=aday, mat=mat, kol=kol, lot_id=lot_id)
    return karar


def m6_rollout(cfg: Config, m4: M4Ciktisi) -> dict[str, rl.RolloutOlcumu]:
    """Her politika icin dunyayi bastan kosar, `baslangic_hafta`da dallanir.

    ISINMA HER POLITIKA ICIN YENIDEN KOSULUR. Durumu kopyalamak (deepcopy)
    daha hizli olurdu ama lot kuyruklari ve miad kovalari derin yapilar;
    yeniden kosmak hem daha ucuz hem TEKRAR URETILEBILIRLIGI mekanik olarak
    garanti ediyor: her politika tam olarak ayni tohumdan ayni isinmayi gorur.
    """
    r = cfg.ope.rollout
    cikti: dict[str, rl.RolloutOlcumu] = {}
    for ad in r.politikalar:
        durum = dunya_kur(cfg, SeedBank(cfg.profil.temel_seed))
        for _ in range(r.baslangic_hafta):
            hafta_adimi(durum)
        karar = (None if ad == "teklif_yok"
                 else _rollout_karar_verici(cfg, m4, ad))
        cikti[ad] = rl.rollout_kos(
            cfg, durum, m4.evren, m4.dunya.eczaneler, m4.dunya.urunler,
            durum.ecz.latent, karar or (lambda t, d: None), ad)
    return cikti


def m6_boru_hatti(cfg: Config, m4: M4Ciktisi) -> M6Ciktisi:
    """Offline tahminciler + kapali dongu + iki sayinin koprusu."""
    t0 = time.perf_counter()
    R = cfg.ope.kayit.tekrar_sayisi
    veri, oracle, q, satir_sayisi = m6_kayit_verisi(cfg, m4)
    prop = ev_ope.propensity_hazirla(cfg, veri)
    hedefler = m6_hedefler(cfg, m4, R)
    ozdeslik = m6_ozdeslik_testi(cfg, veri, prop, q, oracle)
    zaman = {"m6_kayit_sn": round(time.perf_counter() - t0, 2)}

    t0 = time.perf_counter()
    sonuclar, sapmalar, denetimler, ayristirmalar = {}, {}, [], {}
    for ad, kol in hedefler.items():
        sonuclar[ad] = ev_ope.degerlendir(cfg, ad, veri, kol, prop, q)
        sapmalar[ad] = ev_ope.tekrar_sapmasi(cfg, ad, veri, kol, prop, q)
        denetimler += ev_rapor.tahminci_denetimi(ad, sonuclar[ad], oracle.deger(kol))
        ayristirmalar[ad] = ev_rapor.sapma_ayristir(cfg, ad, veri, kol, prop,
                                                    q, oracle)
    zaman["m6_ope_sn"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    rollout = m6_rollout(cfg, m4)
    zaman["m6_rollout_sn"] = round(time.perf_counter() - t0, 2)

    r = cfg.ope.rollout
    taban = rollout["teklif_yok"]
    kopruler, gecikmeli = [], {}
    for ad, o in rollout.items():
        if ad == "teklif_yok":
            continue
        gecikmeli[ad] = ev_rapor.gecikmeli_bedel(o, taban)
        # Ufuk kesiminin yanliligi: sonda rafta duran fazla mal. Net marja
        # KATILMAZ, ayri raporlanir (eval/report.py::terminal_duzeltme).
        for ufuk in r.raporlanan_ufuklar:
            gecikmeli[ad].update({
                f"{k}@{ufuk}": v for k, v
                in ev_rapor.terminal_duzeltme(o, taban, ufuk).items()})
        if ad not in hedefler:
            continue
        artimsal_satir = {t: (sonuclar[ad].deger(t)
                              - sonuclar["teklif_yok"].deger(t))
                          for t in ev_ope.TAHMINCILER}
        for ufuk in r.raporlanan_ufuklar:
            karar_haftasi = min(ufuk, r.teklif_penceresi_hafta)
            karar_haftasi = int(np.ceil(karar_haftasi / r.karar_araligi_hafta))
            for tahminci in ("ips", "snips", "dr"):
                kopruler.append(ev_rapor.kopru(
                    ad, tahminci, ufuk, artimsal_satir[tahminci], satir_sayisi,
                    karar_haftasi,
                    o.net_marj_ufukta(ufuk) - taban.net_marj_ufukta(ufuk),
                    taban.net_marj_ufukta(ufuk)))

    # Egitim penceresiyle ortusme: sizinti gizli kalmasin (core/config.py
    # `_m6_rollout_kilidi` bunu hata saymiyor, sayiyla raporluyor).
    son_egitim = max(m4.egitim_originleri) if m4.egitim_originleri else -1
    ortusme = max(0, min(son_egitim, r.baslangic_hafta + r.ufuk_hafta - 1)
                  - r.baslangic_hafta + 1)

    return M6Ciktisi(
        olcum_originleri=m4.olcum_originleri, veri=veri, prop=prop,
        oracle=oracle, hedefler=hedefler, ope_sonuclari=sonuclar,
        tekrar_sapmalari=sapmalar, denetimler=denetimler,
        ayristirmalar=ayristirmalar, ozdeslik=ozdeslik, rollout=rollout,
        kopruler=kopruler, gecikmeli=gecikmeli, satir_sayisi=float(satir_sayisi),
        egitim_ortusmesi=int(ortusme), zaman=zaman)


def m6_duz_metrikler(c: M6Ciktisi, cfg: Config) -> dict:
    """Sweep tablosuna giren duz M6 metrikleri."""
    duz: dict[str, float] = {}
    duz.update({f"m6.ozdeslik.{k}": float(v) for k, v in c.ozdeslik.items()})
    duz["m6.propensity.ortalama_mutlak_hata"] = c.prop.ortalama_mutlak_hata
    duz["m6.propensity.kalibrasyon_hatasi"] = c.prop.kalibrasyon_hatasi
    duz["m6.propensity.log_orani"] = c.prop.log_orani_ortalamasi
    duz["m6.satir_sayisi"] = c.satir_sayisi
    duz["m6.rollout.egitim_ortusmesi_hafta"] = float(c.egitim_ortusmesi)

    for ad, s in c.ope_sonuclari.items():
        for t in ev_ope.TAHMINCILER:
            duz[f"m6.offline.{ad}.{t}"] = s.deger(t)
        duz[f"m6.offline.{ad}.oracle"] = c.oracle.deger(c.hedefler[ad])
        th = s.teshis
        duz.update({
            f"m6.teshis.{ad}.ess_orani": th.ess_orani,
            f"m6.teshis.{ad}.eslesme_orani": th.eslesme_orani,
            f"m6.teshis.{ad}.agirlik_azami": th.agirlik_azami,
            f"m6.teshis.{ad}.kirpilan_kutle_orani": th.kirpilan_kutle_orani,
            f"m6.teshis.{ad}.ortusme_ihlali_orani": th.ortusme_ihlali_orani,
            f"m6.teshis.{ad}.dusuk_destege_giden_satir_orani":
                th.dusuk_destege_giden_satir_orani,
        })
        duz.update({f"m6.sapma_sd.{ad}.{k}": v
                    for k, v in c.tekrar_sapmalari[ad].items()})

    for d in c.denetimler:
        duz[f"m6.denetim.{d.politika}.{d.tahminci}.sapma"] = d.sapma
        duz[f"m6.denetim.{d.politika}.{d.tahminci}.sapma_yuzde"] = d.sapma_yuzde
        duz[f"m6.denetim.{d.politika}.{d.tahminci}.aralik_kapsiyor"] = float(
            d.araligi_kapsiyor)

    for ad, a in c.ayristirmalar.items():
        duz.update({
            f"m6.ayristirma.{ad}.toplam_sapma": a.toplam_sapma,
            f"m6.ayristirma.{ad}.varyans": a.varyans_kalemi,
            f"m6.ayristirma.{ad}.kirpma": a.kirpma_kalemi,
            f"m6.ayristirma.{ad}.propensity": a.propensity_kalemi,
            f"m6.ayristirma.{ad}.artik": a.artik,
            f"m6.ayristirma.{ad}.ortusme_kor_deger_payi": a.ortusme_kor_deger_payi,
            f"m6.ayristirma.{ad}.kirpma_ortusmeden": a.kirpma_ortusmeden_gelen_pay,
            f"m6.ayristirma.{ad}.ekstrap_zayif": a.ekstrapolasyon_dusuk_destek_hatasi,
            f"m6.ayristirma.{ad}.ekstrap_guclu": a.ekstrapolasyon_yuksek_destek_hatasi,
        })

    taban = c.rollout["teklif_yok"]
    for ad, o in c.rollout.items():
        for ufuk in cfg.ope.rollout.raporlanan_ufuklar:
            v = o.net_marj_ufukta(ufuk)
            t = taban.net_marj_ufukta(ufuk)
            duz[f"m6.online.{ad}.net_marj@{ufuk}"] = v
            duz[f"m6.online.{ad}.artimsal@{ufuk}"] = v - t
            duz[f"m6.online.{ad}.artimsal_yuzde@{ufuk}"] = (
                (v - t) / abs(t) * 100.0 if abs(t) > 1e-9 else float("nan"))
        # Kosulan ufkun SONUNDAKI deger. `raporlanan_ufuklar`dan bagimsiz
        # oldugu icin `ope.rollout.ufuk_hafta` taramasinin tablo sutunu budur:
        # ufuk degistikce ayni ad farkli bir hafta sayisini gosterir ve
        # SPEC 5'in "4 hafta vs uzun ufuk" karsitligi tek sutunda okunur.
        v_son = o.net_marj_ufukta(cfg.ope.rollout.ufuk_hafta)
        t_son = taban.net_marj_ufukta(cfg.ope.rollout.ufuk_hafta)
        duz[f"m6.online.{ad}.artimsal_son"] = v_son - t_son
        duz[f"m6.online.{ad}.artimsal_yuzde_son"] = (
            (v_son - t_son) / abs(t_son) * 100.0 if abs(t_son) > 1e-9 else float("nan"))
        duz[f"m6.online.{ad}.hafta_sayisi"] = float(len(o.haftalar))
        duz[f"m6.online.{ad}.teklif_sayisi"] = float(o.seri("teklif_sayisi").sum())
        duz[f"m6.online.{ad}.kabul_sayisi"] = float(o.seri("kabul_sayisi").sum())
        duz[f"m6.online.{ad}.iade_adet"] = float(o.seri("iade_adet").sum())
        duz[f"m6.online.{ad}.imha_adet"] = float(o.seri("imha_adet").sum())
        duz[f"m6.online.{ad}.bedava_adet"] = float(o.seri("bedava_adet").sum())
        duz[f"m6.online.{ad}.sow_son"] = (float(o.seri("sow_ortalama")[-1])
                                          if o.haftalar else float("nan"))
    for ad, g in c.gecikmeli.items():
        duz.update({f"m6.gecikmeli.{ad}.{k}": float(v) for k, v in g.items()})

    for k in c.kopruler:
        on = f"m6.kopru.{k.politika}.{k.tahminci}@{k.ufuk}"
        duz[f"{on}.offline_tl"] = k.offline_artimsal_tl
        duz[f"{on}.online_tl"] = k.online_artimsal_tl
        duz[f"{on}.offline_yuzde"] = k.offline_yuzde
        duz[f"{on}.online_yuzde"] = k.online_yuzde
        duz[f"{on}.ufuk_kalemi"] = k.ufuk_kalemi
    return duz


# --------------------------------------------------------------------------
# M7: senaryo yorumu (D3/D4) + LLM katmani (D8) + eval harness
# --------------------------------------------------------------------------
@dataclass
class M7Ciktisi:
    kosu: ag_senaryo.SenaryoKosusu
    baglam: ag_arac.AjanBaglami
    harness: hr.HarnessSonucu
    vaka_sayisi: int
    atlanan_vakalar: list[str]
    zaman: dict


# Sweep icinde kosulan vaka kaynaklari. `kayitli` DISARIDA ve sebebi
# yapisal: kayit istemle birlikte dogrulaniyor (agent/client.py), knob
# taramasi ise istemi degistiriyor. Sweep icinde oynatilsaydi her knob
# degeri "kayit bayat" diye kalir ve tablo bir seyi olcmez, kaydin
# tazeligini olcerdi.
SWEEP_VAKA_KAYNAKLARI: tuple[str, ...] = ("sablon",)


def m7_boru_hatti(cfg: Config, m4: M4Ciktisi, kosu_adi: str,
                  kok: Path) -> M7Ciktisi:
    """Senaryolari kosar, ajan baglamini kurar, harness'i calistirir.

    M4'u YENIDEN KOSMAZ (M5/M6 ile ayni disiplin): ayni ogreniciler, ayni
    aday satirlari. Senaryo katmaninin urettigi fark rejim farkidir, model
    farki degil.
    """
    t0 = time.perf_counter()
    hb = hr.baglam_hazirla(cfg, kosu_adi, kok, m4=m4)
    tum = hr.vakalari_yukle()
    vakalar = [v for v in tum if v.kaynak in SWEEP_VAKA_KAYNAKLARI]
    atlanan = [v.ad for v in tum if v.kaynak not in SWEEP_VAKA_KAYNAKLARI]
    sonuc = hr.harness_kos(hb, vakalar)
    return M7Ciktisi(kosu=hb.kosu, baglam=hb.baglam, harness=sonuc,
                     vaka_sayisi=len(vakalar), atlanan_vakalar=atlanan,
                     zaman={"m7_sn": round(time.perf_counter() - t0, 2)})


def m7_duz_metrikler(c: M7Ciktisi) -> dict:
    """Sweep tablosuna giren duz M7 metrikleri: senaryo + harness."""
    return ag_senaryo.duz_metrikler(c.kosu) | hr.duz_metrikler(c.harness)


def kosu_yap(cfg: Config, kosu_id: str, gecersiz: dict, veri_tut: bool,
             tahmin_yaz: bool, kok: Path | None = None,
             asamalar: tuple[str, ...] = ASAMALAR) -> dict:
    dizin = (kok or KOSU_DIZINI) / kosu_id
    if dizin.exists():
        shutil.rmtree(dizin)
    dizin.mkdir(parents=True)

    t0 = time.perf_counter()
    dunya = Run("dunya", kok=dizin)
    dunya_manifest = dunya_yaz(cfg, dunya, gecersiz)
    dunya_sn = round(time.perf_counter() - t0, 2)

    zaman = {"dunya_sn": dunya_sn}
    duz: dict = {}
    icerik = {
        "kosu_id": kosu_id,
        "profil": cfg.profil.ad,
        "config_hash": cfg.hash(),
        "dunya_hash": dunya_manifest["dunya_hash"],
        "temel_seed": cfg.profil.temel_seed,
        "asamalar": list(asamalar),
        "knob": {k: str(v) for k, v in gecersiz.items()},
    }

    b = sonuc = panel_ozeti = None
    if "m2" in asamalar:
        b = boru_hatti(cfg, "dunya", dizin)
        zaman.update(b.zaman)
        izg, panel, bolme, o_test = b.izgara, b.panel, b.bolme, b.o_test
        ufuk, H = cfg.tukenme.hedef.ufuk_hafta, cfg.tukenme.hedef.karar_ufku_hafta

        sonuc, tahmin_kolonlari = {}, {}
        for ad, tah in b.tahminler.items():
            sonuc[ad] = degerlendir(ad, tah, o_test, b.y_test, cfg)
            tahmin_kolonlari[f"{ad}_olasilik"] = tah.olasilik
            tahmin_kolonlari[f"{ad}_skor"] = tah.skor
            tahmin_kolonlari[f"{ad}_sure"] = tah.tukenme_hafta
        tahminciler = b.tahminciler

        gecerli, y_oracle, T = _oracle_hedefleri(o_test, ufuk, H)
        panel_ozeti = {
            "hucre_sayisi": int(izg.talep.shape[0]),
            "origin_sayisi": int(np.unique(panel.origin).size),
            "egitim_satiri": int(bolme.egitim.size),
            "test_satiri": int(bolme.test.size),
            "bolme_sinir_haftasi": bolme.sinir_hafta,
            "olcum_satiri": int(gecerli.sum()),
            "origin_da_zaten_stoksuz_orani": float((~o_test.canli).mean()),
            "listeden_dusme_sansuru_orani": float(o_test.rakip_sansur.mean()),
            "gercek_tukenme_taban_orani": float(y_oracle[gecerli].mean()),
            "gozlemlenebilir_taban_orani": float(b.y_test.mean()),
            "hazard_egitim_periyot_satiri": int(getattr(tahminciler[-1], "egitim_satiri", 0)),
            "kural_literal_n_gun": int(tahminciler[1].n_gun),
            "kural_secilen_n_gun": int(tahminciler[2].n_gun),
        }
        duz.update({f"{ad}.{k}": v for ad, m in sonuc.items()
                    for k, v in m.items() if k != "ad"})
        duz.update({f"panel.{k}": v for k, v in panel_ozeti.items()})
        icerik.update({"panel": panel_ozeti, "tahminciler": sonuc})

    if "m3" in asamalar:
        m3 = m3_boru_hatti(cfg, "dunya", dizin)
        zaman.update(m3.zaman)
        duz.update(m3_duz_metrikler(m3, cfg))
        icerik["m3"] = {
            "originler": m3.originler,
            "aday_satiri": int(m3.teklifler.height),
            "veto": m3.veto.to_dicts(),
            "ihlal": m3_ihlaller(m3, cfg),
        }

    m4 = None
    if {"m4", "m5", "m6", "m7"} & set(asamalar):
        m4 = m4_boru_hatti(cfg, "dunya", dizin)
        zaman.update(m4.zaman)
    if "m4" in asamalar:
        duz.update(m4_duz_metrikler(m4, cfg))
        icerik["m4"] = {
            "egitim_originleri": m4.egitim_originleri,
            "olcum_originleri": m4.olcum_originleri,
            "egitim_satiri": m4.egitim_satiri,
            "ozellik_sayisi": len(m4.ozellik_adlari),
            "politikalar": {ad: vars(o) for ad, o in m4.olcumler.items()},
            "ayristirma": m4.ayristirma,
            "heterojenlik": vars(m4.teshis),
            "cate": {k: v for k, v in m4.cate.items() if k != "kazanc_egrisi"},
            "destek": m4.destek,
            "guven": m4.guven,
            "ihlal": m4_ihlaller(m4, cfg),
        }

    if "m5" in asamalar:
        m5 = m5_boru_hatti(cfg, m4, "dunya", dizin)
        zaman.update(m5.zaman)
        duz.update(m5_duz_metrikler(m5, cfg, m4))
        icerik["m5"] = {
            "originler": m5.originler,
            "lot_sayisi": {str(t): int(l.L) for t, l in m5.lotlar.items()},
            "politikalar": {ad: {k: v for k, v in vars(o).items() if k != "sapma"}
                            for ad, o in m5.olcumler.items()},
            "golge": {ad: ev_tahsis.golge_fiyat_ozeti(
                m5.golge.filter(pl.col("politika") == ad)) for ad in alloc.POLITIKALAR},
            "ihlal": m5_ihlaller(m5, cfg, m4),
        }

    if "m6" in asamalar:
        m6 = m6_boru_hatti(cfg, m4)
        zaman.update(m6.zaman)
        duz.update(m6_duz_metrikler(m6, cfg))
        icerik["m6"] = {
            "olcum_originleri": m6.olcum_originleri,
            "kayit_satiri": int(m6.veri.n),
            "satir_sayisi_origin_basina": m6.satir_sayisi,
            "tekrar_sayisi": cfg.ope.kayit.tekrar_sayisi,
            "propensity_kaynagi": m6.prop.kaynak,
            "ozdeslik": m6.ozdeslik,
            "rollout_penceresi": [cfg.ope.rollout.baslangic_hafta,
                                  cfg.ope.rollout.baslangic_hafta
                                  + cfg.ope.rollout.ufuk_hafta],
            "egitim_ortusmesi_hafta": m6.egitim_ortusmesi,
            "offline": {ad: {t: s.deger(t) for t in ev_ope.TAHMINCILER}
                        | {"oracle": m6.oracle.deger(m6.hedefler[ad]),
                           "teshis": vars(s.teshis)}
                        for ad, s in m6.ope_sonuclari.items()},
            "ayristirma": {ad: vars(a) for ad, a in m6.ayristirmalar.items()},
            "online": {ad: {"net_marj": o.birikimli_net_marj().tolist(),
                            "haftalar": [h.hafta for h in o.haftalar]}
                       for ad, o in m6.rollout.items()},
            "gecikmeli": m6.gecikmeli,
            "kopru": [vars(k) for k in m6.kopruler],
        }

    if "m7" in asamalar:
        m7 = m7_boru_hatti(cfg, m4, "dunya", dizin)
        zaman.update(m7.zaman)
        duz.update(m7_duz_metrikler(m7))
        icerik["m7"] = {
            "origin": m7.kosu.t,
            "politika": m7.kosu.politika,
            "taban_rejim": m7.kosu.taban_ad,
            "rejimler": {ad: vars(o) for ad, o in m7.kosu.ozetler.items()},
            "farklar": {ad: vars(f) for ad, f in m7.kosu.farklar.items()},
            "harness": {
                "vaka_sayisi": m7.vaka_sayisi,
                "atlanan_vakalar": m7.atlanan_vakalar,
                "gecen": m7.harness.gecen,
                "kalan": [s.vaka.ad for s in m7.harness.kalan],
                "vakalar": [{"ad": s.vaka.ad, "tip": s.vaka.tip,
                             "eczane_id": s.eczane_id, "gecti": s.gecti,
                             "olcum": s.olcum} for s in m7.harness.sonuclar],
            },
        }

    icerik["sure"] = zaman
    icerik["duz"] = duz
    (dizin / "metrikler.json").write_text(
        json.dumps(icerik, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    if tahmin_yaz and b is not None:
        pl.DataFrame({
            "eczane_id": panel.anahtar["eczane_id"].to_numpy()[bolme.test],
            "sku_id": panel.anahtar["sku_id"].to_numpy()[bolme.test],
            "origin": panel.origin[bolme.test],
            "olcume_dahil": gecerli,
            "gercek_tukenme_k": o_test.tukenme_k,
            "gercek_olay": o_test.olay,
            # Karar ufkundaki etiket: compare.py'nin eslesmis bootstrap'i
            # metriklerle AYNI hedefi kullansin diye burada uretilir.
            "gercek_karar_olayi": y_oracle,
            "gercek_sure_hafta": T,
            "gozlemlenebilir_olay": b.y_test,
            **tahmin_kolonlari,
        }).write_parquet(dizin / "tahminler.parquet")

    if not veri_tut:
        shutil.rmtree(dunya.kok, ignore_errors=True)
    return icerik


def _teshisler(cfg: Config, panel: Panel, bolme: Bolme, oracle: Oracle,
               y_egitim: np.ndarray) -> dict:
    """TESHIS tahmincileri: teslim edilen model DEGIL, tavan olcumu.

    oracle_etiket : gercek tukenme etiketiyle egitilmis ayni hazard.
                    "Etiket korlugu (bize siparis vekili) ne kadara mal oluyor".
    oracle_ozellik: gercek stok / gercek hiz. Ozellik korlugunun tavani.

    Ikisi de ground_truth okur; bu yuzden degerlendirme tarafinda durur ve
    ciktisi asla feature katmanina donmez.
    """
    ufuk = cfg.tukenme.hedef.ufuk_hafta
    cikti = {}

    o_egitim = oracle.etiketle(
        panel.anahtar["eczane_id"].to_numpy()[bolme.egitim],
        panel.anahtar["sku_id"].to_numpy()[bolme.egitim],
        panel.origin[bolme.egitim], ufuk,
    )
    hz = HazardTahmincisi(cfg, panel).egit(
        panel, bolme.egitim, o_egitim.tukenme_k, o_egitim.izlenen_k)
    cikti["teshis_oracle_etiket"] = hz.tahmin(panel, bolme.test)

    def _kapsama(idx: np.ndarray) -> np.ndarray:
        o = oracle.etiketle(panel.anahtar["eczane_id"].to_numpy()[idx],
                            panel.anahtar["sku_id"].to_numpy()[idx],
                            panel.origin[idx], ufuk)
        return np.clip(o.origin_stogu / np.maximum(o.gercek_hiz, cfg.feature.hiz.min_hiz),
                       0.0, ufuk)

    kal = KovaKalibratoru(cfg.tukenme.degerlendirme.kalibrasyon_kova_sayisi)
    kal.egit(-_kapsama(bolme.egitim), y_egitim)
    kaps = _kapsama(bolme.test)
    cikti["teshis_oracle_ozellik"] = Tahmin(
        olasilik=kal.uygula(-kaps), skor=-kaps, tukenme_hafta=kaps)
    return cikti


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profil", default="fast")
    ap.add_argument("--seed", type=int, default=None, help="profildeki temel_seed'i ezer")
    ap.add_argument("--knob", action="append", default=[], help="yol=deger, tekrarlanabilir")
    ap.add_argument("--ad", default=None, help="kosu_id (varsayilan: profil_hash_seed)")
    ap.add_argument("--veri-tut", action="store_true", help="uretilen dunyayi silme")
    ap.add_argument("--tahmin-yazma", action="store_true", help="tahminler.parquet yazma")
    ap.add_argument("--asama", default="m2,m3,m4,m5",
                    help=f"kosulacak asamalar, virgulle: {','.join(ASAMALAR)}")
    args = ap.parse_args()

    asamalar = tuple(a.strip() for a in args.asama.split(",") if a.strip())
    bilinmeyen = set(asamalar) - set(ASAMALAR)
    if bilinmeyen:
        raise SystemExit(f"bilinmeyen asama: {sorted(bilinmeyen)} (gecerli: {ASAMALAR})")

    gecersiz = knob_ayristir(args.knob)
    if args.seed is not None:
        gecersiz["profil.temel_seed"] = args.seed
    cfg = load_config(args.profil, gecersiz_kilma=gecersiz)
    kosu_id = args.ad or f"{cfg.profil.ad}_{cfg.hash()[:8]}_{cfg.profil.temel_seed}"

    icerik = kosu_yap(cfg, kosu_id, gecersiz, args.veri_tut, not args.tahmin_yazma,
                      asamalar=asamalar)

    print(f"kosu: {KOSU_DIZINI / kosu_id}")
    print(f"config_hash={icerik['config_hash']} seed={icerik['temel_seed']} "
          f"asamalar={asamalar} sure={icerik['sure']}")
    if "m2" in asamalar:
        p = icerik["panel"]
        print(f"panel: {p['hucre_sayisi']} hucre x {p['origin_sayisi']} origin -> "
              f"egitim {p['egitim_satiri']}, test {p['test_satiri']}, "
              f"olcum {p['olcum_satiri']}")
        print(f"gercek tukenme taban orani={p['gercek_tukenme_taban_orani']:.3f} | "
              f"gozlemlenebilir taban orani={p['gozlemlenebilir_taban_orani']:.3f} | "
              f"kural N: literal={p['kural_literal_n_gun']}g, "
              f"secilen={p['kural_secilen_n_gun']}g")
        tablo = pl.DataFrame([
            {"tahminci": ad, "AUC": m["auc"], "PR_AUC": m["pr_auc"],
             "ust%10_kazanc": m["ust_dilim_kazanci"], "brier": m["brier"],
             "kalib_hata": m["kalibrasyon_hatasi"], "MAE_gun": m["mae_gun"],
             "MAE_gun_olayli": m["mae_gun_olayli"], "yanlilik_gun": m["yanlilik_gun"],
             "AUC_gozlemlenebilir": m["auc_gozlemlenebilir"]}
            for ad, m in icerik["tahminciler"].items()
        ])
        with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=200,
                       float_precision=3):
            print(tablo)
    if "m3" in asamalar:
        d, m3 = icerik["duz"], icerik["m3"]
        print(f"\nM3: origin={m3['originler']} aday satiri={m3['aday_satiri']} "
              f"ihlal={m3['ihlal']}")
        with pl.Config(tbl_rows=20, tbl_cols=12, tbl_width_chars=200, float_precision=4):
            print(pl.DataFrame([
                {"uretici": u,
                 "recall@K": d[f"aday.{u}.recall"],
                 "yeni_hucre_recall": d[f"aday.{u}.yeni_recall"],
                 "kapsama": d[f"aday.{u}.kapsama"]}
                for u in list(cfg.politika.aday.URETICILER) + ["hibrit"]]))
        print(f"kisit: havuz recall={d['kisit.havuz_recall']:.4f} -> veto sonrasi="
              f"{d['kisit.veto_sonrasi_recall']:.4f} -> liste={d['kisit.liste_recall']:.4f}"
              f" | veto orani={d['kisit.veto_orani']:.3f}"
              f" | ust dilim veto={d['kisit.ust_dilim_veto_orani']:.3f}")
    if "m4" in asamalar:
        m4_ozet_yaz(icerik)
    if "m5" in asamalar:
        m5_ozet_yaz(icerik)
    if "m6" in asamalar:
        m6_ozet_yaz(icerik, cfg)
    if "m7" in asamalar:
        m7_ozet_yaz(icerik)


def m7_ozet_yaz(icerik: dict) -> None:
    m7 = icerik["m7"]
    print(f"\nM7: origin={m7['origin']} politika={m7['politika']} "
          f"taban={m7['taban_rejim']}")
    tablo = pl.DataFrame([
        {"rejim": ad, "teklif": o["teklif_sayisi"], "vetolu": o["vetolu_satir"],
         "artimsal_marj_TL": o["beklenen_artimsal_marj"],
         "adet": o["teklif_adedi"], "bedava": o["bedava_adet"],
         "ort_mf": o["ortalama_mf"], "ort_vade": o["ortalama_vade"],
         "bekleyemeyen_pay": o["bekleyemeyen_teklif_pay"],
         "erteleme_TL_adet": o["ortalama_erteleme_tl"]}
        for ad, o in m7["rejimler"].items()])
    with pl.Config(tbl_rows=10, tbl_cols=12, tbl_width_chars=200, float_precision=3):
        print(tablo)
    h = m7["harness"]
    print(f"harness: {h['gecen']}/{h['vaka_sayisi']} vaka gecti"
          + (f" | KALAN: {h['kalan']}" if h["kalan"] else "")
          + (f" | atlanan (sweep disi): {h['atlanan_vakalar']}"
             if h["atlanan_vakalar"] else ""))


def m4_ozet_yaz(icerik: dict) -> None:
    d, m4 = icerik["duz"], icerik["m4"]
    print(f"\nM4: egitim origin={m4['egitim_originleri'][0]}..{m4['egitim_originleri'][-1]}"
          f" ({len(m4['egitim_originleri'])} origin, {m4['egitim_satiri']} satir, "
          f"{m4['ozellik_sayisi']} ozellik) | olcum origin={m4['olcum_originleri']} "
          f"| ihlal={m4['ihlal']}")
    tablo = pl.DataFrame([
        {"politika": ad,
         "artimsal_marj_TL": o["artimsal_marj"],
         "teklif": o["teklif_sayisi"],
         "TL/teklif": d[f"m4.{ad}.teklif_basina_artimsal"],
         "negatif_teklif": o["negatif_teklif_sayisi"],
         "yakilan_TL": o["negatif_marj"],
         "ort_MF": o["ortalama_mf"],
         "ort_vade": o["ortalama_vade"]}
        for ad, o in m4["politikalar"].items()])
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=220, float_precision=1):
        print(tablo)
    print(f"MARJ FARKI (uplift_x - propensity) = {d['m4.marj_farki_tl']:,.0f} TL "
          f"[%95: {d['m4.marj_farki_alt']:,.0f}, {d['m4.marj_farki_ust']:,.0f}] "
          f"= artimsalin %{d['m4.marj_farki_yuzde']:.1f}'i")
    print(f"CATE: PEHE T={d['m4.cate.pehe_t']:.4f} X={d['m4.cate.pehe_x']:.4f} | "
          f"sira kor. T={d['m4.cate.sira_kor_t']:.3f} X={d['m4.cate.sira_kor_x']:.3f} | "
          f"heterojenlik sd={d['m4.heterojenlik.cate_sapmasi']:.4f}, "
          f"farkli karar={d['m4.heterojenlik.farkli_karar_orani']:.3f}")


def m5_ozet_yaz(icerik: dict) -> None:
    d, m5 = icerik["duz"], icerik["m5"]
    print(f"\nM5: origin={m5['originler']} lot={m5['lot_sayisi']} "
          f"| ihlal={ {k: v for k, v in m5['ihlal'].items() if v} or '{}' }")
    tablo = pl.DataFrame([
        {"politika": ad,
         "teklif": o["teklif_sayisi"],
         "talep_adet": o["talep_adet"],
         "karsilanmayan": o["karsilanmayan_adet"],
         "stockout": o["stockout_sayisi"],
         "iade_adet": o["iade_adet"],
         "imha_adet": o["imha_adet"],
         "imha_temizlik": o["imha_adet_temizlik"],
         "brut_marj": o["brut_marj"],
         "net_marj": o["net_marj"],
         "memnuniyet": o["memnuniyet_proxy"]}
        for ad, o in m5["politikalar"].items()])
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=240, float_precision=3):
        print(tablo)
    print(f"(a) LP - ranking_only: stockout {d['m5.a.stockout_farki']:+,.1f}, "
          f"karsilanmayan {d['m5.a.karsilanmayan_farki']:+,.0f} adet, "
          f"net marj {d['m5.a.net_marj_farki']:+,.0f} TL")
    print(f"(b) kor iskonto - temizlik yok: imha {d['m5.b.imha_kor_farki']:+,.0f}, "
          f"iade {d['m5.b.iade_kor_farki']:+,.0f}, "
          f"net marj {d['m5.b.net_marj_kor_farki']:+,.0f} TL, "
          f"memnuniyet {d['m5.b.memnuniyet_kor_farki']:+.4f}")
    print(f"(b) hedefli - temizlik yok:     imha {d['m5.b.imha_hedefli_farki']:+,.0f}, "
          f"iade {d['m5.b.iade_hedefli_farki']:+,.0f}, "
          f"net marj {d['m5.b.net_marj_hedefli_farki']:+,.0f} TL, "
          f"memnuniyet {d['m5.b.memnuniyet_hedefli_farki']:+.4f}")
    for ad in ("lp", "hedefli_temizlik"):
        g = m5["golge"].get(ad) or {}
        if g:
            print(f"gölge fiyat [{ad}]: ort {g['golge_ortalama']:.4f} "
                  f"[{g['golge_asgari']:.4f}, {g['golge_azami']:.4f}] TL/adet | "
                  f"negatif lot orani {g['negatif_golge_lot_orani']:.3f} "
                  f"({g['negatif_golge_adet']:,.0f} adet) | temizlik penceresinde "
                  f"negatif {g['temizlik_penceresinde_negatif_orani']:.3f}")


def m6_ozet_yaz(icerik: dict, cfg: Config) -> None:
    d, m6 = icerik["duz"], icerik["m6"]
    oz = m6["ozdeslik"]
    print(f"\nM6: olcum origin={m6['olcum_originleri']} | kayit satiri="
          f"{m6['kayit_satiri']:,} ({m6['tekrar_sayisi']} tekrar x "
          f"{m6['satir_sayisi_origin_basina']:,.0f} satir) | propensity="
          f"{m6['propensity_kaynagi']} | rollout hafta {m6['rollout_penceresi'][0]}"
          f"..{m6['rollout_penceresi'][1]} (egitim ortusmesi "
          f"{m6['egitim_ortusmesi_hafta']} hafta)")
    print(f"ozdeslik testi: kayit politikasi oracle="
          f"{oz['kayit_politikasi_oracle']:.4f} gozlenen="
          f"{oz['kayit_politikasi_gozlenen']:.4f} TL/satir "
          f"(sapma %{oz['ozdeslik_sapma_yuzde']:.2f})")

    hedefler = list(m6["offline"])
    tablo = pl.DataFrame([
        {"politika": ad,
         "oracle": o["oracle"], "IPS": o["ips"], "SNIPS": o["snips"],
         "DR": o["dr"], "dogrudan": o["dogrudan"],
         "IPS_sapma_%": d[f"m6.denetim.{ad}.ips.sapma_yuzde"],
         "DR_sapma_%": d[f"m6.denetim.{ad}.dr.sapma_yuzde"],
         "ESS%": o["teshis"]["ess_orani"] * 100.0,
         "eslesme%": o["teshis"]["eslesme_orani"] * 100.0,
         "w_max": o["teshis"]["agirlik_azami"]}
        for ad, o in m6["offline"].items()])
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=230, float_precision=3):
        print(tablo)
        print(pl.DataFrame([
            {"politika": ad, "toplam_sapma": a["toplam_sapma"],
             "varyans": a["varyans_kalemi"], "kirpma": a["kirpma_kalemi"],
             "propensity": a["propensity_kalemi"],
             "artik": a["toplam_sapma"] - (a["varyans_kalemi"] + a["kirpma_kalemi"]
                                           + a["propensity_kalemi"]),
             "ortusme_ihlal_%": a["ortusme_ihlal_orani"] * 100.0,
             "kor_deger_%": a["ortusme_kor_deger_payi"] * 100.0}
            for ad, a in m6["ayristirma"].items()]))

    ufuklar = cfg.ope.rollout.raporlanan_ufuklar
    print(f"\nKAPALI DONGU (taban = teklif_yok, {len(ufuklar)} ufuk):")
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=230, float_precision=2):
        print(pl.DataFrame([
            {"politika": ad,
             **{f"artimsal@{h}": d[f"m6.online.{ad}.artimsal@{h}"] for h in ufuklar},
             **{f"%@{h}": d[f"m6.online.{ad}.artimsal_yuzde@{h}"] for h in ufuklar},
             "iade": d[f"m6.online.{ad}.iade_adet"],
             "sow_son": d[f"m6.online.{ad}.sow_son"]}
            for ad in cfg.ope.rollout.politikalar if ad != "teklif_yok"]))

    print("\nOFFLINE -> ONLINE KOPRUSU (DR tahmincisi):")
    kopru = [k for k in m6["kopru"] if k["tahminci"] == "dr"]
    with pl.Config(tbl_rows=40, tbl_cols=12, tbl_width_chars=200, float_precision=1):
        print(pl.DataFrame([
            {"politika": k["politika"], "ufuk": k["ufuk"],
             "offline_TL": k["offline_artimsal_tl"],
             "online_TL": k["online_artimsal_tl"],
             "offline_%": k["offline_yuzde"], "online_%": k["online_yuzde"],
             "ufuk_kalemi": k["ufuk_kalemi"]} for k in kopru]))
    for ad in hedefler:
        g = m6["gecikmeli"].get(ad)
        if not g:
            continue
        son = max(ufuklar)
        print(f"gecikmeli bedel [{ad}]: organik siparis farki "
              f"{g['kanibalizm_organik_siparis_farki']:+,.0f} adet | iade farki "
              f"{g['iade_adet_farki']:+,.0f} | bedava {g['bedava_adet']:,.0f} adet | "
              f"SOW son fark {g['sow_son_fark']:+.4f}")
        print(f"  ufuk kesiminin yanliligi @{son}: rafta fazla "
              f"{g[f'terminal_fazla_adet@{son}']:+,.0f} adet -> riskteki marj "
              f"{g[f'terminal_riskli_marj@{son}']:+,.0f} TL "
              f"(artimsalin %{g[f'terminal_riskli_pay@{son}']:.0f}'i, UST SINIR)")


if __name__ == "__main__":
    main()
