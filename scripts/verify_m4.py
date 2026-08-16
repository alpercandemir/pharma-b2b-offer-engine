"""M4 cikis kriteri dogrulamasi.

    python -m scripts.verify_m4 --kosu full
    python -m scripts.verify_m4 --kosu full --hizli   (point-in-time / determinizm atla)

SPEC M4 cikis kriteri: "Propensity-based politika ile uplift-based politika
arasindaki MARJ FARKI olculur. Propensity'nin nasil marj yaktigi sayiyla
gosterilir."

Kontroller iki gruba ayrilir (verify_m1/m2/m3 ile ayni disiplin):
  DURUSTLUK : sizinti siniri (tepki fonksiyonu politikaya gorunmuyor),
              point-in-time ozellikler, propensity loglamasi (D7),
              aksiyon uzayi degismezleri (D1), determinizm.
  KRITER    : marj farki ve ayristirmasi, yakilan marjin sayisi, CATE'in
              tabani gecmesi, HETEROJENLIK TESHISI.

HETEROJENLIK TESHISI NEDEN BIR KONTROL. CLAUDE.md M4 talimati acik: "Iki
politika ayni sonucu veriyorsa simulatorde uplift heterojenligi yok demektir."
Bu cumle burada mekanik hale getirildi: gercek CATE'in sapmasi ve iki
politikanin ayrisan karar orani esiklerin altina duserse kontrol KALIR ve
rapor yerine uyari uretilir.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from core.config import Config, load_config  # noqa: E402
from core.io import DATA_DIR, Run  # noqa: E402
from core.rng import SeedBank  # noqa: E402
from eval import uplift as ev  # noqa: E402
from experiments.run import (m4_boru_hatti, m4_duz_metrikler,  # noqa: E402
                             m4_ihlaller, _origin_blogu)
from features import teklif as ft  # noqa: E402
from features.okuma import GozlemlenebilirKaynak  # noqa: E402
from policy import bandit, scorer  # noqa: E402
from policy import candidates as pol  # noqa: E402
from scripts.verify_m2 import kod_metni  # noqa: E402
from sim.response import (TEKLIF_YOK_KOLU,  # noqa: E402
                          beklenen_miktar_carpani)

REPO_ROOT = Path(__file__).resolve().parent.parent
SEKIL_DIZINI = REPO_ROOT / "reports" / "figures" / "m4"
GECICI = REPO_ROOT / "experiments" / "runs" / "_dogrulama_m4"

# --- CIKIS KRITERI ESIKLERI. Tuning knob'u DEGIL, kriterin kendisi. ---
# Gercek CATE'in (en iyi kol, olasilik olceginde) asgari sapmasi. Altina
# duserse "butun eczaneler tekliflere ayni tepkiyi veriyor" demektir ve
# uplift modellemenin konusu ortadan kalkar.
ESIK_CATE_SAPMASI = 0.02
# Iki politikanin ayrisan karar orani. Sifira yakinsa uplift ile propensity
# ayni politikadir ve M4'un karsilastirmasi anlamsizdir.
ESIK_FARKLI_KARAR = 0.02
# Amac fonksiyonunun TEK BASINA yarattigi fark (oracle uplift - oracle
# propensity), artimsal marja oran olarak. Model hatasindan arindirilmis
# olcum budur; sifirsa iki amac fonksiyonu ozdes davraniyor demektir.
ESIK_ORACLE_FARKI_ORANI = 0.01
# Politikanin ana modulleri tepki fonksiyonunu goremez.
YASAK_TEPKI_ADLARI = ("sim.response", "sim/response", "tepki_hesapla",
                      "TepkiEvreni", "GercekDurum", "ground_truth")
TARANAN_DIZINLER = ("policy", "features", "models")


@dataclass
class Kontrol:
    ad: str
    gecti: bool
    olcum: str


def _sekil_kaydet(ad: str) -> Path:
    SEKIL_DIZINI.mkdir(parents=True, exist_ok=True)
    yol = SEKIL_DIZINI / f"{ad}.png"
    plt.tight_layout()
    plt.savefig(yol, dpi=110)
    plt.close()
    return yol


# --------------------------------------------------------------------------
# DURUSTLUK kontrolleri
# --------------------------------------------------------------------------
def kontrol_tepki_sizintisi() -> Kontrol:
    """Politika / feature / model katmani tepki fonksiyonunu GORMUYOR.

    M4'un en kolay kendini kandirma bicimi: CATE modelinin ozellik
    matrisine gercek kabul olasiligini sizdirmak. Statik tarama.
    """
    bulgular = []
    for dizin in TARANAN_DIZINLER:
        for yol in sorted((REPO_ROOT / dizin).glob("*.py")):
            kod = kod_metni(yol)
            bulgular += [f"{dizin}/{yol.name}:{k}" for k in YASAK_TEPKI_ADLARI
                         if k in kod]
    return Kontrol("Politika/feature/model katmani tepki fonksiyonunu gormuyor",
                   not bulgular,
                   f"{len(bulgular)} bulgu ({len(TARANAN_DIZINLER)} dizin tarandi)"
                   + (f": {bulgular}" if bulgular else ""))


def kontrol_point_in_time(kosu_adi: str, cfg: Config) -> Kontrol:
    """Gelecek silinince ayni origin'in TEKLIF OZELLIKLERI degismemeli."""
    kaynak = GozlemlenebilirKaynak(kosu_adi)
    dunya = pol.dunya_yukle(kaynak, cfg)
    kesme = dunya.W // 2

    hedef = Run("kesilmis", kok=GECICI).prepare()
    for tablo in kaynak.tables():
        df = kaynak.tablo(tablo)
        if "hafta" in df.columns:
            df = df.filter(pl.col("hafta") <= kesme)
        hedef.write_observable(tablo, df)
    kesik_kaynak = GozlemlenebilirKaynak("kesilmis", kok=GECICI)
    kesik = pol.dunya_yukle(kesik_kaynak, cfg)

    td_tam = ft.teklif_dunyasi_yukle(kaynak, cfg, dunya)
    td_kes = ft.teklif_dunyasi_yukle(kesik_kaynak, cfg, kesik)
    a = _origin_blogu(td_tam, cfg, dunya, kesme)
    b = _origin_blogu(td_kes, cfg, kesik, kesme)
    shutil.rmtree(GECICI, ignore_errors=True)

    if a.X.shape != b.X.shape:
        return Kontrol("Point-in-time: gelecek silinince teklif ozellikleri ayni",
                       False, f"sekil farkli: {a.X.shape} vs {b.X.shape}")
    fark = float(np.max(np.abs(np.nan_to_num(a.X) - np.nan_to_num(b.X))))
    return Kontrol("Point-in-time: gelecek silinince teklif ozellikleri ayni",
                   fark == 0.0,
                   f"{a.X.shape[1]} ozellik x {a.X.shape[0]} satir, en buyuk fark={fark:.3g}")


def kontrol_propensity_loglamasi(c, cfg: Config) -> Kontrol:
    """D7: her gosterimde secim olasiligi loglanir ve HER kol pozitif.

    Overlap ihlali M4'te sessiz, M6'da olumcul: propensity'si sifir olan bir
    (satir, kol) ciftinde IPS agirligi tanimsizdir.
    """
    en_kucuk, toplam_hatasi, kapali_kol = 1.0, 0.0, 0
    for b in c.bloklar:
        pi = b.pi
        if pi.shape[0] == 0:
            continue
        toplam_hatasi = max(toplam_hatasi, float(np.abs(pi.sum(axis=1) - 1.0).max()))
        izinli = b.mat.izinli.copy()
        izinli[:, TEKLIF_YOK_KOLU] = True
        en_kucuk = min(en_kucuk, float(pi[izinli].min()))
        kapali_kol += int((pi[~izinli] > 0).sum())
    return Kontrol(
        "D7: propensity loglaniyor, izinli her kol pozitif (overlap)",
        en_kucuk > 0.0 and toplam_hatasi < 1e-9 and kapali_kol == 0,
        f"en kucuk pi={en_kucuk:.5f} (azami IPS agirligi={1/max(en_kucuk,1e-12):.0f}), "
        f"satir toplami hatasi={toplam_hatasi:.2g}, izinsiz kolda kutle={kapali_kol}",
    )


def kontrol_aksiyon_uzayi(cfg: Config, c) -> Kontrol:
    """D1: aksiyon (MF, vade) ciftidir; yuzde iskonto YOK, kanal kisiti tutuyor."""
    uzay = c.bloklar[0].mat.uzay
    beklenen = 1 + len(cfg.politika.aksiyon.mf_oranlari) * len(cfg.politika.aksiyon.vade_gunleri)
    ihlal = m4_ihlaller(c, cfg)
    kritik = {k: v for k, v in ihlal.items() if v}
    # Kanal kisiti fiilen bagliyor mu (vakum kontrolu).
    kapali = sum(int((~b.mat.izinli[:, 1:]).any(axis=1).sum()) for b in c.bloklar)
    satir = sum(b.teklifler.height for b in c.bloklar)
    return Kontrol(
        "D1: aksiyon uzayi (MF, vade) ve kanal kisitlari ihlal edilmiyor",
        not kritik and uzay.A == beklenen and kapali > 0,
        f"{uzay.A} kol (1 kontrol + {beklenen-1} teklif), {kapali}/{satir} satirda "
        f"en az bir kol kapali, ihlal={kritik or '{}'}",
    )


def kontrol_determinizm(cfg: Config, kosu_adi: str) -> Kontrol:
    def _ozet():
        c = m4_boru_hatti(cfg, kosu_adi, DATA_DIR)
        d = m4_duz_metrikler(c, cfg)
        return (round(d["m4.marj_farki_tl"], 6),
                round(d["m4.uplift_x.artimsal_marj"], 6),
                round(d["m4.propensity.artimsal_marj"], 6),
                int(d["m4.uplift_x.teklif_sayisi"]))
    a, b = _ozet(), _ozet()
    return Kontrol("Determinizm: ayni kosu ayni marj farki", a == b, f"{a} == {b}")


# --------------------------------------------------------------------------
# KRITER kontrolleri
# --------------------------------------------------------------------------
def kontrol_heterojenlik(c, cfg: Config) -> Kontrol:
    """CLAUDE.md M4 talimatinin mekanik hali.

    Uc olcu birden esigin altindaysa simulatorde uplift heterojenligi yok
    demektir ve karsilastirma anlamsizdir.
    """
    t = c.teshis
    d = m4_duz_metrikler(c, cfg)
    artimsal = c.olcumler["oracle_propensity"].artimsal_marj
    oracle_orani = (d["m4.oracle_marj_farki_tl"] / abs(artimsal)) if artimsal else 0.0
    gecti = (t.cate_sapmasi > ESIK_CATE_SAPMASI
             and t.farkli_karar_orani > ESIK_FARKLI_KARAR
             and oracle_orani > ESIK_ORACLE_FARKI_ORANI)
    return Kontrol(
        "Heterojenlik: iki politika AYNI politika degil",
        gecti,
        f"gercek CATE sd={t.cate_sapmasi:.4f} (esik {ESIK_CATE_SAPMASI}), "
        f"ust/alt dilim orani={t.cate_dilim_orani:.1f}x, "
        f"farkli karar={t.farkli_karar_orani:.3f} (esik {ESIK_FARKLI_KARAR}), "
        f"amac fonksiyonunun tek basina farki=%{oracle_orani*100:.1f} "
        f"(esik %{ESIK_ORACLE_FARKI_ORANI*100:.0f})",
    )


def kontrol_marj_farki(c, cfg: Config) -> Kontrol:
    """Cikis kriteri: fark OLCULDU ve ayristirmasi toplama esit.

    Kriter farkin POZITIF cikmasi degil - olculmus ve hesabi verilmis
    olmasi. Isaret bir bulgudur, kriter degil (reports/m4.md 5).
    """
    d = m4_duz_metrikler(c, cfg)
    toplam = sum(v for k, v in c.ayristirma.items() if k.endswith("_katki"))
    ozdeslik = abs(toplam - c.ayristirma["toplam_fark"])
    return Kontrol(
        "Marj farki olculdu ve ayristirmasi topluyor",
        ozdeslik < 1e-6,
        f"uplift_x - propensity = {d['m4.marj_farki_tl']:,.0f} TL "
        f"[%95: {d['m4.marj_farki_alt']:,.0f}, {d['m4.marj_farki_ust']:,.0f}], "
        f"ayristirma ozdesligi hatasi={ozdeslik:.2g}",
    )


def kontrol_yakilan_marj(c, cfg: Config) -> Kontrol:
    """Propensity'nin marj YAKTIGI sayiyla gosteriliyor mu.

    Iki propensity bicimi de olculur: ham (tepki maksimizasyonu) ve marj
    farkindali. Yakilan marj = artimsal marji NEGATIF olan tekliflerin
    toplami; oracle uplift'te tanimi geregi sifirdir.
    """
    ham = c.olcumler["propensity_ham"]
    marjli = c.olcumler["propensity"]
    orc = c.olcumler["oracle_uplift"]
    return Kontrol(
        "Propensity'nin yaktigi marj sayiyla gosteriliyor",
        ham.negatif_marj < 0 and orc.negatif_teklif_sayisi == 0,
        f"propensity_ham: {ham.negatif_teklif_sayisi}/{ham.teklif_sayisi} teklif "
        f"negatif, {ham.negatif_marj:,.0f} TL yakildi | "
        f"propensity: {marjli.negatif_teklif_sayisi}/{marjli.teklif_sayisi}, "
        f"{marjli.negatif_marj:,.0f} TL | oracle_uplift: {orc.negatif_teklif_sayisi}",
    )


def kontrol_cate_tabani_geciyor(c, cfg: Config) -> Kontrol:
    """CATE modeli "herkese ayni uplift" tabanini geciyor mu.

    Taban: tahmin edilen CATE'i satir ortalamasiyla degistir. Sira
    korelasyonu tanimsizlasir; bunun yerine kazanc egrisinin AUUC'si
    kullanilir - rassal siralamaya gore alan pozitif olmali.
    """
    auuc = {ad: e["auuc"] for ad, e in c.cate["kazanc_egrisi"].items()}
    return Kontrol(
        "CATE siralamasi rassal tabani geciyor (AUUC > 0)",
        all(v > 0 for v in auuc.values()),
        f"AUUC uplift_t={auuc['uplift_t']:,.0f}, uplift_x={auuc['uplift_x']:,.0f}, "
        f"propensity={auuc['propensity']:,.0f} | PEHE T={c.cate['pehe_t']:.4f} "
        f"X={c.cate['pehe_x']:.4f}",
    )


def kontrol_kisit_veto_yetkisi(c, cfg: Config) -> Kontrol:
    """D6: kisit katmani aksiyon seciminden SONRA da veto yetkisini koruyor.

    Vakum tehlikesi: kredi limiti bu dunyada bagalamiyor (M3'te olculdu),
    bu yuzden kanal kisitinin FIILEN atesledigi ayrica dogrulanir.
    """
    kapali = {}
    for b in c.bloklar:
        for ad, v in b.mat.kapali_sebep.items():
            kapali[ad] = kapali.get(ad, 0) + int(v.sum())
    ihlal = m4_ihlaller(c, cfg)
    return Kontrol(
        "D6: kanal kisitlari aksiyon seciminde de bagliyor",
        kapali.get("mf_kanali_kapali", 0) > 0 and ihlal["mf_kanali_ihlali"] == 0,
        f"kapatilan kol sayisi {kapali}, secim sonrasi kredi vetosu="
        f"{c.kredi_vetosu}, ihlal={ {k: v for k, v in ihlal.items() if v} or '{}'}",
    )


# --------------------------------------------------------------------------
# grafikler
# --------------------------------------------------------------------------
def segment_tablosu(c, cfg: Config) -> pl.DataFrame:
    """PROPENSITY NEREDE MARJ YAKIYOR: taban siparis olasiligi dilimine gore.

    Satirlar "zaten alacak miydi" olasiligina gore ondaliga bolunur. Dilim 9
    kesin alicilardir; propensity oraya teklif yigar cunku p * marj orada en
    buyuktur, ama gercek artimsal marj orada en kucuktur (ihtiyac etkilesimi
    + saturasyon). Farkin nerede olustugunu gosteren tablo budur.
    """
    p0 = np.concatenate([t.taban_olasilik for t in c.tepkiler])
    carpan = c.carpan
    dilim = np.clip((np.argsort(np.argsort(p0)) * 10) // p0.size, 0, 9)
    izin = np.vstack([b.mat.izinli for b in c.bloklar]).copy()
    izin[:, TEKLIF_YOK_KOLU] = False
    cate = np.where(izin, np.vstack([t.uplift for t in c.tepkiler]), -np.inf).max(axis=1)

    art = {}
    kollar = {}
    for ad in ("propensity", "uplift_x"):
        art[ad] = np.concatenate([
            ev.artimsal_marj(t, b.mat, s.kol, carpan)
            for t, b, s in zip(c.tepkiler, c.bloklar, c.secimler[ad])])
        kollar[ad] = np.concatenate([s.kol for s in c.secimler[ad]])

    satirlar = []
    for i in range(10):
        m = dilim == i
        satirlar.append({
            "dilim": i,
            "satir": int(m.sum()),
            "taban_olasilik": float(p0[m].mean()),
            "gercek_CATE": float(cate[m].mean()),
            "propensity_teklif": float((kollar["propensity"][m] != TEKLIF_YOK_KOLU).mean()),
            "uplift_teklif": float((kollar["uplift_x"][m] != TEKLIF_YOK_KOLU).mean()),
            "propensity_marj": float(art["propensity"][m].sum()),
            "uplift_marj": float(art["uplift_x"][m].sum()),
            "fark": float((art["uplift_x"][m] - art["propensity"][m]).sum()),
        })
    return pl.DataFrame(satirlar)


def sekil_politika_karsilastirmasi(c, cfg: Config) -> None:
    sira = ["m3_sabit_kampanya", "propensity_ham", "propensity", "uplift_t",
            "uplift_x", "oracle_propensity", "oracle_uplift"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    deger = [c.olcumler[a].artimsal_marj for a in sira]
    renk = ["#a5613b" if v < 0 else "#3b6ea5" for v in deger]
    renk[sira.index("uplift_x")] = "#5a9e5a"
    ax[0].barh(range(len(sira)), deger, color=renk)
    ax[0].set_yticks(range(len(sira)))
    ax[0].set_yticklabels(sira, fontsize=8)
    ax[0].axvline(0, c="k", lw=1)
    ax[0].set_xlabel("artimsal marj (TL)")
    ax[0].set_title("Politika degeri: teklif vermemeye gore", fontsize=9)

    yakilan = [-c.olcumler[a].negatif_marj for a in sira]
    ax[1].barh(range(len(sira)), yakilan, color="#a5613b")
    ax[1].set_yticks(range(len(sira)))
    ax[1].set_yticklabels(sira, fontsize=8)
    ax[1].set_xlabel("yakilan marj (TL, artimsali negatif teklifler)")
    ax[1].set_title("Propensity nerede marj yakiyor", fontsize=9)

    ayr = [(k[:-6], v) for k, v in c.ayristirma.items() if k.endswith("_katki")]
    ax[2].bar(np.arange(len(ayr)), [v for _, v in ayr], color="#3b6ea5")
    ax[2].axhline(0, c="k", lw=1)
    ax[2].set_xticks(np.arange(len(ayr)))
    ax[2].set_xticklabels([a for a, _ in ayr], rotation=30, fontsize=7, ha="right")
    ax[2].set_ylabel("TL")
    ax[2].set_title("uplift_x - propensity farkinin ayristirmasi", fontsize=9)
    fig.suptitle("M4: propensity ve uplift politikalarinin marj farki", fontsize=10)
    _sekil_kaydet("politika_karsilastirmasi")


def sekil_kazanc_egrisi(c, cfg: Config) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for ad, egri in c.cate["kazanc_egrisi"].items():
        ax[0].plot(egri["x"], egri["y"], marker="o", ms=3, label=f"{ad} (AUUC {egri['auuc']:,.0f})")
    x = np.array(c.cate["kazanc_egrisi"]["uplift_x"]["x"])
    toplam = c.cate["kazanc_egrisi"]["uplift_x"]["y"][-1]
    ax[0].plot(x, x * toplam, "k--", lw=1, label="rassal sira")
    ax[0].set_xlabel("teklif edilen satir orani")
    ax[0].set_ylabel("birikimli GERCEK artimsal marj (TL)")
    ax[0].set_title("Siralamanin degeri (Qini'nin marj karsiligi)", fontsize=9)
    ax[0].legend(fontsize=7)

    # Taban olasilik dilimine gore gercek uplift ve politikalarin teklif orani
    p0 = np.concatenate([t.taban_olasilik for t in c.tepkiler])
    izin = np.vstack([b.mat.izinli for b in c.bloklar]).copy()
    izin[:, TEKLIF_YOK_KOLU] = False
    cate = np.where(izin, np.vstack([t.uplift for t in c.tepkiler]), -np.inf).max(axis=1)
    dilim = np.clip((np.argsort(np.argsort(p0)) * 10) // p0.size, 0, 9)
    ort_cate = [cate[dilim == i].mean() for i in range(10)]
    ax2 = ax[1].twinx()
    ax[1].bar(np.arange(10), ort_cate, color="#5a9e5a", alpha=0.6, label="gercek CATE")
    for ad, renk in (("propensity", "#a5613b"), ("uplift_x", "#3b6ea5")):
        kol = np.concatenate([s.kol for s in c.secimler[ad]])
        oran = [(kol[dilim == i] != TEKLIF_YOK_KOLU).mean() for i in range(10)]
        ax2.plot(np.arange(10), oran, marker="o", ms=4, c=renk, label=ad)
    ax[1].set_xlabel("taban siparis olasiligi dilimi (9 = kesin alici)")
    ax[1].set_ylabel("gercek CATE (kabul olasiligi artisi)")
    ax2.set_ylabel("teklif verme orani")
    ax[1].set_title("Kesin aliciya kim teklif veriyor", fontsize=9)
    ax2.legend(fontsize=7, loc="upper right")
    fig.suptitle("CATE siralamasi ve hedefleme davranisi", fontsize=10)
    _sekil_kaydet("kazanc_egrisi")


def sekil_cate_kalitesi(c, cfg: Config) -> None:
    izin = np.vstack([b.mat.izinli for b in c.bloklar]).copy()
    izin[:, TEKLIF_YOK_KOLU] = False
    gercek = np.vstack([t.uplift for t in c.tepkiler])[izin]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, (ad, tahmin) in enumerate(
            (("T-ogrenici", c.tahmin_cate["t"]), ("X-ogrenici", c.tahmin_cate["x"]))):
        v = tahmin[izin]
        kova = np.clip((np.argsort(np.argsort(v)) * 10) // max(v.size, 1), 0, 9)
        ax[j].plot(range(10), [v[kova == i].mean() for i in range(10)],
                   marker="o", label="tahmin")
        ax[j].plot(range(10), [gercek[kova == i].mean() for i in range(10)],
                   marker="s", label="gercek")
        ax[j].set_xlabel("tahmin edilen CATE dilimi")
        ax[j].set_ylabel("kabul olasiligi artisi")
        ax[j].set_title(f"{ad}: kalibrasyon (PEHE "
                        f"{c.cate['pehe_' + ('t' if j == 0 else 'x')]:.4f})", fontsize=9)
        ax[j].legend(fontsize=8)
    fig.suptitle("CATE tahmini ile gercek uplift: dilim bazli", fontsize=10)
    _sekil_kaydet("cate_kalitesi")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kosu", default="full", help="data/<kosu> altindaki dunya")
    ap.add_argument("--profil", default=None, help="varsayilan: kosu manifestindeki profil")
    ap.add_argument("--hizli", action="store_true",
                    help="point-in-time ve determinizm kontrollerini atla")
    args = ap.parse_args()

    kosu = Run(args.kosu)
    manifest = kosu.read_manifest()
    profil = args.profil or manifest["profil"]
    cfg = load_config(profil)
    print(f"kosu={args.kosu} profil={profil} dunya_config_hash={manifest['config_hash']} "
          f"m4_config_hash={cfg.hash()}")

    c = m4_boru_hatti(cfg, args.kosu, DATA_DIR)
    d = m4_duz_metrikler(c, cfg)
    # Grafikler icin CATE tahminleri (boru hatti bunlari saklamiyor).
    c.carpan = beklenen_miktar_carpani(cfg)
    c.tahmin_cate = {"t": np.vstack([b.p_t - b.p_t[:, [TEKLIF_YOK_KOLU]] for b in c.bloklar]),
                     "x": np.vstack([b.p_x - b.p_x[:, [TEKLIF_YOK_KOLU]] for b in c.bloklar])}

    print(f"egitim origin={c.egitim_originleri[0]}..{c.egitim_originleri[-1]} "
          f"({len(c.egitim_originleri)} origin, {c.egitim_satiri} satir) | "
          f"olcum origin={c.olcum_originleri} | sure={c.zaman}")

    tablo = pl.DataFrame([
        {"politika": ad, "artimsal_marj": o.artimsal_marj, "teklif": o.teklif_sayisi,
         "TL/teklif": o.teklif_basina_artimsal,
         "negatif_teklif": o.negatif_teklif_sayisi, "yakilan_TL": o.negatif_marj,
         "ort_MF": o.ortalama_mf, "ort_vade": o.ortalama_vade,
         "beklenen_kabul": o.beklenen_kabul}
        for ad, o in c.olcumler.items()])
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=220, float_precision=2):
        print(tablo)
    print("\nPROPENSITY NEREDE MARJ YAKIYOR (taban siparis olasiligi dilimi):")
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=220, float_precision=3):
        print(segment_tablosu(c, cfg))
    print("\nayristirma (uplift_x - propensity):")
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=220, float_precision=1):
        print(pl.DataFrame([{k: v for k, v in c.ayristirma.items()}]))

    sekil_politika_karsilastirmasi(c, cfg)
    sekil_kazanc_egrisi(c, cfg)
    sekil_cate_kalitesi(c, cfg)

    kontroller = [
        kontrol_tepki_sizintisi(),
        kontrol_propensity_loglamasi(c, cfg),
        kontrol_aksiyon_uzayi(cfg, c),
        kontrol_kisit_veto_yetkisi(c, cfg),
        kontrol_heterojenlik(c, cfg),
        kontrol_marj_farki(c, cfg),
        kontrol_yakilan_marj(c, cfg),
        kontrol_cate_tabani_geciyor(c, cfg),
    ]
    if not args.hizli:
        kontroller.append(kontrol_point_in_time(args.kosu, cfg))
        kontroller.append(kontrol_determinizm(cfg, args.kosu))

    print("\n" + "=" * 132)
    print(f"{'DURUM':<8}{'KONTROL':<60}OLCUM")
    print("-" * 132)
    for k in kontroller:
        print(f"{'GECTI' if k.gecti else 'KALDI':<8}{k.ad:<60}{k.olcum}")
    print("=" * 132)
    kalan = [k.ad for k in kontroller if not k.gecti]
    print(f"grafikler: {SEKIL_DIZINI}")
    if "Heterojenlik: iki politika AYNI politika degil" in kalan:
        print("\n!!! UPLIFT HETEROJENLIGI YOK: iki politika ayni sonucu veriyor.")
        print("    Simulatorde teklif etkisi eczaneler arasinda ayrismiyor;")
        print("    `tepki.duyarlilik.heterojenlik_carpani` ve")
        print("    `tepki.teklif.ihtiyac_etkilesimi` knob'larina bakin.")
    print(f"SONUC: {len(kontroller) - len(kalan)}/{len(kontroller)} gecti"
          + (f" | KALAN: {kalan}" if kalan else ""))
    sys.exit(1 if kalan else 0)


if __name__ == "__main__":
    main()
