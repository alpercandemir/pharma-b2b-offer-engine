"""M2 cikis kriteri dogrulamasi.

    python -m scripts.verify_m2 --kosu full
    python -m scripts.verify_m2 --kosu full --hizli     (sizinti testlerini atla)

Her kontrol bir GECTI/KALDI satiri ve olcum uretir; grafikler
reports/figures/m2/ altina yazilir. SPEC M2 cikis kriteri:
"Tahmin edilen tukenme gunu ile simulatorun gercek stok sifirlanma gunu
karsilastirmasi; MAE ve kalibrasyon egrisi."

Kontroller iki gruba ayrilir:
  DURUSTLUK  : sizinti, point-in-time, determinizm. Bunlar kalirsa geri kalan
               her sayi supheli olur, once bunlar bakilir.
  KRITER     : MAE + kalibrasyon uretiliyor mu, hazard kurali geciyor mu,
               ve metrik SUPHELI DERECEDE IYI mi (leakage kokusu).
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

from core.config import load_config  # noqa: E402
from core.io import DATA_DIR, Run  # noqa: E402
from eval import metrics as mt  # noqa: E402
from eval.oracle import Oracle  # noqa: E402
from experiments.run import _oracle_hedefleri, boru_hatti, degerlendir  # noqa: E402
from features.okuma import GozlemlenebilirKaynak  # noqa: E402
from features.panel import izgara_kur, panel_kur  # noqa: E402
from sim.calendar import GUN_HAFTA  # noqa: E402

SEKIL_DIZINI = Path(__file__).resolve().parent.parent / "reports" / "figures" / "m2"
GECICI = Path(__file__).resolve().parent.parent / "experiments" / "runs" / "_dogrulama"

# Cikis kriteri esikleri. Tuning knob'u DEGIL, kriterin kendisi: gevsetilirse
# kriter anlamini yitirir (verify_m1.py ile ayni disiplin).
#
# Hazard modelinin gercek tukenmede AUC'si bu tavani asarsa sizintiden
# suphelenilir: gercek stok / gercek hiz ile kurulmus oracle kapsamasi bilgi
# tavanidir, gozlemlenebilir veriyle onu gecmek mumkun olmamali.
ESIK_SUPHE_PAYI = 0.02
# Point-in-time kontrolunde kabul edilen fark: sifir. Kayan nokta yeniden
# hesaplandigi icin bit karsilastirmasi degil, tam esitlik araniyor.
ESIK_PIT_FARK = 0.0


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
YASAK_ADLAR = ("ground_truth", "oku_gercek", "yaz_gercek", "hucre_haftalik",
               "sow_haftalik", "latent_eczane", "tukenme_olaylari")


def kod_metni(yol: Path) -> str:
    """Dosyanin yorum ve docstring'lerden arindirilmis kod metni.

    Duz metin taramasi yetmez: bu modullerin docstring'leri sinirin NEDEN
    boyle kuruldugunu anlatmak icin 'ground_truth' kelimesini kullaniyor.
    Aranan sey aciklama degil, gercek erisim."""
    import ast

    agac = ast.parse(yol.read_text(encoding="utf-8"))
    for dugum in ast.walk(agac):
        if isinstance(dugum, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)) and ast.get_docstring(dugum):
            dugum.body = dugum.body[1:]
    return ast.unparse(agac)


def kontrol_statik_sizinti() -> Kontrol:
    """features/ ve models/ altinda ground_truth'a giden bir yol var mi."""
    kok = Path(__file__).resolve().parent.parent
    bulgular = []
    for dizin in ("features", "models"):
        for yol in sorted((kok / dizin).glob("*.py")):
            kod = kod_metni(yol)
            bulgular += [f"{yol.name}:{k}" for k in YASAK_ADLAR if k in kod]
    return Kontrol("Feature/model katmani ground_truth'a dokunmuyor", not bulgular,
                   f"{len(bulgular)} bulgu" + (f": {bulgular}" if bulgular else ""))


def _kesilmis_dunya(kaynak: GozlemlenebilirKaynak, kesme_haftasi: int) -> Path:
    """Gelecegi silinmis bir dunya kopyasi. Point-in-time testinin zemini."""
    hedef = Run("kesilmis", kok=GECICI).prepare()
    for ad in kaynak.tables():
        df = kaynak.tablo(ad)
        if "hafta" in df.columns:
            df = df.filter(pl.col("hafta") <= kesme_haftasi)
        hedef.write_observable(ad, df)
    return GECICI


def kontrol_point_in_time(kosu_adi: str, cfg) -> Kontrol:
    """Gelecek silinince ayni origin'in ozellikleri DEGISMEMELI.

    Bu, "point-in-time dogru" iddiasinin tek gercek testi. Yorumla degil,
    veriyi kesip yeniden hesaplayarak.
    """
    kaynak = GozlemlenebilirKaynak(kosu_adi)
    izg = izgara_kur(kaynak, cfg)
    kesme = izg.W // 2
    originler = np.array([kesme - 8, kesme - 4, kesme])
    tam = panel_kur(izg, cfg, originler)

    kok = _kesilmis_dunya(kaynak, kesme)
    kesik = panel_kur(izgara_kur(GozlemlenebilirKaynak("kesilmis", kok=kok), cfg),
                      cfg, originler)
    shutil.rmtree(GECICI, ignore_errors=True)

    a = tam.anahtar.with_row_index("i")
    b = kesik.anahtar.with_row_index("j")
    ortak = a.join(b, on=["eczane_id", "sku_id", "origin"], how="inner")
    if ortak.height == 0:
        return Kontrol("Point-in-time: gelecek silinince ozellikler degismiyor",
                       False, "ortak satir yok")
    fark = np.abs(tam.X[ortak["i"].to_numpy()] - kesik.X[ortak["j"].to_numpy()])
    en_buyuk = float(np.nanmax(fark))
    sutun = tam.ozellik_adlari[int(np.nanargmax(fark.max(axis=0)))] if en_buyuk else "-"
    return Kontrol(
        "Point-in-time: gelecek silinince ozellikler degismiyor",
        en_buyuk <= ESIK_PIT_FARK,
        f"{ortak.height} ortak satir, en buyuk fark={en_buyuk:.3g}"
        + (f" ({sutun})" if en_buyuk else ""),
    )


def kontrol_determinizm(cfg, kosu_adi: str) -> Kontrol:
    """Ayni config + ayni dunya -> ayni model, ayni metrik."""
    def _ozet():
        b = boru_hatti(cfg, kosu_adi, DATA_DIR)
        m = degerlendir("hazard", b.tahminler["hazard"], b.o_test, b.y_test, cfg)
        return (round(m["auc"], 10), round(m["mae_gun"], 10), round(m["brier"], 10))
    a, b = _ozet(), _ozet()
    return Kontrol("Determinizm: ayni kosu ayni metrik", a == b, f"{a} == {b}")


def kontrol_bolme(b, cfg) -> Kontrol:
    """Egitim ve test origin'leri zamanda ayrik ve aralarinda tampon var mi."""
    egitim = b.panel.origin[b.bolme.egitim]
    test = b.panel.origin[b.bolme.test]
    bosluk = int(test.min() - egitim.max())
    yeterli = bosluk >= cfg.tukenme.hedef.ufuk_hafta
    return Kontrol(
        "Zaman bolmesi: egitim/test ayrik ve tamponlu", yeterli,
        f"egitim <= hafta {egitim.max()}, test >= hafta {test.min()}, "
        f"bosluk={bosluk} hafta (gereken >= {cfg.tukenme.hedef.ufuk_hafta})",
    )


# --------------------------------------------------------------------------
# KRITER kontrolleri
# --------------------------------------------------------------------------
def kontrol_cikis_kriteri(olcumler: dict) -> Kontrol:
    """MAE ve kalibrasyon uretildi mi (SPEC M2 cikis kriteri)."""
    h = olcumler["hazard"]
    tamam = all(np.isfinite([h["mae_gun"], h["mae_gun_olayli"], h["kalibrasyon_hatasi"]]))
    return Kontrol(
        "Cikis kriteri: MAE + kalibrasyon uretildi", tamam,
        f"MAE={h['mae_gun']:.1f} gun (olayli {h['mae_gun_olayli']:.1f}), "
        f"yanlilik={h['yanlilik_gun']:+.1f} gun, kalibrasyon hatasi={h['kalibrasyon_hatasi']:.3f}",
    )


def kontrol_hazard_kurali_geciyor(b, cfg) -> Kontrol:
    """Hazard modeli 'son N gunde aldi mi' kuralini gecti mi - eslesmis bootstrap."""
    d = cfg.tukenme.degerlendirme
    gecerli, y, _ = _oracle_hedefleri(b.o_test, cfg.tukenme.hedef.ufuk_hafta,
                                      cfg.tukenme.hedef.karar_ufku_hafta)
    yg = y[gecerli]
    hz = b.tahminler["hazard"].skor[gecerli]
    kr = b.tahminler["kural_ikili"].skor[gecerli]
    fark, alt, ust = mt.bootstrap_farki(yg, hz, kr, mt.guvenli_auc,
                                        d.bootstrap_orneklem, d.bootstrap_seed,
                                        grup=b.panel.hucre_idx[b.bolme.test][gecerli])
    return Kontrol(
        "Hazard, 'son N gunde aldi mi' kuralini geciyor", alt > 0,
        f"AUC farki={fark:+.3f} [%95: {alt:+.3f}, {ust:+.3f}] "
        f"(hazard={mt.guvenli_auc(yg, hz):.3f}, kural={mt.guvenli_auc(yg, kr):.3f})",
    )


def ikili_karsilastirmalar(b, cfg) -> pl.DataFrame:
    """Tahminci ciftleri arasindaki AUC farki + eslesmis BLOK bootstrap araligi.

    "X, Y'yi geciyor" cumlesi ancak buradan cikan aralik sifiri disliyorsa
    kurulur. Bootstrap satir degil HUCRE yeniden ornekler: ayni hucrenin
    ardisik origin'leri ortustugu icin satirlar bagimsiz degil.
    """
    d = cfg.tukenme.degerlendirme
    gecerli, y, _ = _oracle_hedefleri(b.o_test, cfg.tukenme.hedef.ufuk_hafta,
                                      cfg.tukenme.hedef.karar_ufku_hafta)
    yg = y[gecerli]
    hucre = b.panel.hucre_idx[b.bolme.test][gecerli]
    ciftler = [("hazard", "kural_ikili"), ("hazard", "defter"),
               ("defter", "kural_ikili"), ("teshis_oracle_etiket", "hazard"),
               ("teshis_oracle_ozellik", "teshis_oracle_etiket")]
    satirlar = []
    for a, c in ciftler:
        if a not in b.tahminler or c not in b.tahminler:
            continue
        sa = b.tahminler[a].skor[gecerli]
        sc = b.tahminler[c].skor[gecerli]
        fark, alt, ust = mt.bootstrap_farki(yg, sa, sc, mt.guvenli_auc,
                                            d.bootstrap_orneklem, d.bootstrap_seed,
                                            grup=hucre)
        satirlar.append({"A": a, "B": c, "AUC_A": mt.guvenli_auc(yg, sa),
                         "AUC_B": mt.guvenli_auc(yg, sc), "A-B": fark,
                         "%2.5": alt, "%97.5": ust,
                         "anlamli": bool(np.isfinite(alt) and (alt > 0) == (ust > 0))})
    return pl.DataFrame(satirlar)


def kontrol_supheli_iyi(olcumler: dict) -> Kontrol:
    """CLAUDE.md 7 / README M2 tuzagi: metrik supheli derecede iyiyse leakage.

    Gozlemlenebilir veriyle kurulan model, GERCEK stok ve GERCEK hizla kurulan
    oracle kapsamasini gecemez. Gecerse ya sizinti vardir ya simulator kolaydir.
    """
    if "teshis_oracle_ozellik" not in olcumler:
        return Kontrol("Sizinti kokusu: model oracle tavanini asmiyor", True,
                       "teshis kapali (tukenme.degerlendirme.oracle_teshisi=false)")
    hz = olcumler["hazard"]["auc"]
    tavan = olcumler["teshis_oracle_ozellik"]["auc"]
    return Kontrol(
        "Sizinti kokusu: model oracle tavanini asmiyor", hz <= tavan + ESIK_SUPHE_PAYI,
        f"hazard AUC={hz:.3f}, oracle-ozellik tavani={tavan:.3f}, "
        f"pay={hz - tavan:+.3f} (izin verilen +{ESIK_SUPHE_PAYI})",
    )


# --------------------------------------------------------------------------
# grafikler
# --------------------------------------------------------------------------
def sekil_ayristirma(olcumler: dict, cfg) -> None:
    sira = ["kural_ikili", "kural", "defter", "sabit", "hazard",
            "teshis_oracle_etiket", "teshis_oracle_ozellik"]
    adlar = [a for a in sira if a in olcumler]
    auc = [olcumler[a]["auc"] for a in adlar]
    gozlem = [olcumler[a]["auc_gozlemlenebilir"] for a in adlar]
    x = np.arange(len(adlar))
    plt.figure(figsize=(10, 4.4))
    plt.bar(x - 0.2, auc, 0.4, label="gercek tukenme (oracle)", color="#3b6ea5")
    plt.bar(x + 0.2, gozlem, 0.4, label="gozlemlenebilir etiket (bize siparis)",
            color="#a5613b")
    plt.axhline(0.5, c="k", ls="--", lw=1)
    plt.xticks(x, adlar, rotation=15, fontsize=8)
    plt.ylabel("AUC")
    plt.ylim(0.35, max(0.8, max(auc + gozlem) + 0.05))
    plt.title(f"Ayirt etme gucu ayristirmasi (karar ufku "
              f"{cfg.tukenme.hedef.karar_ufku_hafta} hafta)")
    plt.legend(fontsize=8)
    _sekil_kaydet("ayristirma")


def sekil_kalibrasyon(b, olcumler: dict, cfg) -> None:
    d = cfg.tukenme.degerlendirme
    gecerli, y, _ = _oracle_hedefleri(b.o_test, cfg.tukenme.hedef.ufuk_hafta,
                                      cfg.tukenme.hedef.karar_ufku_hafta)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    lim = 0.05
    for ad, renk in (("hazard", "#3b6ea5"), ("defter", "#5a9e5a"),
                     ("kural_ikili", "#a5613b")):
        if ad not in b.tahminler:
            continue
        p = b.tahminler[ad].olasilik
        for j, (maske, hedef) in enumerate((
                (gecerli, y[gecerli]),
                (np.ones_like(gecerli, dtype=bool), b.y_test))):
            t, g, _n = mt.kalibrasyon_egrisi(hedef, p[maske], d.kalibrasyon_kova_sayisi)
            ax[j].plot(t, g, marker="o", ms=4, color=renk, label=ad)
            lim = max(lim, float(np.max(t)) if t.size else 0.0,
                      float(np.max(g)) if g.size else 0.0)
    lim = min(1.0, lim * 1.15)
    for j, baslik in enumerate(("Gercek tukenme (oracle)",
                                "Gozlemlenebilir etiket (bize siparis)")):
        ax[j].plot([0, 1], [0, 1], "k--", lw=1)
        ax[j].set_xlim(0, lim)
        ax[j].set_ylim(0, lim)
        ax[j].set_xlabel("tahmin edilen olasilik")
        ax[j].set_ylabel("gozlenen oran")
        ax[j].set_title(baslik, fontsize=10)
        ax[j].legend(fontsize=8)
    fig.suptitle("Kalibrasyon: model egitildigi soruda kalibre, "
                 "sorulan soruda degil", fontsize=11)
    _sekil_kaydet("kalibrasyon")


def sekil_mae(b, olcumler: dict, cfg) -> None:
    ufuk = cfg.tukenme.hedef.ufuk_hafta
    gecerli, _, T = _oracle_hedefleri(b.o_test, ufuk, cfg.tukenme.hedef.karar_ufku_hafta)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    adlar = [a for a in ("sabit", "kural_ikili", "defter", "hazard",
                         "teshis_oracle_etiket") if a in b.tahminler]
    ax[0].bar(adlar, [olcumler[a]["mae_gun"] for a in adlar], color="#3b6ea5")
    ax[0].set_ylabel("MAE (gun)")
    ax[0].set_title(f"Kirpilmis tukenme suresi MAE\n(ufuk {ufuk} hafta, tum canli hucreler)",
                    fontsize=9)
    ax[0].tick_params(axis="x", rotation=20, labelsize=7)
    ax[1].bar(adlar, [olcumler[a]["yanlilik_gun"] for a in adlar], color="#a5613b")
    ax[1].axhline(0, c="k", lw=1)
    ax[1].set_ylabel("ortalama isaretli hata (gun)")
    ax[1].set_title("Yanlilik: eksi = 'erken tukenir' sanma", fontsize=9)
    ax[1].tick_params(axis="x", rotation=20, labelsize=7)

    gercek = T[gecerli] * GUN_HAFTA
    for ad, renk in (("hazard", "#3b6ea5"), ("defter", "#5a9e5a")):
        if ad not in b.tahminler:
            continue
        tahmin = b.tahminler[ad].tukenme_hafta[gecerli] * GUN_HAFTA
        kova = np.clip((np.argsort(np.argsort(tahmin)) * 10 // max(tahmin.size, 1)), 0, 9)
        x = [tahmin[kova == i].mean() for i in range(10)]
        yv = [gercek[kova == i].mean() for i in range(10)]
        ax[2].plot(x, yv, marker="o", ms=4, color=renk, label=ad)
    ax[2].plot([0, ufuk * GUN_HAFTA], [0, ufuk * GUN_HAFTA], "k--", lw=1)
    ax[2].set_xlabel("tahmin edilen tukenme (gun, desil ortalamasi)")
    ax[2].set_ylabel("gercek (gun)")
    ax[2].set_title("Tahmin - gercek hizasi", fontsize=9)
    ax[2].legend(fontsize=8)
    _sekil_kaydet("mae_ve_yanlilik")


def sekil_hiz_ve_stok(b, kosu_adi: str, cfg) -> dict:
    """Hiz ve stok cikariminin dogrulugu: M2'nin kapsam maddeleri."""
    panel, bolme = b.panel, b.bolme
    idx = bolme.test
    o = Oracle(kosu_adi).etiketle(panel.anahtar["eczane_id"].to_numpy()[idx],
                                  panel.anahtar["sku_id"].to_numpy()[idx],
                                  panel.origin[idx], cfg.tukenme.hedef.ufuk_hafta)
    ad = panel.ozellik_adlari
    hiz_akis = panel.X[idx, ad.index(f"hiz_akis_{max(cfg.feature.hiz.pencereler_hafta)}h")]
    hiz_miktar = panel.X[idx, ad.index("hiz_miktar")]
    defter = panel.X[idx, ad.index("defter_stok")]
    gercek_hiz = o.origin_tuketimi           # son ufuk haftanin gercek ortalamasi
    gercek_stok = o.origin_stogu.astype(float)

    gecerli = (gercek_hiz > 0) & (hiz_akis > 0)
    r_akis = float(np.corrcoef(np.log1p(hiz_akis[gecerli]),
                               np.log1p(gercek_hiz[gecerli]))[0, 1])
    r_miktar = float(np.corrcoef(np.log1p(hiz_miktar[gecerli]),
                                 np.log1p(gercek_hiz[gecerli]))[0, 1])
    oran = float(np.median(hiz_akis[gecerli] / gercek_hiz[gecerli]))
    oran_miktar = float(np.median(hiz_miktar[gecerli] / gercek_hiz[gecerli]))
    stok_gecerli = (gercek_stok > 0) & (defter > 0)
    r_stok = float(np.corrcoef(np.log1p(defter[stok_gecerli]),
                               np.log1p(gercek_stok[stok_gecerli]))[0, 1])

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].scatter(gercek_hiz[gecerli], hiz_akis[gecerli], s=3, alpha=0.15, color="#3b6ea5")
    lim = np.percentile(gercek_hiz[gecerli], 99)
    ax[0].plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
    ax[0].plot([0, lim], [0, lim * oran], "r-", lw=1,
               label=f"medyan oran = {oran:.2f}")
    ax[0].set_xlim(0, lim)
    ax[0].set_ylim(0, lim)
    ax[0].set_xlabel("gercek haftalik tuketim")
    ax[0].set_ylabel("akis tahmini (bize gelen / hafta)")
    ax[0].set_title(f"Hiz cikarimi: sekil dogru, SEVIYE degil\nlog-korelasyon r={r_akis:.2f}",
                    fontsize=9)
    ax[0].legend(fontsize=8)

    ax[1].hist(np.clip(hiz_akis[gecerli] / gercek_hiz[gecerli], 0, 3), bins=60,
               color="#3b6ea5")
    ax[1].axvline(oran, c="r", lw=1.5, label=f"medyan={oran:.2f}")
    ax[1].axvline(1.0, c="k", ls="--", lw=1)
    ax[1].set_xlabel("akis tahmini / gercek tuketim")
    ax[1].set_title("Gorunurluk carpani (latent share_of_wallet)", fontsize=9)
    ax[1].legend(fontsize=8)

    ax[2].scatter(gercek_stok[stok_gecerli], defter[stok_gecerli], s=3, alpha=0.15,
                  color="#5a9e5a")
    lim2 = np.percentile(gercek_stok[stok_gecerli], 98)
    ax[2].plot([0, lim2], [0, lim2], "k--", lw=1)
    ax[2].set_xlim(0, lim2)
    ax[2].set_ylim(0, lim2)
    ax[2].set_xlabel("gercek eczane stogu")
    ax[2].set_ylabel("defter tahmini")
    ax[2].set_title(f"Stok cikarimi cok daha zayif\nlog-korelasyon r={r_stok:.2f}",
                    fontsize=9)
    _sekil_kaydet("hiz_ve_stok")
    return {"hiz_akis_log_korelasyon": r_akis, "hiz_miktar_log_korelasyon": r_miktar,
            "akis_gercek_medyan_oran": oran, "miktar_gercek_medyan_oran": oran_miktar,
            "defter_stok_log_korelasyon": r_stok}


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
          f"m2_config_hash={cfg.hash()}")

    b = boru_hatti(cfg, args.kosu, DATA_DIR)
    olcumler = {ad: degerlendir(ad, t, b.o_test, b.y_test, cfg)
                for ad, t in b.tahminler.items()}

    gecerli, y, _ = _oracle_hedefleri(b.o_test, cfg.tukenme.hedef.ufuk_hafta,
                                      cfg.tukenme.hedef.karar_ufku_hafta)
    print(f"panel: {b.izgara.talep.shape[0]} hucre | egitim {b.bolme.egitim.size} satir | "
          f"test {b.bolme.test.size} satir | olcum {int(gecerli.sum())} satir")
    print(f"gercek tukenme taban orani={float(y[gecerli].mean()):.3f} | "
          f"gozlemlenebilir taban orani={float(b.y_test.mean()):.3f} | "
          f"origin'de zaten stoksuz={float((~b.o_test.canli).mean()):.3f} | "
          f"listeden dusme sansuru={float(b.o_test.rakip_sansur.mean()):.3f}")
    print(f"sureler: {b.zaman}")

    tablo = pl.DataFrame([
        {"tahminci": ad, "AUC": m["auc"], "PR_AUC": m["pr_auc"],
         "ust%10": m["ust_dilim_kazanci"], "brier": m["brier"],
         "kalib_hata": m["kalibrasyon_hatasi"], "MAE_gun": m["mae_gun"],
         "MAE_gun_olayli": m["mae_gun_olayli"], "yanlilik_gun": m["yanlilik_gun"],
         "AUC_gozlem": m["auc_gozlemlenebilir"]}
        for ad, m in olcumler.items()])
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=200, float_precision=3):
        print(tablo)

    print("\nikili karsilastirmalar (eslesmis bootstrap, "
          f"{cfg.tukenme.degerlendirme.bootstrap_orneklem} tekrar):")
    with pl.Config(tbl_rows=20, tbl_width_chars=160, float_precision=4):
        print(ikili_karsilastirmalar(b, cfg))

    sekil_ayristirma(olcumler, cfg)
    sekil_kalibrasyon(b, olcumler, cfg)
    sekil_mae(b, olcumler, cfg)
    cikarim = sekil_hiz_ve_stok(b, args.kosu, cfg)
    print("\ncikarim kalitesi: " + " | ".join(f"{k}={v:.3f}" for k, v in cikarim.items()))

    kontroller = [
        kontrol_statik_sizinti(),
        kontrol_bolme(b, cfg),
        kontrol_cikis_kriteri(olcumler),
        kontrol_hazard_kurali_geciyor(b, cfg),
        kontrol_supheli_iyi(olcumler),
    ]
    if not args.hizli:
        kontroller.append(kontrol_point_in_time(args.kosu, cfg))
        kontroller.append(kontrol_determinizm(cfg, args.kosu))

    print("\n" + "=" * 118)
    print(f"{'DURUM':<8}{'KONTROL':<56}OLCUM")
    print("-" * 118)
    for k in kontroller:
        print(f"{'GECTI' if k.gecti else 'KALDI':<8}{k.ad:<56}{k.olcum}")
    print("=" * 118)
    kalan = [k.ad for k in kontroller if not k.gecti]
    print(f"grafikler: {SEKIL_DIZINI}")
    print(f"SONUC: {len(kontroller) - len(kalan)}/{len(kontroller)} gecti"
          + (f" | KALAN: {kalan}" if kalan else ""))
    sys.exit(1 if kalan else 0)


if __name__ == "__main__":
    main()
