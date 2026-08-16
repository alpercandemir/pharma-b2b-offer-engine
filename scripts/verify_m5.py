"""M5 cikis kriteri dogrulamasi.

    uv run python -m scripts.verify_m5 --kosu full
    uv run python -m scripts.verify_m5 --kosu full --hizli   (determinizm atla)

SPEC M5 cikis kriteri:
  (a) Kit SKU senaryosunda ranking-only politika ile LP politikasi
      karsilastirmasi; stockout ve karsilanmayan talep sayilari.
  (b) Kisa miatli lot senaryosunda uc politika: temizlik yok / kor iskonto /
      M2 kuplajli hedefli temizlik. Metrik: imha adedi, iade adedi, net marj,
      eczane memnuniyeti proxy'si.

UC SENARYO, TEK M4 KOSUSU. `tahsis` blogu M4'un hicbir ciktisini etkilemez
(aday havuzu, kol matrisleri, CATE tahminleri ayni), bu yuzden ogrenici bir
kez egitilir ve uc senaryo AYNI tahminlerin uzerinde kosar. Senaryolar
arasindaki fark boylece tahsis katmanina atfedilebilir, model gurultusune
degil.

Kontroller M1-M4 ile ayni disiplinde iki gruba ayrilir:
  DURUSTLUK : sizinti siniri, LP'nin fizibilitesi ve dualitesi, determinizm
  KRITER    : (a) ve (b) karsilastirmalari, D9 isaret degisimi, D6 vetosu,
              M2 kuplajinin FIILEN bagladigi (vakum kontrolu)
"""

from __future__ import annotations

import argparse
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
from eval import allocation as ev  # noqa: E402
from experiments.run import (m4_boru_hatti, m5_boru_hatti,  # noqa: E402
                             m5_duz_metrikler, m5_ihlaller)
from policy import allocate as alloc  # noqa: E402
from scripts.verify_m2 import kod_metni  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SEKIL_DIZINI = REPO_ROOT / "reports" / "figures" / "m5"

# --- CIKIS KRITERI ESIKLERI. Tuning knob'u DEGIL, kriterin kendisi. ---
# Kit senaryoda LP'nin karsilanmayan talebi ranking-only'nin en fazla bu kadari
# olmali. 1.0 olsaydi "LP hic bir sey bozmadi" da gecerdi; kriter LP'nin kit
# kaynagi FIILEN paylastirmasi.
ESIK_LP_KARSILANMAYAN_ORANI = 0.75
# Temizlik rejiminde negatif gölge fiyatli lotlarin asgari orani. Sifirsa
# isaret degisimi hic gerceklesmemis demektir ve D9 uygulanmamistir.
ESIK_NEGATIF_GOLGE_ORANI = 0.01
# M2 kuplajinin FIILEN eledigi (satir, lot) cifti sayisi. Sifirsa kisit
# dekoratiftir - SPEC 2.5'in "sabit tavan koyma" talimati bos kalir.
ESIK_KUPLAJ_ELEMESI = 1
# LP'nin primal fizibilite ve gucli dualite toleransi (goreli).
TOLERANS_DUALITE = 1e-6
# Politika katmani tepki fonksiyonunu goremez (verify_m4 ile ayni liste).
YASAK_TEPKI_ADLARI = ("sim.response", "sim/response", "tepki_hesapla",
                      "TepkiEvreni", "GercekDurum", "ground_truth")


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
# senaryolar
# --------------------------------------------------------------------------
def senaryo_configleri(profil: str) -> dict[str, Config]:
    """Uc senaryo. Dunya UCUNDE DE AYNI (dunya_hash sabit); degisen gorunum.

    `dogal`    : senaryo kadranlari kapali. Dogal dunyada aday talebi stogu
                 yalnizca birkac SKU'da asiyor; kitligin ne yaptigini burada
                 gormek zor - bu da bir bulgudur.
    `kit_stok` : tahsis edilebilir lot adetleri kisilir (cikis kriteri (a)).
    `kisa_miat`: butun lotlar yaslandirilir (cikis kriteri (b)).
    """
    return {
        "dogal": load_config(profil),
        "kit_stok": load_config(profil, gecersiz_kilma={
            "tahsis.senaryo.kit_stok_carpani": 0.25}),
        "kisa_miat": load_config(profil, gecersiz_kilma={
            "tahsis.senaryo.miad_hizlandirma_gun": 60}),
    }


# --------------------------------------------------------------------------
# DURUSTLUK
# --------------------------------------------------------------------------
def kontrol_tepki_sizintisi() -> Kontrol:
    """Tahsis katmani tepki fonksiyonunu ve ground_truth'u GORMUYOR.

    M5'in en kolay kendini kandirma bicimi: LP'ye gercek kabul olasiligini
    ya da gercek tuketim hizini vermek. O zaman "LP daha iyi tahsis ediyor"
    sonucu tahsisten degil, bilgiden gelirdi.
    """
    bulgular = []
    for yol in (REPO_ROOT / "policy" / "allocate.py",):
        kod = kod_metni(yol)
        bulgular += [f"{yol.name}:{k}" for k in YASAK_TEPKI_ADLARI if k in kod]
    return Kontrol("Tahsis katmani tepki fonksiyonunu / ground_truth'u gormuyor",
                   not bulgular,
                   f"{len(bulgular)} bulgu" + (f": {bulgular}" if bulgular else ""))


def kontrol_lp_dualitesi(c, cfg: Config) -> Kontrol:
    """LP fizibil mi ve gucli dualite tutuyor mu.

    Gölge fiyatlar raporun merkezinde; dual cozum dogrulanmadan yorumlanamaz.
    Iki sey sinaniyor:
      primal : hicbir lotun tahsisi kapasitesini asmiyor
      dualite: sum(kazanc*x) + sum(v*r) ile dual amac ayni (HiGHS'in kendi
               dualite acigi raporlanir)
    """
    asim, en_buyuk_asim, lot_sayisi = 0, 0.0, 0
    for ad in ("lp", "hedefli_temizlik"):
        for sonuc in c.sonuclar[ad]:
            lot_gor = c.lotlar[sonuc.t]
            lot_sayisi += lot_gor.L
            cekilis = np.zeros(lot_gor.L)
            i = np.flatnonzero(sonuc.teklif_maskesi)
            if i.size:
                np.add.at(cekilis, sonuc.lot[i], _cekilis(c, sonuc, i))
            fark = cekilis - lot_gor.adet
            asim += int((fark > 1e-6).sum())
            en_buyuk_asim = max(en_buyuk_asim, float(fark.max()) if fark.size else 0.0)
    return Kontrol(
        "LP primal fizibil: hicbir lotun kapasitesi asilmiyor",
        asim == 0,
        f"{lot_sayisi} lot x 2 LP politikasi, asan lot={asim}, "
        f"en buyuk asim={en_buyuk_asim:.3g} adet")


def _cekilis(c, sonuc, i: np.ndarray) -> np.ndarray:
    """Secilen kollarin ARTIMSAL lot rezervasyonu (allocate.py ile ayni formul)."""
    blok = c.blok_haritasi[sonuc.t]
    kol = sonuc.kol[i]
    nominal = blok.mat.adet[i, kol] + blok.mat.bedava[i, kol]
    taban = blok.p_x[i, 0] * blok.mat.adet[i, 0]
    return np.maximum(c.carpan * (blok.p_x[i, kol] * nominal - taban), 0.0)


def kontrol_golge_fiyat_tutarliligi(c, cfg: Config) -> Kontrol:
    """Gölge fiyat >= devam degeri ve artik varken ESIT olmali.

    LP teorisi: lot dengesi esitliginin duali, lot tuketilmediginde (r_l > 0)
    tam olarak devam degerine esittir; tuketildiginde ustune kitlik primi
    biner. Bu ozdeslik tutmuyorsa dual cozum yanlis okunuyordur ve raporun
    butun gölge fiyat yorumu cop olur.
    """
    ihlal, kontrol_edilen, en_buyuk = 0, 0, 0.0
    for ad in ("lp", "hedefli_temizlik"):
        for sonuc in c.sonuclar[ad]:
            g, v = sonuc.golge_fiyat, sonuc.lot_devam_degeri
            sonlu = np.isfinite(g)
            olcek = np.maximum(np.abs(v[sonlu]), 1.0)
            fark = (v[sonlu] - g[sonlu]) / olcek
            ihlal += int((fark > 1e-6).sum())
            en_buyuk = max(en_buyuk, float(fark.max()) if fark.size else 0.0)
            kontrol_edilen += int(sonlu.sum())
    return Kontrol(
        "Gölge fiyat >= lotun devam degeri (LP dualitesi)",
        ihlal == 0,
        f"{kontrol_edilen} lot-dual kontrol edildi, ihlal={ihlal}, "
        f"en buyuk goreli sapma={en_buyuk:.3g}")


def kontrol_determinizm(cfg: Config, m4, kosu_adi: str) -> Kontrol:
    def _ozet():
        c = m5_boru_hatti(cfg, m4, kosu_adi, DATA_DIR)
        d = m5_duz_metrikler(c, cfg, m4)
        return tuple(round(d[k], 6) for k in
                     ("m5.a.stockout_farki", "m5.a.net_marj_farki",
                      "m5.b.net_marj_kor_farki", "m5.b.net_marj_hedefli_farki"))
    a, b = _ozet(), _ozet()
    return Kontrol("Determinizm: ayni kosu ayni M5 sayilari", a == b, f"{a} == {b}")


# --------------------------------------------------------------------------
# KRITER
# --------------------------------------------------------------------------
def kontrol_kit_senaryosu(d: dict) -> Kontrol:
    """(a) Kit SKU: ranking-only vs LP. SPEC M5 cikis kriteri (a)."""
    r_k = d["m5.ranking_only.karsilanmayan_adet"]
    l_k = d["m5.lp.karsilanmayan_adet"]
    oran = l_k / r_k if r_k > 0 else float("inf")
    return Kontrol(
        "(a) Kit SKU: LP karsilanmayan talebi dusuruyor",
        r_k > 0 and oran <= ESIK_LP_KARSILANMAYAN_ORANI,
        f"karsilanmayan {r_k:,.0f} -> {l_k:,.0f} adet (oran {oran:.2f}, esik "
        f"{ESIK_LP_KARSILANMAYAN_ORANI}) | stockout "
        f"{d['m5.ranking_only.stockout_sayisi']:,.1f} -> "
        f"{d['m5.lp.stockout_sayisi']:,.1f} | net marj "
        f"{d['m5.a.net_marj_farki']:+,.0f} TL")


def kontrol_miad_senaryosu(d: dict) -> Kontrol:
    """(b) Kor iskonto zarari AZALTMAK yerine TRANSFER ediyor mu.

    SPEC 2.5: "Kisa miatli stogu eczaneye yikmak zarari transfer eder -
    satamaz, iade eder, iliski zarar gorur." Kriterin sinadigi sey bu
    MEKANIZMANIN gorunur olmasi: imha duser, iade artar.

    NET MARJIN ISARETI KRITER DEGIL, BULGUDUR. `fast`ta negatif, `full`da
    pozitif cikiyor ve sebebi imha maliyetinin lot DSF'ine gore agirliklanmasi
    (reports/m5.md 6.2). Isareti kriter yapmak, dunyaya bagli bir sonucu
    kodun dogrulugu gibi gostermek olurdu.
    """
    imha = d["m5.b.imha_kor_farki"]
    iade = d["m5.b.iade_kor_farki"]
    marj = d["m5.b.net_marj_kor_farki"]
    olculdu = all(np.isfinite(d[f"m5.kor_iskonto.{k}"]) for k in
                  ("imha_adet", "iade_adet", "net_marj", "memnuniyet_proxy"))
    return Kontrol(
        "(b) Kor iskonto: imhayi azaltirken iadeyi artiriyor (zarar TRANSFERI)",
        imha < 0 and iade > 0 and olculdu,
        f"kor iskonto - temizlik yok: imha {imha:+,.0f} adet, iade {iade:+,.0f} "
        f"adet, net marj {marj:+,.0f} TL, memnuniyet "
        f"{d['m5.b.memnuniyet_kor_farki']:+.4f}")


def kontrol_hedefleme_kazanci(d: dict) -> Kontrol:
    """(b) M2 kuplajli hedefleme, kor iskontonun imha kazancini DAHA AZ IADEYLE
    elde ediyor mu.

    SPEC 2.5'in "temizlik bir iskonto degil, HEDEFLEME problemidir" cumlesinin
    olculebilir hali. Iki politika ayni imha azalmasini elde ediyorsa fark
    tamamen iade tarafinda gorunmeli - orasi zararin nereye TRANSFER edildigi.
    """
    imha_kor = d["m5.b.imha_temizlik_kor_farki"]
    imha_hed = d["m5.b.imha_temizlik_hedefli_farki"]
    iade_kor = d["m5.b.iade_kor_farki"]
    iade_hed = d["m5.b.iade_hedefli_farki"]
    oran = iade_hed / iade_kor if iade_kor else float("nan")
    return Kontrol(
        "(b) Hedefli temizlik ayni imha kazancini daha az iadeyle aliyor",
        imha_hed <= imha_kor * 0.9 and 0 < iade_hed < iade_kor,
        f"imha (temizlik penceresi): kor {imha_kor:+,.0f} vs hedefli "
        f"{imha_hed:+,.0f} adet | iade: kor {iade_kor:+,.0f} vs hedefli "
        f"{iade_hed:+,.0f} adet (hedefli, korun iadesinin %{oran*100:.0f}'i)")


def kontrol_isaret_degisimi(d: dict) -> Kontrol:
    """D9: miad rejiminde gölge fiyat NEGATIFE doner, normal rejimde donmez.

    Iki yonlu kontrol. Yalnizca "negatif var mi" sorulsaydi, salvage egrisi
    kapali iken de negatif cikan bir hata gozden kacardi.
    """
    n_h = d.get("m5.golge.hedefli_temizlik.negatif_golge_lot_orani", 0.0)
    n_l = d.get("m5.golge.lp.negatif_golge_lot_orani", 0.0)
    pencere_disi = d.get("m5.golge.hedefli_temizlik.pencere_disi_negatif_lot", 0.0)
    return Kontrol(
        "D9: gölge fiyat temizlik rejiminde isaret degistiriyor",
        n_h >= ESIK_NEGATIF_GOLGE_ORANI and n_l == 0.0 and pencere_disi == 0.0,
        f"negatif gölge lot orani: temizlik rejimi {n_h:.3f} (esik "
        f"{ESIK_NEGATIF_GOLGE_ORANI}), normal rejim {n_l:.3f} | pencere DISINDA "
        f"negatif lot={pencere_disi:.0f} | en dusuk gölge "
        f"{d.get('m5.golge.hedefli_temizlik.golge_asgari', float('nan')):.3f} TL/adet")


def kontrol_rejim_mf_derinligi(d: dict) -> Kontrol:
    """D9 davranis kaniti: negatif gölge fiyat MF derinligini artiriyor mu.

    Isaret degisimi tek basina bir tablo. Kriter, o degisimin KARARI
    degistirmesi: ayni politikanin temizlik penceresindeki MF derinligi,
    penceredisindakinden buyuk olmali.
    """
    ic = d.get("m5.hedefli_temizlik.ortalama_mf_temizlik", float("nan"))
    dis = d.get("m5.hedefli_temizlik.ortalama_mf_normal", float("nan"))
    temiz_teklif = d.get("m5.hedefli_temizlik.teklif_sayisi_temizlik", 0.0)
    return Kontrol(
        "D9: negatif gölge fiyat MF derinligini artiriyor",
        temiz_teklif > 0 and np.isfinite(ic) and np.isfinite(dis) and ic > dis,
        f"hedefli_temizlik ortalama MF: temizlik penceresi {ic:.4f} vs normal "
        f"{dis:.4f} ({temiz_teklif:.0f} temizlik teklifi) | temizlik rejimi "
        f"kapaliyken pencere teklifi="
        f"{d.get('m5.lp.teklif_sayisi_temizlik', 0.0):.0f}")


def kontrol_kuplaj_bagliyor(c, cfg: Config) -> Kontrol:
    """SPEC 2.5 vakum kontrolu: M2 kuplaji FIILEN aday eliyor mu.

    "Sabit bir tavan koyma" talimatinin karsiligi ancak kisit BAGLADIGINDA
    anlamlidir. Suzgecin kendisi sayiyor (policy/allocate.py `Kolonlar.elenen`);
    politikalar arasi kolon sayisi farkina bakmak yaniltirdi, cunku kor
    iskontonun derin-MF dayatmasi da kolon eliyor.
    """
    o = c.olcumler["hedefli_temizlik"]
    toplam = o.elenen.get("hiz_kuplaji", 0)
    pencerede = o.elenen.get("hiz_kuplaji_temizlik", 0)
    return Kontrol(
        "SPEC 2.5: M2 tuketim hizi kuplaji fiilen aday eliyor",
        toplam >= ESIK_KUPLAJ_ELEMESI,
        f"kuplaj {toplam:,} (satir, lot, kol) ucluslunu eledi; {pencerede:,} tanesi "
        f"temizlik penceresinde (esik {ESIK_KUPLAJ_ELEMESI}) | diger suzgecler: "
        f"{ {k: v for k, v in o.elenen.items() if not k.startswith('hiz_')} }")


def kontrol_veto_yetkisi(c, cfg: Config, m4) -> Kontrol:
    """D6: temizlik rejimi vetoyu ASMIYOR (SPEC 2.5 acik hukmu)."""
    ihlal = m5_ihlaller(c, cfg, m4)
    kritik = {k: v for k, v in ihlal.items() if v}
    return Kontrol(
        "D6: temizlik rejimi hicbir vetoyu asmiyor",
        not kritik,
        f"ihlal={kritik or '{}'} (raf omru tabani, hiz kuplaji, stok, kredi, "
        f"izinli kol)")


def kontrol_butunluk_acigi(c) -> Kontrol:
    """LP gevsetmesinden teslim edilebilir politikaya gecisin bedeli.

    Kriter bir esik degil, acigin OLCULMUS olmasi: LP degeri bir UST SINIRDIR
    ve rapor o siniri teslim edilen politika sanmamali.
    """
    satirlar = []
    for ad in ("lp", "hedefli_temizlik"):
        o = c.olcumler[ad]
        oran = o.butunluk_acigi / abs(o.lp_teklif_degeri) if o.lp_teklif_degeri else float("nan")
        satirlar.append(f"{ad}: LP {o.lp_teklif_degeri:,.0f} TL -> yuvarlanmis "
                        f"{o.lp_teklif_degeri - o.butunluk_acigi:,.0f} TL "
                        f"(acik %{oran*100:.2f}, {o.kesirli_sutun:.0f} kesirli sutun)")
    return Kontrol("Butunluk acigi olculdu (LP degeri bir UST SINIR)",
                   all(np.isfinite(c.olcumler[a].butunluk_acigi) for a in
                       ("lp", "hedefli_temizlik")),
                   " | ".join(satirlar))


# --------------------------------------------------------------------------
# grafikler
# --------------------------------------------------------------------------
def sekil_kit_senaryosu(d_kit: dict, d_dogal: dict) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    politikalar = ["ranking_only", "lp"]
    renk = ["#a5613b", "#3b6ea5"]
    for j, (metrik, baslik, birim) in enumerate((
            ("karsilanmayan_adet", "Karsilanmayan talep", "adet"),
            ("stockout_sayisi", "Stockout yasayan teklif", "teklif"),
            ("net_marj", "Net marj", "TL"))):
        x = np.arange(2)
        for k, (etiket, d) in enumerate((("dogal", d_dogal), ("kit stok", d_kit))):
            deger = [d[f"m5.{p}.{metrik}"] for p in politikalar]
            ax[j].bar(x + k * 0.38 - 0.19, deger, width=0.36,
                      color=renk, alpha=1.0 if k else 0.45,
                      label=etiket if j == 0 else None)
        ax[j].set_xticks(x)
        ax[j].set_xticklabels(politikalar, fontsize=8)
        ax[j].set_title(f"{baslik} ({birim})", fontsize=9)
        ax[j].axhline(0, c="k", lw=1)
    ax[0].legend(fontsize=8, title="senaryo", title_fontsize=8)
    fig.suptitle("M5 (a): kit stok altinda ranking-only vs LP tahsisi", fontsize=10)
    _sekil_kaydet("kit_senaryosu")


def sekil_golge_fiyat(c, cfg: Config) -> None:
    """D9'un merkezi grafigi: gölge fiyat x kalan raf omru, iki rejim."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    tetik = cfg.tahsis.temizlik.tetik_gun
    for j, ad in enumerate(("lp", "hedefli_temizlik")):
        t = c.golge.filter(pl.col("politika") == ad)
        g = t["golge_fiyat"].to_numpy().astype(float)
        gun = t["kalan_gun"].to_numpy().astype(float)
        adet = t["adet"].to_numpy().astype(float)
        sonlu = np.isfinite(g)
        boyut = 8 + 60 * adet[sonlu] / max(adet[sonlu].max(), 1.0)
        ax[j].scatter(gun[sonlu], g[sonlu], s=boyut,
                      c=np.where(g[sonlu] < 0, "#a5613b", "#3b6ea5"), alpha=0.6)
        ax[j].axhline(0, c="k", lw=1.2)
        ax[j].axvline(tetik, c="#888", ls="--", lw=1,
                      label=f"temizlik tetigi ({tetik:.0f} gun)")
        ax[j].set_xlabel("lotun kalan raf omru (gun)")
        ax[j].set_ylabel("gölge fiyat (TL/adet)")
        ax[j].set_xscale("symlog")
        negatif = int((g[sonlu] < 0).sum())
        ax[j].set_title(f"{ad}: {negatif}/{sonlu.sum()} lotta negatif", fontsize=9)
        ax[j].legend(fontsize=8)
    fig.suptitle("D9: ayni LP, isaret degisimi — stok varliktan yukumluluge",
                 fontsize=10)
    _sekil_kaydet("golge_fiyat")


def sekil_miad_senaryosu(d: dict) -> None:
    """(b) uc politika, dort metrik — hepsi "TEMIZLIK YOK"a GORE FARK olarak.

    Seviye cizilseydi grafik okunamazdi ve bu bir estetik tercih degil, olcunun
    kendisiyle ilgili: imha seviyesi (~66.000 adet) yapisal fazla stoktan
    besleniyor ve politikalar arasinda binde birkac oynuyor. Politikanin FIILEN
    oynattigi miktar farktir; taban seviye her panelin basligina yaziliyor ki
    farkin hangi buyuklugun uzerine bindigi kaybolmasin.
    """
    sira = ["kor_iskonto", "hedefli_temizlik"]
    etiket = ["kor iskonto", "hedefli temizlik\n(M2 kuplajli)"]
    renk = ["#a5613b", "#5a9e5a"]
    panel = (("imha_adet_temizlik", "Imha (temizlik penceresi)", "adet", "{:+,.0f}"),
             ("iade_adet", "Iade", "adet", "{:+,.0f}"),
             ("net_marj", "Net marj", "TL", "{:+,.0f}"),
             ("memnuniyet_proxy", "Eczane memnuniyeti proxy'si", "", "{:+.4f}"))
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    for j, (metrik, baslik, birim, bicim) in enumerate(panel):
        taban = d[f"m5.lp.{metrik}"]
        fark = [d[f"m5.{p}.{metrik}"] - taban for p in sira]
        ax[j].bar(np.arange(2), fark, color=renk, width=0.55)
        for i, v in enumerate(fark):
            ax[j].annotate(bicim.format(v), (i, v), ha="center", fontsize=8,
                           va="bottom" if v >= 0 else "top",
                           xytext=(0, 4 if v >= 0 else -4), textcoords="offset points")
        ax[j].axhline(0, c="k", lw=1.2)
        ax[j].set_xticks(np.arange(2))
        ax[j].set_xticklabels(etiket, fontsize=8)
        ax[j].set_ylabel(f"temizlik yok'a gore fark ({birim})" if birim else
                         "temizlik yok'a gore fark")
        ax[j].margins(y=0.22)
        ax[j].set_title(f"{baslik}\ntemizlik yok tabani: "
                        + (f"{taban:,.4f}" if metrik == "memnuniyet_proxy"
                           else f"{taban:,.0f} {birim}"), fontsize=9)
    fig.suptitle("M5 (b): kisa miatli lot senaryosunda uc politika — "
                 "kor iskonto imhayi azaltirken iadeyi patlatiyor", fontsize=10)
    _sekil_kaydet("miad_senaryosu")


def sekil_salvage_egrisi(cfg: Config) -> None:
    """Salvage fonksiyonunun kendisi: nerede isaret degistiriyor."""
    t = cfg.tahsis.temizlik
    gun = np.linspace(0, t.tetik_gun * 2, 400)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for egri in ("lineer", "eksponansiyel", "basamakli"):
        alt = cfg.model_copy(update={"tahsis": cfg.tahsis.model_copy(
            update={"temizlik": t.model_copy(update={"deger_egrisi": egri})})})
        _, v = alloc.salvage_degeri(alt, np.full(gun.size, 100.0),
                                    np.full(gun.size, 0.05), gun)
        stil = "-" if egri == t.deger_egrisi else "--"
        ax.plot(gun, v, stil, lw=2 if egri == t.deger_egrisi else 1.2,
                label=egri + (" (aktif)" if egri == t.deger_egrisi else ""))
    ax.axhline(0, c="k", lw=1.2)
    ax.axvline(t.tetik_gun, c="#888", ls=":", lw=1)
    ax.set_xlabel("lotun kalan raf omru (gun)")
    ax.set_ylabel("birim devam degeri (TL/adet, DSF=100, marj=%5)")
    ax.set_title("SPEC 2.5 dinamik salvage: degerin isaret degistirdigi nokta",
                 fontsize=9)
    ax.legend(fontsize=8)
    _sekil_kaydet("salvage_egrisi")


# --------------------------------------------------------------------------
def _tablo(c) -> pl.DataFrame:
    return pl.DataFrame([
        {"politika": ad, "teklif": o.teklif_sayisi,
         "temizlik_teklifi": o.teklif_sayisi_temizlik,
         "talep": o.talep_adet, "karsilanmayan": o.karsilanmayan_adet,
         "stockout": o.stockout_sayisi, "iade": o.iade_adet,
         "imha_temizlik": o.imha_adet_temizlik, "imha": o.imha_adet,
         "brut_marj": o.brut_marj, "net_marj": o.net_marj,
         "memnuniyet": o.memnuniyet_proxy,
         "MF_temizlik": o.ortalama_mf_temizlik, "MF_normal": o.ortalama_mf_normal}
        for ad, o in c.olcumler.items()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kosu", default="full", help="data/<kosu> altindaki dunya")
    ap.add_argument("--profil", default=None, help="varsayilan: kosu manifestindeki profil")
    ap.add_argument("--hizli", action="store_true", help="determinizm kontrolunu atla")
    args = ap.parse_args()

    manifest = Run(args.kosu).read_manifest()
    profil = args.profil or manifest["profil"]
    configler = senaryo_configleri(profil)
    temel = configler["dogal"]
    print(f"kosu={args.kosu} profil={profil} dunya_config_hash={manifest['config_hash']} "
          f"m5_config_hash={temel.hash()}")

    # M4 bir kez: `tahsis` blogu M4'un hicbir ciktisini etkilemiyor.
    m4 = m4_boru_hatti(temel, args.kosu, DATA_DIR)
    print(f"M4 yeniden kullanildi: olcum origin={m4.olcum_originleri}, "
          f"{sum(b.teklifler.height for b in m4.bloklar)} vetosuz aday satiri")

    ciktilar, duzler = {}, {}
    for ad, cfg in configler.items():
        c = m5_boru_hatti(cfg, m4, args.kosu, DATA_DIR)
        c.blok_haritasi = {b.gor.t: b for b in m4.bloklar}
        ciktilar[ad], duzler[ad] = c, m5_duz_metrikler(c, cfg, m4)
        print(f"\n=== SENARYO: {ad} "
              f"(kit_stok_carpani={cfg.tahsis.senaryo.kit_stok_carpani}, "
              f"miad_hizlandirma_gun={cfg.tahsis.senaryo.miad_hizlandirma_gun}) "
              f"| sure={c.zaman}")
        with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=250, float_precision=3):
            print(_tablo(c))

    print("\nGÖLGE FIYAT (temizlik rejimi, kisa miat senaryosu):")
    g = ciktilar["kisa_miat"].golge.filter(pl.col("politika") == "hedefli_temizlik")
    with pl.Config(tbl_rows=12, tbl_cols=12, tbl_width_chars=200, float_precision=3):
        print(g.sort("kalan_gun").select(
            ["lot_id", "kalan_gun", "adet", "dsf", "devam_degeri", "golge_fiyat",
             "temizlik_penceresinde"]).head(12))

    sekil_kit_senaryosu(duzler["kit_stok"], duzler["dogal"])
    sekil_golge_fiyat(ciktilar["kisa_miat"], configler["kisa_miat"])
    sekil_miad_senaryosu(duzler["kisa_miat"])
    sekil_salvage_egrisi(temel)

    kontroller = [
        kontrol_tepki_sizintisi(),
        kontrol_lp_dualitesi(ciktilar["kit_stok"], configler["kit_stok"]),
        kontrol_golge_fiyat_tutarliligi(ciktilar["kisa_miat"], configler["kisa_miat"]),
        kontrol_veto_yetkisi(ciktilar["kisa_miat"], configler["kisa_miat"], m4),
        kontrol_kuplaj_bagliyor(ciktilar["kisa_miat"], configler["kisa_miat"]),
        kontrol_kit_senaryosu(duzler["kit_stok"]),
        kontrol_miad_senaryosu(duzler["kisa_miat"]),
        kontrol_hedefleme_kazanci(duzler["kisa_miat"]),
        kontrol_isaret_degisimi(duzler["kisa_miat"]),
        kontrol_rejim_mf_derinligi(duzler["kisa_miat"]),
        kontrol_butunluk_acigi(ciktilar["kit_stok"]),
    ]
    if not args.hizli:
        kontroller.append(kontrol_determinizm(configler["kisa_miat"], m4, args.kosu))

    print("\n" + "=" * 140)
    print(f"{'DURUM':<8}{'KONTROL':<62}OLCUM")
    print("-" * 140)
    for k in kontroller:
        print(f"{'GECTI' if k.gecti else 'KALDI':<8}{k.ad:<62}{k.olcum}")
    print("=" * 140)
    kalan = [k.ad for k in kontroller if not k.gecti]
    print(f"grafikler: {SEKIL_DIZINI}")
    print(f"SONUC: {len(kontroller) - len(kalan)}/{len(kontroller)} gecti"
          + (f" | KALAN: {kalan}" if kalan else ""))
    sys.exit(1 if kalan else 0)


if __name__ == "__main__":
    main()
