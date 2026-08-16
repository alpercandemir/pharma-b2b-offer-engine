"""M1 cikis kriteri dogrulamasi.

    python -m scripts.verify_m1 --kosu full
    python -m scripts.verify_m1 --kosu full --knob-taramasi talep.dagilim.sifir_sisirme --degerler 0.15,0.34,0.55

Her kontrol bir GECTI/KALDI satiri uretir ve ilgili grafigi
reports/figures/m1/ altina yazar. Iddia yok, olcum var.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from core.config import config_yukle  # noqa: E402
from core.io import LATENT_KOLONLAR, Kosu  # noqa: E402
from core.rng import SeedBankasi  # noqa: E402
from sim.calendar import GUN_HAFTA  # noqa: E402
from sim.world import dunya_kos  # noqa: E402

SEKIL_DIZINI = Path(__file__).resolve().parent.parent / "reports" / "figures" / "m1"

# M1 cikis kriteri esikleri. Bunlar "dunyanin gercekci olma" testleridir,
# tuning knob'u degildir: gevsetilirse kriter anlamini yitirir.
ESIK_SEYREKLIK = 0.80          # tum (eczane, SKU, hafta) hucrelerinin en az %80'i sifir
ESIK_UZUN_KUYRUK = 0.45        # en yuksek %10 SKU hacmin en az %45'ini almali
ESIK_MEVSIMSEL_KONTRAST = 1.8  # mevsimsel kategori oynakligi / kronik oynakligi
ESIK_CESIT_SURUKLENME = 0.10   # cesit orani ufuk boyunca en fazla bu kadar goreli kayabilir
ESIK_ECZANE_KAPSAMA = 12.0     # eczane stogu hafta cinsinden en fazla bu kadar olabilir


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
def kontrol_olcek(kosu: Kosu, cfg) -> Kontrol:
    u = kosu.oku_gozlemlenebilir("urunler").height
    e = kosu.oku_gozlemlenebilir("eczaneler").height
    w = kosu.oku_gozlemlenebilir("takvim").height
    gecti = (e, u, w) == (cfg.profil.eczane_sayisi, cfg.profil.sku_sayisi, cfg.profil.hafta_sayisi)
    return Kontrol("Olcek (eczane x SKU x hafta)", gecti, f"{e} x {u} x {w}")


def kontrol_seyreklik(kosu: Kosu, cfg) -> tuple[Kontrol, Kontrol]:
    h = kosu.oku_gercek("hucre_haftalik")
    P, S, W = cfg.profil.eczane_sayisi, cfg.profil.sku_sayisi, cfg.profil.hafta_sayisi
    toplam_hucre = P * S * W
    sifir_olmayan = (h["gercek_tuketim"] > 0).sum()
    sifir_orani = 1.0 - sifir_olmayan / toplam_hucre

    aktif = h.filter(pl.col("cesitte_var"))
    aktif_sifir = (aktif["gercek_tuketim"] == 0).mean()

    hucre_bazli = (
        h.group_by(["eczane_id", "sku_id"])
        .agg(pl.col("gercek_tuketim").gt(0).sum().alias("dolu_hafta"))
    )
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(hucre_bazli["dolu_hafta"].to_numpy(), bins=50, color="#3b6ea5")
    ax[0].set_title("Hucre basina sifir-olmayan hafta sayisi")
    ax[0].set_xlabel(f"hafta (toplam {W})")
    ax[0].set_ylabel("hucre sayisi")
    ax[1].bar(
        ["tum grid", "cesitte olan"],
        [sifir_orani, aktif_sifir],
        color=["#3b6ea5", "#a5613b"],
    )
    ax[1].axhline(ESIK_SEYREKLIK, ls="--", c="k", lw=1)
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Sifir talep orani")
    for i, v in enumerate([sifir_orani, aktif_sifir]):
        ax[1].text(i, v + 0.02, f"{v:.1%}", ha="center")
    _sekil_kaydet("seyreklik")

    k1 = Kontrol(
        "Seyreklik: cogu hucre cogu hafta sifir",
        sifir_orani >= ESIK_SEYREKLIK,
        f"tum grid {sifir_orani:.1%} sifir (esik {ESIK_SEYREKLIK:.0%}); "
        f"cesitte olan hucrelerde {aktif_sifir:.1%}",
    )

    hacim = (
        h.group_by("sku_id").agg(pl.col("gercek_tuketim").sum())
        .sort("gercek_tuketim", descending=True)["gercek_tuketim"].to_numpy()
    )
    kumulatif = np.cumsum(hacim) / hacim.sum()
    ust_yuzde10 = kumulatif[max(0, int(0.10 * len(kumulatif)) - 1)]
    plt.figure(figsize=(5.5, 4))
    plt.plot(np.arange(1, len(kumulatif) + 1) / len(kumulatif), kumulatif, lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="tekduze")
    plt.axvline(0.10, c="r", ls=":", lw=1)
    plt.xlabel("SKU orani (hacme gore sirali)")
    plt.ylabel("kumulatif hacim payi")
    plt.title(f"Uzun kuyruk: ust %10 SKU = hacmin %{ust_yuzde10*100:.0f}'i")
    plt.legend()
    _sekil_kaydet("uzun_kuyruk")

    k2 = Kontrol(
        "Uzun kuyruk (hacim yogunlasmasi)",
        ust_yuzde10 >= ESIK_UZUN_KUYRUK,
        f"ust %10 SKU hacmin %{ust_yuzde10*100:.1f}'ini aliyor (esik %{ESIK_UZUN_KUYRUK*100:.0f})",
    )
    return k1, k2


def kontrol_durgunluk(kosu: Kosu, cfg) -> tuple[Kontrol, Kontrol]:
    """Dunyanin istenmeyen sistematik kaymasi var mi?

    Ikisi de M1 kabul edildikten SONRA bulunan gercek hatalardi (bkz.
    reports/m1.md 3.6-3.8); bir daha sessizce geri gelmesinler diye kontrol.
    """
    h = kosu.oku_gercek("hucre_haftalik")
    W = cfg.profil.hafta_sayisi
    g = (h.group_by("hafta").agg(pl.col("cesitte_var").sum().alias("aktif")).sort("hafta"))
    seri = g["aktif"].to_numpy().astype(float)
    bas, son = seri[0], seri[-1]
    suruklenme = abs(son - bas) / max(bas, 1)

    aktif = h.filter(pl.col("cesitte_var"))
    haftalik_tuketim = aktif["gercek_tuketim"].sum() / W
    haftalik_stok = (aktif.group_by("hafta").agg(pl.col("gercek_eczane_stogu").sum())
                     ["gercek_eczane_stogu"].mean())
    kapsama = haftalik_stok / max(haftalik_tuketim, 1e-9)

    stok_serisi = (aktif.group_by("hafta").agg(pl.col("gercek_eczane_stogu").sum())
                   .sort("hafta")["gercek_eczane_stogu"].to_numpy())
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(seri, lw=2, color="#3b6ea5")
    ax[0].set_ylim(0, max(seri) * 1.25)
    ax[0].set_title(f"Cesitte olan hucre sayisi\n(goreli kayma {suruklenme:.1%})")
    ax[0].set_xlabel("hafta")
    ax[1].plot(stok_serisi / max(haftalik_tuketim, 1e-9), lw=2, color="#a5613b")
    ax[1].axhline(ESIK_ECZANE_KAPSAMA, ls="--", c="k", lw=1)
    ax[1].set_title(f"Eczane stogu (hafta cinsinden kapsama)\nortalama {kapsama:.1f}")
    ax[1].set_xlabel("hafta")
    _sekil_kaydet("durgunluk")

    return (
        Kontrol("Cesit orani ufuk boyunca durgun", suruklenme <= ESIK_CESIT_SURUKLENME,
                f"hafta 0: {bas:.0f} hucre -> hafta {W-1}: {son:.0f} "
                f"(goreli kayma {suruklenme:.1%}, esik {ESIK_CESIT_SURUKLENME:.0%})"),
        Kontrol("Eczane stogu makul kapsamada", kapsama <= ESIK_ECZANE_KAPSAMA,
                f"ortalama {kapsama:.1f} hafta kapsama (esik {ESIK_ECZANE_KAPSAMA:.0f})"),
    )


def kontrol_iade_kuplaji(kosu: Kosu) -> Kontrol:
    """Iade tuketim hiziyla kupleli mi, yoksa duz bir gun esigi mi?

    SPEC 2.5: ayni kalan raf omru, yavas eczane icin zayi, hizli eczane icin
    sorun degil. Iade orani hucre hizina gore AZALMALI.
    """
    ia = kosu.oku_gozlemlenebilir("iadeler")
    h = kosu.oku_gercek("hucre_haftalik").filter(pl.col("cesitte_var"))
    sevk = kosu.oku_gozlemlenebilir("sevkiyat_satirlari")
    if ia.height == 0:
        return Kontrol("Iade tuketim hiziyla kupleli", False, "hic iade olusmadi")

    hiz = h.group_by(["eczane_id", "sku_id"]).agg(
        (pl.col("gercek_tuketim").sum() / h["hafta"].n_unique()).alias("hafta_hizi"))
    gelen = sevk.group_by(["eczane_id", "sku_id"]).agg(pl.col("adet").sum().alias("sevk"))
    d = (gelen.join(ia.group_by(["eczane_id", "sku_id"]).agg(pl.col("iade_adet").sum()),
                    on=["eczane_id", "sku_id"], how="left")
         .join(hiz, on=["eczane_id", "sku_id"], how="left")
         .with_columns(pl.col("iade_adet").fill_null(0), pl.col("hafta_hizi").fill_null(0.0)))
    d = d.with_columns(
        pl.when(pl.col("hafta_hizi") < 0.1).then(pl.lit("1_cok_yavas"))
        .when(pl.col("hafta_hizi") < 0.5).then(pl.lit("2_yavas"))
        .when(pl.col("hafta_hizi") < 2.0).then(pl.lit("3_orta"))
        .otherwise(pl.lit("4_hizli")).alias("dilim"))
    ozet = (d.group_by("dilim")
            .agg((pl.col("iade_adet").sum() / pl.col("sevk").sum()).alias("iade_orani"),
                 pl.len().alias("hucre"))
            .sort("dilim"))
    oranlar = ozet["iade_orani"].to_numpy()

    plt.figure(figsize=(6, 4))
    plt.bar(ozet["dilim"].to_list(), oranlar, color="#a5613b")
    plt.ylabel("iade / sevk orani")
    plt.title("Iade orani hucre hizina gore\n(yavas hucrede yuksek olmali)")
    _sekil_kaydet("iade_kuplaji")

    return Kontrol(
        "Iade tuketim hiziyla kupleli (duz esik degil)",
        oranlar[0] > oranlar[-1],
        f"cok_yavas={oranlar[0]:.3f} > hizli={oranlar[-1]:.3f} "
        f"(toplam iade/sevk = {ia['iade_adet'].sum()/sevk['adet'].sum():.3f})",
    )


def kontrol_mevsimsellik(kosu: Kosu, cfg) -> Kontrol:
    h = kosu.oku_gercek("hucre_haftalik")
    urun = kosu.oku_gozlemlenebilir("urunler").select(["sku_id", "kategori_kod"])
    takvim = kosu.oku_gozlemlenebilir("takvim").select(["hafta", "ay"])
    birlesik = h.join(urun, on="sku_id").join(takvim, on="hafta")
    aylik = (
        birlesik.group_by(["kategori_kod", "ay"])
        .agg(pl.col("gercek_tuketim").mean().alias("ort"))
        .sort(["kategori_kod", "ay"])
    )

    kronik = [k.kod for k in cfg.urun.kategoriler if k.kronik]
    mevsimsel = [k.kod for k in cfg.urun.kategoriler if not k.kronik and k.kod != "MEDIKAL"]

    def _oynaklik(kod: str) -> float:
        x = aylik.filter(pl.col("kategori_kod") == kod).sort("ay")["ort"].to_numpy()
        return float(x.std() / max(x.mean(), 1e-9))

    kronik_cv = float(np.mean([_oynaklik(k) for k in kronik]))
    mevsim_cv = float(np.mean([_oynaklik(k) for k in mevsimsel]))
    oran = mevsim_cv / max(kronik_cv, 1e-9)

    plt.figure(figsize=(9, 4.5))
    for kod in [k.kod for k in cfg.urun.kategoriler]:
        x = aylik.filter(pl.col("kategori_kod") == kod).sort("ay")
        y = x["ort"].to_numpy()
        plt.plot(x["ay"].to_numpy(), y / y.mean(), marker="o", ms=3,
                 lw=2.2 if kod in kronik else 1.2,
                 ls="-" if kod not in kronik else "--", label=kod)
    plt.axhline(1.0, c="k", lw=0.6)
    plt.xlabel("ay")
    plt.ylabel("normalize ortalama tuketim")
    plt.title(f"Mevsimsellik (kesikli = kronik). mevsimsel/kronik CV orani = {oran:.2f}")
    plt.legend(ncol=6, fontsize=8)
    _sekil_kaydet("mevsimsellik")

    return Kontrol(
        "Mevsimsellik gorunur, kronik duz",
        oran >= ESIK_MEVSIMSEL_KONTRAST,
        f"mevsimsel CV={mevsim_cv:.3f}, kronik CV={kronik_cv:.3f}, oran={oran:.2f} "
        f"(esik {ESIK_MEVSIMSEL_KONTRAST})",
    )


def _etkilenen_skular(satir: dict, urun: pl.DataFrame) -> list[str]:
    """Olay satirindaki 'hedef' alanini SKU listesine cevirir."""
    if satir["kapsam"] == "GLOBAL":
        return urun["sku_id"].to_list()
    if satir["kapsam"] == "KATEGORI_AKUT":
        return urun.filter(pl.col("kategori_kod") == satir["hedef"])["sku_id"].to_list()
    return satir["hedef"].split(",")


def kontrol_olay_etkisi(kosu: Kosu, cfg) -> Kontrol:
    """Olay etudu. Sadece ETKILENEN SKU'lar uzerinden; tum evren uzerinden
    olculurse tek kategoriye vuran olaylar sulanip gorunmez olur."""
    h = kosu.oku_gercek("hucre_haftalik")
    olaylar = kosu.oku_gercek("olaylar_gercek")
    siparis = kosu.oku_gozlemlenebilir("siparisler")
    urun = kosu.oku_gozlemlenebilir("urunler").select(["sku_id", "kategori_kod"])
    W = cfg.profil.hafta_sayisi

    tuketim_sku = (h.group_by(["sku_id", "hafta"]).agg(pl.col("gercek_tuketim").sum())
                   .rename({"gercek_tuketim": "adet"}))
    siparis_sku = (siparis.group_by(["sku_id", "hafta"]).agg(pl.col("talep_adet").sum())
                   .rename({"talep_adet": "adet"}))

    def _seri(tablo: pl.DataFrame, skular: list[str]) -> np.ndarray:
        alt = tablo.filter(pl.col("sku_id").is_in(skular)).group_by("hafta").agg(
            pl.col("adet").sum()).sort("hafta")
        seri = np.zeros(W)
        seri[alt["hafta"].to_numpy()] = alt["adet"].to_numpy()
        return seri

    pencere = np.arange(-6, 7)
    tipler = sorted(set(olaylar["tip"].to_list()))
    fig, axlar = plt.subplots(1, len(tipler), figsize=(3.3 * len(tipler), 3.8), sharey=True)
    axlar = np.atleast_1d(axlar)
    bulgular: dict[tuple[str, str], np.ndarray] = {}

    for ax, tip in zip(axlar, tipler):
        satirlar = olaylar.filter(pl.col("tip") == tip).to_dicts()
        yiginlar = {"tuketim": [], "siparis": []}
        for satir in satirlar:
            skular = _etkilenen_skular(satir, urun)
            t0 = satir["yururluk_hafta"]
            for tablo, ad in ((tuketim_sku, "tuketim"), (siparis_sku, "siparis")):
                seri = _seri(tablo, skular)
                # Taban: olayin penceresi disindaki kendi ortalamasi.
                dis = np.ones(W, dtype=bool)
                dis[max(0, t0 - 8): min(W, t0 + 12)] = False
                taban = max(seri[dis].mean(), 1e-9)
                idx = t0 + pencere
                gecerli = (idx >= 0) & (idx < W)
                v = np.full(len(pencere), np.nan)
                v[gecerli] = seri[idx[gecerli]] / taban
                yiginlar[ad].append(v)
        for ad, renk in (("tuketim", "#3b6ea5"), ("siparis", "#a5613b")):
            if not yiginlar[ad]:
                continue
            ort = np.nanmean(np.array(yiginlar[ad]), axis=0)
            bulgular[(tip, ad)] = ort
            ax.plot(pencere, ort, marker="o", ms=3, label=ad, color=renk)
        ax.axvline(0, c="k", ls="--", lw=1)
        ax.axhline(1.0, c="k", lw=0.5)
        ax.set_title(tip.replace("_", "\n"), fontsize=8)
        ax.set_xlabel("olaya gore hafta")
    axlar[0].set_ylabel("olay oncesi tabana gore kat")
    axlar[0].legend(fontsize=8)
    _sekil_kaydet("olay_etkisi")

    # Kritik iddia (D4): referans kur guncellemesinde SIPARIS one cekilir,
    # TUKETIM degismez. Ayrisma sayisal olarak gosterilmeli.
    kur = "REFERANS_KUR_GUNCELLEME"
    antic = slice(0, 6)      # -6..-1
    parcalar, gecti = [], True
    if (kur, "siparis") in bulgular:
        s_zirve = float(np.nanmax(bulgular[(kur, "siparis")][antic]))
        t_ort = float(np.nanmean(bulgular[(kur, "tuketim")][antic]))
        gecti &= s_zirve >= 1.5 and abs(t_ort - 1.0) < 0.15
        parcalar.append(f"KUR: siparis zirvesi={s_zirve:.2f}x, tuketim={t_ort:.2f}x")
    for tip, beklenen in (("EPIDEMI_DALGASI", "yukari"), ("TITCK_GERI_CEKME", "asagi")):
        if (tip, "tuketim") in bulgular:
            sonra = bulgular[(tip, "tuketim")][7:]
            deger = float(np.nanmax(sonra) if beklenen == "yukari" else np.nanmin(sonra))
            gecti &= (deger > 1.5) if beklenen == "yukari" else (deger < 0.5)
            parcalar.append(f"{tip.split('_')[0]}: tuketim={deger:.2f}x")
    return Kontrol("Olay etkisi dogru kanalda ve olculebilir buyuklukte",
                   gecti, " | ".join(parcalar))


def kontrol_fefo(kosu: Kosu, cfg) -> tuple[Kontrol, Kontrol]:
    lotlar = kosu.oku_gozlemlenebilir("stok_lotlari")
    sevk = kosu.oku_gozlemlenebilir("sevkiyat_satirlari")
    latent = kosu.oku_gercek("latent_eczane").select(["eczane_id", "miad_toleransi_gun"])
    urun = kosu.oku_gozlemlenebilir("urunler").select(["sku_id", "kategori_kod"])
    kat_carpan = {k.kod: k.miad_toleransi_carpani for k in cfg.urun.kategoriler}

    s = (sevk.join(latent, on="eczane_id").join(urun, on="sku_id")
         .with_columns(
             (pl.col("miad_toleransi_gun")
              * pl.col("kategori_kod").replace_strict(kat_carpan, return_dtype=pl.Float64)
              ).alias("gerekli_kalan_gun")))

    tolerans_ihlali = s.filter(pl.col("kalan_raf_omru_gun") < pl.col("gerekli_kalan_gun")).height

    # FEFO replay: sevkiyatlar dosya sirasina gore islenirken, ayni SKU'da daha
    # erken miatli VE o eczane icin uygun VE bakiyesi olan bir lot var miydi?
    lot_bilgi = {
        r["lot_id"]: (r["sku_id"], r["miad_gun_indeksi"], r["adet_giris"])
        for r in lotlar.iter_rows(named=True)
    }
    sku_lotlari: dict[str, list[str]] = {}
    for lid, (sku, miad, _) in lot_bilgi.items():
        sku_lotlari.setdefault(sku, []).append(lid)
    for sku in sku_lotlari:
        sku_lotlari[sku].sort(key=lambda l: lot_bilgi[l][1])
    bakiye = {lid: v[2] for lid, v in lot_bilgi.items()}
    giris_haftasi = dict(zip(lotlar["lot_id"], lotlar["giris_haftasi"]))

    fefo_ihlali = 0
    for r in s.iter_rows(named=True):
        lid, sku, hafta = r["lot_id"], r["sku_id"], r["hafta"]
        bugun = hafta * GUN_HAFTA
        secilen_miad = lot_bilgi[lid][1]
        for aday in sku_lotlari[sku]:
            if lot_bilgi[aday][1] >= secilen_miad:
                break
            if bakiye[aday] <= 0 or giris_haftasi[aday] > hafta:
                continue
            kalan = lot_bilgi[aday][1] - bugun
            if kalan <= 0 or kalan < r["gerekli_kalan_gun"]:
                continue
            fefo_ihlali += 1
            break
        bakiye[lid] -= r["adet"]

    imha = kosu.oku_gozlemlenebilir("imhalar")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].hist(sevk["kalan_raf_omru_gun"].to_numpy(), bins=50, color="#3b6ea5")
    ax[0].set_title("Sevkiyatta kalan raf omru (gun)")
    ax[1].scatter(s["gerekli_kalan_gun"].to_numpy(), s["kalan_raf_omru_gun"].to_numpy(),
                  s=2, alpha=0.15, color="#3b6ea5")
    lim = float(s["kalan_raf_omru_gun"].max())
    ax[1].plot([0, lim], [0, lim], "r--", lw=1)
    ax[1].set_xlabel("eczanenin talep ettigi min raf omru")
    ax[1].set_ylabel("sevk edilen lotun raf omru")
    ax[1].set_title("Miad toleransi ihlal edilmiyor\n(kirmizi cizginin ustu)")
    if imha.height:
        g = imha.group_by("hafta").agg(pl.col("adet").sum()).sort("hafta")
        ax[2].bar(g["hafta"].to_numpy(), g["adet"].to_numpy(), color="#a5613b")
    ax[2].set_title("Haftalik imha edilen adet")
    ax[2].set_xlabel("hafta")
    _sekil_kaydet("fefo_ve_miad")

    k1 = Kontrol("FEFO: daha erken miatli uygun lot atlanmiyor", fefo_ihlali == 0,
                 f"{fefo_ihlali} ihlal / {s.height} sevkiyat satiri")
    k2 = Kontrol("Miad toleransi sevkiyatta ihlal edilmiyor", tolerans_ihlali == 0,
                 f"{tolerans_ihlali} ihlal / {s.height} sevkiyat satiri")
    return k1, k2


def kontrol_miad_toleransi_canli(kosu: Kosu) -> Kontrol:
    """miad_toleransi olu bir alan mi, yoksa gercekten karari degistiriyor mu?"""
    o = kosu.oku_gozlemlenebilir("siparisler")
    latent = kosu.oku_gercek("latent_eczane").select(["eczane_id", "miad_toleransi_gun"])
    toplam = o["talep_adet"].sum()
    red = o["miad_kisiti_nedeniyle_verilemeyen"].sum()
    ecz = (o.group_by("eczane_id")
           .agg((pl.col("miad_kisiti_nedeniyle_verilemeyen").sum()
                 / pl.col("talep_adet").sum()).alias("red_orani"))
           .join(latent, on="eczane_id"))
    r = np.corrcoef(ecz["miad_toleransi_gun"].to_numpy(), ecz["red_orani"].to_numpy())[0, 1]

    plt.figure(figsize=(5.5, 4))
    plt.scatter(ecz["miad_toleransi_gun"].to_numpy(), ecz["red_orani"].to_numpy(),
                s=12, alpha=0.6, color="#3b6ea5")
    plt.xlabel("latent miad_toleransi_gun")
    plt.ylabel("miad nedeniyle karsilanamama orani")
    plt.title(f"miad_toleransi canli mi? korelasyon = {r:.2f}")
    _sekil_kaydet("miad_toleransi_etkisi")

    return Kontrol(
        "Persona miad_toleransi kararı degistiriyor",
        red > 0 and r > 0.2,
        f"talebin %{100*red/toplam:.1f}'i miad kisiti yuzunden karsilanamadi; "
        f"tolerans-red korelasyonu r={r:.2f}",
    )


def kontrol_sizinti(kosu: Kosu) -> Kontrol:
    ihlaller = []
    for ad in kosu.tablolar()["observable"]:
        kolonlar = set(kosu.oku_gozlemlenebilir(ad).columns)
        for k in sorted(kolonlar & LATENT_KOLONLAR):
            ihlaller.append(f"{ad}.{k}")
    return Kontrol("Gozlemlenebilir katmanda latent kolon yok", not ihlaller,
                   f"{len(ihlaller)} ihlal" + (f": {ihlaller}" if ihlaller else ""))


def kontrol_determinizm(cfg) -> Kontrol:
    """Ayni seed -> ayni dunya. Iki kez uret, ozetleri karsilastir."""
    def _ozet():
        d = dunya_kos(cfg, SeedBankasi(cfg.profil.temel_seed))
        return (
            d.siparisler.height, int(d.siparisler["talep_adet"].sum()),
            d.sevkiyat_satirlari.height, int(d.sevkiyat_satirlari["adet"].sum()),
            int(d.hucre_haftalik["gercek_tuketim"].sum()),
            int(d.imhalar["adet"].sum()) if d.imhalar.height else 0,
            round(float(d.sow_haftalik["share_of_wallet"].sum()), 6),
        )
    a, b = _ozet(), _ozet()
    return Kontrol("Determinizm: ayni seed ayni dunya", a == b, f"{a} == {b}")


# --------------------------------------------------------------------------
def _kosu_metrikleri(k: Kosu) -> dict[str, float]:
    h = k.oku_gercek("hucre_haftalik")
    o = k.oku_gozlemlenebilir("siparisler")
    i = k.oku_gozlemlenebilir("imhalar")
    aktif = h.filter(pl.col("cesitte_var"))
    talep = float(o["talep_adet"].sum())
    karsilanan = float(o["karsilanan_adet"].sum())
    imha = float(i["adet"].sum()) if i.height else 0.0
    kayip = float(h["karsilanmayan_hasta_talebi"].sum())
    tuketim = float(h["gercek_tuketim"].sum())
    hucre = o.group_by(["eczane_id", "sku_id"]).len()["len"]
    return {
        "sifir_orani_aktif": float((aktif["gercek_tuketim"] == 0).mean()),
        "tuketim": tuketim,
        "siparis_satiri": float(o.height),
        "siparis_hucre": float(hucre.len()),
        "siparis_p90_hucre": float(hucre.quantile(0.90)),
        "karsilama_orani": karsilanan / max(talep, 1),
        "miad_reddi_orani": float(o["miad_kisiti_nedeniyle_verilemeyen"].sum()) / max(talep, 1),
        "imha_orani": imha / max(karsilanan + imha, 1),
        "iade_orani": (float(k.oku_gozlemlenebilir("iadeler")["iade_adet"].sum())
                       / max(karsilanan, 1)),
        "eczane_kayip_talep_orani": kayip / max(kayip + tuketim, 1),
    }


def knob_taramasi(profil: str, knob: str, degerler: list[str], seeds: int,
                  sabit: list[str] | None = None) -> None:
    """M1'de experiments/sweep.py YOK (o M2'ye ait, SPEC 5b.2).

    Bu, M1 knob'larinin etkisini olcmek icin karsilastirmali kosu kosucusudur:
    her deger icin `seeds` adet farkli seed, metriklerin ortalamasi ve std'si.
    """
    temel = config_yukle(profil).profil.temel_seed
    sabit_args: list[str] = []
    for s in sabit or []:
        sabit_args += ["--knob", s]
    satirlar = []
    for deger in degerler:
        yigin: list[dict[str, float]] = []
        for j in range(seeds):
            subprocess.run(
                [sys.executable, "-m", "scripts.generate_world", "--profil", profil,
                 "--kosu", "_tarama", "--knob", f"{knob}={deger}",
                 "--seed", str(temel + j)] + sabit_args,
                check=True, stdout=subprocess.DEVNULL,
            )
            yigin.append(_kosu_metrikleri(Kosu("_tarama")))
        satir: dict[str, object] = {"deger": deger}
        for ad in yigin[0]:
            v = np.array([y[ad] for y in yigin])
            satir[ad] = round(float(v.mean()), 4)
            if seeds > 1:
                satir[f"{ad}_sd"] = round(float(v.std(ddof=1)), 4)
        satirlar.append(satir)
    baslik = f"\n=== knob taramasi: {knob}  (profil={profil}, seed={seeds}"
    baslik += f", sabit: {', '.join(sabit)})" if sabit else ")"
    print(baslik + " ===")
    with pl.Config(tbl_rows=30, tbl_cols=30, tbl_width_chars=250):
        print(pl.DataFrame(satirlar))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kosu", default="full")
    ap.add_argument("--profil", default=None, help="varsayilan: kosu manifestindeki profil")
    ap.add_argument("--knob-taramasi", default=None)
    ap.add_argument("--degerler", default=None)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--sabit", action="append", default=[],
                    help="taramada sabit tutulacak knob: yol=deger, tekrarlanabilir")
    ap.add_argument("--sadece-tarama", action="store_true",
                    help="kontrolleri atla, sadece knob taramasini kostur")
    args = ap.parse_args()

    if args.knob_taramasi and args.sadece_tarama:
        if not args.degerler:
            raise SystemExit("--knob-taramasi ile --degerler verilmeli")
        knob_taramasi(args.profil or "full", args.knob_taramasi,
                      args.degerler.split(","), args.seeds, args.sabit)
        return

    kosu = Kosu(args.kosu)
    manifest = kosu.manifest_oku()
    profil = args.profil or manifest["profil"]
    cfg = config_yukle(profil)
    print(f"kosu={args.kosu} profil={profil} config_hash={manifest['config_hash']}")

    kontroller: list[Kontrol] = [kontrol_olcek(kosu, cfg)]
    kontroller.extend(kontrol_seyreklik(kosu, cfg))
    kontroller.extend(kontrol_durgunluk(kosu, cfg))
    kontroller.append(kontrol_iade_kuplaji(kosu))
    kontroller.append(kontrol_mevsimsellik(kosu, cfg))
    kontroller.append(kontrol_olay_etkisi(kosu, cfg))
    kontroller.extend(kontrol_fefo(kosu, cfg))
    kontroller.append(kontrol_miad_toleransi_canli(kosu))
    kontroller.append(kontrol_sizinti(kosu))
    kontroller.append(kontrol_determinizm(cfg))

    print("\n" + "=" * 100)
    print(f"{'DURUM':<8}{'KONTROL':<48}OLCUM")
    print("-" * 100)
    for k in kontroller:
        print(f"{'GECTI' if k.gecti else 'KALDI':<8}{k.ad:<48}{k.olcum}")
    print("=" * 100)
    kalan = [k.ad for k in kontroller if not k.gecti]
    print(f"grafikler: {SEKIL_DIZINI}")
    print(f"SONUC: {len(kontroller) - len(kalan)}/{len(kontroller)} gecti"
          + (f" | KALAN: {kalan}" if kalan else ""))

    if args.knob_taramasi:
        if not args.degerler:
            raise SystemExit("--knob-taramasi ile --degerler verilmeli")
        knob_taramasi(profil, args.knob_taramasi, args.degerler.split(","),
                      args.seeds, args.sabit)

    sys.exit(1 if kalan else 0)


if __name__ == "__main__":
    main()
