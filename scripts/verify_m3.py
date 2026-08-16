"""M3 cikis kriteri dogrulamasi.

    python -m scripts.verify_m3 --kosu full
    python -m scripts.verify_m3 --kosu full --hizli   (sizinti/determinizm atla)

SPEC M3 cikis kriteri: "Kirmizi receteli urunun hicbir kosulda oneri
listesinde cikmadigi test; soguk zincir min siparis kurali."

Kontroller iki gruba ayrilir (verify_m1 / verify_m2 ile ayni disiplin):
  DURUSTLUK : sizinti siniri, point-in-time, stok muhasebesi, determinizm.
  KRITER    : kisit degismezleri (STRES ALTINDA) ve aday havuzunun bilgi
              tasiyip tasimadigi.

STRES NEDEN SART. Bu dunyada kirmizi/yesil urun katalogun %1'inden azi ve
varsayilan ayarda kredi limiti hic baglamiyor. "Listede kirmizi cikmadi"
cumlesi bu haliyle bir sey kanitlamaz - urun zaten yok. Bu yuzden kriter
kontrolleri havuzun EN YUKSEK SKORLU urunlerini kirmizi/yesil ilan ederek,
miad baskisini tavana cikararak ve kredi tavanini daraltarak kosuluyor.
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
from eval import aday as ev_aday  # noqa: E402
from experiments.run import m3_boru_hatti, m3_ihlaller  # noqa: E402
from features.okuma import GozlemlenebilirKaynak  # noqa: E402
from policy import candidates as pol  # noqa: E402
from policy.constraints import VETO_SEBEPLERI, kisit_uygula, oneri_listesi  # noqa: E402
from scripts.verify_m2 import YASAK_ADLAR, kod_metni  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SEKIL_DIZINI = REPO_ROOT / "reports" / "figures" / "m3"
GECICI = REPO_ROOT / "experiments" / "runs" / "_dogrulama_m3"

# Cikis kriteri esikleri. Tuning knob'u DEGIL, kriterin kendisi.
# Stres kurulumunda kirmizi/yesil ilan edilen SKU sayisi: havuzun tepesinden
# bu kadari alinir; kucuk olursa veto ateslenmeyebilir ve kontrol vakuma duser.
STRES_SKU_SAYISI = 12
# Stres kredi tavani: varsayilan tavan bu dunyada hic baglamiyor.
STRES_KREDI_TAVANI = 0.02
# Stres miad baskisi: agirlik absurt derecede buyuk, esik butun lotlari
# "baski altinda" sayacak kadar genis. Amac gercekci bir ayar degil, SPEC 2.5'in
# "miad baskisi vetoyu asmaz" cumlesini en agir kosulda sinamak.
STRES_MIAD_AGIRLIGI = 1000.0
STRES_MIAD_ESIK_GUN = 100000.0
# Stres frekans tavani: budama, olculmek istenen kisitlari maskelemesin.
STRES_FREKANS_TAVANI = 50
# Aday havuzunun populerlik tabanini gecmesi beklenen asgari fark (recall).
# Sifir degil: "istatistiksel olarak ayirt edilemez kadar iyi" yeterli sayilmaz.
ESIK_TABAN_FARKI = 0.0


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
# stres kurulumu
# --------------------------------------------------------------------------
def _renklendir(dunya: pol.AdayDunyasi, sku_idler, renk: str) -> None:
    hedef = pl.col("sku_id").is_in(list(sku_idler))
    dunya.urunler = dunya.urunler.with_columns([
        pl.when(hedef).then(pl.lit(renk)).otherwise(pl.col("recete_rengi"))
          .alias("recete_rengi"),
        pl.when(hedef).then(pl.lit(False)).otherwise(pl.col("promosyon_serbest"))
          .alias("promosyon_serbest"),
    ])


def stres_kosusu(kosu_adi: str, profil: str, kok: Path) -> tuple:
    """Kisit katmanini baglayacak sekilde kurulmus tek origin.

    Uc baski ayni anda: (1) havuzun tepesi kirmizi/yesil, (2) her lot kisa
    miatli ve baski agirligi 1000, (3) kredi tavani 50'de bir.
    """
    cfg = load_config(profil, gecersiz_kilma={
        "politika.aday.miad_baskisi_agirligi": STRES_MIAD_AGIRLIGI,
        "politika.aday.miad_baskisi_esik_gun": STRES_MIAD_ESIK_GUN,
        "politika.kisit.kredi_kullanim_tavani": STRES_KREDI_TAVANI,
        "politika.kisit.eczane_haftalik_teklif_tavani": STRES_FREKANS_TAVANI,
    })
    dunya = pol.dunya_yukle(GozlemlenebilirKaynak(kosu_adi, kok=kok), cfg)
    t = pol.origin_haftalari(cfg, dunya.W)[-1]
    gor = pol.gorunum_kur(dunya, cfg, t)
    havuz, _ = pol.aday_havuzu(dunya, cfg, gor)

    tepe = (havuz.sort("skor", descending=True)["sku_id"]
            .unique(maintain_order=True).head(STRES_SKU_SAYISI).to_list())
    yari = len(tepe) // 2
    _renklendir(dunya, tepe[:yari], "KIRMIZI")
    _renklendir(dunya, tepe[yari:], "YESIL")
    return cfg, dunya, gor, kisit_uygula(dunya, cfg, gor, havuz)


# --------------------------------------------------------------------------
# DURUSTLUK kontrolleri
# --------------------------------------------------------------------------
def kontrol_statik_sizinti() -> Kontrol:
    bulgular = []
    for yol in sorted((REPO_ROOT / "policy").glob("*.py")):
        kod = kod_metni(yol)
        bulgular += [f"{yol.name}:{k}" for k in YASAK_ADLAR if k in kod]
    return Kontrol("Politika katmani ground_truth'a dokunmuyor", not bulgular,
                   f"{len(bulgular)} bulgu" + (f": {bulgular}" if bulgular else ""))


def kontrol_point_in_time(kosu_adi: str, cfg: Config) -> Kontrol:
    """Gelecek silinince ayni origin'in uretici skorlari DEGISMEMELI."""
    kaynak = GozlemlenebilirKaynak(kosu_adi)
    dunya = pol.dunya_yukle(kaynak, cfg)
    kesme = dunya.W // 2

    hedef = Run("kesilmis", kok=GECICI).prepare()
    for tablo in kaynak.tables():
        df = kaynak.tablo(tablo)
        if "hafta" in df.columns:
            df = df.filter(pl.col("hafta") <= kesme)
        hedef.write_observable(tablo, df)
    kesik = pol.dunya_yukle(GozlemlenebilirKaynak("kesilmis", kok=GECICI), cfg)

    tam = pol.uretici_skorlari(dunya, cfg, pol.gorunum_kur(dunya, cfg, kesme))
    kes = pol.uretici_skorlari(kesik, cfg, pol.gorunum_kur(kesik, cfg, kesme))
    shutil.rmtree(GECICI, ignore_errors=True)

    farklar = {u: float(np.max(np.abs(tam[u] - kes[u]))) for u in tam}
    en_kotu = max(farklar, key=farklar.get)
    return Kontrol(
        "Point-in-time: gelecek silinince aday skorlari degismiyor",
        farklar[en_kotu] == 0.0,
        f"{len(farklar)} uretici, en buyuk fark={farklar[en_kotu]:.3g} ({en_kotu})",
    )


def kontrol_stok_muhasebesi(kosu_adi: str, cfg: Config) -> Kontrol:
    """Politikanin gordugu stok, kayitli depo stoguyla ozdes olmali.

    Beklenen fark tanimli: kayit hafta SONUNDA alindigi icin gelecek haftanin
    partisini icerir, politika onu tahsis edemez. Artik fark sifir olmali ve
    politikanin gordugu stok kayitli stogu HIC ASMAMALI.
    """
    kaynak = GozlemlenebilirKaynak(kosu_adi)
    dunya = pol.dunya_yukle(kaynak, cfg)
    kayit = kaynak.tablo("depo_stok_haftalik")
    sira = {s: i for i, s in enumerate(dunya.urunler["sku_id"].to_list())}
    en_buyuk, asan = 0.0, 0
    for t in pol.origin_haftalari(cfg, dunya.W) + [dunya.W // 2]:
        gor = pol.gorunum_kur(dunya, cfg, t)
        ref = kayit.filter(pl.col("hafta") == t)
        kayitli = np.zeros(dunya.S)
        kayitli[[sira[s] for s in ref["sku_id"]]] = ref["eldeki_adet"].to_numpy()
        ileri = np.zeros(dunya.S)
        m = dunya.lot_giris == t + 1
        np.add.at(ileri, dunya.lot_s[m], dunya.lot_adet[m])
        en_buyuk = max(en_buyuk, float(np.abs((kayitli - gor.depo_stok) - ileri).max()))
        asan += int((gor.depo_stok > kayitli).sum())
    return Kontrol("Stok muhasebesi: politika olmayan mali soz vermiyor",
                   en_buyuk == 0.0 and asan == 0,
                   f"artik fark={en_buyuk:.3g}, kayitli stogu asan SKU={asan}")


def kontrol_determinizm(cfg: Config, kosu_adi: str) -> Kontrol:
    def _ozet():
        c = m3_boru_hatti(cfg, kosu_adi, DATA_DIR, oracle_hedefi=False)
        liste = oneri_listesi(c.teklifler)
        return (liste.height, round(float(liste["teklif_adedi"].sum()), 6),
                round(float(c.liste["liste_recall"].mean()), 10))
    a, b = _ozet(), _ozet()
    return Kontrol("Determinizm: ayni kosu ayni oneri listesi", a == b, f"{a} == {b}")


# --------------------------------------------------------------------------
# KRITER kontrolleri
# --------------------------------------------------------------------------
def kontrol_kirmizi_yesil(stres: tuple) -> Kontrol:
    """SPEC M3 cikis kriteri. Stres altinda: havuzun tepesi kirmizi/yesil,
    miad baskisi tavanda."""
    cfg, dunya, _, teklifler = stres
    veto = int((teklifler["veto_sebebi"].to_numpy() == "recete_rengi").sum())
    liste = oneri_listesi(teklifler).join(
        dunya.urunler.select(["sku_id", "recete_rengi"]), on="sku_id", how="left")
    sizan = int(liste.filter(
        pl.col("recete_rengi").is_in(list(cfg.politika.kisit.recete_rengi_vetosu))).height)
    return Kontrol(
        "Kirmizi/yesil recete oneri listesinde YOK (stres altinda)",
        sizan == 0 and veto > 0,
        f"{veto} satir vetolandi, listede {sizan} satir "
        f"(veto ateslenmediyse kontrol vakumdur)",
    )


def kontrol_miad_baskisi_vetoyu_asmiyor(stres: tuple) -> Kontrol:
    """SPEC 2.5: miad baskisi promosyon vetosunu asmaz."""
    cfg, dunya, gor, teklifler = stres
    baskili = gor.miad_baskisi[gor.depo_stok > 0]
    renk = teklifler.join(dunya.urunler.select(["sku_id", "recete_rengi"]),
                          on="sku_id", how="left")["recete_rengi"].to_numpy()
    vetolu_renk = np.isin(renk, list(cfg.politika.kisit.recete_rengi_vetosu))
    listede = teklifler["listede"].to_numpy()
    return Kontrol(
        "Miad baskisi promosyon vetosunu ASMIYOR",
        int((vetolu_renk & listede).sum()) == 0 and float(baskili.min()) == 1.0,
        f"baski agirligi={STRES_MIAD_AGIRLIGI:.0f}, stogu olan SKU'larda baski={baskili.min():.2f}, "
        f"kirmizi/yesil listede={int((vetolu_renk & listede).sum())}",
    )


def kontrol_soguk_zincir(stres: tuple) -> Kontrol:
    cfg, _, _, teklifler = stres
    m = cfg.politika.kisit.soguk_zincir_min_siparis_adedi
    liste = oneri_listesi(teklifler).filter(pl.col("soguk_zincir"))
    yukseltme = int(teklifler["soguk_zincir_yukseltildi"].sum())
    ihlal = int(liste.filter(pl.col("teklif_adedi") < m).height)
    return Kontrol(
        "Soguk zincir minimum siparis kurali ihlal edilmiyor",
        ihlal == 0 and liste.height > 0,
        f"listede {liste.height} soguk zincir satiri, en kucuk adet="
        f"{liste['teklif_adedi'].min() if liste.height else float('nan')} "
        f"(min {m}), {yukseltme} satir yukseltildi, ihlal={ihlal}",
    )


def kontrol_kredi(stres: tuple) -> Kontrol:
    cfg, dunya, gor, teklifler = stres
    k = cfg.politika.kisit
    dbs = dunya.eczaneler["dbs_limiti"].to_numpy().astype(float)
    risk = dunya.eczaneler["vade_riski_skoru"].to_numpy().astype(float)
    kalan = np.maximum(dbs * k.kredi_kullanim_tavani
                       * (1.0 - risk * k.vade_riski_cezasi) - gor.acik_bakiye, 0.0)
    liste = oneri_listesi(teklifler)
    yuk = np.zeros(dunya.P)
    np.add.at(yuk, liste["eczane_idx"].to_numpy(), liste["teklif_tutari"].to_numpy())
    veto = int((teklifler["veto_sebebi"].to_numpy() == "kredi_limiti").sum())
    asan = int((yuk > kalan + 1e-6).sum())
    kullanim = float(np.max(yuk / np.maximum(kalan, 1e-9)))
    return Kontrol(
        "Kredi limitini asan teklif uretilmiyor (stres altinda)",
        asan == 0 and veto > 0,
        f"{veto} satir vetolandi, limiti asan eczane={asan}, "
        f"en yuksek limit kullanimi={kullanim:.3f}",
    )


def _eczane_bootstrap(hedef: np.ndarray, a: np.ndarray, b: np.ndarray,
                      tekrar: int, seed: int) -> tuple[float, float, float]:
    """Iki havuzun recall farki icin ECZANE bazli blok bootstrap.

    Bagimsizlik birimi satir degil eczane: ayni eczanenin adaylari ortak
    gecmisten uretiliyor ve bagimsiz degil (M2'de ayni tuzak olculdu).
    """
    rng = np.random.default_rng(seed)
    P = hedef.shape[0]
    pay_a, pay_b = (hedef & a).sum(axis=1), (hedef & b).sum(axis=1)
    payda = hedef.sum(axis=1)
    temel = pay_a.sum() / payda.sum() - pay_b.sum() / payda.sum()
    farklar = np.empty(tekrar)
    for i in range(tekrar):
        idx = rng.integers(0, P, P)
        n = payda[idx].sum()
        farklar[i] = ((pay_a[idx].sum() - pay_b[idx].sum()) / n) if n else np.nan
    g = farklar[np.isfinite(farklar)]
    return float(temel), float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))


def kontrol_havuz_bilgi_tasiyor(c, cfg: Config) -> Kontrol:
    """Hibrit havuz, populerlik tabanini gecmeli - eslesmis blok bootstrap ile.

    Gecmiyorsa aday uretimi kisisellestirilmemis bir top-N listesinden ibaret
    demektir ve M3'un ekledigi bir sey yoktur.
    """
    d = cfg.tukenme.degerlendirme
    dunya = c.dunya
    farklar = []
    for t, gor in c.gorunumler.items():
        skorlar = pol.uretici_skorlari(dunya, cfg, gor)
        hibrit = pol.hibrit_skor(skorlar, cfg, gor)
        hedef = ev_aday.gozlemlenebilir_hedef(
            dunya, gor, cfg.politika.aday.degerlendirme.ufuk_hafta)
        K = cfg.politika.aday.havuz_boyutu_k
        farklar.append(_eczane_bootstrap(
            hedef.matris, ev_aday.ust_k_maskesi(hibrit, K),
            ev_aday.ust_k_maskesi(skorlar["populerlik"], K),
            d.bootstrap_orneklem, d.bootstrap_seed))
    fark = float(np.mean([f[0] for f in farklar]))
    alt = float(np.mean([f[1] for f in farklar]))
    ust = float(np.mean([f[2] for f in farklar]))
    return Kontrol(
        "Aday havuzu populerlik tabanini geciyor", alt > ESIK_TABAN_FARKI,
        f"recall farki={fark:+.3f} [%95: {alt:+.3f}, {ust:+.3f}] "
        f"({len(farklar)} origin ortalamasi)",
    )


def kontrol_cold_start(c, cfg: Config) -> Kontrol:
    """Yeni hucrede sinyal uretilebiliyor mu.

    `tekrar` ureticisi tanimi geregi sifir; CF / sepet / oznitelik komsulugu
    bu boslugu doldurmuyorsa M3'un cold start maddesi karsilanmamis demektir.
    """
    K = cfg.politika.aday.havuz_boyutu_k
    goz = c.olcum.filter((pl.col("k") == K) & (pl.col("hedef") == "gozlemlenebilir"))
    yeni = {s["uretici"]: s["yeni_hucre_recall"] for s
            in goz.group_by("uretici").agg(
                pl.col("yeni_hucre_recall").mean()).iter_rows(named=True)}
    soguk = {s["uretici"]: s["soguk_eczane_recall"] for s
             in goz.group_by("uretici").agg(
                 pl.col("soguk_eczane_recall").mean()).iter_rows(named=True)}
    return Kontrol(
        "Cold start: gecmisi olmayan hucrede aday uretiliyor",
        yeni["hibrit"] > 0 and yeni["soguk_start"] > 0 and yeni["tekrar"] == 0,
        f"yeni hucre recall: hibrit={yeni['hibrit']:.3f}, "
        f"soguk_start={yeni['soguk_start']:.3f}, cf={yeni['cf']:.3f}, "
        f"tekrar={yeni['tekrar']:.3f} | soguk eczane recall: "
        f"hibrit={soguk['hibrit']:.3f}",
    )


def kontrol_ihlaller(c, cfg: Config) -> Kontrol:
    """Varsayilan ayarda uretilen oneri listesinin butun degismezleri."""
    ihlal = m3_ihlaller(c, cfg)
    kalan = {k: v for k, v in ihlal.items() if v}
    return Kontrol("Oneri listesi butun kisit degismezlerini saglıyor",
                   not kalan, f"{len(ihlal)} degismez kontrol edildi, ihlal={kalan}")


# --------------------------------------------------------------------------
# grafikler
# --------------------------------------------------------------------------
def sekil_recall_egrisi(c, cfg: Config) -> None:
    goz = (c.olcum.filter(pl.col("hedef") == "gozlemlenebilir")
           .group_by(["uretici", "k"]).agg(pl.col("recall").mean(),
                                           pl.col("yeni_hucre_recall").mean(),
                                           pl.col("kapsama").mean()).sort("k"))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for uretici in list(cfg.politika.aday.URETICILER) + ["hibrit"]:
        alt = goz.filter(pl.col("uretici") == uretici)
        kalin = 2.2 if uretici == "hibrit" else 1.2
        for j, kolon in enumerate(("recall", "yeni_hucre_recall", "kapsama")):
            ax[j].plot(alt["k"], alt[kolon], marker="o", ms=4, lw=kalin, label=uretici)
    baslik = (f"recall@K (tum hedefler)",
              "recall@K (YALNIZCA yeni hucreler)\n= cross-sell isi",
              "katalog kapsamasi")
    for j, b in enumerate(baslik):
        ax[j].set_xlabel("K (eczane basina aday)")
        ax[j].set_title(b, fontsize=9)
        ax[j].legend(fontsize=7)
    ax[0].set_ylabel("recall")
    fig.suptitle("Aday ureticileri: hicbiri her sutunda birden iyi degil", fontsize=10)
    _sekil_kaydet("recall_egrisi")


def sekil_veto_semasi(c, cfg: Config, stres: tuple) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    oranlar = {ad: float(c.veto[f"veto_{ad}"].mean()) for ad in VETO_SEBEPLERI}
    stres_oranlari = {
        ad: float(((stres[3]["veto_maskesi"].to_numpy() >> bit) & 1).mean())
        for bit, ad in enumerate(VETO_SEBEPLERI)}
    x = np.arange(len(VETO_SEBEPLERI))
    ax[0].bar(x - 0.2, [oranlar[a] for a in VETO_SEBEPLERI], 0.4,
              label="varsayilan", color="#3b6ea5")
    ax[0].bar(x + 0.2, [stres_oranlari[a] for a in VETO_SEBEPLERI], 0.4,
              label="stres", color="#a5613b")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(VETO_SEBEPLERI, rotation=35, fontsize=7, ha="right")
    ax[0].set_ylabel("aday satirlarinin orani")
    ax[0].set_title("Hangi kisit bagliyor", fontsize=9)
    ax[0].legend(fontsize=8)

    lst = c.liste.filter(pl.col("hedef") == "gozlemlenebilir")
    asamalar = ["havuz", "veto sonrasi", "oneri listesi"]
    degerler = [float(lst["havuz_recall"].mean()),
                float(lst["veto_sonrasi_recall"].mean()),
                float(lst["liste_recall"].mean())]
    ax[1].bar(asamalar, degerler, color=["#3b6ea5", "#5a9e5a", "#a5613b"])
    for i, v in enumerate(degerler):
        ax[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax[1].set_ylabel("recall (gozlemlenebilir hedef)")
    ax[1].set_title("D6'nin bedeli: veto ve frekans tavani\nrecall'dan ne goturuyor",
                    fontsize=9)

    teklif = c.teklifler
    dilim = np.clip(
        np.argsort(np.argsort(-teklif["skor"].to_numpy())) * 10 // max(teklif.height, 1),
        0, 9)
    vetolu = teklif["vetolu"].to_numpy()
    ax[2].bar(np.arange(10), [vetolu[dilim == i].mean() for i in range(10)],
              color="#a5613b")
    ax[2].set_xlabel("skor dilimi (0 = en yuksek skorlu %10)")
    ax[2].set_ylabel("veto orani")
    ax[2].set_title("Veto skorun neresini kesiyor", fontsize=9)
    fig.suptitle("Kisit katmani: ML skorunun uzerinde veto yetkisi (D6)", fontsize=10)
    _sekil_kaydet("veto_semasi")


def sekil_soguk_start(c, cfg: Config) -> None:
    K = cfg.politika.aday.havuz_boyutu_k
    goz = (c.olcum.filter((pl.col("k") == K) & (pl.col("hedef") == "gozlemlenebilir"))
           .group_by("uretici").agg(pl.col("soguk_eczane_recall").mean(),
                                    pl.col("sicak_eczane_recall").mean(),
                                    pl.col("yeni_hucre_recall").mean()))
    adlar = list(cfg.politika.aday.URETICILER) + ["hibrit"]
    goz = goz.sort(pl.col("uretici").replace_strict(
        {a: i for i, a in enumerate(adlar)}, return_dtype=pl.Int32))
    x = np.arange(goz.height)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].bar(x - 0.2, goz["soguk_eczane_recall"], 0.4, label="soguk eczane",
              color="#3b6ea5")
    ax[0].bar(x + 0.2, goz["sicak_eczane_recall"], 0.4, label="sicak eczane",
              color="#a5613b")
    ax[0].set_ylabel("recall@K")
    ax[0].set_title("Az gecmisli eczaneler (en alt "
                    f"%{cfg.politika.aday.soguk_start.soguk_dilim * 100:.0f}) "
                    "vs digerleri", fontsize=9)
    ax[0].legend(fontsize=8)
    ax[1].bar(x, goz["yeni_hucre_recall"], 0.5, color="#5a9e5a")
    ax[1].set_ylabel("recall@K (yeni hucreler)")
    ax[1].set_title("Yeni hucre recall'u: cold start'in asil olcusu", fontsize=9)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(goz["uretici"], rotation=20, fontsize=8)
    fig.suptitle("Cold start: eczane oznitelikleri gecmisin yerini ne kadar tutuyor",
                 fontsize=10)
    _sekil_kaydet("soguk_start")


def sekil_miad_ve_veto(c, cfg: Config, stres: tuple) -> None:
    _, dunya, gor, teklifler = stres
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    havuz = c.teklifler
    baski = havuz["miad_baskisi"].to_numpy()
    kova = np.clip((baski * 5).astype(int), 0, 4)
    etiket = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    oran = [float(havuz["listede"].to_numpy()[kova == i].mean())
            if (kova == i).any() else np.nan for i in range(5)]
    ax[0].bar(etiket, oran, color="#3b6ea5")
    ax[0].set_xlabel("SKU'nun miad baskisi (kisa miatli stok orani)")
    ax[0].set_ylabel("adaylarin listeye girme orani")
    ax[0].set_title("Miad baskisi sirayi oynatir...", fontsize=9)

    renk = teklifler.join(dunya.urunler.select(["sku_id", "recete_rengi"]),
                          on="sku_id", how="left")["recete_rengi"].to_numpy()
    vetolu_renk = np.isin(renk, list(cfg.politika.kisit.recete_rengi_vetosu))
    gruplar = ["kirmizi/yesil\n(baski=1.0)", "diger\n(baski=1.0)"]
    degerler = [float(teklifler["listede"].to_numpy()[vetolu_renk].mean()),
                float(teklifler["listede"].to_numpy()[~vetolu_renk].mean())]
    ax[1].bar(gruplar, degerler, color=["#a5613b", "#5a9e5a"])
    for i, v in enumerate(degerler):
        ax[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax[1].set_ylabel("listeye girme orani")
    ax[1].set_title(f"...ama vetoyu ASMAZ (SPEC 2.5)\nbaski tavanda, agirlik {STRES_MIAD_AGIRLIGI:.0f}",
                    fontsize=9)
    fig.suptitle("Miad baskisi promosyon vetosunu asmiyor", fontsize=10)
    _sekil_kaydet("miad_ve_veto")


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
          f"m3_config_hash={cfg.hash()}")

    c = m3_boru_hatti(cfg, args.kosu, DATA_DIR)
    K = cfg.politika.aday.havuz_boyutu_k
    print(f"origin={c.originler} | aday satiri={c.teklifler.height} | "
          f"K={K} | sure={c.zaman}")

    goz = (c.olcum.filter((pl.col("k") == K) & (pl.col("hedef") == "gozlemlenebilir"))
           .group_by("uretici").agg(pl.col("recall").mean(),
                                    pl.col("yeni_hucre_recall").mean(),
                                    pl.col("soguk_eczane_recall").mean(),
                                    pl.col("precision").mean(),
                                    pl.col("kapsama").mean()))
    orc = (c.olcum.filter((pl.col("k") == K) & (pl.col("hedef") == "oracle"))
           .group_by("uretici").agg(pl.col("recall").mean().alias("oracle_recall"),
                                    pl.col("yeni_hucre_recall").mean()
                                    .alias("oracle_yeni_recall")))
    adlar = list(cfg.politika.aday.URETICILER) + ["hibrit"]
    tablo = (goz.join(orc, on="uretici", how="left")
             .sort(pl.col("uretici").replace_strict(
                 {a: i for i, a in enumerate(adlar)}, return_dtype=pl.Int32)))
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=200, float_precision=4):
        print(tablo)

    hedef_sayilari = (c.olcum.filter(pl.col("uretici") == "hibrit")
                      .group_by("hedef").agg(pl.col("hedef_sayisi").mean(),
                                             pl.col("yeni_hedef_sayisi").mean()))
    print(f"\nhedef buyuklukleri (origin ortalamasi):\n{hedef_sayilari}")
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=220, float_precision=4):
        print(f"\nveto ozeti:\n{c.veto}")

    stres = stres_kosusu(args.kosu, profil, DATA_DIR)
    sekil_recall_egrisi(c, cfg)
    sekil_veto_semasi(c, cfg, stres)
    sekil_soguk_start(c, cfg)
    sekil_miad_ve_veto(c, cfg, stres)

    kontroller = [
        kontrol_statik_sizinti(),
        kontrol_stok_muhasebesi(args.kosu, cfg),
        kontrol_kirmizi_yesil(stres),
        kontrol_soguk_zincir(stres),
        kontrol_kredi(stres),
        kontrol_miad_baskisi_vetoyu_asmiyor(stres),
        kontrol_ihlaller(c, cfg),
        kontrol_havuz_bilgi_tasiyor(c, cfg),
        kontrol_cold_start(c, cfg),
    ]
    if not args.hizli:
        kontroller.append(kontrol_point_in_time(args.kosu, cfg))
        kontroller.append(kontrol_determinizm(cfg, args.kosu))

    print("\n" + "=" * 126)
    print(f"{'DURUM':<8}{'KONTROL':<58}OLCUM")
    print("-" * 126)
    for k in kontroller:
        print(f"{'GECTI' if k.gecti else 'KALDI':<8}{k.ad:<58}{k.olcum}")
    print("=" * 126)
    kalan = [k.ad for k in kontroller if not k.gecti]
    print(f"grafikler: {SEKIL_DIZINI}")
    print(f"SONUC: {len(kontroller) - len(kalan)}/{len(kontroller)} gecti"
          + (f" | KALAN: {kalan}" if kalan else ""))
    sys.exit(1 if kalan else 0)


if __name__ == "__main__":
    main()
