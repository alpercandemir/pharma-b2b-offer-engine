"""M7 cikis kriteri dogrulamasi.

    uv run python -m scripts.verify_m7 --kosu full
    uv run python -m scripts.verify_m7 --kosu fast --hizli   (determinizm atla)

SPEC M7 cikis kriteri:
    "Eval harness'ta LLM ciktisi deterministik olarak test ediliyor --
     hallusinasyon, kisit ihlali iddiasi, sayi uydurma yakalaniyor."

Bu cumlenin DOGRU olmasi icin dort ayri seyin ayri ayri kanitlanmasi
gerekiyor ve kontroller tam olarak bu dordu sinar:

  1. LLM KARARA DOKUNMUYOR MU   D8'in kendisi. Ajan katmani karar ureten
                                hicbir fonksiyonu cagirmiyor, ground_truth
                                okumuyor. Kaynak taramasi, yorum degil.
  2. SENARYO KATMANI CANLI MI   D3/D4. Rejimler ayni tabloyu uretiyorsa
                                "kur rejimi altinda politika ne oneriyor"
                                sorusunun cevabi yok demektir. Ayrica kur
                                TAHMIN EDILMIYOR: cikti her zaman butun
                                rejimleri tasiyor.
  3. DENETCILER CANLI MI        Her mutasyon BEKLENEN tipte bulgu uretmeli.
                                Uretmiyorsa denetci olu ve harness'in
                                "her sey temiz" ciktisi anlamsiz.
  4. YANLIS ALARM YOK MU        Temiz brifing sifir bulgu vermeli. Yoksa
                                denetci gurultu uretir ve gercek bulgu
                                gorunmez olur.

M1-M6 ile ayni disiplinde iki gruba ayrilir:
  DURUSTLUK : D8 siniri, sizinti taramasi, senaryonun dunyaya dokunmamasi,
              determinizm
  KRITER    : rejim ayrismasi, kanal canliligi, denetci canliligi, yanlis
              alarm, kosullu okuma (tek rejim one cikarilmiyor)
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

from agent import narrative as nv  # noqa: E402
from agent import scenario as sc  # noqa: E402
from agent import tools as at  # noqa: E402
from core.config import Config, load_config  # noqa: E402
from core.io import DATA_DIR, Run  # noqa: E402
from experiments.run import m4_boru_hatti  # noqa: E402
from harness import denetim as dn  # noqa: E402
from harness import run as hr  # noqa: E402
from scripts.verify_m2 import kod_metni  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SEKIL_DIZINI = REPO_ROOT / "reports" / "figures" / "m7"

# --- CIKIS KRITERI ESIKLERI. Tuning knob'u DEGIL, kriterin kendisi. ---
# Sert rejimin taban rejimden ayrismasi icin gereken ASGARI kol degisimi
# (aday satirlarina oran). Altina duserse senaryo katmani dekoratiftir.
ESIK_REJIM_AYRISMASI = 0.05
# Erteleme kanalinin fiilen olcek degistirmesi icin gereken asgari TL/adet.
ESIK_ERTELEME_TL = 1e-6
# Temiz vakada kabul edilen azami bulgu. Sifir; yanlis alarm bir hata.
ESIK_YANLIS_ALARM = 0
# Taban rejimde kabul edilen azami "talep baskilayan teklif" orani. Sifir
# DEGIL ve olmamali: CATE tahmininde bazi kollarin olasiligi kontrolun
# hafifce altinda kaliyor ve notr rejimde bile birkac satir bu kola dusuyor
# (olculdu: `fast`ta %0,3). Esik, o gurultuyu artefakttan ayirmak icin var --
# taban bunun uzerine cikarsa rejimler arasi karsilastirma zeminini kaybeder.
ESIK_TABAN_BASKILAMA = 0.05
# `kontrol_dunya_dokunmazligi`da bir knob'i oynatma miktari. Kriterin
# kendisi degil, sifirdan farkli olmasi yeten bir tetik.
OYNATMA_SAPMASI = 0.1
# D8 siniri: ajan katmaninin cagirmasi YASAK olan karar fonksiyonlari ve
# modulleri. Bir tanesi bile gecerse LLM karar noktasina girmis demektir.
YASAK_KARAR_ADLARI = ("kisit_uygula", "tahsis_et", "scorer.sec", "teklif_matrisleri",
                      "senaryolari_kos", "acgozlu_tahsis", "lp_tahsisi",
                      "kredi_son_kontrolu")
YASAK_KARAR_IMPORTLARI = ("from policy", "import policy", "from models",
                          "import models", "from sim", "import sim",
                          "from eval", "import eval")
# Gercek tepkiyi / oracle'i okuma yasagi (verify_m4/m5/m6 ile ayni liste).
YASAK_TEPKI_ADLARI = ("sim.response", "sim/response", "tepki_hesapla",
                      "TepkiEvreni", "GercekDurum", "ground_truth", "oracle")
# D8 siniri bu iki dosyada zorlanir: araclar ve anlati.
AJAN_SALT_OKUR = ("agent/tools.py", "agent/narrative.py")


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
# DURUSTLUK
# --------------------------------------------------------------------------
def kontrol_d8_siniri() -> Kontrol:
    """D8: LLM karar noktasinda YOK. Kaynak taramasiyla.

    `agent/tools.py` ve `agent/narrative.py` policy/models/sim/eval
    modullerinden hicbirini import etmez ve karar ureten hicbir fonksiyonu
    cagirmaz. Ederlerse model -- ya da modelin cagirdigi bir arac -- teklif
    listesini degistirebilir hale gelirdi ve D8 ihlal edilmis olurdu.

    Kontrol yorum satirina degil KAYNAGA bagli; iddia kod incelemesiyle
    degil kosuyla dogrulanir.
    """
    bulunan = []
    for dosya in AJAN_SALT_OKUR:
        metin = kod_metni(REPO_ROOT / dosya)
        bulunan += [f"{dosya}:{ad}" for ad in YASAK_KARAR_ADLARI if ad in metin]
        bulunan += [f"{dosya}:{ad}" for ad in YASAK_KARAR_IMPORTLARI if ad in metin]
    return Kontrol(
        ad="D8: ajan katmani karar fonksiyonu cagirmiyor/import etmiyor",
        gecti=not bulunan,
        olcum=f"{len(AJAN_SALT_OKUR)} dosya x "
              f"{len(YASAK_KARAR_ADLARI) + len(YASAK_KARAR_IMPORTLARI)} yasak ad "
              f"tarandi, bulunamadi" if not bulunan else f"BULUNAN: {bulunan}",
    )


def kontrol_ajan_sizintisi() -> Kontrol:
    """Ajan ve harness katmani gercek tepkiyi / oracle'i okumuyor.

    Brifingin butun iddiasi "gozlemlenebilir olgulardan anlatim" uzerine
    kurulu. Bir arac ground_truth'a bir kez bakarsa hem D8 hem M1'in
    gozlemlenebilirlik siniri coker.
    """
    dosyalar = list(AJAN_SALT_OKUR) + ["agent/client.py", "harness/denetim.py",
                                       "harness/mutasyon.py"]
    bulunan = []
    for dosya in dosyalar:
        metin = kod_metni(REPO_ROOT / dosya)
        bulunan += [f"{dosya}:{ad}" for ad in YASAK_TEPKI_ADLARI if ad in metin]
    return Kontrol(
        ad="ajan/harness katmani gercek tepkiyi/oracle'i okumuyor",
        gecti=not bulunan,
        olcum=f"{len(dosyalar)} dosya tarandi, yasak ad bulunamadi"
        if not bulunan else f"BULUNAN: {bulunan}",
    )


def kontrol_dunya_degismedi(cfg: Config, manifest: dict) -> Kontrol:
    """M7 knob'lari dunyayi DEGISTIRMEZ (D3).

    Rejim, kosullu bir OKUMADIR: ayni dunya uzerinde farkli bir soru sorar.
    Dunyayi degistirseydi rejimler arasi fark politika farki degil DUNYA
    farki olurdu ve karsilastirma anlamsiz kalirdi. M5'in `kit_stok_carpani`
    senaryosuyla ayni disiplin.

    Olcu manifeste BAGLI DEGIL: M7 knob'lari fiilen oynatilir ve
    `dunya_hash`in sabit, tam config hash'inin degismis oldugu gosterilir.
    Manifeste bakan bir kontrol, `dunya_hash` alanini tasimayan eski
    kosularda (M1 doneminde uretilmis `full` dunyasi) olculemez hale
    gelirdi -- olculemeyen kontrol gecmis sayilamaz.
    """
    # Oynatma miktarlari ONEMSIZ: sorulan sey "bu knob dunyaya dokunuyor
    # mu", "ne kadar dokunuyor" degil. Her uc knob'in da sifirdan farkli
    # bicimde degismesi yeterli, o yuzden config'e cikarilmadi.
    oynatilmis = load_config(cfg.profil.ad, gecersiz_kilma={
        "senaryo.ikame_ufku_hafta": cfg.senaryo.ikame_ufku_hafta * 2,
        "ajan.brifing_teklif_sayisi": cfg.ajan.brifing_teklif_sayisi + 1,
        "harness.mutasyon_sapmasi": cfg.harness.mutasyon_sapmasi + OYNATMA_SAPMASI,
    })
    dunya_ayni = cfg.dunya_hash() == oynatilmis.dunya_hash()
    config_degisti = cfg.hash() != oynatilmis.hash()
    manifest_hash = manifest.get("dunya_hash")
    manifest_ayni = manifest_hash is None or manifest_hash == cfg.dunya_hash()
    return Kontrol(
        ad="M7 knob'lari dunyayi degistirmiyor (dunya_hash sabit)",
        gecti=dunya_ayni and config_degisti and manifest_ayni,
        olcum=f"senaryo/ajan/harness knob'lari oynatildi -> dunya_hash "
              f"{'sabit' if dunya_ayni else 'DEGISTI'} ({cfg.dunya_hash()}), "
              f"config hash {'degisti' if config_degisti else 'DEGISMEDI'} | "
              f"kosu manifesti: {manifest_hash or 'alan yok (M1 donemi kosusu)'}",
    )


def kontrol_determinizm(cfg: Config, kosu_adi: str, m4) -> Kontrol:
    """Ayni config + ayni seed -> ayni senaryo tablosu ve ayni brifing metni.

    LLM'in kendisi deterministik olmasa da M7'nin OLCULEN kismi olmak
    zorunda: senaryo katmani tamamen deterministik, brifing ise kayittan
    oynatiliyor (agent/client.py).
    """
    a = sc.duz_metrikler(sc.senaryolari_kos(cfg, m4))
    kosu_b = sc.senaryolari_kos(cfg, m4)
    b = sc.duz_metrikler(kosu_b)
    farkli = [k for k in a if a[k] != b.get(k)]
    baglam = at.baglam_kur(cfg, kosu_b)
    ecz = baglam.teklif_veren_eczaneler()[0]
    m1 = nv.brifing_uret(cfg, baglam, ecz, istemci=None).metin
    m2 = nv.brifing_uret(cfg, baglam, ecz, istemci=None).metin
    return Kontrol(
        ad="determinizm: iki senaryo kosusu + iki brifing ayni",
        gecti=not farkli and m1 == m2,
        olcum=f"{len(a)} metrik karsilastirildi, {len(farkli)} fark | brifing "
              f"metni {'ayni' if m1 == m2 else 'FARKLI'} ({len(m1)} karakter)"
              + (f" ({farkli[:3]})" if farkli else ""),
    )


# --------------------------------------------------------------------------
# KRITER: senaryo katmani
# --------------------------------------------------------------------------
def kontrol_rejim_ayrismasi(kosu: sc.SenaryoKosusu, cfg: Config) -> Kontrol:
    """Rejimler FIILEN farkli tablo uretiyor mu.

    Config yuklemesindeki notrluk kilidi (`_m7_senaryo_kilidi`) yalnizca
    TANIMIN olu olmadigini garanti eder: parametreler farkli. Bu kontrol bir
    adim otesini sinar -- parametre farki SONUCA yansiyor mu. Yansimiyorsa
    senaryo katmani dekoratiftir ve raporda oyle yazilmalidir.
    """
    taban = kosu.ozetler[kosu.taban_ad]
    oranlar = {ad: f.kol_degisen_satir / max(kosu.ozetler[ad].aday_satiri, 1)
               for ad, f in kosu.farklar.items()}
    en_buyuk = max(oranlar.values(), default=0.0)
    kazanan = max(oranlar, key=oranlar.get) if oranlar else "-"
    return Kontrol(
        ad=f"rejim ayrismasi: taban ({kosu.taban_ad}) ile sert rejim farkli karar veriyor",
        gecti=en_buyuk > ESIK_REJIM_AYRISMASI,
        olcum=f"azami kol degisim orani %{en_buyuk * 100:.1f} ({kazanan}) "
              f"(esik %{ESIK_REJIM_AYRISMASI * 100:.0f}) | teklif "
              f"{taban.teklif_sayisi} -> "
              + ", ".join(f"{ad}: {o.teklif_sayisi}" for ad, o in kosu.ozetler.items()
                          if ad != kosu.taban_ad),
    )


def kontrol_taban_notr(kosu: sc.SenaryoKosusu) -> Kontrol:
    """Taban rejimde erteleme kalemi TAM SIFIR olmali.

    Taban rejim bir mudahale degil, olcum sifiridir. Kalem sifirdan farkli
    olsaydi butun "tabana gore fark" sutunlari iki mudahalenin farkini
    olcerdi ve rapor yanlis okunurdu.
    """
    o = kosu.ozetler[kosu.taban_ad]
    s = kosu.sonuclar[kosu.taban_ad]
    azami = float(np.max(np.abs(s.erteleme))) if s.erteleme.size else 0.0
    return Kontrol(
        ad=f"taban rejim ({kosu.taban_ad}) notr: erteleme kalemi tam sifir",
        gecti=azami == 0.0,
        olcum=f"azami |erteleme| = {azami:.3e} TL/adet, ortalama "
              f"{o.ortalama_erteleme_tl:.3e}",
    )


def kontrol_erteleme_kanali(kosu: sc.SenaryoKosusu) -> Kontrol:
    """Erteleme kazanci kanali sert rejimde fiilen calisiyor mu.

    Sifirsa `senaryo.ikame_ufku_hafta` ve `fiyat_gecis_katsayisi` birer
    kadran degil sussuz sayilardir; D4'un "asil sinyal guncelleme
    beklentisi" iddiasi da olculemez.
    """
    kalemler = {ad: o.ortalama_erteleme_tl for ad, o in kosu.ozetler.items()
                if ad != kosu.taban_ad}
    en_buyuk = max(kalemler.values(), default=0.0)
    return Kontrol(
        ad="erteleme kazanci kanali canli (sert rejimde > 0)",
        gecti=en_buyuk > ESIK_ERTELEME_TL,
        olcum=" | ".join(f"{ad}: {v:.2f} TL/adet" for ad, v in kalemler.items()),
    )


def kontrol_bekleme_kapisi(kosu: sc.SenaryoKosusu) -> Kontrol:
    """D9 kesismesi: erteleme kapisi HEDEFLEME yapiyor mu.

    Kapi butun satirlara ayni uygulaniyorsa (bekleyemeyen pay = 0 ya da 1)
    rejim yalnizca SEVIYE kaydirir. Aradaysa rejim satirlar arasinda AYRIM
    yapiyor demektir: raf omru guncellemeyi tasimayan lot bekleyemez ve o
    satir sert rejimde de listede kalir. Kontrol GEVSEK -- kapinin
    baglamasi dunyanin lot yasi dagilimina bagli ve bu bir olcum, bir
    tuning hedefi degil.
    """
    paylar = {ad: o.bekleyemeyen_pay for ad, o in kosu.ozetler.items()}
    ayrim = [ad for ad, p in paylar.items() if 0.0 < p < 1.0]
    return Kontrol(
        ad="erteleme kapisi ayrim yapiyor (bekleyemeyen pay 0 ile 1 arasinda)",
        gecti=bool(ayrim),
        olcum=" | ".join(f"{ad}: aday %{p * 100:.1f}, teklif "
                         f"%{kosu.ozetler[ad].bekleyemeyen_teklif_pay * 100:.1f}"
                         for ad, p in paylar.items()),
    )


def kontrol_isaret_donmesi(kosu: sc.SenaryoKosusu) -> Kontrol:
    """Artimsal marj metrigi hangi rejimde OKUNAMAZ hale geliyor.

    Erteleme kalemi buyudukce "teklif yok" kolunun senaryo marji negatife
    doner. Amac fonksiyonu p*marj oldugu icin o bolgede en iyi kol, kabul
    olasiligi EN DUSUK olan kol olabilir: politika fiilen "satmamayi"
    optimize eder ve `beklenen_artimsal_marj` YUKSELIR. Aksiyon uzayinda
    (D1) talebi baskilayan bir kol YOKTUR; bu bir modelleme artefaktidir.

    Kontrol artefakti YASAKLAMIYOR -- yasaklamak M4'un amac fonksiyonunu
    degistirmek olurdu ve M7'nin kapsami disinda (CLAUDE.md 1). Iki sey
    yapiyor: (a) artefakti her kosuda GORUNUR kiliyor, (b) TABAN rejimin
    temiz kaldigini sart kosuyor. (b) olmasaydi karsilastirmanin zemini de
    bulasik olurdu ve "tabana gore fark" sutunlari okunamazdi.
    """
    paylar = {ad: o.talep_baskilayan_teklif_orani for ad, o in kosu.ozetler.items()}
    taban_pay = paylar[kosu.taban_ad]
    bulasik = [ad for ad, p in paylar.items() if p > ESIK_TABAN_BASKILAMA]
    return Kontrol(
        ad="isaret donmesi teshisi canli; taban rejim temiz",
        gecti=taban_pay <= ESIK_TABAN_BASKILAMA,
        olcum=f"taban %{taban_pay * 100:.1f} (esik %{ESIK_TABAN_BASKILAMA * 100:.0f}) | "
        + " | ".join(
            f"{ad}: baskilayan %{p * 100:.1f}, negatif taban marj "
            f"%{kosu.ozetler[ad].negatif_taban_marj_orani * 100:.1f}"
            for ad, p in paylar.items())
        + (f" | ARTIMSAL MARJ TEK BASINA OKUNAMAZ: {bulasik}" if bulasik else ""),
    )


def kontrol_kosullu_okuma(kosu: sc.SenaryoKosusu, cfg: Config,
                          metin: str) -> Kontrol:
    """D3: cikti tek bir rejim degil, HEPSI.

    Iki yerde birden sinanir: senaryo katmani butun rejimleri donduruyor mu,
    ve brifing metni her rejim icin bir bolum tasiyor mu. Biri eksikse
    "kur tahmin edilmiyor, senaryolastiriliyor" cumlesi bos kalir.
    """
    beklenen = [r.ad for r in cfg.senaryo.rejimler]
    eksik_kosu = [ad for ad in beklenen if ad not in kosu.sonuclar]
    eksik_metin = [ad for ad in beklenen
                   if f"{nv.REJIM_BASLIGI}{ad}" not in metin]
    return Kontrol(
        ad="D3: hem senaryo ciktisi hem brifing BUTUN rejimleri tasiyor",
        gecti=not eksik_kosu and not eksik_metin,
        olcum=f"{len(beklenen)} rejim: kosuda eksik={eksik_kosu or 'yok'}, "
              f"brifingde eksik={eksik_metin or 'yok'}",
    )


# --------------------------------------------------------------------------
# KRITER: harness
# --------------------------------------------------------------------------
def kontrol_yanlis_alarm(sonuc: hr.HarnessSonucu) -> Kontrol:
    """Temiz brifing sifir bulgu veriyor mu.

    Gurultulu bir denetci gercek bulguyu gorunmez kilar; "yakaliyor" kadar
    "bos yere bagirmiyor" da cikis kriterinin parcasi.
    """
    temiz = [s for s in sonuc.sonuclar if s.vaka.tip == "temiz"]
    toplam = sum(len(s.bulgular) for s in temiz)
    return Kontrol(
        ad="yanlis alarm yok: temiz vakalar sifir bulgu",
        gecti=toplam <= ESIK_YANLIS_ALARM and bool(temiz),
        olcum=f"{len(temiz)} temiz vaka, toplam {toplam} bulgu "
              f"(esik {ESIK_YANLIS_ALARM})" + (
                  "" if toplam == 0 else
                  f" | ornek: {temiz[0].bulgular[0] if temiz[0].bulgular else ''}"),
    )


def kontrol_denetci_canli(sonuc: hr.HarnessSonucu) -> Kontrol:
    """Her mutant BEKLENEN tipte bulgu uretti mi.

    M7'nin en onemli kontrolu. Denetciler hicbir seyi yakalamasaydi butun
    temiz vakalar yine gecerdi ve harness "her sey yolunda" derdi -- M5'in
    "rejim olu", M6'nin "teshis olu" tuzagi.
    """
    mutant = [s for s in sonuc.sonuclar if s.vaka.tip == "mutant"]
    kalan = [s.vaka.ad for s in mutant if not s.gecti]
    tipler = sorted({t for s in mutant for t in s.vaka.beklenen})
    return Kontrol(
        ad="denetciler canli: her mutasyon beklenen tipte bulgu uretiyor",
        gecti=bool(mutant) and not kalan,
        olcum=f"{len(mutant) - len(kalan)}/{len(mutant)} mutant yakalandi | "
              f"kapsanan tipler: {tipler}"
              + (f" | KALAN: {kalan}" if kalan else ""),
    )


def kontrol_tip_kapsamasi(sonuc: hr.HarnessSonucu) -> Kontrol:
    """Cikis kriterinin saydigi UC belirtinin de bir mutanti var mi.

    SPEC "hallusinasyon, kisit ihlali iddiasi, sayi uydurma" diyor; senaryo
    karistirma da milestone kapsaminda ayrica istendi. Dordunun de kanitli
    olmasi gerekiyor -- kanitsiz kalan tip icin "yakalaniyor" denemez.
    """
    zorunlu = {"hallusinasyon", "kisit_ihlali", "sayi_uydurma", "senaryo_karismasi"}
    kanitli = {t for s in sonuc.sonuclar if s.vaka.tip == "mutant" and s.gecti
               for t in s.vaka.beklenen}
    eksik = sorted(zorunlu - kanitli)
    return Kontrol(
        ad="cikis kriterinin dort belirtisi de kanitli",
        gecti=not eksik,
        olcum=f"kanitli tipler: {sorted(kanitli)}"
              + (f" | EKSIK: {eksik}" if eksik else ""),
    )


def kontrol_kayit_yolu(sonuc: hr.HarnessSonucu) -> Kontrol:
    """Kayitli konusma oynatma yolu fiilen kosuldu mu.

    Determinizm cozumunun kendisi bu yol (agent/client.py): arac dongusu
    calisir, arac sonuclari YENIDEN hesaplanir, defter araclardan kurulur.
    Kosulmadiysa "LLM ciktisi deterministik test ediliyor" cumlesinin
    tasiyicisi sinanmamis demektir.
    """
    kayitli = [s for s in sonuc.sonuclar if s.vaka.kaynak == "kayitli"]
    gecen = [s for s in kayitli if s.gecti and not s.atlandi]
    return Kontrol(
        ad="kayitli konusma oynatildi (determinizm tasiyicisi)",
        gecti=bool(gecen),
        olcum=f"{len(gecen)}/{len(kayitli)} kayitli vaka kosuldu"
              + ("" if gecen else " | kayit yok: --kayit-uret ile uretin"),
    )


# --------------------------------------------------------------------------
# grafikler
# --------------------------------------------------------------------------
def sekil_rejim_tablosu(kosu: sc.SenaryoKosusu) -> Path:
    """Rejim x metrik: senaryo katmaninin tek bakista okunusu."""
    adlar = list(kosu.ozetler)
    x = np.arange(len(adlar))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    ax[0].bar(x, [kosu.ozetler[a].teklif_sayisi for a in adlar])
    ax[0].set_ylabel("teklif sayisi")
    ax[0].set_title("Hacim", fontsize=10)
    ikiz = ax[0].twinx()
    ikiz.plot(x, [kosu.ozetler[a].teklif_adedi for a in adlar], "ko--", ms=5)
    ikiz.set_ylabel("teklif adedi (nokta)")

    ax[1].bar(x, [kosu.ozetler[a].beklenen_artimsal_marj for a in adlar])
    ax[1].axhline(0, c="k", lw=1)
    ax[1].set_ylabel("beklenen artimsal marj (TL)")
    ax[1].set_title("Deger", fontsize=10)

    ax[2].bar(x - 0.2, [kosu.ozetler[a].ortalama_mf for a in adlar], 0.4,
              label="ortalama MF")
    ikiz2 = ax[2].twinx()
    ikiz2.bar(x + 0.2, [kosu.ozetler[a].ortalama_vade for a in adlar], 0.4,
              color="tab:orange", label="ortalama vade")
    ax[2].set_ylabel("MF orani")
    ikiz2.set_ylabel("vade (gun)")
    ax[2].set_title("Aksiyon karmasi (D1'in iki ekseni)", fontsize=10)
    for a in (ax[0], ax[1], ax[2]):
        a.set_xticks(x)
        a.set_xticklabels(adlar)
    fig.suptitle("Kur rejimi altinda politikanin onerisi (D3: hepsi kosullu)",
                 fontsize=10)
    return _sekil_kaydet("rejim_tablosu")


def sekil_erteleme_kapisi(kosu: sc.SenaryoKosusu, cfg: Config) -> Path:
    """Erteleme kaleminin lot raf omrune gore dagilimi: D3 x D9 kesismesi."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for ad, s in kosu.sonuclar.items():
        if s.aday.height == 0:
            continue
        kalan = s.aday["lot_kalan_gun"].to_numpy()
        ax[0].scatter(kalan, s.erteleme, s=8, alpha=0.4, label=ad)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("lotun kalan raf omru (gun)")
    ax[0].set_ylabel("erteleme kazanci (TL/adet)")
    ax[0].set_title("Bekleyemeyen lotta kalem SIFIR (D9 kesismesi)", fontsize=10)
    ax[0].legend(fontsize=8)

    adlar = list(kosu.ozetler)
    x = np.arange(len(adlar))
    ax[1].bar(x - 0.2, [kosu.ozetler[a].bekleyemeyen_pay for a in adlar], 0.4,
              label="aday satirlari")
    ax[1].bar(x + 0.2, [kosu.ozetler[a].bekleyemeyen_teklif_pay for a in adlar], 0.4,
              label="teklif verilen satirlar")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(adlar)
    ax[1].set_ylabel("bekleyemeyen satir orani")
    ax[1].set_title("Kapi kimi bagliyor", fontsize=10)
    ax[1].legend(fontsize=8)
    return _sekil_kaydet("erteleme_kapisi")


def sekil_harness(sonuc: hr.HarnessSonucu) -> Path:
    """Vaka x bulgu tipi matrisi: denetcinin canliligi tek bakista."""
    vakalar = [s.vaka.ad for s in sonuc.sonuclar]
    tipler = list(dn.BULGU_TIPLERI)
    matris = np.array([[dn.tip_sayimi(s.bulgular).get(t, 0) for t in tipler]
                       for s in sonuc.sonuclar], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(vakalar) + 2.2))
    ax.imshow(matris, aspect="auto", cmap="Oranges")
    ax.set_xticks(np.arange(len(tipler)))
    ax.set_xticklabels(tipler, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(vakalar)))
    ax.set_yticklabels(vakalar, fontsize=8)
    for i in range(matris.shape[0]):
        for j in range(matris.shape[1]):
            if matris[i, j]:
                ax.text(j, i, int(matris[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("Vaka x bulgu tipi: temiz satirlar BOS, mutant satirlar DOLU",
                 fontsize=10)
    return _sekil_kaydet("harness_matrisi")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kosu", default="full", help="data/<kosu> altindaki dunya")
    ap.add_argument("--profil", default=None,
                    help="varsayilan: kosu manifestindeki profil")
    ap.add_argument("--hizli", action="store_true", help="determinizm kontrolunu atla")
    args = ap.parse_args()

    manifest = Run(args.kosu).read_manifest()
    profil = args.profil or manifest["profil"]
    cfg = load_config(profil)
    print(f"kosu={args.kosu} profil={profil} "
          f"dunya_config_hash={manifest['config_hash']} m7_config_hash={cfg.hash()}")

    m4 = m4_boru_hatti(cfg, args.kosu, DATA_DIR)
    print(f"M4 yeniden kullanildi: olcum origin={m4.olcum_originleri}")

    hb = hr.baglam_hazirla(cfg, args.kosu, DATA_DIR, m4=m4)
    kosu = hb.kosu
    print(f"\nM7 senaryo: origin={kosu.t} politika={kosu.politika} "
          f"taban={kosu.taban_ad} rejimler={kosu.rejim_adlari}")

    with pl.Config(tbl_rows=10, tbl_cols=14, tbl_width_chars=210, float_precision=3):
        print("\nREJIM TABLOSU (D3: hepsi kosullu okuma, hicbiri tahmin degil):")
        print(sc.ozet_tablosu(kosu))
        print("\nTABANA GORE FARK:")
        print(sc.fark_tablosu(kosu))

    ecz = hb.adaylar[0]
    brifing_metni = nv.brifing_uret(cfg, hb.baglam, ecz, istemci=None).metin
    sonuc = hr.harness_kos(hb, hr.vakalari_yukle())

    print(f"\nHARNESS: {sonuc.gecen}/{len(sonuc.sonuclar)} vaka "
          f"({sonuc.sure_sn} sn)")
    with pl.Config(tbl_rows=30, tbl_cols=8, tbl_width_chars=190):
        print(pl.DataFrame([
            {"vaka": s.vaka.ad, "tip": s.vaka.tip, "eczane": s.eczane_id,
             "gecti": s.gecti, "olcum": s.olcum[:70]} for s in sonuc.sonuclar]))

    sekiller = [
        sekil_rejim_tablosu(kosu),
        sekil_erteleme_kapisi(kosu, cfg),
        sekil_harness(sonuc),
    ]

    kontroller = [
        kontrol_d8_siniri(),
        kontrol_ajan_sizintisi(),
        kontrol_dunya_degismedi(cfg, manifest),
        kontrol_rejim_ayrismasi(kosu, cfg),
        kontrol_taban_notr(kosu),
        kontrol_erteleme_kanali(kosu),
        kontrol_bekleme_kapisi(kosu),
        kontrol_isaret_donmesi(kosu),
        kontrol_kosullu_okuma(kosu, cfg, brifing_metni),
        kontrol_yanlis_alarm(sonuc),
        kontrol_denetci_canli(sonuc),
        kontrol_tip_kapsamasi(sonuc),
        kontrol_kayit_yolu(sonuc),
    ]
    if not args.hizli:
        kontroller.append(kontrol_determinizm(cfg, args.kosu, m4))

    print("\n" + "=" * 140)
    print(f"{'DURUM':<8}{'KONTROL':<66}OLCUM")
    print("-" * 140)
    for k in kontroller:
        print(f"{'GECTI' if k.gecti else 'KALDI':<8}{k.ad:<66}{k.olcum}")
    print("=" * 140)
    kalan = [k.ad for k in kontroller if not k.gecti]
    print(f"grafikler: {SEKIL_DIZINI} ({len(sekiller)} dosya)")
    print(f"SONUC: {len(kontroller) - len(kalan)}/{len(kontroller)} gecti"
          + (f" | KALAN: {kalan}" if kalan else ""))
    sys.exit(1 if kalan else 0)


if __name__ == "__main__":
    main()
