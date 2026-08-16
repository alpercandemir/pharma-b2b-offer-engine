"""M6 cikis kriteri dogrulamasi.

    uv run python -m scripts.verify_m6 --kosu full
    uv run python -m scripts.verify_m6 --kosu fast --hizli   (determinizm atla)

SPEC M6 cikis kriteri:
    "Offline tahmin +%12 dedi, gercek -%3 cikti, neden?" sorusunun
    cevaplanabildigi bir rapor. Varyans, ortusme ihlali, extrapolation.

Bir raporun bu soruyu CEVAPLAYABILMESI icin sirasiyla su dordu gerekir ve
kontroller bu dordu ayri ayri sinar:

  1. OLCEK DOGRU MU        Offline tahminin ve oracle'in ayni buyuklugu
                           olctugu kanitlanmali. Ozdeslik testi: hedef =
                           kayit politikasi alindiginda IPS kayit
                           politikasinin GERCEK degerini bulmali. Bulamiyorsa
                           butun tablo cope gider ve "neden" sorusu sorulamaz.
  2. SAPMA AYRISTIRMASI    Kalemler (varyans / kirpma / propensity) toplami
                           TAM olarak toplam sapmaya esit olmali. Artik
                           sifirdan farkliysa ayristirma yalan soyluyordur.
  3. TESHISLER CANLI MI    Kirpma fiilen kirpiyor mu, ortusme teshisi fiilen
                           isaretliyor mu. Sifirsa kadran dekoratiftir.
  4. IKI SAYI AYRISIYOR MU Offline ile online sistematik olarak ayrisiyor mu.
                           AYRISMIYORSA milestone bir sey ogretmiyor demektir
                           (CLAUDE.md: "offline ile online supheli derecede
                           uyusuyorsa exploration fazla genis").

Kontroller M1-M5 ile ayni disiplinde iki gruba ayrilir:
  DURUSTLUK : sizinti siniri, ozdeslik, ayristirma artigi, isinma sadakati,
              determinizm
  KRITER    : tahminci sapmasi, kirpma/ortusme canliligi, propensity
              bozmasinin yonu, ufka gore ayrisma, kanibalizm kanali
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

from core.config import Config, config_yukle  # noqa: E402
from core.io import VERI_DIZINI, Kosu  # noqa: E402
from core.rng import SeedBankasi  # noqa: E402
from eval import ope as ev_ope  # noqa: E402
from eval import report as ev_rapor  # noqa: E402
from experiments.run import (m4_boru_hatti, m6_boru_hatti,  # noqa: E402
                             m6_duz_metrikler)
from scripts.verify_m2 import kod_metni  # noqa: E402
from sim.world import dunya_kur, hafta_adimi  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
SEKIL_DIZINI = KOK / "reports" / "figures" / "m6"

# --- CIKIS KRITERI ESIKLERI. Tuning knob'u DEGIL, kriterin kendisi. ---
# Ozdeslik testi: hedef = kayit politikasi iken IPS ile oracle arasindaki
# goreli fark. Sifir OLAMAZ (kabul zarlari sonlu orneklem), ama buyukse
# odul olcegi ya da propensity loglamasi bozuktur.
ESIK_OZDESLIK_YUZDE = 5.0
# Sapma ayristirmasinin artigi. Cebirsel bir ozdeslik oldugu icin tolerans
# yalnizca kayan nokta payidir.
TOLERANS_AYRISTIRMA = 1e-9
# Kirpmanin fiilen sildigi agirlik kutlesi. Sifirsa `kirpma_esigi` bir kadran
# degil, sussuz bir sayidir.
ESIK_KIRPMA_KUTLESI = 1e-6
# Ortusme teshisinin isaretledigi asgari satir sayisi (butun politikalar
# toplaminda). Sifirsa teshis olu.
ESIK_ORTUSME_IHLALI = 1
# Offline ile online arasinda beklenen ASGARI ayrisma (ufuk kaleminin taban
# net marja orani). Altina duserse iki sayi "supheli derecede uyusuyor"
# demektir ve CLAUDE.md geregi exploration daraltilip yeniden kosulmalidir.
ESIK_UFUK_AYRISMASI = 0.05
# Isinma sadakati: rollout'un dallanma anindaki durumu taban dunyayla
# BIREBIR ayni olmali (tamsayi karsilastirmasi, tolerans yok).
# Propensity bozmasi kontrolunde kullanilan sicaklik.
BOZMA_SICAKLIGI = 2.0
# OPE katmani gercek tepkiyi goremez (verify_m4/m5 ile ayni liste).
YASAK_TEPKI_ADLARI = ("sim.response", "sim/response", "tepki_hesapla",
                      "TepkiEvreni", "GercekDurum", "ground_truth", "oracle")


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
def kontrol_ope_sizintisi() -> Kontrol:
    """`eval/ope.py` gercek tepkiyi ya da oracle'i goremez.

    M6'nin butun iddiasi "loglardan, gercegi bilmeden tahmin" uzerine kurulu.
    Tahminci dosyasi oracle'a bir kez bakarsa iddia coker; bu yuzden kontrol
    yorum satirina degil KAYNAK TARAMASINA bagli.
    """
    metin = kod_metni(KOK / "eval" / "ope.py")
    bulunan = [a for a in YASAK_TEPKI_ADLARI if a in metin]
    return Kontrol(
        ad="eval/ope.py gercek tepkiyi/oracle'i okumuyor",
        gecti=not bulunan,
        olcum=f"yasak ad bulunamadi ({len(YASAK_TEPKI_ADLARI)} ad tarandi)"
        if not bulunan else f"BULUNAN: {bulunan}",
    )


def kontrol_ozdeslik(c) -> Kontrol:
    """Hedef = kayit politikasi iken IPS, kayit politikasinin gercek degeri.

    OPE'nin en temel ozdesligi ve butun M6 tablosunun on kosulu: iki sayi
    ayni buyuklugu olcmuyorsa "offline +%X dedi" cumlesi anlamsizdir.
    """
    o = c.ozdeslik
    sapma = o["ozdeslik_sapma_yuzde"]
    return Kontrol(
        ad="ozdeslik: hedef=kayit politikasi -> IPS = oracle",
        gecti=bool(np.isfinite(sapma) and sapma < ESIK_OZDESLIK_YUZDE),
        olcum=f"gozlenen {o['kayit_politikasi_gozlenen']:.4f} vs oracle "
              f"{o['kayit_politikasi_oracle']:.4f} TL/satir "
              f"= %{sapma:.2f} (esik %{ESIK_OZDESLIK_YUZDE})",
    )


def kontrol_ayristirma_artigi(c) -> Kontrol:
    """Kalemler toplami TAM olarak toplam sapmaya esit mi.

    Ayristirma bir ablasyon merdivenidir; her basamak bir yaklasikligi geri
    alir ve teleskopik toplam farka esittir. Artik sifirdan farkliysa
    merdivenin bir basamagi yanlis kurulmustur ve raporun butun "sebep"
    yorumu dayanaksiz kalir.
    """
    en_buyuk = max((abs(a.artik) for a in c.ayristirmalar.values()), default=0.0)
    return Kontrol(
        ad="sapma ayristirmasi ozdesligi (varyans+kirpma+propensity = toplam)",
        gecti=en_buyuk < TOLERANS_AYRISTIRMA,
        olcum=f"azami |artik| = {en_buyuk:.3e} TL/satir "
              f"({len(c.ayristirmalar)} politika, tolerans {TOLERANS_AYRISTIRMA:.0e})",
    )


def kontrol_isinma_sadakati(cfg: Config, kosu: str) -> Kontrol:
    """Rollout dallanma aninda TABAN dunyanin ta kendisinde mi.

    Kapali dongunun anlami buna bagli: politika gercek dunyanin devamina
    tepki vermiyorsa "closed-loop" bir benzetimin benzetimidir. Kontrol
    dunyayi iki kez kosar -- biri kesintisiz, biri dallanma haftasinda durup
    -- ve eczane stogu / SOW / depo stogunu TAMSAYI olarak karsilastirir.
    """
    b = cfg.ope.rollout.baslangic_hafta
    durum_a = dunya_kur(cfg, SeedBankasi(cfg.profil.temel_seed))
    for _ in range(b):
        hafta_adimi(durum_a)
    durum_b = dunya_kur(cfg, SeedBankasi(cfg.profil.temel_seed))
    for _ in range(b):
        hafta_adimi(durum_b)

    stok_ayni = bool(np.array_equal(durum_a.kovalar.toplam(),
                                    durum_b.kovalar.toplam()))
    sow_ayni = bool(np.array_equal(durum_a.sow, durum_b.sow))
    depo_ayni = all(durum_a.depo.eldeki_adet(s) == durum_b.depo.eldeki_adet(s)
                    for s in range(durum_a.S))
    return Kontrol(
        ad=f"isinma sadakati: {b}. haftada durum tekrar uretilebilir",
        gecti=stok_ayni and sow_ayni and depo_ayni,
        olcum=f"eczane stogu={'ayni' if stok_ayni else 'FARKLI'}, "
              f"SOW={'ayni' if sow_ayni else 'FARKLI'}, "
              f"depo={'ayni' if depo_ayni else 'FARKLI'} "
              f"(toplam stok {int(durum_a.kovalar.toplam().sum()):,})",
    )


def kontrol_determinizm(cfg: Config, kosu: str) -> Kontrol:
    """Ayni config + ayni seed -> ayni M6 tablosu (CLAUDE.md 5).

    M6 en cok rassal katman: kayit tekrarlari, kabul zarlari, rollout
    haftalari. Tekrar uretilemezse hicbir sapma yorumu kalici degildir.
    """
    m4 = m4_boru_hatti(cfg, kosu, VERI_DIZINI)
    a = m6_duz_metrikler(m6_boru_hatti(cfg, m4), cfg)
    b = m6_duz_metrikler(m6_boru_hatti(cfg, m4), cfg)
    farkli = [k for k in a
              if not (np.isnan(a[k]) and np.isnan(b.get(k, np.nan)))
              and a[k] != b.get(k)]
    return Kontrol(
        ad="determinizm: iki M6 kosusu ayni metrikleri veriyor",
        gecti=not farkli,
        olcum=f"{len(a)} metrik karsilastirildi, {len(farkli)} fark"
              + (f" ({farkli[:3]})" if farkli else ""),
    )


# --------------------------------------------------------------------------
# KRITER
# --------------------------------------------------------------------------
def kontrol_kirpma_canli(c) -> Kontrol:
    """Kirpma fiilen agirlik siliyor mu.

    Silmiyorsa `ope.tahminci.kirpma_esigi` bir kadran degil sussuz bir
    sayidir ve sapma ayristirmasinin "kirpma" kalemi tanimi geregi sifirdir --
    yani rapor bir sebebi olcemez.
    """
    kutle = {ad: s.teshis.kirpilan_kutle_orani for ad, s in c.ope_sonuclari.items()}
    en_buyuk = max(kutle.values(), default=0.0)
    kazanan = max(kutle, key=kutle.get) if kutle else "-"
    return Kontrol(
        ad="kirpma canli: agirlik tavani fiilen kutle siliyor",
        gecti=en_buyuk > ESIK_KIRPMA_KUTLESI,
        olcum=f"azami silinen kutle %{en_buyuk * 100:.2f} ({kazanan}), "
              f"{sum(v > ESIK_KIRPMA_KUTLESI for v in kutle.values())}/{len(kutle)} "
              f"politikada kirpma var",
    )


def kontrol_ortusme_canli(c) -> Kontrol:
    """Ortusme teshisi fiilen satir isaretliyor mu.

    Config yuklemesindeki `_m6_ortusme_kilidi` esigin tabanin ALTINA
    inmedigini garanti ediyor; bu kontrol bir adim otesini sinar: esik
    gecerli olsa da veri gercekten kor bolge iceriyor mu.
    """
    toplam = sum(s.teshis.ortusme_ihlali_orani * s.teshis.n
                 for s in c.ope_sonuclari.values() if np.isfinite(s.teshis.n))
    en_buyuk_ad = max(c.ope_sonuclari,
                      key=lambda a: c.ope_sonuclari[a].teshis.ortusme_ihlali_orani)
    en_buyuk = c.ope_sonuclari[en_buyuk_ad].teshis
    return Kontrol(
        ad="ortusme teshisi canli: kor bolge fiilen isaretleniyor",
        gecti=toplam >= ESIK_ORTUSME_IHLALI,
        olcum=f"toplam {int(toplam):,} isaretli satir | en yuksek "
              f"{en_buyuk_ad}: %{en_buyuk.ortusme_ihlali_orani * 100:.2f} satir, "
              f"odul payi %{en_buyuk.ortusme_ihlali_odul_payi * 100:.2f}",
    )


def kontrol_dr_varyans_kazanci(c) -> Kontrol:
    """DR, IPS'ten daha az oynak mi (bagimsiz kayit tekrarlarinda).

    DR'nin varlik sebebi budur: onem agirligini yalnizca ARTIGA uygular.
    Kazanc yoksa ya sonuc modeli odul mertebesinde hatalidir ya da eslesme
    orani o kadar dusuktur ki duzeltme terimi tahmini yine tek basina tasir.
    Ikisi de raporlanmasi gereken bir bulgudur; bu yuzden kontrol yonu
    "DR <= IPS" seklinde ve gevsek.
    """
    ips = [v["ips_sapma"] for v in c.tekrar_sapmalari.values()]
    dr = [v["dr_sapma"] for v in c.tekrar_sapmalari.values()]
    gecerli = [(a, b) for a, b in zip(ips, dr) if np.isfinite(a) and np.isfinite(b)]
    if not gecerli:
        return Kontrol("DR varyans kazanci", False, "olculebilir tekrar yok")
    kazanan = sum(b <= a for a, b in gecerli)
    ort_ips = float(np.mean([a for a, _ in gecerli]))
    ort_dr = float(np.mean([b for _, b in gecerli]))
    return Kontrol(
        ad="DR varyans kazanci: sd(DR) <= sd(IPS) politikalarin cogunda",
        gecti=kazanan > len(gecerli) / 2,
        olcum=f"{kazanan}/{len(gecerli)} politikada DR daha kararli | "
              f"ort sd: IPS {ort_ips:.4f} vs DR {ort_dr:.4f} TL/satir",
    )


def kontrol_propensity_bozmasi(cfg: Config, kosu: str, m4) -> Kontrol:
    """Propensity kalibrasyonu bozulunca sapma ONGORULEN yonde kayiyor mu.

    Ongoru cebirsel ve TEK YONLU DEGILDIR -- ilk yazdigimda oyle sandim ve
    yanildim (reports/m6.md "beklentiyle gercegin ayristigi yer"). Dogru
    ifade su:

        IPS agirligi eslesen satirlarda w = 1 / pi_kullanilan. O satirlarda
        kullanilan propensity gercekten BUYUKSE agirlik kuculur ve tahmin
        ASAGI kayar; KUCUKSE yukari kayar. Yani

            isaret(propensity kalemi) = -isaret( E[pi_kullanilan - pi_log] )
                                        eslesen satirlar uzerinde

    Sicaklik > 1'in dagilimi "duzlestirmesi" bu isareti tek basina
    belirlemez: hangi kollarin buyudugu, hedef politikanin AGIRLIKLI OLARAK
    hangi kolu sectigine baglidir. Bu dunyada hedeflerin %71-100'u kol 0
    ("teklif yok") secer; kol 0 kutlenin buyuk kismini tasidigi icin
    duzlestirmede KUCULUR, agirlik BUYUR ve tahmin YUKARI kayar. Kontrol bu
    yuzden sabit bir yon degil, satir bazinda olculen yonu sinar.
    """
    bozuk = config_yukle(cfg.profil.ad, gecersiz_kilma={
        "ope.propensity.sicaklik": BOZMA_SICAKLIGI,
        "ope.rollout.politikalar": ["teklif_yok"],
        "ope.rollout.ufuk_hafta": 1,
        "ope.rollout.raporlanan_ufuklar": [1],
        "ope.rollout.teklif_penceresi_hafta": 1,
    })
    c = m6_boru_hatti(bozuk, m4)
    v, p = c.veri, c.prop
    tutan, toplam, kalemler = 0, 0, []
    for ad, kol in c.hedefler.items():
        esles = v.kol == kol
        if not esles.any():
            continue
        fark = float((p.propensity[esles] - v.propensity[esles]).mean())
        kalem = c.ayristirmalar[ad].propensity_kalemi
        kalemler.append(kalem)
        toplam += 1
        tutan += int(np.sign(kalem) == -np.sign(fark))
    canli = any(abs(k) > TOLERANS_AYRISTIRMA for k in kalemler)
    return Kontrol(
        ad=f"propensity bozmasi (sicaklik={BOZMA_SICAKLIGI}) olculen yonde kaydiriyor",
        gecti=canli and tutan > toplam / 2,
        olcum=f"{tutan}/{toplam} politikada isaret ongoruyle uyustu | "
              f"kalem araligi [{min(kalemler):+.3f}, {max(kalemler):+.3f}] TL/satir | "
              f"kalibrasyon hatasi {c.prop.kalibrasyon_hatasi:.4f}, "
              f"log oran {c.prop.log_orani_ortalamasi:+.3f}",
    )


def kontrol_ufuk_ayrismasi(c, cfg: Config) -> Kontrol:
    """Offline ile online SISTEMATIK olarak ayrisiyor mu.

    M6'nin varlik sebebi bu ayrismadir. Iki sayi cakisirsa milestone bir sey
    ogretmiyor demektir ve CLAUDE.md'nin talimati acik: "offline ile online
    supheli derecede uyusuyorsa exploration fazla genis demektir -- daralt ve
    tekrar kos". Bu yuzden kontrol AYRISMANIN VARLIGINI arar, yoklugunu degil.
    """
    son = max(cfg.ope.rollout.raporlanan_ufuklar)
    ilgili = [k for k in c.kopruler if k.ufuk == son and k.tahminci == "dr"]
    if not ilgili:
        return Kontrol("ufuk ayrismasi", False, "kopru satiri yok")
    oranlar = [abs(k.ufuk_kalemi) / max(abs(k.taban_net_marj_tl), 1e-9)
               for k in ilgili]
    en_buyuk = max(oranlar)
    isaret_donen = [k.politika for k in ilgili if k.isaret_dondu]
    return Kontrol(
        ad=f"offline/online ayrismasi @{son} hafta (exploration cok genis degil)",
        gecti=en_buyuk > ESIK_UFUK_AYRISMASI,
        olcum=f"azami |ufuk kalemi| / taban = %{en_buyuk * 100:.1f} "
              f"(esik %{ESIK_UFUK_AYRISMASI * 100:.0f}) | isaret donen politika: "
              f"{isaret_donen or 'yok'}",
    )


def kontrol_kanibalizm(c) -> Kontrol:
    """Teklif veren politika organik siparisi FIILEN yiyor mu.

    Gecikmeli bedelin uc kanalindan biri (sim/rollout.py). Sifirsa kapali
    dongu tek adimlik degerlendirmeden yapisal olarak farksiz olurdu ve
    "ufuk kalemi" diye bir sey olmazdi.
    """
    fark = {ad: g["kanibalizm_organik_siparis_farki"] for ad, g in c.gecikmeli.items()}
    negatif = sum(v < 0 for v in fark.values())
    return Kontrol(
        ad="kanibalizm kanali canli: teklif organik siparisi dusuruyor",
        gecti=negatif == len(fark) and len(fark) > 0,
        olcum=f"{negatif}/{len(fark)} politikada organik siparis azaldi | "
              + ", ".join(f"{a}: {v:+,.0f} adet" for a, v in list(fark.items())[:3]),
    )


def kontrol_iade_kanali(c) -> Kontrol:
    """Derin MF iadeyi artiriyor mu (SPEC 2.5'in alici tarafi direnci).

    Kapali dongunun ikinci gecikmeli kanali. `agresif` bedava adet yukluyor;
    eczane emebileceginden fazlasini iade ediyor. Tek adimlik tahminci bu
    kalemi tanimi geregi goremez.
    """
    if "agresif" not in c.gecikmeli:
        return Kontrol("iade kanali", True, "agresif politika kosulmadi, atlandi")
    a = c.gecikmeli["agresif"]
    digerleri = [g["iade_adet_farki"] for ad, g in c.gecikmeli.items()
                 if ad != "agresif"]
    en_buyuk_diger = max(digerleri) if digerleri else 0.0
    return Kontrol(
        ad="iade kanali: agresif MF iadeyi en cok artiran politika",
        gecti=a["iade_adet_farki"] > max(en_buyuk_diger, 0.0),
        olcum=f"agresif +{a['iade_adet_farki']:,.0f} adet iade "
              f"({a['bedava_adet']:,.0f} bedava adet) vs digerlerinin azamisi "
              f"{en_buyuk_diger:+,.0f}",
    )


def kontrol_tahminci_sapmasi(c) -> Kontrol:
    """Tahminciler oracle'i %95 blok bootstrap araliginda yakaliyor mu.

    GEVSEK bir kontrol ve bilerek oyle: amac tahmincinin mukemmel olmasi
    degil, sapmasinin OLCULEBILIR ve ARALIKLA TUTARLI olmasi. Hicbir politika
    yakalanamiyorsa ya odul olcegi kaymistir ya aralik yanlis kuruluyordur.
    """
    dr = [d for d in c.denetimler if d.tahminci == "dr"]
    kapsayan = sum(d.araligi_kapsiyor for d in dr)
    ort = float(np.mean([abs(d.sapma_yuzde) for d in dr
                         if np.isfinite(d.sapma_yuzde)]))
    return Kontrol(
        ad="DR tahmincisi oracle'i %95 araliginda yakaliyor (>= yarisinda)",
        gecti=kapsayan >= len(dr) / 2,
        olcum=f"{kapsayan}/{len(dr)} politikada aralik oracle'i kapsiyor | "
              f"ortalama |sapma| %{ort:.1f}",
    )


# --------------------------------------------------------------------------
# grafikler
# --------------------------------------------------------------------------
def sekil_tahminci_denetimi(c) -> Path:
    """Politika x tahminci: tahmin - oracle. M6'nin (A) yarisi."""
    politikalar = sorted(c.ope_sonuclari)
    tahminciler = ("ips_kirpmasiz", "ips", "snips", "dogrudan", "dr")
    x = np.arange(len(politikalar))
    genislik = 0.16
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for j, t in enumerate(tahminciler):
        y = [c.ope_sonuclari[p].deger(t) - c.oracle.deger(c.hedefler[p])
             for p in politikalar]
        ax.bar(x + (j - 2) * genislik, y, genislik, label=t)
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(politikalar, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("tahmin - oracle (TL/satir)")
    ax.set_title("Tahminci sapmasi: sentetik dunyada oracle biliniyor", fontsize=10)
    ax.legend(fontsize=8, ncol=5)
    return _sekil_kaydet("tahminci_denetimi")


def sekil_sapma_ayristirmasi(c) -> Path:
    """Yigilmis kalemler: varyans / kirpma / propensity = toplam sapma."""
    politikalar = sorted(c.ayristirmalar)
    kalemler = [("varyans", "varyans_kalemi"), ("kirpma", "kirpma_kalemi"),
                ("propensity", "propensity_kalemi")]
    x = np.arange(len(politikalar))
    fig, ax = plt.subplots(figsize=(12, 4.6))
    pozitif = np.zeros(len(politikalar))
    negatif = np.zeros(len(politikalar))
    for ad, alan in kalemler:
        v = np.array([getattr(c.ayristirmalar[p], alan) for p in politikalar])
        taban = np.where(v >= 0, pozitif, negatif)
        ax.bar(x, v, 0.6, bottom=taban, label=ad)
        pozitif += np.where(v >= 0, v, 0.0)
        negatif += np.where(v < 0, v, 0.0)
    toplam = [c.ayristirmalar[p].toplam_sapma for p in politikalar]
    ax.plot(x, toplam, "ko", ms=6, label="toplam sapma (ozdeslik)")
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(politikalar, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("IPS - oracle (TL/satir)")
    ax.set_title("Sapmanin kaynagi: kalemler toplami noktaya OTURMALI", fontsize=10)
    ax.legend(fontsize=8)
    return _sekil_kaydet("sapma_ayristirmasi")


def sekil_ufuk_egrisi(c, cfg: Config) -> Path:
    """Birikimli artimsal net marj, hafta hafta. M6'nin (B) yarisi."""
    taban = c.rollout["teklif_yok"].birikimli_net_marj()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for ad, o in c.rollout.items():
        if ad == "teklif_yok":
            continue
        v = o.birikimli_net_marj() - taban[:len(o.haftalar)]
        ax[0].plot(np.arange(1, v.size + 1), v, marker="", label=ad)
    ax[0].axhline(0, c="k", ls="--", lw=1)
    for u in cfg.ope.rollout.raporlanan_ufuklar:
        ax[0].axvline(u, c="0.8", lw=0.8)
    ax[0].set_xlabel("rollout haftasi")
    ax[0].set_ylabel("birikimli artimsal net marj (TL)")
    ax[0].set_title("Kapali dongu: teklif_yok'a gore", fontsize=10)
    ax[0].legend(fontsize=8)

    # Sag panel: ayni politikanin offline tahmini ile online gerceklesmesi.
    ufuklar = cfg.ope.rollout.raporlanan_ufuklar
    politikalar = sorted({k.politika for k in c.kopruler})
    x = np.arange(len(ufuklar))
    for p in politikalar:
        off = [next((k.offline_yuzde for k in c.kopruler
                     if k.politika == p and k.ufuk == u and k.tahminci == "dr"),
                    np.nan) for u in ufuklar]
        on = [next((k.online_yuzde for k in c.kopruler
                    if k.politika == p and k.ufuk == u and k.tahminci == "dr"),
                   np.nan) for u in ufuklar]
        cizgi, = ax[1].plot(x, off, "o--", ms=4, label=f"{p} offline")
        ax[1].plot(x, on, "s-", ms=4, c=cizgi.get_color(), label=f"{p} online")
    ax[1].axhline(0, c="k", ls="--", lw=1)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([str(u) for u in ufuklar])
    ax[1].set_xlabel("degerlendirme ufku (hafta)")
    ax[1].set_ylabel("taban net marja gore %")
    ax[1].set_title("Offline (kesikli) vs kapali dongu (duz)", fontsize=10)
    ax[1].legend(fontsize=6, ncol=2)
    return _sekil_kaydet("ufuk_egrisi")


def sekil_agirlik_dagilimi(c) -> Path:
    """Onem agirliklarinin kuyrugu: IPS varyansinin gorsel kaynagi."""
    cfg_esik = None
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for ad in sorted(c.ope_sonuclari):
        s = c.ope_sonuclari[ad]
        ax[1].bar(ad, s.teshis.ess_orani)
        cfg_esik = s.teshis.agirlik_azami if cfg_esik is None else cfg_esik
    ax[1].set_ylabel("ESS / n")
    ax[1].set_title("Etkin orneklem orani", fontsize=10)
    ax[1].tick_params(axis="x", rotation=25, labelsize=7)

    for ad in sorted(c.ope_sonuclari):
        s = c.ope_sonuclari[ad]
        ax[0].scatter(s.teshis.eslesme_orani, s.teshis.agirlik_azami, s=40)
        ax[0].annotate(ad, (s.teshis.eslesme_orani, s.teshis.agirlik_azami),
                       fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[0].set_xlabel("eslesme orani (hedef loglarda gorulme sikligi)")
    ax[0].set_ylabel("azami ham agirlik 1/pi")
    ax[0].set_yscale("log")
    ax[0].set_title("Ortusme: az eslesen politika buyuk agirlik tasir", fontsize=10)
    return _sekil_kaydet("agirlik_dagilimi")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kosu", default="full", help="data/<kosu> altindaki dunya")
    ap.add_argument("--profil", default=None, help="varsayilan: kosu manifestindeki profil")
    ap.add_argument("--hizli", action="store_true",
                    help="determinizm ve propensity bozma kontrollerini atla")
    args = ap.parse_args()

    manifest = Kosu(args.kosu).manifest_oku()
    profil = args.profil or manifest["profil"]
    cfg = config_yukle(profil)
    print(f"kosu={args.kosu} profil={profil} "
          f"dunya_config_hash={manifest['config_hash']} m6_config_hash={cfg.hash()}")

    m4 = m4_boru_hatti(cfg, args.kosu, VERI_DIZINI)
    print(f"M4 yeniden kullanildi: olcum origin={m4.olcum_originleri}, "
          f"{sum(b.teklifler.height for b in m4.bloklar)} vetosuz aday satiri")

    c = m6_boru_hatti(cfg, m4)
    r = cfg.ope.rollout
    print(f"\nM6: {c.veri.n:,} loglanmis satir "
          f"({cfg.ope.kayit.tekrar_sayisi} tekrar x {c.satir_sayisi:,.0f} satir) | "
          f"propensity={c.prop.kaynak} | rollout={r.baslangic_hafta}.."
          f"{r.baslangic_hafta + r.ufuk_hafta} | sure={c.zaman}")

    print("\nOFFLINE TAHMIN vs ORACLE (TL/satir):")
    with pl.Config(tbl_rows=20, tbl_cols=14, tbl_width_chars=220, float_precision=4):
        print(pl.DataFrame([
            {"politika": ad, "oracle": c.oracle.deger(c.hedefler[ad]),
             **{t: s.deger(t) for t in ev_ope.TAHMINCILER},
             "ESS/n": s.teshis.ess_orani,
             "eslesme": s.teshis.eslesme_orani,
             "kirpilan_kutle": s.teshis.kirpilan_kutle_orani}
            for ad, s in c.ope_sonuclari.items()]))

    print("\nSAPMANIN KAYNAGI:")
    with pl.Config(tbl_rows=20, tbl_cols=14, tbl_width_chars=220, float_precision=4):
        print(ev_rapor.ayristirma_tablosu(list(c.ayristirmalar.values())))

    print("\nOFFLINE -> ONLINE KOPRUSU (DR):")
    with pl.Config(tbl_rows=40, tbl_cols=10, tbl_width_chars=200, float_precision=1):
        print(ev_rapor.kopru_tablosu([k for k in c.kopruler if k.tahminci == "dr"]))

    sekiller = [
        sekil_tahminci_denetimi(c),
        sekil_sapma_ayristirmasi(c),
        sekil_ufuk_egrisi(c, cfg),
        sekil_agirlik_dagilimi(c),
    ]

    kontroller = [
        kontrol_ope_sizintisi(),
        kontrol_ozdeslik(c),
        kontrol_ayristirma_artigi(c),
        kontrol_isinma_sadakati(cfg, args.kosu),
        kontrol_kirpma_canli(c),
        kontrol_ortusme_canli(c),
        kontrol_dr_varyans_kazanci(c),
        kontrol_tahminci_sapmasi(c),
        kontrol_ufuk_ayrismasi(c, cfg),
        kontrol_kanibalizm(c),
        kontrol_iade_kanali(c),
    ]
    if not args.hizli:
        kontroller.append(kontrol_propensity_bozmasi(cfg, args.kosu, m4))
        kontroller.append(kontrol_determinizm(cfg, args.kosu))

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
