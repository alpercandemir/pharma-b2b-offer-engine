"""Aksiyon uzayi (D1), marj aritmetigi ve aksiyon secimi.

D1 TARTISMAYA KAPALI: aksiyon `(mal_fazlasi_orani, vade_gunu)` ciftidir.
Bu dosyada yuzde iskonto yoktur; fiyat regule oldugu icin tavizin fiili
kanallari MF ve vadedir (SPEC 1). Aksiyon uzayi:

    kol 0        : TEKLIF YOK (kontrol). Eczane yine siparis verebilir;
                   o organik siparisin marji `taban_vade_gun` ile hesaplanir.
    kol 1..A-1   : (mf_orani, vade_gunu) capraz carpimi.

`(mf=0, vade=taban)` kolu bilerek vardir: BEDELSIZ teklif. Marji kol 0 ile
birebir ayni, kabul olasiligi farkli (gorunurluk etkisi). Uplift'in "bedava
kazanc" bandi burasidir ve bir politika bunu bulamiyorsa uplift'i degil
seviyeyi optimize ediyordur.

VADE IKI YONLUDUR. Taban vadenin ALTINDAKI kol bir taviz degil, marj
HASADIDIR: net isletme sermayesi = musteri vadesi - tedarikci vadesi, ve
musteri vadesi tedarikci vadesinin altindayken fonlama kalemi POZITIFTIR.
Tek yonlu bir aksiyon uzayi (sadece "daha iyi sartlar") bu kaldiraci
gorunmez kilardi.

KISIT KATMANI BURADA DA VETO YETKISINI KORUR (D6):
  - `mf_izinli=false` (SGK geri odeme kapsami) satirda MF kollari kapalidir,
    vade kollari acik (SPEC 2.5).
  - Koli katina yuvarlama adedi BUYUTUR; buyuyen adet emilim tavanini
    asiyorsa o satirda MF kanali kapanir.
  - MF orani koli yuvarlamasindan sonra tek bir bedava adet bile uretmiyorsa
    kol kapalidir: "7 adetlik satirda 10+1" anlamsizdir (SPEC 2.1).
  - Aksiyon secildikten SONRA portfoy kredi limiti yeniden kontrol edilir;
    yuvarlama tutari buyuttugu icin bu kontrol M3'unkinin tekrari degildir.

Bu dosya ground_truth okumaz. Gercek tepki `sim/response.py`de.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from core.config import Config
from policy.candidates import AdayDunyasi, OriginGorunumu
from policy.constraints import portfoy_kredi_vetosu

# Yillik gun sayisi. Vade maliyeti gun bazli hesaplanir; takvim sabiti.
GUN_YIL = 365.0
# Kontrol kolunun indeksi. Butun [n, A] matrislerinde 0. sutun "teklif yok".
TEKLIF_YOK = 0


@dataclass(frozen=True)
class AksiyonUzayi:
    """Kollarin (mf_orani, vade_gunu) tablosu. Kol 0 = teklif yok."""

    mf: np.ndarray            # [A] kol 0 icin 0.0
    vade: np.ndarray          # [A] kol 0 icin taban_vade_gun
    adlar: tuple[str, ...]

    @property
    def A(self) -> int:
        return self.mf.size

    @property
    def teklif_kollari(self) -> np.ndarray:
        return np.arange(1, self.A)


def aksiyon_uzayi(cfg: Config) -> AksiyonUzayi:
    a = cfg.politika.aksiyon
    mf = [0.0]
    vade = [float(a.taban_vade_gun)]
    adlar = ["teklif_yok"]
    for m in a.mf_oranlari:
        for v in a.vade_gunleri:
            mf.append(float(m))
            vade.append(float(v))
            adlar.append(f"mf{m:.2f}_v{v}")
    return AksiyonUzayi(mf=np.array(mf), vade=np.array(vade), adlar=tuple(adlar))


# --------------------------------------------------------------------------
# adet ve MF yuvarlamasi
# --------------------------------------------------------------------------
def koli_yuvarlamasi(adet: np.ndarray, koli: np.ndarray, mf: float, cfg: Config,
                     emilim_tavani: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                         np.ndarray]:
    """(efektif_adet, bedava_adet, yuvarlandi). SPEC 2.1 + reports/m3.md 8 borcu.

    Iki kural, ayri ayri:

    (1) BEDAVA ADET TAM SAYIDIR: `floor(adet * mf)`. Yarim kutu mal fazlasi
        diye bir sey yok. Sonuc sifirsa "10+1" o satirda anlamsizdir ve MF
        kanali kapanir - M3'un birakigi borcun tam karsiligi ("10+1 teklifi
        7 adetlik bir satirda anlamsiz").

    (2) TEKLIF ADEDI KOLI KATINA yuvarlanir, AMA emilim tavanini asmamak
        sartiyla. Yuvarlama adedi buyutur; kucuk hucrede koli 30 iken 4
        adetlik teklifi 30'a cikarmak eczaneye emebileceginin yedi kati mal
        yikmak olur. Tavani asiyorsa yuvarlama ATLANIR (kanal kapanmaz):
        "N adet + floor(N*mf) bedava" hala gecerli bir MF teklifidir.

        ILK UYGULAMA KANALI KAPATIYORDU: yuvarlama kosulsuz yapilinca MF
        kollari satirlarin yalnizca %27'sinde acik kaliyordu ve aksiyon
        uzayi fiilen tek boyuta (vade) iniyordu. Olcum reports/m4.md 7.3'te.
    """
    if mf <= 0.0:
        return adet.astype(float), np.zeros(adet.size), np.zeros(adet.size, dtype=bool)
    efektif = adet.astype(float)
    yuvarlandi = np.zeros(adet.size, dtype=bool)
    if cfg.politika.aksiyon.koli_katina_yuvarla:
        k = np.maximum(koli.astype(float), 1.0)
        aday = np.ceil(adet / k) * k
        yuvarlandi = aday <= emilim_tavani
        efektif = np.where(yuvarlandi, aday, efektif)
    return efektif, np.floor(efektif * mf), yuvarlandi


# --------------------------------------------------------------------------
# marj
# --------------------------------------------------------------------------
def brut_marj(adet: np.ndarray, bedava: np.ndarray, dsf: np.ndarray,
              depo_marji: np.ndarray, vade_gun: float, vade_riski: np.ndarray,
              cfg: Config) -> np.ndarray:
    """Bir satirin bir kol altindaki beklenen brut marji (TL), KABUL SARTIYLA.

        ciro          = adet * dsf
        urun marji    = ciro * depo_kar_marji
        MF maliyeti   = bedava * birim_maliyet * (1 - tedarikci_destegi)
        fonlama       = ciro * yillik_oran * (vade - tedarikci_vade) / 365
        kredi zarari  = ciro * vade_riski * temerrut_katsayisi * vade / 365

    Fonlama kaleminin ISARETI degisebilir: musteri vadesi tedarikci
    vademizin altindaysa tedarikci bizi finanse ediyordur ve kalem marja
    EKLENIR. Bu, aksiyon uzayinin vade boyutunu iki yonlu yapan mekanizmadir.
    """
    s = cfg.politika.skor
    ciro = adet * dsf
    birim_maliyet = dsf * (1.0 - depo_marji)
    mf_maliyeti = bedava * birim_maliyet * (1.0 - s.tedarikci_mf_destek_orani)
    net_vade = vade_gun - s.tedarikci_vade_gun
    fonlama = ciro * s.yillik_fonlama_orani * net_vade / GUN_YIL
    kredi_zarari = ciro * vade_riski * s.temerrut_ceza_katsayisi * vade_gun / GUN_YIL
    return ciro * depo_marji - mf_maliyeti - fonlama - kredi_zarari


@dataclass
class TeklifMatrisleri:
    """Bir origin'in aday satirlari x kollar tablolari.

    Hepsi GOZLEMLENEBILIR buyukluklerden kurulur; tepki olasiligi burada yok.
    """

    uzay: AksiyonUzayi
    adet: np.ndarray          # [n, A] kol altinda sevk edilecek adet
    bedava: np.ndarray        # [n, A] bedava adet
    marj: np.ndarray          # [n, A] kabul sartiyla brut marj (TL)
    izinli: np.ndarray        # [n, A] kol acik mi
    yuvarlandi: np.ndarray    # [n, A] adet koli katina yuvarlandi mi
    kapali_sebep: dict[str, np.ndarray]   # sebep -> [n] kac kol kapatti


def teklif_matrisleri(dunya: AdayDunyasi, cfg: Config,
                      teklifler: pl.DataFrame) -> TeklifMatrisleri:
    """Aday satirlari icin kol bazli adet / bedava / marj / izin tablolari."""
    uzay = aksiyon_uzayi(cfg)
    n, A = teklifler.height, uzay.A
    s_idx = teklifler["sku_idx"].to_numpy()
    p_idx = teklifler["eczane_idx"].to_numpy()
    urun, ecz = dunya.urunler, dunya.eczaneler

    adet0 = teklifler["teklif_adedi"].to_numpy().astype(float)
    dsf = dunya.dsf[s_idx]
    depo_marji = urun["depo_kar_marji"].to_numpy()[s_idx]
    koli = urun["koli_ici_adet"].to_numpy()[s_idx]
    vade_riski = ecz["vade_riski_skoru"].to_numpy().astype(float)[p_idx]
    mf_izinli = teklifler["mf_izinli"].to_numpy().astype(bool)
    # Emilim tavani: yuvarlanmis adet de eczanenin emme kapasitesine tabidir.
    hiz = teklifler["hiz_tahmini"].to_numpy() * cfg.politika.aday.hiz_telafi_katsayisi
    emilim_tavani = hiz * cfg.politika.kisit.azami_kapsama_hafta

    adet = np.empty((n, A))
    bedava = np.empty((n, A))
    marj = np.empty((n, A))
    izinli = np.ones((n, A), dtype=bool)
    yuvarlandi = np.zeros((n, A), dtype=bool)
    kapali = {ad: np.zeros(n, dtype=np.int32)
              for ad in ("mf_kanali_kapali", "mf_bedava_sifir", "yuvarlama_atlandi")}

    for a in range(A):
        m = float(uzay.mf[a])
        adet[:, a], bedava[:, a], yuvarlandi[:, a] = koli_yuvarlamasi(
            adet0, koli, m, cfg, emilim_tavani)
        marj[:, a] = brut_marj(adet[:, a], bedava[:, a], dsf, depo_marji,
                               float(uzay.vade[a]), vade_riski, cfg)
        if a == TEKLIF_YOK or m <= 0.0:
            continue
        kanal = ~mf_izinli
        bos = bedava[:, a] < 1.0
        izinli[:, a] = ~(kanal | bos)
        kapali["mf_kanali_kapali"] += kanal
        kapali["mf_bedava_sifir"] += (~kanal & bos)
        kapali["yuvarlama_atlandi"] += (~kanal & ~bos & ~yuvarlandi[:, a])

    return TeklifMatrisleri(uzay=uzay, adet=adet, bedava=bedava, marj=marj,
                            izinli=izinli, yuvarlandi=yuvarlandi,
                            kapali_sebep=kapali)


# --------------------------------------------------------------------------
# aksiyon secimi
# --------------------------------------------------------------------------
@dataclass
class Secim:
    """Bir politikanin cikardigi karar: satir basina kol indeksi."""

    ad: str
    kol: np.ndarray           # [n] 0 = teklif yok
    kazanc: np.ndarray        # [n] politikanin KENDI amac fonksiyonundaki kazanc

    @property
    def teklif_maskesi(self) -> np.ndarray:
        return self.kol != TEKLIF_YOK


def en_iyi_kol(deger: np.ndarray, izinli: np.ndarray) -> np.ndarray:
    """[n] satir basina en yuksek degerli IZINLI teklif kolu (kol 0 haric).

    Beraberlik bozma: en kucuk kol indeksi. Rassal degil, tekrar uretilebilir.
    """
    maskeli = np.where(izinli, deger, -np.inf)
    maskeli[:, TEKLIF_YOK] = -np.inf
    return np.argmax(maskeli, axis=1)


def sec(ad: str, deger: np.ndarray, taban_deger: np.ndarray, izinli: np.ndarray,
        eczane_idx: np.ndarray, tavan: int, esik: float) -> Secim:
    """Frekans tavani altinda aksiyon secimi.

    `deger`      : [n, A] politikanin kol basina amac degeri
    `taban_deger`: [n] "teklif vermeme"nin ayni amac fonksiyonundaki degeri.
                   Propensity tabanli politikada bu SIFIRDIR - ayrim burada.
    `esik`       : kazanc bu esigin altindaysa teklif verilmez.

    Eczane basina en yuksek kazancli `tavan` satir teklife donusur. Tavan
    M3'un frekans tavaninin aynisidir (`kisit.eczane_haftalik_teklif_tavani`):
    saha kapasitesi aksiyon secimiyle degismedi.
    """
    n = deger.shape[0]
    kol = np.zeros(n, dtype=np.int32)
    if n == 0:
        return Secim(ad=ad, kol=kol, kazanc=np.zeros(0))
    aday_kol = en_iyi_kol(deger, izinli)
    en_iyi_deger = deger[np.arange(n), aday_kol]
    # Hicbir teklif kolu acik olmayan satir (-inf) elenir.
    acik = np.isfinite(en_iyi_deger)
    kazanc = np.where(acik, en_iyi_deger - taban_deger, -np.inf)

    sira = np.lexsort((-kazanc, eczane_idx))
    sayac: dict[int, int] = {}
    for i in sira:
        if not acik[i] or kazanc[i] <= esik:
            continue
        e = int(eczane_idx[i])
        adet = sayac.get(e, 0)
        if adet < tavan:
            kol[i] = aday_kol[i]
            sayac[e] = adet + 1
    return Secim(ad=ad, kol=kol, kazanc=np.where(np.isfinite(kazanc), kazanc, 0.0))


def kredi_son_kontrolu(dunya: AdayDunyasi, cfg: Config, gor: OriginGorunumu,
                       teklifler: pl.DataFrame, mat: TeklifMatrisleri,
                       secim: Secim) -> np.ndarray:
    """Secilen tekliflerin portfoy kredi limiti kontrolu. [n] vetolanan maske.

    M3'un kredi vetosunun tekrari DEGIL: koli yuvarlamasi teklif adedini ve
    dolayisiyla tutari buyutuyor, aksiyon secimi de hangi satirin listeye
    girecegini degistiriyor. D6'nin veto yetkisi aksiyon seciminden SONRA da
    gecerlidir.
    """
    n = secim.kol.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    p = teklifler["eczane_idx"].to_numpy()
    dsf = dunya.dsf[teklifler["sku_idx"].to_numpy()]
    tutar = mat.adet[np.arange(n), secim.kol] * dsf
    return portfoy_kredi_vetosu(dunya, cfg, p, tutar, secim.kazanc,
                                secim.teklif_maskesi, gor)
