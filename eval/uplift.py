"""M4 olcumu: politika marji, marj farkinin ayristirilmasi, CATE kalitesi.

SENTETIK OLDUGU ICIN KARSI-OLGUSAL ELIMIZDE. `sim/response.py` her satir
icin her kolun gercek kabul olasiligini veriyor, dolayisiyla bir politikanin
degeri TAHMIN EDILMEZ, HESAPLANIR:

    V(pi) = sum_i  p_gercek(i, kol_i) * brut_marj(i, kol_i) * E[miktar carpani]

Bu M6 DEGILDIR. M6'nin sorusu "loglardan, gercegi bilmeden, bu degeri ne
kadar dogru tahmin edebiliriz" (IPS/SNIPS/DR). M4'te gercegi biliyoruz ve
politikalari onun uzerinde karsilastiriyoruz; OPE'nin denetleyecegi sayi
tam olarak buradaki sayidir.

MARJ FARKININ AYRISTIRILMASI. Iki politikanin toplam farki satir bazinda
tam olarak su bes sinifa bolunur ve toplamlari farka ESITTIR (ozdeslik
scripts/verify_m4.py'de sinaniyor):

    ikisi_de_vermedi          0 katki
    sadece_propensity_verdi   -artimsal(propensity kolu)   <- yakilan marj
    sadece_uplift_verdi       +artimsal(uplift kolu)       <- kacirilan firsat
    farkli_kol                artimsal(u) - artimsal(p)
    ayni_kol                  0 katki
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policy.scorer import TEKLIF_YOK, Secim, TeklifMatrisleri
from sim.response import Tepki


def satir_degeri(tepki: Tepki, mat: TeklifMatrisleri, kol: np.ndarray,
                 carpan: float) -> np.ndarray:
    """[n] satirin secilen kol altindaki GERCEK beklenen marji (TL)."""
    n = kol.size
    if n == 0:
        return np.zeros(0)
    idx = np.arange(n)
    return tepki.olasilik[idx, kol] * mat.marj[idx, kol] * carpan


def artimsal_marj(tepki: Tepki, mat: TeklifMatrisleri, kol: np.ndarray,
                  carpan: float) -> np.ndarray:
    """[n] teklif vermemeye gore gercek artimsal marj. Kol 0'da tam sifir."""
    taban = satir_degeri(tepki, mat, np.zeros(kol.size, dtype=int), carpan)
    return np.where(kol == TEKLIF_YOK, 0.0, satir_degeri(tepki, mat, kol, carpan) - taban)


@dataclass
class PolitikaOlcumu:
    ad: str
    toplam_marj: float          # V(pi), teklif verilmeyen satirlar dahil
    artimsal_marj: float        # V(pi) - V(hic teklif yok)
    teklif_sayisi: int
    satir_sayisi: int
    negatif_teklif_sayisi: int  # artimsal marji < 0 olan teklifler
    negatif_marj: float         # onlarin toplami (yakilan marj)
    ortalama_mf: float
    ortalama_vade: float
    beklenen_kabul: float       # teklif edilen satirlarda ortalama p_gercek

    @property
    def teklif_basina_artimsal(self) -> float:
        return self.artimsal_marj / self.teklif_sayisi if self.teklif_sayisi else float("nan")


def politika_olcumu(ad: str, tepki: Tepki, mat: TeklifMatrisleri, secim: Secim,
                    carpan: float) -> PolitikaOlcumu:
    kol = secim.kol
    n = kol.size
    artimsal = artimsal_marj(tepki, mat, kol, carpan)
    teklif = kol != TEKLIF_YOK
    negatif = teklif & (artimsal < 0)
    idx = np.arange(n)
    return PolitikaOlcumu(
        ad=ad,
        toplam_marj=float(satir_degeri(tepki, mat, kol, carpan).sum()),
        artimsal_marj=float(artimsal.sum()),
        teklif_sayisi=int(teklif.sum()),
        satir_sayisi=n,
        negatif_teklif_sayisi=int(negatif.sum()),
        negatif_marj=float(artimsal[negatif].sum()),
        ortalama_mf=float(mat.uzay.mf[kol][teklif].mean()) if teklif.any() else 0.0,
        ortalama_vade=float(mat.uzay.vade[kol][teklif].mean()) if teklif.any() else 0.0,
        beklenen_kabul=float(tepki.olasilik[idx, kol][teklif].mean()) if teklif.any() else 0.0,
    )


# --------------------------------------------------------------------------
# ayristirma
# --------------------------------------------------------------------------
AYRISTIRMA_SINIFLARI = ("ikisi_de_vermedi", "sadece_a_verdi", "sadece_b_verdi",
                        "farkli_kol", "ayni_kol")


def marj_farki_ayristirmasi(tepki: Tepki, mat: TeklifMatrisleri, a: Secim,
                            b: Secim, carpan: float) -> dict:
    """V(a) - V(b) farkini satir sinifina gore ayristirir.

    `a` uplift, `b` propensity olarak cagrilir: pozitif katki uplift'in
    kazanci, negatif katki kaybi demektir.
    """
    art_a = artimsal_marj(tepki, mat, a.kol, carpan)
    art_b = artimsal_marj(tepki, mat, b.kol, carpan)
    va, vb = a.kol != TEKLIF_YOK, b.kol != TEKLIF_YOK
    siniflar = {
        "ikisi_de_vermedi": ~va & ~vb,
        "sadece_a_verdi": va & ~vb,
        "sadece_b_verdi": ~va & vb,
        "farkli_kol": va & vb & (a.kol != b.kol),
        "ayni_kol": va & vb & (a.kol == b.kol),
    }
    cikti: dict[str, float] = {}
    for ad, maske in siniflar.items():
        cikti[f"{ad}_satir"] = int(maske.sum())
        cikti[f"{ad}_katki"] = float((art_a[maske] - art_b[maske]).sum())
    cikti["toplam_fark"] = float(art_a.sum() - art_b.sum())
    return cikti


# --------------------------------------------------------------------------
# CATE kalitesi
# --------------------------------------------------------------------------
def pehe(tahmin: np.ndarray, gercek: np.ndarray, izinli: np.ndarray) -> float:
    """sqrt(E[(tau_tahmin - tau_gercek)^2]) -- yalnizca IZINLI kollarda.

    Izinsiz kolda CATE tanimsizdir (o aksiyon o satirda hic verilemez);
    metrige katmak modeli olmayan bir hatadan sorumlu tutardi.
    """
    m = izinli.copy()
    m[:, TEKLIF_YOK] = False
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((tahmin[m] - gercek[m]) ** 2)))


def sira_korelasyonu(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman. Politika siralamayla calisiyor; seviye degil sira onemli."""
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def kazanc_egrisi(skor: np.ndarray, gercek_artimsal: np.ndarray,
                  dilim: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Qini egrisinin marj karsiligi.

    Satirlar TAHMIN EDILEN artimsal marja gore siralanir; egri o sirada
    birikimli GERCEK artimsal marji verir. Rassal sira dogrusal bir referans
    cizer; ikisinin arasindaki alan (AUUC) siralamanin degeridir.
    """
    n = skor.size
    if n == 0:
        return np.zeros(0), np.zeros(0), float("nan")
    sira = np.argsort(-skor, kind="stable")
    birikim = np.cumsum(gercek_artimsal[sira])
    kesim = np.linspace(0, n, dilim + 1)[1:].astype(int)
    x = kesim / n
    y = birikim[np.maximum(kesim - 1, 0)]
    toplam = birikim[-1]
    rassal = x * toplam
    auuc = float(np.trapezoid(y - rassal, x)) if n > 1 else float("nan")
    return x, y, auuc


# --------------------------------------------------------------------------
# heterojenlik teshisi
# --------------------------------------------------------------------------
@dataclass
class HeterojenlikTeshisi:
    """CLAUDE.md M4 talimatinin mekanik karsiligi.

    "Iki politika ayni sonucu veriyorsa simulatorde uplift heterojenligi yok
    demektir." Bu sinif o cumleyi olculebilir hale getirir; esikler
    scripts/verify_m4.py'de cikis kriteri olarak duruyor.
    """

    cate_sapmasi: float          # gercek en iyi kol CATE'inin sd'si (olasilik)
    cate_dilim_orani: float      # ust %10 / alt %10 ortalama CATE orani
    artimsal_sapma: float        # gercek artimsal marjin sd'si (TL)
    farkli_karar_orani: float    # iki politikanin ayrisan satir orani
    marj_farki: float            # V(uplift) - V(propensity)


def heterojenlik_teshisi(tepki: Tepki, mat: TeklifMatrisleri, uplift: Secim,
                         propensity: Secim, carpan: float) -> HeterojenlikTeshisi:
    izinli = mat.izinli.copy()
    izinli[:, TEKLIF_YOK] = False
    gercek_cate = np.where(izinli, tepki.uplift, -np.inf)
    en_iyi = gercek_cate.max(axis=1)
    en_iyi = en_iyi[np.isfinite(en_iyi)]
    if en_iyi.size:
        alt = np.quantile(en_iyi, 0.10)
        ust = np.quantile(en_iyi, 0.90)
        ort_alt = en_iyi[en_iyi <= alt].mean()
        ort_ust = en_iyi[en_iyi >= ust].mean()
        oran = float(ort_ust / ort_alt) if ort_alt > 0 else float("inf")
    else:
        oran = float("nan")

    art = np.where(izinli, tepki.olasilik * mat.marj * carpan
                   - (tepki.olasilik[:, [0]] * mat.marj[:, [0]] * carpan), np.nan)
    return HeterojenlikTeshisi(
        cate_sapmasi=float(en_iyi.std()) if en_iyi.size else float("nan"),
        cate_dilim_orani=oran,
        artimsal_sapma=float(np.nanstd(art)),
        farkli_karar_orani=float((uplift.kol != propensity.kol).mean())
        if uplift.kol.size else float("nan"),
        marj_farki=float(artimsal_marj(tepki, mat, uplift.kol, carpan).sum()
                         - artimsal_marj(tepki, mat, propensity.kol, carpan).sum()),
    )


# --------------------------------------------------------------------------
# guven araligi
# --------------------------------------------------------------------------
def eczane_bootstrap(fark: np.ndarray, eczane_idx: np.ndarray, tekrar: int,
                     seed: int) -> tuple[float, float, float]:
    """Satir bazli marj farkinin ECZANE blok bootstrap'i.

    Bagimsizlik birimi satir degil eczane: ayni eczanenin satirlari ortak
    frekans tavanini paylasiyor ve secimleri birbirine bagli (M2/M3'te ayni
    disiplin). Satir bazli bootstrap araligi oldugundan dar gosterirdi.
    """
    temel = float(fark.sum())
    if fark.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    gruplar = [np.flatnonzero(eczane_idx == e) for e in np.unique(eczane_idx)]
    G = len(gruplar)
    ornek = np.empty(tekrar)
    for i in range(tekrar):
        secim = rng.integers(0, G, G)
        ornek[i] = sum(fark[gruplar[j]].sum() for j in secim)
    return temel, float(np.percentile(ornek, 2.5)), float(np.percentile(ornek, 97.5))


def kol_destegi(kol: np.ndarray, propensity: np.ndarray, A: int) -> dict:
    """Overlap ozeti: kol basina orneklem ve en kucuk loglanan propensity.

    En kucuk propensity M6'nin IPS agirliginin ust sinirini belirler
    (1/pi_min). Simdiden raporlanmasinin sebebi: kayit politikasi M4'te
    seciliyor, bedelini M6 oduyor.
    """
    sayim = np.bincount(kol, minlength=A)
    return {
        "kol_orneklemi_min": int(sayim.min()),
        "kol_orneklemi_ort": float(sayim.mean()),
        "propensity_min": float(propensity.min()) if propensity.size else float("nan"),
        "azami_ips_agirligi": float(1.0 / propensity.min()) if propensity.size else float("nan"),
    }
