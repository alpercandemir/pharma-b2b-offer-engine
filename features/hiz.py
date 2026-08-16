"""Tuketim hizi cikarimi - SPEC M2 "siparis miktarindan tuketim hizi cikarimi".

Iki bagimsiz tahminci var ve ikisinin FARKLI yanlilik yapisi olmasi M2'nin
asil bulgusudur:

  AKIS tahmincisi   : bize gelen adet / gecen hafta.
      Eczanenin siparisi tedarikciler arasinda bolunur (multi-homing). Bir
      hucrede bize gelen akis, gercek tuketimin share_of_wallet kadarlik
      parcasidir:   hiz_akis ~= sow * hiz_gercek.
      sow LATENT oldugu icin bu yanlilik duzeltilemez.

  MIKTAR tahmincisi : ortalama siparis adedi / varsayilan dongu.
      Eczane order-up-to calisir ve bir siparisin TAMAMI tek tedarikciye
      gider. Yani siparis MIKTARI sow ile olceklenmez; sadece siparis
      SIKLIGI olceklenir. Miktar tabanli hiz bu yuzden sow'dan bagimsizdir
      (bedeli: dongu uzunlugu bilinmiyor, global bir varsayimla giriliyor).

Ikisinin orani gozlenen payin (share_of_wallet) tahminidir:
      sow_tahmini ~= hiz_akis / hiz_miktar
Bu, latent bir buyuklugu gozlemlenebilir iki tahmincinin oraniyla yakalama
denemesidir; ne kadar tuttugu reports/m2.md 3.1'de olculuyor.

Seyrek hucre problemi: uzun kuyrukta hucre basina 1-2 siparis var. Ham hucre
hizi tek bir siparise baglanir. Havuzlama (shrinkage) hucre hizini eczane x
kategori ortalamasina n/(n+k) agirligiyla ceker.
"""

from __future__ import annotations

import numpy as np

from core.config import Config


def gerileme_toplami(kumulatif: np.ndarray, hafta: int, pencere: int) -> np.ndarray:
    """[n, W] kumulatif matristen [n] pencere toplami (hafta dahil)."""
    bas = hafta - pencere + 1
    ust = kumulatif[:, hafta]
    return ust - kumulatif[:, bas - 1] if bas > 0 else ust


def gerileme_hizi(kumulatif: np.ndarray, hafta: int, pencere: int) -> np.ndarray:
    """Pencere basina ortalama haftalik adet. Kosu basinda pencere kirpilir."""
    etkin = min(pencere, hafta + 1)
    return gerileme_toplami(kumulatif, hafta, pencere) / etkin


def ewma_matrisi(matris: np.ndarray, alfa: float) -> np.ndarray:
    """[n, W] haftalik seriden [n, W] EWMA. t sutunu t dahil gecmisi kullanir."""
    cikti = np.empty_like(matris, dtype=np.float64)
    hafiza = np.zeros(matris.shape[0], dtype=np.float64)
    for w in range(matris.shape[1]):
        hafiza = alfa * matris[:, w] + (1.0 - alfa) * hafiza
        cikti[:, w] = hafiza
    return cikti


def havuzlanmis_hiz(
    ham_hiz: np.ndarray, gozlem_sayisi: np.ndarray, grup_idx: np.ndarray,
    guc: float,
) -> np.ndarray:
    """Hucre hizini grup ortalamasina dogru ceker (James-Stein tarzi shrinkage).

    agirlik = n / (n + guc); n = hucrenin gozlem (siparis) sayisi.
    Grup ortalamasi hacim agirlikli DEGIL, hucre basina alinir: birkac buyuk
    hucre grubu domine etmesin.
    """
    grup_sayisi = int(grup_idx.max()) + 1 if grup_idx.size else 0
    toplam = np.bincount(grup_idx, weights=ham_hiz, minlength=grup_sayisi)
    adet = np.bincount(grup_idx, minlength=grup_sayisi)
    grup_ort = np.where(adet > 0, toplam / np.maximum(adet, 1), 0.0)
    agirlik = gozlem_sayisi / (gozlem_sayisi + guc)
    return agirlik * ham_hiz + (1.0 - agirlik) * grup_ort[grup_idx]


def gozlenen_pay_tahmini(
    hiz_akis: np.ndarray, hiz_miktar: np.ndarray, eczane_idx: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """[n] hucre bazli akis/miktar oranindan [n] eczane bazli pay tahmini.

    Hucre bazinda oran cok gurultulu; eczane duzeyinde MEDYAN alinir
    (ortalama, siparis vermemis hucrelerin sifirlariyla cokerdi).
    """
    h = cfg.feature.hiz
    gecerli = (hiz_miktar > h.min_hiz) & (hiz_akis > 0)
    P = int(eczane_idx.max()) + 1 if eczane_idx.size else 0
    cikti = np.full(P, np.nan)
    if gecerli.any():
        oran = np.zeros_like(hiz_akis)
        oran[gecerli] = hiz_akis[gecerli] / hiz_miktar[gecerli]
        sira = np.argsort(eczane_idx[gecerli], kind="stable")
        e = eczane_idx[gecerli][sira]
        o = oran[gecerli][sira]
        sinirlar = np.searchsorted(e, np.arange(P + 1))
        for p in range(P):
            dilim = o[sinirlar[p]: sinirlar[p + 1]]
            if dilim.size:
                cikti[p] = float(np.median(dilim))
    genel = float(np.nanmedian(cikti)) if np.isfinite(cikti).any() else 1.0
    cikti = np.where(np.isfinite(cikti), cikti, genel)
    return np.clip(cikti, h.gozlenen_pay_tabani, 1.0)
