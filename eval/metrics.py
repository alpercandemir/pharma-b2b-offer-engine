"""M2 cikis kriteri metrikleri: MAE, kalibrasyon, ayirt etme.

SPEC M2 cikis kriteri: "Tahmin edilen tukenme gunu ile simulatorun gercek stok
sifirlanma gunu karsilastirmasi; MAE ve kalibrasyon egrisi."

Iki metrik ailesi ve ikisinin de neden gerekli oldugu:

  ZAMAN (MAE) - "kac gun sonra biter" sorusu.
      Sagdan sansur var: ufuk icinde tukenmeyen hucrenin gercek suresi
      bilinmiyor. Cozum kirpilmis (restricted) sure: her iki tarafta da
      min(T, ufuk) alinir. Boylece tum canli hucreler metrige girer ve
      "sadece hizli tukenenlere bakma" secim yanliligi olusmaz.
      Ayrica sansursuz alt kume icin ayri MAE raporlanir.

  KARAR (AUC / PR-AUC / kalibrasyon) - "onumuzdeki H hafta icinde tukenir mi".
      Teklif karari bu forma baglanacak (M4). Origin'ler W-H'den kucuk
      secildigi icin bu etikette sansur YOKTUR.

Kirpilmis MAE'nin tuzagi: ufuk icinde tukenme orani dusukse sabit tahmin
(hep "ufuk") guclu bir taban olur. Bu yuzden `sabit` tahminci metrik
tablosunda hep bulunur - onu gecemeyen bir model MAE'de bir sey ogrenmemistir.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def guvenli_auc(y: np.ndarray, skor: np.ndarray) -> float:
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, skor))


def guvenli_pr_auc(y: np.ndarray, skor: np.ndarray) -> float:
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    return float(average_precision_score(y, skor))


def brier(y: np.ndarray, olasilik: np.ndarray) -> float:
    return float(np.mean((olasilik - y) ** 2)) if y.size else float("nan")


def kalibrasyon_egrisi(y: np.ndarray, olasilik: np.ndarray, kova: int):
    """Esit frekansli kovalarda (ortalama tahmin, gozlenen oran, adet)."""
    if y.size == 0:
        return np.array([]), np.array([]), np.array([])
    sira = np.argsort(olasilik, kind="stable")
    parcalar = np.array_split(sira, min(kova, max(1, y.size)))
    tahmin, gozlem, adet = [], [], []
    for p in parcalar:
        if p.size == 0:
            continue
        tahmin.append(float(olasilik[p].mean()))
        gozlem.append(float(y[p].mean()))
        adet.append(int(p.size))
    return np.array(tahmin), np.array(gozlem), np.array(adet)


def beklenen_kalibrasyon_hatasi(y: np.ndarray, olasilik: np.ndarray, kova: int) -> float:
    tahmin, gozlem, adet = kalibrasyon_egrisi(y, olasilik, kova)
    if tahmin.size == 0:
        return float("nan")
    return float(np.sum(adet * np.abs(tahmin - gozlem)) / adet.sum())


# Beraberlik bozma tohumu. Sabit skorlu (bilgisiz) bir tahminci sirali veride
# argsort'un ilk satirlarini secer ve panelin origin sirasindan sahte bir
# kazanc devsirir. Tohum sabit oldugu icin olcum yine tekrar uretilebilir.
BERABERLIK_TOHUMU = 20260812


def ust_dilim_kazanci(y: np.ndarray, skor: np.ndarray, dilim: float = 0.10) -> float:
    """En riskli %10'daki olay orani / genel olay orani. 1.0 = sinyal yok."""
    if y.size == 0 or y.mean() == 0:
        return float("nan")
    k = max(1, int(round(dilim * y.size)))
    karistir = np.random.default_rng(BERABERLIK_TOHUMU).permutation(y.size)
    ust = karistir[np.argsort(-skor[karistir], kind="stable")[:k]]
    return float(y[ust].mean() / y.mean())


def mae(tahmin: np.ndarray, gercek: np.ndarray) -> float:
    return float(np.mean(np.abs(tahmin - gercek))) if tahmin.size else float("nan")


def yanlilik(tahmin: np.ndarray, gercek: np.ndarray) -> float:
    """Ortalama isaretli hata. Pozitif = model gec tukeniyor sanıyor."""
    return float(np.mean(tahmin - gercek)) if tahmin.size else float("nan")


def _grup_dilimleri(grup: np.ndarray) -> list[np.ndarray]:
    """Grup etiketlerinden grup basina satir indeksi listesi."""
    sira = np.argsort(grup, kind="stable")
    sirali = grup[sira]
    sinir = np.flatnonzero(np.r_[True, sirali[1:] != sirali[:-1]])
    return np.split(sira, sinir[1:])


def bootstrap_farki(y: np.ndarray, a: np.ndarray, b: np.ndarray, olcut,
                    tekrar: int, seed: int,
                    grup: np.ndarray | None = None) -> tuple[float, float, float]:
    """olcut(y, a) - olcut(y, b) icin (fark, %2.5, %97.5).

    Ayni satirlar ustunde eslesmis (paired) bootstrap: iki tahminci ayni
    orneklemi gordugu icin fark varyansi tek tek metriklerin varyansindan
    kucuktur. Aksi halde "ikisi de gurultulu, fark anlamsiz" denirdi.

    `grup` verilirse BLOK bootstrap yapilir: satirlar degil gruplar (hucreler)
    yeniden orneklenir. Panelde ayni hucrenin ardisik origin'leri buyuk olcude
    ortusuyor (origin araligi 2, ufuk 12 hafta -> %83 ortusme); satir bazli
    bootstrap bu satirlari bagimsiz sayar ve araligi OLDUGUNDAN DAR gosterir.
    Blok bootstrap bagimsizlik birimini dogru yere, hucreye koyar.
    """
    rng = np.random.default_rng(seed)
    temel = olcut(y, a) - olcut(y, b)
    if y.size == 0:
        return float("nan"), float("nan"), float("nan")
    dilimler = _grup_dilimleri(grup) if grup is not None else None
    farklar = np.empty(tekrar)
    for i in range(tekrar):
        if dilimler is None:
            idx = rng.integers(0, y.size, y.size)
        else:
            secim = rng.integers(0, len(dilimler), len(dilimler))
            idx = np.concatenate([dilimler[j] for j in secim])
        farklar[i] = olcut(y[idx], a[idx]) - olcut(y[idx], b[idx])
    gecerli = farklar[np.isfinite(farklar)]
    if gecerli.size == 0:
        return temel, float("nan"), float("nan")
    return temel, float(np.percentile(gecerli, 2.5)), float(np.percentile(gecerli, 97.5))
