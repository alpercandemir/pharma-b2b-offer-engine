"""Eldeki stok tahmini - SPEC M2 "eldeki stok tahmini".

Defter (ledger) yaklasimi:
    stok(t) = clip( stok(t-1) + bize_gelen_sevkiyat(t) - tahmini_tuketim(t) )

Iki yapisal korluk var ve ikisi de duzeltilemez:
  1. Rakip depolardan gelen mal defterde YOKTUR. Eczaneye giren adedin
     ortalama %60'ini gormuyoruz.
  2. Tahmini tuketim de ayni sow ile olceklidir (features/hiz.py).

Ilginc sonuc - ORAN SADELESMESI: her iki taraf da yaklasik ayni carpanla
kucukse, seviye tahminleri yanlis ama ORANLARI dogru kalir:
    stok_tahmini / hiz_tahmini ~= (sow * stok_gercek) / (sow * hiz_gercek)
Yani mutlak stok cikarilamazken TUKENME SURESI cikarilabilir olmali.
`stok.varsayilan_gozlenen_pay` knob'u bu sadelesmenin gecerli olup olmadigini
sinamak icin var: degistirildiginde stok ve hiz seviyeleri degisir, tukenme
suresi degismemelidir. Ne kadar degismedigi reports/m2.md 3.2'de olculuyor.

Tavan: eczane order-up-to calisir, sonsuz stok tutmaz. Defter tavani
`stok.tavan_kapsama_hafta` x hiz'dir; olmadigi durumda tek bir buyuk siparis
defteri aylarca sisik birakiyor (m2.md 3.2).
"""

from __future__ import annotations

import numpy as np

from core.config import Config


def defter_stogu(sevkiyat: np.ndarray, hiz: np.ndarray, cfg: Config) -> np.ndarray:
    """[n, W] sevkiyat ve [n, W] hiz tahmininden [n, W] eldeki stok tahmini.

    Hiz haftalik guncellenir; defter hafta hafta ilerletilir. Point-in-time:
    t sutunu yalnizca <= t haftalarini kullanir.
    """
    s = cfg.feature.stok
    n, W = sevkiyat.shape
    telafi = 1.0 / s.varsayilan_gozlenen_pay
    stok = hiz[:, 0] * s.baslangic_kapsama_hafta
    cikti = np.empty((n, W), dtype=np.float64)
    for w in range(W):
        stok = np.maximum(0.0, stok + sevkiyat[:, w] * telafi - hiz[:, w] * telafi)
        # Order-up-to tavani: eczanenin elinde bu kadar haftalik maldan
        # fazlasi olamaz. Tavan, o hafta gelen sevkiyatin altina inemez.
        tavan = np.maximum(hiz[:, w] * telafi * s.tavan_kapsama_hafta,
                           sevkiyat[:, w] * telafi)
        cikti[:, w] = stok = np.minimum(stok, tavan)
    return cikti


def tukenme_haftasi(stok: np.ndarray, hiz: np.ndarray, cfg: Config,
                    ufuk: float) -> np.ndarray:
    """stok / hiz -> kac hafta sonra biter. Ufukta kirpilir.

    hiz `min_hiz`in altindaysa "olcemedik" demektir; sonuc ufka dayanir.
    """
    h = cfg.feature.hiz
    olculebilir = hiz >= h.min_hiz
    ham = np.where(olculebilir, stok / np.maximum(hiz, h.min_hiz), ufuk)
    return np.clip(ham, 0.0, ufuk)
