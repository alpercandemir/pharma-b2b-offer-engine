"""Takvim: hafta indeksi, ay, ramazan, yil sonu, nobet rotasyonu.

Mevsimsellik AY duzeyinde tanimli (config/products.yaml), hafta duzeyine
buradan tasinir. Bir haftanin ayi, haftanin persembesine (baslangic + 3 gun)
gore belirlenir: ay sinirindaki haftalarin tek bir aya atanmasi icin.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from core.config import Config

# Bir hafta 7 gundur. Domain sabiti.
GUN_HAFTA = 7
# Haftanin temsil gunu: persembe. Ay atamasini tek bir kurala baglar.
HAFTA_TEMSIL_GUN_OFSETI = 3


def hafta_tarihleri(baslangic: date, hafta_sayisi: int) -> list[date]:
    return [baslangic + timedelta(days=GUN_HAFTA * w) for w in range(hafta_sayisi)]


def takvim_kur(cfg: Config) -> pl.DataFrame:
    baslangic = cfg.profil.baslangic_tarihi
    W = cfg.profil.hafta_sayisi
    tarihler = hafta_tarihleri(baslangic, W)
    temsil = [t + timedelta(days=HAFTA_TEMSIL_GUN_OFSETI) for t in tarihler]

    ramazan_payi = []
    for t in tarihler:
        gunler = [t + timedelta(days=i) for i in range(GUN_HAFTA)]
        icinde = 0
        for g in gunler:
            for pencere in cfg.sim.takvim.ramazan_pencereleri:
                if pencere.baslangic <= g <= pencere.bitis:
                    icinde += 1
                    break
        ramazan_payi.append(icinde / GUN_HAFTA)

    yil_sonu_aylari = set(cfg.sim.takvim.yil_sonu_stoklama_aylari)

    return pl.DataFrame(
        {
            "hafta": np.arange(W, dtype=np.int32),
            "hafta_basi_tarih": tarihler,
            "yil": [t.year for t in temsil],
            "ay": np.array([t.month for t in temsil], dtype=np.int8),
            "ramazan_payi": np.array(ramazan_payi, dtype=np.float32),
            "yil_sonu_stoklama": np.array(
                [t.month in yil_sonu_aylari for t in temsil], dtype=bool
            ),
        }
    )


def nobet_gun_sayilari(
    periyot: np.ndarray, ofset: np.ndarray, hafta_sayisi: int
) -> np.ndarray:
    """[P, W] -> o hafta icindeki nobet gunu sayisi (0, 1 veya 2).

    Rotasyon periyodu 7 gun degilse nobet gunu haftadan haftaya kayar; bu
    haftalik seride gercek bir dalgalanma yaratir. Periyot 7 ise sabit 1 olur.
    """
    P = periyot.shape[0]
    gun_idx = np.arange(hafta_sayisi * GUN_HAFTA)
    nobet_gun = ((gun_idx[None, :] + ofset[:, None]) % periyot[:, None]) == 0
    return nobet_gun.reshape(P, hafta_sayisi, GUN_HAFTA).sum(axis=2).astype(np.float32)
