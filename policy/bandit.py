"""Kayit politikasi ve propensity loglama (D7).

D7: "Her gosterimde secim olasiligi (propensity) loglanir. Loglanmazsa
off-policy evaluation imkansiz; her degisiklik canli A/B'ye mahkum."

M4'te bu dosyanin isi, egitim verisini ureten politikayi TANIMLAMAK ve her
satir icin pi(a|x)'i tam olarak yazmaktir. Uc parcali bir karisim:

    pi(teklif yok | x) = 1 - q(x)
    pi(a | x)          = q(x) * [ eps / |izinli| + (1-eps) * w_a / sum_b w_b ]

    q(x) : teklif verme olasiligi; aday skoruyla artar. SIFIR EGILIMLI
           DEGIL -- gercek loglar hep bir politikadan gelir ve o politika
           yuksek skorlu satirlara daha cok teklif verir. Confounding'i
           bilerek uretiyoruz: ogreticinin isi bunu duzeltmek, veri toplama
           bicimini secmek degil.
    w_a  : eski saha kurali -- "riskli / kucuk eczaneye daha derin MF ve
           daha uzun vade". Kollar arasi orneklem dengesizligini bu uretir
           ve X-ogrenicinin T'ye gore avantajinin sinandigi yer burasidir.
    eps  : kesif payi. VARLIK SEBEBI OVERLAP: eps > 0 oldugu surece her
           izinli kolun propensity'si pozitiftir. eps = 0 olsaydi bazi
           (satir, kol) ciftleri hic gozlenmez, o bolgede karsi-olgusal
           tahmin ekstrapolasyon olurdu ve M6'nin IPS/DR tahmincisi
           tanimsiz kalirdi. config/uplift.yaml bunu yuklemede zorluyor.

Thompson sampling (SPEC 3'te bu dosyaya yazili) M6'nin isi: posterior
guncellemesi kapali dongu rollout'u gerektiriyor, M4'te dongu yok.

Bu dosya ground_truth okumaz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from core.config import Config
from core.rng import SeedBank
from policy.scorer import TEKLIF_YOK, AksiyonUzayi

# Teklif verme olasiliginin kirpma sinirlari. Ne 0 ne 1 olabilir: 0'da o
# satir hic teklif gormez (kol destegi kaybolur), 1'de kontrol grubu kaybolur
# ve taban tepki hic olculemez. Olcum sabiti, tuning kadrani degil.
Q_ALT, Q_UST = 0.05, 0.95
MIN_SAPMA = 1e-6


@dataclass
class KayitCiktisi:
    kol: np.ndarray           # [n] secilen kol
    propensity: np.ndarray    # [n] pi(secilen kol | x)
    pi: np.ndarray            # [n, A] tum kollarin olasiligi


def _yuzdelik(v: np.ndarray) -> np.ndarray:
    """[0, 1] yuzdelik sira. Skor olcegi ureticiye gore degisir; sira degismez."""
    if v.size == 0:
        return v
    sira = np.argsort(np.argsort(v, kind="stable"))
    return sira / max(v.size - 1, 1)


def kayit_olasiliklari(dunya, cfg: Config, uzay: AksiyonUzayi,
                       teklifler: pl.DataFrame, izinli: np.ndarray) -> np.ndarray:
    """[n, A] kayit politikasinin olasilik matrisi. Satirlar 1.0'a toplanir."""
    k = cfg.uplift.kayit
    n, A = teklifler.height, uzay.A
    if n == 0:
        return np.zeros((0, A))

    q = np.clip(k.teklif_taban_olasiligi
                + k.skor_egilimi * (_yuzdelik(teklifler["skor"].to_numpy()) - 0.5),
                Q_ALT, Q_UST)

    # Eski saha kurali: riskli ve kucuk eczaneye daha derin MF / uzun vade.
    p_idx = teklifler["eczane_idx"].to_numpy()
    ecz = dunya.eczaneler
    risk = ecz["vade_riski_skoru"].to_numpy().astype(float)
    olcek = np.log(ecz["aylik_recete_adedi"].to_numpy().astype(float))
    z = lambda v: (v - v.mean()) / max(float(v.std()), MIN_SAPMA)  # noqa: E731
    tercih = np.tanh(z(risk) - z(olcek))[p_idx]

    mf_norm = uzay.mf / max(float(uzay.mf.max()), MIN_SAPMA)
    vade_ara = float(uzay.vade.max() - uzay.vade.min())
    vade_norm = (uzay.vade - uzay.vade.min()) / max(vade_ara, MIN_SAPMA)
    kol_derinligi = mf_norm + vade_norm                       # [A]

    agirlik = np.exp(k.derin_mf_egilimi * tercih[:, None] * kol_derinligi[None, :])
    agirlik[:, TEKLIF_YOK] = 0.0
    agirlik = np.where(izinli, agirlik, 0.0)
    toplam = agirlik.sum(axis=1, keepdims=True)

    duzgun = np.where(izinli, 1.0, 0.0)
    duzgun[:, TEKLIF_YOK] = 0.0
    duzgun_toplam = duzgun.sum(axis=1, keepdims=True)

    pi = np.zeros((n, A))
    acik = (toplam[:, 0] > 0) & (duzgun_toplam[:, 0] > 0)
    karisim = (k.kesif_orani * duzgun / np.maximum(duzgun_toplam, 1.0)
               + (1.0 - k.kesif_orani) * agirlik / np.maximum(toplam, MIN_SAPMA))
    pi[acik] = (q[acik, None] * karisim[acik])
    pi[:, TEKLIF_YOK] = 1.0 - pi[:, 1:].sum(axis=1)
    return pi


def kayit_kosusu(dunya, cfg: Config, seedler: SeedBank, uzay: AksiyonUzayi,
                 teklifler: pl.DataFrame, izinli: np.ndarray,
                 t: int) -> KayitCiktisi:
    """Bir origin'de kayit politikasini kosar ve propensity'yi loglar."""
    pi = kayit_olasiliklari(dunya, cfg, uzay, teklifler, izinli)
    n = pi.shape[0]
    if n == 0:
        return KayitCiktisi(np.zeros(0, dtype=np.int32), np.zeros(0), pi)
    # Asama adi `kayit.seed`i tasir: knob degisince loglanan aksiyonlar
    # degisir, dunyanin cekilis akisi degismez (core/rng.py).
    rng = seedler.generator(f"kayit_politikasi_{cfg.uplift.kayit.seed}_{t}")
    kumulatif = np.cumsum(pi, axis=1)
    kol = (rng.random(n)[:, None] > kumulatif).sum(axis=1).astype(np.int32)
    kol = np.minimum(kol, uzay.A - 1)
    return KayitCiktisi(kol=kol, propensity=pi[np.arange(n), kol], pi=pi)
