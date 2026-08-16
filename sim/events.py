"""Rejim / olay katmani. SPEC 2.4.

Uc ayri kanal uretir; hangisinin hangi olaydan etkilendigi kasitlidir:

  tuketim_carpani [S, W]  -> hastanin tuketimi (epidemi, geri cekme)
  antisipasyon    [S, W]  -> ECZANENIN siparis stoklamasi (referans kur, yil sonu)
  ikmal_blok      [S, W]  -> BIZIM ikmalimiz (tedarik krizi, geri cekme)

Bu ayrim D2/D4'un ogretmek istedigi seyin ta kendisi: referans kur
guncellemesinde TUKETIM DEGISMEZ, sadece siparis one cekilir. Siparis
serisinden tuketim hizi cikarmaya calisan model tam burada yanilir.

D4 / leakage: antisipasyon penceresi olayin yururlugunden ONCE baslar ve o
pencerede kamuya duyuru yoktur. Gozlemlenebilir olay tablosuna olay ancak
yururluk haftasinda dusdur (gorunur_hafta == yururluk_hafta). Antisipasyon
baslangici sadece ground_truth'a yazilir.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from core.config import Config
from core.rng import SeedBank

# Antisipasyon penceresi icinde siddet dogrusal olarak yukselir: olaya bir
# hafta kala tam siddet. Rampanin sekli domain kararidir, seviye knob'dur.
ANTISIPASYON_RAMPA_MIN = 0.25


class OlayDunyasi:
    def __init__(
        self,
        tuketim_carpani: np.ndarray,
        antisipasyon: np.ndarray,
        ikmal_blok: np.ndarray,
        referans_kur: np.ndarray,
        fiyat_endeksi: np.ndarray,
        gozlemlenebilir: pl.DataFrame,
        gercek: pl.DataFrame,
    ) -> None:
        self.tuketim_carpani = tuketim_carpani
        self.antisipasyon = antisipasyon
        self.ikmal_blok = ikmal_blok
        self.referans_kur = referans_kur
        self.fiyat_endeksi = fiyat_endeksi
        self.gozlemlenebilir = gozlemlenebilir
        self.gercek = gercek


def _olusum_haftalari(rng, W: int, min_ara: int, max_ara: int) -> list[int]:
    haftalar = []
    t = int(rng.integers(0, max(1, max_ara)))
    while t < W:
        haftalar.append(t)
        t += int(rng.integers(min_ara, max_ara + 1))
    return haftalar


def olaylari_uret(
    cfg: Config,
    seedler: SeedBank,
    urunler: pl.DataFrame,
    latent_populerlik: np.ndarray,
    takvim: pl.DataFrame,
) -> OlayDunyasi:
    rng = seedler.generator("olaylar")
    S = cfg.profil.sku_sayisi
    W = cfg.profil.hafta_sayisi

    tuketim = np.ones((S, W), dtype=np.float64)
    antisipasyon = np.zeros((S, W), dtype=np.float64)
    ikmal_blok = np.zeros((S, W), dtype=bool)

    kategori_kod = urunler["kategori_kod"].to_numpy()
    etken = urunler["etken_madde"].to_numpy()
    akut_kodlar = {k.kod for k in cfg.urun.kategoriler if k.akut}

    satirlar: list[dict] = []
    kur_artislari: list[tuple[int, float]] = []

    for tanim in cfg.olay.olaylar:
        for yururluk in _olusum_haftalari(rng, W, tanim.min_ara_hafta, tanim.max_ara_hafta):
            sure = int(rng.integers(tanim.sure_hafta_min, tanim.sure_hafta_max + 1))
            antic = int(rng.integers(tanim.antisipasyon_hafta_min,
                                     tanim.antisipasyon_hafta_max + 1))

            if tanim.kapsam == "GLOBAL":
                hedef_idx = np.arange(S)
                hedef_ad = "GLOBAL"
            elif tanim.kapsam == "KATEGORI_AKUT":
                kod = sorted(akut_kodlar)[int(rng.integers(0, len(akut_kodlar)))]
                hedef_idx = np.flatnonzero(kategori_kod == kod)
                hedef_ad = kod
            else:  # SKU
                n = max(1, int(round(tanim.etkilenen_sku_orani * S)))
                hedef_idx = rng.choice(S, size=n, replace=False)
                hedef_ad = ",".join(sorted(urunler["sku_id"].to_numpy()[hedef_idx]))

            if hedef_idx.size == 0:
                continue

            bitis = min(W, yururluk + sure)
            if yururluk < W:
                # tuketim etkisi
                if tanim.tuketim_carpani != 1.0:
                    tuketim[np.ix_(hedef_idx, np.arange(yururluk, bitis))] *= tanim.tuketim_carpani
                # kalici seviye kaymasi (SGK listesine girme/cikma)
                if tanim.kalici_seviye_kaymasi != 0.0:
                    yon = 1.0 if rng.random() < tanim.kalici_kayma_yukari_olasiligi else -1.0
                    tuketim[np.ix_(hedef_idx, np.arange(yururluk, W))] *= (
                        1.0 + yon * tanim.kalici_seviye_kaymasi
                    )
                # ikmal blogu
                if tanim.ikmal_bloklar:
                    ikmal_blok[np.ix_(hedef_idx, np.arange(yururluk, bitis))] = True
                # geri cekmede ayni etken maddedeki urunlere kayma
                if tanim.tip == "TITCK_GERI_CEKME":
                    _ikame_uygula(
                        tuketim, hedef_idx, etken, latent_populerlik,
                        yururluk, bitis, cfg.olay.ikame.geri_cekmede_ikame_orani,
                    )
                if tanim.tip == "REFERANS_KUR_GUNCELLEME":
                    artis = float(rng.normal(cfg.olay.referans_kur.guncelleme_artis_ort,
                                             cfg.olay.referans_kur.guncelleme_artis_sigma))
                    kur_artislari.append((yururluk, max(0.0, artis)))

            # antisipasyon: yururlukten ONCE, duyurusuz
            antic_bas = max(0, yururluk - antic)
            if antic > 0 and antic_bas < yururluk:
                pencere = np.arange(antic_bas, min(yururluk, W))
                if pencere.size:
                    rampa = np.linspace(ANTISIPASYON_RAMPA_MIN, 1.0, pencere.size)
                    ek = tanim.antisipasyon_siddeti * rampa
                    blok = antisipasyon[np.ix_(hedef_idx, pencere)]
                    antisipasyon[np.ix_(hedef_idx, pencere)] = np.maximum(blok, ek[None, :])

            satirlar.append(
                {
                    "olay_id": f"OLY{len(satirlar):04d}",
                    "tip": tanim.tip,
                    "kapsam": tanim.kapsam,
                    "hedef": hedef_ad,
                    "yururluk_hafta": yururluk,
                    "bitis_hafta": bitis,
                    "gorunur_hafta": yururluk,
                    "antisipasyon_baslangic_hafta": antic_bas,
                    "antisipasyon_siddeti": tanim.antisipasyon_siddeti,
                }
            )

    # Yil sonu stoklama beklentisi: olay degil, takvim kaynakli antisipasyon.
    yil_sonu = takvim["yil_sonu_stoklama"].to_numpy()
    if yil_sonu.any():
        antisipasyon[:, yil_sonu] = np.maximum(
            antisipasyon[:, yil_sonu], cfg.sim.takvim.yil_sonu_stoklama_yogunlugu
        )

    referans_kur, fiyat_endeksi = _kur_ve_fiyat(cfg, W, kur_artislari)

    gercek = pl.DataFrame(satirlar) if satirlar else pl.DataFrame(
        schema={"olay_id": pl.Utf8, "tip": pl.Utf8, "kapsam": pl.Utf8, "hedef": pl.Utf8,
                "yururluk_hafta": pl.Int64, "bitis_hafta": pl.Int64, "gorunur_hafta": pl.Int64,
                "antisipasyon_baslangic_hafta": pl.Int64, "antisipasyon_siddeti": pl.Float64}
    )
    gozlemlenebilir = gercek.drop(["antisipasyon_baslangic_hafta", "antisipasyon_siddeti"])

    return OlayDunyasi(
        tuketim_carpani=tuketim, antisipasyon=antisipasyon, ikmal_blok=ikmal_blok,
        referans_kur=referans_kur, fiyat_endeksi=fiyat_endeksi,
        gozlemlenebilir=gozlemlenebilir, gercek=gercek,
    )


def _ikame_uygula(tuketim, geri_cekilen_idx, etken, populerlik, bas, bit, oran) -> None:
    """Geri cekilen urunun kaybolan talebi ayni etken maddedeki urunlere kayar."""
    for s in geri_cekilen_idx:
        kardesler = np.flatnonzero((etken == etken[s]) & (np.arange(len(etken)) != s))
        if kardesler.size == 0:
            continue
        kardes_hacim = populerlik[kardesler].sum()
        if kardes_hacim <= 0:
            continue
        carpan = 1.0 + oran * populerlik[s] / kardes_hacim
        tuketim[np.ix_(kardesler, np.arange(bas, bit))] *= carpan


def _kur_ve_fiyat(cfg: Config, W: int, artislar: list[tuple[int, float]]):
    rk = cfg.olay.referans_kur
    kur = np.full(W, rk.baslangic_deger, dtype=np.float64)
    endeks = np.ones(W, dtype=np.float64)
    for hafta, artis in sorted(artislar):
        kur[hafta:] *= 1.0 + artis
        gecikmeli = min(W, hafta + rk.fiyat_gecis_gecikme_hafta)
        endeks[gecikmeli:] *= 1.0 + rk.fiyat_gecis_katsayisi * artis
    return kur, endeks
