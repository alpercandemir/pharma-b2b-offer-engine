"""Eczane evreni ve persona uretimi. SPEC 2.2.

Gozlemlenebilir / latent ayrimi burada baslar:
  gozlemlenebilir : konum, hastane yakinligi, sosyoekonomik index, turizm,
                    nobet rotasyonu, ciro bandi, recete adedi, kredi.
  latent          : share_of_wallet, stokculuk_egilimi, miad_toleransi_gun,
                    kapsama hedefi, gozden gecirme periyodu, buyukluk,
                    kategori affinitesi.

CLAUDE.md 7: share_of_wallet latent kalir. Sistemin en buyuk belirsizligi bu.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from core.config import Config
from core.rng import SeedBankasi, agirlikli_secim

# Sosyoekonomik index [0, 1] araliginda uretilir; affinite formulunde [-1, +1]
# olceginde kullanilir (config/pharmacies.yaml basindaki formul). Donusum sabiti.
SOSYO_MERKEZ = 0.5
SOSYO_OLCEK = 2.0


class EczaneEvreni:
    def __init__(
        self,
        master: pl.DataFrame,
        latent: pl.DataFrame,
        affinite: np.ndarray,
        buyukluk: np.ndarray,
        sow0: np.ndarray,
        stokculuk: np.ndarray,
        miad_toleransi: np.ndarray,
        kapsama_hafta: np.ndarray,
        gozden_gecirme: np.ndarray,
        gozden_gecirme_ofset: np.ndarray,
        nobet_periyot: np.ndarray,
        nobet_ofset: np.ndarray,
        turizm: np.ndarray,
    ) -> None:
        self.master = master
        self.latent = latent
        self.affinite = affinite          # [P, K] log-affinite -> exp'lenmis
        self.buyukluk = buyukluk
        self.sow0 = sow0
        self.stokculuk = stokculuk
        self.miad_toleransi = miad_toleransi
        self.kapsama_hafta = kapsama_hafta
        self.gozden_gecirme = gozden_gecirme
        self.gozden_gecirme_ofset = gozden_gecirme_ofset
        self.nobet_periyot = nobet_periyot
        self.nobet_ofset = nobet_ofset
        self.turizm = turizm


def eczane_evreni_kur(cfg: Config, seedler: SeedBankasi) -> EczaneEvreni:
    rng = seedler.uretec("eczane_evreni")
    P = cfg.profil.eczane_sayisi
    ec = cfg.eczane

    il_paylari = np.array([i.pay for i in ec.iller], dtype=float)
    il_idx = rng.choice(len(ec.iller), size=P, p=il_paylari / il_paylari.sum())
    il_ad = [ec.iller[i].ad for i in il_idx]
    ilce_no = rng.integers(1, ec.cografya.ilce_sayisi_il_basina + 1, size=P)
    semt_no = rng.integers(1, ec.cografya.semt_sayisi_ilce_basina + 1, size=P)

    turizm = np.array(
        [rng.random() < ec.iller[i].turizm_olasiligi for i in il_idx], dtype=bool
    )
    sosyo = np.clip(
        np.array([rng.normal(ec.iller[i].sosyoekonomik_ort,
                             ec.iller[i].sosyoekonomik_sigma) for i in il_idx]),
        0.0, 1.0,
    )

    mesafe_km = np.exp(rng.normal(ec.konum.hastane_mesafesi_log_ort,
                                  ec.konum.hastane_mesafesi_log_sigma, P))
    yakinlik = np.exp(-mesafe_km / ec.konum.mesafe_olcegi_km)

    buyukluk = np.exp(rng.normal(0.0, ec.olcek.buyukluk_log_sigma, P))
    buyukluk = buyukluk / buyukluk.mean()
    aylik_recete = np.maximum(1, np.round(ec.olcek.aylik_recete_taban * buyukluk)).astype(int)
    sinirlar = np.array(ec.olcek.ciro_bandi_sinirlari)
    bant_adlari = np.array(["S", "M", "L", "XL"])
    ciro_bandi = bant_adlari[np.searchsorted(sinirlar, buyukluk, side="right")]

    nobet_periyot = agirlikli_secim(
        rng, ec.nobet.rotasyon_periyodu_gun_secenekleri,
        ec.nobet.rotasyon_periyodu_agirliklari, P,
    ).astype(np.int64)
    nobet_ofset = rng.integers(0, nobet_periyot)

    vade_riski = np.clip(rng.normal(ec.kredi.vade_riski_ort, ec.kredi.vade_riski_sigma, P), 0.0, 1.0)
    dbs_carpan = np.maximum(
        ec.kredi.dbs_limiti_carpani_min,
        rng.normal(ec.kredi.dbs_limiti_carpani_ort, ec.kredi.dbs_limiti_carpani_sigma, P)
    )
    dbs_limiti = np.round(aylik_recete * ec.kredi.ortalama_recete_tutari * dbs_carpan, 2)
    sgk_orani = np.clip(
        rng.normal(ec.recete_karmasi.sgk_recete_orani_ort,
                   ec.recete_karmasi.sgk_recete_orani_sigma, P), 0.0, 1.0
    )

    master = pl.DataFrame(
        {
            "eczane_id": [f"ECZ{i:04d}" for i in range(P)],
            "il": il_ad,
            "ilce": [f"{a}-ILCE{n}" for a, n in zip(il_ad, ilce_no)],
            "semt": [f"{a}-ILCE{n}-SEMT{s}" for a, n, s in zip(il_ad, ilce_no, semt_no)],
            "hastane_yakinligi_km": np.round(mesafe_km, 3),
            "semt_sosyoekonomik_index": np.round(sosyo, 4),
            "turizm_bolgesi": turizm,
            "nobet_rotasyon_gun": nobet_periyot,
            "nobet_rotasyon_ofset": nobet_ofset,
            "aylik_ciro_bandi": ciro_bandi,
            "aylik_recete_adedi": aylik_recete,
            "vade_riski_skoru": np.round(vade_riski, 4),
            "dbs_limiti": dbs_limiti,
            "sgk_recete_orani": np.round(sgk_orani, 4),
        }
    )

    # --- latent persona ---
    sow0 = np.clip(
        rng.beta(ec.latent_share_of_wallet.beta_a, ec.latent_share_of_wallet.beta_b, P),
        ec.latent_share_of_wallet.min, ec.latent_share_of_wallet.max,
    )
    stokculuk = np.minimum(
        np.exp(rng.normal(ec.latent_stokculuk.log_ort, ec.latent_stokculuk.log_sigma, P)),
        ec.latent_stokculuk.ust_sinir,
    )
    miad_tol = np.clip(
        rng.normal(ec.latent_miad_toleransi.taban_gun_ort,
                   ec.latent_miad_toleransi.taban_gun_sigma, P),
        ec.latent_miad_toleransi.min_gun, ec.latent_miad_toleransi.max_gun,
    )
    sd = ec.latent_siparis_davranisi
    kapsama = np.maximum(
        sd.kapsama_hafta_min,
        rng.normal(sd.kapsama_hafta_ort, sd.kapsama_hafta_sigma, P),
    )
    gozden_gecirme = agirlikli_secim(
        rng, sd.gozden_gecirme_periyodu_secenekleri,
        sd.gozden_gecirme_periyodu_agirliklari, P,
    ).astype(np.int64)
    gozden_gecirme_ofset = rng.integers(0, gozden_gecirme)

    # Kategori affinitesi: hastane yakinligi Rx miksini, sosyoekonomik index
    # dermokozmetik/TEG payini, turizm yaz kategorilerini surukler (SPEC 2.2).
    kodlar = [k.kod for k in cfg.urun.kategoriler]
    K = len(kodlar)
    log_aff = np.zeros((P, K))
    for j, kod in enumerate(kodlar):
        satir = ec.kategori_egilimi.tablo[kod]
        log_aff[:, j] = (
            satir.taban
            + satir.hastane_kats * yakinlik
            + satir.sosyo_kats * (sosyo - SOSYO_MERKEZ) * SOSYO_OLCEK
            + satir.turizm_kats * turizm.astype(float)
            + rng.normal(0.0, ec.kategori_egilimi.gurultu_sigma, P)
        )
    affinite = np.exp(log_aff)

    latent = pl.DataFrame(
        {
            "eczane_id": master["eczane_id"],
            "share_of_wallet": np.round(sow0, 5),
            "stokculuk_egilimi": np.round(stokculuk, 5),
            "miad_toleransi_gun": np.round(miad_tol, 2),
            "kapsama_hafta": np.round(kapsama, 4),
            "gozden_gecirme_periyodu": gozden_gecirme,
            "latent_buyukluk": np.round(buyukluk, 5),
            **{f"latent_affinite_{kod}": np.round(affinite[:, j], 5)
               for j, kod in enumerate(kodlar)},
        }
    )

    return EczaneEvreni(
        master=master, latent=latent, affinite=affinite, buyukluk=buyukluk,
        sow0=sow0, stokculuk=stokculuk, miad_toleransi=miad_tol,
        kapsama_hafta=kapsama, gozden_gecirme=gozden_gecirme,
        gozden_gecirme_ofset=gozden_gecirme_ofset,
        nobet_periyot=nobet_periyot, nobet_ofset=nobet_ofset, turizm=turizm,
    )
