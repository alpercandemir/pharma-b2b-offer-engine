"""Urun evreni uretimi. SPEC 2.1.

Cikti iki parcali:
  - gozlemlenebilir urun master (fiyat, regulasyon bayraklari, lojistik)
  - latent populerlik (talep olcegi) -> ground_truth
"""

from __future__ import annotations

import numpy as np
import polars as pl

from core.config import Config, Kategori
from core.rng import SeedBankasi, agirlikli_secim

# ATC kodu son iki hanesi 01..99 arasindadir. Format sabiti.
ATC_SIRA_MAX = 99
_HARFLER = "ABCDEFGHJKLMNPR"


def _marj_kademesi(cfg: Config, psf: float) -> tuple[float, float]:
    for kademe in cfg.urun.marj_kademeleri:
        if kademe.psf_ust_siniri is None or psf <= kademe.psf_ust_siniri:
            return kademe.depo_marji, kademe.eczane_marji
    raise AssertionError("marj kademesi bulunamadi; son kademe null olmali")


def _promosyon_serbest(cfg: Config, urun_tipi: str, recete_rengi: str, sgk: bool) -> bool:
    kural = cfg.urun.promosyon_serbest_kurali
    if recete_rengi in kural.recete_rengi_vetosu:
        return False
    if urun_tipi in kural.urun_tipi_serbest:
        return True
    # Geriye RX kaliyor.
    return kural.rx_sgk_kapsaminda_serbest if sgk else kural.rx_sgk_disi_serbest


def _kategori_atamalari(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """SKU basina kategori indeksi; paylara gore deterministik kota + karistirma."""
    S = cfg.profil.sku_sayisi
    paylar = np.array([k.pay for k in cfg.urun.kategoriler], dtype=float)
    kotalar = np.floor(paylar * S).astype(int)
    # Yuvarlamadan artan SKU'lari en buyuk paylardan basla dagit.
    artan = S - kotalar.sum()
    for i in np.argsort(-paylar)[:artan]:
        kotalar[i] += 1
    idx = np.repeat(np.arange(len(paylar)), kotalar)
    rng.shuffle(idx)
    return idx


def urun_evreni_kur(cfg: Config, seedler: SeedBankasi) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = seedler.uretec("urun_evreni")
    S = cfg.profil.sku_sayisi
    kategoriler: list[Kategori] = cfg.urun.kategoriler
    kat_idx = _kategori_atamalari(cfg, rng)

    sku_id = [f"SKU{i:04d}" for i in range(S)]
    kategori_kod, atc_kodu, etken_madde = [], [], []
    urun_tipi, recete_rengi, sgk, tedarik_guclugu = [], [], [], []
    psf, dsf, depo_marji, eczane_marji = [], [], [], []
    soguk_zincir, koli, hacim, its = [], [], [], []

    # Kategori basina etken madde havuzu: ayni INN'i paylasan SKU'lar esdeger grubu.
    inn_havuzu: dict[str, list[str]] = {}
    for k_i, kat in enumerate(kategoriler):
        n_kat = int((kat_idx == k_i).sum())
        n_inn = max(1, int(round(n_kat * kat.etken_madde_orani)))
        inn_havuzu[kat.kod] = [f"{kat.kod}-INN-{j:02d}" for j in range(n_inn)]

    koli_secim = agirlikli_secim(
        rng, cfg.urun.evren.koli_ici_adet_secenekleri,
        cfg.urun.evren.koli_ici_adet_agirliklari, S,
    )

    for i in range(S):
        kat = kategoriler[kat_idx[i]]
        kategori_kod.append(kat.kod)

        if kat.atc_prefix is None:
            atc_kodu.append(None)
        else:
            alt = _HARFLER[rng.integers(0, len(_HARFLER))]
            kim = _HARFLER[rng.integers(0, len(_HARFLER))]
            sira = int(rng.integers(1, ATC_SIRA_MAX + 1))
            atc_kodu.append(f"{kat.atc_prefix}{alt}{kim}{sira:02d}")

        havuz = inn_havuzu[kat.kod]
        etken_madde.append(havuz[int(rng.integers(0, len(havuz)))])

        tipler = list(kat.urun_tipi_dagilimi)
        t = tipler[int(rng.choice(len(tipler), p=[kat.urun_tipi_dagilimi[x] for x in tipler]))]
        urun_tipi.append(t)

        renkler = list(kat.recete_rengi_dagilimi)
        if t == "RX":
            r = renkler[int(rng.choice(len(renkler), p=[kat.recete_rengi_dagilimi[x] for x in renkler]))]
        else:
            r = "NORMAL"  # recete rengi sadece receteli urunde tanimli
        recete_rengi.append(r)

        s_geri_odeme = bool(rng.random() < kat.sgk_olasiligi) if t == "RX" else False
        sgk.append(s_geri_odeme)
        tedarik_guclugu.append(bool(rng.random() < kat.tedarik_guclugu_olasiligi))

        p = float(np.exp(rng.normal(kat.fiyat_log_ort, kat.fiyat_log_sigma)))
        dm, em = _marj_kademesi(cfg, p)
        psf.append(round(p, 2))
        dsf.append(round(p * (1.0 - em), 2))
        depo_marji.append(dm)
        eczane_marji.append(em)

        sz = bool(rng.random() < kat.soguk_zincir_olasiligi)
        soguk_zincir.append(sz)
        koli.append(int(koli_secim[i]))
        hacim.append(
            float(np.exp(rng.normal(cfg.urun.evren.birim_hacim_log_ort,
                                    cfg.urun.evren.birim_hacim_log_sigma)))
        )
        its_p = cfg.urun.evren.its_olasiligi_rx if t == "RX" else cfg.urun.evren.its_olasiligi_rx_disi
        its.append(bool(rng.random() < its_p))

    promosyon = [
        _promosyon_serbest(cfg, urun_tipi[i], recete_rengi[i], sgk[i]) for i in range(S)
    ]

    urunler = pl.DataFrame(
        {
            "sku_id": sku_id,
            "kategori_kod": kategori_kod,
            "atc_kodu": atc_kodu,
            "etken_madde": etken_madde,
            "urun_tipi": urun_tipi,
            "recete_rengi": recete_rengi,
            "sgk_geri_odeme": sgk,
            "titck_tedarik_guclugu": tedarik_guclugu,
            "promosyon_serbest": promosyon,
            "psf": psf,
            "dsf": dsf,
            "kdv_orani": [cfg.urun.evren.kdv_orani] * S,
            "depo_kar_marji": depo_marji,
            "eczane_kar_marji": eczane_marji,
            "soguk_zincir": soguk_zincir,
            "koli_ici_adet": koli,
            "birim_hacim": [round(h, 4) for h in hacim],
            "its_serilestirilmis": its,
        }
    )

    # Latent: SKU hacim olcegi. Uzun kuyruk buradan gelir.
    populerlik = np.exp(rng.normal(0.0, cfg.urun.evren.populerlik_log_sigma, S))
    populerlik = populerlik / populerlik.mean()
    latent = pl.DataFrame({"sku_id": sku_id, "latent_populerlik": populerlik})

    return urunler, latent
