"""Teklif satiri ozellikleri: CATE modelinin gordugu tek sey.

Point-in-time kurali M2/M3 ile ayni ve burada da testle sinaniyor
(tests/test_uplift.py::test_teklif_ozellikleri_point_in_time): origin'den
sonraki butun veri silindiginde ayni origin'in ozellik matrisi BIT BAZINDA
ayni kalmali.

NE VAR, NE YOK. Uplift'in gercek suruculeri (sim/response.py):

    MF duyarliligi   <- sosyoekonomik index    VAR
                        eczane olcegi          VAR
                        urun tipi              VAR
                        share_of_wallet        YOK (latent)
                        kisisel gurultu        YOK (latent)
    Vade duyarliligi <- vade_riski_skoru       VAR
                        dbs_limiti             VAR
                        stokculuk_egilimi      YOK (latent)
    Taban tepki      <- gercek stok / hiz      YOK -> `tahmini_kapsama_hafta`
                        share_of_wallet        YOK (latent)

Yani CATE ogrenilebilir ama TAM ogrenilemez. Bu bilincli: latent surucu
olmasaydi model tavana carpar ve "uplift modeli mukemmel calisiyor" gibi
yanlis bir sonuc uretirdik (CLAUDE.md 7).

Bu dosya ground_truth okumaz. features/okuma.py disinda bir kaynak yok.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from core.config import Config
from features.okuma import GozlemlenebilirKaynak
from features.panel import UZAK_GECMIS_HAFTA, _olay_matrisleri
from policy.candidates import AdayDunyasi, OriginGorunumu

# Sifira bolme korumasi. Sayisal sabit.
EPSILON = 1e-9
# Agac modelinde kategorik islenecek kolonlar.
KATEGORIK_ADLAR = frozenset({"ecz_il", "ecz_ciro_bandi", "sku_kategori",
                             "sku_urun_tipi", "ay"})


@dataclass
class TeklifDunyasi:
    """Aday dunyasinin ustune M4'un ihtiyac duydugu gozlemlenebilir tablolar.

    Fiyat haftalik degisiyor (referans kur gecisi, M1); teklif marji ve
    tepki modeli origin haftasinin fiyatiyla kurulmali. AdayDunyasi statik
    urun master'indaki fiyati tasiyordu - M3 icin yeterliydi, M4 icin degil.
    """

    aday: AdayDunyasi
    dsf_hafta: np.ndarray      # [S, W]
    psf_hafta: np.ndarray      # [S, W]
    referans_kur: np.ndarray   # [W]
    fiyat_endeksi: np.ndarray  # [W]
    ay: np.ndarray             # [W]
    ramazan: np.ndarray        # [W]
    yil_sonu: np.ndarray       # [W]
    olay_aktif: np.ndarray     # [S, W]
    olay_gecen: np.ndarray     # [S, W]
    kur_gecen: np.ndarray      # [W]
    kategori_kod: np.ndarray   # [S] tamsayi kod
    urun_tipi_kod: np.ndarray  # [S]
    il_kod: np.ndarray         # [P]
    ciro_kod: np.ndarray       # [P]


def _kod(seri: pl.Series) -> np.ndarray:
    duzey = {d: i for i, d in enumerate(sorted(seri.unique().to_list()))}
    return np.array([duzey[v] for v in seri.to_list()], dtype=float)


def teklif_dunyasi_yukle(kaynak: GozlemlenebilirKaynak, cfg: Config,
                         aday: AdayDunyasi) -> TeklifDunyasi:
    urun, ecz = aday.urunler, aday.eczaneler
    takvim = kaynak.tablo("takvim").sort("hafta")
    makro = kaynak.tablo("makro_haftalik").sort("hafta")
    fiyat = kaynak.tablo("urun_fiyat_haftalik")
    W, S = takvim.height, aday.S

    sira = {s: i for i, s in enumerate(urun["sku_id"].to_list())}
    idx = np.array([sira[s] for s in fiyat["sku_id"]])
    hafta = fiyat["hafta"].to_numpy().astype(int)
    dsf_h = np.zeros((S, W))
    psf_h = np.zeros((S, W))
    dsf_h[idx, hafta] = fiyat["dsf"].to_numpy()
    psf_h[idx, hafta] = fiyat["psf"].to_numpy()

    aktif, gecen, kur_gecen = _olay_matrisleri(kaynak.tablo("olaylar"), urun, W)
    return TeklifDunyasi(
        aday=aday, dsf_hafta=dsf_h, psf_hafta=psf_h,
        referans_kur=makro["referans_avro_kuru"].to_numpy(),
        fiyat_endeksi=makro["fiyat_endeksi"].to_numpy(),
        ay=takvim["ay"].to_numpy().astype(float),
        ramazan=takvim["ramazan_payi"].to_numpy().astype(float),
        yil_sonu=takvim["yil_sonu_stoklama"].to_numpy().astype(float),
        olay_aktif=aktif, olay_gecen=gecen, kur_gecen=kur_gecen,
        kategori_kod=_kod(urun["kategori_kod"]), urun_tipi_kod=_kod(urun["urun_tipi"]),
        il_kod=_kod(ecz["il"]), ciro_kod=_kod(ecz["aylik_ciro_bandi"]),
    )


def _hucre_gecmisi(dunya: AdayDunyasi, cfg: Config, t: int) -> dict[str, np.ndarray]:
    """[P, S] son siparis izleri. Yalnizca hafta <= t siparislerinden."""
    sec = dunya.sip_w <= t
    p, s = dunya.sip_p[sec], dunya.sip_s[sec]
    w, adet = dunya.sip_w[sec].astype(float), dunya.sip_adet[sec]

    son_hafta = np.full((dunya.P, dunya.S), -UZAK_GECMIS_HAFTA)
    np.maximum.at(son_hafta, (p, s), w)
    # Son siparisin adedi: en son haftaya ait satirlarin toplami.
    son_adet = np.zeros((dunya.P, dunya.S))
    sonda = w == son_hafta[p, s]
    np.add.at(son_adet, (p[sonda], s[sonda]), adet[sonda])
    return {"gecen_hafta_siparis": t - son_hafta, "son_siparis_adedi": son_adet}


def ozellik_matrisi(td: TeklifDunyasi, cfg: Config, gor: OriginGorunumu,
                    teklifler: pl.DataFrame,
                    uretici_skorlari: dict[str, np.ndarray]
                    ) -> tuple[np.ndarray, list[str], list[int]]:
    """[n, F] ozellik matrisi + ad listesi + kategorik kolon indeksleri."""
    d = td.aday
    t = gor.t
    p = teklifler["eczane_idx"].to_numpy()
    s = teklifler["sku_idx"].to_numpy()
    ecz, urun = d.eczaneler, d.urunler

    gecmis = _hucre_gecmisi(d, cfg, t)
    gecen = gecmis["gecen_hafta_siparis"][p, s]
    son_adet = gecmis["son_siparis_adedi"][p, s]
    akis = gor.akis_hizi[p, s]
    # Defter mantiginin ucuz hali (M2'nin `son_siparis_tukenme_hafta`si):
    # son siparisten bugune kadar tuketilmis olani dus, kalani hiza bol.
    kalan_adet = np.maximum(son_adet - akis * np.minimum(gecen, UZAK_GECMIS_HAFTA), 0.0)
    tahmini_kapsama = kalan_adet / np.maximum(akis, EPSILON)

    dbs = ecz["dbs_limiti"].to_numpy().astype(float)
    risk = ecz["vade_riski_skoru"].to_numpy().astype(float)
    kalan_limit = np.maximum(
        dbs * cfg.politika.kisit.kredi_kullanim_tavani
        * (1.0 - risk * cfg.politika.kisit.vade_riski_cezasi) - gor.acik_bakiye, 0.0)

    sku_hacim = gor.agirlikli_adet.sum(axis=0)
    sku_eczane = gor.ikili.sum(axis=0).astype(float)

    ozellik: dict[str, np.ndarray] = {
        # --- aday havuzundan gelenler ---
        "aday_skor": teklifler["skor"].to_numpy(),
        "aday_sira": teklifler["sira"].to_numpy().astype(float),
        "yeni_hucre": teklifler["yeni_hucre"].to_numpy().astype(float),
        "miad_baskisi": teklifler["miad_baskisi"].to_numpy(),
        "hiz_tahmini": teklifler["hiz_tahmini"].to_numpy(),
        "teklif_adedi": teklifler["teklif_adedi"].to_numpy().astype(float),
        "teklif_tutari": teklifler["teklif_tutari"].to_numpy(),
        "lot_kalan_gun": teklifler["lot_kalan_gun"].to_numpy().astype(float),
        "mf_izinli": teklifler["mf_izinli"].to_numpy().astype(float),
        **{f"uretici_{ad}": skor[p, s] for ad, skor in uretici_skorlari.items()},
        # --- hucre gecmisi ---
        "gecen_hafta_siparis": gecen,
        "son_siparis_adedi": son_adet,
        "akis_hizi": akis,
        "agirlikli_siparis_sayisi": gor.agirlikli_sayi[p, s],
        "tahmini_kalan_adet": kalan_adet,
        "tahmini_kapsama_hafta": tahmini_kapsama,
        # --- eczane ---
        "ecz_aylik_recete_adedi": ecz["aylik_recete_adedi"].to_numpy().astype(float)[p],
        "ecz_hastane_yakinligi_km": ecz["hastane_yakinligi_km"].to_numpy()[p],
        "ecz_sosyoekonomik": ecz["semt_sosyoekonomik_index"].to_numpy()[p],
        "ecz_turizm": ecz["turizm_bolgesi"].to_numpy()[p].astype(float),
        "ecz_nobet_rotasyon_gun": ecz["nobet_rotasyon_gun"].to_numpy().astype(float)[p],
        "ecz_sgk_recete_orani": ecz["sgk_recete_orani"].to_numpy()[p],
        "ecz_vade_riski": risk[p],
        "ecz_dbs_limiti": dbs[p],
        "ecz_acik_bakiye": gor.acik_bakiye[p],
        "ecz_kalan_kredi_limiti": kalan_limit[p],
        "ecz_siparis_satiri": gor.eczane_siparis_satiri[p],
        "ecz_il": td.il_kod[p],
        "ecz_ciro_bandi": td.ciro_kod[p],
        # --- urun ---
        "sku_dsf": td.dsf_hafta[s, t],
        "sku_psf": td.psf_hafta[s, t],
        "sku_depo_kar_marji": urun["depo_kar_marji"].to_numpy()[s],
        "sku_koli_ici_adet": urun["koli_ici_adet"].to_numpy().astype(float)[s],
        "sku_birim_hacim": urun["birim_hacim"].to_numpy()[s],
        "sku_soguk_zincir": urun["soguk_zincir"].to_numpy()[s].astype(float),
        "sku_sgk_geri_odeme": urun["sgk_geri_odeme"].to_numpy()[s].astype(float),
        "sku_promosyon_serbest": urun["promosyon_serbest"].to_numpy()[s].astype(float),
        "sku_hacim": sku_hacim[s],
        "sku_eczane_sayisi": sku_eczane[s],
        "sku_depo_stogu": gor.depo_stok[s],
        "sku_kategori": td.kategori_kod[s],
        "sku_urun_tipi": td.urun_tipi_kod[s],
        # --- zaman / makro ---
        "ay": np.full(p.size, td.ay[t]),
        "ramazan_payi": np.full(p.size, td.ramazan[t]),
        "yil_sonu_stoklama": np.full(p.size, td.yil_sonu[t]),
        "referans_avro_kuru": np.full(p.size, td.referans_kur[t]),
        "fiyat_endeksi": np.full(p.size, td.fiyat_endeksi[t]),
        "kur_olayindan_gecen_hafta": np.full(p.size, td.kur_gecen[t]),
        "olay_aktif": td.olay_aktif[s, t].astype(float),
        "olaydan_gecen_hafta": td.olay_gecen[s, t],
    }
    adlar = list(ozellik)
    X = np.column_stack([ozellik[ad] for ad in adlar]).astype(np.float32)
    kategorik = [i for i, ad in enumerate(adlar) if ad in KATEGORIK_ADLAR]
    return X, adlar, kategorik
