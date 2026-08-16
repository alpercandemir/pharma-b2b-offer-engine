"""Point-in-time panel: (hucre, origin) satirlari + gozlemlenebilir etiket.

Origin = tahmin ani (bir haftanin sonu). Kural:
    ozellikler  yalnizca hafta <= origin verisini kullanir,
    etiket      yalnizca hafta >  origin verisini kullanir.
Leakage guard bunu iki testle sinar (tests/test_features.py):
  - gelecek silinince ayni origin'in ozellik satirlari BIT BAZINDA ayni kalmali
  - features/ altinda ground_truth okuyan hicbir yol olmamali

Etiket (gozlemlenebilir): origin'den sonraki ilk BIZE SIPARIS haftasi.
Bu, gercek tukenmenin bozuk bir izdusumudur:
  - eczane stok sifirlanmadan ONCE siparis verir (emniyet stogu)  -> erken
  - siparisin sadece sow kadari bize gelir                        -> gec
Iki yanlilik ters yonlu; net etkisi olculuyor (reports/m2.md 3.3).

Bu dosya ground_truth/ okumaz. Gercek tukenme etiketi eval/oracle.py'de.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from core.config import Config
from features.hiz import (ewma_matrisi, gerileme_hizi, gerileme_toplami,
                          gozlenen_pay_tahmini, havuzlanmis_hiz)
from features.okuma import GozlemlenebilirKaynak
from features.stok import defter_stogu, tukenme_haftasi
from sim.calendar import GUN_HAFTA

# Fiyat degisim ozelliginin gerileme penceresi (hafta). Referans kur gecisinin
# fiyata yansimasi 1-2 hafta surer; bir ceyreklik pencere bunu yakalar.
FIYAT_DEGISIM_PENCERESI = 13
# "Hic olmadi" durumunda gecen-hafta ozelliklerine yazilan deger. NaN degil,
# cunku "cok uzun zaman once" ile "hic" ayni yonde okunmali; agac modeli buyuk
# sayiyi dogal olarak en uc kovaya koyar. Olcek: kosu uzunlugundan buyuk olmak
# zorunda degil, siralamayi bozmamasi yeterli.
UZAK_GECMIS_HAFTA = 999.0


@dataclass
class Izgara:
    """Gozlemlenebilir veriden kurulmus hucre x hafta izgarasi."""

    eczane_id: np.ndarray
    sku_id: np.ndarray
    eczane_idx: np.ndarray
    sku_idx: np.ndarray
    W: int
    talep: np.ndarray            # [n, W] bize verilen siparis adedi
    sevk: np.ndarray             # [n, W] bizim karsiladigimiz adet
    miad_red: np.ndarray         # [n, W] miad kisiti nedeniyle veremedigimiz
    eczane_tablo: pl.DataFrame
    urun_tablo: pl.DataFrame
    takvim: pl.DataFrame
    makro: pl.DataFrame
    olay_aktif: np.ndarray       # [S, W]
    olay_gecen: np.ndarray       # [S, W] son gorunur olaydan gecen hafta
    kur_gecen: np.ndarray        # [W] son referans kur guncellemesinden gecen hafta


@dataclass
class Panel:
    anahtar: pl.DataFrame        # eczane_id, sku_id, origin, hucre_idx
    X: np.ndarray                # [m, F] sabit ozellikler (float32)
    ozellik_adlari: list[str]
    kategorik_idx: list[int]
    origin: np.ndarray           # [m] origin haftasi
    hucre_idx: np.ndarray        # [m] izgaradaki hucre indeksi
    etiket_k: np.ndarray         # [m] ilk "bize siparis" gecikmesi (0 = olmadi)
    izlenen_k: np.ndarray        # [m] kac periyot gozlenebiliyor
    takvim_k: np.ndarray         # [W, 3] ay / ramazan_payi / yil_sonu


# --------------------------------------------------------------------------
# izgara
# --------------------------------------------------------------------------
def _olay_matrisleri(olaylar: pl.DataFrame, urunler: pl.DataFrame, W: int):
    """Gozlemlenebilir olay tablosundan [S, W] matrisleri.

    Olay ancak `gorunur_hafta`da gorunur (M1: antisipasyon penceresi
    duyurusuzdur). Burada gorunur_hafta'dan onceki hicbir bilgi kullanilmaz.
    """
    S = urunler.height
    sku_sira = {s: i for i, s in enumerate(urunler["sku_id"].to_list())}
    kategori = urunler["kategori_kod"].to_numpy()
    aktif = np.zeros((S, W), dtype=bool)
    son = np.full((S, W), -UZAK_GECMIS_HAFTA)
    kur_son = np.full(W, -UZAK_GECMIS_HAFTA)

    for satir in olaylar.iter_rows(named=True):
        gorunur = int(satir["gorunur_hafta"])
        bitis = int(satir["bitis_hafta"])
        if gorunur >= W:
            continue
        if satir["kapsam"] == "GLOBAL":
            idx = np.arange(S)
        elif satir["kapsam"] == "KATEGORI_AKUT":
            idx = np.flatnonzero(kategori == satir["hedef"])
        else:
            idx = np.array([sku_sira[s] for s in satir["hedef"].split(",")
                            if s in sku_sira], dtype=int)
        if idx.size == 0:
            continue
        aktif[np.ix_(idx, np.arange(gorunur, min(bitis, W)))] = True
        son[np.ix_(idx, np.arange(gorunur, W))] = gorunur
        if satir["tip"] == "REFERANS_KUR_GUNCELLEME":
            kur_son[gorunur:] = gorunur

    haftalar = np.arange(W)
    return aktif, haftalar[None, :] - son, haftalar - kur_son


def izgara_kur(kaynak: GozlemlenebilirKaynak, cfg: Config) -> Izgara:
    siparis = kaynak.tablo("siparisler")
    eczane = kaynak.tablo("eczaneler").sort("eczane_id")
    urun = kaynak.tablo("urunler").sort("sku_id")
    takvim = kaynak.tablo("takvim").sort("hafta")
    makro = kaynak.tablo("makro_haftalik").sort("hafta")
    olaylar = kaynak.tablo("olaylar")
    W = takvim.height

    ecz_sira = {e: i for i, e in enumerate(eczane["eczane_id"].to_list())}
    sku_sira = {s: i for i, s in enumerate(urun["sku_id"].to_list())}

    # Hucre evreni: bize EN AZ BIR KEZ siparis vermis (eczane, SKU) ciftleri.
    # Hic gormedigimiz hucre icin hiz tahmini yoktur; aday evreni budur.
    hucre = (siparis.select(["eczane_id", "sku_id"]).unique()
             .sort(["eczane_id", "sku_id"]))
    n = hucre.height
    hucre_sira = {k: i for i, k in enumerate(zip(hucre["eczane_id"], hucre["sku_id"]))}

    satir_idx = np.array([hucre_sira[(e, s)] for e, s
                          in zip(siparis["eczane_id"], siparis["sku_id"])])
    hafta = siparis["hafta"].to_numpy().astype(int)
    talep = np.zeros((n, W))
    sevk = np.zeros((n, W))
    miad_red = np.zeros((n, W))
    np.add.at(talep, (satir_idx, hafta), siparis["talep_adet"].to_numpy().astype(float))
    np.add.at(sevk, (satir_idx, hafta), siparis["karsilanan_adet"].to_numpy().astype(float))
    np.add.at(miad_red, (satir_idx, hafta),
              siparis["miad_kisiti_nedeniyle_verilemeyen"].to_numpy().astype(float))

    aktif, gecen, kur_gecen = _olay_matrisleri(olaylar, urun, W)

    return Izgara(
        eczane_id=hucre["eczane_id"].to_numpy(),
        sku_id=hucre["sku_id"].to_numpy(),
        eczane_idx=np.array([ecz_sira[e] for e in hucre["eczane_id"]]),
        sku_idx=np.array([sku_sira[s] for s in hucre["sku_id"]]),
        W=W, talep=talep, sevk=sevk, miad_red=miad_red,
        eczane_tablo=eczane, urun_tablo=urun, takvim=takvim, makro=makro,
        olay_aktif=aktif, olay_gecen=gecen, kur_gecen=kur_gecen,
    )


# --------------------------------------------------------------------------
# panel
# --------------------------------------------------------------------------
def origin_haftalari(cfg: Config, W: int) -> np.ndarray:
    """Panelin origin haftalari. Son origin, en az bir periyot gozlenebilecek
    sekilde secilir."""
    p = cfg.feature.panel
    son = W - 2                      # origin+1 haftasinin gozlenebilmesi icin
    return np.arange(p.ilk_origin_hafta, son + 1, p.origin_araligi_hafta)


def _kod(seri: pl.Series) -> tuple[np.ndarray, list[str]]:
    duzey = sorted(seri.unique().to_list())
    esle = {d: i for i, d in enumerate(duzey)}
    return np.array([esle[v] for v in seri.to_list()], dtype=np.float64), duzey


def panel_kur(izg: Izgara, cfg: Config, originler: np.ndarray | None = None) -> Panel:
    p, h, W = cfg.feature.panel, cfg.feature.hiz, izg.W
    ufuk = cfg.tukenme.hedef.ufuk_hafta
    originler = origin_haftalari(cfg, W) if originler is None else np.asarray(originler)

    n = izg.talep.shape[0]
    kum_talep = np.cumsum(izg.talep, axis=1)
    kum_sevk = np.cumsum(izg.sevk, axis=1)
    kum_red = np.cumsum(izg.miad_red, axis=1)
    var = izg.talep > 0
    kum_var = np.cumsum(var, axis=1)
    ewma = ewma_matrisi(izg.sevk, h.ewma_alfa)

    # son siparis / ilk siparis / son sevkiyat izleri [n, W]
    haftalar = np.arange(W, dtype=np.float64)

    def _son_iz(maske: np.ndarray) -> np.ndarray:
        """Her hafta icin, o haftaya kadarki (dahil) son True'nun hafta indeksi."""
        return np.maximum.accumulate(
            np.where(maske, haftalar[None, :], -UZAK_GECMIS_HAFTA), axis=1)

    son_siparis_h = _son_iz(var)
    son_sevk_h = _son_iz(izg.sevk > 0)
    ilk_iz = np.minimum.accumulate(np.where(var, haftalar[None, :], np.inf), axis=1)
    ilk_siparis_h = np.where(np.isfinite(ilk_iz), ilk_iz, np.nan)
    son_adet = np.zeros((n, W))
    tasima = np.zeros(n)
    for w in range(W):
        tasima = np.where(var[:, w], izg.talep[:, w], tasima)
        son_adet[:, w] = tasima

    # defter icin referans hiz: en uzun pencere (en dusuk varyans)
    uzun = max(h.pencereler_hafta)
    hiz_defter = np.empty((n, W))
    for w in range(W):
        hiz_defter[:, w] = gerileme_hizi(kum_sevk, w, uzun)
    defter = defter_stogu(izg.sevk, hiz_defter, cfg)

    # statik tablolar -> hucre hizasi
    ecz, urn = izg.eczane_tablo, izg.urun_tablo
    ciro_kod, _ = _kod(ecz["aylik_ciro_bandi"])
    il_kod, _ = _kod(ecz["il"])
    kat_kod, _ = _kod(urn["kategori_kod"])
    tip_kod, _ = _kod(urn["urun_tipi"])
    renk_kod, _ = _kod(urn["recete_rengi"])
    e, s = izg.eczane_idx, izg.sku_idx
    statik_ecz = {
        "ecz_aylik_recete_adedi": ecz["aylik_recete_adedi"].to_numpy()[e],
        "ecz_hastane_yakinligi_km": ecz["hastane_yakinligi_km"].to_numpy()[e],
        "ecz_sosyoekonomik": ecz["semt_sosyoekonomik_index"].to_numpy()[e],
        "ecz_turizm": ecz["turizm_bolgesi"].to_numpy()[e].astype(float),
        "ecz_nobet_rotasyon_gun": ecz["nobet_rotasyon_gun"].to_numpy()[e],
        "ecz_sgk_recete_orani": ecz["sgk_recete_orani"].to_numpy()[e],
        "ecz_vade_riski": ecz["vade_riski_skoru"].to_numpy()[e],
        "ecz_ciro_bandi": ciro_kod[e],
        "ecz_il": il_kod[e],
    }
    statik_sku = {
        "sku_psf": urn["psf"].to_numpy()[s],
        "sku_koli_ici_adet": urn["koli_ici_adet"].to_numpy()[s].astype(float),
        "sku_birim_hacim": urn["birim_hacim"].to_numpy()[s],
        "sku_soguk_zincir": urn["soguk_zincir"].to_numpy()[s].astype(float),
        "sku_sgk_geri_odeme": urn["sgk_geri_odeme"].to_numpy()[s].astype(float),
        "sku_promosyon_serbest": urn["promosyon_serbest"].to_numpy()[s].astype(float),
        "sku_tedarik_guclugu": urn["titck_tedarik_guclugu"].to_numpy()[s].astype(float),
        "sku_kategori": kat_kod[s],
        "sku_urun_tipi": tip_kod[s],
        "sku_recete_rengi": renk_kod[s],
    }
    kategorik_adlar = {"ecz_ciro_bandi", "ecz_il", "sku_kategori", "sku_urun_tipi",
                       "sku_recete_rengi", "ay"}

    kur = izg.makro["referans_avro_kuru"].to_numpy()
    endeks = izg.makro["fiyat_endeksi"].to_numpy()
    # havuzlama grubu: eczane x kategori (uzun kuyrukta tek hucre yetmez)
    grup = (izg.eczane_idx * (int(kat_kod.max()) + 1) + kat_kod[s].astype(int))
    grup_idx = np.unique(grup, return_inverse=True)[1]

    bloklar: list[tuple] = []
    for t in originler:
        yakin_siparis = gerileme_toplami(kum_var, t, p.aday_pencere_hafta)
        sec = np.flatnonzero(yakin_siparis >= p.min_siparis_sayisi)
        if sec.size == 0:
            continue

        gecen = t - son_siparis_h[sec, t]
        toplam_siparis = kum_var[sec, t]
        hiz_akislari = {f"hiz_akis_{k}h": gerileme_hizi(kum_sevk, t, k)[sec]
                        for k in h.pencereler_hafta}
        hiz_uzun = hiz_akislari[f"hiz_akis_{uzun}h"]
        ort_adet = np.where(toplam_siparis > 0, kum_talep[sec, t] / np.maximum(toplam_siparis, 1), 0.0)
        hiz_miktar = ort_adet / h.varsayilan_dongu_hafta

        # eczane duzeyinde gozlenen pay tahmini (akis / miktar orani)
        pay_tahmini = gozlenen_pay_tahmini(hiz_uzun, hiz_miktar, izg.eczane_idx[sec], cfg)
        pay_hucre = pay_tahmini[izg.eczane_idx[sec]]

        hiz_havuz = havuzlanmis_hiz(hiz_uzun, toplam_siparis, grup_idx[sec], h.havuzlama_gucu)
        # Defter, akisi `varsayilan_gozlenen_pay` ile telafi ederek kuruluyor;
        # tukenme suresi de AYNI telafili hizla hesaplanmali ki oran sadelessin
        # (features/stok.py). Telafili hiz burada bir kez hesaplanir.
        telafi = 1.0 / cfg.feature.stok.varsayilan_gozlenen_pay
        hiz_telafili = hiz_uzun * telafi
        defter_stok = defter[sec, t]
        defter_tuk = tukenme_haftasi(defter_stok, hiz_telafili, cfg, ufuk)
        # Son siparis tabanli tahmin: siparis MIKTARI seyreltilmemis (siparisin
        # tamami tek tedarikciye gider), hiz seyreltilmis. Bu ikisini bir arada
        # kullanmak defterdeki sadelesmeyi KIRAR ve sureyi 1/pay kadar uzun
        # gosterir. Kasitli: iki tahmincinin yanlilik yapisi farkli olsun diye.
        kalan_adet = np.maximum(son_adet[sec, t] - hiz_telafili * gecen, 0.0)
        son_tuk = tukenme_haftasi(kalan_adet, hiz_telafili, cfg, ufuk)

        # eczane duzeyi dinamik ozetler (son 52 hafta)
        ecz_pencere = max(h.pencereler_hafta)
        hucre_siparis = gerileme_toplami(kum_var, t, ecz_pencere).astype(float)
        hucre_adet = gerileme_toplami(kum_talep, t, ecz_pencere)
        P = izg.eczane_tablo.height
        ecz_satir = np.bincount(izg.eczane_idx, weights=hucre_siparis, minlength=P)
        ecz_adet = np.bincount(izg.eczane_idx, weights=hucre_adet, minlength=P)
        ecz_hucre = np.bincount(izg.eczane_idx, weights=(hucre_siparis > 0).astype(float),
                                minlength=P)
        S = izg.urun_tablo.height
        sku_hacim = np.bincount(izg.sku_idx, weights=hucre_adet, minlength=S)
        sku_ecz = np.bincount(izg.sku_idx, weights=(hucre_siparis > 0).astype(float),
                              minlength=S)

        fiyat_bas = max(0, t - FIYAT_DEGISIM_PENCERESI)
        ozellik = {
            "gecen_hafta_siparis": gecen,
            "gecen_hafta_sevkiyat": t - son_sevk_h[sec, t],
            "ilk_siparisten_hafta": t - ilk_siparis_h[sec, t],
            "siparis_sayisi_toplam": toplam_siparis,
            **{f"siparis_sayisi_{k}h": gerileme_toplami(kum_var, t, k)[sec].astype(float)
               for k in h.pencereler_hafta},
            **hiz_akislari,
            "hiz_ewma": ewma[sec, t],
            "hiz_havuzlu": hiz_havuz,
            "hiz_miktar": hiz_miktar,
            "hiz_duzeltilmis": hiz_uzun / pay_hucre,
            "son_siparis_adedi": son_adet[sec, t],
            "ortalama_siparis_adedi": ort_adet,
            "siparis_araligi_ort": np.where(
                toplam_siparis > 1,
                (t - ilk_siparis_h[sec, t]) / np.maximum(toplam_siparis - 1, 1),
                UZAK_GECMIS_HAFTA),
            "karsilama_orani_hucre": np.where(kum_talep[sec, t] > 0,
                                              kum_sevk[sec, t] / np.maximum(kum_talep[sec, t], 1),
                                              np.nan),
            "miad_reddi_orani_hucre": np.where(kum_talep[sec, t] > 0,
                                               kum_red[sec, t] / np.maximum(kum_talep[sec, t], 1),
                                               np.nan),
            "kumulatif_sevk": kum_sevk[sec, t],
            "defter_stok": defter_stok,
            "defter_kapsama_hafta": defter_stok / np.maximum(hiz_telafili, h.min_hiz),
            "defter_tukenme_hafta": defter_tuk,
            "son_siparis_kalan_adet": kalan_adet,
            "son_siparis_tukenme_hafta": son_tuk,
            "ecz_siparis_satiri": ecz_satir[izg.eczane_idx[sec]],
            "ecz_aktif_hucre": ecz_hucre[izg.eczane_idx[sec]],
            "ecz_ortalama_satir_adedi": np.where(
                ecz_satir[izg.eczane_idx[sec]] > 0,
                ecz_adet[izg.eczane_idx[sec]] / np.maximum(ecz_satir[izg.eczane_idx[sec]], 1),
                np.nan),
            "ecz_gozlenen_pay_tahmini": pay_hucre,
            **{ad: v[sec] for ad, v in statik_ecz.items()},
            "sku_hacim": sku_hacim[izg.sku_idx[sec]],
            "sku_eczane_sayisi": sku_ecz[izg.sku_idx[sec]],
            **{ad: v[sec] for ad, v in statik_sku.items()},
            "referans_avro_kuru": np.full(sec.size, kur[t]),
            "fiyat_endeksi": np.full(sec.size, endeks[t]),
            "fiyat_degisim": np.full(sec.size, endeks[t] / endeks[fiyat_bas] - 1.0),
            "kur_olayindan_gecen_hafta": np.full(sec.size, izg.kur_gecen[t]),
            "olaydan_gecen_hafta": izg.olay_gecen[izg.sku_idx[sec], t],
            "olay_aktif": izg.olay_aktif[izg.sku_idx[sec], t].astype(float),
        }
        bloklar.append((t, sec, ozellik))

    if not bloklar:
        raise ValueError("panel bos: origin araligi veya aday kurali cok dar")

    adlar = list(bloklar[0][2])
    X = np.vstack([np.column_stack([blok[ad] for ad in adlar]) for _, _, blok in bloklar])
    origin = np.concatenate([np.full(sec.size, t) for t, sec, _ in bloklar])
    hucre_idx = np.concatenate([sec for _, sec, _ in bloklar])

    # gozlemlenebilir etiket: origin sonrasi ilk "bize siparis"
    etiket_k = np.zeros(origin.size, dtype=np.int32)
    izlenen_k = np.zeros(origin.size, dtype=np.int32)
    imlec = 0
    for t, sec, _ in bloklar:
        # Kosunun son haftasindaki origin hic gozlenemez (gozlenebilir = 0);
        # etiketi 0 kalir ve kisi-periyot acilimina hic girmez.
        gozlenebilir = max(0, min(ufuk, W - 1 - t))
        if gozlenebilir:
            ileri = var[sec, t + 1: t + 1 + gozlenebilir]
            k = np.where(ileri.any(axis=1), ileri.argmax(axis=1) + 1, 0)
        else:
            k = np.zeros(sec.size, dtype=int)
        etiket_k[imlec: imlec + sec.size] = k
        izlenen_k[imlec: imlec + sec.size] = gozlenebilir
        imlec += sec.size

    takvim_k = np.column_stack([
        izg.takvim["ay"].to_numpy().astype(float),
        izg.takvim["ramazan_payi"].to_numpy().astype(float),
        izg.takvim["yil_sonu_stoklama"].to_numpy().astype(float),
    ])

    tam_adlar = adlar + ["k", "ay", "ramazan_payi", "yil_sonu_stoklama"]
    anahtar = pl.DataFrame({
        "eczane_id": izg.eczane_id[hucre_idx],
        "sku_id": izg.sku_id[hucre_idx],
        "origin": origin.astype(np.int32),
        "hucre_idx": hucre_idx.astype(np.int32),
    })
    return Panel(
        anahtar=anahtar, X=X.astype(np.float32), ozellik_adlari=tam_adlar,
        kategorik_idx=[i for i, ad in enumerate(tam_adlar) if ad in kategorik_adlar],
        origin=origin.astype(np.int32), hucre_idx=hucre_idx.astype(np.int32),
        etiket_k=etiket_k, izlenen_k=izlenen_k, takvim_k=takvim_k,
    )


def gun(hafta: np.ndarray | float) -> np.ndarray | float:
    """Hafta -> gun. Simulator haftalik cozunurlukte; SPEC gun istiyor."""
    return hafta * GUN_HAFTA
