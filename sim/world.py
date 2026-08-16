"""Sentetik dunya: haftalik kapali dongu.

Zincir (bir hafta):
    gelen sevkiyatlar -> hasta tuketimi -> eczane stogu duser
    -> eczane gozden gecirir -> siparis verir (koli katina yuvarlanmis)
    -> siparis BIZE veya RAKIBE gider (latent share_of_wallet)
    -> bize gelen siparis FEFO + miad toleransi ile karsilanir
    -> karsilanamayan rakibe kayar, SOW kalici zarar gorur
    -> bizim depo periyodik ikmal yapar (yeni lot)
    -> miadi gecen lotlar imha edilir

Modelin gorebildigi yer sadece "bize gelen siparis" ve "bizim sevkiyatimiz".
Tuketim, eczane stogu, rakip siparisi, share_of_wallet gorunmez.

CLAUDE.md 7 geregi zorluk kaynaklari:
  1. share_of_wallet latent VE zamanla kayiyor -> gecmis oran bayatliyor
  2. koli yuvarlamasi tuketim -> siparis eslemesini kiriyor
  3. eczane kendi EWMA'sina gore siparis veriyor, gercek hiza gore degil,
     ve o EWMA satislarla (sansurlu) besleniyor
  4. referans kur antisipasyonunda TUKETIM DEGISMIYOR, sadece siparis one
     cekiliyor -> siparis serisinden tuketim cikaran model yaniliyor
  5. cesit (assortment) zaman icinde degisiyor
  6. bizim stoksuzlugumuz siparisi gorunmez kiliyor (rakibe gidiyor)

M6'DA NE DEGISTI (yalnizca BICIM, davranis degil)
=================================================
Haftalik dongunun govdesi `hafta_adimi()`e, dongu oncesi kurulum
`dunya_kur()`a ayrildi; `dunya_kos()` artik ikisini surer. Sebep: M6'nin
kapali dongusu (sim/rollout.py) dunyayi ADIM ADIM surmek ve araya teklif
sevkiyati sokmak zorunda. Tek parca bir `for w` dongusu bunu imkansiz
kiliyordu ve dinamigi ikinci kez yazmak "closed-loop" iddiasini bosa
cikarirdi: politika o zaman dunyaya degil, dunyanin kopyasina tepki verirdi.

Cekilis sirasi BIRE BIR korundu; `tests/test_world.py::test_refactor_dunyayi
_degistirmedi` uretilen tablolarin sha256'sini M5 sonrasi anlik goruntuyle
karsilastirir. M1-M5 sonuclari bu yuzden gecerli kalir.

TEKLIF ENJEKSIYONU. `hafta_adimi(durum, teklif_sevk=..., teklif_miad=...)`
cagrildiginda kabul edilmis teklifin sevkiyati (a) eczanenin STOK
POZISYONUNA girer -- bu yuzden o hafta organik siparisi KUCULUR, kanibalizm
buradan dogar; (b) yoldaki sevkiyata eklenir ve tedarik suresi sonunda
eczanenin miad kovalarina duser -- fazla mal orada yaslanir ve IADE olur;
(c) depo cikisina sayilir. Uc kanal da dunyanin kendi mekanizmalari; teklif
icin yeni bir kural yazilmadi.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import polars as pl

from core.config import Config
from core.rng import SeedBank
from sim.calendar import GUN_HAFTA, nobet_gun_sayilari, takvim_kur
from sim.events import OlayDunyasi, olaylari_uret
from sim.lots import (EczaneLotKovalari, ImhaSatiri, LotDeposu, TahsisSatiri,
                      kalan_raf_omru_cek)
from sim.pharmacies import EczaneEvreni, eczane_evreni_kur
from sim.products import urun_evreni_kur

# Bir yil 12 aydir; aylik mevsimsellik vektorunun indeksi icin.
AY_SAYISI = 12
# Ciro/limit gibi TL buyukluklerinde iki ondalik. Bicim sabiti.
TL_ONDALIK = 2
# log(0) korumasi. Sayisal sabit, knob degil.
LOG_EPSILON = 1e-9


@dataclass
class DunyaCiktisi:
    takvim: pl.DataFrame
    urunler: pl.DataFrame
    eczaneler: pl.DataFrame
    olaylar_gozlemlenebilir: pl.DataFrame
    olaylar_gercek: pl.DataFrame
    siparisler: pl.DataFrame
    sevkiyat_satirlari: pl.DataFrame
    stok_lotlari: pl.DataFrame
    depo_stok_haftalik: pl.DataFrame
    imhalar: pl.DataFrame
    iadeler: pl.DataFrame
    urun_fiyat_haftalik: pl.DataFrame
    makro_haftalik: pl.DataFrame
    latent_urun: pl.DataFrame
    latent_eczane: pl.DataFrame
    hucre_haftalik: pl.DataFrame
    rakip_siparisleri: pl.DataFrame
    sow_haftalik: pl.DataFrame
    tukenme_olaylari: pl.DataFrame


def _mevsimsellik_matrisi(cfg: Config, urunler: pl.DataFrame, takvim: pl.DataFrame) -> np.ndarray:
    """[S, W] aylik mevsimsellik + ramazan carpani."""
    kod_indeks = {k.kod: i for i, k in enumerate(cfg.urun.kategoriler)}
    aylik = np.array([k.mevsimsellik for k in cfg.urun.kategoriler])          # [K, 12]
    ramazan = np.array([k.ramazan_carpani for k in cfg.urun.kategoriler])     # [K]
    kat_idx = np.array([kod_indeks[k] for k in urunler["kategori_kod"].to_numpy()])
    ay = takvim["ay"].to_numpy() - 1
    ramazan_payi = takvim["ramazan_payi"].to_numpy()
    taban = aylik[kat_idx][:, ay]                                            # [S, W]
    ramazan_carpan = 1.0 + (ramazan[kat_idx][:, None] - 1.0) * ramazan_payi[None, :]
    return taban * ramazan_carpan


def _turizm_matrisi(cfg: Config, ecz: EczaneEvreni, takvim: pl.DataFrame) -> np.ndarray:
    t = cfg.sim.talep.turizm
    ay = takvim["ay"].to_numpy()
    carpan = np.ones(len(ay))
    carpan[np.isin(ay, t.zirve_aylari)] = t.zirve_carpani
    carpan[np.isin(ay, t.omuz_aylari)] = t.omuz_carpani
    return 1.0 + (carpan[None, :] - 1.0) * ecz.turizm[:, None].astype(float)


def _iade_isle(cfg: Config, w: int, iade_adet, iade_kalan_gun, dsf,
               iade_kayit: list, imha_kayit: list, sebep: str):
    """Eczane iadesini kaydeder ve eczane basina toplami dondurur.

    Iadenin `depoya_iade_orani` kadari bize fiziken geri doner: satilamaz
    (raf omru esigin altinda), imha edilir ve eczaneye kredi verilir. Kalani
    eczanede zayidir. Ikisi de gercek maliyet.
    """
    ia = cfg.sim.iade
    if not iade_adet.any():
        return np.zeros(iade_adet.shape[0])
    imha_orani = cfg.lot.maliyet.imha_birim_maliyeti_dsf_orani
    for p_i, s_i in zip(*np.nonzero(iade_adet)):
        adet = int(iade_adet[p_i, s_i])
        donen = int(round(adet * ia.depoya_iade_orani))
        kredi = donen * dsf[s_i] * ia.kredi_orani
        iade_kayit.append((w, int(p_i), int(s_i), adet, donen,
                           float(iade_kalan_gun[p_i, s_i]), float(kredi)))
        if donen > 0:
            imha_kayit.append(
                ImhaSatiri(hafta=w, sku_idx=int(s_i), lot_id=None, adet=donen,
                           birim_maliyet=float(dsf[s_i]),
                           imha_maliyeti=float(donen * dsf[s_i] * imha_orani),
                           kaynak=f"eczane_iadesi:{sebep}")
            )
    return iade_adet.sum(axis=1).astype(float)


@dataclass
class DunyaKayitlari:
    """Kosu boyunca biriken ham kayitlar. Cikti tablolari bunlardan kurulur."""

    tuketim_3d: np.ndarray
    stok_3d: np.ndarray
    cesit_3d: np.ndarray
    karsilanmayan_3d: np.ndarray
    depo_stok_kayit: np.ndarray
    sow_kayit: np.ndarray
    rakip_kayit: list = field(default_factory=list)
    siparis_kayit: list = field(default_factory=list)
    sevk_kayit: list = field(default_factory=list)
    imha_kayit: list = field(default_factory=list)
    tukenme_kayit: list = field(default_factory=list)
    iade_kayit: list = field(default_factory=list)


@dataclass
class DunyaDurumu:
    """Simulasyonun bir andaki TAM durumu + hafta boyunca degismeyen tablolar.

    `hafta_adimi` bunu yerinde gunceller. `w` bir sonraki kosulacak haftadir.
    """

    cfg: Config
    rng: np.random.Generator
    P: int
    S: int
    W: int
    w: int
    # --- sabitler ---
    takvim: pl.DataFrame
    urunler: pl.DataFrame
    ecz: EczaneEvreni
    olay: OlayDunyasi
    latent_urun: pl.DataFrame
    kat_idx: np.ndarray
    akut_mask: np.ndarray
    miad_kat_carpani: np.ndarray
    dsf: np.ndarray
    psf: np.ndarray
    depo_marji: np.ndarray
    soguk: np.ndarray
    soguk_pas: np.ndarray
    min_siparis: np.ndarray
    lam_base: np.ndarray
    cesit_olasiligi: np.ndarray
    mevsim: np.ndarray
    turizm_c: np.ndarray
    nobet_c: np.ndarray
    # --- degisen durum ---
    assort: np.ndarray
    hic_cesitte_oldu: np.ndarray
    kovalar: EczaneLotKovalari
    ewma: np.ndarray
    ewma_var: np.ndarray
    sow: np.ndarray
    yolda: deque
    depo: LotDeposu
    depo_cikis_ewma: np.ndarray
    depo_cikis_var_ewma: np.ndarray
    onceki_stok_sifir: np.ndarray
    kayit: DunyaKayitlari

    @property
    def eczane_stogu(self) -> np.ndarray:
        return self.kovalar.toplam()


def dunya_kur(cfg: Config, seedler: SeedBank) -> DunyaDurumu:
    """Haftalik dongu oncesi butun kurulum. Cekilis sirasi M1'den degismedi."""
    P, S, W = cfg.profil.eczane_sayisi, cfg.profil.sku_sayisi, cfg.profil.hafta_sayisi
    rng = seedler.generator("dunya_dongusu")

    takvim = takvim_kur(cfg)
    urunler, latent_urun = urun_evreni_kur(cfg, seedler)
    ecz = eczane_evreni_kur(cfg, seedler)
    populerlik = latent_urun["latent_populerlik"].to_numpy()
    olay = olaylari_uret(cfg, seedler, urunler, populerlik, takvim)

    kod_indeks = {k.kod: i for i, k in enumerate(cfg.urun.kategoriler)}
    kat_idx = np.array([kod_indeks[k] for k in urunler["kategori_kod"].to_numpy()])
    akut_mask = np.array([cfg.urun.kategoriler[i].akut for i in kat_idx])
    miad_kat_carpani = np.array([cfg.urun.kategoriler[i].miad_toleransi_carpani for i in kat_idx])
    dsf = urunler["dsf"].to_numpy()
    psf = urunler["psf"].to_numpy()
    depo_marji = urunler["depo_kar_marji"].to_numpy()
    soguk = urunler["soguk_zincir"].to_numpy()

    # --- latent talep yogunlugu [P, S] ---
    tc = cfg.sim.talep
    hucre_gurultu = rng.gamma(tc.yogunluk.hucre_gurultu_shape, size=(P, S))
    hucre_gurultu /= tc.yogunluk.hucre_gurultu_shape
    lam_base = (
        tc.yogunluk.taban_adet_hafta
        * populerlik[None, :]
        * ecz.buyukluk[:, None]
        * ecz.affinite[:, kat_idx]
        * hucre_gurultu
    )

    # --- cesit (yapisal sifirlar) ---
    cs = tc.cesitlendirme
    skor = (
        np.log(np.maximum(populerlik, LOG_EPSILON))[None, :] * cs.populerlik_agirligi
        + np.log(np.maximum(ecz.affinite[:, kat_idx], LOG_EPSILON)) * cs.affinite_agirligi
        + np.log(np.maximum(ecz.buyukluk, LOG_EPSILON))[:, None] * cs.buyukluk_agirligi
    )
    olasilik = 1.0 / (1.0 + np.exp(-(skor + np.log(cs.taban_oran / (1.0 - cs.taban_oran)))))
    assort = rng.random((P, S)) < olasilik
    hic_cesitte_oldu = assort.copy()

    mevsim = _mevsimsellik_matrisi(cfg, urunler, takvim)
    turizm_c = _turizm_matrisi(cfg, ecz, takvim)
    nobet_gun = nobet_gun_sayilari(ecz.nobet_periyot, ecz.nobet_ofset, W)
    nobet_c = 1.0 + nobet_gun * (tc.nobet.akut_carpani - 1.0) / GUN_HAFTA

    env = cfg.sim.envanter
    ts = cfg.sim.tedarikci_secimi
    ik = cfg.sim.ikmal
    L = env.tedarik_suresi_hafta

    min_siparis = np.where(soguk, env.soguk_zincir_minimum_siparis_adedi,
                           env.minimum_siparis_adedi).astype(np.int64)
    kovalar = EczaneLotKovalari(env.eczane_lot_bolme_sayisi, P, S)
    baslangic_stok = (np.maximum(np.ceil(lam_base * env.baslangic_kapsama_hafta),
                                 min_siparis[None, :]) * assort).astype(np.int64)
    soguk_pas = np.broadcast_to(soguk[None, :], (P, S))
    kovalar.ekle(baslangic_stok, kalan_raf_omru_cek(cfg, rng, (P, S), soguk_pas))
    ph_stock = kovalar.toplam()
    ewma = lam_base * assort
    # Eczanenin talep sapmasi tahmini. Baslangicta Poisson varsayimi (var = ort).
    ewma_var = lam_base * assort
    sow = ecz.sow0.copy()
    # Yoldaki sevkiyat hem adet hem miad tasir: eczaneye giden malin raf omru
    # onun ileride iade olup olmayacagini belirler.
    yolda = deque((np.zeros((P, S), dtype=np.int64),
                   np.full((P, S), 0, dtype=np.int64)) for _ in range(L))

    depo = LotDeposu(cfg=cfg, dsf=dsf, depo_marji=depo_marji, soguk_zincir=soguk)
    beklenen_haftalik_cikis = (lam_base * assort).sum(axis=0) * sow.mean()
    for s in range(S):
        adet = int(np.ceil(max(beklenen_haftalik_cikis[s] * ik.baslangic_kapsama_hafta,
                               ik.minimum_parti_adet)))
        depo.parti_yarat(s, adet, 0, rng)
    depo_cikis_ewma = beklenen_haftalik_cikis.copy()
    depo_cikis_var_ewma = beklenen_haftalik_cikis.copy()

    kayit = DunyaKayitlari(
        tuketim_3d=np.zeros((P, S, W), dtype=np.int32),
        stok_3d=np.zeros((P, S, W), dtype=np.int32),
        cesit_3d=np.zeros((P, S, W), dtype=bool),
        karsilanmayan_3d=np.zeros((P, S, W), dtype=np.int32),
        depo_stok_kayit=np.zeros((S, W), dtype=np.int64),
        sow_kayit=np.zeros((P, W), dtype=np.float64),
    )

    return DunyaDurumu(
        cfg=cfg, rng=rng, P=P, S=S, W=W, w=0,
        takvim=takvim, urunler=urunler, ecz=ecz, olay=olay,
        latent_urun=latent_urun, kat_idx=kat_idx, akut_mask=akut_mask,
        miad_kat_carpani=miad_kat_carpani, dsf=dsf, psf=psf,
        depo_marji=depo_marji, soguk=soguk, soguk_pas=soguk_pas,
        min_siparis=min_siparis, lam_base=lam_base, cesit_olasiligi=olasilik,
        mevsim=mevsim, turizm_c=turizm_c, nobet_c=nobet_c,
        assort=assort, hic_cesitte_oldu=hic_cesitte_oldu, kovalar=kovalar,
        ewma=ewma, ewma_var=ewma_var, sow=sow, yolda=yolda, depo=depo,
        depo_cikis_ewma=depo_cikis_ewma,
        depo_cikis_var_ewma=depo_cikis_var_ewma,
        onceki_stok_sifir=np.zeros((P, S), dtype=bool),
        kayit=kayit,
    )


def hafta_adimi(d: DunyaDurumu, teklif_sevk: np.ndarray | None = None,
                teklif_miad_agirlikli: np.ndarray | None = None) -> None:
    """Bir haftayi kosar ve `d`yi yerinde ilerletir.

    `teklif_sevk` [P, S] : o hafta KABUL EDILMIS tekliflerin sevk adedi
                           (bedava adet dahil). Cagiran taraf bu adedi depodan
                           zaten dusmus ve sevk kaydini yazmis olmali - lot
                           secimi bir POLITIKA kararidir, dunyanin isi degil.
    `teklif_miad_agirlikli` [P, S] : ayni adetlerin adet-agirlikli mutlak miad
                           gunu toplami (gelen malin raf omru eczanede iade
                           riskini belirler).

    Ikisi de None ise dunya M1'deki gibi, teklifsiz kosar.
    """
    cfg, rng = d.cfg, d.rng
    P, S, w = d.P, d.S, d.w
    tc, env = cfg.sim.talep, cfg.sim.envanter
    cs = tc.cesitlendirme
    ts, ik = cfg.sim.tedarikci_secimi, cfg.sim.ikmal
    L = env.tedarik_suresi_hafta
    k_ = d.kayit
    ecz, olay = d.ecz, d.olay

    bugun = w * GUN_HAFTA
    gelen_adet_w, gelen_miad_w = d.yolda.popleft()
    d.kovalar.ekle(gelen_adet_w, gelen_miad_w)
    ph_stock = d.kovalar.toplam()

    # --- 1) hasta tuketimi ---
    kat_sok = np.exp(rng.normal(0.0, tc.dagilim.kategori_hafta_soku_sigma,
                                len(cfg.urun.kategoriler)))[d.kat_idx]
    ecz_sok = np.exp(rng.normal(0.0, tc.dagilim.eczane_hafta_soku_sigma, P))
    lam = (
        d.lam_base
        * (d.mevsim[:, w] * olay.tuketim_carpani[:, w] * kat_sok)[None, :]
        * d.turizm_c[:, w][:, None]
        * ecz_sok[:, None]
    )
    lam[:, d.akut_mask] *= d.nobet_c[:, w][:, None]
    lam = np.where(d.assort, lam, 0.0)

    k = tc.dagilim.negbin_shape
    gecici = rng.gamma(k, scale=lam / k)
    talep = rng.poisson(gecici)
    talep = np.where(rng.random((P, S)) < tc.dagilim.sifir_sisirme, 0, talep)

    satis = np.minimum(talep, ph_stock)
    karsilanmayan = talep - satis
    d.kovalar.tuket(satis)

    # --- 1b) eczanede yaslanan stok: satilamaz, IADE olur (SPEC 2.5) ---
    iade_adet, iade_kalan_gun = d.kovalar.satilamayacagi_bosalt(
        bugun, gunluk_hiz=d.ewma / GUN_HAFTA,
        guvenlik_marji_gun=cfg.sim.iade.eczaci_guvenlik_marji_gun,
        degerlendirme_esigi_gun=cfg.sim.iade.degerlendirme_esigi_gun,
    )
    ph_stock = d.kovalar.toplam()
    iade_hafta_p = _iade_isle(
        cfg, w, iade_adet, iade_kalan_gun, d.dsf, k_.iade_kayit, k_.imha_kayit,
        "eczane_miad",
    )

    k_.tuketim_3d[:, :, w] = satis
    k_.karsilanmayan_3d[:, :, w] = karsilanmayan
    k_.cesit_3d[:, :, w] = d.assort

    # gercek tukenme olayi: stok bu hafta sifira dustu (M2 oracle'i icin)
    simdi_sifir = (ph_stock == 0) & d.assort
    yeni_tukenen = simdi_sifir & ~d.onceki_stok_sifir
    for p_i, s_i in zip(*np.nonzero(yeni_tukenen)):
        k_.tukenme_kayit.append((w, int(p_i), int(s_i)))
    d.onceki_stok_sifir = simdi_sifir

    # --- 2) eczanenin kendi tahmini (sansurlu: sadece SATISI gorur) ---
    sapma_kare = (satis - d.ewma) ** 2
    d.ewma = np.where(d.assort,
                      env.talep_ewma_alfa * satis + (1 - env.talep_ewma_alfa) * d.ewma,
                      0.0)
    d.ewma_var = np.where(
        d.assort,
        env.talep_varyans_ewma_alfa * sapma_kare
        + (1 - env.talep_varyans_ewma_alfa) * d.ewma_var,
        0.0,
    )

    # --- 3) siparis karari ---
    yolda_toplam = (np.sum([a for a, _ in d.yolda], axis=0) if len(d.yolda)
                    else np.zeros((P, S), dtype=np.int64))
    pozisyon = ph_stock + yolda_toplam
    if teklif_sevk is not None:
        # KANIBALIZM KANALI. Kabul edilen teklif eczanenin stok pozisyonuna
        # girer; o hafta (ve yoldayken sonraki haftalarda) organik siparisi
        # bu kadar kuculur. M6'nin uzun ufukta agresif iskontoyu kaybettiren
        # mekanizmasi burasidir ve teklif icin yazilmis ayri bir kural degil,
        # eczanenin kendi (s, S) politikasinin sonucudur.
        pozisyon = pozisyon + teklif_sevk
    antic = olay.antisipasyon[:, w]
    antic_carpani = np.minimum(
        1.0 + ecz.stokculuk[:, None] * antic[None, :] * env.antisipasyon_kapsama_kazanci,
        env.antisipasyon_azami_carpan,
    )
    kapsama = ecz.kapsama_hafta[:, None] * antic_carpani
    # Emniyet stogu: sapma x sqrt(tedarik suresi + gozden gecirme araligi)
    risk_penceresi = np.sqrt(L + ecz.gozden_gecirme[:, None])
    emniyet = env.emniyet_z_katsayisi * np.sqrt(np.maximum(d.ewma_var, 0.0)) * risk_penceresi
    hedef = d.ewma * (kapsama + L) + emniyet
    # Max-stok tavani: eczaci NORMALDE bu kadar haftadan fazla stok tutmaz.
    # Tavan antisipasyon carpaniyla birlikte gevser -- stoklama zaten bilincli
    # olarak normal tavanin uzerine cikmaktir; tavan sabit kalsaydi rejim
    # olaylarinin siparis kanali sonerdi.
    # min_siparis eklenmezse cok yavas hucreler hic siparis veremez.
    hedef = np.minimum(
        hedef,
        d.ewma * env.azami_kapsama_hafta * antic_carpani + d.min_siparis[None, :],
    )
    yeniden_siparis_noktasi = d.ewma * (L + env.emniyet_stogu_hafta) + emniyet

    gozden_gecir = ((w + ecz.gozden_gecirme_ofset) % ecz.gozden_gecirme) == 0
    gozden_gecir_hucre = np.broadcast_to(gozden_gecir[:, None], (P, S)).copy()
    if env.stoksuzlukta_acil_gozden_gecirme:
        gozden_gecir_hucre |= (ph_stock == 0) & d.assort

    # Stoklama rejimi: beklenti yeterince gucluyse nokta kosulu aranmaz.
    stoklama = (ecz.stokculuk[:, None] * antic[None, :]) > env.antisipasyon_siparis_esigi
    ham = hedef - pozisyon
    ver = (gozden_gecir_hucre & d.assort
           & ((pozisyon <= yeniden_siparis_noktasi) | stoklama) & (ham > 0))
    adet = np.maximum(np.ceil(np.maximum(ham, 0.0)), d.min_siparis[None, :])
    adet = np.where(ver, adet, 0).astype(np.int64)

    # --- 4) tedarikci secimi (multi-homing) ---
    p_bize = np.clip(d.sow[:, None] + rng.normal(0.0, ts.siparis_gurultusu, (P, S)), 0.0, 1.0)
    bize = ver & (rng.random((P, S)) < p_bize)
    rakibe = ver & ~bize

    # Teklif sevkiyati yoldaki mala eklenir: tedarik suresi sonunda eczanenin
    # miad kovalarina duser ve oradan itibaren dunyanin normal kurallarina
    # (FEFO tuketim, satilamayani iade) tabidir.
    gelen_sevk = (np.zeros((P, S), dtype=np.int64) if teklif_sevk is None
                  else teklif_sevk.astype(np.int64).copy())
    gelen_miad_agirlikli = (np.zeros((P, S), dtype=np.float64)
                            if teklif_miad_agirlikli is None
                            else teklif_miad_agirlikli.astype(float).copy())

    # Rakip depodan gelen mal da miat tasir; bizim kuyrugumuzdan gelmedigi
    # icin raf omru ayri cekilir (ayni dagilim).
    rakip_kalan = kalan_raf_omru_cek(cfg, rng, (P, S), d.soguk_pas)
    rakip_adet = np.where(rakibe, adet, 0)
    gelen_sevk += rakip_adet
    gelen_miad_agirlikli += rakip_adet * (bugun + rakip_kalan)
    for p_i, s_i in zip(*np.nonzero(rakibe)):
        k_.rakip_kayit.append((w, int(p_i), int(s_i), int(adet[p_i, s_i])))

    # --- 5) bizim karsilamamiz: FEFO + miad toleransi ---
    istenen_toplam = np.zeros(P)
    eksik_toplam = np.zeros(P)
    haftalik_cikis = np.zeros(S)
    if teklif_sevk is not None:
        # Teklif sevkiyati da depo cikisidir: ikmal EWMA'si onu gormezse depo
        # sistematik olarak az siparis verir.
        haftalik_cikis += teklif_sevk.sum(axis=0).astype(float)
    for p_i, s_i in zip(*np.nonzero(bize)):
        istenen = int(adet[p_i, s_i])
        gerekli_gun = ecz.miad_toleransi[p_i] * d.miad_kat_carpani[s_i]
        karsilanan, satirlar, miad_reddi = d.depo.tahsis_et(
            sku_idx=int(s_i), adet=istenen, hafta=w, eczane_idx=int(p_i),
            gerekli_kalan_gun=gerekli_gun,
        )
        k_.sevk_kayit.extend(satirlar)
        for t in satirlar:
            gelen_miad_agirlikli[p_i, s_i] += t.adet * (bugun + t.kalan_raf_omru_gun)
        haftalik_cikis[s_i] += karsilanan
        eksik_adet = istenen - karsilanan
        # Atlanan lot sonradan baska lottan karsilanmis olabilir; sadece
        # gercekten eksige donusen kismi rapor et.
        etkili_miad_reddi = min(miad_reddi, eksik_adet)
        k_.siparis_kayit.append(
            (w, int(p_i), int(s_i), istenen, karsilanan, etkili_miad_reddi))
        gelen_sevk[p_i, s_i] += karsilanan
        eksik = eksik_adet
        istenen_toplam[p_i] += istenen
        eksik_toplam[p_i] += eksik
        if eksik > 0:
            if ts.karsilanamayan_siparis_rakibe_gider:
                gelen_sevk[p_i, s_i] += eksik
                gelen_miad_agirlikli[p_i, s_i] += eksik * (
                    bugun + rakip_kalan[p_i, s_i])
                k_.rakip_kayit.append((w, int(p_i), int(s_i), eksik))

    gelen_miad = np.where(
        gelen_sevk > 0,
        np.rint(gelen_miad_agirlikli / np.maximum(gelen_sevk, 1)).astype(np.int64),
        0,
    )
    d.yolda.append((gelen_sevk, gelen_miad))

    # --- 6) SOW dinamigi ---
    # Ceza ikili bayrak degil, o haftaki karsilayamama ORANI ile olcekli:
    # tek bir kalemi eksik gonderdik diye tum pay kaybedilmez.
    eksik_oran = np.where(istenen_toplam > 0, eksik_toplam / np.maximum(istenen_toplam, 1), 0.0)
    # Iade de iliskiyi yipratir: eczaneye satamayacagi mal gonderilmis demektir.
    iade_oran = np.where(istenen_toplam > 0,
                         iade_hafta_p / np.maximum(istenen_toplam, 1), 0.0)
    # Toparlanma ortalamaya donustur: pay yapisal seviyesine geri cekilir.
    d.sow = d.sow + ts.sow_toparlanma_hizi * (ecz.sow0 - d.sow)
    d.sow -= ts.stoksuzluk_sow_cezasi * eksik_oran
    d.sow -= cfg.sim.iade.sow_cezasi * np.minimum(iade_oran, 1.0)
    d.sow = d.sow + rng.normal(0.0, ts.sow_rassal_yuruyus_sigma, P)
    d.sow = np.clip(d.sow, cfg.eczane.latent_share_of_wallet.min,
                    cfg.eczane.latent_share_of_wallet.max)
    k_.sow_kayit[:, w] = d.sow

    # --- 7) bizim ikmalimiz ---
    depo_sapma_kare = (haftalik_cikis - d.depo_cikis_ewma) ** 2
    d.depo_cikis_ewma = (env.talep_ewma_alfa * haftalik_cikis
                         + (1 - env.talep_ewma_alfa) * d.depo_cikis_ewma)
    d.depo_cikis_var_ewma = (env.talep_varyans_ewma_alfa * depo_sapma_kare
                             + (1 - env.talep_varyans_ewma_alfa) * d.depo_cikis_var_ewma)
    if w % ik.periyot_hafta == 0:
        depo_emniyet = (ik.emniyet_z_katsayisi
                        * np.sqrt(np.maximum(d.depo_cikis_var_ewma, 0.0))
                        * np.sqrt(ik.periyot_hafta))
        for s in range(S):
            if olay.ikmal_blok[s, w]:
                continue
            hedef_adet = d.depo_cikis_ewma[s] * ik.hedef_kapsama_hafta + depo_emniyet[s]
            acik = hedef_adet - d.depo.eldeki_adet(s)
            if acik <= 0:
                continue
            gurultu = float(np.exp(rng.normal(0.0, ik.siparis_gurultusu_sigma)))
            miktar = int(max(ik.minimum_parti_adet, round(acik * gurultu)))
            # Ikmal tahsisTEN SONRA calisir: bu parti ancak gelecek hafta
            # tahsis edilebilir. giris_haftasi = kullanilabilir oldugu hafta.
            d.depo.parti_yarat(s, miktar, w + 1, rng)

    k_.imha_kayit.extend(d.depo.miadi_gecenleri_imha_et(w))
    for s in range(S):
        k_.depo_stok_kayit[s, w] = d.depo.eldeki_adet(s)

    # --- 8) cesit degisimi (seviye durgun, kompozisyon degisken) ---
    cikar = (d.assort & (ph_stock == 0)
             & (rng.random((P, S)) < cs.haftalik_churn_orani))
    ekle = np.zeros((P, S), dtype=bool)
    n_cikan = int(cikar.sum())
    aday = (~d.assort) & ~cikar
    if n_cikan and aday.any():
        agirlik = (d.cesit_olasiligi * aday).ravel()
        toplam = agirlik.sum()
        if toplam > 0:
            secim = rng.choice(P * S, size=min(n_cikan, int(aday.sum())),
                               replace=False, p=agirlik / toplam)
            ekle.ravel()[secim] = True
    if ekle.any():
        baslangic_ewma = np.maximum(
            cs.yeni_cesit_deneme_adedi, d.min_siparis[None, :]
        ) / ecz.kapsama_hafta[:, None]
        d.ewma = np.where(ekle, baslangic_ewma, d.ewma)
    d.assort = (d.assort | ekle) & ~cikar
    d.hic_cesitte_oldu |= d.assort
    # Listeden cikan urunun eczanedeki kalan stogu havada kalmaz: iade olur.
    dusenler = d.kovalar.maskeyi_bosalt(~d.assort)
    if cfg.sim.iade.cesitten_cikarmada_iade and dusenler.any():
        kalan_gun_tahmini = np.full((P, S), cfg.sim.iade.degerlendirme_esigi_gun,
                                    dtype=np.float64)
        _iade_isle(cfg, w, dusenler, kalan_gun_tahmini, d.dsf, k_.iade_kayit,
                   k_.imha_kayit, "cesitten_cikarma")
    k_.stok_3d[:, :, w] = d.kovalar.toplam()
    d.w = w + 1


def dunya_kos(cfg: Config, seedler: SeedBank) -> DunyaCiktisi:
    d = dunya_kur(cfg, seedler)
    for _ in range(d.W):
        hafta_adimi(d)
    k = d.kayit
    return _ciktilari_topla(
        cfg, d.takvim, d.urunler, d.ecz, d.olay, d.latent_urun,
        k.tuketim_3d, k.stok_3d, k.cesit_3d, k.karsilanmayan_3d,
        d.hic_cesitte_oldu, d.lam_base, k.rakip_kayit, k.siparis_kayit,
        k.sevk_kayit, k.imha_kayit, k.depo_stok_kayit, k.sow_kayit,
        k.tukenme_kayit, k.iade_kayit, d.depo, d.psf, d.dsf,
    )


def _ciktilari_topla(
    cfg, takvim, urunler, ecz, olay, latent_urun,
    tuketim_3d, stok_3d, cesit_3d, karsilanmayan_3d, hic_cesitte_oldu,
    lam_base, rakip_kayit, siparis_kayit, sevk_kayit, imha_kayit,
    depo_stok_kayit, sow_kayit, tukenme_kayit, iade_kayit, depo, psf, dsf,
) -> DunyaCiktisi:
    P, S, W = cfg.profil.eczane_sayisi, cfg.profil.sku_sayisi, cfg.profil.hafta_sayisi
    ecz_id = ecz.master["eczane_id"].to_numpy()
    sku_id = urunler["sku_id"].to_numpy()
    hafta_tarih = takvim["hafta_basi_tarih"].to_list()

    def _sip_df(kayit, ad_adet: list[str]) -> pl.DataFrame:
        if not kayit:
            return pl.DataFrame(schema={"hafta": pl.Int32, "eczane_id": pl.Utf8,
                                        "sku_id": pl.Utf8, **{a: pl.Int64 for a in ad_adet}})
        dizi = np.array(kayit, dtype=np.int64)
        veri = {
            "hafta": dizi[:, 0].astype(np.int32),
            "eczane_id": ecz_id[dizi[:, 1]],
            "sku_id": sku_id[dizi[:, 2]],
        }
        for j, ad in enumerate(ad_adet):
            veri[ad] = dizi[:, 3 + j]
        return pl.DataFrame(veri)

    siparisler = _sip_df(
        siparis_kayit,
        ["talep_adet", "karsilanan_adet", "miad_kisiti_nedeniyle_verilemeyen"],
    )
    if siparisler.height:
        siparisler = siparisler.join(takvim.select(["hafta", "hafta_basi_tarih"]),
                                     on="hafta", how="left")
    rakip = _sip_df(rakip_kayit, ["rakip_siparis_adedi"])

    sevkiyat = pl.DataFrame(
        {
            "hafta": np.array([t.hafta for t in sevk_kayit], dtype=np.int32),
            "eczane_id": ecz_id[[t.eczane_idx for t in sevk_kayit]] if sevk_kayit else [],
            "sku_id": sku_id[[t.sku_idx for t in sevk_kayit]] if sevk_kayit else [],
            "lot_id": [t.lot_id for t in sevk_kayit],
            "adet": np.array([t.adet for t in sevk_kayit], dtype=np.int64),
            "kalan_raf_omru_gun": np.array([t.kalan_raf_omru_gun for t in sevk_kayit],
                                           dtype=np.int32),
        }
    )

    tum_lot = depo.tum_lotlar()
    baslangic = cfg.profil.baslangic_tarihi
    stok_lotlari = pl.DataFrame(
        {
            "lot_id": [l.lot_id for l in tum_lot],
            "sku_id": sku_id[[l.sku_idx for l in tum_lot]] if tum_lot else [],
            "giris_haftasi": np.array([l.giris_haftasi for l in tum_lot], dtype=np.int32),
            "miad_gun_indeksi": np.array([l.miad_gun for l in tum_lot], dtype=np.int32),
            "miad_tarihi": [baslangic + timedelta(days=l.miad_gun) for l in tum_lot],
            "adet_giris": np.array([l.adet_giris for l in tum_lot], dtype=np.int64),
            "birim_maliyet": np.round([l.birim_maliyet for l in tum_lot], TL_ONDALIK),
        }
    )

    imhalar = pl.DataFrame(
        {
            "hafta": np.array([i.hafta for i in imha_kayit], dtype=np.int32),
            "sku_id": sku_id[[i.sku_idx for i in imha_kayit]] if imha_kayit else [],
            "lot_id": [i.lot_id for i in imha_kayit],
            "adet": np.array([i.adet for i in imha_kayit], dtype=np.int64),
            "imha_maliyeti": np.round([i.imha_maliyeti for i in imha_kayit], TL_ONDALIK),
            "kaynak": [i.kaynak for i in imha_kayit],
        }
    )

    iadeler = (
        pl.DataFrame(
            {
                "hafta": np.array([r[0] for r in iade_kayit], dtype=np.int32),
                "eczane_id": ecz_id[[r[1] for r in iade_kayit]],
                "sku_id": sku_id[[r[2] for r in iade_kayit]],
                "iade_adet": np.array([r[3] for r in iade_kayit], dtype=np.int64),
                "depoya_donen_adet": np.array([r[4] for r in iade_kayit], dtype=np.int64),
                "kalan_raf_omru_gun": np.round([r[5] for r in iade_kayit], 1),
                "kredi_tutari": np.round([r[6] for r in iade_kayit], TL_ONDALIK),
            }
        )
        if iade_kayit
        else pl.DataFrame(schema={"hafta": pl.Int32, "eczane_id": pl.Utf8, "sku_id": pl.Utf8,
                                  "iade_adet": pl.Int64, "depoya_donen_adet": pl.Int64,
                                  "kalan_raf_omru_gun": pl.Float64, "kredi_tutari": pl.Float64})
    )

    depo_stok = pl.DataFrame(
        {
            "hafta": np.repeat(np.arange(W, dtype=np.int32), S),
            "sku_id": np.tile(sku_id, W),
            "eldeki_adet": depo_stok_kayit.T.reshape(-1),
        }
    )

    endeks = olay.fiyat_endeksi
    fiyat = pl.DataFrame(
        {
            "hafta": np.repeat(np.arange(W, dtype=np.int32), S),
            "sku_id": np.tile(sku_id, W),
            "psf": np.round((psf[None, :] * endeks[:, None]).reshape(-1), TL_ONDALIK),
            "dsf": np.round((dsf[None, :] * endeks[:, None]).reshape(-1), TL_ONDALIK),
        }
    )
    makro = pl.DataFrame(
        {
            "hafta": np.arange(W, dtype=np.int32),
            "referans_avro_kuru": np.round(olay.referans_kur, 4),
            "fiyat_endeksi": np.round(endeks, 6),
        }
    )

    aktif_p, aktif_s = np.nonzero(hic_cesitte_oldu)
    n = aktif_p.size
    hucre = pl.DataFrame(
        {
            "hafta": np.repeat(np.arange(W, dtype=np.int32), n),
            "eczane_id": np.tile(ecz_id[aktif_p], W),
            "sku_id": np.tile(sku_id[aktif_s], W),
            "gercek_tuketim": tuketim_3d[aktif_p, aktif_s, :].T.reshape(-1),
            "gercek_eczane_stogu": stok_3d[aktif_p, aktif_s, :].T.reshape(-1),
            "karsilanmayan_hasta_talebi": karsilanmayan_3d[aktif_p, aktif_s, :].T.reshape(-1),
            "cesitte_var": cesit_3d[aktif_p, aktif_s, :].T.reshape(-1),
            "latent_tuketim_hizi": np.round(
                np.tile(lam_base[aktif_p, aktif_s], W), 6),
        }
    )

    sow_df = pl.DataFrame(
        {
            "hafta": np.repeat(np.arange(W, dtype=np.int32), P),
            "eczane_id": np.tile(ecz_id, W),
            "share_of_wallet": np.round(sow_kayit.T.reshape(-1), 5),
        }
    )

    tukenme = pl.DataFrame(
        {
            "gercek_tukenme_haftasi": np.array([t[0] for t in tukenme_kayit], dtype=np.int32),
            "eczane_id": ecz_id[[t[1] for t in tukenme_kayit]] if tukenme_kayit else [],
            "sku_id": sku_id[[t[2] for t in tukenme_kayit]] if tukenme_kayit else [],
        }
    )

    return DunyaCiktisi(
        takvim=takvim, urunler=urunler, eczaneler=ecz.master,
        olaylar_gozlemlenebilir=olay.gozlemlenebilir, olaylar_gercek=olay.gercek,
        siparisler=siparisler, sevkiyat_satirlari=sevkiyat, stok_lotlari=stok_lotlari,
        depo_stok_haftalik=depo_stok, imhalar=imhalar, iadeler=iadeler,
        urun_fiyat_haftalik=fiyat, makro_haftalik=makro,
        latent_urun=latent_urun, latent_eczane=ecz.latent,
        hucre_haftalik=hucre, rakip_siparisleri=rakip, sow_haftalik=sow_df,
        tukenme_olaylari=tukenme,
    )
