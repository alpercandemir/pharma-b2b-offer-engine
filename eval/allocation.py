"""M5 olcumu: karsilama, stockout, imha/iade ve memnuniyet proxy'si.

SENTETIK OLDUGU ICIN SONUCU BILIYORUZ. Politika TAHMIN EDILEN kabul
olasiligiyla plan yapar (policy/allocate.py); burasi GERCEK olasiligi
(sim/response.py) ornekler ve plan ile gerceklesenin arasindaki farki olcer.
Stockout tam olarak bu farktan dogar: LP stogu beklenen cekilise gore ayirir,
gerceklesen cekilis ondan sapar.

UC AYRI SONUC KATMANI, UCU DE AYRI OLCULUR
==========================================
1. KARSILAMA (cikis kriteri (a)):
   Kabul edilen teklifler oncelik sirasinda lottan cekilir. Lot bitince kalan
   talep KARSILANMAZ. `ranking_only` ayni lotu birden fazla eczaneye soz
   verdigi icin burada acik verir; LP'nin lot dengesi kisiti bunu yapisal
   olarak engeller ve geriye yalnizca TAHMIN HATASI kaynakli acik kalir.

2. ECZANE TARAFI (cikis kriteri (b)): teslim edilen mal eczanede de yaslanir.
       satilabilir = gunluk_hiz x (lot_kalan_gun - eczaci_guvenlik_marji)
                     - eczanenin ELINDEKI stok
   Fazlasi IADE olur; iadenin `depoya_iade_orani` kadari bize doner ve imha
   edilir. Kor iskontonun zarari azaltmayip TRANSFER ettigi yer burasi.
   Bu hesap sim/lots.py `satilamayacagi_bosalt`in tek donemlik karsiligidir
   ve GERCEK (latent) tuketim hizini kullanir - politikanin tahminini degil.

3. DEPO TARAFI (cikis kriteri (b)): teklif edilmeyip lotta kalan adet miadina
   kadar ORGANIK talebin ne kadarini gorur? Projeksiyon SKU'nun gozlemlenen
   sevk hizindan FEFO sirasinda yapilir; gorulemeyen kisim IMHA sayilir.

MARJ MUHASEBESI (raporda da acikca yazili)
==========================================
    net_marj = satilan adedin brut marji                      (M4 tabani)
             - iade edilen adedin marji                       (satis geri alinir)
             - donen adet   x dsf x imha_orani                (iade islem maliyeti)
             - depoda imha  x dsf x imha_orani                (imha maliyeti)
SPEC 2.5 miad sonrasi degeri "-imha_maliyeti" olarak tanimliyor; bu yuzden
imha kaleminde islem maliyeti var, lotun DEFTER DEGERI yok. Defter degeri
ayri bir teshis kolonu olarak (`imha_batik_tutari`) raporlanir ki hangi
tabanda okundugu tartisilabilir olsun.

Bu modul ground_truth okur ve YALNIZCA olcumde kullanilir; ciktisi hicbir
zaman politika ya da feature katmanina donmez (eval/oracle.py ile ayni sinir).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from core.config import Config
from core.rng import SeedBank
from policy.allocate import Kolonlar, LotGorunumu, TahsisSonucu
from policy.candidates import AdayDunyasi, OriginGorunumu
from policy.scorer import TEKLIF_YOK, TeklifMatrisleri
from sim.calendar import GUN_HAFTA
from sim.response import GercekDurum, Tepki

EPSILON = 1e-12


# --------------------------------------------------------------------------
# organik cikis projeksiyonu (depo tarafi)
# --------------------------------------------------------------------------
def organik_sevk_hizi(dunya: AdayDunyasi, cfg: Config, t: int) -> np.ndarray:
    """[S] SKU basina haftalik sevk hizi, GOZLEMLENEBILIR gecmisten.

    Depoda kalan adedin miadina kadar organik talep tarafindan tuketilip
    tuketilemeyecegini projelendirmek icin. Kapali dongu degil (o M6): bu bir
    PROJEKSIYONDUR ve politikalar arasinda aynidir, dolayisiyla karsilastirmayi
    yanlilamaz - yalnizca imha seviyesini kaydirir.
    """
    pencere = cfg.tahsis.degerlendirme.organik_cikis_pencere_hafta
    sec = (dunya.sevk_w <= t) & (dunya.sevk_w > t - pencere)
    toplam = np.zeros(dunya.S)
    np.add.at(toplam, dunya.sevk_s[sec], dunya.sevk_adet[sec])
    return toplam / float(min(pencere, t + 1))


def depo_imhasi(lotlar: LotGorunumu, kalan_adet: np.ndarray,
                sevk_hizi: np.ndarray) -> np.ndarray:
    """[L] lotta kalan adedin miadina kadar satilamayacak kismi.

    FEFO kumulatifi: bir lotun onunde duran (daha erken miatli) stok once
    satilir. j'inci lotun miadina kadar toplam satilabilir adet
    `hiz x kalan_gun/7`; ondan onceki lotlarin stogu dusulur.
    sim/lots.py `satilamayacagi_bosalt` ile ayni mantik, depo tarafinda.
    """
    imha = np.zeros(lotlar.L)
    for sku, idx in lotlar.sku_lotlari.items():
        sira = idx[np.argsort(lotlar.kalan_gun[idx], kind="stable")]   # FEFO
        stok = kalan_adet[sira]
        kapasite = sevk_hizi[sku] * np.maximum(lotlar.kalan_gun[sira], 0.0) / GUN_HAFTA
        onceki = np.cumsum(stok) - stok
        satilabilir = np.clip(kapasite - onceki, 0.0, stok)
        imha[sira] = stok - satilabilir
    return imha


# --------------------------------------------------------------------------
# olcum
# --------------------------------------------------------------------------
@dataclass
class TahsisOlcumu:
    """Bir politikanin bir origin kumesindeki sonucu. Ortalamalar tekrar uzeri."""

    ad: str
    teklif_sayisi: float
    planlanan_cekilis: float          # LP/acgozlunun ayirdigi beklenen adet
    beklenen_artimsal_marj: float     # M4 tabani, deterministik (kiyas noktasi)
    # --- karsilama (a) ---
    kabul_sayisi: float
    talep_adet: float
    karsilanan_adet: float
    karsilanmayan_adet: float
    stockout_sayisi: float
    stockout_eczane_sayisi: float
    # --- miad (b) ---
    teslim_adet: float
    satilan_adet: float
    iade_adet: float
    depoya_donen_adet: float
    imha_adet: float                  # depo lotlarinda miadinda kalan
    imha_adet_temizlik: float         # bunun temizlik penceresindeki lotlardan gelen kismi
    teslim_adet_temizlik: float       # temizlik penceresindeki lottan teslim edilen
    iade_imhasi_adet: float           # iadeden gelen imha
    ortalama_teslim_raf_omru: float
    # --- para ---
    brut_marj: float
    iade_geri_alinan_marj: float
    imha_islem_maliyeti: float
    iade_islem_maliyeti: float
    iade_kredi_tutari: float          # teshis: eczaneye odenen nakit
    imha_batik_tutari: float          # teshis: imha edilen adedin defter degeri
    net_marj: float
    # --- memnuniyet ---
    memnuniyet_proxy: float
    sow_kaybi_iade: float
    sow_kaybi_stoksuzluk: float
    # --- D9 davranis kaniti ---
    teklif_sayisi_temizlik: float = 0.0
    ortalama_mf_temizlik: float = float("nan")
    ortalama_mf_normal: float = float("nan")
    # --- kolon suzgeclerinin vakum kontrolu ---
    elenen: dict = field(default_factory=dict)
    # --- LP teshisi ---
    lp_teklif_degeri: float = float("nan")
    butunluk_acigi: float = float("nan")
    kesirli_sutun: float = float("nan")
    kolon_sayisi: float = float("nan")
    sapma: dict = field(default_factory=dict)   # tekrarlar arasi standart sapma

    @property
    def karsilama_orani(self) -> float:
        return self.karsilanan_adet / self.talep_adet if self.talep_adet else float("nan")


ORTALANAN = (
    "kabul_sayisi", "talep_adet", "karsilanan_adet", "karsilanmayan_adet",
    "stockout_sayisi", "stockout_eczane_sayisi", "teslim_adet", "satilan_adet",
    "iade_adet", "depoya_donen_adet", "imha_adet", "imha_adet_temizlik",
    "teslim_adet_temizlik", "iade_imhasi_adet",
    "ortalama_teslim_raf_omru", "brut_marj", "iade_geri_alinan_marj",
    "imha_islem_maliyeti", "iade_islem_maliyeti", "iade_kredi_tutari",
    "imha_batik_tutari", "net_marj", "memnuniyet_proxy", "sow_kaybi_iade",
    "sow_kaybi_stoksuzluk",
)


def _tekrar(cfg: Config, dunya: AdayDunyasi, gor: OriginGorunumu,
            lotlar: LotGorunumu, teklifler: pl.DataFrame, mat: TeklifMatrisleri,
            tepki: Tepki, sonuc: TahsisSonucu, kolonlar: Kolonlar,
            durum: GercekDurum, sevk_hizi: np.ndarray,
            rng: np.random.Generator) -> dict:
    """Tek bir karsilama ornegi. Butun sayilar bu origin'e ait."""
    ia = cfg.sim.iade
    ts = cfg.sim.tedarikci_secimi
    n = sonuc.kol.size
    idx = np.arange(n)
    teklif = sonuc.teklif_maskesi
    p_idx = teklifler["eczane_idx"].to_numpy()
    s_idx = teklifler["sku_idx"].to_numpy()
    dsf = dunya.dsf[s_idx]

    # --- kabul ve miktar cekilisi (gercek olasilikla) ---
    p_gercek = tepki.olasilik[idx, sonuc.kol]
    kabul = teklif & (rng.random(n) < p_gercek)
    sigma = cfg.tepki.miktar.kabul_gurultu_sigma
    carpan = (np.maximum(cfg.tepki.miktar.asgari_kabul_orani,
                         np.exp(rng.normal(-0.5 * sigma * sigma, sigma, n)))
              if sigma > 0 else np.ones(n))

    nominal = mat.adet[idx, sonuc.kol] + mat.bedava[idx, sonuc.kol]
    talep = np.where(kabul, nominal * carpan, 0.0)

    # --- lottan karsilama: politikanin oncelik sirasinda ---
    # Baslangic kapasitesi lotun HAM kalani; once teklif ALMAYAN satirlarin
    # organik siparisi dusulur (teklif alan satirda organik siparis yerini
    # teklife birakir, iki kez sayilmaz). Bu, LP'nin artimsal stok
    # muhasebesinin (`taban_talebini_dus`) olcum tarafindaki birebir esidir.
    kalan = lotlar.ham_adet.copy()
    if lotlar.taban_satir.size == n:
        tekliftsiz = ~teklif & (lotlar.taban_satir_lot >= 0)
        np.add.at(kalan, lotlar.taban_satir_lot[tekliftsiz],
                  -lotlar.taban_satir[tekliftsiz])
        kalan = np.maximum(kalan, 0.0)
    karsilanan = np.zeros(n)
    kazanc = np.where(sonuc.kolon >= 0, kolonlar.kazanc[np.maximum(sonuc.kolon, 0)], 0.0)
    for i in np.flatnonzero(kabul)[np.argsort(-kazanc[kabul], kind="stable")]:
        l = int(sonuc.lot[i])
        if l < 0:
            continue
        ver = min(talep[i], kalan[l])
        kalan[l] -= ver
        karsilanan[i] = ver
    karsilanmayan = talep - karsilanan
    stockout = karsilanmayan > EPSILON

    pay = np.where(talep > EPSILON, karsilanan / np.maximum(talep, EPSILON), 0.0)
    brut = mat.marj[idx, sonuc.kol] * carpan * pay * kabul

    # --- eczane tarafi: teslim edilen mal orada da yaslanir ---
    lot_gun = np.where(sonuc.lot >= 0, lotlar.kalan_gun[np.maximum(sonuc.lot, 0)], 0.0)
    ecz_stok, latent_hiz, _ = durum.hucre_durumu(
        teklifler["eczane_id"].to_numpy(), teklifler["sku_id"].to_numpy(), gor.t)
    gunluk = latent_hiz / GUN_HAFTA
    kapasite = gunluk * np.maximum(lot_gun - ia.eczaci_guvenlik_marji_gun, 0.0)
    satilabilir = np.maximum(kapasite - ecz_stok, 0.0)
    iade = np.maximum(karsilanan - satilabilir, 0.0)
    satilan = karsilanan - iade
    donen = iade * ia.depoya_iade_orani

    satis_payi = np.where(karsilanan > EPSILON, satilan / np.maximum(karsilanan, EPSILON), 0.0)
    brut_satilan = brut * satis_payi
    geri_alinan = brut - brut_satilan

    # --- depo tarafi: teklif edilmeyip kalan adet miadina kadar satilir mi ---
    imha_lot = depo_imhasi(lotlar, kalan, sevk_hizi)
    # Politikanin FIILEN oynatabilecegi imha: temizlik penceresindeki lotlar.
    # Toplam imha yapisal fazla stoktan da beslenir ve politikalar arasinda
    # neredeyse sabittir; (b) karsilastirmasinin okunabilir paydasi budur.
    pencerede = lotlar.kalan_gun < cfg.tahsis.temizlik.tetik_gun
    imha_dsf = dunya.dsf[lotlar.sku_idx]
    imha_orani = cfg.lot.maliyet.imha_birim_maliyeti_dsf_orani

    # --- memnuniyet: dunyanin SOW cezalarinin tek donemlik karsiligi ---
    teslim_p = np.bincount(p_idx, weights=karsilanan, minlength=dunya.P)
    iade_p = np.bincount(p_idx, weights=iade, minlength=dunya.P)
    talep_p = np.bincount(p_idx, weights=talep, minlength=dunya.P)
    eksik_p = np.bincount(p_idx, weights=karsilanmayan, minlength=dunya.P)
    dokunulan = talep_p > EPSILON
    iade_cezasi = ia.sow_cezasi * np.minimum(
        np.where(teslim_p > EPSILON, iade_p / np.maximum(teslim_p, EPSILON), 0.0), 1.0)
    stoksuz_cezasi = ts.stoksuzluk_sow_cezasi * np.minimum(
        np.where(dokunulan, eksik_p / np.maximum(talep_p, EPSILON), 0.0), 1.0)
    memnuniyet = (1.0 - float((iade_cezasi + stoksuz_cezasi)[dokunulan].mean())
                  if dokunulan.any() else float("nan"))

    iade_islem = float((donen * dsf).sum() * imha_orani)
    imha_islem = float((imha_lot * imha_dsf).sum() * imha_orani)
    net = float(brut_satilan.sum()) - iade_islem - imha_islem

    return {
        "kabul_sayisi": float(kabul.sum()),
        "talep_adet": float(talep.sum()),
        "karsilanan_adet": float(karsilanan.sum()),
        "karsilanmayan_adet": float(karsilanmayan.sum()),
        "stockout_sayisi": float(stockout.sum()),
        "stockout_eczane_sayisi": float((eksik_p > EPSILON).sum()),
        "teslim_adet": float(karsilanan.sum()),
        "satilan_adet": float(satilan.sum()),
        "iade_adet": float(iade.sum()),
        "depoya_donen_adet": float(donen.sum()),
        "imha_adet": float(imha_lot.sum()),
        "imha_adet_temizlik": float(imha_lot[pencerede].sum()),
        "teslim_adet_temizlik": float(
            karsilanan[(sonuc.lot >= 0) & pencerede[np.maximum(sonuc.lot, 0)]].sum()),
        "iade_imhasi_adet": float(donen.sum()),
        "ortalama_teslim_raf_omru": float(
            (lot_gun * karsilanan).sum() / max(karsilanan.sum(), EPSILON)),
        "brut_marj": float(brut.sum()),
        "iade_geri_alinan_marj": float(geri_alinan.sum()),
        "imha_islem_maliyeti": imha_islem,
        "iade_islem_maliyeti": iade_islem,
        "iade_kredi_tutari": float((donen * dsf).sum() * ia.kredi_orani),
        "imha_batik_tutari": float((imha_lot * lotlar.birim_maliyet).sum()),
        "net_marj": net,
        "memnuniyet_proxy": memnuniyet,
        "sow_kaybi_iade": float(iade_cezasi[dokunulan].sum()),
        "sow_kaybi_stoksuzluk": float(stoksuz_cezasi[dokunulan].sum()),
    }


def politika_olcumu(cfg: Config, dunya: AdayDunyasi, gor: OriginGorunumu,
                    lotlar: LotGorunumu, teklifler: pl.DataFrame,
                    mat: TeklifMatrisleri, tepki: Tepki, sonuc: TahsisSonucu,
                    kolonlar: Kolonlar, durum: GercekDurum,
                    sevk_hizi: np.ndarray, carpan: float) -> dict:
    """Bir (origin, politika) icin tekrarlanmis karsilama olcumu.

    Tekrar sayisi `tahsis.degerlendirme.ornek_sayisi`. Seed origin ve politika
    adina gomulu: ayni kosu iki kez calisinca ayni sayilar.
    """
    d = cfg.tahsis.degerlendirme
    idx = np.arange(sonuc.kol.size)
    # Seed origin ve politika adina gomulu (core/rng.py disiplini): ayni kosu
    # iki kez calisinca ayni ornekler, farkli politikalar ORTAK carpanlar.
    seedler = SeedBank(d.ornek_seed)
    ornekler = []
    for r in range(d.ornek_sayisi):
        rng = seedler.generator(f"karsilama_{gor.t}_{sonuc.politika}_{r}")
        ornekler.append(_tekrar(cfg, dunya, gor, lotlar, teklifler, mat, tepki,
                                sonuc, kolonlar, durum, sevk_hizi, rng))

    ozet = {ad: float(np.mean([o[ad] for o in ornekler])) for ad in ORTALANAN}
    sapma = {ad: float(np.std([o[ad] for o in ornekler], ddof=1))
             for ad in ORTALANAN} if d.ornek_sayisi > 1 else {}

    # Deterministik kiyas noktasi: M4'un olctugu beklenen artimsal marj.
    # Karsilama ve miad KATILMAZ - M5'in ekledigi seyin ne oldugu ancak bu
    # sayiyla `net_marj` arasindaki farkta gorunur.
    beklenen = float(((tepki.olasilik[idx, sonuc.kol] * mat.marj[idx, sonuc.kol]
                       - tepki.olasilik[:, TEKLIF_YOK] * mat.marj[:, TEKLIF_YOK])
                      * carpan)[sonuc.teklif_maskesi].sum())
    secilen = sonuc.kolon[sonuc.kolon >= 0]

    # D9'un DAVRANIS kaniti: negatif gölge fiyat rejiminde ayni politikanin
    # MF derinligi artmali. Isaret degisimi bir tabloda gorunur (gölge fiyat),
    # etkisi burada gorunur - "normalde irrasyonel bir MF derinligi burada
    # rasyonel hale gelir" cumlesinin sayisi.
    teklif = sonuc.teklif_maskesi
    pencerede = np.zeros(sonuc.kol.size, dtype=bool)
    var = sonuc.lot >= 0
    pencerede[var] = lotlar.kalan_gun[sonuc.lot[var]] < cfg.tahsis.temizlik.tetik_gun
    mf = mat.uzay.mf[sonuc.kol]
    ozet.update({
        "teklif_sayisi": float(teklif.sum()),
        "teklif_sayisi_temizlik": float((teklif & pencerede).sum()),
        "ortalama_mf_temizlik": float(mf[teklif & pencerede].mean())
        if (teklif & pencerede).any() else float("nan"),
        "ortalama_mf_normal": float(mf[teklif & ~pencerede].mean())
        if (teklif & ~pencerede).any() else float("nan"),
        "planlanan_cekilis": float(kolonlar.cekilen[secilen].sum()),
        "elenen": dict(kolonlar.elenen),
        "beklenen_artimsal_marj": beklenen,
        "lp_teklif_degeri": sonuc.lp_teklif_degeri,
        "butunluk_acigi": sonuc.lp_teklif_degeri - sonuc.yuvarlanmis_deger,
        "kesirli_sutun": float(sonuc.kesirli_sutun),
        "kolon_sayisi": float(sonuc.kolon_sayisi),
    })
    return {"ozet": ozet, "sapma": sapma}


def _agirlikli(parcalar: list[dict], alan: str, agirlik_alani: str) -> float:
    """Origin'ler boyunca teklif sayisiyla agirlikli ortalama (NaN'lar atlanir)."""
    v = np.array([p["ozet"][alan] for p in parcalar], dtype=float)
    w = np.array([p["ozet"][agirlik_alani] for p in parcalar], dtype=float)
    gecerli = np.isfinite(v) & (w > 0)
    if not gecerli.any():
        return float("nan")
    return float((v[gecerli] * w[gecerli]).sum() / w[gecerli].sum())


def olcum_birlestir(ad: str, parcalar: list[dict]) -> TahsisOlcumu:
    """Origin'ler boyunca toplama. Marj ve adetler TOPLANIR, oranlar agirlikli.

    Politika degeri bir TOPLAM buyuklugudur (M4 ile ayni disiplin); origin
    sayisi tabloda yazili. Memnuniyet ve ortalama raf omru orandir: teslim
    adediyle agirliklandirilir.
    """
    toplanan = {ad_: sum(p["ozet"][ad_] for p in parcalar) for ad_ in ORTALANAN}
    agirlik = np.array([p["ozet"]["teslim_adet"] for p in parcalar])
    for oran in ("ortalama_teslim_raf_omru",):
        v = np.array([p["ozet"][oran] for p in parcalar])
        toplanan[oran] = float((v * agirlik).sum() / max(agirlik.sum(), EPSILON))
    toplanan["memnuniyet_proxy"] = float(
        np.nanmean([p["ozet"]["memnuniyet_proxy"] for p in parcalar]))

    sapma = {}
    if parcalar and parcalar[0]["sapma"]:
        # Origin'ler bagimsiz: varyanslar toplanir.
        for ad_ in ORTALANAN:
            sapma[ad_] = float(np.sqrt(sum(p["sapma"][ad_] ** 2 for p in parcalar)))

    return TahsisOlcumu(
        ad=ad,
        teklif_sayisi=sum(p["ozet"]["teklif_sayisi"] for p in parcalar),
        teklif_sayisi_temizlik=sum(p["ozet"]["teklif_sayisi_temizlik"] for p in parcalar),
        ortalama_mf_temizlik=_agirlikli(parcalar, "ortalama_mf_temizlik",
                                        "teklif_sayisi_temizlik"),
        ortalama_mf_normal=_agirlikli(parcalar, "ortalama_mf_normal", "teklif_sayisi"),
        planlanan_cekilis=sum(p["ozet"]["planlanan_cekilis"] for p in parcalar),
        elenen={ad_: sum(p["ozet"]["elenen"].get(ad_, 0) for p in parcalar)
                for ad_ in sorted({k for p in parcalar for k in p["ozet"]["elenen"]})},
        beklenen_artimsal_marj=sum(p["ozet"]["beklenen_artimsal_marj"] for p in parcalar),
        lp_teklif_degeri=float(np.sum([p["ozet"]["lp_teklif_degeri"] for p in parcalar])),
        butunluk_acigi=float(np.sum([p["ozet"]["butunluk_acigi"] for p in parcalar])),
        kesirli_sutun=float(np.sum([p["ozet"]["kesirli_sutun"] for p in parcalar])),
        kolon_sayisi=float(np.sum([p["ozet"]["kolon_sayisi"] for p in parcalar])),
        sapma=sapma,
        **{k: v for k, v in toplanan.items()},
    )


# --------------------------------------------------------------------------
# gölge fiyat ozeti
# --------------------------------------------------------------------------
def golge_fiyat_tablosu(cfg: Config, dunya: AdayDunyasi, lotlar: LotGorunumu,
                        sonuc: TahsisSonucu) -> pl.DataFrame:
    """Lot basina gölge fiyat + devam degeri + rejim etiketi.

    D9'un gorulecegi tablo budur: `kalan_gun < isaret_esigi` olan lotlarda
    `golge_fiyat` NEGATIF olmali. Isaret degisimi bir iddia degil, bu
    tablodan okunan bir sayidir.
    """
    if lotlar.L == 0:
        return pl.DataFrame()
    tetik = cfg.tahsis.temizlik.tetik_gun
    return pl.DataFrame({
        "origin": np.full(lotlar.L, sonuc.t, dtype=np.int32),
        "politika": [sonuc.politika] * lotlar.L,
        "lot_id": lotlar.lot_id.astype(str),
        "sku_idx": lotlar.sku_idx,
        "kalan_gun": lotlar.kalan_gun,
        "adet": lotlar.adet,
        "dsf": dunya.dsf[lotlar.sku_idx],
        "devam_degeri": sonuc.lot_devam_degeri,
        "golge_fiyat": sonuc.golge_fiyat,
        "temizlik_penceresinde": lotlar.kalan_gun < tetik,
    })


def golge_fiyat_ozeti(tablo: pl.DataFrame) -> dict:
    """Sweep tablosuna giren duz gölge fiyat metrikleri."""
    if tablo.height == 0 or tablo["golge_fiyat"].is_null().all():
        return {}
    g = tablo["golge_fiyat"].to_numpy().astype(float)
    adet = tablo["adet"].to_numpy().astype(float)
    pencerede = tablo["temizlik_penceresinde"].to_numpy().astype(bool)
    sonlu = np.isfinite(g)
    if not sonlu.any():
        return {}
    negatif = sonlu & (g < 0)
    return {
        "golge_ortalama": float(g[sonlu].mean()),
        "golge_medyan": float(np.median(g[sonlu])),
        "golge_azami": float(g[sonlu].max()),
        "golge_asgari": float(g[sonlu].min()),
        "negatif_golge_lot_orani": float(negatif.sum() / sonlu.sum()),
        "negatif_golge_adet": float(adet[negatif].sum()),
        "temizlik_penceresi_lot_orani": float(pencerede[sonlu].mean()),
        "temizlik_penceresinde_negatif_orani": float(
            negatif[pencerede & sonlu].sum() / max((pencerede & sonlu).sum(), 1)),
        "pencere_disi_negatif_lot": float((negatif & ~pencerede).sum()),
    }
