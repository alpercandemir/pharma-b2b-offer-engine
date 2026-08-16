"""Kit stok altinda tahsis (D5) ve miad rejimi (D9). SPEC M5 + 2.5.

D5: SKORLAMA ILE TAHSIS AYRI KATMANLARDIR
=========================================
M4 "bu satira hangi aksiyon" sorusunu cozdu ve her satiri BAGIMSIZ ele aldi.
Stok paylasilan bir kaynak: ayni lotu 200 eczaneye soz vermek bir siralama
hatasi degil, bir MUHASEBE hatasidir. Bu dosya o kaynagi paylastirir.

    max  sum_{i,l,a} c[i,l,a] x[i,l,a]  +  sum_l v_l r_l
    s.t. sum_{l,a} x[i,l,a]        <= 1          her aday satiri i
         sum_{i,a} u[i,l,a] x + r_l = S_l        her lot l   <- KIT KAYNAK
         sum_{i in p} x            <= tavan      her eczane p (frekans)
         sum_{i in p} tutar x      <= limit_p    her eczane p (DBS, D6)
         x >= 0,  r_l >= 0

`x[i,l,a]` = i satirina l lotundan a kolunu verme orani (LP gevsetmesi;
tam sayili politikaya yuvarlama asagida). `r_l` = lotta ELDE KALAN adet.

D9: TEMIZLIK AYRI BIR MOTOR DEGIL, BU LP'NIN BIR REJIMI
=======================================================
Butun mekanizma `v_l`de, lotun BIRIM DEVAM DEGERINDE:

    kalan_gun >= tetik_gun  ->  v = +dsf * depo_marji * realizasyon   (varlik)
    temizlik penceresi      ->  v = deger_egrisi ile azalir
    miad sonrasi            ->  v = -dsf * imha_orani                 (yukumluluk)

Lot kisiti bir ESITLIKTIR, bu yuzden duali ISARETSIZDIR. `v_l > 0` iken elde
kalan adet oduldur, gölge fiyat pozitiftir ve LP stogu KORUR. `v_l < 0` iken
elde kalan adet cezadir, gölge fiyat NEGATIFE DONER ve LP adedi disari itmek
icin marjdan taviz vermeye razi olur - normalde irrasyonel bir MF derinligi
burada rasyonel hale gelir. Ayri bir amac fonksiyonu, ayri bir kod yolu,
ayri bir "temizlik motoru" YOK. SPEC 2.5'in "ayni LP, isaret degisimi"
cumlesinin birebir karsiligi budur ve `scripts/verify_m5.py` isaret degisimini
bir cikis kriteri olarak sinar.

M2 KUPLAJI (SPEC 2.5): TEMIZLIK BIR ISKONTO DEGIL, HEDEFLEME PROBLEMIDIR
========================================================================
    max_teklif_adedi = tuketim_hizi * (kalan_gun - eczaci_marji) * guvenlik
Tuketim hizi M2'nin ciktisidir (`OriginGorunumu.hiz_tahmini`, features/hiz.py).
SABIT BIR TAVAN YOK: 40 gun kalan 100 adet, haftada 50 satan eczane icin
sorun degil; haftada 1 satan eczane icin tamamen zayidir. Sonuc sifirsa o
eczane o lot icin ADAY DEGILDIR, iskonto ne kadar derin olursa olsun.

KISIT KATMANI BURADA DA VETO YETKISINI KORUR (D6)
=================================================
Aday kumesi M3'un vetodan gecmis satirlaridir; miad baskisi vetoyu ASMAZ
(kirmizi/yesil recete, tedarik guclugu). Temizlik rejiminin gevsettigi TEK
sey lot raf omru tabanidir ve o taban da sifir olamaz (core/config.py
`_m5_miad_kilidi`). Kredi limiti LP'nin kendi kisiti olarak girer.

Bu dosya ground_truth okumaz. Gercek tepki sim/response.py'de, karsilama ve
imha/iade olcumu eval/allocation.py'de.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

from core.config import Config
from policy.candidates import AdayDunyasi, OriginGorunumu
from policy.scorer import TEKLIF_YOK, TeklifMatrisleri
from sim.calendar import GUN_HAFTA

# Kesirli LP cozumunde "sifir sayilmayan" esik. Sayisal sabit, knob degil:
# HiGHS'in dondurdugu degerler 1e-12 mertebesinde artik tasiyor.
SIFIR_ESIGI = 1e-9
# Sifira bolme korumasi.
EPSILON = 1e-12


# --------------------------------------------------------------------------
# politika tanimlari
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TahsisPolitikasi:
    """Dort politika, TEK bir tahsis makinesinin dort ayari.

    Ayri kod yollari degil ayri parametrelestirmeler olmalari D9'un iddiasidir:
    "temizlik ayri bir motor degil, tahsis katmaninin bir rejimidir".
    """

    ad: str
    lp: bool                    # LP mi (True) yoksa acgozlu siralama mi
    stok_ayirma: bool           # acgozluyken lot stogu rezerve edilir mi
    temizlik_rejimi: bool       # kisa miatli lot teklif edilebilir mi
    temizlik_degeri: bool       # salvage egrisi (negatif gölge fiyat) devrede mi
    temizlik_hiz_kuplaji: bool  # kisa miatli kolonda SPEC 2.5 adet kisiti
    temizlik_derin_mf: bool     # kisa miatli kolonda en derin MF dayatilir mi


POLITIKALAR: dict[str, TahsisPolitikasi] = {
    # (a) KIT SKU: taban cizgisi. M4'un teslim ettigi politika - satir bazinda
    # en iyi kol, eczane basina frekans tavani, lot FEFO. Stok PAYLASTIRILMAZ:
    # ayni lot birden fazla eczaneye soz verilir. M3/M4'ten devralinan borc.
    "ranking_only": TahsisPolitikasi(
        ad="ranking_only", lp=False, stok_ayirma=False, temizlik_rejimi=False,
        temizlik_degeri=False, temizlik_hiz_kuplaji=True, temizlik_derin_mf=False),
    # (a) KIT SKU: LP. (b)'de ayni politika "temizlik yok" olarak okunur -
    # bu bir isim oyunu degil, D9'un kendisi: temizlik ayri motor olsaydi
    # burada iki farkli nesne olurdu.
    "lp": TahsisPolitikasi(
        ad="lp", lp=True, stok_ayirma=True, temizlik_rejimi=False,
        temizlik_degeri=False, temizlik_hiz_kuplaji=True, temizlik_derin_mf=False),
    # (b) KOR ISKONTO: sahanin refleksi. Kisa miatli lot ACILIR, en derin MF
    # dayatilir, adet tuketim hiziyla SINIRLANMAZ. Zarari azaltmaz, TRANSFER
    # eder: eczane satamaz, iade eder, iliski zarar gorur (SPEC 2.5).
    "kor_iskonto": TahsisPolitikasi(
        ad="kor_iskonto", lp=False, stok_ayirma=True, temizlik_rejimi=True,
        temizlik_degeri=False, temizlik_hiz_kuplaji=False, temizlik_derin_mf=True),
    # (b) HEDEFLI TEMIZLIK: ayni LP, negatif gölge fiyat rejimi + M2 kuplaji.
    "hedefli_temizlik": TahsisPolitikasi(
        ad="hedefli_temizlik", lp=True, stok_ayirma=True, temizlik_rejimi=True,
        temizlik_degeri=True, temizlik_hiz_kuplaji=True, temizlik_derin_mf=False),
}


# --------------------------------------------------------------------------
# lot gorunumu (senaryo uygulanmis)
# --------------------------------------------------------------------------
@dataclass
class LotGorunumu:
    """Origin'de tahsis edilebilir lotlar, senaryo katsayilari uygulanmis.

    Senaryo dunyayi DEGISTIRMEZ (dunya_hash sabit kalir); LP'ye verilen
    gorunumu degistirir. D3'un "kur tahmin edilmez, senaryolastirilir"
    disiplininin stok tarafindaki karsiligi.
    """

    lot_id: np.ndarray        # [L]
    sku_idx: np.ndarray       # [L]
    adet: np.ndarray          # [L] TEKLIF PROGRAMINA kalan adet (taban dusulmus)
    ham_adet: np.ndarray      # [L] senaryo sonrasi lot kalani (taban dusulmeden)
    taban_talebi: np.ndarray  # [L] teklif verilmese de gelecek organik cekilis
    # Taban cekilisin SATIR bazli hali. Origin'e ait oldugu icin burada durur;
    # eval/allocation.py teklif ALMAYAN satirlarin taban talebini lottan duser
    # (teklif alan satirda organik siparis yerini teklife birakir).
    taban_satir: np.ndarray   # [n] satir basina beklenen taban cekilis
    taban_satir_lot: np.ndarray  # [n] o cekilisin gittigi lot indeksi, -1 = yok
    kalan_gun: np.ndarray     # [L] senaryo sonrasi kalan raf omru
    birim_maliyet: np.ndarray  # [L] lotun defter birim maliyeti (imha teshisi)
    birim_deger: np.ndarray   # [L] normal rejim devam degeri (TL/adet)
    salvage: np.ndarray       # [L] temizlik rejimi devam degeri (TL/adet)
    sku_lotlari: dict[int, np.ndarray]   # sku_idx -> FEFO sirali lot indeksleri
    lot_sirasi: dict[str, int]

    @property
    def L(self) -> int:
        return self.lot_id.size

    @property
    def canli(self) -> np.ndarray:
        return self.kalan_gun > 0.0


def _deger_egrisi(cfg: Config, u: np.ndarray) -> np.ndarray:
    """[0, 1] normalize kalan omurden [0, 1] deger payi. SPEC 5 salvage_curve."""
    t = cfg.tahsis.temizlik
    if t.deger_egrisi == "lineer":
        return u
    if t.deger_egrisi == "eksponansiyel":
        return u ** t.egri_ussu
    return (u >= t.basamak_esigi).astype(float)


def salvage_degeri(cfg: Config, dsf: np.ndarray, depo_marji: np.ndarray,
                   kalan_gun: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(normal_birim_deger, temizlik_salvage) - ikisi de TL/adet.

    `normal`: lot temizlik penceresinin disindaymis gibi degerlenir. Politika
              temizligi bilmiyorsa gordugu deger budur ve HER ZAMAN POZITIFTIR.
    `salvage`: SPEC 2.5'in dinamik degeri. Miad sonrasi -imha_maliyeti, temizlik
              penceresinde egriyle azalir, penceredisinda normale esittir.

    Isaret degisim noktasi lineer egride kapali formda:
        v = 0  <=>  kalan_gun / tetik = imha / (normal + imha)
    Bu oran `raporlanabilir_isaret_esigi` ile disariya verilir; temizlik
    penceresinin nerede yukumluluge dondugu bir iddia degil, bir sayidir.
    """
    t = cfg.tahsis.temizlik
    normal = dsf * depo_marji * t.normal_realizasyon_orani
    imha = dsf * t.imha_birim_maliyeti_dsf_orani
    u = np.clip(kalan_gun / t.tetik_gun, 0.0, 1.0)
    pencerede = kalan_gun < t.tetik_gun
    egri = _deger_egrisi(cfg, u)
    salvage = np.where(pencerede, -imha + (normal + imha) * egri, normal)
    return normal, salvage


def isaret_esigi_gun(cfg: Config, dsf: float, depo_marji: float) -> float:
    """Salvage'in sifiri kestigi kalan gun (lineer egri icin kapali form)."""
    t = cfg.tahsis.temizlik
    normal = dsf * depo_marji * t.normal_realizasyon_orani
    imha = dsf * t.imha_birim_maliyeti_dsf_orani
    if normal + imha <= 0:
        return float("nan")
    pay = imha / (normal + imha)
    if t.deger_egrisi == "eksponansiyel":
        pay = pay ** (1.0 / t.egri_ussu)
    elif t.deger_egrisi == "basamakli":
        pay = t.basamak_esigi
    return float(t.tetik_gun * pay)


def lot_gorunumu(dunya: AdayDunyasi, cfg: Config, gor: OriginGorunumu) -> LotGorunumu:
    """Origin gorunumundeki lotlara senaryoyu uygular ve degerlerini hesaplar."""
    s = cfg.tahsis.senaryo
    maliyet_haritasi = dict(zip(dunya.lot_id.astype(str), dunya.lot_birim_maliyet))
    lot_id, sku_idx, adet, kalan_gun, maliyet = [], [], [], [], []
    for sku, liste in sorted(gor.lotlar.items()):
        for lot in liste:                              # FEFO sirali geliyor
            lot_id.append(lot.lot_id)
            sku_idx.append(sku)
            adet.append(np.floor(lot.kalan_adet * s.kit_stok_carpani))
            kalan_gun.append(lot.kalan_gun - s.miad_hizlandirma_gun)
            maliyet.append(maliyet_haritasi[str(lot.lot_id)])
    lot_id = np.array(lot_id, dtype=object)
    sku_idx = np.array(sku_idx, dtype=np.int32)
    adet = np.array(adet, dtype=float)
    kalan_gun = np.array(kalan_gun, dtype=float)
    maliyet = np.array(maliyet, dtype=float)

    dsf = dunya.dsf[sku_idx] if sku_idx.size else np.zeros(0)
    marji = (dunya.urunler["depo_kar_marji"].to_numpy()[sku_idx] if sku_idx.size
             else np.zeros(0))
    normal, salvage = salvage_degeri(cfg, dsf, marji, kalan_gun)

    sku_lotlari: dict[int, np.ndarray] = {}
    for i, s_i in enumerate(sku_idx):
        sku_lotlari.setdefault(int(s_i), []).append(i)
    sku_lotlari = {k: np.array(v, dtype=np.int32) for k, v in sku_lotlari.items()}

    return LotGorunumu(lot_id=lot_id, sku_idx=sku_idx, adet=adet.copy(),
                       ham_adet=adet, taban_talebi=np.zeros_like(adet),
                       taban_satir=np.zeros(0), taban_satir_lot=np.zeros(0, dtype=np.int32),
                       kalan_gun=kalan_gun, birim_maliyet=maliyet,
                       birim_deger=normal, salvage=salvage,
                       sku_lotlari=sku_lotlari,
                       lot_sirasi={str(l): i for i, l in enumerate(lot_id)})


def taban_talebini_dus(dunya: AdayDunyasi, cfg: Config, lotlar: LotGorunumu,
                       teklifler: pl.DataFrame, mat: TeklifMatrisleri,
                       p_tahmin: np.ndarray, carpan: float) -> LotGorunumu:
    """Teklif verilmese de gelecek organik cekilisi lot stogundan duser.

    NEDEN ZORUNLU. LP'nin amac fonksiyonu ARTIMSAL marjdir (M4 ile ayni
    taban): "teklif ver" ile "verme" arasindaki fark. Stok muhasebesi de ayni
    tabanda olmak ZORUNDA. Aksi halde LP, artimsal bir kazanc icin BRUT bir
    stok bedeli oder ve sistematik olarak stogu tutmayi secer - ilk uygulamada
    tam olarak bu oldu: LP 900 slot yerine 47 teklif cikardi, cunku bir adedi
    bugun satmanin artimsal kazanci (kucuk) o adedin devam degerinden (tam
    marj) her zaman kucuktu.

    Taban cekilis satirin FEFO lotuna yazilir; kalan kapasite TEKLIF
    PROGRAMINA ayrilan stoktur ve butun politikalar icin AYNIDIR (dolayisiyla
    karsilastirmayi yanlilamaz).
    """
    if teklifler.height == 0 or lotlar.L == 0:
        return lotlar
    taban = p_tahmin[:, TEKLIF_YOK] * carpan * mat.adet[:, TEKLIF_YOK]
    s_idx = teklifler["sku_idx"].to_numpy()
    talep = np.zeros(lotlar.L)
    satir_lot = np.full(teklifler.height, -1, dtype=np.int32)
    for i in range(teklifler.height):
        adaylar = lotlar.sku_lotlari.get(int(s_idx[i]))
        if adaylar is None:
            continue
        canli = adaylar[lotlar.kalan_gun[adaylar] > 0.0]
        if canli.size:
            satir_lot[i] = int(canli[0])             # FEFO: en erken miatli lot
            talep[canli[0]] += taban[i]
    lotlar.taban_talebi = talep
    lotlar.taban_satir = taban
    lotlar.taban_satir_lot = satir_lot
    lotlar.adet = np.maximum(lotlar.ham_adet - talep, 0.0)
    return lotlar


# --------------------------------------------------------------------------
# kolon uretimi
# --------------------------------------------------------------------------
@dataclass
class Kolonlar:
    """LP'nin karar sutunlari: (aday satiri, lot, kol) ucluleri."""

    satir: np.ndarray       # [C]
    lot: np.ndarray         # [C]
    kol: np.ndarray         # [C]
    kazanc: np.ndarray      # [C] amac fonksiyonu katsayisi (TL)
    cekilen: np.ndarray     # [C] lottan BEKLENEN adet cekilisi
    nominal: np.ndarray     # [C] kabul halinde sevk edilecek adet (adet + bedava)
    tutar: np.ndarray       # [C] beklenen kredi kullanimi (TL)
    temizlik: np.ndarray    # [C] lot temizlik penceresinde mi
    # Suzgec basina elenen sutun sayisi. Vakum kontrolu icin: bir kisit
    # hic elemiyorsa dekoratiftir ve raporda oyle yazilmali.
    elenen: dict[str, int] = field(default_factory=dict)

    @property
    def C(self) -> int:
        return self.satir.size


def azami_teklif_adedi(cfg: Config, hiz_haftalik: np.ndarray,
                       kalan_gun: np.ndarray) -> np.ndarray:
    """SPEC 2.5 kuplaji. `hiz_haftalik` M2'nin tuketim hizi ciktisi (adet/hafta).

    Sonuc bir ADET tavanidir; sifir ya da negatifse o (eczane, lot) cifti
    aday degildir. Sabit gun esigi yerine bunun kullanilmasinin sebebi
    SPEC 2.5'te yazili: ayni miad, farkli hizdaki iki eczaneye ayni seyi
    soyleyemez.
    """
    t = cfg.tahsis.temizlik
    gunluk = hiz_haftalik / GUN_HAFTA
    return gunluk * np.maximum(kalan_gun - t.eczaci_marji_gun, 0.0) * t.guvenlik_katsayisi


def kolon_uret(dunya: AdayDunyasi, cfg: Config, gor: OriginGorunumu,
               lotlar: LotGorunumu, teklifler: pl.DataFrame,
               mat: TeklifMatrisleri, p_tahmin: np.ndarray, carpan: float,
               pol: TahsisPolitikasi) -> Kolonlar:
    """Politikanin gordugu butun (satir, lot, kol) secenekleri.

    Uygunluk suzgecleri, uygulanma sirasiyla:
      1. lot SKU'su satirin SKU'suyla ayni ve lot canli (kalan_gun > 0)
      2. raf omru tabani  - normal rejimde `kisit.asgari_kalan_raf_omru_gun`,
         temizlik rejiminde `temizlik.asgari_kalan_raf_omru_gun`; ikisi de
         soguk zincirde `soguk_zincir_raf_omru_carpani` ile carpilir (D6)
      3. kol izinli mi (M3/M4 kanal kisitlari - SGK'da MF kapali vs.)
      4. lotta fiziken yeterli adet var mi (teklif satiri TEK lot tasir)
      5. SPEC 2.5 adet tavani (`temizlik_hiz_kuplaji` acikken)
      6. `temizlik_derin_mf` acikken: temizlik penceresindeki lotta yalnizca
         en derin izinli MF kolu birakilir (kor iskontonun tanimi)
    """
    k = cfg.politika.kisit
    n, A = teklifler.height, mat.uzay.A
    if n == 0 or lotlar.L == 0:
        return _bos_kolonlar()

    s_idx = teklifler["sku_idx"].to_numpy()
    soguk = teklifler["soguk_zincir"].to_numpy().astype(bool)
    yeni = teklifler["yeni_hucre"].to_numpy().astype(float)
    hiz = teklifler["hiz_tahmini"].to_numpy() * cfg.politika.aday.hiz_telafi_katsayisi
    dsf = dunya.dsf[s_idx]

    carpan_soguk = np.where(soguk, k.soguk_zincir_raf_omru_carpani, 1.0)
    esik_normal = k.asgari_kalan_raf_omru_gun * carpan_soguk
    esik_temizlik = (cfg.tahsis.temizlik.asgari_kalan_raf_omru_gun * carpan_soguk
                     if pol.temizlik_rejimi else esik_normal)

    # Satir basina aday lotlar: FEFO sirasinda ilk `aday_lot_sayisi` uygun lot.
    satir_l, lot_l = [], []
    aday_lot = cfg.tahsis.lp.aday_lot_sayisi
    for i in range(n):
        adaylar = lotlar.sku_lotlari.get(int(s_idx[i]))
        if adaylar is None:
            continue
        uygun = adaylar[(lotlar.kalan_gun[adaylar] >= esik_temizlik[i])
                        & (lotlar.adet[adaylar] > 0.0)]
        for l in uygun[:aday_lot]:
            satir_l.append(i)
            lot_l.append(int(l))
    if not satir_l:
        return _bos_kolonlar()
    satir_l = np.array(satir_l, dtype=np.int32)
    lot_l = np.array(lot_l, dtype=np.int32)

    # (satir, lot) x kol capraz carpimi.
    kollar = np.arange(1, A)
    satir = np.repeat(satir_l, kollar.size)
    lot = np.repeat(lot_l, kollar.size)
    kol = np.tile(kollar, satir_l.size)

    nominal = mat.adet[satir, kol] + mat.bedava[satir, kol]
    p = p_tahmin[satir, kol]
    p0 = p_tahmin[satir, TEKLIF_YOK]
    # ARTIMSAL stok cekilisi: taban (teklif yok) cekilisi `taban_talebini_dus`
    # ile lot kapasitesinden zaten dusuldu; burada yalnizca teklifin EKLEDIGI
    # adet rezerve edilir. Marj artimsal, stok brut olsaydi LP stogu tutardi.
    cekilen = np.maximum(carpan * (p * nominal - p0 * mat.adet[satir, TEKLIF_YOK]), 0.0)
    kazanc = (p * carpan * mat.marj[satir, kol]
              - p0 * carpan * mat.marj[satir, TEKLIF_YOK])
    if cfg.tahsis.lp.sow_buyutme_agirligi > 0.0:
        # SPEC 5 "tahsis hedefi: kisa vadeli marj vs share-of-wallet buyutme".
        # share_of_wallet LATENT; prim gozlemlenebilir vekile (yeni hucre) bagli.
        kazanc = kazanc + (cfg.tahsis.lp.sow_buyutme_agirligi
                           * p * carpan * mat.adet[satir, kol] * dsf[satir]
                           * yeni[satir])
    tutar = p * carpan * mat.adet[satir, kol] * dsf[satir]
    temizlik_kolonu = lotlar.kalan_gun[lot] < cfg.tahsis.temizlik.tetik_gun

    elenen: dict[str, int] = {}
    tut = mat.izinli[satir, kol]
    elenen["kol_izinli_degil"] = int((~tut).sum())
    # (4) teklif satiri TEK lot referansi tasir: sevkiyatin tamami o lottan
    # cikar, o yuzden fizibilite BRUT adet uzerinden bakilir. LP'nin ARTIMSAL
    # stok muhasebesi bundan ayri bir sey.
    yeter = nominal <= lotlar.ham_adet[lot] * k.depo_stok_yeterlilik_carpani + SIFIR_ESIGI
    elenen["lot_yetersiz"] = int((tut & ~yeter).sum())
    tut &= yeter
    # (5) SPEC 2.5 adet tavani. VAKUM KONTROLU: `elenen["hiz_kuplaji"]` sifirsa
    # kuplaj dekoratiftir - "sabit tavan koyma" talimati bos kalmis demektir.
    kuple = np.ones(satir.size, dtype=bool) if pol.temizlik_hiz_kuplaji else ~temizlik_kolonu
    tavan = azami_teklif_adedi(cfg, hiz[satir], lotlar.kalan_gun[lot])
    sigar = ~kuple | (nominal <= tavan + SIFIR_ESIGI)
    elenen["hiz_kuplaji"] = int((tut & ~sigar).sum())
    elenen["hiz_kuplaji_temizlik"] = int((tut & ~sigar & temizlik_kolonu).sum())
    tut &= sigar
    # (6) kor iskonto: temizlik penceresinde yalnizca en derin izinli MF
    if pol.temizlik_derin_mf:
        derin = ~temizlik_kolonu | _en_derin_mf_maskesi(cfg, mat, satir, kol)
        elenen["derin_mf_dayatmasi"] = int((tut & ~derin).sum())
        tut &= derin

    return Kolonlar(satir=satir[tut], lot=lot[tut], kol=kol[tut],
                    kazanc=kazanc[tut], cekilen=cekilen[tut], nominal=nominal[tut],
                    tutar=tutar[tut], temizlik=temizlik_kolonu[tut], elenen=elenen)


def _en_derin_mf_maskesi(cfg: Config, mat: TeklifMatrisleri, satir: np.ndarray,
                         kol: np.ndarray) -> np.ndarray:
    """Satir basina en derin IZINLI MF orani + taban vade kolu.

    "Kor iskonto"nun tanimi: hedefleme yok, adet kuplaji yok, tek bir derin
    kampanya. MF kanali kapali satirda (SGK) en derin MF sifirdir ve kol
    fiilen taban vadeli bedelsiz teklife duser - SPEC 2.5'in "SGK'da temizlik
    MF yerine VADE ile yapilir" kuralinin kor karsiligidir.
    """
    mf = mat.uzay.mf
    taban = mat.uzay.vade == float(cfg.politika.aksiyon.taban_vade_gun)
    izinli = mat.izinli[satir]                       # [C', A]
    aday = izinli & taban[None, :]
    aday[:, TEKLIF_YOK] = False
    en_derin = np.max(np.where(aday, mf[None, :], -np.inf), axis=1)
    return (mf[kol] == en_derin) & taban[kol]


def _bos_kolonlar() -> Kolonlar:
    bos_i = np.zeros(0, dtype=np.int32)
    bos_f = np.zeros(0)
    return Kolonlar(satir=bos_i, lot=bos_i, kol=bos_i, kazanc=bos_f,
                    cekilen=bos_f, nominal=bos_f, tutar=bos_f,
                    temizlik=np.zeros(0, dtype=bool), elenen={})


# --------------------------------------------------------------------------
# sonuc
# --------------------------------------------------------------------------
@dataclass
class TahsisSonucu:
    """Bir origin'de bir politikanin teslim ettigi tahsis."""

    politika: str
    t: int
    kol: np.ndarray             # [n] 0 = teklif yok
    lot: np.ndarray             # [n] lot indeksi, -1 = yok
    kolon: np.ndarray           # [n] secilen kolonun indeksi, -1 = yok
    golge_fiyat: np.ndarray     # [L] lot basina gölge fiyat (TL/adet)
    lot_devam_degeri: np.ndarray  # [L] LP'ye verilen v_l
    slot_golge: np.ndarray      # [P] frekans tavani gölge fiyati
    kredi_golge: np.ndarray     # [P] DBS limiti gölge fiyati
    lp_degeri: float            # LP amac degeri (r_l dahil)
    lp_teklif_degeri: float     # yalnizca teklif kolonlarinin katkisi
    yuvarlanmis_deger: float    # tam sayili politikanin ayni amaca gore degeri
    kesirli_sutun: int          # 0 < x < 1 olan sutun sayisi
    kolon_sayisi: int
    durum: str

    @property
    def teklif_maskesi(self) -> np.ndarray:
        return self.kol != TEKLIF_YOK


def _bos_sonuc(pol: TahsisPolitikasi, t: int, n: int, L: int,
               P: int) -> TahsisSonucu:
    return TahsisSonucu(
        politika=pol.ad, t=t, kol=np.zeros(n, dtype=np.int32),
        lot=np.full(n, -1, dtype=np.int32), kolon=np.full(n, -1, dtype=np.int32),
        golge_fiyat=np.zeros(L), lot_devam_degeri=np.zeros(L),
        slot_golge=np.zeros(P), kredi_golge=np.zeros(P),
        lp_degeri=0.0, lp_teklif_degeri=0.0, yuvarlanmis_deger=0.0,
        kesirli_sutun=0, kolon_sayisi=0, durum="bos")


# --------------------------------------------------------------------------
# acgozlu tahsis (ranking-only ve kor iskonto)
# --------------------------------------------------------------------------
def acgozlu_tahsis(dunya: AdayDunyasi, cfg: Config, gor: OriginGorunumu,
                   lotlar: LotGorunumu, teklifler: pl.DataFrame,
                   kolonlar: Kolonlar, oncelik: np.ndarray,
                   pol: TahsisPolitikasi, n: int) -> TahsisSonucu:
    """Siralamaya dayali tahsis: satir basina en iyi kolon, sonra kuyruk.

    `stok_ayirma=False` ise lot stogu HIC dusulmez - M4'un teslim ettigi
    davranis budur ve kit SKU'da ayni lot birden fazla eczaneye soz verilir.
    Bu bir hata degil, olculecek TABAN CIZGISIDIR (SPEC M5 (a)).

    Gölge fiyat uretmez: acgozlu bir siralamanin duali yoktur. Kit kaynagin
    fiyatini gormek icin LP gerekir - (a) karsilastirmasinin ikinci okumasi
    tam olarak budur.
    """
    k = cfg.politika.kisit
    kol = np.zeros(n, dtype=np.int32)
    lot = np.full(n, -1, dtype=np.int32)
    kolon = np.full(n, -1, dtype=np.int32)
    if kolonlar.C == 0:
        return _bos_sonuc(pol, gor.t, n, lotlar.L, dunya.P)

    # Satir basina en iyi kolon: once FEFO lot (en kucuk lot indeksi = en erken
    # miad), sonra oncelik. Lot tercihi FEFO KALIR - acgozlu politikanin lot
    # secme yetkisi yok, bu yetkiyi LP'ye veren sey D5'in kendisi.
    sira = np.lexsort((-oncelik, kolonlar.lot, kolonlar.satir))
    en_iyi: dict[int, int] = {}
    for c in sira:
        i = int(kolonlar.satir[c])
        if i not in en_iyi:
            en_iyi[i] = int(c)
    aday = np.array(sorted(en_iyi.values()), dtype=np.int64)

    p_idx = teklifler["eczane_idx"].to_numpy()
    kalan_limit = _kalan_kredi_limiti(dunya, cfg, gor)
    kalan_stok = lotlar.adet.copy()
    sayac = np.zeros(dunya.P, dtype=np.int32)
    kullanilan = np.zeros(dunya.P)
    esik = cfg.politika.skor.asgari_teklif_marji

    for c in aday[np.argsort(-oncelik[aday], kind="stable")]:
        if oncelik[c] <= esik:
            continue
        i = int(kolonlar.satir[c])
        p = int(p_idx[i])
        if sayac[p] >= k.eczane_haftalik_teklif_tavani:
            continue
        if kullanilan[p] + kolonlar.tutar[c] > kalan_limit[p]:
            continue
        l = int(kolonlar.lot[c])
        if pol.stok_ayirma and kalan_stok[l] < kolonlar.cekilen[c]:
            continue
        kol[i] = kolonlar.kol[c]
        lot[i] = l
        kolon[i] = int(c)
        sayac[p] += 1
        kullanilan[p] += kolonlar.tutar[c]
        if pol.stok_ayirma:
            kalan_stok[l] -= kolonlar.cekilen[c]

    secilen = kolon[kolon >= 0]
    return TahsisSonucu(
        politika=pol.ad, t=gor.t, kol=kol, lot=lot, kolon=kolon,
        golge_fiyat=np.full(lotlar.L, np.nan), lot_devam_degeri=np.zeros(lotlar.L),
        slot_golge=np.full(dunya.P, np.nan), kredi_golge=np.full(dunya.P, np.nan),
        lp_degeri=float("nan"),
        lp_teklif_degeri=float(kolonlar.kazanc[secilen].sum()),
        yuvarlanmis_deger=float(kolonlar.kazanc[secilen].sum()),
        kesirli_sutun=0, kolon_sayisi=kolonlar.C, durum="acgozlu")


def _kalan_kredi_limiti(dunya: AdayDunyasi, cfg: Config,
                        gor: OriginGorunumu) -> np.ndarray:
    """[P] DBS limitinin kalani. policy/constraints.py ile AYNI formul."""
    k = cfg.politika.kisit
    dbs = dunya.eczaneler["dbs_limiti"].to_numpy().astype(float)
    risk = dunya.eczaneler["vade_riski_skoru"].to_numpy().astype(float)
    tavan = dbs * k.kredi_kullanim_tavani * (1.0 - risk * k.vade_riski_cezasi)
    return np.maximum(tavan - gor.acik_bakiye, 0.0)


# --------------------------------------------------------------------------
# LP
# --------------------------------------------------------------------------
def lp_tahsisi(dunya: AdayDunyasi, cfg: Config, gor: OriginGorunumu,
               lotlar: LotGorunumu, teklifler: pl.DataFrame, kolonlar: Kolonlar,
               pol: TahsisPolitikasi, n: int) -> TahsisSonucu:
    """Kit stok altinda tahsis LP'si + gölge fiyatlar (D5 / D9).

    Degiskenler: [C teklif sutunu | L lot artik sutunu].
    Lot kisiti ESITLIKTIR ve duali isaretsizdir - gölge fiyatin isaret
    degistirebilmesinin tek sebebi budur. `<=` yazilsaydi dual her zaman
    negatif olmayan olurdu ve D9 uygulanamazdi.
    """
    lp = cfg.tahsis.lp
    C, L, P = kolonlar.C, lotlar.L, dunya.P
    if C == 0:
        return _bos_sonuc(pol, gor.t, n, L, P)

    v = lotlar.salvage if pol.temizlik_degeri else lotlar.birim_deger
    hedef = np.concatenate([kolonlar.kazanc, v])

    p_idx = teklifler["eczane_idx"].to_numpy()
    kolon_ecz = p_idx[kolonlar.satir]

    # --- esitlik: lot dengesi ---
    A_eq = coo_matrix(
        (np.concatenate([kolonlar.cekilen, np.ones(L)]),
         (np.concatenate([kolonlar.lot, np.arange(L)]),
          np.concatenate([np.arange(C), C + np.arange(L)]))),
        shape=(L, C + L)).tocsc()
    b_eq = lotlar.adet

    # --- esitsizlikler: satir / frekans / kredi ---
    satir_veri = (np.ones(C), (kolonlar.satir, np.arange(C)))
    bloklar = [coo_matrix(satir_veri, shape=(n, C + L))]
    b_ub = [np.ones(n)]
    bloklar.append(coo_matrix((np.ones(C), (kolon_ecz, np.arange(C))),
                              shape=(P, C + L)))
    b_ub.append(np.full(P, float(cfg.politika.kisit.eczane_haftalik_teklif_tavani)))
    if lp.kredi_kisiti:
        bloklar.append(coo_matrix((kolonlar.tutar, (kolon_ecz, np.arange(C))),
                                  shape=(P, C + L)))
        b_ub.append(_kalan_kredi_limiti(dunya, cfg, gor))
    A_ub = vstack(bloklar).tocsc()
    b_ub = np.concatenate(b_ub)

    # r_l'nin UST SINIRI YOK ve bu bilincli. `S_l` ile sinirlansaydi, hic
    # tahsis almayan lotta r_l sinira dayanir, dualin bir kismini o sinir
    # kisiti emer ve gölge fiyat devam degerinin ALTINA duserdi - "gölge fiyat
    # >= devam degeri" ozdesligi kirilir ve tablodaki sayilar yorumlanamaz
    # hale gelirdi (scripts/verify_m5.py bu ozdesligi sinar). Ust sinir zaten
    # gereksiz: cekilis katsayilari negatif olmadigi icin r_l <= S_l otomatik.
    ust = np.concatenate([np.ones(C), np.full(L, np.inf)])
    sonuc = linprog(-hedef, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=np.column_stack([np.zeros(C + L), ust]),
                    method="highs",
                    options={"time_limit": lp.cozucu_zaman_siniri_sn})
    if not sonuc.success:
        raise RuntimeError(
            f"LP cozulemedi (origin {gor.t}, politika {pol.ad}): {sonuc.message}. "
            f"Kosu durduruluyor: cozulmemis bir LP'nin metrikleri okunmamali.")

    x = sonuc.x[:C]
    r = sonuc.x[C:]
    # linprog MINIMIZE ediyor ve -hedef verildi: dual isaretleri geri cevrilir.
    # gölge fiyat = d(deger)/d(kapasite) = -marginal
    golge = -np.asarray(sonuc.eqlin.marginals)
    ineq = -np.asarray(sonuc.ineqlin.marginals)
    slot_golge = ineq[n:n + P]
    kredi_golge = ineq[n + P:n + 2 * P] if lp.kredi_kisiti else np.full(P, np.nan)

    kesirli = int(((x > SIFIR_ESIGI) & (x < 1.0 - SIFIR_ESIGI)).sum())
    lp_deger = float(hedef @ sonuc.x)
    lp_teklif = float(kolonlar.kazanc @ x)

    kol = np.zeros(n, dtype=np.int32)
    lot = np.full(n, -1, dtype=np.int32)
    kolon = np.full(n, -1, dtype=np.int32)
    yuvarlanmis = float("nan")
    if lp.butunluk_yuvarlamasi:
        kol, lot, kolon, yuvarlanmis = _tam_sayiya_yuvarla(
            dunya, cfg, gor, lotlar, teklifler, kolonlar, x, v, n)

    return TahsisSonucu(
        politika=pol.ad, t=gor.t, kol=kol, lot=lot, kolon=kolon,
        golge_fiyat=golge, lot_devam_degeri=v, slot_golge=slot_golge,
        kredi_golge=kredi_golge, lp_degeri=lp_deger, lp_teklif_degeri=lp_teklif,
        yuvarlanmis_deger=yuvarlanmis, kesirli_sutun=kesirli,
        kolon_sayisi=C, durum=str(sonuc.message)[:60])


def _tam_sayiya_yuvarla(dunya: AdayDunyasi, cfg: Config, gor: OriginGorunumu,
                        lotlar: LotGorunumu, teklifler: pl.DataFrame,
                        kolonlar: Kolonlar, x: np.ndarray, v: np.ndarray, n: int):
    """LP gevsetmesinden teslim edilebilir (tam sayili) politikaya gecis.

    Sutunlar LP agirligina gore azalan sirada gezilir; ayni kisitlar (satir
    tekligi, lot stogu, frekans tavani, kredi limiti) tam sayili olarak
    uygulanir. Beraberlik bozma: LP agirligi, sonra REJIM DEGERI, sonra sutun
    indeksi - rassal degil, tekrar uretilebilir.

    ASGARI MARJ ESIGI REJIM DEGERINE UYGULANIR, ham kazanca degil:

        rejim_degeri = kazanc - cekilen x v_lot

    yani "teklifin marji" eksi "o adetleri lotta birakmanin degeri". Ham
    kazanca uygulansaydi miad rejimi CALISMAZDI: lotun devam degeri negatifken
    marji NEGATIF olan bir teklif rasyonel olabilir (SPEC 2.5, "normalde
    irrasyonel bir MF derinligi burada rasyoneldir") ve esik onu elerdi.
    Normal rejimde ayni ifade esigi SIKILASTIRIR - stogun firsat maliyeti.

    Yuvarlamanin bedeli (`lp_teklif_degeri - yuvarlanmis_deger`) butunluk
    acigidir ve raporlanir: LP'nin verdigi sayi bir UST SINIRDIR, teslim
    edilen politika o siniri yakalamak zorunda degil.
    """
    k = cfg.politika.kisit
    p_idx = teklifler["eczane_idx"].to_numpy()
    kalan_limit = _kalan_kredi_limiti(dunya, cfg, gor)
    kalan_stok = lotlar.adet.copy()
    sayac = np.zeros(dunya.P, dtype=np.int32)
    kullanilan = np.zeros(dunya.P)

    kol = np.zeros(n, dtype=np.int32)
    lot = np.full(n, -1, dtype=np.int32)
    kolon = np.full(n, -1, dtype=np.int32)
    kullanildi = np.zeros(n, dtype=bool)

    rejim_degeri = kolonlar.kazanc - kolonlar.cekilen * v[kolonlar.lot]
    sira = np.lexsort((np.arange(kolonlar.C), -rejim_degeri, -x))
    for c in sira:
        if x[c] <= SIFIR_ESIGI or rejim_degeri[c] <= cfg.politika.skor.asgari_teklif_marji:
            continue
        i = int(kolonlar.satir[c])
        if kullanildi[i]:
            continue
        p = int(p_idx[i])
        if sayac[p] >= k.eczane_haftalik_teklif_tavani:
            continue
        if cfg.tahsis.lp.kredi_kisiti and kullanilan[p] + kolonlar.tutar[c] > kalan_limit[p]:
            continue
        l = int(kolonlar.lot[c])
        if kalan_stok[l] < kolonlar.cekilen[c]:
            continue
        kullanildi[i] = True
        kol[i] = kolonlar.kol[c]
        lot[i] = l
        kolon[i] = int(c)
        sayac[p] += 1
        kullanilan[p] += kolonlar.tutar[c]
        kalan_stok[l] -= kolonlar.cekilen[c]

    secilen = kolon[kolon >= 0]
    return kol, lot, kolon, float(kolonlar.kazanc[secilen].sum())


# --------------------------------------------------------------------------
# giris noktasi
# --------------------------------------------------------------------------
def tahsis_et(dunya: AdayDunyasi, cfg: Config, gor: OriginGorunumu,
              lotlar: LotGorunumu, teklifler: pl.DataFrame,
              mat: TeklifMatrisleri, p_tahmin: np.ndarray, carpan: float,
              pol: TahsisPolitikasi) -> tuple[TahsisSonucu, Kolonlar]:
    """Bir origin, bir politika: kolonlari uret, tahsis et.

    `p_tahmin` politikanin TAHMIN ETTIGI kabul olasiliklaridir (M4'un
    X-ogrenicisi). Gercek olasilik burada YOK: planlama tahminle yapilir,
    stockout tam olarak bu fark yuzunden dogar ve olcumu eval/allocation.py
    gercek olasilikla ornekleyerek yapar.
    """
    kolonlar = kolon_uret(dunya, cfg, gor, lotlar, teklifler, mat, p_tahmin,
                          carpan, pol)
    n = teklifler.height
    if pol.lp:
        return lp_tahsisi(dunya, cfg, gor, lotlar, teklifler, kolonlar, pol, n), kolonlar
    return acgozlu_tahsis(dunya, cfg, gor, lotlar, teklifler, kolonlar,
                          kolonlar.kazanc, pol, n), kolonlar
