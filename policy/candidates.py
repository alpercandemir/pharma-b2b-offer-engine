"""Aday havuzu uretimi (retrieval). SPEC M3: "CF / market basket ile aday havuzu".

Bes uretici, hepsi AYNI gozlemlenebilir katmandan ve hepsi point-in-time:

  tekrar      : hucrenin kendi gecmisi (zaman azalimli siparis sayisi).
                Bugun sahada calisan sistem budur; taban cizgisi.
  cf          : item-item collaborative filtering. Eczane x SKU alim
                matrisinden SKU-SKU kosinus benzerligi, destek kirpmali.
                "Bu eczanenin aldiklarina benzeyen urun".
  sepet       : market basket. Ayni SEPETTE (birkac haftalik siparis blogu)
                birlikte gorulen SKU'lardan lift/confidence kurallari.
                CF'ten farki zaman olcegi: CF eczane duzeyinde bir yil boyunca
                birlikte ALINMAYI olcer, sepet AYNI SIPARISTE birlikte
                gorulmeyi. Ikisi farkli sinyal: birincisi cesit benzerligi,
                ikincisi siparis tamamlayiciligi.
  soguk_start : eczane OZNITELIKLERINDEN komsuluk (SPEC M3 "cold start icin
                eczane attribute'lari"). Hic siparis gecmisi olmayan hucrede
                CF ve sepet sessizdir; benzer eczanelerin (hastane yakinligi,
                sosyoekonomik, turizm, olcek, il) aldigi urun tek sinyaldir.
  populerlik  : global hacim sirasi. Kisisellestirme yok; recall tabani.

Hibrit bunlari ECZANE ICI SIRAYA cevirip agirlikli toplar. Ham skorlari
toplamak yanlis olurdu: kosinus toplami, lift ve satin alma orani ayni
birimde degil.

KISITLARDAN HABERSIZ. Bu bilincli (D6): kirmizi recete de, kredi limiti asan
teklif de havuza girer ve policy/constraints.py tarafindan cikarilir. Aday
uretimi kisiti onceden bilseydi vetonun bedeli olculemezdi.

POINT-IN-TIME: butun matrisler `gorunum_kur(dunya, cfg, t)` icinde
hafta <= t verisinden kurulur. Gelecek silinince ayni origin'in skorlari
degismemeli - tests/test_candidates.py bunu kesip yeniden hesaplayarak sinar.
Bu dosya ground_truth okumaz; hedef kumesi ve oracle tavani eval/aday.py'de.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from core.config import Config
from features.okuma import GozlemlenebilirKaynak
from sim.calendar import GUN_HAFTA

# Yariomurden azalim katsayisi: agirlik = YARIM ** (gecen_hafta / yariomur).
# Matematik sabiti, knob degil; kadran `aday.yariomur_hafta`.
YARIM = 0.5
# Sifira bolme korumasi (birimsiz oranlarda). Sayisal sabit.
EPSILON = 1e-12


# --------------------------------------------------------------------------
# statik dunya (origin'den bagimsiz)
# --------------------------------------------------------------------------
@dataclass
class AdayDunyasi:
    """Gozlemlenebilir katmanin aday uretimi icin gereken parcasi.

    Tablolar ham tutulur; origin'e gore filtreleme `gorunum_kur`da yapilir.
    Boylece "hangi satirlar hangi origin'de kullanildi" tek yerde gorulur.
    """

    eczaneler: pl.DataFrame          # eczane_id'ye gore sirali
    urunler: pl.DataFrame            # sku_id'ye gore sirali
    P: int
    S: int
    W: int
    # siparisler
    sip_p: np.ndarray
    sip_s: np.ndarray
    sip_w: np.ndarray
    sip_adet: np.ndarray
    # sevkiyat satirlari (lot kalanlari + acik bakiye)
    sevk_p: np.ndarray
    sevk_s: np.ndarray
    sevk_w: np.ndarray
    sevk_lot: np.ndarray
    sevk_adet: np.ndarray
    # depo lotlari
    lot_id: np.ndarray
    lot_s: np.ndarray
    lot_giris: np.ndarray
    lot_miad_gun: np.ndarray
    lot_adet: np.ndarray
    lot_birim_maliyet: np.ndarray
    # imhalar (lot bazli olanlar)
    imha_lot: np.ndarray
    imha_w: np.ndarray
    imha_adet: np.ndarray
    # urun ozellikleri
    dsf: np.ndarray
    soguk_zincir: np.ndarray
    # eczane olcegi (teklif adedi tahmininde kullanilir)
    eczane_olcegi: np.ndarray


def _sira_haritasi(seri: pl.Series) -> dict:
    return {v: i for i, v in enumerate(seri.to_list())}


def dunya_yukle(kaynak: GozlemlenebilirKaynak, cfg: Config) -> AdayDunyasi:
    eczane = kaynak.tablo("eczaneler").sort("eczane_id")
    urun = kaynak.tablo("urunler").sort("sku_id")
    siparis = kaynak.tablo("siparisler")
    sevkiyat = kaynak.tablo("sevkiyat_satirlari")
    lotlar = kaynak.tablo("stok_lotlari")
    imha = kaynak.tablo("imhalar")
    W = kaynak.tablo("takvim").height

    ecz_sira = _sira_haritasi(eczane["eczane_id"])
    sku_sira = _sira_haritasi(urun["sku_id"])

    def _p(df: pl.DataFrame) -> np.ndarray:
        return np.array([ecz_sira[e] for e in df["eczane_id"]], dtype=np.int32)

    def _s(df: pl.DataFrame) -> np.ndarray:
        return np.array([sku_sira[s] for s in df["sku_id"]], dtype=np.int32)

    # Yalnizca lot bazli imhalar: eczane iadesinden gelen satirlarda lot_id
    # yoktur (sim/world.py) ve depo lot kalanini etkilemez.
    imha_lotlu = imha.filter(pl.col("lot_id").is_not_null())

    olcek = eczane["aylik_recete_adedi"].to_numpy().astype(float)
    olcek = olcek / max(float(np.median(olcek)), EPSILON)

    return AdayDunyasi(
        eczaneler=eczane, urunler=urun,
        P=eczane.height, S=urun.height, W=W,
        sip_p=_p(siparis), sip_s=_s(siparis),
        sip_w=siparis["hafta"].to_numpy().astype(np.int32),
        sip_adet=siparis["talep_adet"].to_numpy().astype(float),
        sevk_p=_p(sevkiyat), sevk_s=_s(sevkiyat),
        sevk_w=sevkiyat["hafta"].to_numpy().astype(np.int32),
        sevk_lot=sevkiyat["lot_id"].to_numpy(),
        sevk_adet=sevkiyat["adet"].to_numpy().astype(float),
        lot_id=lotlar["lot_id"].to_numpy(),
        lot_s=_s(lotlar),
        lot_giris=lotlar["giris_haftasi"].to_numpy().astype(np.int32),
        lot_miad_gun=lotlar["miad_gun_indeksi"].to_numpy().astype(float),
        lot_adet=lotlar["adet_giris"].to_numpy().astype(float),
        lot_birim_maliyet=lotlar["birim_maliyet"].to_numpy().astype(float),
        imha_lot=imha_lotlu["lot_id"].to_numpy(),
        imha_w=imha_lotlu["hafta"].to_numpy().astype(np.int32),
        imha_adet=imha_lotlu["adet"].to_numpy().astype(float),
        dsf=urun["dsf"].to_numpy().astype(float),
        soguk_zincir=urun["soguk_zincir"].to_numpy(),
        eczane_olcegi=olcek,
    )


# --------------------------------------------------------------------------
# origin gorunumu (point-in-time)
# --------------------------------------------------------------------------
@dataclass
class Lot:
    lot_id: str
    kalan_adet: float
    kalan_gun: float


@dataclass
class OriginGorunumu:
    """t haftasinin sonunda, YALNIZCA hafta <= t verisinden kurulmus durum."""

    t: int
    agirlikli_adet: np.ndarray       # [P, S] zaman azalimli siparis adedi
    agirlikli_sayi: np.ndarray       # [P, S] zaman azalimli siparis haftasi sayisi
    ikili: np.ndarray                # [P, S] pencerede en az bir siparis
    akis_hizi: np.ndarray            # [P, S] adet / hafta (azalimsiz)
    hiz_tahmini: np.ndarray          # [P, S] teklif adedi icin kullanilan hiz
    son_sepet: np.ndarray            # [P, S] son sepet penceresindeki SKU'lar
    sepet_matrisi: np.ndarray        # [B, S] kural madenciligi icin sepetler
    eczane_siparis_satiri: np.ndarray  # [P] penceredeki siparis satiri sayisi
    depo_stok: np.ndarray            # [S] eldeki adet
    miad_baskisi: np.ndarray         # [S] kisa miatli adet / eldeki adet
    lotlar: dict[int, list[Lot]]     # sku_idx -> FEFO sirali lot listesi
    acik_bakiye: np.ndarray          # [P] tahsil edilmemis sevkiyat tutari


def _pencere_maskesi(hafta: np.ndarray, t: int, pencere: int) -> np.ndarray:
    return (hafta <= t) & (hafta > t - pencere)


def gorunum_kur(dunya: AdayDunyasi, cfg: Config, t: int) -> OriginGorunumu:
    a = cfg.politika.aday
    P, S = dunya.P, dunya.S
    sec = _pencere_maskesi(dunya.sip_w, t, a.pencere_hafta)
    p, s = dunya.sip_p[sec], dunya.sip_s[sec]
    w, adet = dunya.sip_w[sec], dunya.sip_adet[sec]
    agirlik = YARIM ** ((t - w) / a.yariomur_hafta)

    agirlikli_adet = np.zeros((P, S))
    agirlikli_sayi = np.zeros((P, S))
    ham_adet = np.zeros((P, S))
    np.add.at(agirlikli_adet, (p, s), adet * agirlik)
    np.add.at(agirlikli_sayi, (p, s), agirlik)
    np.add.at(ham_adet, (p, s), adet)
    ikili = ham_adet > 0

    etkin_pencere = float(min(a.pencere_hafta, t + 1))
    akis_hizi = ham_adet / etkin_pencere

    # Yeni hucrede akis hizi sifirdir; teklif adedi icin bir beklenti gerekir.
    # SKU'nun kendisini alan eczanelerdeki ortalama hizi, bu eczanenin
    # olcegiyle olceklenir. Ikisi de gozlemlenebilir; uydurma sabit yok.
    alan_sayisi = ikili.sum(axis=0)
    sku_ort_hiz = akis_hizi.sum(axis=0) / np.maximum(alan_sayisi, 1)
    beklenen_hiz = sku_ort_hiz[None, :] * dunya.eczane_olcegi[:, None]
    hiz_tahmini = np.where(akis_hizi > 0, akis_hizi, beklenen_hiz)

    # --- sepetler: (eczane, blok) ikilisi bir sepettir ---
    blok = (t - w) // a.sepet.pencere_hafta
    blok_sayisi = int(np.ceil(a.pencere_hafta / a.sepet.pencere_hafta))
    sepet_matrisi = np.zeros((P * blok_sayisi, S))
    gecerli = blok < blok_sayisi
    sepet_matrisi[(p[gecerli] * blok_sayisi + blok[gecerli]), s[gecerli]] = 1.0
    son_sepet = np.zeros((P, S), dtype=bool)
    son = blok == 0
    son_sepet[p[son], s[son]] = True

    eczane_siparis_satiri = np.bincount(p, minlength=P).astype(float)

    return OriginGorunumu(
        t=t, agirlikli_adet=agirlikli_adet, agirlikli_sayi=agirlikli_sayi,
        ikili=ikili, akis_hizi=akis_hizi, hiz_tahmini=hiz_tahmini,
        son_sepet=son_sepet, sepet_matrisi=sepet_matrisi,
        eczane_siparis_satiri=eczane_siparis_satiri,
        **_stok_gorunumu(dunya, cfg, t),
        acik_bakiye=_acik_bakiye(dunya, cfg, t),
    )


def _stok_gorunumu(dunya: AdayDunyasi, cfg: Config, t: int) -> dict:
    """Depo lotlarinin t haftasindaki kalani, miad baskisi ve FEFO sirasi.

    Lot kalani gozlemlenebilir tablolardan yeniden kurulur:
        kalan = giris - (t'ye kadar sevk edilen) - (t'ye kadar imha edilen)

    `depo_stok_haftalik` tablosuyla ozdeslik (tests/test_candidates.py):
        kayitli_stok(t) - yeniden_kurulan(t) = giris_haftasi == t+1 olan lotlar
    Aradaki fark tesadufi degil: depo stok kaydi hafta SONUNDA, ikmalden sonra
    aliniyor ve gelecek haftanin partisini iceriyor. Politika t haftasinda o
    partiyi tahsis EDEMEZ (sim/lots.py giris_haftasi kurali), bu yuzden burada
    bilerek TAHSIS EDILEBILIR stok kuruluyor - kayitli stoktan hep kucuk ya da
    esit. Kisit katmani hicbir zaman olmayan mali soz vermez.
    """
    gun = t * GUN_HAFTA
    lot_var = dunya.lot_giris <= t
    lot_sira = {l: i for i, l in enumerate(dunya.lot_id)}

    kullanilan = np.zeros(dunya.lot_id.size)
    for lot_dizi, hafta_dizi, adet_dizi in (
        (dunya.sevk_lot, dunya.sevk_w, dunya.sevk_adet),
        (dunya.imha_lot, dunya.imha_w, dunya.imha_adet),
    ):
        sec = hafta_dizi <= t
        idx = np.array([lot_sira[l] for l in lot_dizi[sec]], dtype=int)
        if idx.size:
            np.add.at(kullanilan, idx, adet_dizi[sec])

    kalan = np.maximum(dunya.lot_adet - kullanilan, 0.0) * lot_var
    kalan_gun = dunya.lot_miad_gun - gun
    canli = (kalan > 0) & (kalan_gun > 0)

    depo_stok = np.zeros(dunya.S)
    np.add.at(depo_stok, dunya.lot_s[canli], kalan[canli])
    baskili = canli & (kalan_gun < cfg.politika.aday.miad_baskisi_esik_gun)
    baski_adet = np.zeros(dunya.S)
    np.add.at(baski_adet, dunya.lot_s[baskili], kalan[baskili])
    miad_baskisi = baski_adet / np.maximum(depo_stok, EPSILON)

    lotlar: dict[int, list[Lot]] = {}
    for i in np.flatnonzero(canli):
        lotlar.setdefault(int(dunya.lot_s[i]), []).append(
            Lot(lot_id=str(dunya.lot_id[i]), kalan_adet=float(kalan[i]),
                kalan_gun=float(kalan_gun[i])))
    for liste in lotlar.values():
        liste.sort(key=lambda l: (l.kalan_gun, l.lot_id))   # FEFO
    return {"depo_stok": depo_stok, "miad_baskisi": miad_baskisi, "lotlar": lotlar}


def _acik_bakiye(dunya: AdayDunyasi, cfg: Config, t: int) -> np.ndarray:
    """Tahsil edilmemis sayilan sevkiyat tutari (TL), eczane basina.

    Gercek cari hesap tablosu bu POC'ta yok; vade penceresi icindeki
    sevkiyatin DSF tutari acik bakiye vekilidir (TUNING.md M3-B4).
    """
    sec = _pencere_maskesi(dunya.sevk_w, t, cfg.politika.kisit.acik_bakiye_vade_hafta)
    bakiye = np.zeros(dunya.P)
    np.add.at(bakiye, dunya.sevk_p[sec],
              dunya.sevk_adet[sec] * dunya.dsf[dunya.sevk_s[sec]])
    return bakiye


# --------------------------------------------------------------------------
# ureticiler: hepsi [P, S] skor matrisi dondurur
# --------------------------------------------------------------------------
def uretici_tekrar(dunya: AdayDunyasi, cfg: Config, gor: OriginGorunumu) -> np.ndarray:
    """Hucrenin kendi gecmisi. Yeni hucrede sifir - tanimi geregi kor."""
    return gor.agirlikli_sayi


def uretici_populerlik(dunya: AdayDunyasi, cfg: Config,
                       gor: OriginGorunumu) -> np.ndarray:
    """Global hacim. Kisisellestirme yok; tum eczanelerde ayni sira."""
    hacim = gor.agirlikli_adet.sum(axis=0)
    return np.broadcast_to(hacim[None, :], (dunya.P, dunya.S)).copy()


def _sku_benzerligi(cfg: Config, gor: OriginGorunumu) -> np.ndarray:
    """[S, S] destek kirpmali kosinus benzerligi, komsu sayisiyla budanmis."""
    b = cfg.politika.aday.benzerlik
    A = np.log1p(gor.agirlikli_adet)
    norm = np.sqrt((A * A).sum(axis=0))
    A_n = A / np.maximum(norm, EPSILON)[None, :]
    sim = A_n.T @ A_n

    ortak = gor.ikili.astype(float).T @ gor.ikili.astype(float)
    sim = sim * (ortak / (ortak + b.kirpma))
    sim[ortak < b.min_ortak_eczane] = 0.0
    np.fill_diagonal(sim, 0.0)

    # Her SKU icin yalnizca en yakin komsular kalir.
    if b.komsu_sku_sayisi < sim.shape[1]:
        esik = np.partition(sim, -b.komsu_sku_sayisi, axis=1)[:, -b.komsu_sku_sayisi]
        sim = np.where(sim >= esik[:, None], sim, 0.0)
    return sim


def uretici_cf(dunya: AdayDunyasi, cfg: Config, gor: OriginGorunumu) -> np.ndarray:
    """Item-item CF: eczanenin aldiklarina benzeyen SKU'lar."""
    A = np.log1p(gor.agirlikli_adet)
    norm = np.sqrt((A * A).sum(axis=1))
    A_n = A / np.maximum(norm, EPSILON)[:, None]
    return A_n @ _sku_benzerligi(cfg, gor)


def _sepet_kurallari(cfg: Config, gor: OriginGorunumu) -> np.ndarray:
    """[S, S] kural gucu: kaynak -> hedef, confidence * ln(lift).

    Kural ancak min_destek ve min_lift'i birlikte geciyorsa kalir. Yalniz
    confidence populer urunu her seye baglar; yalniz lift seyrek ciftlerde
    patlar. Ikisi birlikte.
    """
    sp = cfg.politika.aday.sepet
    M = gor.sepet_matrisi
    n_sepet = float((M.sum(axis=1) > 0).sum())
    if n_sepet == 0:
        return np.zeros((M.shape[1], M.shape[1]))
    birlikte = M.T @ M
    destek = np.diag(birlikte).copy()
    np.fill_diagonal(birlikte, 0.0)

    payda = np.maximum(destek[:, None] * destek[None, :], EPSILON)
    lift = birlikte * n_sepet / payda
    guven = birlikte / np.maximum(destek[:, None], EPSILON)
    gecerli = (birlikte >= sp.min_destek) & (lift >= sp.min_lift)
    return np.where(gecerli, guven * np.log(np.maximum(lift, 1.0)), 0.0)


def uretici_sepet(dunya: AdayDunyasi, cfg: Config, gor: OriginGorunumu) -> np.ndarray:
    """Market basket: eczanenin SON sepetindeki urunlerin tetikledigi kurallar."""
    return gor.son_sepet.astype(float) @ _sepet_kurallari(cfg, gor)


def _eczane_benzerligi(dunya: AdayDunyasi, cfg: Config) -> np.ndarray:
    """[P, P] oznitelik tabanli eczane benzerligi. Siparis gecmisi KULLANMAZ.

    Cold start'in tanimi bu: hucre hakkinda hicbir islem verisi yokken elde
    kalan tek sey eczanenin kendisidir (SPEC 2.2: konum ve hastane yakinligi
    recete miksini belirleyen en guclu ozellik).
    """
    ag = cfg.politika.aday.soguk_start.oznitelik_agirliklari
    e = dunya.eczaneler
    surekli = {
        "hastane_yakinligi": np.log1p(e["hastane_yakinligi_km"].to_numpy().astype(float)),
        "sosyoekonomik": e["semt_sosyoekonomik_index"].to_numpy().astype(float),
        "turizm": e["turizm_bolgesi"].to_numpy().astype(float),
        "olcek": np.log(np.maximum(dunya.eczane_olcegi, EPSILON)),
        "sgk_recete_orani": e["sgk_recete_orani"].to_numpy().astype(float),
        "nobet": e["nobet_rotasyon_gun"].to_numpy().astype(float),
    }
    kare = np.zeros((dunya.P, dunya.P))
    for ad, v in surekli.items():
        z = (v - v.mean()) / max(float(v.std()), EPSILON)
        kare += ag[ad] * (z[:, None] - z[None, :]) ** 2
    il = e["il"].to_numpy()
    kare += ag["il"] * (il[:, None] != il[None, :]).astype(float)
    return 1.0 / (1.0 + np.sqrt(kare))


def uretici_soguk_start(dunya: AdayDunyasi, cfg: Config,
                        gor: OriginGorunumu) -> np.ndarray:
    """Benzer eczanelerin aldigi urun. Kendi gecmisini KULLANMAZ."""
    sc = cfg.politika.aday.soguk_start
    benzerlik = _eczane_benzerligi(dunya, cfg)
    np.fill_diagonal(benzerlik, -np.inf)          # kendi kendinin komsusu degil
    k = min(sc.komsu_eczane_sayisi, dunya.P - 1)
    komsu = np.argpartition(-benzerlik, k - 1, axis=1)[:, :k]
    agirlik = np.take_along_axis(benzerlik, komsu, axis=1)

    alim = gor.ikili.astype(float)
    skor = np.einsum("pk,pks->ps", agirlik, alim[komsu])
    return skor / np.maximum(agirlik.sum(axis=1), EPSILON)[:, None]


URETICILER = {
    "tekrar": uretici_tekrar,
    "cf": uretici_cf,
    "sepet": uretici_sepet,
    "soguk_start": uretici_soguk_start,
    "populerlik": uretici_populerlik,
}


# --------------------------------------------------------------------------
# hibrit + havuz
# --------------------------------------------------------------------------
def sira_normalize(skor: np.ndarray) -> np.ndarray:
    """[P, S] skoru eczane ICINDE (0, 1] yuzdelik siraya cevirir; sifir sifir kalir.

    Ureticilerin ham skorlari ayni birimde degil (kosinus toplami vs lift vs
    satin alma orani). Karistirmadan once sira uzayina tasinmalari sart.
    """
    n = skor.shape[1]
    sira = np.argsort(np.argsort(-skor, axis=1, kind="stable"), axis=1)
    return np.where(skor > 0, 1.0 - sira / n, 0.0)


def uretici_skorlari(dunya: AdayDunyasi, cfg: Config,
                     gor: OriginGorunumu) -> dict[str, np.ndarray]:
    return {ad: f(dunya, cfg, gor) for ad, f in URETICILER.items()}


def hibrit_skor(skorlar: dict[str, np.ndarray], cfg: Config,
                gor: OriginGorunumu) -> np.ndarray:
    """Sira birlestirme + miad baskisi carpani."""
    a = cfg.politika.aday
    toplam = np.zeros_like(next(iter(skorlar.values())))
    for ad, agirlik in a.karisim_agirliklari.items():
        toplam += agirlik * sira_normalize(skorlar[ad])
    toplam /= sum(a.karisim_agirliklari.values())
    # Miad baskisi SIRALAMAYI oynatir, veto yetkisi YOKTUR (D6 / SPEC 2.5).
    return toplam * (1.0 + a.miad_baskisi_agirligi * gor.miad_baskisi[None, :])


def havuz_cikar(skor: np.ndarray, k: int, dunya: AdayDunyasi,
                gor: OriginGorunumu) -> pl.DataFrame:
    """Eczane basina en yuksek k adayi tablolastirir.

    Beraberlik bozma: (skor, sku_idx) sirasi. Rassal degil, tekrar uretilebilir.
    """
    P, S = skor.shape
    k = min(k, S)
    sira = np.argsort(-skor, axis=1, kind="stable")[:, :k]
    p = np.repeat(np.arange(P), k)
    s = sira.reshape(-1)
    skorlar = skor[p, s]
    # Skoru sifir olan aday yoktur: hicbir uretici sinyal vermemistir.
    tut = skorlar > 0
    p, s, skorlar = p[tut], s[tut], skorlar[tut]
    sirano = np.tile(np.arange(k), P)[tut]

    return pl.DataFrame({
        "origin": np.full(p.size, gor.t, dtype=np.int32),
        "eczane_idx": p.astype(np.int32),
        "sku_idx": s.astype(np.int32),
        "eczane_id": dunya.eczaneler["eczane_id"].to_numpy()[p],
        "sku_id": dunya.urunler["sku_id"].to_numpy()[s],
        "sira": sirano.astype(np.int32),
        "skor": skorlar,
        "yeni_hucre": ~gor.ikili[p, s],
        "miad_baskisi": gor.miad_baskisi[s],
        "hiz_tahmini": gor.hiz_tahmini[p, s],
    })


def teklif_adedi(hiz: np.ndarray, cfg: Config) -> np.ndarray:
    """Talebe gore teklif adedi. Kisit katmani bunu YUKSELTEBILIR ya da veto eder.

    adet = tavan(hiz * telafi * kapsama). Koli katina yuvarlama YOK: SPEC 2.1
    koli kati kuralini MF oranlari icin koyuyor (M4), normal siparis icin
    degil (sim/sim.yaml ile ayni kabul).

    `hiz_telafi_katsayisi` M2'nin olctugu seyreltmeyi telafi eder: bize gelen
    akis gercek tuketimin share_of_wallet kadarlik parcasidir, telafisiz adet
    sistematik olarak kucuk cikar (reports/m2.md 12.4).
    """
    a = cfg.politika.aday
    return np.maximum(1.0, np.ceil(hiz * a.hiz_telafi_katsayisi * a.teklif_kapsama_hafta))


def aday_havuzu(dunya: AdayDunyasi, cfg: Config, gor: OriginGorunumu,
                k: int | None = None) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Teslim edilen havuz + uretici skor matrisleri (olcum icin)."""
    skorlar = uretici_skorlari(dunya, cfg, gor)
    hibrit = hibrit_skor(skorlar, cfg, gor)
    skorlar["hibrit"] = hibrit
    havuz = havuz_cikar(hibrit, k or cfg.politika.aday.havuz_boyutu_k, dunya, gor)
    havuz = havuz.with_columns(
        pl.Series("teklif_adedi", teklif_adedi(havuz["hiz_tahmini"].to_numpy(), cfg))
    )
    return havuz, skorlar


def origin_haftalari(cfg: Config, W: int) -> list[int]:
    """Olcum origin'leri: kosunun sonundan geriye, ufuk kadar araliklarla.

    Aralik = ufuk: degerlendirme pencereleri ORTUSMESIN. Ortusen pencereler
    ayni siparisleri birden fazla origin'de hedef sayar ve recall'un guven
    araligini oldugundan dar gosterir (M2'de ayni tuzak olculdu).
    """
    d = cfg.politika.aday.degerlendirme
    son = W - 1 - d.ufuk_hafta
    originler = [son - i * d.ufuk_hafta for i in range(d.origin_sayisi)]
    en_erken = cfg.feature.panel.ilk_origin_hafta
    gecerli = sorted(t for t in originler if t >= en_erken)
    if not gecerli:
        raise ValueError(
            f"M3 origin'i kalmadi: W={W}, ufuk={d.ufuk_hafta}, "
            f"origin_sayisi={d.origin_sayisi}, isinma={en_erken}")
    return gecerli
