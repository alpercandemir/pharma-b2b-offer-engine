"""Lot boyutu, miad dagilimi ve FEFO tahsisi. SPEC 2.5.

Stok SKU seviyesinde tutulmaz. Miad partinin niteligidir:
    lot(lot_id, sku_id, miad_tarihi, adet, maliyet, giris_tarihi)

Tahsis FEFO calisir AMA kor degildir: eczacinin kabul edecegi minimum kalan
raf omru (latent miad_toleransi x kategori carpani) altindaki lot o eczaneye
verilemez, sirada en onde olsa bile. Bu M1'de miad_toleransi'ni yasayan bir
parametre yapar; aksi halde olu bir alan olurdu.

M5'in salvage value / temizlik rejimi BURADA YOK. Burada sadece lotlarin
dogmasi, FEFO tuketilmesi ve miadi gecenin imhasi var.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.config import Config
from sim.calendar import GUN_HAFTA


@dataclass
class Lot:
    lot_id: str
    sku_idx: int
    miad_gun: int          # koşu baslangicindan itibaren mutlak gun indeksi
    giris_haftasi: int       # lotun TAHSIS EDILEBILIR oldugu ilk hafta
    adet_giris: int
    adet_kalan: int
    birim_maliyet: float


@dataclass
class TahsisSatiri:
    hafta: int
    eczane_idx: int
    sku_idx: int
    lot_id: str
    adet: int
    kalan_raf_omru_gun: int


@dataclass
class ImhaSatiri:
    hafta: int
    sku_idx: int
    lot_id: str | None
    adet: int
    birim_maliyet: float
    imha_maliyeti: float
    kaynak: str            # "depo_miad" | "eczane_iadesi"


@dataclass
class LotDeposu:
    cfg: Config
    dsf: np.ndarray
    depo_marji: np.ndarray
    soguk_zincir: np.ndarray
    _lotlar: dict[int, list[Lot]] = field(default_factory=dict)
    _tarihce: list[Lot] = field(default_factory=list)
    _sayac: int = 0

    def parti_yarat(self, sku_idx: int, adet: int, hafta: int,
                    rng: np.random.Generator) -> Lot:
        lot_cfg = self.cfg.lot
        rf = lot_cfg.raf_omru
        toplam = float(np.exp(rng.normal(rf.toplam_gun_log_ort, rf.toplam_gun_log_sigma)))
        if self.soguk_zincir[sku_idx]:
            toplam *= rf.soguk_zincir_carpani
        toplam = float(np.clip(toplam, rf.min_gun, rf.max_gun))

        g = lot_cfg.giris
        if rng.random() < g.kisa_miatli_parti_olasiligi:
            kalan = float(rng.normal(g.kisa_miatli_kalan_gun_ort, g.kisa_miatli_kalan_gun_sigma))
            kalan = max(g.kisa_miatli_min_gun, min(kalan, toplam))
        else:
            tuketilmis = float(np.clip(
                rng.normal(g.tuketilmis_oran_ort, g.tuketilmis_oran_sigma),
                0.0, g.tuketilmis_oran_ust_sinir))
            kalan = toplam * (1.0 - tuketilmis)

        maliyet = (
            self.dsf[sku_idx]
            * (1.0 - self.depo_marji[sku_idx])
            * float(np.exp(rng.normal(0.0, lot_cfg.maliyet.parti_pazarligi_sigma)))
        )

        lot = Lot(
            lot_id=f"LOT{self._sayac:06d}",
            sku_idx=sku_idx,
            miad_gun=int(hafta * GUN_HAFTA + round(kalan)),
            giris_haftasi=hafta,
            adet_giris=int(adet),
            adet_kalan=int(adet),
            birim_maliyet=float(maliyet),
        )
        self._sayac += 1
        kuyruk = self._lotlar.setdefault(sku_idx, [])
        kuyruk.append(lot)
        kuyruk.sort(key=lambda l: l.miad_gun)   # FEFO sirasi
        self._tarihce.append(lot)
        return lot

    def eldeki_adet(self, sku_idx: int) -> int:
        return sum(l.adet_kalan for l in self._lotlar.get(sku_idx, ()))

    def tahsis_et(
        self, sku_idx: int, adet: int, hafta: int, eczane_idx: int,
        gerekli_kalan_gun: float, oncelikli_lot: str | None = None,
    ) -> tuple[int, list[TahsisSatiri], int]:
        """FEFO + miad toleransi.

        (karsilanan_adet, satirlar, tolerans_nedeniyle_atlanan_adet) doner.
        Ucuncu deger olcum icin: karsilayamamanin ne kadari gercek stoksuzluk,
        ne kadari "stok var ama eczaci bu miadi kabul etmez" ayrilabilsin diye.

        `oncelikli_lot` (M6): tahsis katmani (policy/allocate.py) bir lot
        SECMISSE once o denenir, yetmezse FEFO'ya donulur. Miad toleransi ve
        raf omru kontrolu o lotta da AYNEN uygulanir -- politikanin lot secmesi
        eczacinin kabul esigini gecersiz kilmaz (SPEC 2.5). Varsayilan None'da
        davranis M1'deki gibidir.
        """
        kuyruk = self._lotlar.get(sku_idx)
        if not kuyruk or adet <= 0:
            return 0, [], 0
        bugun = hafta * GUN_HAFTA
        tol_uygula = self.cfg.lot.tahsis.miad_toleransi_uygulanir
        satirlar: list[TahsisSatiri] = []
        kalan_talep = int(adet)
        atlanan = 0

        sirali = kuyruk if self.cfg.lot.tahsis.fefo_aktif else sorted(
            kuyruk, key=lambda l: -l.miad_gun)
        if oncelikli_lot is not None:
            sirali = sorted(sirali, key=lambda l: l.lot_id != oncelikli_lot)
        for lot in sirali:
            if kalan_talep <= 0:
                break
            if lot.adet_kalan <= 0:
                continue
            kalan_raf = lot.miad_gun - bugun
            if kalan_raf <= 0:
                continue
            if tol_uygula and kalan_raf < gerekli_kalan_gun:
                atlanan += min(lot.adet_kalan, kalan_talep)
                continue          # eczaci bu kadar kisa miatli lotu kabul etmez
            ver = min(lot.adet_kalan, kalan_talep)
            lot.adet_kalan -= ver
            kalan_talep -= ver
            satirlar.append(
                TahsisSatiri(hafta=hafta, eczane_idx=eczane_idx, sku_idx=sku_idx,
                             lot_id=lot.lot_id, adet=ver, kalan_raf_omru_gun=int(kalan_raf))
            )
        return int(adet) - kalan_talep, satirlar, atlanan

    def miadi_gecenleri_imha_et(self, hafta: int) -> list[ImhaSatiri]:
        bugun = hafta * GUN_HAFTA
        cikti: list[ImhaSatiri] = []
        for sku_idx, kuyruk in self._lotlar.items():
            birim_imha = self.dsf[sku_idx] * self.cfg.lot.maliyet.imha_birim_maliyeti_dsf_orani
            for lot in kuyruk:
                if lot.adet_kalan > 0 and lot.miad_gun <= bugun:
                    cikti.append(
                        ImhaSatiri(hafta=hafta, sku_idx=sku_idx, lot_id=lot.lot_id,
                                   adet=lot.adet_kalan, birim_maliyet=lot.birim_maliyet,
                                   imha_maliyeti=float(birim_imha * lot.adet_kalan),
                                   kaynak="depo_miad")
                    )
                    lot.adet_kalan = 0
            self._lotlar[sku_idx] = [l for l in kuyruk if l.adet_kalan > 0]
        return cikti

    def tum_lotlar(self) -> list[Lot]:
        """Kosu boyunca yaratilmis TUM lotlar (tukenmis/imha edilmis dahil)."""
        return list(self._tarihce)


# --------------------------------------------------------------------------
# Eczane tarafi: miad eczanede de yasar
# --------------------------------------------------------------------------
# Eczane stogu skaler tutulamaz. Kisa miatli mal eczaneye gidince orada
# yaslanir, satilmaz ve IADE olur (SPEC 2.5 "alici tarafi direnci"). Bu,
# miad_toleransi'ni cift yonlu yapar: siparis aninda ret + sonradan pismanlik.
#
# Depodaki lot izlemesinin aynisini eczane basina yapmak 200x300 hucrede
# Python dongusu demek olurdu. Bunun yerine hucre basina SABIT SAYIDA
# (config: eczane_lot_bolme_sayisi) miad kovasi tutulur ve tum islemler
# vektorize calisir. Kova dolu iken gelen sevkiyat, miadi en yakin kovaya
# adet-agirlikli ortalama ile birlestirilir.
#
# BASITLESTIRME: birlestirme hucre ici miad dagilimini kabalastirir. Kova
# sayisi arttikca yaklasim gerceklesir; varsayilan 4'te eczaneye gelen
# sevkiyatlarin buyuk cogunlugu tek lottan geldigi icin bozulma kucuk.

# Bos kovanin miadi. Siralamada en sona dussun diye buyuk secildi.
MIAD_SENTINEL = 10**9


class EczaneLotKovalari:
    """[K, P, S] miad kovalari. Tum islemler vektorize."""

    def __init__(self, kova_sayisi: int, P: int, S: int) -> None:
        self.K = kova_sayisi
        self.adet = np.zeros((kova_sayisi, P, S), dtype=np.int64)
        self.miad = np.full((kova_sayisi, P, S), MIAD_SENTINEL, dtype=np.int64)

    def toplam(self) -> np.ndarray:
        return self.adet.sum(axis=0)

    def _bosalti_temizle(self) -> None:
        self.miad = np.where(self.adet > 0, self.miad, MIAD_SENTINEL)

    def ekle(self, gelen_adet: np.ndarray, gelen_miad: np.ndarray) -> None:
        var = gelen_adet > 0
        if not var.any():
            return
        bos = self.adet == 0
        bos_var = bos.any(axis=0)
        hedef = np.where(bos_var, bos.argmax(axis=0),
                         np.abs(self.miad - gelen_miad[None, :, :]).argmin(axis=0))
        hedef = hedef[None, :, :]

        mevcut_adet = np.take_along_axis(self.adet, hedef, 0)[0]
        mevcut_miad = np.take_along_axis(self.miad, hedef, 0)[0]
        yeni_adet = mevcut_adet + gelen_adet
        # Adet-agirlikli ortalama miad; bos kovada gelenin miadi aynen gecer.
        yeni_miad = np.where(
            mevcut_adet > 0,
            np.rint((mevcut_adet * mevcut_miad + gelen_adet * gelen_miad)
                    / np.maximum(yeni_adet, 1)).astype(np.int64),
            gelen_miad,
        )
        np.put_along_axis(self.adet, hedef, np.where(var, yeni_adet, mevcut_adet)[None], 0)
        np.put_along_axis(self.miad, hedef, np.where(var, yeni_miad, mevcut_miad)[None], 0)
        self._bosalti_temizle()

    def tuket(self, adet: np.ndarray) -> None:
        """FEFO: en erken miatli kovadan basla."""
        sira = np.argsort(self.miad, axis=0, kind="stable")
        s_adet = np.take_along_axis(self.adet, sira, 0)
        onceki = np.cumsum(s_adet, axis=0) - s_adet
        alinan = np.clip(adet[None, :, :] - onceki, 0, s_adet)
        np.put_along_axis(self.adet, sira, s_adet - alinan, 0)
        self._bosalti_temizle()

    def satilamayacagi_bosalt(
        self, bugun: int, gunluk_hiz: np.ndarray, guvenlik_marji_gun: float,
        degerlendirme_esigi_gun: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Eczanenin miada kadar satamayacagi FAZLAYI bosaltir.

        SPEC 2.5'in tuketim hizi kuplajinin alici tarafindaki karsiligi:

            satabilecegi = gunluk_hiz x (kalan_gun - eczaci_guvenlik_marji)

        FEFO sirasinda k'inci kovaya kadar biriken stok (`gerek`), o kovanin
        miadina kadar satilabilecek miktari (`kapasite`) asiyorsa aradaki fark
        satilamaz. Iade miktari bu farkin kovalar boyunca maksimumudur ve en
        kisa miatli stoktan dusulur.

        Sabit bir gun esigi yerine bunun kullanilmasinin sebebi: 40 gun kalan
        100 adet, haftada 50 satan eczane icin sorun degil; haftada 1 satan
        eczane icin tamamen zayidir. Ayni esik ikisine ayni seyi soyleyemez.

        `degerlendirme_esigi_gun`: eczaci iki yil miatli stoga bakip karar
        vermez; sadece bu pencereye giren kovalar degerlendirilir.

        (iade_adet [P,S], adet-agirlikli ortalama kalan gun [P,S]) doner.
        """
        sira = np.argsort(self.miad, axis=0, kind="stable")
        s_adet = np.take_along_axis(self.adet, sira, 0)
        s_miad = np.take_along_axis(self.miad, sira, 0)
        kalan = np.clip(s_miad - bugun, 0, None)

        degerlendir = (s_adet > 0) & (kalan <= degerlendirme_esigi_gun)
        kapasite = gunluk_hiz[None, :, :] * np.clip(kalan - guvenlik_marji_gun, 0, None)
        gerek = np.cumsum(s_adet, axis=0)
        fazla = np.where(degerlendir, np.clip(gerek - kapasite, 0, None), 0.0)
        iade_toplam = np.rint(fazla.max(axis=0)).astype(np.int64)
        iade_toplam = np.minimum(iade_toplam, s_adet.sum(axis=0))

        # En kisa miatlidan dus.
        onceki = np.cumsum(s_adet, axis=0) - s_adet
        dusulen = np.clip(iade_toplam[None, :, :] - onceki, 0, s_adet)
        agirlikli = (dusulen * kalan).sum(axis=0)
        ortalama_kalan = np.where(iade_toplam > 0,
                                  agirlikli / np.maximum(iade_toplam, 1), 0.0)
        np.put_along_axis(self.adet, sira, s_adet - dusulen, 0)
        self._bosalti_temizle()
        return iade_toplam, ortalama_kalan

    def maskeyi_bosalt(self, maske: np.ndarray) -> np.ndarray:
        """Verilen [P,S] maskesindeki hucrelerin tum stogunu bosaltir."""
        adet = (self.adet * maske[None, :, :]).sum(axis=0)
        self.adet = np.where(maske[None, :, :], 0, self.adet)
        self._bosalti_temizle()
        return adet


def kalan_raf_omru_cek(cfg: Config, rng: np.random.Generator, boyut: tuple,
                       soguk_maske: np.ndarray) -> np.ndarray:
    """Vektorize raf omru cekilisi (gun). Depo disi kaynaklar icin.

    Baslangic eczane stogu ve rakip depodan gelen sevkiyat da miat tasir;
    bunlar bizim lot kuyrugumuzdan gelmedigi icin ayri cekilir. Dagilim
    depoya giren partilerle aynidir (config/lots.yaml).
    """
    rf, g = cfg.lot.raf_omru, cfg.lot.giris
    toplam = np.exp(rng.normal(rf.toplam_gun_log_ort, rf.toplam_gun_log_sigma, boyut))
    toplam = np.where(soguk_maske, toplam * rf.soguk_zincir_carpani, toplam)
    toplam = np.clip(toplam, rf.min_gun, rf.max_gun)

    tuketilmis = np.clip(rng.normal(g.tuketilmis_oran_ort, g.tuketilmis_oran_sigma, boyut),
                         0.0, g.tuketilmis_oran_ust_sinir)
    normal = toplam * (1.0 - tuketilmis)
    kisa = np.clip(rng.normal(g.kisa_miatli_kalan_gun_ort, g.kisa_miatli_kalan_gun_sigma, boyut),
                   g.kisa_miatli_min_gun, None)
    kisa_mi = rng.random(boyut) < g.kisa_miatli_parti_olasiligi
    return np.rint(np.where(kisa_mi, np.minimum(kisa, toplam), normal)).astype(np.int64)
