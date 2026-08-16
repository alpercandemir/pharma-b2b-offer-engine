"""Oracle: simulatorun gercegi. SADECE OLCUM.

SPEC 3: "sim/ Sentetik dunya (ground truth burada, model asla goremez)".
Bu dosya ground_truth/ okuyan yerdir ve ciktisi hicbir zaman feature
katmanina donmez. Egitim yolu features/ + models/ uzerinden gecer; oracle
yalnizca cikis kriterini olcen verify scriptlerinde ve experiments/run.py'de
DEGERLENDIRME tarafinda cagrilir.

IKI ORACLE, IKI SORU
====================
M2 (`Oracle`)             : "bu hucrenin stogu ne zaman sifirlandi"
M6 (`KarsiOlgusalOracle`) : "bu politikanin degeri TAM OLARAK nedir"

Ikincisi M6'nin varlik sebebidir. Sentetik dunyada `sim/response.py` her
satirin her kol altindaki GERCEK kabul olasiligini veriyor; dolayisiyla bir
politikanin degeri tahmin edilmez, HESAPLANIR. IPS/SNIPS/DR'nin sapmasi bu
sayiya gore olculur -- gercek hayatta imkansiz olan denetim tam olarak budur
ve POC'un ogretmek istedigi sey de bu denetimi bir kez gormektir.

M2'nin iki rakip sonu, karistirilmamali (reports/m1.md borc #6):
  TUKENME  : eldeki stok sifira dustu -> asil hedef.
  LISTEDEN DUSME : hucre cesitten cikarildi, kalan stok iade edildi.
     Bu da stogu sifirlar ama tuketimden degil, karardan gelir. Rakip risk
     olarak SANSUR sayilir; tukenme olayi olarak sayilirsa model "eczane bu
     urunu birakti" vakasini "stogu bitti" sanir ve metrik yalanci sekilde
     iyilesir.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

KOK = Path(__file__).resolve().parent.parent
VERI_DIZINI = KOK / "data"

# Cesit bayragi ile stok kaydi arasindaki bir haftalik kayit gecikmesi
# (sim/world.py: cesit haftanin basinda, stok haftanin sonunda yazilir).
# Domain sabiti degil, KAYIT sabiti: simulatorun yazma sirasindan geliyor.
CESIT_KAYIT_GECIKMESI = 1


@dataclass
class OracleEtiketleri:
    tukenme_k: np.ndarray      # [m] ilk stok-sifir gecikmesi (0 = ufukta olmadi)
    dusme_k: np.ndarray        # [m] listeden dusme gecikmesi (0 = olmadi)
    izlenen_k: np.ndarray      # [m] gozlenebilen periyot sayisi
    origin_stogu: np.ndarray   # [m] origin haftasindaki gercek stok
    gercek_hiz: np.ndarray     # [m] latent haftalik tuketim hizi
    origin_tuketimi: np.ndarray  # [m] son ufuk kadar haftanin gercek tuketim ort.

    @property
    def canli(self) -> np.ndarray:
        """Origin'de gercekten stogu olan satirlar. Tukenme metrikleri bunlarda."""
        return self.origin_stogu > 0

    @property
    def rakip_sansur(self) -> np.ndarray:
        """Listeden dusme yuzunden sifirlanmis satirlar: metrikten cikarilir.

        Kayit sirasi (sim/world.py): bir hucre w haftasinda cesitten cikarilinca
        kalan stok o hafta IADE edilir (stok w'de sifirlanir) ama cesit bayragi
        w'nin BASINDA yazildigi icin dusus ancak w+1'de gorunur. Bu yuzden
        dusme, tukenmeden bir hafta SONRA gorunse bile sebep odur.
        """
        gecikmeli = self.dusme_k > 0
        tukenmeden_once = (self.tukenme_k == 0) | (
            self.dusme_k <= self.tukenme_k + CESIT_KAYIT_GECIKMESI)
        return gecikmeli & tukenmeden_once

    @property
    def olay(self) -> np.ndarray:
        """Ufuk icinde GERCEKTEN tukendi: sifirlanma tuketimden geldi."""
        return (self.tukenme_k > 0) & ~self.rakip_sansur


class Oracle:
    """Gercek tukenme etiketleri. Matrisler ilk kullanimda bir kez kurulur.

    Ayni kosuda etiketle() birkac kez cagriliyor (egitim, test, teshis);
    2.4 milyon satirlik ground_truth'u her seferinde pivotlamak olcum
    suresini uc katina cikariyordu.
    """

    def __init__(self, kosu_adi: str, kok: Path | None = None) -> None:
        dizin = (kok or VERI_DIZINI) / kosu_adi / "ground_truth"
        self.hucre = pl.read_parquet(dizin / "hucre_haftalik.parquet")
        self._matrisler: tuple | None = None

    def _hazirla(self) -> tuple:
        if self._matrisler is not None:
            return self._matrisler
        h = self.hucre
        W = int(h["hafta"].max()) + 1
        e = h["eczane_id"].to_numpy()
        s = h["sku_id"].to_numpy()
        anahtar = np.char.add(np.char.add(e.astype(str), "|"), s.astype(str))
        benzersiz, satir = np.unique(anahtar, return_inverse=True)
        sira = {a: i for i, a in enumerate(benzersiz)}
        n = benzersiz.size

        hafta = h["hafta"].to_numpy().astype(int)
        stok = np.zeros((n, W), dtype=np.int64)
        cesit = np.zeros((n, W), dtype=bool)
        tuketim = np.zeros((n, W), dtype=np.int64)
        hiz = np.zeros(n)
        stok[satir, hafta] = h["gercek_eczane_stogu"].to_numpy()
        cesit[satir, hafta] = h["cesitte_var"].to_numpy()
        tuketim[satir, hafta] = h["gercek_tuketim"].to_numpy()
        hiz[satir] = h["latent_tuketim_hizi"].to_numpy()
        self._matrisler = (sira, stok, cesit, tuketim, hiz, W)
        return self._matrisler

    def etiketle(self, eczane_id: np.ndarray, sku_id: np.ndarray,
                 origin: np.ndarray, ufuk: int) -> OracleEtiketleri:
        """Panel anahtarlarini gercek tukenme etiketleriyle esler."""
        sira, stok, cesit, tuketim, hiz, W = self._hazirla()
        idx = np.array([sira.get(f"{e}|{s}", -1) for e, s in zip(eczane_id, sku_id)])
        if (idx < 0).any():
            raise KeyError("panel hucresi ground_truth'ta yok - kosular uyusmuyor")

        m = idx.size
        tukenme_k = np.zeros(m, dtype=np.int32)
        dusme_k = np.zeros(m, dtype=np.int32)
        izlenen_k = np.zeros(m, dtype=np.int32)
        origin_tuketimi = np.zeros(m)
        for t in np.unique(origin):
            sec = np.flatnonzero(origin == t)
            gozlenebilir = min(ufuk, W - 1 - int(t))
            izlenen_k[sec] = gozlenebilir
            if gozlenebilir <= 0:
                continue
            dilim = slice(int(t) + 1, int(t) + 1 + gozlenebilir)
            sifir = stok[idx[sec], dilim] == 0
            dusus = ~cesit[idx[sec], dilim]
            tukenme_k[sec] = np.where(sifir.any(1), sifir.argmax(1) + 1, 0)
            dusme_k[sec] = np.where(dusus.any(1), dusus.argmax(1) + 1, 0)
            gecmis = slice(max(0, int(t) - ufuk + 1), int(t) + 1)
            origin_tuketimi[sec] = tuketim[idx[sec], gecmis].mean(axis=1)

        return OracleEtiketleri(
            tukenme_k=tukenme_k, dusme_k=dusme_k, izlenen_k=izlenen_k,
            origin_stogu=stok[idx, origin.astype(int)],
            gercek_hiz=hiz[idx], origin_tuketimi=origin_tuketimi,
        )


# --------------------------------------------------------------------------
# M6: karsi-olgusal politika degeri
# --------------------------------------------------------------------------
@dataclass
class KarsiOlgusalOracle:
    """Her (satir, kol) icin GERCEK beklenen odul. OPE'nin denetleyicisi.

        v[i, a] = p_gercek(i, a) x brut_marj(i, a) x E[miktar carpani]

    `p_gercek` sim/response.py'nin uplift ground truth'u, `brut_marj`
    policy/scorer.py'nin marj aritmetigi. Ikisinin carpimi M4'un
    `politika_olcumu`yla AYNI buyukluktur; M6 onu SATIR BASINA normalize
    ederek raporlar cunku tahmincilerin cikardigi sayi da satir basinadir.

    IZINSIZ KOL. Politikanin secemeyecegi kol icin deger yine hesaplanir ama
    `izinli` maskesi ayri tutulur: bir tahminci izinsiz kola agirlik
    veriyorsa bu bir hatadir ve gorunmesi gerekir, sessizce maskelenmesi
    degil.
    """

    deger_matrisi: np.ndarray    # [n, A] gercek beklenen odul (TL)
    izinli: np.ndarray           # [n, A]
    eczane_idx: np.ndarray       # [n]

    @property
    def n(self) -> int:
        return self.deger_matrisi.shape[0]

    def satir_degeri(self, kol: np.ndarray) -> np.ndarray:
        """[n] verilen aksiyon vektorunun gercek beklenen odulu."""
        if self.n == 0:
            return np.zeros(0)
        return self.deger_matrisi[np.arange(self.n), kol]

    def deger(self, kol: np.ndarray) -> float:
        """V(pi), SATIR BASINA (TL/satir)."""
        return float(self.satir_degeri(kol).mean()) if self.n else float("nan")

    def karisim_degeri(self, pi: np.ndarray) -> float:
        """Stokastik bir politikanin gercek degeri: sum_a pi(a|x) v(x, a).

        Kayit politikasinin kendi degeri buradan cikar ve OPE'nin en temel
        ozdeslik testini verir: hedef = kayit politikasi alindiginda IPS bu
        sayiyi bulmali. Bulamiyorsa hata tahmincide degil, loglamadadir.
        """
        if self.n == 0:
            return float("nan")
        return float((pi * self.deger_matrisi).sum(axis=1).mean())

    def en_iyi_kol(self) -> np.ndarray:
        """[n] izinli kollar icinde gercek degeri en yuksek olani. TAVAN."""
        maskeli = np.where(self.izinli, self.deger_matrisi, -np.inf)
        return np.argmax(maskeli, axis=1)


def karsi_olgusal_oracle(tepki, mat, carpan: float,
                         eczane_idx: np.ndarray) -> KarsiOlgusalOracle:
    """`sim/response.py` tepkisi + `policy/scorer.py` marjindan oracle kurar."""
    return KarsiOlgusalOracle(
        deger_matrisi=tepki.olasilik * mat.marj * carpan,
        izinli=mat.izinli.copy(),
        eczane_idx=eczane_idx,
    )


def oracle_birlestir(parcalar: list[KarsiOlgusalOracle]) -> KarsiOlgusalOracle:
    var = [p for p in parcalar if p.n]
    if not var:
        return parcalar[0]
    return KarsiOlgusalOracle(
        deger_matrisi=np.vstack([p.deger_matrisi for p in var]),
        izinli=np.vstack([p.izinli for p in var]),
        eczane_idx=np.concatenate([p.eczane_idx for p in var]),
    )
