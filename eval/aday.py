"""M3 olcumu: aday havuzu recall'u ve vetonun bedeli.

Iki hedef tanimi var ve farklari M2'nin etiket korlugu bulgusunun (m2.md 3.3)
aday tarafindaki karsiligidir:

  GOZLEMLENEBILIR hedef : origin'den sonraki ufukta BIZE gelen siparisler.
      Sahada olculebilecek tek sey budur. Ama eczane ayni hafta ayni urunu
      rakipten de almis olabilir; o satirlar hedefte GORUNMEZ.

  ORACLE hedefi         : ayni ufukta eczanenin GERCEKTEN tukettigi urunler
      (ground_truth/hucre_haftalik). Tedarikcisi kim olursa olsun gercek
      ihtiyac budur. Aday uretiminin tavani buna gore olculur.

Gozlemlenebilir recall, oracle recall'undan sistematik olarak YUKSEK cikar:
paydasi kucuk (yalnizca bize gelen). "Recall %60" cumlesi hangi hedefe gore
sorulmadan okunamaz - M2'deki "MAE 18 gun" tuzaginin aynisi.

Bu dosya ground_truth okuyan iki yerden biridir (digeri eval/oracle.py) ve
ciktisi asla policy/ katmanina donmez.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from policy.candidates import AdayDunyasi, OriginGorunumu

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# "Ust dilim" = en yuksek skorlu %10. eval/metrics.py'deki `ust_dilim_kazanci`
# ile ayni dilim; iki metrik ayni yeri isaret etsin diye. Olcum tanimi, knob degil.
UST_DILIM = 0.10


@dataclass
class Hedef:
    """[P, S] hedef maskesi + hedefin yeni hucre olan alt kumesi."""

    ad: str
    matris: np.ndarray
    yeni: np.ndarray

    @property
    def sayi(self) -> int:
        return int(self.matris.sum())

    @property
    def yeni_sayi(self) -> int:
        return int(self.yeni.sum())


def gozlemlenebilir_hedef(dunya: AdayDunyasi, gor: OriginGorunumu,
                          ufuk: int) -> Hedef:
    """Origin sonrasi ufukta BIZE gelen siparisler."""
    sec = (dunya.sip_w > gor.t) & (dunya.sip_w <= gor.t + ufuk)
    matris = np.zeros((dunya.P, dunya.S), dtype=bool)
    matris[dunya.sip_p[sec], dunya.sip_s[sec]] = True
    return Hedef("gozlemlenebilir", matris, matris & ~gor.ikili)


def oracle_hedef(kosu_adi: str, dunya: AdayDunyasi, gor: OriginGorunumu,
                 ufuk: int, kok: Path | None = None) -> Hedef:
    """Origin sonrasi ufukta eczanenin GERCEKTEN tukettigi urunler.

    Tedarikciden bagimsiz gercek ihtiyac. Yalnizca olcumde kullanilir.
    """
    yol = (kok or DATA_DIR) / kosu_adi / "ground_truth" / "hucre_haftalik.parquet"
    h = (pl.read_parquet(yol, columns=["hafta", "eczane_id", "sku_id", "gercek_tuketim"])
         .filter((pl.col("hafta") > gor.t) & (pl.col("hafta") <= gor.t + ufuk)
                 & (pl.col("gercek_tuketim") > 0)))
    ecz = {e: i for i, e in enumerate(dunya.eczaneler["eczane_id"].to_list())}
    sku = {s: i for i, s in enumerate(dunya.urunler["sku_id"].to_list())}
    matris = np.zeros((dunya.P, dunya.S), dtype=bool)
    if h.height:
        matris[[ecz[e] for e in h["eczane_id"]], [sku[s] for s in h["sku_id"]]] = True
    return Hedef("oracle", matris, matris & ~gor.ikili)


# --------------------------------------------------------------------------
def ust_k_maskesi(skor: np.ndarray, k: int) -> np.ndarray:
    """[P, S] skordan eczane basina en iyi k adayin maskesi.

    Skoru sifir olan aday sayilmaz: uretici o hucre icin sinyal uretmemistir,
    havuza doldurma amaciyla koymak recall'u sahte biçimde yukseltirdi.
    """
    P, S = skor.shape
    k = min(k, S)
    maske = np.zeros((P, S), dtype=bool)
    sira = np.argsort(-skor, axis=1, kind="stable")[:, :k]
    np.put_along_axis(maske, sira, True, axis=1)
    return maske & (skor > 0)


def recall(hedef: np.ndarray, havuz: np.ndarray,
           eczane_maskesi: np.ndarray | None = None) -> float:
    if eczane_maskesi is not None:
        hedef = hedef & eczane_maskesi[:, None]
        havuz = havuz & eczane_maskesi[:, None]
    toplam = hedef.sum()
    return float((hedef & havuz).sum() / toplam) if toplam else float("nan")


def precision(hedef: np.ndarray, havuz: np.ndarray) -> float:
    n = havuz.sum()
    return float((hedef & havuz).sum() / n) if n else float("nan")


def kapsama(havuz: np.ndarray) -> float:
    """Havuzun degdigi farkli SKU orani. Populerlik tabani burada cokerken
    kisisellestirilmis uretici katalogun genisine yayilir."""
    return float((havuz.any(axis=0)).sum() / havuz.shape[1])


def soguk_eczaneler(gor: OriginGorunumu, dilim: float) -> np.ndarray:
    """Siparis satiri sayisina gore en alttaki `dilim` kadar eczane.

    Mutlak esik yerine dilim: esik profile bagli ve `full`da hic baglamiyordu
    (en az gecmisli eczanenin bile 13 siparis satiri var), metrik NaN
    ciktiyordu. Dilim her profilde baglar.
    """
    esik = np.quantile(gor.eczane_siparis_satiri, dilim)
    return gor.eczane_siparis_satiri <= esik


def origin_olcumu(dunya: AdayDunyasi, gor: OriginGorunumu,
                  skorlar: dict[str, np.ndarray], hedefler: list[Hedef],
                  k_degerleri: list[int], soguk_dilim: float) -> list[dict]:
    """Bir origin icin uretici x K x hedef metrik satirlari."""
    soguk = soguk_eczaneler(gor, soguk_dilim)
    satirlar = []
    for uretici, skor in skorlar.items():
        for k in k_degerleri:
            havuz = ust_k_maskesi(skor, k)
            for hedef in hedefler:
                satirlar.append({
                    "origin": gor.t, "uretici": uretici, "k": k,
                    "hedef": hedef.ad,
                    "hedef_sayisi": hedef.sayi,
                    "yeni_hedef_sayisi": hedef.yeni_sayi,
                    "havuz_satiri": int(havuz.sum()),
                    "recall": recall(hedef.matris, havuz),
                    "yeni_hucre_recall": recall(hedef.yeni, havuz),
                    "soguk_eczane_recall": recall(hedef.matris, havuz, soguk),
                    "sicak_eczane_recall": recall(hedef.matris, havuz, ~soguk),
                    "precision": precision(hedef.matris, havuz),
                    "kapsama": kapsama(havuz),
                })
    return satirlar


def liste_olcumu(dunya: AdayDunyasi, gor: OriginGorunumu, teklifler: pl.DataFrame,
                 hedefler: list[Hedef]) -> list[dict]:
    """Vetodan ONCEKI havuz ile SONRAKI oneri listesinin recall karsilastirmasi.

    D6'nin bedeli budur: kisit katmani ML skorunu veto ettikce recall duser.
    Bedeli olcmeden "kisit koyduk" demek anlamsiz.
    """
    def _maske(df: pl.DataFrame) -> np.ndarray:
        m = np.zeros((dunya.P, dunya.S), dtype=bool)
        if df.height:
            m[df["eczane_idx"].to_numpy(), df["sku_idx"].to_numpy()] = True
        return m

    havuz = _maske(teklifler)
    veto_sonrasi = _maske(teklifler.filter(~pl.col("vetolu")))
    liste = _maske(teklifler.filter(pl.col("listede")))
    satirlar = []
    for hedef in hedefler:
        satirlar.append({
            "origin": gor.t, "hedef": hedef.ad,
            "havuz_recall": recall(hedef.matris, havuz),
            "veto_sonrasi_recall": recall(hedef.matris, veto_sonrasi),
            "liste_recall": recall(hedef.matris, liste),
            "havuz_yeni_recall": recall(hedef.yeni, havuz),
            "liste_yeni_recall": recall(hedef.yeni, liste),
            "havuz_precision": precision(hedef.matris, havuz),
            "liste_precision": precision(hedef.matris, liste),
        })
    return satirlar


def veto_ozeti(teklifler: pl.DataFrame, sebepler: tuple[str, ...]) -> dict:
    """Veto oranlari + ust dilimde vetonun agirligi.

    `ust_dilim_veto_orani` D6'nin resmi: skor sirasinin en tepesindeki
    tekliflerin ne kadari kisit tarafindan iptal ediliyor. Bu oran sifirsa
    kisit katmani ML'in zaten yapmadigi bir sey yapmiyordur.
    """
    n = teklifler.height
    if n == 0:
        return {"aday_satiri": 0}
    vetolu = teklifler["vetolu"].to_numpy()
    skor = teklifler["skor"].to_numpy()
    ust = skor >= np.quantile(skor, 1.0 - UST_DILIM)
    ozet = {
        "aday_satiri": n,
        "veto_orani": float(vetolu.mean()),
        "ust_dilim_veto_orani": float(vetolu[ust].mean()),
        "liste_satiri": int(teklifler["listede"].sum()),
        "frekans_budamasi": int((~vetolu).sum() - teklifler["listede"].sum()),
        "mf_kapali_orani": float(
            (~teklifler["mf_izinli"].to_numpy().astype(bool))[~vetolu].mean()),
        "soguk_zincir_yukseltme_orani": float(
            teklifler["soguk_zincir_yukseltildi"].to_numpy().mean()),
    }
    maske = teklifler["veto_maskesi"].to_numpy()
    for bit, ad in enumerate(sebepler):
        ozet[f"veto_{ad}"] = float(((maske >> bit) & 1).mean())
    return ozet
