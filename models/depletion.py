"""Tukenme zamani modeli (D2) ve karsisina konan naif kural.

Dort tahminci, HEPSI ayni gozlemlenebilir etiketle beslenir:

  sabit        : taban oran. MAE'de kirpilmis sureyi hep medyan tahmin eder.
                 Bunu gecemeyen bir model bir sey ogrenmemistir.
  kural        : "son N gunde aldi mi". D2'nin karsisina koydugu naif kural.
                 En iyi sansi verilir: N egitim doneminde AUC'yi maksimize
                 edecek sekilde secilir, siralama icin surekli "son siparisten
                 gecen gun" kullanilir, olasiligi kova kalibrasyonuyla uretilir.
  defter       : stok_tahmini / hiz_tahmini. Ogrenme yok, muhasebe var.
  hazard       : ayrik zamanli hazard. h(k) = P(olay = k | olay >= k, x),
                 k = 1..ufuk. Sagdan sansuru dogal olarak isler.

Ayrik zamanli hazard neden Cox degil:
  - Sansur ve baglar (ties) haftalik izgarada cok yogun; ayrik zaman bunun
    dogal formulasyonu.
  - Cikti dogrudan OLASILIK: "H hafta icinde tukenme" kalibre edilebilir ve
    M4'un beklenen marj hesabina girebilir. Cox'un kismi olabilirligi taban
    hazard'i vermez.
  - Etkiler oransal degil: mevsimsellik ve olaylar hazard'i sekil olarak
    degistirir. Ayrik zamanda k bir ozelliktir, oransallik varsayimi yoktur.

Tahmin edilen tukenme suresi = kirpilmis ortalama omur (restricted mean):
    T_tahmin = sum_{k=1..ufuk} S(k),   S(k) = prod_{j<=k} (1 - h(j))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from core.config import Config
from features.panel import Panel
from sim.calendar import GUN_HAFTA


@dataclass
class Tahmin:
    """Bir tahmincinin bir satir kumesi uzerindeki ciktisi."""

    olasilik: np.ndarray        # P(olay <= karar_ufku)
    skor: np.ndarray           # siralama skoru (buyuk = daha erken tukenir)
    tukenme_hafta: np.ndarray  # kirpilmis tukenme suresi tahmini


class KovaKalibratoru:
    """Skoru olasiliga cevirir: egitimde esit frekansli kovalar, kova frekansi.

    Isotonic yerine kova: az sayida knob, sizmaya kapali, ve kalibrasyon
    egrisinde ne oldugu gozle gorulur. Kova sayisi config'ten gelir.
    """

    def __init__(self, kova_sayisi: int) -> None:
        self.kova_sayisi = kova_sayisi
        self.sinirlar: np.ndarray | None = None
        self.degerler: np.ndarray | None = None

    def egit(self, skor: np.ndarray, y: np.ndarray) -> "KovaKalibratoru":
        q = np.linspace(0, 1, self.kova_sayisi + 1)[1:-1]
        self.sinirlar = np.unique(np.quantile(skor, q))
        kova = np.searchsorted(self.sinirlar, skor, side="right")
        n = self.sinirlar.size + 1
        toplam = np.bincount(kova, weights=y, minlength=n)
        adet = np.bincount(kova, minlength=n)
        genel = float(y.mean()) if y.size else 0.0
        self.degerler = np.where(adet > 0, toplam / np.maximum(adet, 1), genel)
        return self

    def uygula(self, skor: np.ndarray) -> np.ndarray:
        if self.sinirlar is None or self.degerler is None:
            raise RuntimeError("kalibrator egitilmedi")
        return self.degerler[np.searchsorted(self.sinirlar, skor, side="right")]


# --------------------------------------------------------------------------
# ortak yardimcilar
# --------------------------------------------------------------------------
def gozlemlenebilir_olay(panel: Panel, idx: np.ndarray, ufuk: int) -> np.ndarray:
    """Egitim etiketi: origin sonrasi `ufuk` hafta icinde bize siparis geldi mi."""
    k = panel.etiket_k[idx]
    return ((k > 0) & (k <= ufuk)).astype(int)


def _k_sutunlari(panel: Panel, idx: np.ndarray, k: np.ndarray) -> np.ndarray:
    """[len(idx), 4] -> k ve k periyodunun takvimi (ay, ramazan, yil sonu).

    Takvim gelecege ait ama DETERMINISTIK ve kamuya acik: gercek bir sistem de
    onumuzdeki haftanin ayini bilir. Sizinti degil.
    """
    W = panel.takvim_k.shape[0]
    hafta = np.clip(panel.origin[idx] + k, 0, W - 1).astype(int)
    return np.column_stack([k.astype(np.float32), panel.takvim_k[hafta]])


def _tasarim(panel: Panel, idx: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.hstack([panel.X[idx], _k_sutunlari(panel, idx, k)]).astype(np.float32)


def kisi_periyot(panel: Panel, idx: np.ndarray, etiket_k: np.ndarray,
                 izlenen_k: np.ndarray, ufuk: int) -> tuple[np.ndarray, np.ndarray]:
    """(satir, periyot) tablosu: her satir olay veya sansure kadar acilir."""
    olay_k = np.where((etiket_k > 0) & (etiket_k <= ufuk), etiket_k, 0)
    son = np.where(olay_k > 0, olay_k, np.minimum(izlenen_k, ufuk))
    son = np.maximum(son, 0)
    tekrar = son.astype(int)
    satir = np.repeat(np.arange(idx.size), tekrar)
    # her satirin kendi icinde 1..tekrar sayaci (vektorize)
    baslangic = np.repeat(np.cumsum(tekrar) - tekrar, tekrar)
    k = np.arange(satir.size) - baslangic + 1
    y = (k == olay_k[satir]).astype(np.int8)
    return satir, np.column_stack([k, y]).astype(np.int32)


# --------------------------------------------------------------------------
# tahminciler
# --------------------------------------------------------------------------
class HazardTahmincisi:
    ad = "hazard"

    def __init__(self, cfg: Config, panel: Panel) -> None:
        self.cfg = cfg
        m = cfg.tukenme.model
        self.model = HistGradientBoostingClassifier(
            learning_rate=m.ogrenme_orani,
            max_iter=m.azami_agac,
            max_leaf_nodes=m.azami_yaprak,
            min_samples_leaf=m.min_yaprak_ornegi,
            l2_regularization=m.l2_duzenlilestirme,
            max_features=m.ozellik_orani,
            early_stopping=m.erken_durdurma,
            validation_fraction=m.dogrulama_orani,
            n_iter_no_change=m.sabir,
            random_state=m.seed,
            categorical_features=panel.kategorik_idx,
        )

    def egit(self, panel: Panel, idx: np.ndarray, etiket_k: np.ndarray,
             izlenen_k: np.ndarray) -> "HazardTahmincisi":
        ufuk = self.cfg.tukenme.hedef.ufuk_hafta
        m = self.cfg.tukenme.model
        satir, ky = kisi_periyot(panel, idx, etiket_k, izlenen_k, ufuk)
        if m.azami_egitim_satiri and satir.size > m.azami_egitim_satiri:
            # Satir tavani: (hucre, origin) ciftleri seed'li alt orneklenir,
            # periyotlar BOLUNMEZ - yoksa sansur yapisi bozulur.
            rng = np.random.default_rng(m.seed)
            pay = m.azami_egitim_satiri / satir.size
            tut = rng.random(idx.size) < pay
            kalan = tut[satir]
            satir, ky = satir[kalan], ky[kalan]
        X = _tasarim(panel, idx[satir], ky[:, 0])
        self.model.fit(X, ky[:, 1])
        self.egitim_satiri = int(X.shape[0])
        return self

    def hazard_egrisi(self, panel: Panel, idx: np.ndarray) -> np.ndarray:
        """[len(idx), ufuk] hazard matrisi. k dongusu bellek icin."""
        ufuk = self.cfg.tukenme.hedef.ufuk_hafta
        h = np.empty((idx.size, ufuk))
        for j, k in enumerate(range(1, ufuk + 1)):
            X = _tasarim(panel, idx, np.full(idx.size, k))
            h[:, j] = self.model.predict_proba(X)[:, 1]
        return h

    def tahmin(self, panel: Panel, idx: np.ndarray) -> Tahmin:
        H = self.cfg.tukenme.hedef.karar_ufku_hafta
        h = self.hazard_egrisi(panel, idx)
        S = np.cumprod(1.0 - h, axis=1)
        olasilik = 1.0 - S[:, H - 1]
        return Tahmin(olasilik=olasilik, skor=olasilik, tukenme_hafta=S.sum(axis=1))


class KuralTahmincisi:
    """'Son N gunde aldi mi' - D2'nin karsisina koydugu kural.

    `ikili=True` kuralin LITERAL hali: config'teki sabit N (varsayilan 30 gun),
        tek esik, iki olasilik seviyesi. Sahada kullanilan bicim budur.
    `ikili=False` kurala verilebilecek EN IYI hal: N egitim doneminde AUC'yi
        maksimize edecek sekilde secilir, siralama surekli "gecen gun"
        uzerinden yapilir, olasilik kova kalibrasyonuyla uretilir.

    Ikisi de raporlanir: aradaki fark "kurali daha iyi kullansaydik ne
    kazanirdik", ikisinin de 0.5'e yakinligi "kural bu soruyu cevaplamiyor".
    """

    def __init__(self, cfg: Config, panel: Panel, ikili: bool) -> None:
        self.cfg = cfg
        self.ikili = ikili
        self.ad = "kural_ikili" if ikili else "kural"
        self.sutun = panel.ozellik_adlari.index("gecen_hafta_siparis")
        self.kalibrator = KovaKalibratoru(cfg.tukenme.degerlendirme.kalibrasyon_kova_sayisi)

    def _gecen_gun(self, panel: Panel, idx: np.ndarray) -> np.ndarray:
        return panel.X[idx, self.sutun].astype(float) * GUN_HAFTA

    def egit(self, panel: Panel, idx: np.ndarray, y: np.ndarray) -> "KuralTahmincisi":
        gecen = self._gecen_gun(panel, idx)
        varsayilan = self.cfg.tukenme.taban_kural.son_n_gun
        if self.ikili:
            # Literal kural: N config'ten gelir, veriye uydurulmaz.
            self.n_gun = varsayilan
        else:
            adaylar = self.cfg.tukenme.taban_kural.n_gun_adaylari
            tek_sinif = y.size == 0 or y.min() == y.max()
            skorlar = [float("nan") if tek_sinif else roc_auc_score(y, (gecen > n).astype(float))
                       for n in adaylar]
            gecerli = [(s, n) for s, n in zip(skorlar, adaylar) if np.isfinite(s)]
            self.n_gun = max(gecerli)[1] if gecerli else varsayilan
        self.kalibrator.egit((gecen > self.n_gun).astype(float) if self.ikili else gecen, y)
        return self

    def tahmin(self, panel: Panel, idx: np.ndarray) -> Tahmin:
        ufuk = self.cfg.tukenme.hedef.ufuk_hafta
        gecen = self._gecen_gun(panel, idx)
        ham = (gecen > self.n_gun).astype(float) if self.ikili else gecen
        # Kuralin zaman tahmini: stok, son siparisten N gun sonra biter.
        kalan = np.clip((self.n_gun - gecen) / GUN_HAFTA, 0.0, ufuk)
        return Tahmin(olasilik=self.kalibrator.uygula(ham), skor=ham, tukenme_hafta=kalan)


class DefterTahmincisi:
    """stok_tahmini / hiz_tahmini. Ogrenme yok; sadece muhasebe + kalibrasyon."""

    ad = "defter"

    def __init__(self, cfg: Config, panel: Panel, sutun: str = "defter_tukenme_hafta") -> None:
        self.cfg = cfg
        self.sutun_adi = sutun
        self.sutun = panel.ozellik_adlari.index(sutun)
        self.kalibrator = KovaKalibratoru(cfg.tukenme.degerlendirme.kalibrasyon_kova_sayisi)

    def egit(self, panel: Panel, idx: np.ndarray, y: np.ndarray) -> "DefterTahmincisi":
        self.kalibrator.egit(-panel.X[idx, self.sutun].astype(float), y)
        return self

    def tahmin(self, panel: Panel, idx: np.ndarray) -> Tahmin:
        ufuk = self.cfg.tukenme.hedef.ufuk_hafta
        ttd = np.clip(panel.X[idx, self.sutun].astype(float), 0.0, ufuk)
        return Tahmin(olasilik=self.kalibrator.uygula(-ttd), skor=-ttd, tukenme_hafta=ttd)


class SabitTahminci:
    """Taban oran. MAE'de kirpilmis surenin medyani, olasilikta taban oran."""

    ad = "sabit"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def egit(self, panel: Panel, idx: np.ndarray, y: np.ndarray) -> "SabitTahminci":
        ufuk = self.cfg.tukenme.hedef.ufuk_hafta
        k = panel.etiket_k[idx]
        sure = np.where((k > 0) & (k <= ufuk), k, ufuk)
        self.oran = float(y.mean()) if y.size else 0.0
        self.sure = float(np.median(sure)) if sure.size else float(ufuk)
        return self

    def tahmin(self, panel: Panel, idx: np.ndarray) -> Tahmin:
        return Tahmin(olasilik=np.full(idx.size, self.oran),
                      skor=np.zeros(idx.size),
                      tukenme_hafta=np.full(idx.size, self.sure))
