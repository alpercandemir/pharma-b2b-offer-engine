"""CATE: T-ogrenici ve X-ogrenici. SPEC M4.

Sorulan soru "bu satir siparis verir mi" DEGIL, "bu teklif siparis
olasiligini NE KADAR DEGISTIRIR". Ikisi ayni model ailesiyle tahmin edilir
ama farkli tahmincilerdir ve M4'un olctugu marj farki bu ayrimdan doguyor.

Kollar: 0 = teklif yok (kontrol), 1..A-1 = (mf_orani, vade_gunu).

T-OGRENICI
    Her kol icin ayri bir mu_a(x) = E[Y | X=x, A=a] modeli.
        tau_a(x) = mu_a(x) - mu_0(x)
    Avantaji: hicbir yapisal varsayim yok, kollar birbirini kirletmiyor.
    Dezavantaji: iki gurultulu tahminin FARKI aliniyor. Kol orneklemi
    kucukse fark, sinyalden cok gurultu tasir - ve kollar dengesiz oldugu
    icin (kayit politikasi kontrole agirlik veriyor) tam da bu durumdayiz.

X-OGRENICI
    Ayni mu'lardan baslar, sonra ETKIYI dogrudan modeller:
        tedavi gorenlerde   D1_i = Y_i - mu_0(X_i)
        kontrolde           D0_i = mu_a(X_i) - Y_i
        tau_1a = D1 uzerine regresyon (tedavi orneklemi)
        tau_0a = D0 uzerine regresyon (kontrol orneklemi)
        tau_a(x) = g_a(x) * tau_0a(x) + (1 - g_a(x)) * tau_1a(x)
        g_a(x) = pi_a / (pi_0 + pi_a)          -- KAYIT PROPENSITY'SI (D7)
    Kucuk kolda g_a kucuktur, agirlik buyuk kontrol orneklemiyle egitilmis
    tau_1a'ya kayar. Yani X-ogrenici "hangi orneklem daha bilgiliyse ona
    yaslan" der. D7'nin loglanmis propensity'si burada dogrudan kullanilir;
    loglanmamis olsaydi g tahmin edilmek zorunda kalir ve X-ogrenicinin T'ye
    gore avantaji tahmin hatasina karisirdi.

DESTEK DISI KOL. Bir kolda `min_kol_orneklemi`den az gozlem varsa o kol
icin model KURULMAZ ve tau_a = 0 dondurulur (mu_a = mu_0). Uydurma yerine
"bilmiyorum" demek: ekstrapolasyon yapan bir CATE tahmini, politikayi hic
gozlenmemis bir kola surukler ve M6'nin OPE'si o kolda tanimsizdir.

Bu dosya ground_truth okumaz.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

from core.config import Config


def egitim_originleri(cfg: Config, W: int) -> list[int]:
    """CATE egitiminin origin'leri: olcum penceresinden ONCE ve tamponlu.

    Olcum origin'leri M3'unkilerle ayni (`politika.aday.degerlendirme`) --
    ayni satirlar uzerinde aday uretimi, kisit ve aksiyon secimi ust uste
    okunabilsin diye. Egitim penceresi onlarin en erkenine `sinir_tamponu`
    kadar mesafede biter; ortusme config yuklemesinde zaten reddediliyor
    (core/config.py `_m4_zaman_kilidi`).

    Pencere tasinca EN GEC origin'ler tutulur: olcume zaman olarak en yakin
    donem. Rejim olaylari ve mevsimsellik nedeniyle uzak gecmis daha az
    temsili.
    """
    e = cfg.uplift.egitim
    d = cfg.politika.aday.degerlendirme
    son_olcum = W - 1 - d.ufuk_hafta
    ilk_olcum = son_olcum - (d.origin_sayisi - 1) * d.ufuk_hafta
    sinir = ilk_olcum - e.sinir_tamponu_hafta
    originler = list(range(e.ilk_origin_hafta, sinir + 1, e.origin_araligi_hafta))
    if not originler:
        raise ValueError(
            f"CATE egitim origin'i kalmadi: W={W}, ilk_origin={e.ilk_origin_hafta}, "
            f"ilk olcum origin'i={ilk_olcum}, tampon={e.sinir_tamponu_hafta}")
    return originler[-e.azami_origin_sayisi:]


def _model_argumanlari(cfg: Config, kategorik: list[int]) -> dict:
    m = cfg.uplift.model
    return dict(
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
        categorical_features=kategorik,
    )


class _SabitSiniflandirici:
    """Tek sinifli (ya da destek disi) kol icin sabit olasilik dondurur.

    HistGradientBoosting tek sinifli etikette hata veriyor; bu durumda
    "ogrenecek bir sey yok" cevabini uydurmadan vermek gerekiyor.
    """

    def __init__(self, oran: float) -> None:
        self.oran = float(oran)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.full(X.shape[0], self.oran)
        return np.column_stack([1.0 - p, p])


@dataclass
class TOgrenici:
    """Kol basina mu_a(x). Hem propensity hem uplift politikasi bunu kullanir.

    Iki politikanin AYNI olasilik matrisini kullanmasi bilincli: boylece
    olculen marj farki model kalitesinden degil, YALNIZCA amac
    fonksiyonundan gelir (reports/m4.md 5).
    """

    cfg: Config
    A: int
    kategorik: list[int]
    modeller: dict[int, object] = field(default_factory=dict)
    kol_orneklemi: dict[int, int] = field(default_factory=dict)
    destekli: set[int] = field(default_factory=set)

    def egit(self, X: np.ndarray, kol: np.ndarray, y: np.ndarray) -> "TOgrenici":
        asgari = self.cfg.uplift.model.min_kol_orneklemi
        genel = float(y.mean()) if y.size else 0.0
        # Kontrol kolu ONCE egitilir: destek disi kollar onun modelini
        # AYNEN kullanir, boylece tau_a tam olarak sifir olur ("bilmiyorum"),
        # sabit bir orana dusmez (o, sahte bir CATE uretirdi).
        for a in [0] + list(range(1, self.A)):
            sec = kol == a
            n = int(sec.sum())
            self.kol_orneklemi[a] = n
            ya = y[sec]
            if n < asgari or ya.size == 0 or ya.min() == ya.max():
                self.modeller[a] = (self.modeller[0] if a > 0 and 0 in self.modeller
                                    else _SabitSiniflandirici(
                                        float(ya.mean()) if ya.size else genel))
                continue
            model = HistGradientBoostingClassifier(
                **_model_argumanlari(self.cfg, self.kategorik))
            model.fit(X[sec], ya)
            self.modeller[a] = model
            self.destekli.add(a)
        return self

    def olasilik(self, X: np.ndarray) -> np.ndarray:
        """[n, A] her kol altinda tahmini kabul olasiligi."""
        cikti = np.empty((X.shape[0], self.A))
        for a in range(self.A):
            cikti[:, a] = self.modeller[a].predict_proba(X)[:, 1]
        return cikti

    def cate(self, X: np.ndarray) -> np.ndarray:
        p = self.olasilik(X)
        return p - p[:, [0]]


@dataclass
class XOgrenici:
    """T-ogreniciyi birinci asama olarak kullanan X-ogrenici.

    Birinci asamayi yeniden egitmez: X-ogreniciNIN ilk adimi zaten
    T-ogrenicidir. Ayri egitmek hem iki kat sure hem de "iki model farkli
    mu kullaniyor" karisikligi anlamina gelirdi.
    """

    cfg: Config
    t_ogrenici: TOgrenici
    kategorik: list[int]
    tau0: dict[int, object] = field(default_factory=dict)
    tau1: dict[int, object] = field(default_factory=dict)

    @property
    def A(self) -> int:
        return self.t_ogrenici.A

    def egit(self, X: np.ndarray, kol: np.ndarray, y: np.ndarray) -> "XOgrenici":
        t = self.t_ogrenici
        asgari = self.cfg.uplift.model.min_kol_orneklemi
        kontrol = kol == 0
        mu0_hepsi = t.modeller[0].predict_proba(X)[:, 1]
        for a in range(1, self.A):
            if a not in t.destekli:
                continue
            tedavi = kol == a
            mua_kontrol = t.modeller[a].predict_proba(X[kontrol])[:, 1]
            D1 = y[tedavi] - mu0_hepsi[tedavi]
            D0 = mua_kontrol - y[kontrol]
            if tedavi.sum() >= asgari:
                self.tau1[a] = HistGradientBoostingRegressor(
                    **_model_argumanlari(self.cfg, self.kategorik)).fit(X[tedavi], D1)
            if kontrol.sum() >= asgari:
                self.tau0[a] = HistGradientBoostingRegressor(
                    **_model_argumanlari(self.cfg, self.kategorik)).fit(X[kontrol], D0)
        return self

    def cate(self, X: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """[n, A] CATE. `pi` kayit politikasinin olasilik matrisi (D7)."""
        x = self.cfg.uplift.x_ogrenici
        tau = np.zeros((X.shape[0], self.A))
        for a in range(1, self.A):
            if a not in self.tau1 and a not in self.tau0:
                continue
            g = np.clip(pi[:, a] / np.maximum(pi[:, 0] + pi[:, a], 1e-12),
                        x.egilim_kirpma_alt, x.egilim_kirpma_ust)
            t1 = self.tau1[a].predict(X) if a in self.tau1 else 0.0
            t0 = self.tau0[a].predict(X) if a in self.tau0 else 0.0
            tau[:, a] = g * t0 + (1.0 - g) * t1
        return tau

    def olasilik(self, X: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """mu_0 + tau_a, [0, 1] araligina kirpilmis.

        Kirpma sart: tau bir REGRESYON ciktisi, olasilik olmak zorunda degil.
        Kirpilmadan marj hesabina girerse olasilik 1'i asan satirlar politika
        siralamasini bozar.
        """
        mu0 = self.t_ogrenici.modeller[0].predict_proba(X)[:, 1]
        return np.clip(mu0[:, None] + self.cate(X, pi), 0.0, 1.0)
