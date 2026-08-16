"""Off-policy degerlendirme: IPS / SNIPS / Doubly-Robust (SPEC 3 `eval/ope.py`).

SORU. Elimizde bir KAYIT politikasinin (policy/bandit.py, D7) urettigi loglar
var: her satir icin secilen kol, o kolun secilme olasiligi ve GERCEKLESEN
odul. Baska bir politikayi canliya almadan degerini tahmin edebilir miyiz?

    V(pi) = E_x[ r(x, pi(x)) ]

Uc tahminci, uc farkli takas:

    IPS    yansiz ama VARYANSLI. w_i = 1[a_i = pi(x_i)] / pi_log(a_i|x_i);
           tek bir dusuk-propensity satiri tahmini tasiyabilir. Kirpma
           varyansi kirar ve YANLILIK ekler -- M6'nin merkezindeki takas.
    SNIPS  agirliklari kendi toplamiyla normalize eder. Kucuk orneklemde
           yanli, ama varyansi cok daha dusuk ve olcek kaymasina dayanikli:
           agirliklarin ortalamasi 1'den saparsa IPS olceginden kopar, SNIPS
           kopmaz.
    DR     bir SONUC MODELI q(x, a) ekler ve onem agirligini yalnizca ARTIGA
           uygular. Model dogruysa varyans duser; propensity dogruysa yanlilik
           kalmaz. "Cift saglam" adi bundan: ikisinden BIRI dogru olsun yeter.
           Bedeli: q'nun destegi olmayan (x, a) bolgesinde EKSTRAPOLE etmesi.

BU DOSYA ORACLE OKUMAZ. Bilerek: burada uretilen her teshis (ESS, agirlik
dagilimi, ortusme, kol destegi, propensity kalibrasyonu) gercek hayatta da
elde olan seylerdir. Tahmincinin gercekten ne kadar saptigi ancak
`eval/oracle.py` ile bilinir ve o karsilastirma `eval/report.py`de yapilir.
Ayrimi korumak M6'nin ogrettigi seyin ta kendisi: canlida hangi teshise
bakarak "bu tahmine guvenme" diyebilirdik?

ODUL OLCEGI. Bir satirin odulu `gerceklesen_miktar_carpani x brut_marj(kol)`
(TL). Kol 0 ("teklif yok") da bir odul tasir: eczane organik siparisini yine
verebilir. Boylece V(pi) M4'un `politika_olcumu`yla AYNI olcektedir ve oracle
karsilastirmasi birim cevirmeden yapilabilir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from core.config import Config

# Sifira bolme korumasi. Sayisal sabit, knob degil.
EPSILON = 1e-12
# Kontrol kolunun indeksi (policy.scorer.TEKLIF_YOK ile ayni olmak zorunda).
TEKLIF_YOK = 0


# --------------------------------------------------------------------------
# loglanmis veri
# --------------------------------------------------------------------------
@dataclass
class LoglanmisVeri:
    """Bir kayit kosusunun ciktisi. OPE'nin gorebildigi HER SEY burada.

    Gercek kabul olasiliklari, gercek CATE, oracle degeri -- hicbiri yok.
    Bu sinifin alanlarina bir ground_truth buyuklugu eklenirse M6'nin butun
    iddiasi coker; tests/test_ope.py alan listesini kilitler.
    """

    X: np.ndarray             # [n, F] ozellik matrisi (sonuc modeli icin)
    kol: np.ndarray           # [n] loglanan aksiyon
    propensity: np.ndarray    # [n] pi_log(a_i | x_i), gosterim aninda yazildi
    pi_log: np.ndarray        # [n, A] tum kollarin olasiligi
    odul: np.ndarray          # [n] gerceklesen marj (TL)
    izinli: np.ndarray        # [n, A]
    eczane_idx: np.ndarray    # [n] blok bootstrap birimi
    origin: np.ndarray        # [n]
    tekrar: np.ndarray        # [n] kacinci bagimsiz kayit kosusu

    @property
    def n(self) -> int:
        return self.kol.size

    @property
    def A(self) -> int:
        return self.pi_log.shape[1]


def birlestir(parcalar: list[LoglanmisVeri]) -> LoglanmisVeri:
    var = [p for p in parcalar if p.n]
    if not var:
        bos = parcalar[0]
        return bos
    return LoglanmisVeri(
        X=np.vstack([p.X for p in var]),
        kol=np.concatenate([p.kol for p in var]),
        propensity=np.concatenate([p.propensity for p in var]),
        pi_log=np.vstack([p.pi_log for p in var]),
        odul=np.concatenate([p.odul for p in var]),
        izinli=np.vstack([p.izinli for p in var]),
        eczane_idx=np.concatenate([p.eczane_idx for p in var]),
        origin=np.concatenate([p.origin for p in var]),
        tekrar=np.concatenate([p.tekrar for p in var]),
    )


# --------------------------------------------------------------------------
# propensity: loglanan / bozulmus / kestirilmis
# --------------------------------------------------------------------------
def sicaklik_uygula(pi: np.ndarray, sicaklik: float) -> np.ndarray:
    """[n, A] olasilik matrisine sicaklik uygular ve yeniden normalize eder.

    p'  ~ p ** (1 / sicaklik)

    sicaklik > 1 dagilimi DUZLESTIRIR (log politikasini oldugundan daha
    kesifci sanmak), < 1 KESKINLESTIRIR (oldugundan daha kararli sanmak).
    Bu bir tuning kadrani degil, KONTROLLU BIR BOZMA: kalibrasyon hatasinin
    yonunu bilerek verip IPS'in hangi yone kaydigini gostermek icin var.

    1.0'da fonksiyon birim fonksiyondur (yeniden normalizasyon dahil aynidir).
    """
    if sicaklik == 1.0:
        return pi
    guclu = np.power(np.maximum(pi, 0.0), 1.0 / sicaklik)
    toplam = guclu.sum(axis=1, keepdims=True)
    return np.where(toplam > 0, guclu / np.maximum(toplam, EPSILON), pi)


@dataclass
class PropensityCiktisi:
    propensity: np.ndarray     # [n] kullanilacak pi_hat(a_i | x_i)
    pi: np.ndarray             # [n, A]
    kaynak: str
    # Loglanana gore kalibrasyon teshisi. SENTETIK LUKSU: gercek hayatta
    # "dogru" propensity bilinmediginden bu tablo cikarilamaz; burada
    # cikarilabiliyor ve kalibrasyon hatasinin IPS'i ne kadar kaydirdigi
    # olculebiliyor.
    ortalama_mutlak_hata: float
    kalibrasyon_hatasi: float   # kova bazli |ortalama tahmin - ortalama gercek|
    log_orani_ortalamasi: float  # E[log(pi_hat / pi_log)] -- yonu gosterir


def _kova_kalibrasyonu(tahmin: np.ndarray, gercek: np.ndarray, kova: int) -> float:
    if tahmin.size == 0:
        return float("nan")
    sira = np.argsort(tahmin, kind="stable")
    parcalar = np.array_split(sira, min(kova, max(1, tahmin.size)))
    agirlik, hata = 0.0, 0.0
    for p in parcalar:
        if p.size == 0:
            continue
        hata += p.size * abs(float(tahmin[p].mean()) - float(gercek[p].mean()))
        agirlik += p.size
    return hata / agirlik if agirlik else float("nan")


class PropensityModeli:
    """Loglanan aksiyonu ozelliklerden kestiren cok sinifli model.

    NEDEN VAR. D7 propensity'nin loglanmasini sart kosuyor ve bu POC'ta
    loglaniyor. Ama sahada en sik gorulen ariza tam olarak bunun eksikligidir:
    log ya hic tutulmamistir ya da tutan sistem degismistir. O durumda
    propensity VERIDEN kestirilir ve IPS'in paydasi artik bir olcum degil,
    bir TAHMINDIR. `ope.propensity.kaynak = tahmin` bu dunyayi kurar.

    Kayit politikasi bir softmax karisimidir (policy/bandit.py); agac modeli
    onu tam ogrenemez. Kalan hata M6'nin "propensity kalibrasyonu" kalemidir
    ve uydurulmus degil, MEKANIK olarak dogar.
    """

    def __init__(self, cfg: Config, A: int) -> None:
        m = cfg.ope.propensity.model
        self.A = A
        self.alt = cfg.ope.propensity.kirpma_alt
        self.model = HistGradientBoostingClassifier(
            learning_rate=m.ogrenme_orani, max_iter=m.azami_agac,
            max_leaf_nodes=m.azami_yaprak, min_samples_leaf=m.min_yaprak_ornegi,
            l2_regularization=m.l2_duzenlilestirme, random_state=m.seed,
            early_stopping=False,
        )
        self._siniflar: np.ndarray | None = None

    def egit(self, X: np.ndarray, kol: np.ndarray) -> "PropensityModeli":
        self.model.fit(X, kol)
        self._siniflar = self.model.classes_
        return self

    def olasilik(self, X: np.ndarray) -> np.ndarray:
        """[n, A] tam olasilik matrisi. Egitimde gorulmemis kol tam sifirdir.

        Sifir birakiliyor cunku alt kirpma ZATEN uygulanacak; burada uydurma
        bir taban vermek "model o kolu gordu" yalanini kurardi.
        """
        ham = self.model.predict_proba(X)
        pi = np.zeros((X.shape[0], self.A))
        pi[:, self._siniflar.astype(int)] = ham
        return pi


def propensity_hazirla(cfg: Config, veri: LoglanmisVeri) -> PropensityCiktisi:
    """OPE'nin kullanacagi propensity'yi uretir ve kalibrasyonunu olcer."""
    p = cfg.ope.propensity
    idx = np.arange(veri.n)
    if p.kaynak == "tahmin":
        model = PropensityModeli(cfg, veri.A).egit(veri.X, veri.kol)
        pi = model.olasilik(veri.X)
    else:
        pi = veri.pi_log
    pi = sicaklik_uygula(pi, p.sicaklik)
    prop = np.clip(pi[idx, veri.kol], p.kirpma_alt, 1.0)

    gercek = veri.propensity
    oran = np.log(np.maximum(prop, EPSILON)) - np.log(np.maximum(gercek, EPSILON))
    return PropensityCiktisi(
        propensity=prop, pi=pi,
        kaynak=f"{p.kaynak}(sicaklik={p.sicaklik})",
        ortalama_mutlak_hata=float(np.mean(np.abs(prop - gercek))) if veri.n else float("nan"),
        kalibrasyon_hatasi=_kova_kalibrasyonu(prop, gercek, p.kalibrasyon_kova_sayisi),
        log_orani_ortalamasi=float(oran.mean()) if veri.n else float("nan"),
    )


# --------------------------------------------------------------------------
# onem agirliklari
# --------------------------------------------------------------------------
@dataclass
class Agirliklar:
    ham: np.ndarray           # [n] kirpmasiz w
    kirpik: np.ndarray        # [n] min(w, tavan)
    esles: np.ndarray         # [n] a_i == pi(x_i)
    tavan: float

    @property
    def kirpilan_satir(self) -> np.ndarray:
        return self.ham > self.tavan

    @property
    def silinen_kutle(self) -> float:
        """Kirpmanin attigi agirlik kutlesi / ham kutle. Yanliligin olcegi."""
        toplam = float(self.ham.sum())
        return float((self.ham - self.kirpik).sum() / toplam) if toplam > 0 else 0.0


def onem_agirligi(kol_hedef: np.ndarray, veri: LoglanmisVeri,
                  propensity: np.ndarray, tavan: float) -> Agirliklar:
    """w_i = 1[a_i = pi(x_i)] / pi_log(a_i | x_i), tavanla kirpilmis."""
    esles = veri.kol == kol_hedef
    ham = np.where(esles, 1.0 / np.maximum(propensity, EPSILON), 0.0)
    return Agirliklar(ham=ham, kirpik=np.minimum(ham, tavan), esles=esles,
                      tavan=tavan)


# --------------------------------------------------------------------------
# tahminciler
# --------------------------------------------------------------------------
def ips(odul: np.ndarray, w: np.ndarray) -> float:
    """(1/n) sum w_i r_i. Yansiz -- ama yalnizca w kirpilmamissa."""
    return float(np.mean(w * odul)) if odul.size else float("nan")


def snips(odul: np.ndarray, w: np.ndarray) -> float:
    """sum(w r) / sum(w). Kendinden normalize; sum(w)=0 ise tanimsiz."""
    payda = float(w.sum())
    if odul.size == 0 or payda <= EPSILON:
        return float("nan")
    return float((w * odul).sum() / payda)


def dogrudan_yontem(q_hedef: np.ndarray) -> float:
    """(1/n) sum q(x_i, pi(x_i)). Sadece sonuc modeli; onem agirligi YOK.

    DR'nin "model tarafi". Tek basina raporlaniyor cunku DR ile arasindaki
    fark, duzeltme teriminin ne kadar is yaptigini gosterir: ikisi ayniysa
    duzeltme olu demektir (eslesme orani cok dusuk).
    """
    return float(np.mean(q_hedef)) if q_hedef.size else float("nan")


def doubly_robust(odul: np.ndarray, w: np.ndarray, q_secilen: np.ndarray,
                  q_hedef: np.ndarray) -> float:
    """(1/n) sum [ q(x, pi(x)) + w_i (r_i - q(x_i, a_i)) ].

    Onem agirligi yalnizca ARTIGI olcekler. Sonuc modeli iyiyse artik kucuktur
    ve buyuk bir w bile tahmini ucurmaz -- IPS'e gore varyans kazanci buradan
    gelir. Model kotuyse DR, IPS'e geri doner (artik = odul mertebesinde).
    """
    if odul.size == 0:
        return float("nan")
    return float(np.mean(q_hedef + w * (odul - q_secilen)))


# --------------------------------------------------------------------------
# teshis (yalnizca loglardan -- oracle yok)
# --------------------------------------------------------------------------
@dataclass
class OPETeshisi:
    n: int
    eslesme_orani: float           # hedef politika loglarda ne siklikta gorundu
    ess: float                     # etkin orneklem (sum w)^2 / sum w^2
    ess_orani: float               # ess / n
    agirlik_ortalamasi: float      # yansizlikta 1.0 olmali
    agirlik_azami: float
    agirlik_p99: float
    kirpilan_satir_orani: float
    kirpilan_kutle_orani: float
    ortusme_ihlali_orani: float    # pi_log(pi(x)|x) < esik olan satir orani
    ortusme_ihlali_odul_payi: float  # o satirlarin IPS toplamindaki payi
    dusuk_destekli_kol_sayisi: int
    dusuk_destege_giden_satir_orani: float  # hedefin destegi zayif kola gonderdigi pay


def teshis(cfg: Config, veri: LoglanmisVeri, kol_hedef: np.ndarray,
           ag: Agirliklar, pi: np.ndarray) -> OPETeshisi:
    """Canlida da elde olacak teshisler. Hicbiri gercek degeri kullanmaz."""
    o = cfg.ope.ortusme
    n = veri.n
    if n == 0:
        nan = float("nan")
        return OPETeshisi(0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 0, nan)

    w = ag.kirpik
    kare = float((w * w).sum())
    ess = float(w.sum() ** 2 / kare) if kare > 0 else 0.0

    # Ortusme: hedef politikanin sectigi kolun LOGLANAN olasiligi.
    idx = np.arange(n)
    hedef_prop = pi[idx, kol_hedef]
    ihlal = hedef_prop < o.esik
    ips_katki = w * veri.odul
    toplam_katki = float(np.abs(ips_katki).sum())

    # Destek: bir kol loglarda kac kez secildi. Hedef politikanin destegi zayif
    # bir kola gonderdigi satirlar DR'nin ekstrapole ettigi yerdir.
    sayim = np.bincount(veri.kol, minlength=veri.A)
    zayif = sayim < o.dusuk_destek_orneklemi
    # Hicbir satirda izinli olmayan kol "destegi zayif" sayilmaz: o kol
    # zaten hicbir politikanin secebilecegi bir sey degil.
    kullanilabilir = veri.izinli.any(axis=0)
    zayif &= kullanilabilir

    return OPETeshisi(
        n=n,
        eslesme_orani=float(ag.esles.mean()),
        ess=ess,
        ess_orani=ess / n,
        agirlik_ortalamasi=float(w.mean()),
        agirlik_azami=float(ag.ham.max()),
        agirlik_p99=float(np.percentile(ag.ham, 99)),
        kirpilan_satir_orani=float(ag.kirpilan_satir.mean()),
        kirpilan_kutle_orani=ag.silinen_kutle,
        ortusme_ihlali_orani=float(ihlal.mean()),
        ortusme_ihlali_odul_payi=(float(np.abs(ips_katki[ihlal]).sum() / toplam_katki)
                                  if toplam_katki > 0 else 0.0),
        dusuk_destekli_kol_sayisi=int(zayif.sum()),
        dusuk_destege_giden_satir_orani=float(zayif[kol_hedef].mean()),
    )


# --------------------------------------------------------------------------
# bir politikanin tam degerlendirmesi
# --------------------------------------------------------------------------
@dataclass
class OPESonucu:
    ad: str
    ips: float
    ips_kirpmasiz: float
    snips: float
    dogrudan: float
    dr: float
    teshis: OPETeshisi
    # Blok bootstrap araliklari (bagimsizlik birimi ECZANE, M4/M5 ile ayni).
    ips_alt: float
    ips_ust: float
    dr_alt: float
    dr_ust: float

    def deger(self, tahminci: str) -> float:
        return {"ips": self.ips, "ips_kirpmasiz": self.ips_kirpmasiz,
                "snips": self.snips, "dogrudan": self.dogrudan,
                "dr": self.dr}[tahminci]


TAHMINCILER = ("ips", "ips_kirpmasiz", "snips", "dogrudan", "dr")


def _blok_bootstrap(veri: LoglanmisVeri, olcut, tekrar: int,
                    seed: int) -> tuple[float, float]:
    """Eczane blok bootstrap'i: satirlar degil ECZANELER yeniden orneklenir.

    Ayni eczanenin satirlari ortak frekans tavanini ve ortak kredi limitini
    paylasiyor; satir bazli bootstrap onlari bagimsiz sayar ve araligi
    oldugundan DAR gosterir (M4'te olculmus ayni tuzak).
    """
    if veri.n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    gruplar = [np.flatnonzero(veri.eczane_idx == e)
               for e in np.unique(veri.eczane_idx)]
    G = len(gruplar)
    ornek = np.empty(tekrar)
    for i in range(tekrar):
        secim = rng.integers(0, G, G)
        idx = np.concatenate([gruplar[j] for j in secim])
        ornek[i] = olcut(idx)
    gecerli = ornek[np.isfinite(ornek)]
    if gecerli.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(gecerli, 2.5)), float(np.percentile(gecerli, 97.5))


def degerlendir(cfg: Config, ad: str, veri: LoglanmisVeri, kol_hedef: np.ndarray,
                prop: PropensityCiktisi, q: np.ndarray) -> OPESonucu:
    """Bir hedef politikanin butun tahmincilerle offline degeri.

    `q` : [n, A] sonuc modeli tahmini (TL). M4'un T-ogrenicisinden kurulur:
          q(x, a) = p_hat(a | x) x marj(x, a) x miktar_carpani. Yeni bir model
          egitilmiyor -- DR'nin sonuc modeli POLITIKANIN KENDI modelidir ve
          M6'nin sordugu soru tam olarak "o modelin hatasi tahmini nasil
          bozuyor".

    Deger BIRIM SATIR BASINA (TL/satir) doner. Toplam degil: kayit tekrar
    sayisi degisince toplam degisir ama satir basina deger degismez, ve
    oracle ile karsilastirma ancak ayni normalizasyonda anlamlidir.
    """
    t = cfg.ope.tahminci
    d = cfg.ope.degerlendirme
    idx = np.arange(veri.n)
    ag = onem_agirligi(kol_hedef, veri, prop.propensity, t.kirpma_esigi)
    ag_dr = onem_agirligi(kol_hedef, veri, prop.propensity, t.dr_kirpma_esigi)
    q_hedef = q[idx, kol_hedef] if veri.n else np.zeros(0)
    q_secilen = q[idx, veri.kol] if veri.n else np.zeros(0)

    def _ips(i: np.ndarray) -> float:
        return ips(veri.odul[i], ag.kirpik[i])

    def _dr(i: np.ndarray) -> float:
        return doubly_robust(veri.odul[i], ag_dr.kirpik[i], q_secilen[i], q_hedef[i])

    ips_alt, ips_ust = _blok_bootstrap(veri, _ips, d.bootstrap_orneklem, d.bootstrap_seed)
    dr_alt, dr_ust = _blok_bootstrap(veri, _dr, d.bootstrap_orneklem, d.bootstrap_seed)

    return OPESonucu(
        ad=ad,
        ips=ips(veri.odul, ag.kirpik),
        ips_kirpmasiz=ips(veri.odul, ag.ham),
        snips=snips(veri.odul, ag.kirpik),
        dogrudan=dogrudan_yontem(q_hedef),
        dr=doubly_robust(veri.odul, ag_dr.kirpik, q_secilen, q_hedef),
        teshis=teshis(cfg, veri, kol_hedef, ag, prop.pi),
        ips_alt=ips_alt, ips_ust=ips_ust, dr_alt=dr_alt, dr_ust=dr_ust,
    )


def tekrar_sapmasi(cfg: Config, ad: str, veri: LoglanmisVeri,
                   kol_hedef: np.ndarray, prop: PropensityCiktisi,
                   q: np.ndarray) -> dict[str, float]:
    """Tahmincilerin BAGIMSIZ kayit kosulari arasindaki sapmasi.

    M6'nin sapma ayristirmasinda "varyans" kalemi budur ve bootstrap'tan
    FARKLIDIR: bootstrap ayni loglari yeniden ornekler, burasi dunyayi ayni
    politikayla YENIDEN loglar. Ikincisi tahmincinin gercek tekrar
    varyansidir; birincisi onun orneklem-ici vekilidir. Ikisinin farki
    raporda ayri bir satir (reports/m6.md).
    """
    tekrarlar = np.unique(veri.tekrar)
    if tekrarlar.size < 2:
        return {f"{t}_sapma": float("nan") for t in TAHMINCILER}
    toplayici: dict[str, list[float]] = {t: [] for t in TAHMINCILER}
    for r in tekrarlar:
        sec = veri.tekrar == r
        alt = LoglanmisVeri(
            X=veri.X[sec], kol=veri.kol[sec], propensity=veri.propensity[sec],
            pi_log=veri.pi_log[sec], odul=veri.odul[sec], izinli=veri.izinli[sec],
            eczane_idx=veri.eczane_idx[sec], origin=veri.origin[sec],
            tekrar=veri.tekrar[sec])
        alt_prop = PropensityCiktisi(
            propensity=prop.propensity[sec], pi=prop.pi[sec], kaynak=prop.kaynak,
            ortalama_mutlak_hata=prop.ortalama_mutlak_hata,
            kalibrasyon_hatasi=prop.kalibrasyon_hatasi,
            log_orani_ortalamasi=prop.log_orani_ortalamasi)
        sonuc = degerlendir(cfg, ad, alt, kol_hedef[sec], alt_prop, q[sec])
        for t in TAHMINCILER:
            toplayici[t].append(sonuc.deger(t))
    return {f"{t}_sapma": float(np.std(np.array(v)[np.isfinite(v)], ddof=1))
            if np.isfinite(v).sum() > 1 else float("nan")
            for t, v in toplayici.items()}
