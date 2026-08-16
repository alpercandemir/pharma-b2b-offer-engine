"""Teklife tepki fonksiyonu: uplift'in GROUND TRUTH'u (SPEC 3 `sim/response.py`).

Bu dosya dunyanin parcasidir. Modeller ve politika buraya ASLA bakamaz;
`scripts/verify_m4.py` policy/, features/ ve models/ altinda bu modulun
adinin gecmedigini statik olarak dogrular.

MODEL
=====
Bir (eczane, SKU, origin) satirinin `a` kolu altinda siparis verme olasiligi:

    p(a) = sigmoid( taban_logit + teklif_logit(a) )        a >= 1
    p(0) = sigmoid( taban_logit )                          teklif yok

`taban_logit` "zaten alacak miydi" sorusudur -- PROPENSITY'nin kendisi.
`teklif_logit(a)` teklifin katkisidir -- UPLIFT'in kaynagi. Ikisi ayri
mekanizma ve ayri surucu kumesi tarafindan belirlenir; M4'un olctugu marj
farki bu ayrimin sayisal karsiligidir.

HETEROJENLIK (CLAUDE.md 7)
==========================
Teklif etkisi eczaneye gore degisir ve suruculerin bir kismi LATENT'tir:

    MF duyarliligi   <- sosyoekonomik index (gozlemlenebilir),
                        eczane olcegi (gozlemlenebilir),
                        share_of_wallet (LATENT),
                        log-normal kisisel gurultu (LATENT)
    Vade duyarliligi <- vade_riski_skoru (gozlemlenebilir),
                        dbs_limiti (gozlemlenebilir),
                        stokculuk_egilimi (LATENT),
                        log-normal kisisel gurultu (LATENT)

Gozlemlenebilir surucu olmasaydi CATE ogrenilemezdi ve karsilastirma
anlamsiz olurdu; latent surucu olmasaydi model tavana carpar ve "uplift
modeli mukemmel calisiyor" gibi yanlis bir sonuc cikardi.

SATURASYON. Logit uzayindaki SABIT bir teklif etkisi bile olasilik uzayinda
heterojen uplift uretir: taban olasiligi yuksek satirlarda sigmoid'in tureviu
kucuktur. Yani "kesin alicinin uplift'i dusuktur" bu modelde bir kural degil,
mekanizmanin sonucudur. `duyarlilik.heterojenlik_carpani = 0` yapildiginda
tasarlanmis heterojenlik kapanir ama saturasyon kaynakli heterojenlik KALIR;
olculen marj farkinin tabani budur (reports/m4.md 7.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
import polars as pl

from core.config import Config
from core.rng import SeedBank

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# Sifira bolme korumasi (hiz paydasi). Sayisal sabit, knob degil.
EPSILON = 1e-9
# Kontrol kolunun indeksi. policy.scorer.TEKLIF_YOK ile ayni olmak zorunda;
# sim/ politikadan import etmez (katman yonu), bu yuzden burada tekrarlanir
# ve tests/test_uplift.py ikisinin esitligini sinar.
TEKLIF_YOK_KOLU = 0
# Standartlastirmada sifir sapma korumasi.
MIN_SAPMA = 1e-6


def _z(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / max(float(v.std()), MIN_SAPMA)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sayisal olarak kararli sigmoid. Buyuk negatif logit'te exp tasar."""
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


def _normal_kumulatif(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def beklenen_miktar_carpani(cfg: Config) -> float:
    """E[max(c, X)], X ~ LogN(-s^2/2, s). Kapali form; Monte Carlo degil.

    Kabul edilen adet teklif edilen adede birebir esit degil (eczane kendi
    ihtiyacina gore kirpar ya da buyutur). Carpanin beklenen degeri BIRDEN
    BUYUKTUR cunku asagidan kirpiliyor; beklenen marj hesabi bu sapmayi
    yok saymaz - saysaydi olculen politika degeri sistematik olarak eksik
    cikardi ve fark tablolarina yanlilik olarak sizardi.
    """
    s = cfg.tepki.miktar.kabul_gurultu_sigma
    c = cfg.tepki.miktar.asgari_kabul_orani
    if s <= 0.0:
        return max(1.0, c)
    mu = -0.5 * s * s
    d = (log(max(c, EPSILON)) - mu) / s
    return c * _normal_kumulatif(d) + exp(mu + 0.5 * s * s) * _normal_kumulatif(s - d)


# --------------------------------------------------------------------------
# ground truth okuyucu
# --------------------------------------------------------------------------
class GercekDurum:
    """`ground_truth/` katmanindan origin anindaki gercek durum.

    eval/oracle.py ile ayni tabloyu okur ama farkli soru sorar: oracle
    "ne zaman tukendi" der, burasi "su anda ne kadar ihtiyaci var" der.
    Matrisler ilk kullanimda bir kez kurulur.
    """

    def __init__(self, kosu_adi: str, kok: Path | None = None) -> None:
        dizin = (kok or DATA_DIR) / kosu_adi / "ground_truth"
        h = pl.read_parquet(
            dizin / "hucre_haftalik.parquet",
            columns=["hafta", "eczane_id", "sku_id", "gercek_eczane_stogu",
                     "cesitte_var", "latent_tuketim_hizi"])
        W = int(h["hafta"].max()) + 1
        anahtar = np.char.add(np.char.add(h["eczane_id"].to_numpy().astype(str), "|"),
                              h["sku_id"].to_numpy().astype(str))
        benzersiz, satir = np.unique(anahtar, return_inverse=True)
        hafta = h["hafta"].to_numpy().astype(int)
        n = benzersiz.size
        self._sira = {a: i for i, a in enumerate(benzersiz)}
        self._stok = np.zeros((n, W), dtype=np.int32)
        self._cesit = np.zeros((n, W), dtype=bool)
        self._hiz = np.zeros(n)
        self._stok[satir, hafta] = h["gercek_eczane_stogu"].to_numpy()
        self._cesit[satir, hafta] = h["cesitte_var"].to_numpy()
        self._hiz[satir] = h["latent_tuketim_hizi"].to_numpy()

        sow = pl.read_parquet(dizin / "sow_haftalik.parquet")
        ecz = sow["eczane_id"].unique().sort().to_list()
        self._sow_sira = {e: i for i, e in enumerate(ecz)}
        self._sow = np.zeros((len(ecz), W))
        self._sow[[self._sow_sira[e] for e in sow["eczane_id"]],
                  sow["hafta"].to_numpy().astype(int)] = sow["share_of_wallet"].to_numpy()

        self.latent_eczane = pl.read_parquet(dizin / "latent_eczane.parquet")
        self.W = W

    def hucre_durumu(self, eczane_id: np.ndarray, sku_id: np.ndarray,
                     t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(stok, latent hiz, cesitte_var) - hic cesitte olmamis hucre = (0, 0, False)."""
        idx = np.array([self._sira.get(f"{e}|{s}", -1)
                        for e, s in zip(eczane_id, sku_id)])
        var = idx >= 0
        guvenli = np.where(var, idx, 0)
        return (np.where(var, self._stok[guvenli, t], 0).astype(float),
                np.where(var, self._hiz[guvenli], 0.0),
                var & self._cesit[guvenli, t])

    def sow(self, eczane_id: np.ndarray, t: int) -> np.ndarray:
        return self._sow[[self._sow_sira[e] for e in eczane_id], t]


# --------------------------------------------------------------------------
# latent tepki parametreleri
# --------------------------------------------------------------------------
@dataclass
class TepkiEvreni:
    """Eczane ve hucre duzeyinde KALICI latent tepki parametreleri.

    Origin'den bagimsiz: ayni eczane her origin'de ayni duyarliliga sahip.
    Origin'ler arasi degisseydi hicbir model ogrenemezdi ve "CATE ogrenilemez"
    sonucu modelin degil dunyanin ozelligi olurdu.
    """

    mf_duyarliligi: np.ndarray      # [P]
    vade_duyarliligi: np.ndarray    # [P]
    hucre_gurultusu: np.ndarray     # [P, S]
    miad_toleransi: np.ndarray      # [P] gun
    kapsama_hedefi: np.ndarray      # [P] hafta
    urun_mf_carpani: np.ndarray     # [S]
    dsf_z: np.ndarray               # [S]


def tepki_evreni_kur(cfg: Config, seedler: SeedBank, eczaneler: pl.DataFrame,
                     urunler: pl.DataFrame, latent_eczane: pl.DataFrame) -> TepkiEvreni:
    """Latent duyarliliklari kurar.

    Kendi seed asamasini kullanir (`teklif_tepkisi`): M1 dunyasinin cekilis
    akisina DOKUNMAZ, bu yuzden `dunya_hash` degismez ve M1/M2/M3 sonuclari
    gecerli kalir (core/rng.py'deki asama ayrimi tam bunun icin var).
    """
    d = cfg.tepki.duyarlilik
    h = d.heterojenlik_carpani
    rng = seedler.generator("teklif_tepkisi")
    P, S = eczaneler.height, urunler.height

    # Latent tablo eczane_id'ye gore hizalanir; sira varsayimi yapilmaz.
    lat = eczaneler.select("eczane_id").join(latent_eczane, on="eczane_id", how="left")
    sow0 = lat["share_of_wallet"].to_numpy().astype(float)
    stokculuk = lat["stokculuk_egilimi"].to_numpy().astype(float)
    miad_tol = lat["miad_toleransi_gun"].to_numpy().astype(float)
    kapsama = lat["kapsama_hafta"].to_numpy().astype(float)

    sosyo = eczaneler["semt_sosyoekonomik_index"].to_numpy().astype(float)
    olcek = eczaneler["aylik_recete_adedi"].to_numpy().astype(float)
    risk = eczaneler["vade_riski_skoru"].to_numpy().astype(float)
    dbs = eczaneler["dbs_limiti"].to_numpy().astype(float)

    mf_us = h * (d.mf_sosyoekonomik * _z(sosyo)
                 + d.mf_olcek * _z(np.log(olcek))
                 + d.mf_sow * _z(sow0)
                 + d.mf_log_sigma * rng.normal(0.0, 1.0, P))
    vade_us = h * (d.vade_riski * _z(risk)
                   + d.vade_dbs * _z(np.log(dbs))
                   + d.vade_stokculuk * _z(np.log(stokculuk))
                   + d.vade_log_sigma * rng.normal(0.0, 1.0, P))

    tip = urunler["urun_tipi"].to_numpy()
    carpan = np.array([cfg.tepki.urun_tipi_mf_carpani[t] for t in tip])

    return TepkiEvreni(
        mf_duyarliligi=np.exp(mf_us),
        vade_duyarliligi=np.exp(vade_us),
        hucre_gurultusu=rng.normal(0.0, cfg.tepki.taban.hucre_gurultu_sigma, (P, S)),
        miad_toleransi=miad_tol, kapsama_hedefi=kapsama,
        urun_mf_carpani=carpan,
        dsf_z=_z(np.log(urunler["dsf"].to_numpy().astype(float))),
    )


# --------------------------------------------------------------------------
# tepki
# --------------------------------------------------------------------------
@dataclass
class Tepki:
    """Bir origin'in aday satirlari icin GERCEK kabul olasiliklari."""

    olasilik: np.ndarray      # [n, A]
    taban_logit: np.ndarray   # [n]
    ihtiyac: np.ndarray       # [n] gercek ihtiyac (0-1)

    @property
    def taban_olasilik(self) -> np.ndarray:
        return self.olasilik[:, 0]

    @property
    def uplift(self) -> np.ndarray:
        """[n, A] kabul olasiligindaki artis (kol 0 = 0)."""
        return self.olasilik - self.olasilik[:, [0]]


def tepki_hesapla(cfg: Config, evren: TepkiEvreni, durum: GercekDurum,
                  uzay, teklifler: pl.DataFrame, t: int,
                  adet: np.ndarray | None = None) -> Tepki:
    """Aday satirlari x kollar icin gercek kabul olasiligi matrisi.

    `uzay` : policy.scorer.AksiyonUzayi. Aksiyon uzayinin TANIMI politikanin
             kararidir; dunya yalnizca verilen aksiyona nasil tepki verecegini
             bilir.
    `adet` : [n, A] kol basina sevk edilecek adet. Verilmezse asiri adet
             direnci hesaplanmaz (yalnizca testlerde).
    """
    tb, tk = cfg.tepki.taban, cfg.tepki.teklif
    n, A = teklifler.height, uzay.A
    if n == 0:
        return Tepki(np.zeros((0, A)), np.zeros(0), np.zeros(0))

    p_idx = teklifler["eczane_idx"].to_numpy()
    s_idx = teklifler["sku_idx"].to_numpy()
    ecz_id = teklifler["eczane_id"].to_numpy()
    sku_id = teklifler["sku_id"].to_numpy()

    stok, hiz, cesitte = durum.hucre_durumu(ecz_id, sku_id, t)
    kapsama = stok / np.maximum(hiz, EPSILON)
    ihtiyac = np.where(cesitte, np.exp(-kapsama / tb.ihtiyac_referans_hafta),
                       tb.cesit_disi_ihtiyac)
    sow = durum.sow(ecz_id, t)

    taban = (tb.kesme
             + tb.ihtiyac_katsayisi * ihtiyac
             + tb.sow_katsayisi * (sow - 0.5) * 2.0
             + tb.yeni_hucre_cezasi * teklifler["yeni_hucre"].to_numpy().astype(float)
             + tb.fiyat_katsayisi * evren.dsf_z[s_idx]
             + evren.hucre_gurultusu[p_idx, s_idx])

    # Alici tarafi miad direnci (SPEC 2.5): teklif edilen lotun kalan raf
    # omru eczacinin toleransinin altina dustukce kabul duser. Yalnizca
    # TEKLIF kollarinda: organik siparis lotu biz secmiyoruz.
    lot_gun = teklifler["lot_kalan_gun"].to_numpy().astype(float)
    gerekli = np.maximum(evren.miad_toleransi[p_idx], EPSILON)
    eksik = np.clip(1.0 - lot_gun / gerekli, 0.0, 1.0)
    miad_direnci = cfg.tepki.miad.direnc_katsayisi * eksik

    mf_gucu = evren.mf_duyarliligi[p_idx] * evren.urun_mf_carpani[s_idx]
    vade_gucu = evren.vade_duyarliligi[p_idx]
    taban_vade = float(cfg.politika.aksiyon.taban_vade_gun)
    # Acil ihtiyac teklif esnekligini SONDURUR: stogu bitmek uzere olan
    # eczane zaten siparis verecektir (D2). "Kesin alici" ile "ikna
    # edilebilir" ayrimini ureten mekanizma budur.
    ihtiyac_carpani = np.exp(tk.ihtiyac_etkilesimi * ihtiyac)

    # Asiri adet direnci: teklif eczanenin kapsama hedefinin kac katini
    # kapatiyor. Bu terim olmasa "adedi buyut" sinirsiz bir marj kaldiraci
    # olurdu (config/response.yaml).
    mk = cfg.tepki.miktar
    esik_adet = np.maximum(hiz * evren.kapsama_hedefi[p_idx] * mk.kapsama_toleransi,
                           mk.asgari_esik_adet)

    def _asiri(a: int) -> np.ndarray | float:
        if adet is None:
            return 0.0
        return mk.asiri_adet_direnci * np.maximum(
            adet[:, a] / np.maximum(esik_adet, EPSILON) - 1.0, 0.0)

    logit = np.repeat(taban[:, None], A, axis=1)
    # Asiri adet direnci KONTROL KOLUNA DA uygulanir. Karsi-olgusal soru
    # "teklif olmasa AYNI SEPETI alir miydi": sepet eczanenin emebileceginden
    # buyukse cevap teklifsiz de hayirdir. Yalnizca teklif kollarina
    # uygulansaydi kontrol yapay olarak cazip gorunur ve butun politikalar
    # sistematik olarak az teklif verirdi.
    logit[:, TEKLIF_YOK_KOLU] = taban + _asiri(TEKLIF_YOK_KOLU)
    for a in range(1, A):
        mf_etki = (tk.mf_taban_etkisi
                   * (uzay.mf[a] / tk.mf_referans_orani) ** tk.mf_azalan_us
                   * mf_gucu) if uzay.mf[a] > 0 else 0.0
        vade_etki = (tk.vade_taban_etkisi
                     * ((uzay.vade[a] / taban_vade) ** tk.vade_azalan_us - 1.0)
                     * vade_gucu)
        logit[:, a] = (taban
                       + (tk.taban_etki + mf_etki + vade_etki) * ihtiyac_carpani
                       + miad_direnci + _asiri(a))

    return Tepki(olasilik=_sigmoid(logit), taban_logit=taban, ihtiyac=ihtiyac)


def sonuc_ornekle(cfg: Config, seedler: SeedBank, tepki: Tepki,
                  kol: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Gerceklesen (kabul, miktar_carpani). Origin bazli seed'li.

    Miktar carpani yalnizca KABUL edilen satirlarda anlamli; reddedilende 0.
    """
    n = kol.size
    rng = seedler.generator(f"tepki_ornekleme_{t}")
    if n == 0:
        return np.zeros(0, dtype=np.int8), np.zeros(0)
    p = tepki.olasilik[np.arange(n), kol]
    kabul = (rng.random(n) < p).astype(np.int8)
    s = cfg.tepki.miktar.kabul_gurultu_sigma
    carpan = np.maximum(cfg.tepki.miktar.asgari_kabul_orani,
                        np.exp(rng.normal(-0.5 * s * s, s, n))) if s > 0 else np.ones(n)
    return kabul, np.where(kabul > 0, carpan, 0.0)
