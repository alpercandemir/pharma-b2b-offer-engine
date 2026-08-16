"""Politika karsilastirma raporu: offline tahmin vs kapali dongu (SPEC 3).

M6'NIN CIKIS KRITERI TEK BIR SORUDUR:

    "Offline tahminci +%X dedi, closed-loop rollout %Y cikti. NEDEN?"

Bu dosya o "neden"i iki bagimsiz parcaya ayirir ve her parcayi kendi
icinde ayristirir.

    (A) TAHMINCI HATASI  = offline tahmin - GERCEK tek adimlik deger
        Ayni ani, ayni satirlar, ayni odul tanimi. Fark yalnizca
        tahmincinin kendisinden gelir. Uc kaleme ayrilir ve toplamlari
        farka TAM ESITTIR (scripts/verify_m6.py bu ozdesligi sinar):

            varyans     : sonlu orneklem + kabul zarlari.
                          V_ips(dogru pi, kirpmasiz) - V_oracle
            kirpma      : agirlik tavaninin sildigi kutle.
                          V_ips(dogru pi, kirpik) - V_ips(dogru pi, kirpmasiz)
            propensity  : kullanilan paydanin dogru paydadan farki.
                          V_ips(kullanilan pi, kirpik) - V_ips(dogru pi, kirpik)

        ORTUSME ve EKSTRAPOLASYON bu toplamin icinde birer KALEM DEGIL,
        birer SEBEPTIR ve ayri ayri raporlanir:
          - ortusme ihlali kirpma kaleminin nereden geldigini soyler
            (loglarin kor oldugu bolgede duran gercek deger),
          - ekstrapolasyon DR'ye ozgudur: sonuc modelinin destegi olmayan
            kollardaki hatasi, destegi olan kollardakiyle yan yana konur.

    (B) UFUK HATASI      = kapali dongu gerceklesen - offline tahmin
        Tahminci mukemmel olsaydi bile kalirdi. Offline odul TEK ADIMLIKTIR;
        kanibalizm, iade ve SOW erozyonu haftalar sonra gelir (sim/rollout.py).
        `ope.rollout.raporlanan_ufuklar` bu kalemi ufka gore tablolastirir ve
        SPEC 5'in "kisa ufukta agresif iskonto kazanir, uzun ufukta kaybeder"
        karsitligi tam olarak burada okunur.

OLCEK KOPRUSU. Offline tahminciler SATIR BASINA TL uretir, rollout ufuk
boyunca TOPLAM TL. Ikisini ayni birime tasimak icin offline tahmin karar
haftasi basina satir sayisi ve karar haftasi sayisiyla carpilir. Bu carpim
bir varsayimi ACIKCA yazar: "her haftanin karari bagimsizdir ve yalnizca ani
odulu vardir". Kapali dongunun ihlal ettigi varsayim tam olarak budur;
koprunun kendisi bu yuzden ogreticidir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from core.config import Config
from eval import ope as ev_ope
from eval.oracle import KarsiOlgusalOracle
from sim.rollout import RolloutOlcumu

EPSILON = 1e-12


# --------------------------------------------------------------------------
# (A) tahminci hatasi
# --------------------------------------------------------------------------
@dataclass
class TahminciDenetimi:
    """Bir politikanin bir tahmincisinin oracle'a gore konumu."""

    politika: str
    tahminci: str
    tahmin: float          # TL/satir
    oracle: float          # TL/satir
    sapma: float
    sapma_yuzde: float     # oracle'a gore
    alt: float
    ust: float
    araligi_kapsiyor: bool  # %95 blok bootstrap araligi oracle'i iceriyor mu


def _yuzde(pay: float, payda: float) -> float:
    return float(pay / abs(payda) * 100.0) if abs(payda) > EPSILON else float("nan")


def tahminci_denetimi(pol: str, sonuc: ev_ope.OPESonucu,
                      oracle_deger: float) -> list[TahminciDenetimi]:
    aralik = {"ips": (sonuc.ips_alt, sonuc.ips_ust),
              "dr": (sonuc.dr_alt, sonuc.dr_ust)}
    cikti = []
    for t in ev_ope.TAHMINCILER:
        alt, ust = aralik.get(t, (float("nan"), float("nan")))
        tahmin = sonuc.deger(t)
        cikti.append(TahminciDenetimi(
            politika=pol, tahminci=t, tahmin=tahmin, oracle=oracle_deger,
            sapma=tahmin - oracle_deger,
            sapma_yuzde=_yuzde(tahmin - oracle_deger, oracle_deger),
            alt=alt, ust=ust,
            araligi_kapsiyor=bool(np.isfinite(alt) and np.isfinite(ust)
                                  and alt <= oracle_deger <= ust),
        ))
    return cikti


@dataclass
class SapmaAyristirmasi:
    """IPS sapmasinin tam ayristirmasi. Kalemler farka ESIT toplanir."""

    politika: str
    oracle: float
    ips: float
    toplam_sapma: float
    varyans_kalemi: float
    kirpma_kalemi: float
    propensity_kalemi: float
    # --- sebepler (kalem degil, teshis) ---
    ortusme_ihlal_orani: float
    ortusme_kor_deger_payi: float      # hedef degerinin kor bolgedeki payi
    kirpma_ortusmeden_gelen_pay: float  # kirpma kaleminin kac katı ihlal satirindan
    ekstrapolasyon_dusuk_destek_hatasi: float   # q - oracle, zayif destekli kollarda
    ekstrapolasyon_yuksek_destek_hatasi: float  # ayni hata, destekli kollarda
    dusuk_destege_giden_satir_orani: float

    @property
    def artik(self) -> float:
        """Ozdeslik artigi. Sifir olmali; degilse ayristirma yalan soyluyor."""
        return self.toplam_sapma - (self.varyans_kalemi + self.kirpma_kalemi
                                    + self.propensity_kalemi)


def sapma_ayristir(cfg: Config, pol: str, veri: ev_ope.LoglanmisVeri,
                   kol_hedef: np.ndarray, prop: ev_ope.PropensityCiktisi,
                   q: np.ndarray, oracle: KarsiOlgusalOracle) -> SapmaAyristirmasi:
    """Ablasyon merdiveni: her basamakta BIR yaklasiklik geri alinir."""
    t = cfg.ope.tahminci
    o = cfg.ope.ortusme
    n = veri.n
    idx = np.arange(n)
    oracle_deger = oracle.deger(kol_hedef)

    # Merdivenin basamaklari.
    dogru_ham = ev_ope.onem_agirligi(kol_hedef, veri, veri.propensity, np.inf)
    dogru_kirpik = ev_ope.onem_agirligi(kol_hedef, veri, veri.propensity, t.kirpma_esigi)
    kullanilan = ev_ope.onem_agirligi(kol_hedef, veri, prop.propensity, t.kirpma_esigi)

    v_ham = ev_ope.ips(veri.odul, dogru_ham.kirpik)
    v_kirpik = ev_ope.ips(veri.odul, dogru_kirpik.kirpik)
    v_son = ev_ope.ips(veri.odul, kullanilan.kirpik)

    # --- sebep 1: ortusme. Hedef kolun LOGLANAN olasiligi esigin altinda mi.
    hedef_prop = veri.pi_log[idx, kol_hedef] if n else np.zeros(0)
    kor = hedef_prop < o.esik
    hedef_oracle = oracle.satir_degeri(kol_hedef)
    toplam_deger = float(np.abs(hedef_oracle).sum())
    kor_pay = (float(np.abs(hedef_oracle[kor]).sum() / toplam_deger)
               if toplam_deger > EPSILON else 0.0)
    # Kirpma kaleminin ne kadari kor bolgeden geliyor: silinen agirlik
    # kutlesinin kor satirlardaki payi.
    silinen = (dogru_ham.kirpik - dogru_kirpik.kirpik) * veri.odul
    silinen_toplam = float(np.abs(silinen).sum())
    kirpma_kor_payi = (float(np.abs(silinen[kor]).sum() / silinen_toplam)
                       if silinen_toplam > EPSILON else 0.0)

    # --- sebep 2: ekstrapolasyon. Sonuc modelinin destek disi kollardaki hatasi.
    sayim = np.bincount(veri.kol, minlength=veri.A)
    zayif_kol = (sayim < o.dusuk_destek_orneklemi) & veri.izinli.any(axis=0)
    zayif = zayif_kol[kol_hedef] if n else np.zeros(0, dtype=bool)
    q_hedef = q[idx, kol_hedef] if n else np.zeros(0)
    hata = q_hedef - hedef_oracle

    return SapmaAyristirmasi(
        politika=pol, oracle=oracle_deger, ips=v_son,
        toplam_sapma=v_son - oracle_deger,
        varyans_kalemi=v_ham - oracle_deger,
        kirpma_kalemi=v_kirpik - v_ham,
        propensity_kalemi=v_son - v_kirpik,
        ortusme_ihlal_orani=float(kor.mean()) if n else float("nan"),
        ortusme_kor_deger_payi=kor_pay,
        kirpma_ortusmeden_gelen_pay=kirpma_kor_payi,
        ekstrapolasyon_dusuk_destek_hatasi=(float(hata[zayif].mean())
                                            if zayif.any() else float("nan")),
        ekstrapolasyon_yuksek_destek_hatasi=(float(hata[~zayif].mean())
                                             if (~zayif).any() else float("nan")),
        dusuk_destege_giden_satir_orani=float(zayif.mean()) if n else float("nan"),
    )


# --------------------------------------------------------------------------
# (B) offline -> online koprusu
# --------------------------------------------------------------------------
@dataclass
class KopruOlcumu:
    """Bir politikanin bir ufuktaki offline/online karsilastirmasi."""

    politika: str
    tahminci: str
    ufuk: int
    offline_artimsal_tl: float    # tek adimlik tahminin ufka olceklenmisi
    online_artimsal_tl: float     # kapali dongude gerceklesen
    taban_net_marj_tl: float      # teklif_yok'un ayni ufuktaki net marji
    offline_yuzde: float          # +%X
    online_yuzde: float           # %Y
    ufuk_kalemi: float            # online - offline (TL)

    @property
    def isaret_dondu(self) -> bool:
        return (np.sign(self.offline_artimsal_tl)
                != np.sign(self.online_artimsal_tl))


def kopru(pol: str, tahminci: str, ufuk: int, offline_satir_basina: float,
          satir_sayisi: float, karar_haftasi: int,
          online_artimsal: float, taban_net_marj: float) -> KopruOlcumu:
    """Satir basina offline tahmini ufka olcekler ve online ile yan yana koyar.

    `karar_haftasi` = ufuk icinde teklif VERILEN hafta sayisi. Teklif
    penceresi ufuktan kisaysa offline tahmin de o kadar haftayla carpilir;
    aksi halde kopru verilmemis teklifleri de sayardi.
    """
    offline_tl = offline_satir_basina * satir_sayisi * karar_haftasi
    return KopruOlcumu(
        politika=pol, tahminci=tahminci, ufuk=ufuk,
        offline_artimsal_tl=offline_tl, online_artimsal_tl=online_artimsal,
        taban_net_marj_tl=taban_net_marj,
        offline_yuzde=_yuzde(offline_tl, taban_net_marj),
        online_yuzde=_yuzde(online_artimsal, taban_net_marj),
        ufuk_kalemi=online_artimsal - offline_tl,
    )


def terminal_duzeltme(olcum: RolloutOlcumu, taban: RolloutOlcumu,
                      ufuk: int) -> dict:
    """UFUK KESIMININ YANLILIGI: sonda eczane rafinda duran fazla mal.

    Net marj sevk aninda YAZILIR; iade ve imha ancak GERCEKLESTIKLERINDE
    dusulur. Ufuk kesildiginde teklif veren politikanin gonderdigi malin bir
    kismi hala eczane rafindadir: marji yazilmis, akibeti belli degildir. O
    stok ya tuketilir (marj hakedilmistir) ya iade olur (marj geri gider).

    Bu bir hata degil, OLCUNUN KENDISININ bir varsayimi -- ve teklif veren
    politikanin LEHINE calisir. Sessiz birakmak yerine ust siniriyla birlikte
    raporlaniyor:

        fazla_adet   = stok(politika, ufuk) - stok(taban, ufuk)
        riskli_marj  = fazla_adet x (politikanin sevk basina gerceklesen marji)

    `riskli_marj` bir DUZELTME DEGIL, UST SINIRDIR: fazla stogun TAMAMI iade
    olsaydi artimsal deger bu kadar duserdi. Gercek duzeltme ikisinin
    arasindadir ve ancak ufuk uzatilarak olculur (`ope.rollout.ufuk_hafta`).
    """
    if not olcum.haftalar or not taban.haftalar:
        return {"terminal_fazla_adet": float("nan"),
                "terminal_riskli_marj": float("nan"),
                "terminal_riskli_pay": float("nan")}
    kesim = min(ufuk, len(olcum.haftalar)) - 1
    fazla = float(olcum.seri("eczane_stogu")[kesim]
                  - taban.seri("eczane_stogu")[min(ufuk, len(taban.haftalar)) - 1])
    sevk = (olcum.ufukta("teklif_sevk_adet", ufuk)
            + olcum.ufukta("organik_sevk_adet", ufuk))
    marj = (olcum.ufukta("teklif_brut_marj", ufuk)
            + olcum.ufukta("organik_brut_marj", ufuk))
    birim = marj / sevk if sevk > EPSILON else 0.0
    riskli = fazla * birim
    artimsal = olcum.net_marj_ufukta(ufuk) - taban.net_marj_ufukta(ufuk)
    return {
        "terminal_fazla_adet": fazla,
        "terminal_riskli_marj": riskli,
        "terminal_riskli_pay": _yuzde(riskli, artimsal),
    }


def ufuk_tablosu(olcumler: dict[str, RolloutOlcumu], taban_ad: str,
                 ufuklar: list[int]) -> pl.DataFrame:
    """Politika x ufuk: birikimli net marj ve tabana gore artimsal.

    Tek bir rollout butun onekleri verir; ayri kosu gerekmez. Isaretin ufka
    gore dondugu yer bu tabloda gorunur.
    """
    taban = olcumler[taban_ad]
    satirlar = []
    for ad, o in olcumler.items():
        for h in ufuklar:
            t = taban.net_marj_ufukta(h)
            v = o.net_marj_ufukta(h)
            satirlar.append({
                "politika": ad, "ufuk": h, "net_marj": v,
                "artimsal": v - t, "artimsal_yuzde": _yuzde(v - t, t),
                "teklif": o.ufukta("teklif_sayisi", h),
                "kabul": o.ufukta("kabul_sayisi", h),
                "bedava_adet": o.ufukta("bedava_adet", h),
                "iade_adet": o.ufukta("iade_adet", h),
                "imha_adet": o.ufukta("imha_adet", h),
                "karsilanmayan": o.ufukta("karsilanmayan_siparis_adet", h),
                "sow_son": (o.seri("sow_ortalama")[:h][-1]
                            if o.haftalar else float("nan")),
            })
    return pl.DataFrame(satirlar)


def gecikmeli_bedel(olcum: RolloutOlcumu, taban: RolloutOlcumu) -> dict:
    """Kanibalizm / iade / SOW kanallarinin tabana gore buyuklugu.

    Uc kanal da `sim/rollout.py`nin dokumanindaki mekanizmalardir; burada
    OLCULUR. Hicbiri offline tahmincinin gorebilecegi bir sey degildir, ve
    ufuk kaleminin nereden geldigini bu uclu soyler.
    """
    def _fark(alan: str) -> float:
        return float(olcum.seri(alan).sum() - taban.seri(alan).sum())

    sow_ = olcum.seri("sow_ortalama")
    sow_t = taban.seri("sow_ortalama")
    return {
        # Kanibalizm: teklif hacmi organik siparisi ne kadar yedi.
        "kanibalizm_organik_siparis_farki": _fark("organik_siparis_adet"),
        "kanibalizm_organik_marj_farki": _fark("organik_brut_marj"),
        "teklif_sevk_adet": float(olcum.seri("teklif_sevk_adet").sum()),
        "bedava_adet": float(olcum.seri("bedava_adet").sum()),
        # Iade: eczaneye emebileceginden fazlasi gitti mi.
        "iade_adet_farki": _fark("iade_adet"),
        "iade_maliyet_farki": _fark("iade_islem_maliyeti") + _fark("iade_marj_geri_alma"),
        "imha_adet_farki": _fark("imha_adet"),
        # SOW: kalici iliski kaybi (son hafta seviyesi).
        "sow_son_fark": float(sow_[-1] - sow_t[-1]) if sow_.size else float("nan"),
        "sow_ortalama_fark": float(sow_.mean() - sow_t.mean()) if sow_.size else float("nan"),
        "eczane_stogu_son_fark": float(olcum.seri("eczane_stogu")[-1]
                                       - taban.seri("eczane_stogu")[-1])
        if olcum.haftalar else float("nan"),
    }


# --------------------------------------------------------------------------
# tablolar (CLI)
# --------------------------------------------------------------------------
def denetim_tablosu(denetimler: list[TahminciDenetimi]) -> pl.DataFrame:
    return pl.DataFrame([{
        "politika": d.politika, "tahminci": d.tahminci,
        "tahmin_TL/satir": d.tahmin, "oracle_TL/satir": d.oracle,
        "sapma": d.sapma, "sapma_%": d.sapma_yuzde,
        "aralik_kapsiyor": d.araligi_kapsiyor,
    } for d in denetimler])


def ayristirma_tablosu(ayr: list[SapmaAyristirmasi]) -> pl.DataFrame:
    return pl.DataFrame([{
        "politika": a.politika, "oracle": a.oracle, "IPS": a.ips,
        "toplam_sapma": a.toplam_sapma, "varyans": a.varyans_kalemi,
        "kirpma": a.kirpma_kalemi, "propensity": a.propensity_kalemi,
        "artik": a.artik, "ortusme_ihlal_%": a.ortusme_ihlal_orani * 100.0,
        "kor_deger_%": a.ortusme_kor_deger_payi * 100.0,
        "ekstrap_zayif": a.ekstrapolasyon_dusuk_destek_hatasi,
        "ekstrap_guclu": a.ekstrapolasyon_yuksek_destek_hatasi,
    } for a in ayr])


def kopru_tablosu(kopruler: list[KopruOlcumu]) -> pl.DataFrame:
    return pl.DataFrame([{
        "politika": k.politika, "tahminci": k.tahminci, "ufuk": k.ufuk,
        "offline_TL": k.offline_artimsal_tl, "online_TL": k.online_artimsal_tl,
        "offline_%": k.offline_yuzde, "online_%": k.online_yuzde,
        "ufuk_kalemi_TL": k.ufuk_kalemi, "isaret_dondu": k.isaret_dondu,
    } for k in kopruler])
