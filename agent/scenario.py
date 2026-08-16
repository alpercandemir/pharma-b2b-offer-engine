"""Kur rejimi altinda KOSULLU okuma. D3 + D4.

    D3: "Kur tahmin hedefi degil, senaryo girdisidir."
    D4: "Asil makro sinyal piyasa kuru degil, referans kur guncelleme
         BEKLENTISIDIR."

BU DOSYADA KUR TAHMINI YOKTUR ve olmayacak. Girdi bir rejim tanimidir
(config/scenarios.yaml), cikti rejim basina bir tablodur. Hicbir fonksiyon
"hangi rejim gerceklesecek" sorusunu sormaz, hicbiri tek bir rejimi
digerlerinin onune koymaz. `senaryolari_kos` HER ZAMAN butun rejimleri
dondurur; tek rejim dondurmesi mumkun degil.

LLM DE YOKTUR (D8). Bu katman tamamen deterministiktir; ayni config + ayni
seed iki kez kosunca ayni tabloyu verir. LLM bu tabloyu yalnizca OKUR
(agent/narrative.py) ve okudugunu uydurmadigi harness/ tarafindan
deterministik olarak sinanir.

POLITIKA YENIDEN YAZILMADI. Aday havuzu, kisit vetosu, kol matrisleri,
CATE ve aksiyon secimi M3-M6'nin ta kendisidir; senaryo yalnizca UC GIRDIYI
degistirir:

  1. `politika.skor.yillik_fonlama_orani`  x fonlama_orani_carpani
     (var olan knob'in rejim altinda gecersiz kilinmasi; yeni aritmetik yok)
  2. `hiz_tahmini`                          x antisipasyon_talep_carpani
     (teklif adedi ve emilim tavani birlikte kayar)
  3. brut marja ERTELEME KAZANCI kalemi     (bu dosyanin tek yeni formulu)

Erteleme kazanci (kanal 1):

    pay          = kirp(1 - beklenti_hafta / senaryo.ikame_ufku_hafta, 0, 1)
    bekleyebilir = kalan_gun - beklenti_hafta*7 >= kisit.asgari_kalan_raf_omru_gun
    birim        = dsf * kur_artisi * fiyat_gecisi * realizasyon * pay * bekleyebilir
    marj[i, a]  -= (adet[i, a] + bedava[i, a]) * birim[i]

Okunusu: depodaki adedin maliyeti gecmiste sabitlendi; referans kur
guncellemesi DSF'i yukari tasiyinca AYNI adet daha yuksek marjla satilir.
Bugun satmanin firsat maliyeti bu farktir. `realizasyon` ("beklersem
zaten satilir mi") M5'in `tahsis.temizlik.normal_realizasyon_orani`
knob'inin ta kendisidir -- ayni inanc iki yerde iki farkli isimle
yasamasin diye yeni knob acilmadi.

IKI KNOB'SIZ ESIK, ikisi de var olan knob'lardan turuyor:

  - `pay`: guncelleme `ikame_ufku_hafta`nin otesindeyse bugunun karari
    etkilenmez ve kalem TAM SIFIRDIR (taban rejim boyle notrlesir).
  - `bekleyebilir`: beklemek, ayni mali guncellemeden SONRA teklif etmek
    demektir; o gun de politikanin kendi raf omru tabani gecerlidir. Yeni
    bir esik uydurmak yerine `politika.kisit.asgari_kalan_raf_omru_gun`
    kullanildi.

Ikinci carpan D9 ile kesisme noktasi ve bu katmanin en ogretici sonucu:
raf omru `beklenti + taban`i tasimayan lot bekleyemez, erteleme kazanci
SIFIRDIR ve o satir sert rejimde de teklif listesinde kalir. Kapinin
fiilen bagladigi satir orani her kosuda `bekleyemeyen_pay` olarak
raporlanir; sifirsa kanal dekoratiftir ve rapor bunu yazar.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import polars as pl

from core.config import Config, Rejim
from features import teklif as ft
from policy import bandit
from policy import candidates as pol_aday
from policy import scorer
from policy.constraints import VETO_SEBEPLERI, kisit_uygula
from policy.scorer import TEKLIF_YOK, TeklifMatrisleri
from sim.calendar import GUN_HAFTA

# Olgu paketinin BICIM sabitleri; knob degil. Bu sayilar modele verilen
# sayi defterinin cozunurlugudur (harness/denetim.py defteri bu degerler
# uzerinden kurar), o yuzden keyfi degil:
#   - TL buyuklukleri iki ondalik. `sim/world.py::TL_ONDALIK` ile ayni
#     bicim; para orada da iki basamak yazilir.
#   - ORANLAR dort ondalik. Daha kaba yuvarlanirsa iki rejimin ayni oran
#     alani (or. `depo_kar_marji`) yuvarlama sonrasi AYNI sayiya cokebilir
#     ve takas edilmis bir sayi mesru gorunerek `senaryo_karismasi`
#     denetiminden kacar. Cozunurluk burada bir denetim sartidir.
#   - Indeks ve gun buyuklukleri kendi dogal cozunurluklerinde.
TL_ONDALIK = 2
ORAN_ONDALIK = 4
INDEKS_ONDALIK = 3
GUN_ONDALIK = 1


# --------------------------------------------------------------------------
# rejim altinda config
# --------------------------------------------------------------------------
def rejim_config(cfg: Config, rejim: Rejim) -> Config:
    """Fonlama kanalinin rejim altindaki hali.

    `load_config(profil, gecersiz_kilma=...)` KULLANILMAZ: cagiran taraf
    (sweep) zaten knob gecersiz kilmis olabilir ve profili diskten yeniden
    okumak o gecersiz kilmalari sessizce silerdi. Bu yuzden dogrudan
    kopyalanir; degistirilen tek alan fonlama orani.
    """
    s = cfg.politika.skor
    yeni_skor = s.model_copy(
        update={"yillik_fonlama_orani": s.yillik_fonlama_orani
                * rejim.fonlama_orani_carpani})
    return cfg.model_copy(
        update={"politika": cfg.politika.model_copy(update={"skor": yeni_skor})})


def beklenti_payi(cfg: Config, rejim: Rejim) -> float:
    """Guncellemenin, stogun elde tutulabilecegi ufkun ICINDE olma payi.

    Ufkun disindaki bir guncelleme bugunun kararini etkilemez: kalem tam
    olarak sifirdir ve taban rejim bu yuzden marj aritmetigine hicbir sey
    eklemez (core/config.py `_m7_senaryo_kilidi` bunu ayrica siniyor).
    """
    return float(np.clip(
        1.0 - rejim.guncelleme_beklentisi_hafta / cfg.senaryo.ikame_ufku_hafta,
        0.0, 1.0))


def bekleyebilir(cfg: Config, rejim: Rejim, lot_kalan_gun: np.ndarray,
                 soguk_zincir: np.ndarray) -> np.ndarray:
    """[n] lot guncellemeyi bekleyip SONRA teklif edilebilir mi.

    Politikanin kendi raf omru tabani, bekleme suresi dusuldukten sonra da
    saglanmali. Soguk zincir carpani burada da uygulanir: o urunlerde
    tolerans penceresi dar (SPEC 2.5) ve bekleme lüksü daha da azdir.
    """
    k = cfg.politika.kisit
    taban = k.asgari_kalan_raf_omru_gun * np.where(
        soguk_zincir, k.soguk_zincir_raf_omru_carpani, 1.0)
    return lot_kalan_gun - rejim.guncelleme_beklentisi_hafta * GUN_HAFTA >= taban


def erteleme_kazanci(cfg: Config, rejim: Rejim, dsf: np.ndarray,
                     lot_kalan_gun: np.ndarray,
                     soguk_zincir: np.ndarray) -> np.ndarray:
    """[n] satir basina birim erteleme kazanci (TL/adet). Dosya basligindaki formul."""
    pay = beklenti_payi(cfg, rejim)
    kapi = bekleyebilir(cfg, rejim, lot_kalan_gun, soguk_zincir)
    return (dsf * rejim.referans_kur_artisi * rejim.fiyat_gecis_katsayisi
            * cfg.tahsis.temizlik.normal_realizasyon_orani * pay
            * kapi.astype(float))


def senaryo_talebi(havuz: pl.DataFrame, cfg: Config, rejim: Rejim) -> pl.DataFrame:
    """Antisipasyon kanali: hiz tahminini ve teklif adedini birlikte kaydirir.

    Aday SIRALAMASI degistirilmez -- carpan butun satirlara ayni uygulandigi
    icin sira zaten degismezdi, ama sinir bilerek burada: rejim "kime teklif
    edelim"i degil "ne kadar ve hangi sartla"yi degistiriyor. Basitlestirme
    ve bedeli reports/m7.md 7'de yazili.
    """
    if rejim.antisipasyon_talep_carpani == 1.0 or havuz.height == 0:
        return havuz
    hiz = havuz["hiz_tahmini"].to_numpy() * rejim.antisipasyon_talep_carpani
    return havuz.with_columns([
        pl.Series("hiz_tahmini", hiz),
        pl.Series("teklif_adedi", pol_aday.teklif_adedi(hiz, cfg)),
    ])


# --------------------------------------------------------------------------
# rejim kosusu
# --------------------------------------------------------------------------
@dataclass
class RejimSonucu:
    """Tek rejimin tek origin'deki tam ciktisi."""

    rejim: Rejim
    t: int
    cfg: Config                 # rejim altindaki config (fonlama kaydirilmis)
    gor: pol_aday.OriginGorunumu
    tumu: pl.DataFrame          # veto DAHIL butun havuz (vetonun bedeli gorunur)
    aday: pl.DataFrame          # veto sonrasi satirlar
    mat: TeklifMatrisleri       # erteleme kalemi UYGULANMIS marj
    ham_marj: np.ndarray        # [n, A] erteleme kalemi uygulanmadan once
    erteleme: np.ndarray        # [n] birim erteleme kazanci (TL/adet)
    bekleyebilir: np.ndarray    # [n] lot guncellemeyi bekleyip sonra teklif edilebilir mi
    p_x: np.ndarray             # [n, A] CATE kabul olasiligi
    kol: np.ndarray             # [n] secilen kol (kredi son kontrolu sonrasi)
    kredi_vetosu: np.ndarray    # [n] secim sonrasi vetolanan satirlar
    # M4'un beklenen miktar carpani. Butun rejimlerde AYNI (dunyanin
    # tepki parametrelerinden geliyor, rejimden degil); rejimler arasi
    # karsilastirmayi degistirmez ama M4/M5 tablolariyla ayni olcekte
    # kalmak icin tasinir.
    carpan: float

    @property
    def teklif_maskesi(self) -> np.ndarray:
        return self.kol != TEKLIF_YOK

    def satir_indeksi(self) -> np.ndarray:
        return np.arange(self.kol.size)

    def secilen(self, alan: np.ndarray) -> np.ndarray:
        """[n, A] matristen secilen kolun degerleri."""
        return alan[self.satir_indeksi(), self.kol]


def _rejim_kos(cfg: Config, m4, rejim: Rejim, t: int) -> RejimSonucu:
    """Bir rejim altinda tam politika hatti. M4/M6 ile AYNI fonksiyonlar."""
    dunya = m4.dunya
    s_cfg = rejim_config(cfg, rejim)

    gor = pol_aday.gorunum_kur(dunya, cfg, t)
    havuz, skorlar = pol_aday.aday_havuzu(dunya, cfg, gor)
    havuz = senaryo_talebi(havuz, cfg, rejim)
    tumu = kisit_uygula(dunya, s_cfg, gor, havuz)
    aday = tumu.filter(~pl.col("vetolu"))

    mat = scorer.teklif_matrisleri(dunya, s_cfg, aday)
    ham_marj = mat.marj.copy()
    dsf = dunya.dsf[aday["sku_idx"].to_numpy()] if aday.height else np.zeros(0)
    kalan_gun = aday["lot_kalan_gun"].to_numpy().astype(float)
    soguk = aday["soguk_zincir"].to_numpy().astype(bool)
    kapi = bekleyebilir(cfg, rejim, kalan_gun, soguk)
    erteleme = erteleme_kazanci(cfg, rejim, dsf, kalan_gun, soguk)
    mat = replace(mat, marj=ham_marj - (mat.adet + mat.bedava) * erteleme[:, None])

    X, _, _ = ft.ozellik_matrisi(m4.td, cfg, gor, aday, skorlar)
    pi = bandit.kayit_olasiliklari(dunya, cfg, mat.uzay, aday, mat.izinli)
    p_t = m4.t_ogr.olasilik(X)
    p_x = m4.x_ogr.olasilik(X, pi)

    secim = _secim_yap(cfg, mat, aday, p_t, p_x, m4.carpan)
    # D6: kisit katmani aksiyon seciminden SONRA da veto yetkisini korur.
    veto = scorer.kredi_son_kontrolu(dunya, s_cfg, gor, aday, mat, secim)
    kol = np.where(veto, TEKLIF_YOK, secim.kol).astype(np.int32)

    return RejimSonucu(rejim=rejim, t=t, cfg=s_cfg, gor=gor, tumu=tumu,
                       aday=aday, mat=mat,
                       ham_marj=ham_marj, erteleme=erteleme, bekleyebilir=kapi,
                       p_x=p_x, kol=kol, kredi_vetosu=veto, carpan=m4.carpan)


def _secim_yap(cfg: Config, mat: TeklifMatrisleri, aday: pl.DataFrame,
               p_t: np.ndarray, p_x: np.ndarray, carpan: float) -> scorer.Secim:
    """Teslim politikasinin aksiyon secimi.

    Import GEC yapiliyor cunku `experiments/run.py` M7 asamasi icin bu
    dosyayi import ediyor; modul duzeyinde karsilikli import dongu kurardi.
    Politikalarin ikinci bir kopyasini yazmak yerine geciktirilmis import
    tercih edildi: senaryo altinda olculen politika ile M4/M6'da olculen
    politika AYNI kod olmak zorunda, yoksa iki tablo karsilastirilamaz
    (M6'nin "rollout'ta hat kisaltilmadi" disiplini).
    """
    from experiments.run import _gozlemlenebilir_politikalar

    return _gozlemlenebilir_politikalar(
        cfg, mat, aday, p_t, p_x, carpan)[cfg.senaryo.politika]


# --------------------------------------------------------------------------
# ozet ve fark
# --------------------------------------------------------------------------
@dataclass
class RejimOzeti:
    """Bir rejimin tek satirlik okunusu. Butun sayilar gozlemlenebilir."""

    ad: str
    aday_satiri: int
    teklif_sayisi: int
    vetolu_satir: int
    beklenen_artimsal_marj: float   # sum p(a)*marj(a) - p(0)*marj(0)
    beklenen_brut_marj: float
    teklif_adedi: int
    bedava_adet: int
    ortalama_mf: float
    ortalama_vade: float
    mf_teklif_sayisi: int
    # Erteleme kapisinin fiilen bagladigi satir orani. Sifirsa kanal butun
    # satirlara AYNI uygulaniyor demektir; o zaman rejim yalnizca seviye
    # kaydiriyor, HEDEFLEME degistirmiyor ve rapor bunu boyle yazmali.
    bekleyemeyen_pay: float             # aday satirlari uzerinde
    bekleyemeyen_teklif_pay: float      # teklif VERILEN satirlar uzerinde
    ortalama_erteleme_tl: float
    # --- ISARET DONMESI TESHISI (reports/m7.md 6.1) ---
    # Erteleme kalemi buyudukce "teklif yok" kolunun senaryo marji negatife
    # doner. O rejimde amac fonksiyonu p*marj oldugu icin en iyi kol, marji
    # en az kotu olan degil KABUL OLASILIGI EN DUSUK olan kol haline
    # gelebilir: politika fiilen "satmamayi" optimize eder. Aksiyon uzayinda
    # (D1) boyle bir kol YOK -- bu bir modelleme artefaktidir ve olculmeden
    # birakilirsa "sok rejiminde artimsal marj yukseldi" diye YANLIS okunur.
    negatif_taban_marj_orani: float     # kol 0 senaryo marji < 0 olan satirlar
    talep_baskilayan_teklif_orani: float  # p(secilen) < p(teklif yok) olan teklifler
    veto_dagilimi: dict[str, int]


def rejim_ozeti(cfg: Config, s: RejimSonucu) -> RejimOzeti:
    idx, kol = s.satir_indeksi(), s.kol
    teklif = s.teklif_maskesi
    p, marj = s.p_x, s.mat.marj
    artimsal = float((p[idx, kol] * marj[idx, kol]
                      - p[:, TEKLIF_YOK] * marj[:, TEKLIF_YOK]).sum()
                     * s.carpan) if kol.size else 0.0
    mf = s.mat.uzay.mf[kol][teklif]
    vade = s.mat.uzay.vade[kol][teklif]
    bekleyemez = ~s.bekleyebilir
    return RejimOzeti(
        ad=s.rejim.ad,
        aday_satiri=int(s.aday.height),
        teklif_sayisi=int(teklif.sum()),
        vetolu_satir=int(s.tumu["vetolu"].sum()) if s.tumu.height else 0,
        beklenen_artimsal_marj=artimsal,
        beklenen_brut_marj=float((p[idx, kol] * marj[idx, kol]).sum()
                                 * s.carpan) if kol.size else 0.0,
        teklif_adedi=int(s.secilen(s.mat.adet)[teklif].sum()) if teklif.any() else 0,
        bedava_adet=int(s.secilen(s.mat.bedava)[teklif].sum()) if teklif.any() else 0,
        ortalama_mf=float(mf.mean()) if mf.size else 0.0,
        ortalama_vade=float(vade.mean()) if vade.size else 0.0,
        mf_teklif_sayisi=int((mf > 0).sum()) if mf.size else 0,
        bekleyemeyen_pay=float(bekleyemez.mean()) if bekleyemez.size else 0.0,
        bekleyemeyen_teklif_pay=(float(bekleyemez[teklif].mean())
                                 if teklif.any() else 0.0),
        ortalama_erteleme_tl=float(s.erteleme.mean()) if s.erteleme.size else 0.0,
        negatif_taban_marj_orani=(float((marj[:, TEKLIF_YOK] < 0).mean())
                                  if kol.size else 0.0),
        talep_baskilayan_teklif_orani=(
            float((p[idx, kol][teklif] < p[:, TEKLIF_YOK][teklif]).mean())
            if teklif.any() else 0.0),
        veto_dagilimi={ad: int((s.tumu["veto_sebebi"] == ad).sum())
                       for ad in VETO_SEBEPLERI} if s.tumu.height else {},
    )


@dataclass
class RejimFarki:
    """Taban rejime gore fark. D3'un ciktisi tam olarak budur."""

    ad: str
    kol_degisen_satir: int
    teklife_giren: int          # tabanda teklif yok, bu rejimde var
    teklifden_cikan: int        # tabanda teklif var, bu rejimde yok
    teklif_sayisi_farki: int
    artimsal_marj_farki: float
    ortalama_mf_farki: float
    ortalama_vade_farki: float
    bedava_adet_farki: int
    bekleyemeyen_teklif_pay_farki: float
    veto_farki: int


def rejim_farki(taban: RejimSonucu, taban_ozet: RejimOzeti,
                s: RejimSonucu, ozet: RejimOzeti) -> RejimFarki:
    """Iki rejimin SATIR BAZINDA karsilastirmasi.

    Satirlar (eczane_id, sku_id) uzerinden eslesir; kume degisebilir cunku
    antisipasyon kanali kisit vetolarini oynatir (emilim tavani, lot
    yeterliligi). Eslesmeyen satirlar "giren"/"cikan" olarak sayilir --
    silinmez. M3'un "vetolanan satir tabloda kalir" disiplininin senaryo
    tarafindaki karsiligi.
    """
    a = _kol_haritasi(taban)
    b = _kol_haritasi(s)
    ortak = set(a) & set(b)
    degisen = sum(1 for k in ortak if a[k] != b[k])
    giren = sum(1 for k in b if b[k] != TEKLIF_YOK and a.get(k, TEKLIF_YOK) == TEKLIF_YOK)
    cikan = sum(1 for k in a if a[k] != TEKLIF_YOK and b.get(k, TEKLIF_YOK) == TEKLIF_YOK)
    return RejimFarki(
        ad=s.rejim.ad,
        kol_degisen_satir=degisen,
        teklife_giren=giren,
        teklifden_cikan=cikan,
        teklif_sayisi_farki=ozet.teklif_sayisi - taban_ozet.teklif_sayisi,
        artimsal_marj_farki=ozet.beklenen_artimsal_marj - taban_ozet.beklenen_artimsal_marj,
        ortalama_mf_farki=ozet.ortalama_mf - taban_ozet.ortalama_mf,
        ortalama_vade_farki=ozet.ortalama_vade - taban_ozet.ortalama_vade,
        bedava_adet_farki=ozet.bedava_adet - taban_ozet.bedava_adet,
        bekleyemeyen_teklif_pay_farki=(ozet.bekleyemeyen_teklif_pay
                                       - taban_ozet.bekleyemeyen_teklif_pay),
        veto_farki=ozet.vetolu_satir - taban_ozet.vetolu_satir,
    )


def _kol_haritasi(s: RejimSonucu) -> dict[tuple[str, str], int]:
    if s.aday.height == 0:
        return {}
    return {(e, k): int(c) for e, k, c in
            zip(s.aday["eczane_id"], s.aday["sku_id"], s.kol)}


# --------------------------------------------------------------------------
# giris noktasi
# --------------------------------------------------------------------------
def eczane_master(cfg: Config, s: RejimSonucu, dunya) -> list[dict]:
    """Brifinge giren eczane profili. Yalnizca gozlemlenebilir kolonlar.

    Ajan katmani dunyayi kendisi okumaz (D8 mekanik siniri: agent/tools.py
    policy/sim import etmez); profil burada, karar katmaninda hazirlanir ve
    duz veri olarak devredilir.
    """
    tavan = cfg.politika.kisit.eczane_haftalik_teklif_tavani
    cikti = []
    for i, satir in enumerate(dunya.eczaneler.iter_rows(named=True)):
        cikti.append({
            "eczane_id": satir["eczane_id"], "il": satir["il"],
            "ilce": satir["ilce"],
            "hastane_yakinligi_km": round(float(satir["hastane_yakinligi_km"]), TL_ONDALIK),
            "semt_sosyoekonomik_index": round(float(satir["semt_sosyoekonomik_index"]), INDEKS_ONDALIK),
            "turizm_bolgesi": bool(satir["turizm_bolgesi"]),
            "aylik_ciro_bandi": satir["aylik_ciro_bandi"],
            "aylik_recete_adedi": int(satir["aylik_recete_adedi"]),
            "vade_riski_skoru": round(float(satir["vade_riski_skoru"]), ORAN_ONDALIK),
            "dbs_limiti_tl": round(float(satir["dbs_limiti"]), TL_ONDALIK),
            "acik_bakiye_tl": round(float(s.gor.acik_bakiye[i]), TL_ONDALIK),
            "sgk_recete_orani": round(float(satir["sgk_recete_orani"]), ORAN_ONDALIK),
            "haftalik_teklif_tavani": int(tavan),
        })
    return cikti


def urun_master(s: RejimSonucu, dunya) -> list[dict]:
    """Brifinge giren urun bilgisi. Regulasyon bayraklari D6'nin gerekcesi."""
    dsf = dunya.dsf
    cikti = []
    for i, satir in enumerate(dunya.urunler.iter_rows(named=True)):
        cikti.append({
            "sku_id": satir["sku_id"], "kategori_kod": satir["kategori_kod"],
            "atc_kodu": satir["atc_kodu"], "etken_madde": satir["etken_madde"],
            "urun_tipi": satir["urun_tipi"], "recete_rengi": satir["recete_rengi"],
            "sgk_geri_odeme": bool(satir["sgk_geri_odeme"]),
            "titck_tedarik_guclugu": bool(satir["titck_tedarik_guclugu"]),
            "promosyon_serbest": bool(satir["promosyon_serbest"]),
            "soguk_zincir": bool(satir["soguk_zincir"]),
            "koli_ici_adet": int(satir["koli_ici_adet"]),
            "dsf_tl": round(float(dsf[i]), TL_ONDALIK),
            "depo_kar_marji": round(float(satir["depo_kar_marji"]), ORAN_ONDALIK),
        })
    return cikti


def lot_master(s: RejimSonucu, dunya) -> list[dict]:
    """Origin'de depoda FIILEN duran lotlar (teklif edilmis olsun olmasin).

    Hallusinasyon denetcisinin "bu lot var mi" sorusu bu kumeye sorulur.
    Teklif edilen lotlarla sinirlansaydi, var olan ama teklif edilmemis bir
    lota atif "hallusinasyon" diye isaretlenirdi -- yanlis bir suclama;
    o durumun dogru adi kisit/oneri uyusmazligidir ve ayri denetlenir.
    """
    sku_id = dunya.urunler["sku_id"].to_list()
    cikti = []
    for sku_idx, lotlar in s.gor.lotlar.items():
        for lot in lotlar:
            cikti.append({
                "lot_id": lot.lot_id, "sku_id": sku_id[sku_idx],
                "kalan_adet": int(lot.kalan_adet),
                "kalan_gun": round(float(lot.kalan_gun), GUN_ONDALIK),
            })
    return sorted(cikti, key=lambda d: d["lot_id"])


@dataclass
class SenaryoKosusu:
    """Butun rejimlerin tek origin'deki ciktisi. TEK REJIM DONMEZ (D3)."""

    t: int
    taban_ad: str
    politika: str
    sonuclar: dict[str, RejimSonucu]
    ozetler: dict[str, RejimOzeti]
    farklar: dict[str, RejimFarki]      # taban haric
    eczane_master: list[dict]
    urun_master: list[dict]
    lot_master: list[dict]

    @property
    def rejim_adlari(self) -> list[str]:
        return list(self.sonuclar)

    def taban(self) -> RejimSonucu:
        return self.sonuclar[self.taban_ad]


def senaryolari_kos(cfg: Config, m4, t: int | None = None) -> SenaryoKosusu:
    """Butun rejimleri ayni origin uzerinde kosar ve tabana gore farki verir.

    `t` verilmezse M4'un SON olcum origin'i kullanilir: brifing "bu hafta ne
    onerelim" sorusuna cevap verdigi icin en guncel karar noktasi dogru
    varsayilan.
    """
    origin = m4.olcum_originleri[-1] if t is None else t
    if origin not in m4.olcum_originleri:
        raise ValueError(
            f"origin {origin} M4'un olcum origin'lerinde yok "
            f"({m4.olcum_originleri}); senaryo katmani M4'un hattini yeniden "
            f"kullanir, yeni bir origin uretmez")

    sonuclar, ozetler = {}, {}
    for rejim in cfg.senaryo.rejimler:
        s = _rejim_kos(cfg, m4, rejim, origin)
        sonuclar[rejim.ad] = s
        ozetler[rejim.ad] = rejim_ozeti(cfg, s)

    taban_ad = cfg.senaryo.taban_ad
    farklar = {ad: rejim_farki(sonuclar[taban_ad], ozetler[taban_ad],
                               sonuclar[ad], ozetler[ad])
               for ad in sonuclar if ad != taban_ad}
    return SenaryoKosusu(
        t=origin, taban_ad=taban_ad, politika=cfg.senaryo.politika,
        sonuclar=sonuclar, ozetler=ozetler, farklar=farklar,
        eczane_master=eczane_master(cfg, sonuclar[taban_ad], m4.dunya),
        urun_master=urun_master(sonuclar[taban_ad], m4.dunya),
        lot_master=lot_master(sonuclar[taban_ad], m4.dunya))


def ozet_tablosu(kosu: SenaryoKosusu) -> pl.DataFrame:
    """Rejim x metrik tablosu. Rapor ve dogrulama scripti bunu basar."""
    return pl.DataFrame([
        {"rejim": ad, "aday": o.aday_satiri, "teklif": o.teklif_sayisi,
         "vetolu": o.vetolu_satir, "artimsal_marj": o.beklenen_artimsal_marj,
         "adet": o.teklif_adedi, "bedava": o.bedava_adet,
         "ort_mf": o.ortalama_mf, "ort_vade": o.ortalama_vade,
         "mf_teklifi": o.mf_teklif_sayisi,
         "bekleyemeyen_pay": o.bekleyemeyen_pay,
         "erteleme_tl_adet": o.ortalama_erteleme_tl,
         "negatif_taban_marj": o.negatif_taban_marj_orani,
         "talep_baskilayan": o.talep_baskilayan_teklif_orani}
        for ad, o in kosu.ozetler.items()])


def fark_tablosu(kosu: SenaryoKosusu) -> pl.DataFrame:
    return pl.DataFrame([
        {"rejim": f.ad, "kol_degisen": f.kol_degisen_satir,
         "teklife_giren": f.teklife_giren, "teklifden_cikan": f.teklifden_cikan,
         "teklif_farki": f.teklif_sayisi_farki,
         "artimsal_marj_farki": f.artimsal_marj_farki,
         "mf_farki": f.ortalama_mf_farki, "vade_farki": f.ortalama_vade_farki,
         "bedava_farki": f.bedava_adet_farki,
         "bekleyemeyen_pay_farki": f.bekleyemeyen_teklif_pay_farki,
         "veto_farki": f.veto_farki}
        for f in kosu.farklar.values()])


def duz_metrikler(kosu: SenaryoKosusu) -> dict[str, float]:
    """Sweep tablosuna giren duz senaryo metrikleri."""
    duz: dict[str, float] = {}
    for ad, o in kosu.ozetler.items():
        duz.update({
            f"m7.senaryo.{ad}.teklif_sayisi": float(o.teklif_sayisi),
            f"m7.senaryo.{ad}.aday_satiri": float(o.aday_satiri),
            f"m7.senaryo.{ad}.vetolu_satir": float(o.vetolu_satir),
            f"m7.senaryo.{ad}.artimsal_marj": o.beklenen_artimsal_marj,
            f"m7.senaryo.{ad}.teklif_adedi": float(o.teklif_adedi),
            f"m7.senaryo.{ad}.bedava_adet": float(o.bedava_adet),
            f"m7.senaryo.{ad}.ortalama_mf": o.ortalama_mf,
            f"m7.senaryo.{ad}.ortalama_vade": o.ortalama_vade,
            f"m7.senaryo.{ad}.mf_teklif_sayisi": float(o.mf_teklif_sayisi),
            f"m7.senaryo.{ad}.bekleyemeyen_pay": o.bekleyemeyen_pay,
            f"m7.senaryo.{ad}.bekleyemeyen_teklif_pay": o.bekleyemeyen_teklif_pay,
            f"m7.senaryo.{ad}.erteleme_tl_adet": o.ortalama_erteleme_tl,
            f"m7.senaryo.{ad}.negatif_taban_marj_orani": o.negatif_taban_marj_orani,
            f"m7.senaryo.{ad}.talep_baskilayan_orani": o.talep_baskilayan_teklif_orani,
        })
    for ad, f in kosu.farklar.items():
        duz.update({
            f"m7.fark.{ad}.kol_degisen": float(f.kol_degisen_satir),
            f"m7.fark.{ad}.teklife_giren": float(f.teklife_giren),
            f"m7.fark.{ad}.teklifden_cikan": float(f.teklifden_cikan),
            f"m7.fark.{ad}.teklif_farki": float(f.teklif_sayisi_farki),
            f"m7.fark.{ad}.artimsal_marj_farki": f.artimsal_marj_farki,
            f"m7.fark.{ad}.mf_farki": f.ortalama_mf_farki,
            f"m7.fark.{ad}.vade_farki": f.ortalama_vade_farki,
            f"m7.fark.{ad}.bedava_farki": float(f.bedava_adet_farki),
            f"m7.fark.{ad}.bekleyemeyen_pay_farki": f.bekleyemeyen_teklif_pay_farki,
            f"m7.fark.{ad}.veto_farki": float(f.veto_farki),
        })
    return duz
