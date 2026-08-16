"""Kapali dongu: politika aksiyon alir, DUNYA tepki verir (SPEC 3, M6).

OFFLINE DEGERLENDIRME ILE FARKI TEK CUMLEDE. `eval/ope.py` bir kararin ANI
odulunu tahmin eder; burasi ayni karari dunyaya uygular ve sonrasini haftalar
boyunca takip eder. M6'nin cikis kriteri ("offline +%12 dedi, gercek -%3
cikti, neden?") bu iki sayinin farkindan dogar.

DUNYA YENIDEN YAZILMADI. `sim/world.py::hafta_adimi` ne ise o kosuluyor;
rollout yalnizca haftanin basina kabul edilmis teklifin sevkiyatini enjekte
ediyor. Bu onemli: dinamigin ikinci bir kopyasi olsaydi "politika dunyaya
tepki veriyor" iddiasi dogru olmazdi -- politika dunyanin kopyasina tepki
verirdi ve sapmanin kaynagi olculemezdi.

GECIKMELI BEDELIN UC KANALI (hicbiri M6 icin yazilmadi, hepsi M1'de vardi):

  1. KANIBALIZM. Kabul edilen teklif eczanenin stok pozisyonuna girer;
     eczane kendi (s, S) politikasiyla sonraki haftalarda daha az siparis
     verir. Bugun satilan adet gelecekten calinmistir.
  2. IADE. Teklif sevkiyati eczanenin miad kovalarina duser. Emebileceginden
     fazlasi orada yaslanir, satilamaz ve iade olur (SPEC 2.5).
  3. SOW EROZYONU. Iade ve karsilanamayan siparis latent share_of_wallet'i
     dusurur; dusen SOW sonraki haftalarda siparisin RAKIBE gitme olasiligini
     artirir. Kalici bir kayiptir ve tek adimlik hicbir tahminci goremez.

Vade maliyeti ise tam tersine ANINDA gorunur (marj aritmetiginde), gecikmeli
degil. Agresif teklifin kisa ufukta kazanip uzun ufukta kaybetmesi bu iki
zaman olceginin yarismasidir.

KATMAN NOTU. Bu dosya `policy/`den import eder ve mimaride ters yonde duran
tek yer burasidir. Sebep tasarimin kendisi: kapali dongu tanimi geregi iki
katmani birbirine baglar ve SPEC 3 rollout'u `sim/` altina koyuyor. Cevrim
yok (policy -> sim.calendar var, sim.rollout -> policy var, ikisi ayri
modul). KARAR MANTIGI burada DEGIL: hangi satira hangi kolun verilecegi bir
`karar_verici` geri cagirmasindan gelir (experiments/run.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import polars as pl

from core.config import Config
from core.rng import SeedBankasi
from policy.candidates import AdayDunyasi
from policy.scorer import TEKLIF_YOK, TeklifMatrisleri, brut_marj
from sim.calendar import GUN_HAFTA
from sim.response import Tepki, TepkiEvreni, sonuc_ornekle, tepki_hesapla
from sim.world import DunyaDurumu, hafta_adimi

# Sifira bolme korumasi. Sayisal sabit, knob degil.
EPSILON = 1e-12


# --------------------------------------------------------------------------
# canli durum: tepki fonksiyonunun ground_truth yerine bakacagi yer
# --------------------------------------------------------------------------
class CanliDurum:
    """`sim.response.GercekDurum` ile ayni arayuz, ama CANLI simulasyondan.

    `GercekDurum` diske yazilmis ground_truth'u okur ve dolayisiyla TABAN
    dunyayi bilir. Rollout'ta dunya teklifler yuzunden taban dunyadan ayrilir;
    tepki fonksiyonunun ayrilmis duruma bakmasi sart. Aksi halde eczane
    deposunda olmayan bir stoga gore karar verir ve kanibalizm hic gorunmezdi.
    """

    def __init__(self, durum: DunyaDurumu, ecz_id: np.ndarray,
                 sku_id: np.ndarray, latent_eczane: pl.DataFrame) -> None:
        self.durum = durum
        self.latent_eczane = latent_eczane
        self.W = durum.W
        self._ecz_sira = {e: i for i, e in enumerate(ecz_id)}
        self._sku_sira = {s: i for i, s in enumerate(sku_id)}

    def _idx(self, ecz_id: np.ndarray, sku_id: np.ndarray) -> tuple:
        return (np.array([self._ecz_sira[e] for e in ecz_id], dtype=int),
                np.array([self._sku_sira[s] for s in sku_id], dtype=int))

    def hucre_durumu(self, eczane_id: np.ndarray, sku_id: np.ndarray,
                     t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(stok, latent hiz, cesitte_var). `t` yalnizca tutarlilik kontrolu."""
        if t != self.durum.w - 1:
            raise ValueError(
                f"canli durum {self.durum.w - 1} haftasinin sonunda, tepki {t} "
                f"icin soruldu: karar ani ile dunya ani uyusmuyor")
        p, s = self._idx(eczane_id, sku_id)
        stok = self.durum.kovalar.toplam()
        return (stok[p, s].astype(float), self.durum.lam_base[p, s],
                self.durum.assort[p, s])

    def sow(self, eczane_id: np.ndarray, t: int) -> np.ndarray:
        p = np.array([self._ecz_sira[e] for e in eczane_id], dtype=int)
        return self.durum.sow[p]


# --------------------------------------------------------------------------
# karar arayuzu
# --------------------------------------------------------------------------
@dataclass
class TeklifPlani:
    """Bir karar haftasinin ciktisi. Dunyanin uygulayacagi tek sey bu.

    `teklifler` ve `mat` tepki fonksiyonunun ihtiyaci olan satir kimligi ve
    kol tablolarini tasir; `kol` politikanin secimidir (0 = teklif yok).
    `lot_id` M5'in tahsis katmanindan gelir; None ise dunya FEFO uygular.
    """

    teklifler: pl.DataFrame
    mat: TeklifMatrisleri
    kol: np.ndarray
    lot_id: np.ndarray | None = None

    @property
    def teklif_maskesi(self) -> np.ndarray:
        return self.kol != TEKLIF_YOK


# Karar verici: karar haftasi t ve o haftanin gozlemlenebilir aday dunyasi
# verilir, bir plan doner. None donmesi "bu hafta teklif yok" demektir.
KararVerici = Callable[[int, AdayDunyasi], "TeklifPlani | None"]


# --------------------------------------------------------------------------
# gozlemlenebilir gorunum: canli kayitlardan
# --------------------------------------------------------------------------
def canli_aday_dunyasi(durum: DunyaDurumu, eczaneler: pl.DataFrame,
                       urunler: pl.DataFrame) -> AdayDunyasi:
    """Rollout'un KENDI gecmisinden aday dunyasi kurar.

    Politika taban dunyanin loglarini degil, kendi urettigi loglari gorur --
    kapali dongunun ikinci yarisi budur. Teklif kaynakli siparis ve sevkiyat
    satirlari da icindedir: gercek sistemde de o satirlar veri ambarina duser
    ve bir sonraki haftanin hiz tahminini besler. (Bu, M2'nin hiz tahminini
    politikanin kendi hacmiyle sismesine yol acar; olculuyor ve raporlaniyor.)

    `dunya_yukle`nin parquet + id->indeks yolu kullanilmaz: kayitlar zaten
    indeks uzayinda ve rollout bu fonksiyonu her karar haftasinda cagiriyor.
    """
    k = durum.kayit
    sip = np.array(k.siparis_kayit, dtype=np.int64).reshape(-1, 6)
    sevk = k.sevk_kayit
    imha = [i for i in k.imha_kayit if i.lot_id is not None]
    lotlar = durum.depo.tum_lotlar()

    olcek = eczaneler["aylik_recete_adedi"].to_numpy().astype(float)
    olcek = olcek / max(float(np.median(olcek)), EPSILON)

    return AdayDunyasi(
        eczaneler=eczaneler, urunler=urunler,
        P=durum.P, S=durum.S, W=durum.W,
        sip_p=sip[:, 1].astype(np.int32), sip_s=sip[:, 2].astype(np.int32),
        sip_w=sip[:, 0].astype(np.int32), sip_adet=sip[:, 3].astype(float),
        sevk_p=np.array([t.eczane_idx for t in sevk], dtype=np.int32),
        sevk_s=np.array([t.sku_idx for t in sevk], dtype=np.int32),
        sevk_w=np.array([t.hafta for t in sevk], dtype=np.int32),
        sevk_lot=np.array([t.lot_id for t in sevk], dtype=object),
        sevk_adet=np.array([t.adet for t in sevk], dtype=float),
        lot_id=np.array([l.lot_id for l in lotlar], dtype=object),
        lot_s=np.array([l.sku_idx for l in lotlar], dtype=np.int32),
        lot_giris=np.array([l.giris_haftasi for l in lotlar], dtype=np.int32),
        lot_miad_gun=np.array([l.miad_gun for l in lotlar], dtype=float),
        lot_adet=np.array([l.adet_giris for l in lotlar], dtype=float),
        lot_birim_maliyet=np.array([l.birim_maliyet for l in lotlar], dtype=float),
        imha_lot=np.array([i.lot_id for i in imha], dtype=object),
        imha_w=np.array([i.hafta for i in imha], dtype=np.int32),
        imha_adet=np.array([i.adet for i in imha], dtype=float),
        dsf=durum.dsf, soguk_zincir=durum.soguk, eczane_olcegi=olcek,
    )


# --------------------------------------------------------------------------
# teklifin uygulanmasi
# --------------------------------------------------------------------------
@dataclass
class TeklifSonucu:
    sevk: np.ndarray               # [P, S] sevk edilen adet (bedava dahil)
    miad_agirlikli: np.ndarray     # [P, S] adet-agirlikli mutlak miad gunu
    teklif_sayisi: int
    kabul_sayisi: int
    kabul_olasiligi_gercek: float  # teklif edilen satirlarda ortalama p_gercek
    kabul_olasiligi_tahmin: float  # ayni satirlarda politikanin tahmini
    faturalanan_adet: float
    bedava_adet: float
    karsilanamayan_adet: float     # kabul edildi ama depoda yoktu
    brut_marj: float               # TL, gerceklesen adet uzerinden
    ortalama_mf: float
    ortalama_vade: float


def _bos_teklif_sonucu(P: int, S: int) -> TeklifSonucu:
    return TeklifSonucu(
        sevk=np.zeros((P, S), dtype=np.int64),
        miad_agirlikli=np.zeros((P, S)), teklif_sayisi=0, kabul_sayisi=0,
        kabul_olasiligi_gercek=float("nan"), kabul_olasiligi_tahmin=float("nan"),
        faturalanan_adet=0.0, bedava_adet=0.0, karsilanamayan_adet=0.0,
        brut_marj=0.0, ortalama_mf=float("nan"), ortalama_vade=float("nan"))


def teklifi_uygula(cfg: Config, durum: DunyaDurumu, seedler: SeedBankasi,
                   plan: TeklifPlani, tepki: Tepki, w: int,
                   p_tahmin: np.ndarray | None = None) -> TeklifSonucu:
    """Kabul ornekler, depodan tahsis eder, sevkiyat matrisini uretir.

    ADET MUHASEBESI. Kabul edilen satirda eczane teklif edilen adedi birebir
    almaz (`sim/response.py` miktar carpani); carpan hem faturalanan hem
    BEDAVA adede uygulanir - MF orani teklifin tanimidir, kabul edilen
    miktarla birlikte olceklenir.

    DEPO GERCEK. Kabul edilen adet depoda yoksa sevk edilmez ve marj
    dogmaz. Bu satirlar `karsilanamayan_adet`e yazilir: M5'in LP'si beklenen
    degere gore plan yapiyordu, kapali dongude planin tutmadigi yer burasi.
    """
    P, S = durum.P, durum.S
    sonuc = _bos_teklif_sonucu(P, S)
    n = plan.kol.size
    if n == 0:
        return sonuc

    kabul, carpan = sonuc_ornekle(cfg, seedler, tepki, plan.kol, w)
    verilen = plan.teklif_maskesi
    sonuc.teklif_sayisi = int(verilen.sum())
    if sonuc.teklif_sayisi == 0:
        return sonuc

    idx = np.arange(n)
    uzay = plan.mat.uzay
    sonuc.kabul_olasiligi_gercek = float(tepki.olasilik[idx, plan.kol][verilen].mean())
    if p_tahmin is not None:
        sonuc.kabul_olasiligi_tahmin = float(p_tahmin[idx, plan.kol][verilen].mean())
    sonuc.ortalama_mf = float(uzay.mf[plan.kol][verilen].mean())
    sonuc.ortalama_vade = float(uzay.vade[plan.kol][verilen].mean())

    alindi = verilen & (kabul > 0)
    sonuc.kabul_sayisi = int(alindi.sum())
    if sonuc.kabul_sayisi == 0:
        return sonuc

    p_idx = plan.teklifler["eczane_idx"].to_numpy()
    s_idx = plan.teklifler["sku_idx"].to_numpy()
    risk = durum.ecz.master["vade_riski_skoru"].to_numpy().astype(float)
    bugun = w * GUN_HAFTA

    for i in np.flatnonzero(alindi):
        a = int(plan.kol[i])
        p_i, s_i = int(p_idx[i]), int(s_idx[i])
        fatura = int(round(plan.mat.adet[i, a] * carpan[i]))
        bedava = int(round(plan.mat.bedava[i, a] * carpan[i]))
        istenen = fatura + bedava
        if istenen <= 0:
            continue
        gerekli_gun = durum.ecz.miad_toleransi[p_i] * durum.miad_kat_carpani[s_i]
        lot = None if plan.lot_id is None else plan.lot_id[i]
        karsilanan, satirlar, _ = durum.depo.tahsis_et(
            sku_idx=s_i, adet=istenen, hafta=w, eczane_idx=p_i,
            gerekli_kalan_gun=gerekli_gun,
            oncelikli_lot=(lot if lot else None),
        )
        durum.kayit.sevk_kayit.extend(satirlar)
        for t in satirlar:
            sonuc.miad_agirlikli[p_i, s_i] += t.adet * (bugun + t.kalan_raf_omru_gun)
        sonuc.sevk[p_i, s_i] += karsilanan
        sonuc.karsilanamayan_adet += istenen - karsilanan
        # Teklif de bir siparistir: politikanin kendi hacmi bir sonraki
        # haftanin gozlemlenebilir katmanina duser (canli_aday_dunyasi).
        durum.kayit.siparis_kayit.append(
            (w, p_i, s_i, istenen, karsilanan, 0))
        if karsilanan <= 0:
            continue
        # Kismi karsilamada MF orani korunur: bedava pay ayni oranda kisilir.
        pay = karsilanan / max(istenen, 1)
        f_ger, b_ger = fatura * pay, bedava * pay
        sonuc.faturalanan_adet += f_ger
        sonuc.bedava_adet += b_ger
        sonuc.brut_marj += float(brut_marj(
            np.array([f_ger]), np.array([b_ger]),
            np.array([durum.dsf[s_i]]), np.array([durum.depo_marji[s_i]]),
            float(uzay.vade[a]), np.array([risk[p_i]]), cfg)[0])
    return sonuc


# --------------------------------------------------------------------------
# haftalik olcum
# --------------------------------------------------------------------------
@dataclass
class HaftaOlcumu:
    hafta: int
    teklif_sayisi: int
    kabul_sayisi: int
    teklif_brut_marj: float
    organik_brut_marj: float
    teklif_sevk_adet: float
    organik_sevk_adet: float
    faturalanan_adet: float
    bedava_adet: float
    iade_adet: float
    iade_marj_geri_alma: float
    iade_islem_maliyeti: float
    imha_adet: float
    imha_islem_maliyeti: float
    karsilanmayan_siparis_adet: float
    organik_siparis_adet: float
    sow_ortalama: float
    eczane_stogu: float
    kabul_olasiligi_gercek: float
    kabul_olasiligi_tahmin: float
    ortalama_mf: float
    ortalama_vade: float

    @property
    def net_marj(self) -> float:
        """M5 (reports/m5.md 3.2) ile AYNI muhasebe, gerceklesen adetlerle.

            net = teklif brut + organik brut
                  - iade edilen adedin marji      (satis geri alinir)
                  - iade islem maliyeti
                  - depo imha islem maliyeti
        """
        return (self.teklif_brut_marj + self.organik_brut_marj
                - self.iade_marj_geri_alma - self.iade_islem_maliyeti
                - self.imha_islem_maliyeti)


@dataclass
class RolloutOlcumu:
    ad: str
    haftalar: list[HaftaOlcumu] = field(default_factory=list)

    def seri(self, alan: str) -> np.ndarray:
        return np.array([getattr(h, alan) for h in self.haftalar], dtype=float)

    def birikimli_net_marj(self) -> np.ndarray:
        return np.cumsum(np.array([h.net_marj for h in self.haftalar]))

    def ufukta(self, alan: str, ufuk: int) -> float:
        """Ilk `ufuk` haftanin toplami. Ufuk kosulan haftadan uzunsa kirpilir."""
        v = self.seri(alan)[:ufuk]
        return float(v.sum())

    def net_marj_ufukta(self, ufuk: int) -> float:
        b = self.birikimli_net_marj()
        if b.size == 0:
            return float("nan")
        return float(b[min(ufuk, b.size) - 1])


def _hafta_olcumu(cfg: Config, durum: DunyaDurumu, w: int, ts: TeklifSonucu,
                  imleç: dict) -> HaftaOlcumu:
    """Haftanin dunya kayitlarindan gerceklesen buyuklukleri toplar."""
    k = durum.kayit
    imha_orani = cfg.lot.maliyet.imha_birim_maliyeti_dsf_orani
    dsf, marj = durum.dsf, durum.depo_marji
    risk = durum.ecz.master["vade_riski_skoru"].to_numpy().astype(float)
    taban_vade = float(cfg.politika.aksiyon.taban_vade_gun)

    organik = k.sevk_kayit[imleç["sevk_organik"]:]
    o_marj = 0.0
    o_adet = 0.0
    if organik:
        adet = np.array([t.adet for t in organik], dtype=float)
        s = np.array([t.sku_idx for t in organik])
        p = np.array([t.eczane_idx for t in organik])
        o_adet = float(adet.sum())
        o_marj = float(brut_marj(adet, np.zeros(adet.size), dsf[s], marj[s],
                                 taban_vade, risk[p], cfg).sum())

    iade = k.iade_kayit[imleç["iade"]:]
    i_adet = i_marj = i_islem = 0.0
    if iade:
        ia = np.array([r[3] for r in iade], dtype=float)
        donen = np.array([r[4] for r in iade], dtype=float)
        s = np.array([r[2] for r in iade])
        i_adet = float(ia.sum())
        i_marj = float((ia * dsf[s] * marj[s]).sum())
        i_islem = float((donen * dsf[s] * imha_orani).sum())

    imha = [i for i in k.imha_kayit[imleç["imha"]:] if i.lot_id is not None]
    m_adet = float(sum(i.adet for i in imha))
    m_islem = float(sum(i.imha_maliyeti for i in imha))

    siparis = k.siparis_kayit[imleç["siparis"]:]
    sip_istenen = float(sum(r[3] for r in siparis))
    sip_karsilanan = float(sum(r[4] for r in siparis))

    return HaftaOlcumu(
        hafta=w,
        teklif_sayisi=ts.teklif_sayisi, kabul_sayisi=ts.kabul_sayisi,
        teklif_brut_marj=ts.brut_marj, organik_brut_marj=o_marj,
        teklif_sevk_adet=float(ts.sevk.sum()), organik_sevk_adet=o_adet,
        faturalanan_adet=ts.faturalanan_adet, bedava_adet=ts.bedava_adet,
        iade_adet=i_adet, iade_marj_geri_alma=i_marj, iade_islem_maliyeti=i_islem,
        imha_adet=m_adet, imha_islem_maliyeti=m_islem,
        karsilanmayan_siparis_adet=sip_istenen - sip_karsilanan,
        organik_siparis_adet=sip_istenen,
        sow_ortalama=float(durum.sow.mean()),
        eczane_stogu=float(durum.kovalar.toplam().sum()),
        kabul_olasiligi_gercek=ts.kabul_olasiligi_gercek,
        kabul_olasiligi_tahmin=ts.kabul_olasiligi_tahmin,
        ortalama_mf=ts.ortalama_mf, ortalama_vade=ts.ortalama_vade,
    )


# --------------------------------------------------------------------------
# surucu
# --------------------------------------------------------------------------
def rollout_kos(cfg: Config, durum: DunyaDurumu, evren: TepkiEvreni,
                eczaneler: pl.DataFrame, urunler: pl.DataFrame,
                latent_eczane: pl.DataFrame, karar_verici: KararVerici,
                ad: str, tahmin_olasiligi: Callable | None = None
                ) -> RolloutOlcumu:
    """`ope.rollout.ufuk_hafta` kadar hafta kosar, her hafta teklifi uygular.

    ORTAK RASSAL SAYI (CRN). Her rollout haftasi kendi tohumunu
    `ope.rollout.seed`ten turetir; boylece iki politika AYNI tuketim
    cekilisini, AYNI tedarikci secim gurultusunu ve AYNI kabul zarlarini
    gorur. Aralarindaki fark politika farkidir. Hafta ICINDE (ikmal
    dongusunde) cekilis sayisi veriye bagli oldugu icin akislar ayrisir; buyuk
    varyans kaynaklari -- tuketim, tedarikci secimi, kabul -- haftanin
    basinda ve ayni akistan cekiliyor.

    UFUK kosulan hafta sayisidir; teklif yalnizca ilk
    `teklif_penceresi_hafta` haftada verilir. Ikisi ayri: "4 hafta mi 52 hafta
    mi bakiyorsun" ile "kac hafta teklif veriyorsun" farkli sorulardir ve
    SPEC 5'in ogretici karsitligi birincisidir.
    """
    r = cfg.ope.rollout
    seedler = SeedBankasi(r.seed)
    baslangic = durum.w
    ecz_id = eczaneler["eczane_id"].to_numpy()
    sku_id = urunler["sku_id"].to_numpy()
    canli = CanliDurum(durum, ecz_id, sku_id, latent_eczane)
    olcum = RolloutOlcumu(ad=ad)
    son_plan: TeklifPlani | None = None
    son_tahmin: np.ndarray | None = None

    for adim in range(r.ufuk_hafta):
        w = baslangic + adim
        if w >= durum.W:
            break
        # Karar, HAFTANIN BASINDA ve w-1'in sonundaki durumla alinir; M4/M5'in
        # origin tanimiyla ayni (gorunum_kur(t) hafta <= t verisini kullanir).
        yeni_karar = (adim % r.karar_araligi_hafta) == 0
        teklif_penceresinde = adim < r.teklif_penceresi_hafta
        if yeni_karar and teklif_penceresinde:
            aday_dunya = canli_aday_dunyasi(durum, eczaneler, urunler)
            son_plan = karar_verici(w - 1, aday_dunya)
            son_tahmin = (tahmin_olasiligi(son_plan) if son_plan is not None
                          and tahmin_olasiligi is not None else None)
        elif not teklif_penceresinde:
            son_plan, son_tahmin = None, None

        if son_plan is not None and son_plan.teklif_maskesi.any():
            tepki = tepki_hesapla(cfg, evren, canli, son_plan.mat.uzay,
                                  son_plan.teklifler, w - 1, son_plan.mat.adet)
            ts = teklifi_uygula(cfg, durum, seedler, son_plan, tepki, w, son_tahmin)
        else:
            ts = _bos_teklif_sonucu(durum.P, durum.S)

        # Imlecler teklif uygulandiktan SONRA alinir: boylece haftalik olcumun
        # "organik" kalemleri teklif satirlarini icermez ve ikisi ayri ayri
        # okunabilir (teklifin kendi sevkiyati `ts`de zaten var).
        imleç = {"sevk_organik": len(durum.kayit.sevk_kayit),
                 "iade": len(durum.kayit.iade_kayit),
                 "imha": len(durum.kayit.imha_kayit),
                 "siparis": len(durum.kayit.siparis_kayit)}

        durum.rng = seedler.uretec(f"rollout_dunya_{w}")
        hafta_adimi(durum, teklif_sevk=ts.sevk,
                    teklif_miad_agirlikli=ts.miad_agirlikli)
        olcum.haftalar.append(_hafta_olcumu(cfg, durum, w, ts, imleç))
    return olcum
