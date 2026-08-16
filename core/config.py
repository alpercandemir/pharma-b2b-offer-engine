"""Config yukleme ve sema dogrulama.

CLAUDE.md 2 (sihirli sayi yasagi) burada MEKANIK olarak uygulanir:

  - Hicbir alanin Python tarafinda varsayilan degeri yoktur. Eksik knob
    -> yuklemede ValidationError. Sessizce varsayilana dusme yok.
  - `extra="forbid"`: YAML'da fazla/yanlis yazilmis anahtar -> hata.
    Boylece "config'e yazdim ama kod okumuyor" sinifi sessiz hata olmaz.

Config hash'i her kosunun manifest'ine yazilir; iki kosunun ayni config'le
kostugu boylece kanitlanabilir.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

UrunTipi = Literal["RX", "OTC", "TEG", "DERMOKOZMETIK", "MEDIKAL"]
ReceteRengi = Literal["NORMAL", "KIRMIZI", "YESIL", "MOR", "TURUNCU"]
OlayTipi = Literal[
    "REFERANS_KUR_GUNCELLEME",
    "SGK_LISTE_GUNCELLEME",
    "TITCK_GERI_CEKME",
    "TEDARIK_KRIZI",
    "EPIDEMI_DALGASI",
]
OlayKapsami = Literal["GLOBAL", "SKU", "KATEGORI_AKUT"]


class Kati(BaseModel):
    """Tum config modellerinin tabani: fazla anahtar yasak, varsayilan yok."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------
# profil
# --------------------------------------------------------------------------
class Profil(Kati):
    ad: str
    eczane_sayisi: int
    sku_sayisi: int
    hafta_sayisi: int
    baslangic_tarihi: date
    temel_seed: int


class ProfileFile(Kati):
    profil: Profil


# --------------------------------------------------------------------------
# products.yaml
# --------------------------------------------------------------------------
class UrunEvreni(Kati):
    populerlik_log_sigma: float
    koli_ici_adet_secenekleri: list[int]
    koli_ici_adet_agirliklari: list[float]
    birim_hacim_log_ort: float
    birim_hacim_log_sigma: float
    kdv_orani: float
    its_olasiligi_rx: float
    its_olasiligi_rx_disi: float

    @model_validator(mode="after")
    def _uzunluk(self):
        if len(self.koli_ici_adet_secenekleri) != len(self.koli_ici_adet_agirliklari):
            raise ValueError("koli_ici_adet secenek ve agirlik uzunluklari esit degil")
        return self


class MarjKademesi(Kati):
    psf_ust_siniri: float | None
    depo_marji: float
    eczane_marji: float


class PromosyonKurali(Kati):
    recete_rengi_vetosu: list[ReceteRengi]
    urun_tipi_serbest: list[UrunTipi]
    rx_sgk_disi_serbest: bool
    rx_sgk_kapsaminda_serbest: bool


class Kategori(Kati):
    kod: str
    ad: str
    atc_prefix: str | None
    pay: float
    urun_tipi_dagilimi: dict[UrunTipi, float]
    akut: bool
    kronik: bool
    soguk_zincir_olasiligi: float
    mevsimsellik: list[float]
    ramazan_carpani: float
    miad_toleransi_carpani: float
    etken_madde_orani: float
    fiyat_log_ort: float
    fiyat_log_sigma: float
    recete_rengi_dagilimi: dict[ReceteRengi, float]
    sgk_olasiligi: float
    tedarik_guclugu_olasiligi: float

    @model_validator(mode="after")
    def _kontrol(self):
        if len(self.mevsimsellik) != 12:
            raise ValueError(f"{self.kod}: mevsimsellik 12 aylik olmali")
        if abs(sum(self.urun_tipi_dagilimi.values()) - 1.0) > 1e-6:
            raise ValueError(f"{self.kod}: urun_tipi_dagilimi 1.0 toplamiyor")
        if abs(sum(self.recete_rengi_dagilimi.values()) - 1.0) > 1e-6:
            raise ValueError(f"{self.kod}: recete_rengi_dagilimi 1.0 toplamiyor")
        return self


class UrunConfig(Kati):
    evren: UrunEvreni
    marj_kademeleri: list[MarjKademesi]
    promosyon_serbest_kurali: PromosyonKurali
    kategoriler: list[Kategori]

    @model_validator(mode="after")
    def _kontrol(self):
        if abs(sum(k.pay for k in self.kategoriler) - 1.0) > 1e-6:
            raise ValueError("kategori paylari 1.0 toplamiyor")
        if self.marj_kademeleri[-1].psf_ust_siniri is not None:
            raise ValueError("son marj kademesinin psf_ust_siniri null olmali")
        return self


# --------------------------------------------------------------------------
# pharmacies.yaml
# --------------------------------------------------------------------------
class Il(Kati):
    ad: str
    pay: float
    turizm_olasiligi: float
    sosyoekonomik_ort: float
    sosyoekonomik_sigma: float


class Cografya(Kati):
    ilce_sayisi_il_basina: int
    semt_sayisi_ilce_basina: int


class Konum(Kati):
    hastane_mesafesi_log_ort: float
    hastane_mesafesi_log_sigma: float
    mesafe_olcegi_km: float


class Olcek(Kati):
    buyukluk_log_sigma: float
    aylik_recete_taban: int
    ciro_bandi_sinirlari: list[float]


class Nobet(Kati):
    rotasyon_periyodu_gun_secenekleri: list[int]
    rotasyon_periyodu_agirliklari: list[float]


class Kredi(Kati):
    vade_riski_ort: float
    vade_riski_sigma: float
    ortalama_recete_tutari: float
    dbs_limiti_carpani_ort: float
    dbs_limiti_carpani_sigma: float
    dbs_limiti_carpani_min: float


class ReceteKarmasi(Kati):
    sgk_recete_orani_ort: float
    sgk_recete_orani_sigma: float


class LatentSow(Kati):
    beta_a: float
    beta_b: float
    min: float
    max: float


class LatentStokculuk(Kati):
    log_ort: float
    log_sigma: float
    ust_sinir: float


class LatentMiadToleransi(Kati):
    taban_gun_ort: float
    taban_gun_sigma: float
    min_gun: float
    max_gun: float


class LatentSiparisDavranisi(Kati):
    kapsama_hafta_ort: float
    kapsama_hafta_sigma: float
    kapsama_hafta_min: float
    gozden_gecirme_periyodu_secenekleri: list[int]
    gozden_gecirme_periyodu_agirliklari: list[float]


class KategoriEgilimSatiri(Kati):
    taban: float
    hastane_kats: float
    sosyo_kats: float
    turizm_kats: float


class KategoriEgilimi(Kati):
    gurultu_sigma: float
    tablo: dict[str, KategoriEgilimSatiri]


class EczaneConfig(Kati):
    iller: list[Il]
    cografya: Cografya
    konum: Konum
    olcek: Olcek
    nobet: Nobet
    kredi: Kredi
    recete_karmasi: ReceteKarmasi
    latent_share_of_wallet: LatentSow
    latent_stokculuk: LatentStokculuk
    latent_miad_toleransi: LatentMiadToleransi
    latent_siparis_davranisi: LatentSiparisDavranisi
    kategori_egilimi: KategoriEgilimi

    @model_validator(mode="after")
    def _kontrol(self):
        if abs(sum(i.pay for i in self.iller) - 1.0) > 1e-6:
            raise ValueError("il paylari 1.0 toplamiyor")
        return self


# --------------------------------------------------------------------------
# sim.yaml
# --------------------------------------------------------------------------
class RamazanPenceresi(Kati):
    baslangic: date
    bitis: date


class TakvimConfig(Kati):
    ramazan_pencereleri: list[RamazanPenceresi]
    yil_sonu_stoklama_aylari: list[int]
    yil_sonu_stoklama_yogunlugu: float


class Cesitlendirme(Kati):
    taban_oran: float
    populerlik_agirligi: float
    affinite_agirligi: float
    buyukluk_agirligi: float
    haftalik_churn_orani: float
    yeni_cesit_deneme_adedi: float


class Yogunluk(Kati):
    taban_adet_hafta: float
    hucre_gurultu_shape: float


class Dagilim(Kati):
    negbin_shape: float
    sifir_sisirme: float
    kategori_hafta_soku_sigma: float
    eczane_hafta_soku_sigma: float


class NobetTalep(Kati):
    akut_carpani: float


class TurizmTalep(Kati):
    zirve_carpani: float
    zirve_aylari: list[int]
    omuz_carpani: float
    omuz_aylari: list[int]


class TalepConfig(Kati):
    cesitlendirme: Cesitlendirme
    yogunluk: Yogunluk
    dagilim: Dagilim
    nobet: NobetTalep
    turizm: TurizmTalep


class EnvanterConfig(Kati):
    tedarik_suresi_hafta: int
    emniyet_stogu_hafta: float
    emniyet_z_katsayisi: float
    talep_varyans_ewma_alfa: float
    azami_kapsama_hafta: float
    minimum_siparis_adedi: int
    soguk_zincir_minimum_siparis_adedi: int
    talep_ewma_alfa: float
    baslangic_kapsama_hafta: float
    stoksuzlukta_acil_gozden_gecirme: bool
    antisipasyon_kapsama_kazanci: float
    antisipasyon_siparis_esigi: float
    antisipasyon_azami_carpan: float
    eczane_lot_bolme_sayisi: int


class TedarikciSecimi(Kati):
    siparis_gurultusu: float
    karsilanamayan_siparis_rakibe_gider: bool
    stoksuzluk_sow_cezasi: float
    sow_toparlanma_hizi: float
    sow_rassal_yuruyus_sigma: float


class IadeConfig(Kati):
    eczaci_guvenlik_marji_gun: float
    degerlendirme_esigi_gun: float
    depoya_iade_orani: float
    kredi_orani: float
    sow_cezasi: float
    cesitten_cikarmada_iade: bool


class IkmalConfig(Kati):
    periyot_hafta: int
    hedef_kapsama_hafta: float
    emniyet_z_katsayisi: float
    baslangic_kapsama_hafta: float
    siparis_gurultusu_sigma: float
    minimum_parti_adet: int


class SimConfig(Kati):
    takvim: TakvimConfig
    talep: TalepConfig
    envanter: EnvanterConfig
    iade: IadeConfig
    tedarikci_secimi: TedarikciSecimi
    ikmal: IkmalConfig


# --------------------------------------------------------------------------
# events.yaml
# --------------------------------------------------------------------------
class ReferansKur(Kati):
    baslangic_deger: float
    guncelleme_artis_ort: float
    guncelleme_artis_sigma: float
    fiyat_gecis_katsayisi: float
    fiyat_gecis_gecikme_hafta: int


class OlayTanimi(Kati):
    tip: OlayTipi
    kapsam: OlayKapsami
    min_ara_hafta: int
    max_ara_hafta: int
    antisipasyon_hafta_min: int
    antisipasyon_hafta_max: int
    antisipasyon_siddeti: float
    tuketim_carpani: float
    sure_hafta_min: int
    sure_hafta_max: int
    kalici_seviye_kaymasi: float
    kalici_kayma_yukari_olasiligi: float
    etkilenen_sku_orani: float
    ikmal_bloklar: bool


class Ikame(Kati):
    geri_cekmede_ikame_orani: float


class OlayConfig(Kati):
    referans_kur: ReferansKur
    olaylar: list[OlayTanimi]
    ikame: Ikame


# --------------------------------------------------------------------------
# lots.yaml
# --------------------------------------------------------------------------
class RafOmru(Kati):
    toplam_gun_log_ort: float
    toplam_gun_log_sigma: float
    min_gun: int
    max_gun: int
    soguk_zincir_carpani: float


class LotGirisi(Kati):
    tuketilmis_oran_ort: float
    tuketilmis_oran_sigma: float
    tuketilmis_oran_ust_sinir: float
    kisa_miatli_parti_olasiligi: float
    kisa_miatli_kalan_gun_ort: float
    kisa_miatli_kalan_gun_sigma: float
    kisa_miatli_min_gun: int


class LotMaliyeti(Kati):
    parti_pazarligi_sigma: float
    imha_birim_maliyeti_dsf_orani: float


class LotTahsisi(Kati):
    fefo_aktif: bool
    miad_toleransi_uygulanir: bool


class LotConfig(Kati):
    raf_omru: RafOmru
    giris: LotGirisi
    maliyet: LotMaliyeti
    tahsis: LotTahsisi


# --------------------------------------------------------------------------
# features.yaml  (M2)
# --------------------------------------------------------------------------
class PanelConfig(Kati):
    ilk_origin_hafta: int
    origin_araligi_hafta: int
    aday_pencere_hafta: int
    min_siparis_sayisi: int
    egitim_orani: float

    @model_validator(mode="after")
    def _kontrol(self):
        if not 0.0 < self.egitim_orani < 1.0:
            raise ValueError("panel.egitim_orani (0, 1) araliginda olmali")
        if self.origin_araligi_hafta < 1:
            raise ValueError("panel.origin_araligi_hafta >= 1 olmali")
        return self


class HizConfig(Kati):
    pencereler_hafta: list[int]
    ewma_alfa: float
    havuzlama_gucu: float
    varsayilan_dongu_hafta: float
    min_hiz: float
    gozlenen_pay_tabani: float

    @model_validator(mode="after")
    def _kontrol(self):
        if not self.pencereler_hafta:
            raise ValueError("hiz.pencereler_hafta bos olamaz")
        if self.min_hiz <= 0:
            raise ValueError("hiz.min_hiz > 0 olmali (sifira bolme korumasi)")
        return self


class StokConfig(Kati):
    tavan_kapsama_hafta: float
    baslangic_kapsama_hafta: float
    varsayilan_gozlenen_pay: float

    @model_validator(mode="after")
    def _kontrol(self):
        if not 0.0 < self.varsayilan_gozlenen_pay <= 1.0:
            raise ValueError("stok.varsayilan_gozlenen_pay (0, 1] araliginda olmali")
        return self


class FeatureConfig(Kati):
    panel: PanelConfig
    hiz: HizConfig
    stok: StokConfig


# --------------------------------------------------------------------------
# depletion.yaml  (M2)
# --------------------------------------------------------------------------
class TukenmeHedefi(Kati):
    ufuk_hafta: int
    karar_ufku_hafta: int
    sinir_tamponu_hafta: int

    @model_validator(mode="after")
    def _kontrol(self):
        if self.karar_ufku_hafta > self.ufuk_hafta:
            raise ValueError("hedef.karar_ufku_hafta, ufuk_hafta'yi asamaz")
        if self.sinir_tamponu_hafta < self.ufuk_hafta:
            raise ValueError(
                "hedef.sinir_tamponu_hafta < ufuk_hafta: egitim etiketi test "
                "penceresine tasar (zaman sizintisi)"
            )
        return self


class HazardModeli(Kati):
    ogrenme_orani: float
    azami_agac: int
    azami_yaprak: int
    min_yaprak_ornegi: int
    l2_duzenlilestirme: float
    ozellik_orani: float
    erken_durdurma: bool
    dogrulama_orani: float
    sabir: int
    seed: int
    azami_egitim_satiri: int


class TabanKural(Kati):
    son_n_gun: int
    n_gun_adaylari: list[int]


class Degerlendirme(Kati):
    kalibrasyon_kova_sayisi: int
    bootstrap_orneklem: int
    bootstrap_seed: int
    oracle_teshisi: bool


class TukenmeConfig(Kati):
    hedef: TukenmeHedefi
    model: HazardModeli
    taban_kural: TabanKural
    degerlendirme: Degerlendirme


# --------------------------------------------------------------------------
# policy.yaml  (M3)
# --------------------------------------------------------------------------
class BenzerlikConfig(Kati):
    min_ortak_eczane: int
    kirpma: float
    komsu_sku_sayisi: int

    @model_validator(mode="after")
    def _kontrol(self):
        if self.min_ortak_eczane < 1:
            raise ValueError("benzerlik.min_ortak_eczane >= 1 olmali")
        if self.komsu_sku_sayisi < 1:
            raise ValueError("benzerlik.komsu_sku_sayisi >= 1 olmali")
        return self


class SepetConfig(Kati):
    pencere_hafta: int
    min_destek: int
    min_lift: float

    @model_validator(mode="after")
    def _kontrol(self):
        if self.pencere_hafta < 1:
            raise ValueError("sepet.pencere_hafta >= 1 olmali")
        return self


class SogukStartConfig(Kati):
    komsu_eczane_sayisi: int
    soguk_dilim: float
    oznitelik_agirliklari: dict[str, float]

    # Eczane benzerliginde kullanilan oznitelik adlari. Config'te fazla ya da
    # eksik ad varsa yukleme duser: "agirligi yazdim ama kod okumuyor" sinifi
    # sessiz hata olmasin.
    OZNITELIKLER: ClassVar[frozenset[str]] = frozenset(
        {"il", "hastane_yakinligi", "sosyoekonomik", "turizm", "olcek",
         "sgk_recete_orani", "nobet"})

    @model_validator(mode="after")
    def _kontrol(self):
        if set(self.oznitelik_agirliklari) != set(self.OZNITELIKLER):
            eksik = self.OZNITELIKLER - set(self.oznitelik_agirliklari)
            fazla = set(self.oznitelik_agirliklari) - self.OZNITELIKLER
            raise ValueError(
                f"soguk_start.oznitelik_agirliklari kod ile ortusmuyor. "
                f"eksik={sorted(eksik)} fazla={sorted(fazla)}")
        if self.komsu_eczane_sayisi < 1:
            raise ValueError("soguk_start.komsu_eczane_sayisi >= 1 olmali")
        if not 0.0 < self.soguk_dilim < 1.0:
            raise ValueError("soguk_start.soguk_dilim (0, 1) araliginda olmali")
        return self


class AdayDegerlendirme(Kati):
    ufuk_hafta: int
    origin_sayisi: int
    k_degerleri: list[int]

    @model_validator(mode="after")
    def _kontrol(self):
        if self.ufuk_hafta < 1 or self.origin_sayisi < 1:
            raise ValueError("degerlendirme.ufuk_hafta ve origin_sayisi >= 1 olmali")
        if not self.k_degerleri or min(self.k_degerleri) < 1:
            raise ValueError("degerlendirme.k_degerleri bos olamaz ve >= 1 olmali")
        return self


class AdayConfig(Kati):
    pencere_hafta: int
    yariomur_hafta: float
    havuz_boyutu_k: int
    benzerlik: BenzerlikConfig
    sepet: SepetConfig
    soguk_start: SogukStartConfig
    karisim_agirliklari: dict[str, float]
    miad_baskisi_agirligi: float
    miad_baskisi_esik_gun: float
    teklif_kapsama_hafta: float
    hiz_telafi_katsayisi: float
    degerlendirme: AdayDegerlendirme

    # Hibrit karisima giren ureticiler. policy/candidates.py'deki uretici
    # adlariyla birebir ayni olmali.
    URETICILER: ClassVar[tuple[str, ...]] = (
        "tekrar", "cf", "sepet", "soguk_start", "populerlik")

    @model_validator(mode="after")
    def _kontrol(self):
        if set(self.karisim_agirliklari) != set(self.URETICILER):
            eksik = set(self.URETICILER) - set(self.karisim_agirliklari)
            fazla = set(self.karisim_agirliklari) - set(self.URETICILER)
            raise ValueError(
                f"aday.karisim_agirliklari ureticilerle ortusmuyor. "
                f"eksik={sorted(eksik)} fazla={sorted(fazla)}")
        if sum(self.karisim_agirliklari.values()) <= 0:
            raise ValueError("aday.karisim_agirliklari toplami > 0 olmali")
        if self.havuz_boyutu_k < 1:
            raise ValueError("aday.havuz_boyutu_k >= 1 olmali")
        if self.yariomur_hafta <= 0:
            raise ValueError("aday.yariomur_hafta > 0 olmali")
        if self.hiz_telafi_katsayisi <= 0:
            raise ValueError("aday.hiz_telafi_katsayisi > 0 olmali")
        return self


class KisitConfig(Kati):
    recete_rengi_vetosu: list[ReceteRengi]
    tedarik_guclugu_veto: bool
    sgk_kapsaminda_mf_serbest: bool
    asgari_kalan_raf_omru_gun: float
    soguk_zincir_raf_omru_carpani: float
    soguk_zincir_min_siparis_adedi: int
    soguk_zincir_min_altinda: Literal["yukselt", "veto"]
    azami_kapsama_hafta: float
    depo_stok_yeterlilik_carpani: float
    acik_bakiye_vade_hafta: int
    kredi_kullanim_tavani: float
    vade_riski_cezasi: float
    eczane_haftalik_teklif_tavani: int

    @model_validator(mode="after")
    def _kontrol(self):
        if self.eczane_haftalik_teklif_tavani < 1:
            raise ValueError("kisit.eczane_haftalik_teklif_tavani >= 1 olmali")
        if not 0.0 <= self.vade_riski_cezasi <= 1.0:
            raise ValueError("kisit.vade_riski_cezasi [0, 1] araliginda olmali")
        if self.kredi_kullanim_tavani < 0:
            raise ValueError("kisit.kredi_kullanim_tavani >= 0 olmali")
        return self


class AksiyonConfig(Kati):
    mf_oranlari: list[float]
    vade_gunleri: list[int]
    taban_vade_gun: int
    koli_katina_yuvarla: bool

    @model_validator(mode="after")
    def _kontrol(self):
        if not self.mf_oranlari or min(self.mf_oranlari) < 0.0:
            raise ValueError("aksiyon.mf_oranlari bos olamaz ve negatif olamaz")
        if 0.0 not in self.mf_oranlari:
            raise ValueError(
                "aksiyon.mf_oranlari 0.0 icermeli: MF'siz teklif (yalnizca vade) "
                "SGK kapsamindaki urunde tek acik kanaldir (SPEC 2.5)")
        if len(set(self.mf_oranlari)) != len(self.mf_oranlari):
            raise ValueError("aksiyon.mf_oranlari tekrarli deger tasiyor")
        if not self.vade_gunleri or min(self.vade_gunleri) < 1:
            raise ValueError("aksiyon.vade_gunleri bos olamaz ve >= 1 olmali")
        if len(set(self.vade_gunleri)) != len(self.vade_gunleri):
            raise ValueError("aksiyon.vade_gunleri tekrarli deger tasiyor")
        if self.taban_vade_gun not in self.vade_gunleri:
            raise ValueError(
                f"aksiyon.taban_vade_gun ({self.taban_vade_gun}) vade_gunleri "
                f"icinde olmali: karsi-olgusal marj taban vadeyle hesaplaniyor, "
                f"aksiyon uzayinda karsiligi yoksa 'teklif verme' ile "
                f"'MF'siz taban vadeli teklif' karsilastirilamaz")
        return self


class SkorConfig(Kati):
    tedarikci_mf_destek_orani: float
    yillik_fonlama_orani: float
    tedarikci_vade_gun: int
    temerrut_ceza_katsayisi: float
    asgari_teklif_marji: float

    @model_validator(mode="after")
    def _kontrol(self):
        if not 0.0 <= self.tedarikci_mf_destek_orani <= 1.0:
            raise ValueError("skor.tedarikci_mf_destek_orani [0, 1] araliginda olmali")
        if self.yillik_fonlama_orani < 0.0:
            raise ValueError("skor.yillik_fonlama_orani >= 0 olmali")
        if self.temerrut_ceza_katsayisi < 0.0:
            raise ValueError("skor.temerrut_ceza_katsayisi >= 0 olmali")
        return self


class PolitikaConfig(Kati):
    aday: AdayConfig
    kisit: KisitConfig
    aksiyon: AksiyonConfig
    skor: SkorConfig


# --------------------------------------------------------------------------
# response.yaml  (M4 - simulator tarafi, tepki ground truth'u)
# --------------------------------------------------------------------------
class TepkiTaban(Kati):
    kesme: float
    ihtiyac_katsayisi: float
    ihtiyac_referans_hafta: float
    cesit_disi_ihtiyac: float
    sow_katsayisi: float
    yeni_hucre_cezasi: float
    fiyat_katsayisi: float
    hucre_gurultu_sigma: float

    @model_validator(mode="after")
    def _kontrol(self):
        if self.ihtiyac_referans_hafta <= 0:
            raise ValueError("tepki.taban.ihtiyac_referans_hafta > 0 olmali")
        return self


class TepkiTeklif(Kati):
    taban_etki: float
    mf_taban_etkisi: float
    mf_referans_orani: float
    mf_azalan_us: float
    vade_taban_etkisi: float
    vade_azalan_us: float
    ihtiyac_etkilesimi: float

    @model_validator(mode="after")
    def _kontrol(self):
        if self.mf_referans_orani <= 0:
            raise ValueError("tepki.teklif.mf_referans_orani > 0 olmali")
        return self


class TepkiDuyarlilik(Kati):
    heterojenlik_carpani: float
    mf_log_sigma: float
    mf_sosyoekonomik: float
    mf_olcek: float
    mf_sow: float
    vade_log_sigma: float
    vade_riski: float
    vade_dbs: float
    vade_stokculuk: float

    @model_validator(mode="after")
    def _kontrol(self):
        if self.heterojenlik_carpani < 0:
            raise ValueError("tepki.duyarlilik.heterojenlik_carpani >= 0 olmali")
        return self


class TepkiMiad(Kati):
    direnc_katsayisi: float


class TepkiMiktar(Kati):
    kabul_gurultu_sigma: float
    asgari_kabul_orani: float
    asiri_adet_direnci: float
    kapsama_toleransi: float
    asgari_esik_adet: float

    @model_validator(mode="after")
    def _kontrol(self):
        if self.kapsama_toleransi <= 0:
            raise ValueError("tepki.miktar.kapsama_toleransi > 0 olmali")
        return self


class TepkiConfig(Kati):
    taban: TepkiTaban
    teklif: TepkiTeklif
    duyarlilik: TepkiDuyarlilik
    urun_tipi_mf_carpani: dict[UrunTipi, float]
    miad: TepkiMiad
    miktar: TepkiMiktar

    # Butun urun tipleri tabloda olmak zorunda: eksik tip sessizce 0 carpan
    # alsaydi o tipte MF kanali gorunmez bicimde olurdu.
    URUN_TIPLERI: ClassVar[frozenset[str]] = frozenset(
        {"RX", "OTC", "TEG", "DERMOKOZMETIK", "MEDIKAL"})

    @model_validator(mode="after")
    def _kontrol(self):
        if set(self.urun_tipi_mf_carpani) != set(self.URUN_TIPLERI):
            eksik = self.URUN_TIPLERI - set(self.urun_tipi_mf_carpani)
            fazla = set(self.urun_tipi_mf_carpani) - self.URUN_TIPLERI
            raise ValueError(
                f"tepki.urun_tipi_mf_carpani urun tipleriyle ortusmuyor. "
                f"eksik={sorted(eksik)} fazla={sorted(fazla)}")
        return self


# --------------------------------------------------------------------------
# uplift.yaml  (M4 - kayit politikasi + ogrenici)
# --------------------------------------------------------------------------
class KayitPolitikasi(Kati):
    teklif_taban_olasiligi: float
    skor_egilimi: float
    derin_mf_egilimi: float
    kesif_orani: float
    seed: int

    @model_validator(mode="after")
    def _kontrol(self):
        if not 0.0 < self.kesif_orani <= 1.0:
            raise ValueError(
                "kayit.kesif_orani (0, 1] araliginda olmali: sifirda bazi "
                "kollarin propensity'si 0 olur, overlap kirilir ve M6'nin "
                "off-policy degerlendirmesi o kollarda tanimsiz kalir (D7)")
        if not 0.0 < self.teklif_taban_olasiligi < 1.0:
            raise ValueError("kayit.teklif_taban_olasiligi (0, 1) araliginda olmali")
        return self


class UpliftEgitim(Kati):
    ilk_origin_hafta: int
    origin_araligi_hafta: int
    azami_origin_sayisi: int
    sinir_tamponu_hafta: int

    @model_validator(mode="after")
    def _kontrol(self):
        if self.origin_araligi_hafta < 1 or self.azami_origin_sayisi < 1:
            raise ValueError("egitim.origin_araligi_hafta ve azami_origin_sayisi >= 1 olmali")
        if self.sinir_tamponu_hafta < 0:
            raise ValueError("egitim.sinir_tamponu_hafta >= 0 olmali")
        return self


class UpliftModeli(Kati):
    ogrenme_orani: float
    azami_agac: int
    azami_yaprak: int
    min_yaprak_ornegi: int
    l2_duzenlilestirme: float
    ozellik_orani: float
    erken_durdurma: bool
    dogrulama_orani: float
    sabir: int
    seed: int
    min_kol_orneklemi: int


class XOgreniciConfig(Kati):
    egilim_kirpma_alt: float
    egilim_kirpma_ust: float

    @model_validator(mode="after")
    def _kontrol(self):
        if not 0.0 < self.egilim_kirpma_alt < self.egilim_kirpma_ust < 1.0:
            raise ValueError("x_ogrenici kirpma sinirlari 0 < alt < ust < 1 olmali")
        return self


class UpliftDegerlendirme(Kati):
    bootstrap_orneklem: int
    bootstrap_seed: int
    qini_dilim_sayisi: int


class UpliftConfig(Kati):
    kayit: KayitPolitikasi
    egitim: UpliftEgitim
    model: UpliftModeli
    x_ogrenici: XOgreniciConfig
    degerlendirme: UpliftDegerlendirme


# --------------------------------------------------------------------------
# allocation.yaml  (M5 - tahsis LP'si + miad rejimi)
# --------------------------------------------------------------------------
class TahsisLP(Kati):
    aday_lot_sayisi: int
    cozucu_zaman_siniri_sn: float
    butunluk_yuvarlamasi: bool
    kredi_kisiti: bool
    sow_buyutme_agirligi: float

    @model_validator(mode="after")
    def _kontrol(self):
        if self.aday_lot_sayisi < 1:
            raise ValueError("tahsis.lp.aday_lot_sayisi >= 1 olmali")
        if self.cozucu_zaman_siniri_sn <= 0:
            raise ValueError("tahsis.lp.cozucu_zaman_siniri_sn > 0 olmali")
        if self.sow_buyutme_agirligi < 0:
            raise ValueError("tahsis.lp.sow_buyutme_agirligi >= 0 olmali")
        return self


class TemizlikConfig(Kati):
    tetik_gun: float
    deger_egrisi: Literal["lineer", "eksponansiyel", "basamakli"]
    egri_ussu: float
    basamak_esigi: float
    guvenlik_katsayisi: float
    eczaci_marji_gun: float
    asgari_kalan_raf_omru_gun: float
    imha_birim_maliyeti_dsf_orani: float
    normal_realizasyon_orani: float

    @model_validator(mode="after")
    def _kontrol(self):
        if self.tetik_gun <= 0:
            raise ValueError("temizlik.tetik_gun > 0 olmali")
        if self.egri_ussu <= 0:
            raise ValueError("temizlik.egri_ussu > 0 olmali")
        if not 0.0 < self.basamak_esigi < 1.0:
            raise ValueError("temizlik.basamak_esigi (0, 1) araliginda olmali")
        if not 0.0 < self.guvenlik_katsayisi <= 1.0:
            raise ValueError("temizlik.guvenlik_katsayisi (0, 1] araliginda olmali")
        if self.eczaci_marji_gun < 0:
            raise ValueError("temizlik.eczaci_marji_gun >= 0 olmali")
        if self.asgari_kalan_raf_omru_gun <= 0:
            raise ValueError(
                "temizlik.asgari_kalan_raf_omru_gun > 0 olmali: hicbir rejimde "
                "miadi dolmak uzere olan mal eczaneye yikilmaz (SPEC 2.5)")
        if self.imha_birim_maliyeti_dsf_orani < 0:
            raise ValueError("temizlik.imha_birim_maliyeti_dsf_orani >= 0 olmali")
        if not 0.0 <= self.normal_realizasyon_orani <= 1.0:
            raise ValueError("temizlik.normal_realizasyon_orani [0, 1] araliginda olmali")
        return self


class KitSenaryosu(Kati):
    kit_stok_carpani: float
    miad_hizlandirma_gun: float

    @model_validator(mode="after")
    def _kontrol(self):
        if not 0.0 < self.kit_stok_carpani <= 1.0:
            raise ValueError(
                "senaryo.kit_stok_carpani (0, 1] araliginda olmali: senaryo stogu "
                "kisitlar, dunyada olmayan mal yaratmaz")
        if self.miad_hizlandirma_gun < 0:
            raise ValueError("senaryo.miad_hizlandirma_gun >= 0 olmali")
        return self


class TahsisDegerlendirme(Kati):
    ornek_sayisi: int
    ornek_seed: int
    organik_cikis_pencere_hafta: int

    @model_validator(mode="after")
    def _kontrol(self):
        if self.ornek_sayisi < 1:
            raise ValueError("degerlendirme.ornek_sayisi >= 1 olmali")
        if self.organik_cikis_pencere_hafta < 1:
            raise ValueError("degerlendirme.organik_cikis_pencere_hafta >= 1 olmali")
        return self


class TahsisConfig(Kati):
    lp: TahsisLP
    temizlik: TemizlikConfig
    senaryo: KitSenaryosu
    degerlendirme: TahsisDegerlendirme


# --------------------------------------------------------------------------
# ope.yaml  (M6 - off-policy degerlendirme + kapali dongu rollout)
# --------------------------------------------------------------------------
class OPEKayit(Kati):
    tekrar_sayisi: int
    seed: int

    @model_validator(mode="after")
    def _kontrol(self):
        if self.tekrar_sayisi < 1:
            raise ValueError("ope.kayit.tekrar_sayisi >= 1 olmali")
        return self


class OPETahminci(Kati):
    kirpma_esigi: float
    dr_kirpma_esigi: float

    @model_validator(mode="after")
    def _kontrol(self):
        if self.kirpma_esigi < 1.0 or self.dr_kirpma_esigi < 1.0:
            raise ValueError(
                "ope.tahminci kirpma esikleri >= 1 olmali: agirlik w = 1/pi ve "
                "pi <= 1 oldugu icin 1'in altinda bir tavan HER satiri kirpar, "
                "tahminci de olceginden kopar")
        return self


class OPEPropensityModeli(Kati):
    ogrenme_orani: float
    azami_agac: int
    azami_yaprak: int
    min_yaprak_ornegi: int
    l2_duzenlilestirme: float
    seed: int


class OPEPropensity(Kati):
    kaynak: Literal["loglanan", "tahmin"]
    sicaklik: float
    kirpma_alt: float
    model: OPEPropensityModeli
    kalibrasyon_kova_sayisi: int

    @model_validator(mode="after")
    def _kontrol(self):
        if self.sicaklik <= 0:
            raise ValueError("ope.propensity.sicaklik > 0 olmali")
        if not 0.0 < self.kirpma_alt < 1.0:
            raise ValueError("ope.propensity.kirpma_alt (0, 1) araliginda olmali")
        if self.kalibrasyon_kova_sayisi < 2:
            raise ValueError("ope.propensity.kalibrasyon_kova_sayisi >= 2 olmali")
        return self


class OPEOrtusme(Kati):
    esik: float
    dusuk_destek_orneklemi: int

    @model_validator(mode="after")
    def _kontrol(self):
        if not 0.0 < self.esik < 1.0:
            raise ValueError("ope.ortusme.esik (0, 1) araliginda olmali")
        if self.dusuk_destek_orneklemi < 1:
            raise ValueError("ope.ortusme.dusuk_destek_orneklemi >= 1 olmali")
        return self


class OPERollout(Kati):
    baslangic_hafta: int
    ufuk_hafta: int
    raporlanan_ufuklar: list[int]
    teklif_penceresi_hafta: int
    karar_araligi_hafta: int
    seed: int
    politikalar: list[str]

    # Rollout'ta kosulabilen politikalar. experiments/run.py'deki
    # `ROLLOUT_POLITIKALARI` ile birebir ayni olmali: config'e yazilip kodun
    # tanimadigi bir ad sessizce atlanmasin.
    POLITIKALAR: ClassVar[frozenset[str]] = frozenset(
        {"teklif_yok", "m3_sabit_kampanya", "propensity", "uplift_t", "uplift_x",
         "agresif", "agresif_vade", "lp"})

    @model_validator(mode="after")
    def _kontrol(self):
        if self.ufuk_hafta < 1:
            raise ValueError("ope.rollout.ufuk_hafta >= 1 olmali")
        if self.baslangic_hafta < 1:
            raise ValueError("ope.rollout.baslangic_hafta >= 1 olmali")
        if not self.raporlanan_ufuklar:
            raise ValueError("ope.rollout.raporlanan_ufuklar bos olamaz")
        if min(self.raporlanan_ufuklar) < 1 or max(self.raporlanan_ufuklar) > self.ufuk_hafta:
            raise ValueError(
                f"ope.rollout.raporlanan_ufuklar [1, ufuk_hafta={self.ufuk_hafta}] "
                f"araliginda olmali: kosulmamis bir hafta raporlanamaz")
        if sorted(set(self.raporlanan_ufuklar)) != list(self.raporlanan_ufuklar):
            raise ValueError("ope.rollout.raporlanan_ufuklar artan ve tekrarsiz olmali")
        if not 1 <= self.teklif_penceresi_hafta <= self.ufuk_hafta:
            raise ValueError(
                "ope.rollout.teklif_penceresi_hafta [1, ufuk_hafta] araliginda olmali")
        if self.karar_araligi_hafta < 1:
            raise ValueError("ope.rollout.karar_araligi_hafta >= 1 olmali")
        bilinmeyen = set(self.politikalar) - self.POLITIKALAR
        if bilinmeyen:
            raise ValueError(
                f"ope.rollout.politikalar tanimsiz ad tasiyor: {sorted(bilinmeyen)}. "
                f"Gecerli: {sorted(self.POLITIKALAR)}")
        if "teklif_yok" not in self.politikalar:
            raise ValueError(
                "ope.rollout.politikalar 'teklif_yok' icermeli: kapali dongude "
                "artimsal deger ancak teklifsiz dunyaya gore tanimlidir")
        return self


class OPEDegerlendirme(Kati):
    bootstrap_orneklem: int
    bootstrap_seed: int


class OPEConfig(Kati):
    kayit: OPEKayit
    tahminci: OPETahminci
    propensity: OPEPropensity
    ortusme: OPEOrtusme
    rollout: OPERollout
    degerlendirme: OPEDegerlendirme


# --------------------------------------------------------------------------
# scenarios.yaml  (M7 - kur rejimi altinda kosullu okuma, D3/D4)
# --------------------------------------------------------------------------
class Rejim(Kati):
    ad: str
    aciklama: str
    guncelleme_beklentisi_hafta: float
    referans_kur_artisi: float
    fiyat_gecis_katsayisi: float
    antisipasyon_talep_carpani: float
    fonlama_orani_carpani: float

    @model_validator(mode="after")
    def _kontrol(self):
        if self.guncelleme_beklentisi_hafta <= 0:
            raise ValueError(f"{self.ad}: guncelleme_beklentisi_hafta > 0 olmali")
        if self.referans_kur_artisi < 0:
            raise ValueError(f"{self.ad}: referans_kur_artisi >= 0 olmali")
        if not 0.0 <= self.fiyat_gecis_katsayisi <= 1.0:
            raise ValueError(f"{self.ad}: fiyat_gecis_katsayisi [0, 1] araliginda olmali")
        if self.antisipasyon_talep_carpani <= 0:
            raise ValueError(f"{self.ad}: antisipasyon_talep_carpani > 0 olmali")
        if self.fonlama_orani_carpani < 0:
            raise ValueError(f"{self.ad}: fonlama_orani_carpani >= 0 olmali")
        return self

    @property
    def notr(self) -> bool:
        """Rejim marj aritmetigine ve talebe hicbir sey yapmiyor mu."""
        return (self.referans_kur_artisi == 0.0
                and self.fiyat_gecis_katsayisi == 0.0
                and self.antisipasyon_talep_carpani == 1.0
                and self.fonlama_orani_carpani == 1.0)


class SenaryoConfig(Kati):
    politika: str
    taban_ad: str
    ikame_ufku_hafta: float
    rejimler: list[Rejim]

    # Senaryo altinda kosulabilen teslim politikalari. experiments/run.py
    # `_gozlemlenebilir_politikalar`in anahtarlariyla birebir ayni olmali.
    # `teklif_yok` bilerek DISARIDA: hicbir teklif vermeyen bir politikanin
    # rejim duyarliligi tanimi geregi sifirdir, senaryo katmani olu olurdu.
    POLITIKALAR: ClassVar[frozenset[str]] = frozenset(
        {"m3_sabit_kampanya", "propensity_ham", "propensity", "uplift_t",
         "uplift_x"})

    @model_validator(mode="after")
    def _kontrol(self):
        if self.politika not in self.POLITIKALAR:
            raise ValueError(
                f"senaryo.politika tanimsiz: {self.politika}. "
                f"Gecerli: {sorted(self.POLITIKALAR)}")
        if self.ikame_ufku_hafta <= 0:
            raise ValueError("senaryo.ikame_ufku_hafta > 0 olmali")
        adlar = [r.ad for r in self.rejimler]
        if len(set(adlar)) != len(adlar):
            raise ValueError(f"senaryo.rejimler tekrarli ad tasiyor: {adlar}")
        return self

    def rejim(self, ad: str) -> Rejim:
        for r in self.rejimler:
            if r.ad == ad:
                return r
        raise KeyError(f"rejim yok: {ad}")


# --------------------------------------------------------------------------
# agent.yaml  (M7 - LLM katmani, D8)
# --------------------------------------------------------------------------
class AjanConfig(Kati):
    istemci: Literal["kayitli", "sablon", "anthropic"]
    kayit_dizini: str
    model: str
    azami_token: int
    sicaklik: float
    azami_tur: int
    brifing_teklif_sayisi: int
    brifing_veto_sayisi: int
    kol_ekonomisi_kol_sayisi: int

    @model_validator(mode="after")
    def _kontrol(self):
        if self.azami_tur < 1:
            raise ValueError("ajan.azami_tur >= 1 olmali")
        if self.azami_token < 1:
            raise ValueError("ajan.azami_token >= 1 olmali")
        if self.sicaklik < 0.0:
            raise ValueError("ajan.sicaklik >= 0 olmali")
        if self.brifing_teklif_sayisi < 1:
            raise ValueError(
                "ajan.brifing_teklif_sayisi >= 1 olmali: sifirda brifing hicbir "
                "teklif tasimaz ve sayi denetcisinin karsilastiracagi olgu kalmaz")
        if self.brifing_veto_sayisi < 0:
            raise ValueError("ajan.brifing_veto_sayisi >= 0 olmali")
        if self.kol_ekonomisi_kol_sayisi < 2:
            raise ValueError(
                "ajan.kol_ekonomisi_kol_sayisi >= 2 olmali: tek kollu tablo bir "
                "karsilastirma degildir")
        return self


# --------------------------------------------------------------------------
# harness.yaml  (M7 - deterministik denetim)
# --------------------------------------------------------------------------
class HarnessConfig(Kati):
    sayi_toleransi_bagil: float
    mutasyon_sapmasi: float
    yoksayilan_tamsayi_ust_siniri: float
    yuvarlama_basamaklari: list[int]
    temiz_vaka_bulgu_tavani: int

    @model_validator(mode="after")
    def _kontrol(self):
        if self.sayi_toleransi_bagil <= 0.0:
            raise ValueError(
                "harness.sayi_toleransi_bagil > 0 olmali: brifing sayilari "
                "yuvarlanmis yazilir, sifir tolerans her satiri ihlal sayardi")
        if self.mutasyon_sapmasi <= 0.0:
            raise ValueError("harness.mutasyon_sapmasi > 0 olmali")
        if self.yoksayilan_tamsayi_ust_siniri < 0:
            raise ValueError("harness.yoksayilan_tamsayi_ust_siniri >= 0 olmali")
        if not self.yuvarlama_basamaklari or min(self.yuvarlama_basamaklari) < 0:
            raise ValueError("harness.yuvarlama_basamaklari bos olamaz ve >= 0 olmali")
        if self.temiz_vaka_bulgu_tavani < 0:
            raise ValueError("harness.temiz_vaka_bulgu_tavani >= 0 olmali")
        return self


# --------------------------------------------------------------------------
# butun
# --------------------------------------------------------------------------
class Config(Kati):
    profil: Profil
    urun: UrunConfig
    eczane: EczaneConfig
    sim: SimConfig
    olay: OlayConfig
    lot: LotConfig
    feature: FeatureConfig
    tukenme: TukenmeConfig
    politika: PolitikaConfig
    tepki: TepkiConfig
    uplift: UpliftConfig
    tahsis: TahsisConfig
    ope: OPEConfig
    senaryo: SenaryoConfig
    ajan: AjanConfig
    harness: HarnessConfig

    @model_validator(mode="after")
    def _capraz_kontrol(self):
        kategori_kodlari = {k.kod for k in self.urun.kategoriler}
        egilim_kodlari = set(self.eczane.kategori_egilimi.tablo)
        if kategori_kodlari != egilim_kodlari:
            eksik = kategori_kodlari - egilim_kodlari
            fazla = egilim_kodlari - kategori_kodlari
            raise ValueError(
                f"kategori_egilimi tablosu kategorilerle ortusmuyor. "
                f"eksik={sorted(eksik)} fazla={sorted(fazla)}"
            )
        # D6 mekanik kilidi: regulasyonun yasakladigi recete renkleri
        # politikanin veto listesinde OLMAK ZORUNDA. Politika daha KATI
        # olabilir, daha gevsek olamaz. Boylece "kirmizi/yesil hicbir kosulda
        # onerilmez" iddiasi kod incelemesine degil, config yuklemesine
        # baglanir: gevsetme denemesi kosuyu dusurur.
        regulasyon = set(self.urun.promosyon_serbest_kurali.recete_rengi_vetosu)
        politika = set(self.politika.kisit.recete_rengi_vetosu)
        if not regulasyon <= politika:
            raise ValueError(
                f"kisit.recete_rengi_vetosu regulasyondan gevsek: "
                f"eksik={sorted(regulasyon - politika)}. Promosyonu yasak olan "
                f"recete rengi politika veto listesinden cikarilamaz (D6)."
            )
        self._m4_zaman_kilidi()
        self._m5_miad_kilidi()
        self._m6_ortusme_kilidi()
        self._m6_rollout_kilidi()
        self._m7_senaryo_kilidi()
        self._m7_harness_kilidi()
        return self

    def _m7_senaryo_kilidi(self) -> None:
        """D3 mekanik kilidi: taban NOTR, en az bir rejim NOTR DEGIL.

        Iki ayri olu-kadran hatasini birden kapatir:

          (a) Taban rejim notr degilse "baz senaryoya gore fark" cumlesi
              anlamsizdir: karsilastirmanin sifir noktasi kendisi bir
              mudahale olur ve rapordaki butun deltalar iki mudahalenin
              farkini olcer.
          (b) Butun rejimler notrse senaryo katmani OLU olur: uc rejim de
              ayni tabloyu uretir, "kur rejimi altinda politika ne oneriyor"
              sorusunun cevabi tanimi geregi "hep ayni sey" cikar ve
              harness'in senaryo karistirma denetcisi hicbir seyi
              ayirt edemez (iki rejimin sayilari zaten esittir).

        M5'in "temizlik penceresi bos, rejim olu" ve M6'nin "ortusme esigi
        tabanin altinda, teshis olu" kilitleriyle ayni disiplin.
        """
        s = self.senaryo
        adlar = [r.ad for r in s.rejimler]
        if s.taban_ad not in adlar:
            raise ValueError(
                f"senaryo.taban_ad ({s.taban_ad}) rejimler arasinda yok: {adlar}")
        taban = s.rejim(s.taban_ad)
        if not taban.notr:
            raise ValueError(
                f"taban rejim '{taban.ad}' notr degil (artis="
                f"{taban.referans_kur_artisi}, gecis={taban.fiyat_gecis_katsayisi}, "
                f"talep={taban.antisipasyon_talep_carpani}, fonlama="
                f"{taban.fonlama_orani_carpani}): butun farklar bu rejime gore "
                f"yaziliyor, notr olmayan taban 'fark' kelimesini anlamsiz kilar (D3)")
        if all(r.notr for r in s.rejimler):
            raise ValueError(
                "senaryo.rejimler icinde notr olmayan tek bir rejim yok: butun "
                "rejimler ayni tabloyu uretir, senaryo katmani olu (D3)")

    def _m7_harness_kilidi(self) -> None:
        """M7 mekanik kilidi: sayi denetcisi kendi mutantini yakalayabilmeli.

        Harness'in "sayi uydurma yakalaniyor" iddiasinin kaniti, temiz
        ciktidan uretilmis bozuk varyantin yakalanmasidir (harness/
        mutasyon.py). Mutasyon bir sayiyi `mutasyon_sapmasi` kadar oynatiyor;
        eslesme toleransi bundan genisse bozuk sayi da "eslesti" sayilir ve
        denetci OLU olur -- ustelik sessizce, cunku harness "butun vakalar
        temiz" der. Yon tek tarafli: tolerans daraltilabilir (kati olcum),
        genisletilemez.
        """
        h = self.harness
        if h.sayi_toleransi_bagil >= h.mutasyon_sapmasi:
            raise ValueError(
                f"harness.sayi_toleransi_bagil ({h.sayi_toleransi_bagil}) >= "
                f"mutasyon_sapmasi ({h.mutasyon_sapmasi}): bozulmus sayi da "
                f"tolerans icinde kalir, sayi denetcisi olu")

    def _m6_rollout_kilidi(self) -> None:
        """Rollout penceresi dunyanin icinde kalmali.

        Ufuk dunyanin sonunu asarsa rollout sessizce kisa keser ve
        `raporlanan_ufuklar`daki en buyuk deger KOSULMAMIS bir hafta sayisini
        etiketler -- yani tablo "52 hafta" yazarken 36 hafta gosterir. Bu, en
        kotu turden bir hata: kod calisir, sayi uretir, sayi yanlistir.

        Egitim penceresiyle ORTUSME burada HATA DEGIL, uyari degeri: 52
        haftalik varyanti kosabilmek icin bilerek serbest. Ortusen hafta
        sayisi `m6.rollout.egitim_ortusmesi_hafta` olarak her kosuda
        raporlanir ki sizinti gizli kalmasin.
        """
        r = self.ope.rollout
        son = r.baslangic_hafta + r.ufuk_hafta
        if son > self.profil.hafta_sayisi:
            raise ValueError(
                f"rollout penceresi dunyayi asiyor: baslangic={r.baslangic_hafta} "
                f"+ ufuk={r.ufuk_hafta} = {son} > hafta_sayisi="
                f"{self.profil.hafta_sayisi}")

    def _m6_ortusme_kilidi(self) -> None:
        """M6 mekanik kilidi: ortusme teshisi OLU olamaz.

        Kayit politikasi (policy/bandit.py) her izinli kola bir TABAN olasilik
        garanti eder:

            taban = q_alt x kesif_orani / |teklif kolu|

        `ope.ortusme.esik` bu tabanin ALTINA inerse hicbir satir "ortusme
        ihlali" olarak isaretlenemez: teshis her kosuda sifir doner ve rapora
        "ortusme sorunu yok" diye gecer. Bu, sorunun yoklugu degil, olcunun
        yoklugudur -- M5'in "temizlik penceresi bos, rejim olu" kilidiyle ayni
        disiplin (SPEC 5b.1: kadrani olmayan sayi knob degildir).

        Ters yon (esik cok BUYUK) kasten serbest: butun satirlari ihlal saymak
        yanlis bir olcum degil, kati bir olcumdur ve `m6.ortusme.ihlal_orani`
        metriginde acikca gorunur.
        """
        kol = len(self.politika.aksiyon.mf_oranlari) * len(self.politika.aksiyon.vade_gunleri)
        taban = self.KAYIT_Q_ALT * self.uplift.kayit.kesif_orani / max(kol, 1)
        if self.ope.ortusme.esik <= taban:
            raise ValueError(
                f"ope.ortusme.esik ({self.ope.ortusme.esik}) kayit politikasinin "
                f"taban propensity'sinin ({taban:.5f}) altinda ya da esit: hicbir "
                f"satir ortusme ihlali olarak isaretlenemez, teshis olu. "
                f"(kesif_orani={self.uplift.kayit.kesif_orani}, teklif kolu={kol})")

    def _m5_miad_kilidi(self) -> None:
        """D9 mekanik kilidi: temizlik rejimi bir GEVSETMEDIR, sikilastirma degil.

        Temizlik tabani normal tabandan buyuk olsaydi "rejim" adi yaniltici
        olurdu: temizlik penceresine giren lot normalden daha da erisilmez
        hale gelirdi ve D9'un "ayni LP, isaret degisimi" iddiasi bosa duserdi.
        Ayrica tetik gunu normal tabanin altina inerse temizlik penceresi ile
        teklif edilebilirlik penceresi hic kesismez ve rejim OLU olur - bu da
        kod incelemesine degil config yuklemesine baglanir.
        """
        t = self.tahsis.temizlik
        normal = self.politika.kisit.asgari_kalan_raf_omru_gun
        if t.asgari_kalan_raf_omru_gun > normal:
            raise ValueError(
                f"temizlik.asgari_kalan_raf_omru_gun ({t.asgari_kalan_raf_omru_gun}) "
                f"normal tabandan ({normal}) buyuk: temizlik rejimi gevsetme "
                f"olmali, sikilastirma degil (D9)")
        if t.tetik_gun <= t.asgari_kalan_raf_omru_gun:
            raise ValueError(
                f"temizlik.tetik_gun ({t.tetik_gun}) temizlik tabaninin "
                f"({t.asgari_kalan_raf_omru_gun}) altinda ya da esit: temizlik "
                f"penceresi bos, rejim olu")

    def _m4_zaman_kilidi(self) -> None:
        """M4 mekanik kilidi: egitim origin'leri olcum origin'lerine giremez.

        Uplift ogreticisinin egitim penceresi ile politikanin olculdugu
        origin'ler ortusurse "modelin gordugu hafta uzerinde politika
        karsilastirmasi" yapilmis olur. Bu, kod incelemesine degil config
        yuklemesine baglanir: ortusen ayar kosuyu dusurur.
        """
        d = self.politika.aday.degerlendirme
        e = self.uplift.egitim
        W = self.profil.hafta_sayisi
        son = W - 1 - d.ufuk_hafta
        olcum = [son - i * d.ufuk_hafta for i in range(d.origin_sayisi)]
        ilk_olcum = min(olcum)
        son_egitim = e.ilk_origin_hafta + (e.azami_origin_sayisi - 1) * e.origin_araligi_hafta
        sinir = ilk_olcum - e.sinir_tamponu_hafta
        if e.ilk_origin_hafta > sinir:
            raise ValueError(
                f"uplift.egitim.ilk_origin_hafta ({e.ilk_origin_hafta}) olcum "
                f"penceresine tasiyor: ilk olcum origin'i {ilk_olcum}, tampon "
                f"{e.sinir_tamponu_hafta} -> egitim en gec {sinir}. hafta_sayisi={W}")
        # Uyari degil hata degil: egitim origin sayisi tamponla kirpilir
        # (models/uplift.py egitim_originleri). Burada yalnizca hic origin
        # kalmadigi durum yakalanir.
        if son_egitim < e.ilk_origin_hafta:
            raise ValueError("uplift.egitim penceresi bos")

    # policy/bandit.py::Q_ALT'in kopyasi. core/ politikadan import EDEMEZ
    # (katman yonu: policy -> core), bu yuzden sabit burada tekrarlanir ve
    # tests/test_ope.py ikisinin esitligini sinar -- sim/response.py ile
    # policy/scorer.py arasindaki TEKLIF_YOK ikizinde kullanilan disiplinin
    # aynisi.
    KAYIT_Q_ALT: ClassVar[float] = 0.05

    # Dunyayi belirleyen bolumler. feature/tukenme bunlarin disindadir:
    # bir feature knob'u degistiginde uretilen dunya AYNI kalir ve iki kosu
    # satir bazinda eslesmis olarak karsilastirilabilir (experiments/compare.py).
    DUNYA_BOLUMLERI: ClassVar[tuple[str, ...]] = (
        "profil", "urun", "eczane", "sim", "olay", "lot")

    def _hash(self, bolumler: tuple[str, ...] | None = None) -> str:
        govde = self.model_dump(mode="json")
        if bolumler is not None:
            govde = {k: govde[k] for k in bolumler}
        ham = json.dumps(govde, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(ham).hexdigest()[:16]

    def hash(self) -> str:
        """Tum config'in hash'i. Iki kosunun ayni ayarla kostugunun kaniti."""
        return self._hash()

    def dunya_hash(self) -> str:
        """Yalnizca simulasyonu belirleyen bolumlerin hash'i."""
        return self._hash(self.DUNYA_BOLUMLERI)


def _yukle(yol: Path) -> dict:
    with yol.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(profil: str, config_dir: Path | None = None,
                 gecersiz_kilma: dict[str, object] | None = None) -> Config:
    """Profil adiyla tum config'i yukler.

    gecersiz_kilma: nokta yollu knob -> deger ("talep.dagilim.sifir_sisirme": 0.5).
    Sweep kosucusunun (M2) kullanacagi giris noktasi; M1'de dogrulama scripti
    karsilastirmali kosu icin kullanir.
    """
    cd = config_dir or CONFIG_DIR
    ham = {
        "profil": _yukle(cd / "profiles" / f"{profil}.yaml")["profil"],
        "urun": _yukle(cd / "products.yaml"),
        "eczane": _yukle(cd / "pharmacies.yaml"),
        "sim": _yukle(cd / "sim.yaml"),
        "olay": _yukle(cd / "events.yaml"),
        "lot": _yukle(cd / "lots.yaml"),
        "feature": _yukle(cd / "features.yaml"),
        "tukenme": _yukle(cd / "depletion.yaml"),
        "politika": _yukle(cd / "policy.yaml"),
        "tepki": _yukle(cd / "response.yaml")["tepki"],
        "uplift": _yukle(cd / "uplift.yaml")["uplift"],
        "tahsis": _yukle(cd / "allocation.yaml")["tahsis"],
        "ope": _yukle(cd / "ope.yaml")["ope"],
        "senaryo": _yukle(cd / "scenarios.yaml")["senaryo"],
        "ajan": _yukle(cd / "agent.yaml")["ajan"],
        "harness": _yukle(cd / "harness.yaml")["harness"],
    }
    for yol, deger in (gecersiz_kilma or {}).items():
        _nokta_yaz(ham, yol, deger)
    return Config.model_validate(ham)


def _nokta_yaz(agac: dict, yol: str, deger: object) -> None:
    parcalar = yol.split(".")
    dugum = agac
    for p in parcalar[:-1]:
        if p not in dugum:
            raise KeyError(f"config yolu yok: {yol} (kirilan parca: {p})")
        dugum = dugum[p]
    if parcalar[-1] not in dugum:
        raise KeyError(f"config yolu yok: {yol} (kirilan parca: {parcalar[-1]})")
    dugum[parcalar[-1]] = deger
