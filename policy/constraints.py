"""Hard veto katmani. D6: "Kisit katmaninin ML skoru uzerinde veto yetkisi vardir".

Bu dosyanin tek isi HAYIR demek. Skoru degistirmez, agirliklandirmaz, ceza
puani vermez. Bir satir ya listede kalir ya cikar; arada bir sey yoktur.
Aday uretimi (policy/candidates.py) bu katmandan habersiz calisir - vetonun
bedeli ancak boyle olculebilir (reports/m3.md 4).

Veto sebepleri ve dayanaklari:

| sebep             | dayanak |
|-------------------|---------|
| `recete_rengi`    | Kirmizi/yesil recete: promosyon yasak, kontrollu dagitim (SPEC 2.1) |
| `tedarik_guclugu` | TITCK listesindeki urun kampanya degil TAHSIS problemi (SPEC 2.1) |
| `emilim_tavani`   | Teklif adedi eczanenin emebilecegi haftalik ihtiyaci asiyor |
| `raf_omru`        | Stok var ama tamami politikanin asgari raf omrunun altinda (SPEC 2.5) |
| `depo_stogu`      | Depoda yeterli adet yok |
| `lot_yetersiz`    | Yeterli adet var ama tek lotta yok (teklif satiri tek lot referansi tasir) |
| `soguk_zincir_min`| Min siparis adedine cikarilamiyor (`soguk_zincir_min_altinda: veto`) |
| `kredi_limiti`    | Eczanenin acik bakiyesi + teklif tutari DBS limitini asiyor |

MIAD BASKISI VETOYU ASMAZ. SPEC 2.5 acik: "promosyon_serbest = false olan
urunlerde temizlik kampanyasi YAPILAMAZ - miad baskisi bu vetoyu asmaz."
Bu dosyada miad baskisina bakan tek satir yok; baski yalnizca aday
uretiminde SIRALAMAYI oynatir. Testi tests/test_constraints.py'de.

VETO DEGIL, KANAL KISITI: SGK geri odeme kapsamindaki urunde iskonto
serbestisi kisitli (SPEC 2.5). Teklif listede KALIR, `mf_izinli=false` ile
isaretlenir; aksiyonu M4 secer. Aksiyon secimi bu katmanda YOK.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from core.config import Config
from policy.candidates import AdayDunyasi, OriginGorunumu

# Veto sebepleri, UYGULANMA sirasiyla. Bir satira ilk baglayan sebep yazilir;
# `veto_maskesi` kolonu hangi sebeplerin AYNI ANDA bagladigini da tasir.
VETO_SEBEPLERI: tuple[str, ...] = (
    "recete_rengi",
    "tedarik_guclugu",
    "emilim_tavani",
    "depo_stogu",
    "raf_omru",
    "lot_yetersiz",
    "soguk_zincir_min",
    "kredi_limiti",
)

# Lot bulunamadigi satirlarda kalan raf omru kolonuna yazilan deger.
# NaN degil: "lot yok" ile "lotun raf omru sifir" farkli seyler ve tabloda
# ayirt edilebilmeli.
LOT_YOK_GUN = -1.0


def _regulasyon_vetolari(dunya: AdayDunyasi, cfg: Config,
                         s: np.ndarray) -> dict[str, np.ndarray]:
    k = cfg.politika.kisit
    urun = dunya.urunler
    renk = urun["recete_rengi"].to_numpy()[s]
    return {
        "recete_rengi": np.isin(renk, list(k.recete_rengi_vetosu)),
        "tedarik_guclugu": (urun["titck_tedarik_guclugu"].to_numpy()[s]
                            & k.tedarik_guclugu_veto),
    }


def _lot_sec(gor: OriginGorunumu, sku_idx: int, adet: float, esik_gun: float,
             yeterlilik: float) -> tuple[str | None, float, str | None]:
    """FEFO sirasinda, raf omru esigini gecen ve adedi tek basina karsilayan
    ilk lot. (lot_id, kalan_gun, veto_sebebi) doner.

    Tek lot sarti bilincli bir BASITLESTIRME: SPEC 2.5 "teklif satir bazinda
    lot referansi tasir" diyor; cok lotlu bolme ve lotun eczaneler arasinda
    paylastirilmasi M5'in tahsis problemidir (reports/m3.md 8).
    """
    gerekli = adet * yeterlilik
    lotlar = gor.lotlar.get(sku_idx, ())
    if not lotlar:
        return None, LOT_YOK_GUN, "depo_stogu"
    if sum(l.kalan_adet for l in lotlar) < gerekli:
        return None, LOT_YOK_GUN, "depo_stogu"
    uygun = [l for l in lotlar if l.kalan_gun >= esik_gun]
    if not uygun:
        return None, LOT_YOK_GUN, "raf_omru"
    for lot in uygun:                                  # FEFO sirali
        if lot.kalan_adet >= gerekli:
            return lot.lot_id, lot.kalan_gun, None
    if sum(l.kalan_adet for l in uygun) < gerekli:
        return None, LOT_YOK_GUN, "raf_omru"
    return None, LOT_YOK_GUN, "lot_yetersiz"


def kisit_uygula(dunya: AdayDunyasi, cfg: Config, gor: OriginGorunumu,
                 havuz: pl.DataFrame) -> pl.DataFrame:
    """Aday havuzunu vetolar, teklif adedini ve lotu kesinlestirir.

    Havuzdaki HER satir ciktida da vardir - vetolananlar silinmez,
    `vetolu=true` ile isaretlenir. Vetonun bedeli ancak silinmemis bir tablo
    uzerinde olculebilir.
    """
    k = cfg.politika.kisit
    a = cfg.politika.aday
    if havuz.height == 0:
        return _bos_cikti(havuz)

    p = havuz["eczane_idx"].to_numpy()
    s = havuz["sku_idx"].to_numpy()
    urun, eczane = dunya.urunler, dunya.eczaneler
    soguk = urun["soguk_zincir"].to_numpy()[s]
    sgk = urun["sgk_geri_odeme"].to_numpy()[s]
    promosyon_serbest = urun["promosyon_serbest"].to_numpy()[s]
    dsf = dunya.dsf[s]
    hiz = havuz["hiz_tahmini"].to_numpy() * a.hiz_telafi_katsayisi

    veto = {ad: np.zeros(havuz.height, dtype=bool) for ad in VETO_SEBEPLERI}
    veto.update(_regulasyon_vetolari(dunya, cfg, s))

    # --- adet: talebe gore baslar, soguk zincir minimumu YUKSELTEBILIR ---
    adet = havuz["teklif_adedi"].to_numpy().astype(float)
    min_adet = np.where(soguk, k.soguk_zincir_min_siparis_adedi, 1.0)
    altinda = adet < min_adet
    yukseltildi = altinda & soguk
    if k.soguk_zincir_min_altinda == "yukselt":
        adet = np.where(altinda, min_adet, adet)
    else:
        veto["soguk_zincir_min"] |= yukseltildi

    # --- emilim tavani: yukseltilmis adet de bu tavana tabidir ---
    veto["emilim_tavani"] = adet > hiz * k.azami_kapsama_hafta

    # --- lot secimi: raf omru + depo stogu, KESINLESMIS adet uzerinden ---
    esik = k.asgari_kalan_raf_omru_gun * np.where(
        soguk, k.soguk_zincir_raf_omru_carpani, 1.0)
    lot_id: list[str | None] = [None] * havuz.height
    lot_gun = np.full(havuz.height, LOT_YOK_GUN)
    onceki = np.zeros(havuz.height, dtype=bool)
    for ad in ("recete_rengi", "tedarik_guclugu", "emilim_tavani", "soguk_zincir_min"):
        onceki |= veto[ad]
    for i in np.flatnonzero(~onceki):
        secilen, gun, sebep = _lot_sec(gor, int(s[i]), float(adet[i]), float(esik[i]),
                                       k.depo_stok_yeterlilik_carpani)
        lot_id[i], lot_gun[i] = secilen, gun
        if sebep is not None:
            veto[sebep][i] = True

    tutar = adet * dsf
    kalan_veto = onceki | veto["depo_stogu"] | veto["raf_omru"] | veto["lot_yetersiz"]

    # --- kredi limiti: SATIR degil PORTFOY kisiti ---
    veto["kredi_limiti"] = _kredi_vetosu(
        dunya, cfg, p, tutar, havuz["skor"].to_numpy(), ~kalan_veto, gor)

    vetolu = np.zeros(havuz.height, dtype=bool)
    sebep = np.full(havuz.height, "", dtype=object)
    maske = np.zeros(havuz.height, dtype=np.int64)
    for bit, ad in enumerate(VETO_SEBEPLERI):
        maske |= veto[ad].astype(np.int64) << bit
        yeni = veto[ad] & ~vetolu
        sebep[yeni] = ad
        vetolu |= veto[ad]

    # --- frekans tavani: veto DEGIL, budama. Veto'dan sonra uygulanir ---
    listede = _frekans_tavani(p, havuz["skor"].to_numpy(), ~vetolu,
                              k.eczane_haftalik_teklif_tavani)

    # --- kanal kisiti (veto degil): SGK kapsaminda MF kapali, vade acik ---
    mf_kapali = sgk.astype(bool) & (not k.sgk_kapsaminda_mf_serbest)
    mf_izinli = ~veto["recete_rengi"] & ~mf_kapali

    return havuz.with_columns([
        pl.Series("teklif_adedi", adet),
        pl.Series("teklif_tutari", tutar),
        pl.Series("lot_id", lot_id, dtype=pl.Utf8),
        pl.Series("lot_kalan_gun", lot_gun),
        pl.Series("soguk_zincir", soguk),
        pl.Series("sgk_geri_odeme", sgk),
        pl.Series("promosyon_serbest", promosyon_serbest),
        pl.Series("soguk_zincir_yukseltildi", yukseltildi),
        pl.Series("vetolu", vetolu),
        pl.Series("veto_sebebi", sebep.tolist(), dtype=pl.Utf8),
        pl.Series("veto_maskesi", maske),
        pl.Series("mf_izinli", mf_izinli),
        pl.Series("vade_izinli", ~vetolu),
        pl.Series("listede", listede),
    ])


def _kredi_vetosu(dunya: AdayDunyasi, cfg: Config, p: np.ndarray,
                  tutar: np.ndarray, skor: np.ndarray, aday: np.ndarray,
                  gor: OriginGorunumu) -> np.ndarray:
    """Eczane basina kumulatif limit kontrolu.

    Satir bazli kontrol yetmez: her teklif tek basina limitin altinda kalip
    toplamda limiti asabilir. Bu yuzden eczanenin adaylari skora gore
    sirali gezilir ve kumulatif tutar tavani asinca satir vetolanir.
    Asan satir ATLANIR (durulmaz): daha kucuk bir teklif hala sigabilir.

    Etkin tavan = DBS limiti x kullanim_tavani x (1 - vade_riski x ceza).
    Riskli eczanede limit daralir; `vade_riski_skoru` gozlemlenebilir.
    """
    k = cfg.politika.kisit
    dbs = dunya.eczaneler["dbs_limiti"].to_numpy().astype(float)
    risk = dunya.eczaneler["vade_riski_skoru"].to_numpy().astype(float)
    tavan = dbs * k.kredi_kullanim_tavani * (1.0 - risk * k.vade_riski_cezasi)
    kalan_limit = np.maximum(tavan - gor.acik_bakiye, 0.0)

    veto = np.zeros(p.size, dtype=bool)
    sira = np.lexsort((-skor, p))                 # once eczane, sonra skor (buyuk once)
    kullanilan = np.zeros(dunya.P)
    for i in sira:
        if not aday[i]:
            continue
        e = p[i]
        if kullanilan[e] + tutar[i] > kalan_limit[e]:
            veto[i] = True
            continue
        kullanilan[e] += tutar[i]
    return veto


def portfoy_kredi_vetosu(dunya: AdayDunyasi, cfg: Config, p: np.ndarray,
                         tutar: np.ndarray, oncelik: np.ndarray,
                         aday: np.ndarray, gor: OriginGorunumu) -> np.ndarray:
    """`_kredi_vetosu`nun disariya acik hali.

    M4 aksiyon secimden SONRA ayni kontrolu tekrar cagirir: koli yuvarlamasi
    tutari buyutuyor ve secilen satir kumesi degisiyor (policy/scorer.py).
    Kural ve sira ayni kalsin diye ayri bir uygulama yazilmadi.
    """
    return _kredi_vetosu(dunya, cfg, p, tutar, oncelik, aday, gor)


def _frekans_tavani(p: np.ndarray, skor: np.ndarray, uygun: np.ndarray,
                    tavan: int) -> np.ndarray:
    """Eczane basina en yuksek skorlu `tavan` kadar satir listede kalir."""
    listede = np.zeros(p.size, dtype=bool)
    sira = np.lexsort((-skor, p))
    sayac: dict[int, int] = {}
    for i in sira:
        if not uygun[i]:
            continue
        e = int(p[i])
        n = sayac.get(e, 0)
        if n < tavan:
            listede[i] = True
            sayac[e] = n + 1
    return listede


def _bos_cikti(havuz: pl.DataFrame) -> pl.DataFrame:
    """Bos havuz icin sema-uyumlu bos tablo (sweep'te bos origin dusmesin)."""
    return havuz.with_columns([
        pl.Series("teklif_tutari", [], dtype=pl.Float64),
        pl.Series("lot_id", [], dtype=pl.Utf8),
        pl.Series("lot_kalan_gun", [], dtype=pl.Float64),
        pl.Series("soguk_zincir", [], dtype=pl.Boolean),
        pl.Series("sgk_geri_odeme", [], dtype=pl.Boolean),
        pl.Series("promosyon_serbest", [], dtype=pl.Boolean),
        pl.Series("soguk_zincir_yukseltildi", [], dtype=pl.Boolean),
        pl.Series("vetolu", [], dtype=pl.Boolean),
        pl.Series("veto_sebebi", [], dtype=pl.Utf8),
        pl.Series("veto_maskesi", [], dtype=pl.Int64),
        pl.Series("mf_izinli", [], dtype=pl.Boolean),
        pl.Series("vade_izinli", [], dtype=pl.Boolean),
        pl.Series("listede", [], dtype=pl.Boolean),
    ])


def oneri_listesi(teklifler: pl.DataFrame) -> pl.DataFrame:
    """Fiilen sahaya cikan liste. "Oneri listesi" ifadesi bunu kasteder."""
    return teklifler.filter(pl.col("listede"))
