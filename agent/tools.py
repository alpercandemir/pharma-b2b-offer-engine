"""LLM'in cagirabilecegi fonksiyonlar (tool use). D8'in mekanik siniri.

D8: "LLM karar noktasinda YOKTUR; orkestrasyon, aciklama ve senaryo
yorumundadir."

BU DOSYA SALT OKURDUR ve bu bir yorum degil, kontrol edilebilir bir olgu:

  - policy/, models/, sim/, eval/ modullerinden HICBIRINI import etmez.
  - Karar ureten hicbir fonksiyonu (kisit_uygula, scorer.sec, tahsis_et,
    senaryolari_kos) cagirmaz; adlari bu dosyada gecmez.
  - Aldigi tek girdi, senaryo katmaninin ONCEDEN uretmis oldugu duz veri
    (`AjanBaglami`). Bir arac cagrisi hicbir sayiyi yeniden hesaplamaz,
    yalnizca hazir tablodan okur.

`scripts/verify_m7.py::kontrol_d8_siniri` bu uc maddeyi kaynak taramasiyla
siniyor. Model ne kadar ikna edici konusursa konussun teklif listesini
degistiremez; degistirebilseydi D8 ihlal edilmis olurdu.

TIP IMPORTU BILEREK YOK. `AjanBaglami`yi kuran `baglam_kur` senaryo
kosusunu tipsiz alir (`kosu`); `from agent.scenario import ...` yazmak
dosyayi policy zincirine baglardi ve yukaridaki taramanin anlami zayiflardi.
Bedeli tip ipucu kaybi, kazanci mekanik bir sinir.

SAYI DEFTERI. Her arac cagrisinin dondurdugu her sayi `SayiDefteri`ne
senaryo etiketiyle yazilir. Harness'in "sayi uydurma" denetcisi tam olarak
bu defteri kullanir: brifingdeki bir sayi ya bir aracin dondurdugu sayidir
ya da UYDURMADIR. Ucuncu ihtimal yok.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Senaryoya bagli olmayan olgular bu etiketle deftere girer.
GENEL = "genel"

# Arac ciktilarinin varsayilan cozunurlugu; bicim sabiti, knob degil.
# `agent/scenario.py::ORAN_ONDALIK` ile ayni deger ve ayni gerekce: bu
# sayilar sayi defterine bu haliyle girer, daha kaba yuvarlama iki rejimin
# oranlarini ayni degere cokertip `senaryo_karismasi` denetimini korlestirir.
ORAN_ONDALIK = 4

# Veto sebeplerinin insan diline cevrilmis dayanaklari. policy/constraints.py
# tablosunun birebir karsiligi; sebep adlari oradan gelir. Metin uretmek bu
# katmanin isi, sebebi URETMEK degil.
VETO_ACIKLAMASI: dict[str, str] = {
    "recete_rengi": "Kirmizi/yesil recete: promosyon ve kampanya yasak, "
                    "kontrollu dagitim. Miad ya da stok baskisi bu vetoyu asmaz.",
    "tedarik_guclugu": "TITCK tedarik guclugu listesinde: bu bir kampanya "
                       "degil tahsis problemi.",
    "emilim_tavani": "Teklif adedi eczanenin haftalik emebilecegi ihtiyaci asiyor.",
    "depo_stogu": "Depoda yeterli adet yok.",
    "raf_omru": "Eldeki lotlarin kalan raf omru politikanin asgari esiginin altinda.",
    "lot_yetersiz": "Yeterli adet var ama tek lotta yok; teklif satiri tek lot "
                    "referansi tasimak zorunda.",
    "soguk_zincir_min": "Soguk zincir asgari siparis adedine cikarilamiyor.",
    "kredi_limiti": "Eczanenin acik bakiyesi + teklif tutari DBS limitini asiyor.",
}


# --------------------------------------------------------------------------
# sayi defteri
# --------------------------------------------------------------------------
# Metin icine gomulmus sayilari da yakalar ("10+1", "%15" gibi ifadeler
# arac ciktisinda string olarak donuyor ve brifingde aynen kullanilabilmeli).
_SAYI = re.compile(r"-?\d+(?:[.,]\d+)?")


@dataclass
class SayiDefteri:
    """Modele FIILEN verilmis sayilar, senaryo etiketiyle.

    `genel` etiketi senaryodan bagimsiz olgulari tasir (eczane profili, lot
    miadi). Rejim etiketli sayilar ise yalnizca o rejimin bolumunde mesrudur;
    harness'in senaryo karistirma denetcisi bu ayrima dayanir.
    """

    kayitlar: dict[str, set[float]] = field(default_factory=dict)

    def ekle(self, etiket: str, deger: Any) -> None:
        for sayi in _sayilari_cikar(deger):
            self.kayitlar.setdefault(etiket, set()).add(sayi)

    def etiketler(self) -> list[str]:
        return sorted(self.kayitlar)

    def kume(self, etiket: str) -> set[float]:
        return self.kayitlar.get(etiket, set())

    def tum(self) -> set[float]:
        return {v for k in self.kayitlar.values() for v in k}

    def izinli(self, etiket: str) -> set[float]:
        """Bir bolumde mesru sayilar: o bolumun kendi sayilari + genel."""
        return self.kume(etiket) | self.kume(GENEL)

    def toplam_sayi(self) -> int:
        return len(self.tum())


def _sayilari_cikar(deger: Any) -> list[float]:
    if isinstance(deger, bool):
        return []
    if isinstance(deger, (int, float)):
        return [float(deger)]
    if isinstance(deger, str):
        return [float(p.replace(",", ".")) for p in _SAYI.findall(deger)]
    if isinstance(deger, dict):
        return [s for v in deger.values() for s in _sayilari_cikar(v)]
    if isinstance(deger, (list, tuple, set)):
        return [s for v in deger for s in _sayilari_cikar(v)]
    return []


# --------------------------------------------------------------------------
# baglam: senaryo katmaninin duz veriye cevrilmis ciktisi
# --------------------------------------------------------------------------
@dataclass
class AjanBaglami:
    """Araclarin gorebilecegi HER SEY. Disinda bir kaynak yok.

    Butun alanlar duz Python veri yapisi: model ne isterse istesin burada
    olmayan bir olguya erisemez ve harness "var olmayan varlik" denetimini
    tam olarak bu kumelere karsi yapar.
    """

    origin: int
    politika: str
    taban_rejim: str
    rejim_parametreleri: dict[str, dict]     # rejim -> parametre sozlugu
    rejim_ozetleri: dict[str, dict]          # rejim -> ozet sozlugu
    rejim_farklari: dict[str, dict]          # rejim -> tabana gore fark
    eczaneler: dict[str, dict]               # eczane_id -> profil
    urunler: dict[str, dict]                 # sku_id -> urun
    lotlar: dict[str, dict]                  # lot_id -> lot
    # (rejim, eczane_id) -> teklif satirlari (artimsal marja gore sirali)
    teklifler: dict[tuple[str, str], list[dict]]
    # (rejim, eczane_id) -> vetolanmis satirlar
    vetolar: dict[tuple[str, str], list[dict]]
    # (rejim, eczane_id, sku_id) -> kol ekonomisi tablosu
    kol_ekonomisi: dict[tuple[str, str, str], list[dict]]

    @property
    def rejimler(self) -> list[str]:
        return list(self.rejim_parametreleri)

    def eczane_var(self, eczane_id: str) -> bool:
        return eczane_id in self.eczaneler

    def teklif_veren_eczaneler(self) -> list[str]:
        return sorted({e for (_, e), satirlar in self.teklifler.items() if satirlar})


def _yuvarla(x: float, basamak: int = ORAN_ONDALIK) -> float:
    return float(round(float(x), basamak))


def baglam_kur(cfg, kosu, kol_sayisi: int | None = None) -> AjanBaglami:
    """`agent.scenario.SenaryoKosusu` -> duz veri.

    `kosu` bilerek TIPSIZ (dosya basligi): bu modul senaryo katmanina
    import baglantisi kurmaz, yalnizca ciktisini okur.
    """
    import numpy as np      # yalnizca dizi okumak icin; karar uretmez

    kol_sayisi = kol_sayisi or cfg.ajan.kol_ekonomisi_kol_sayisi
    eczaneler, urunler, lotlar = {}, {}, {}
    teklifler: dict[tuple[str, str], list[dict]] = {}
    vetolar: dict[tuple[str, str], list[dict]] = {}
    kol_ekonomisi: dict[tuple[str, str, str], list[dict]] = {}
    rejim_par, rejim_ozet, rejim_fark = {}, {}, {}

    for satir in kosu.eczane_master:
        eczaneler[satir["eczane_id"]] = satir
    for satir in kosu.urun_master:
        urunler[satir["sku_id"]] = satir
    for satir in kosu.lot_master:
        lotlar[satir["lot_id"]] = satir

    for ad, s in kosu.sonuclar.items():
        r = s.rejim
        rejim_par[ad] = {
            "rejim": ad,
            "aciklama": r.aciklama,
            "guncelleme_beklentisi_hafta": r.guncelleme_beklentisi_hafta,
            "referans_kur_artisi": r.referans_kur_artisi,
            "fiyat_gecis_katsayisi": r.fiyat_gecis_katsayisi,
            "antisipasyon_talep_carpani": r.antisipasyon_talep_carpani,
            "fonlama_orani_carpani": r.fonlama_orani_carpani,
            "taban_mi": ad == kosu.taban_ad,
        }
        o = kosu.ozetler[ad]
        rejim_ozet[ad] = {"rejim": ad} | {k: _duz(v) for k, v in vars(o).items()
                                          if k not in ("ad", "veto_dagilimi")}
        rejim_ozet[ad]["veto_dagilimi"] = dict(o.veto_dagilimi)
        if ad in kosu.farklar:
            f = kosu.farklar[ad]
            rejim_fark[ad] = {k: _duz(v) for k, v in vars(f).items() if k != "ad"}

        aday = s.aday
        if aday.height == 0:
            continue
        idx = np.arange(aday.height)
        kol = s.kol
        satirlar = _teklif_satirlari(s, aday, idx, kol, np)
        for satir in satirlar:
            teklifler.setdefault((ad, satir["eczane_id"]), []).append(satir)
            kol_ekonomisi[(ad, satir["eczane_id"], satir["sku_id"])] = _kol_tablosu(
                s, aday, satir["_satir"], kol_sayisi, np)
        for satir in _veto_satirlari(s):
            vetolar.setdefault((ad, satir["eczane_id"]), []).append(satir)

    for anahtar, liste in teklifler.items():
        liste.sort(key=lambda d: -d["artimsal_beklenen_marj_tl"])
        for satir in liste:
            satir.pop("_satir", None)
    for liste in vetolar.values():
        liste.sort(key=lambda d: -d["skor"])

    return AjanBaglami(
        origin=kosu.t, politika=kosu.politika, taban_rejim=kosu.taban_ad,
        rejim_parametreleri=rejim_par, rejim_ozetleri=rejim_ozet,
        rejim_farklari=rejim_fark, eczaneler=eczaneler, urunler=urunler,
        lotlar=lotlar, teklifler=teklifler, vetolar=vetolar,
        kol_ekonomisi=kol_ekonomisi)


def _duz(v):
    return v if isinstance(v, (int, float, str, bool)) else str(v)


def _teklif_satirlari(s, aday, idx, kol, np) -> list[dict]:
    mf = s.mat.uzay.mf[kol]
    vade = s.mat.uzay.vade[kol]
    adet = s.mat.adet[idx, kol]
    bedava = s.mat.bedava[idx, kol]
    marj = s.mat.marj[idx, kol]
    p = s.p_x[idx, kol]
    p0 = s.p_x[:, 0]
    marj0 = s.mat.marj[:, 0]
    artimsal = (p * marj - p0 * marj0) * s.carpan
    teklif = kol != 0
    eczane_id = aday["eczane_id"].to_list()
    sku_id = aday["sku_id"].to_list()
    lot_id = aday["lot_id"].to_list()
    kalan = aday["lot_kalan_gun"].to_numpy()
    hiz = aday["hiz_tahmini"].to_numpy()
    mf_izinli = aday["mf_izinli"].to_numpy()
    soguk = aday["soguk_zincir"].to_numpy()
    sgk = aday["sgk_geri_odeme"].to_numpy()

    cikti = []
    for i in np.flatnonzero(teklif):
        cikti.append({
            "_satir": int(i),
            "rejim": s.rejim.ad,
            "eczane_id": eczane_id[i],
            "sku_id": sku_id[i],
            "lot_id": lot_id[i],
            "mf_orani": _yuvarla(mf[i]),
            "mf_ifadesi": _mf_ifadesi(float(mf[i]), float(adet[i]), float(bedava[i])),
            "vade_gun": int(vade[i]),
            "adet": int(adet[i]),
            "bedava_adet": int(bedava[i]),
            "kabul_olasiligi": _yuvarla(p[i]),
            "teklifsiz_kabul_olasiligi": _yuvarla(p0[i]),
            "kabul_sartiyla_brut_marj_tl": _yuvarla(marj[i], 2),
            "artimsal_beklenen_marj_tl": _yuvarla(artimsal[i], 2),
            "lot_kalan_gun": _yuvarla(kalan[i], 1),
            "lot_bekleyebilir": bool(s.bekleyebilir[i]),
            "erteleme_tl_adet": _yuvarla(s.erteleme[i], 2),
            "haftalik_hiz_tahmini": _yuvarla(hiz[i], 2),
            "mf_kanali_acik": bool(mf_izinli[i]),
            "soguk_zincir": bool(soguk[i]),
            "sgk_geri_odeme": bool(sgk[i]),
        })
    return cikti


def _mf_ifadesi(mf: float, adet: float, bedava: float) -> str:
    """Sahanin dilindeki karsilik: "10+1". MF yoksa acikca yazilir."""
    if mf <= 0.0 or bedava < 1.0:
        return "MF yok"
    return f"{int(adet)}+{int(bedava)}"


def _kol_tablosu(s, aday, i: int, kol_sayisi: int, np) -> list[dict]:
    """Tek satirin kol ekonomisi: her kolun marji, kabul olasiligi, artimsali.

    reports/m6.md 6.2'nin kol tablosunun satir bazli hali. M6 bu tabloyu
    "LLM katmaninin uretecegi en somut aciklama" diye isaret etmisti; burada
    o tablo bir arac ciktisi haline geldi ve hicbir sayisi uydurulamaz.
    """
    izinli = s.mat.izinli[i]
    p, marj = s.p_x[i], s.mat.marj[i]
    artimsal = (p * marj - p[0] * marj[0]) * s.carpan
    sira = [0] + sorted((a for a in range(1, s.mat.uzay.A) if izinli[a]),
                        key=lambda a: -artimsal[a])
    cikti = []
    for a in sira[:kol_sayisi]:
        cikti.append({
            "kol": s.mat.uzay.adlar[a],
            "mf_orani": _yuvarla(s.mat.uzay.mf[a]),
            "vade_gun": int(s.mat.uzay.vade[a]),
            "adet": int(s.mat.adet[i, a]),
            "bedava_adet": int(s.mat.bedava[i, a]),
            "kabul_olasiligi": _yuvarla(p[a]),
            "kabul_sartiyla_brut_marj_tl": _yuvarla(marj[a], 2),
            "artimsal_beklenen_marj_tl": _yuvarla(artimsal[a], 2),
            "izinli": bool(izinli[a]),
            "secilen": bool(a == int(s.kol[i])),
        })
    return cikti


def _veto_satirlari(s) -> list[dict]:
    vetolu = s.tumu.filter(s.tumu["vetolu"])
    cikti = []
    for satir in vetolu.iter_rows(named=True):
        sebep = satir["veto_sebebi"]
        cikti.append({
            "rejim": s.rejim.ad,
            "eczane_id": satir["eczane_id"],
            "sku_id": satir["sku_id"],
            "veto_sebebi": sebep,
            "dayanak": VETO_ACIKLAMASI.get(sebep, "bilinmeyen sebep"),
            "skor": _yuvarla(satir["skor"], ORAN_ONDALIK),
            "istenen_adet": int(satir["teklif_adedi"]),
        })
    return cikti


# --------------------------------------------------------------------------
# arac semalari (Anthropic tool use)
# --------------------------------------------------------------------------
def _sema(ad: str, aciklama: str, alanlar: dict[str, str],
          zorunlu: list[str]) -> dict:
    return {
        "name": ad,
        "description": aciklama,
        "input_schema": {
            "type": "object",
            "properties": {k: {"type": "string", "description": v}
                           for k, v in alanlar.items()},
            "required": zorunlu,
        },
    }


ARAC_SEMALARI: list[dict] = [
    _sema("eczane_profili",
          "Eczanenin gozlemlenebilir profili: konum, olcek, kredi durumu. "
          "Senaryodan bagimsizdir.",
          {"eczane_id": "ECZ0000 bicimli eczane kimligi"}, ["eczane_id"]),
    _sema("teklif_listesi",
          "Politikanin bu eczane icin bu rejim altinda URETTIGI teklifler. "
          "Liste burada uretilmez, hazir tablodan okunur.",
          {"eczane_id": "eczane kimligi", "rejim": "rejim adi (baz/yuksek/sok)"},
          ["eczane_id", "rejim"]),
    _sema("kisit_gerekcesi",
          "Bu eczanede aday havuzuna girip KISIT KATMANI tarafindan "
          "vetolanmis satirlar ve veto dayanaklari.",
          {"eczane_id": "eczane kimligi", "rejim": "rejim adi"},
          ["eczane_id", "rejim"]),
    _sema("kol_ekonomisi",
          "Tek bir (eczane, urun) satirinda her aksiyon kolunun kabul "
          "olasiligi, kabul sartiyla marji ve artimsal beklenen marji.",
          {"eczane_id": "eczane kimligi", "sku_id": "SKU0000 bicimli urun kimligi",
           "rejim": "rejim adi"}, ["eczane_id", "sku_id", "rejim"]),
    _sema("lot_bilgisi",
          "Bir lotun urunu, kalan raf omru ve rejim basina 'guncellemeyi "
          "bekleyebilir mi' durumu.",
          {"lot_id": "LOT000000 bicimli lot kimligi"}, ["lot_id"]),
    _sema("senaryo_karsilastir",
          "Butun rejimlerin parametreleri, ozet metrikleri ve tabana gore "
          "farklari. Eczane verilirse o eczanenin rejim basina teklif ozeti "
          "de doner.",
          {"eczane_id": "istege bagli eczane kimligi"}, []),
]

ARAC_ADLARI: tuple[str, ...] = tuple(s["name"] for s in ARAC_SEMALARI)


# --------------------------------------------------------------------------
# arac govdeleri
# --------------------------------------------------------------------------
def _yok(ad: str, kimlik: str) -> dict:
    """Var olmayan varlik icin ACIK hata.

    Sessizce bos liste donmek modelin "demek ki yok" diye uydurmasina davet
    olurdu; arac bunun yerine kimligin bilinmedigini soyler ve gecerli
    orneklerden birkacini verir.
    """
    return {"hata": f"{ad} bulunamadi: {kimlik}"}


def eczane_profili(b: AjanBaglami, eczane_id: str) -> dict:
    if eczane_id not in b.eczaneler:
        return _yok("eczane", eczane_id)
    return dict(b.eczaneler[eczane_id])


def teklif_listesi(b: AjanBaglami, eczane_id: str, rejim: str) -> dict:
    if eczane_id not in b.eczaneler:
        return _yok("eczane", eczane_id)
    if rejim not in b.rejim_parametreleri:
        return _yok("rejim", rejim)
    satirlar = b.teklifler.get((rejim, eczane_id), [])
    return {"rejim": rejim, "eczane_id": eczane_id, "origin_haftasi": b.origin,
            "politika": b.politika, "teklif_sayisi": len(satirlar),
            "teklifler": satirlar}


def kisit_gerekcesi(b: AjanBaglami, eczane_id: str, rejim: str) -> dict:
    if eczane_id not in b.eczaneler:
        return _yok("eczane", eczane_id)
    if rejim not in b.rejim_parametreleri:
        return _yok("rejim", rejim)
    satirlar = b.vetolar.get((rejim, eczane_id), [])
    return {"rejim": rejim, "eczane_id": eczane_id,
            "vetolanan_satir_sayisi": len(satirlar), "vetolar": satirlar}


def kol_ekonomisi(b: AjanBaglami, eczane_id: str, sku_id: str, rejim: str) -> dict:
    anahtar = (rejim, eczane_id, sku_id)
    if anahtar not in b.kol_ekonomisi:
        return {"hata": f"(eczane={eczane_id}, sku={sku_id}, rejim={rejim}) "
                        f"icin teklif satiri yok; kol ekonomisi yalnizca "
                        f"teklif verilen satirlarda tanimli."}
    return {"rejim": rejim, "eczane_id": eczane_id, "sku_id": sku_id,
            "kollar": b.kol_ekonomisi[anahtar]}


def lot_bilgisi(b: AjanBaglami, lot_id: str) -> dict:
    if lot_id not in b.lotlar:
        return _yok("lot", lot_id)
    kayit = dict(b.lotlar[lot_id])
    bekleme = {}
    for rejim in b.rejimler:
        satir = next((s for (r, _), liste in b.teklifler.items() if r == rejim
                      for s in liste if s["lot_id"] == lot_id), None)
        if satir is not None:
            bekleme[rejim] = satir["lot_bekleyebilir"]
    kayit["rejimde_bekleyebilir"] = bekleme
    return kayit


def senaryo_karsilastir(b: AjanBaglami, eczane_id: str | None = None) -> dict:
    cikti: dict[str, Any] = {
        "origin_haftasi": b.origin, "politika": b.politika,
        "taban_rejim": b.taban_rejim,
        "parametreler": b.rejim_parametreleri,
        "ozetler": b.rejim_ozetleri,
        "tabana_gore_fark": b.rejim_farklari,
    }
    if eczane_id:
        if eczane_id not in b.eczaneler:
            return _yok("eczane", eczane_id)
        cikti["eczane_id"] = eczane_id
        cikti["eczane_ozeti"] = {
            rejim: {
                "teklif_sayisi": len(b.teklifler.get((rejim, eczane_id), [])),
                "vetolanan_satir": len(b.vetolar.get((rejim, eczane_id), [])),
                "toplam_artimsal_marj_tl": _yuvarla(sum(
                    s["artimsal_beklenen_marj_tl"]
                    for s in b.teklifler.get((rejim, eczane_id), [])), 2),
            } for rejim in b.rejimler}
    return cikti


GOVDELER = {
    "eczane_profili": eczane_profili,
    "teklif_listesi": teklif_listesi,
    "kisit_gerekcesi": kisit_gerekcesi,
    "kol_ekonomisi": kol_ekonomisi,
    "lot_bilgisi": lot_bilgisi,
    "senaryo_karsilastir": senaryo_karsilastir,
}


def cagir(b: AjanBaglami, ad: str, girdi: dict, defter: SayiDefteri) -> dict:
    """Tek arac cagrisi. Sonucun her sayisi deftere islenir.

    Etiket, cagrinin `rejim` argumanidir; yoksa GENEL. Bu etiket harness'in
    senaryo karistirma denetcisinin dayanagi: `sok` bolumunde yalnizca `sok`
    ve `genel` sayilari mesrudur.
    """
    if ad not in GOVDELER:
        sonuc = {"hata": f"bilinmeyen arac: {ad}. Gecerli: {list(GOVDELER)}"}
        defter.ekle(GENEL, sonuc)
        return sonuc
    izinli = _girdiyi_suz(ad, girdi)
    sonuc = GOVDELER[ad](b, **izinli)
    defter.ekle(str(girdi.get("rejim") or GENEL), sonuc)
    return sonuc


def _girdiyi_suz(ad: str, girdi: dict) -> dict:
    """Semada olmayan anahtarlari eler; model uydurma parametre gonderirse
    TypeError yerine sessizce yoksayilir ve arac yine calisir."""
    sema = next(s for s in ARAC_SEMALARI if s["name"] == ad)
    alanlar = set(sema["input_schema"]["properties"])
    return {k: v for k, v in girdi.items() if k in alanlar}
