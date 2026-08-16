"""KAM / saha icin teklif brifingi. D8'in "aciklama" tarafi.

BU KATMAN KARAR VERMEZ. Teklif listesi, kol secimi ve veto M3-M6 tarafindan
uretilmis olarak gelir; burada yapilan tek sey onu okunur hale getirmektir.
Model bir sayiyi degistiremez, bir satiri listeye ekleyemez, bir vetoyu
kaldiramaz -- yapabildigi tek sey ANLATMAK, ve yanlis anlattigi harness
tarafindan deterministik olarak yakalanir.

BICIM SOZLESMESI. Brifing serbest metin DEGIL, denetlenebilir bir yapidir:

    # Teklif brifingi | <eczane_id> | hafta <origin> | politika <ad>
    ## Eczane
    ## Rejim: <ad>                (her rejim icin bir bolum, hepsi zorunlu)
    ## Kisit notlari
    ## Oneri kaydi
    ```oneri
    - senaryo: <ad>
      eczane_id: ...
      sku_id: ...
      lot_id: ...
      mf_orani: ...
      vade_gun: ...
      adet: ...
      bedava_adet: ...
    ```

Neden makine okunur blok: "kisit ihlali iddiasi" ve "senaryo karistirma"
duz metinden guvenilir bicimde cikarilamaz -- cikarilmaya calisilsaydi
denetci bir metin siniflandirici olurdu ve DETERMINISTIK olmazdi. Blok
sayesinde iddia kesin, karsilastirma tam.

Neden her rejim icin ayri bolum: D3 tek bir rejimi one cikarmayi yasakliyor.
Bolumlerden biri eksikse harness `bicim_ihlali` uretir; yani "sadece sok
senaryosunu anlatan" bir brifing gecemez.

UC URETIM YOLU, TEK BICIM. `sablon_metni` LLM'siz deterministik referansi
uretir; `brifing_uret` ayni bicimi modelden ister. Denetciler ikisini
ayirt etmez -- ayirt etseydi "LLM ciktisi test ediliyor" cumlesi bos olurdu.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from agent import tools as at
from agent.client import ModelTuru
from core.config import Config

# Bicim sozlesmesinin sabitleri. Denetci (harness/denetim.py) bunlari
# import eder; iki yerde iki ayri string durmasin.
BASLIK_ONEKI = "# Teklif brifingi"
REJIM_BASLIGI = "## Rejim: "
ECZANE_BASLIGI = "## Eczane"
KISIT_BASLIGI = "## Kisit notlari"
ONERI_BASLIGI = "## Oneri kaydi"
ONERI_BLOGU = "oneri"
# Oneri kaydinda zorunlu alanlar. Eksigi `bicim_ihlali`dir: yarim bir oneri
# denetlenemez ve sahada da uygulanamaz.
ONERI_ALANLARI: tuple[str, ...] = (
    "senaryo", "eczane_id", "sku_id", "lot_id", "mf_orani", "vade_gun",
    "adet", "bedava_adet")


# --------------------------------------------------------------------------
# olgu paketi
# --------------------------------------------------------------------------
@dataclass
class Brifing:
    """Modele verilen olgularin TAMAMI. Disinda bir kaynak yok."""

    eczane_id: str
    origin: int
    politika: str
    taban_rejim: str
    eczane: dict
    rejimler: dict[str, dict]        # rejim -> {parametre, ozet, fark}
    teklifler: dict[str, list[dict]]  # rejim -> teklif satirlari (kirpilmis)
    vetolar: dict[str, list[dict]]    # rejim -> veto satirlari (kirpilmis)
    kol_ekonomisi: dict[str, list[dict]]  # rejim -> en iyi satirin kol tablosu
    urunler: dict[str, dict] = field(default_factory=dict)

    @property
    def rejim_adlari(self) -> list[str]:
        return list(self.rejimler)


def brifing_kur(cfg: Config, b: at.AjanBaglami, eczane_id: str) -> Brifing:
    """Tek eczanenin olgu paketi. Kirpma boyutlari `ajan.*` knob'lari."""
    if not b.eczane_var(eczane_id):
        raise KeyError(f"eczane yok: {eczane_id}")
    n_teklif = cfg.ajan.brifing_teklif_sayisi
    n_veto = cfg.ajan.brifing_veto_sayisi

    rejimler, teklifler, vetolar, kol_ekonomisi, urunler = {}, {}, {}, {}, {}
    for ad in b.rejimler:
        rejimler[ad] = {
            "parametre": b.rejim_parametreleri[ad],
            "ozet": b.rejim_ozetleri[ad],
            "tabana_gore_fark": b.rejim_farklari.get(ad, {}),
        }
        satirlar = b.teklifler.get((ad, eczane_id), [])[:n_teklif]
        teklifler[ad] = satirlar
        vetolar[ad] = b.vetolar.get((ad, eczane_id), [])[:n_veto]
        if satirlar:
            en_iyi = satirlar[0]
            kol_ekonomisi[ad] = b.kol_ekonomisi.get(
                (ad, eczane_id, en_iyi["sku_id"]), [])
        for satir in satirlar + vetolar[ad]:
            if satir["sku_id"] in b.urunler:
                urunler[satir["sku_id"]] = b.urunler[satir["sku_id"]]

    return Brifing(eczane_id=eczane_id, origin=b.origin, politika=b.politika,
                   taban_rejim=b.taban_rejim, eczane=b.eczaneler[eczane_id],
                   rejimler=rejimler, teklifler=teklifler, vetolar=vetolar,
                   kol_ekonomisi=kol_ekonomisi, urunler=urunler)


def olgu_paketi(brifing: Brifing) -> dict:
    """Modele JSON olarak verilen govde."""
    return {
        "origin_haftasi": brifing.origin,
        "politika": brifing.politika,
        "taban_rejim": brifing.taban_rejim,
        "eczane": brifing.eczane,
        "urunler": brifing.urunler,
        "rejimler": brifing.rejimler,
        "teklifler": brifing.teklifler,
        "vetolar": brifing.vetolar,
        "kol_ekonomisi": brifing.kol_ekonomisi,
    }


def defter_kur(brifing: Brifing) -> at.SayiDefteri:
    """Olgu paketinin sayilarini rejim etiketleriyle deftere isler.

    Etiketleme onemli: `sok` bolumunde yalnizca `sok` ve `genel` sayilari
    mesru. Rejime bagli olmayan olgular (eczane profili, urun master)
    GENEL etiketiyle girer ve her bolumde serbesttir.
    """
    defter = at.SayiDefteri()
    defter.ekle(at.GENEL, brifing.origin)
    defter.ekle(at.GENEL, brifing.eczane)
    defter.ekle(at.GENEL, brifing.urunler)
    for ad in brifing.rejim_adlari:
        defter.ekle(ad, brifing.rejimler[ad])
        defter.ekle(ad, brifing.teklifler[ad])
        defter.ekle(ad, brifing.vetolar[ad])
        defter.ekle(ad, brifing.kol_ekonomisi.get(ad, []))
    return defter


# --------------------------------------------------------------------------
# istem
# --------------------------------------------------------------------------
SISTEM_ISTEMI = """Sen bir B2B ilac dagitim sirketinin saha ekibine (KAM) \
haftalik teklif brifingi yazan bir asistansin.

KARARI SEN VERMIYORSUN. Teklif listesi, mal fazlasi orani, vade ve lot \
secimi karar motoru tarafindan uretildi. Senin isin bu karari saha ekibine \
anlatmak ve kur rejimi altinda nasil degistigini yorumlamak.

KESIN KURALLAR:
1. SAYI UYDURMA. Yazdigin her sayi ya olgu paketinde ya da bir arac \
cagrisinin dondurdugu sonucta bulunmali. Hesaplama yapma, oran turetme, \
yuvarlanmis tahmin yazma. Elinde olmayan bir sayiya ihtiyacin varsa "veri \
yok" yaz.
2. REJIMLERI KARISTIRMA. Her rejimin sayisi yalnizca kendi bolumunde \
kullanilir. Baz rejimin marjini sok bolumune yazmak en agir hatadir.
3. KISITI DELME. Vetolanmis bir urunu onerme. MF kanali kapali bir satirda \
mal fazlasi teklif etme. Kirmizi/yesil receteli urun hicbir kosulda \
onerilemez.
4. VAR OLMAYAN VARLIK UYDURMA. Yalnizca araclarin dondurdugu eczane, urun \
ve lot kimliklerini kullan.
5. TEK REJIMI ONE CIKARMA. Kur tahmin edilmiyor; uc rejim de kosullu bir \
okumadir ve ucu de yazilmalidir.

BICIM (birebir uy):

# Teklif brifingi | <eczane_id> | hafta <origin> | politika <ad>

## Eczane
<eczanenin kisa profili>

## Rejim: <rejim adi>          (her rejim icin ayri bolum, hepsi zorunlu)
<o rejimde ne oneriliyor, neden, tabana gore ne degisti>

## Kisit notlari
<vetolanan satirlar ve dayanaklari>

## Oneri kaydi
```oneri
- senaryo: <rejim adi>
  eczane_id: <...>
  sku_id: <...>
  lot_id: <...>
  mf_orani: <...>
  vade_gun: <...>
  adet: <...>
  bedava_adet: <...>
```

Oneri kaydinda YALNIZCA teklif listesinde fiilen bulunan satirlar yer alir \
ve alanlar teklif listesindeki degerlerle birebir ayni olmalidir."""


def kullanici_istemi(brifing: Brifing) -> str:
    return (
        f"{brifing.eczane_id} icin {brifing.origin}. haftanin teklif "
        f"brifingini yaz. Rejimler: {', '.join(brifing.rejim_adlari)}; taban "
        f"rejim {brifing.taban_rejim}.\n\n"
        f"Olgu paketi (JSON):\n```json\n"
        f"{json.dumps(olgu_paketi(brifing), ensure_ascii=False, indent=2)}\n```\n\n"
        f"Eksik gordugun bir olgu varsa araclari cagir. Arac cagirmadan da "
        f"yazabilirsin; olgu paketi tek basina yeterli."
    )


# --------------------------------------------------------------------------
# uretim
# --------------------------------------------------------------------------
@dataclass
class BrifingCiktisi:
    eczane_id: str
    metin: str
    defter: at.SayiDefteri
    brifing: Brifing
    arac_cagrilari: list[dict] = field(default_factory=list)
    tur_sayisi: int = 0
    kesildi: bool = False       # azami tur asildi mi
    kaynak: str = "sablon"


def brifing_uret(cfg: Config, b: at.AjanBaglami, eczane_id: str,
                 istemci=None) -> BrifingCiktisi:
    """Arac dongusu. `istemci` None ise deterministik sablon uretir.

    Arac SONUCLARI her zaman `agent/tools.py` tarafindan O AN hesaplanir --
    kayitli oynatmada bile (agent/client.py basligi). Defter bu yuzden
    modelin gordugu sayilarin GERCEK kumesidir.
    """
    brifing = brifing_kur(cfg, b, eczane_id)
    defter = defter_kur(brifing)
    if istemci is None:
        return BrifingCiktisi(eczane_id=eczane_id, metin=sablon_metni(brifing),
                              defter=defter, brifing=brifing, kaynak="sablon")

    sistem = SISTEM_ISTEMI
    mesajlar: list[dict] = [{"role": "user", "content": kullanici_istemi(brifing)}]
    cagrilar: list[dict] = []
    metin, tur, kesildi = "", 0, False
    while tur < cfg.ajan.azami_tur:
        cevap: ModelTuru = istemci.konus(sistem, mesajlar, at.ARAC_SEMALARI)
        tur += 1
        if cevap.metin:
            metin = cevap.metin
        if not cevap.araclar:
            break
        mesajlar.append({"role": "assistant",
                         "content": _asistan_blogu(cevap)})
        sonuclar = []
        for c in cevap.araclar:
            sonuc = at.cagir(b, c.ad, c.girdi, defter)
            cagrilar.append({"ad": c.ad, "girdi": c.girdi})
            sonuclar.append({"type": "tool_result", "tool_use_id": c.kimlik,
                             "content": json.dumps(sonuc, ensure_ascii=False)})
        mesajlar.append({"role": "user", "content": sonuclar})
    else:
        kesildi = True

    return BrifingCiktisi(eczane_id=eczane_id, metin=metin, defter=defter,
                          brifing=brifing, arac_cagrilari=cagrilar,
                          tur_sayisi=tur, kesildi=kesildi, kaynak="llm")


def _asistan_blogu(cevap: ModelTuru) -> list[dict]:
    bloklar: list[dict] = []
    if cevap.metin:
        bloklar.append({"type": "text", "text": cevap.metin})
    for c in cevap.araclar:
        bloklar.append({"type": "tool_use", "id": c.kimlik, "name": c.ad,
                        "input": c.girdi})
    return bloklar


# --------------------------------------------------------------------------
# deterministik sablon (denetcilerin TEMIZ referansi)
# --------------------------------------------------------------------------
def sayi_bicimle(x: float) -> str:
    """Turkce ondalik ayraciyla sayi. Defterdeki degerin YUVARLANMISI.

    BINLIK AYRACI YOK. "2.629" yazsaydik denetci bunu 2,629 mu 2629 mu diye
    ayirt etmek zorunda kalir ve belirsizligi TOLERANS lehine cozerdi --
    yani sayi uydurma denetcisinin kadranini genisletirdi. Okunabilirligi
    biraz feda edip belirsizligi tamamen kaldirmak daha ucuz.
    """
    v = float(x)
    if v.is_integer():
        return str(int(v))
    return f"{v:.2f}".replace(".", ",")


def sablon_metni(brifing: Brifing) -> str:
    """Olgu paketinden bicim sozlesmesine uygun metin.

    KURAL: burada hicbir sayi HESAPLANMAZ. Yalnizca olgu paketindekiler
    yazilir. Sablonun kendisi de "sayi uydurma" denetiminden gecmek
    zorunda; gecmiyorsa denetci degil sablon yanlistir.
    """
    e = brifing.eczane
    satirlar = [
        f"{BASLIK_ONEKI} | {brifing.eczane_id} | hafta {brifing.origin} | "
        f"politika {brifing.politika}",
        "",
        ECZANE_BASLIGI,
        f"{e['il']} / {e['ilce']}, {e['aylik_ciro_bandi']} ciro bandi, aylik "
        f"{sayi_bicimle(e['aylik_recete_adedi'])} recete. Hastaneye "
        f"{sayi_bicimle(e['hastane_yakinligi_km'])} km. Vade riski "
        f"{sayi_bicimle(e['vade_riski_skoru'])}, DBS limiti "
        f"{sayi_bicimle(e['dbs_limiti_tl'])} TL, acik bakiye "
        f"{sayi_bicimle(e['acik_bakiye_tl'])} TL. Haftalik teklif tavani "
        f"{e['haftalik_teklif_tavani']} satir.",
        "",
    ]
    for ad in brifing.rejim_adlari:
        satirlar += _rejim_bolumu(brifing, ad)
    satirlar += _kisit_bolumu(brifing)
    satirlar += _oneri_bolumu(brifing)
    return "\n".join(satirlar).rstrip() + "\n"


def _rejim_bolumu(brifing: Brifing, ad: str) -> list[str]:
    p = brifing.rejimler[ad]["parametre"]
    o = brifing.rejimler[ad]["ozet"]
    f = brifing.rejimler[ad]["tabana_gore_fark"]
    teklifler = brifing.teklifler[ad]
    satirlar = [
        f"{REJIM_BASLIGI}{ad}",
        f"{p['aciklama']}. Guncelleme beklentisi "
        f"{sayi_bicimle(p['guncelleme_beklentisi_hafta'])} hafta, varsayilan referans "
        f"kur artisi {sayi_bicimle(p['referans_kur_artisi'])}, fiyat gecis katsayisi "
        f"{sayi_bicimle(p['fiyat_gecis_katsayisi'])}, talep carpani "
        f"{sayi_bicimle(p['antisipasyon_talep_carpani'])}, fonlama carpani "
        f"{sayi_bicimle(p['fonlama_orani_carpani'])}.",
        "",
        f"Portfoy genelinde bu rejimde {sayi_bicimle(o['teklif_sayisi'])} teklif "
        f"cikiyor; ortalama MF {sayi_bicimle(o['ortalama_mf'])}, ortalama vade "
        f"{sayi_bicimle(o['ortalama_vade'])} gun, beklenen artimsal marj "
        f"{sayi_bicimle(o['beklenen_artimsal_marj'])} TL. Erteleme kazanci adet "
        f"basina {sayi_bicimle(o['ortalama_erteleme_tl'])} TL.",
    ]
    if f:
        satirlar.append(
            f"Taban rejime gore: {sayi_bicimle(f['teklif_sayisi_farki'])} teklif farki, "
            f"{sayi_bicimle(f['kol_degisen_satir'])} satirda kol degisti, "
            f"{sayi_bicimle(f['teklife_giren'])} satir teklife girdi, "
            f"{sayi_bicimle(f['teklifden_cikan'])} satir cikti; artimsal marj farki "
            f"{sayi_bicimle(f['artimsal_marj_farki'])} TL.")
    satirlar.append("")
    if not teklifler:
        satirlar += ["Bu eczane icin bu rejimde teklif yok.", ""]
        return satirlar
    satirlar.append("Bu eczane icin onerilen satirlar:")
    for t in teklifler:
        urun = brifing.urunler.get(t["sku_id"], {})
        satirlar.append(
            f"- {t['sku_id']} ({urun.get('urun_tipi', 'bilinmiyor')}, "
            f"{urun.get('kategori_kod', 'bilinmiyor')}), lot {t['lot_id']}: "
            f"{t['mf_ifadesi']}, {t['vade_gun']} gun vade, "
            f"{sayi_bicimle(t['adet'])} adet + {sayi_bicimle(t['bedava_adet'])} bedava. "
            f"Kabul olasiligi {sayi_bicimle(t['kabul_olasiligi'])} (teklifsiz "
            f"{sayi_bicimle(t['teklifsiz_kabul_olasiligi'])}), artimsal beklenen marj "
            f"{sayi_bicimle(t['artimsal_beklenen_marj_tl'])} TL. Lotun kalan raf omru "
            f"{sayi_bicimle(t['lot_kalan_gun'])} gun; guncellemeyi "
            f"{'bekleyebilir' if t['lot_bekleyebilir'] else 'BEKLEYEMEZ'}.")
    kollar = brifing.kol_ekonomisi.get(ad, [])
    if kollar:
        satirlar += ["", f"En iyi satirin ({teklifler[0]['sku_id']}) kol ekonomisi:"]
        for k in kollar:
            satirlar.append(
                f"- {k['kol']}: kabul {sayi_bicimle(k['kabul_olasiligi'])}, kabul "
                f"sartiyla marj {sayi_bicimle(k['kabul_sartiyla_brut_marj_tl'])} TL, "
                f"artimsal {sayi_bicimle(k['artimsal_beklenen_marj_tl'])} TL"
                + (" (secilen)" if k["secilen"] else ""))
    satirlar.append("")
    return satirlar


def _kisit_bolumu(brifing: Brifing) -> list[str]:
    satirlar = [KISIT_BASLIGI]
    yazildi = False
    for ad in brifing.rejim_adlari:
        for v in brifing.vetolar[ad]:
            satirlar.append(
                f"- [{ad}] {v['sku_id']}: {v['veto_sebebi']} -- {v['dayanak']}")
            yazildi = True
    if not yazildi:
        satirlar.append("- Bu eczanede vetolanmis aday satiri yok.")
    satirlar.append("")
    return satirlar


def _oneri_bolumu(brifing: Brifing) -> list[str]:
    satirlar = [ONERI_BASLIGI, f"```{ONERI_BLOGU}"]
    bos = True
    for ad in brifing.rejim_adlari:
        for t in brifing.teklifler[ad]:
            bos = False
            satirlar += [
                f"- senaryo: {ad}",
                f"  eczane_id: {t['eczane_id']}",
                f"  sku_id: {t['sku_id']}",
                f"  lot_id: {t['lot_id']}",
                f"  mf_orani: {t['mf_orani']}",
                f"  vade_gun: {t['vade_gun']}",
                f"  adet: {t['adet']}",
                f"  bedava_adet: {t['bedava_adet']}",
            ]
    if bos:
        satirlar.append("[]")
    satirlar.append("```")
    return satirlar
