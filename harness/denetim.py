"""Deterministik denetciler. SPEC M7 cikis kriterinin ta kendisi.

    "Eval harness'ta LLM ciktisi DETERMINISTIK olarak test ediliyor --
     hallusinasyon, kisit ihlali iddiasi, sayi uydurma yakalaniyor."

Bes bulgu tipi, hicbiri bir metin siniflandiricisina dayanmiyor:

| tip | soru | dayanak |
|---|---|---|
| `bicim_ihlali`      | Brifing sozlesmeye uyuyor mu | agent/narrative.py bicimi |
| `hallusinasyon`     | Atif yapilan varlik var mi | ajan baglaminin kimlik kumeleri |
| `kisit_ihlali`      | Vetolanmis satir onerildi mi, kapali kanalda MF verildi mi | policy/constraints.py ciktisi |
| `senaryo_karismasi` | Bir rejimin sayisi baska rejimin bolumunde mi | rejim etiketli sayi defteri |
| `sayi_uydurma`      | Metindeki sayi modele fiilen verilmis mi | agent/tools.py sayi defteri |

Artı bir tip, dortlunun disinda kalan ve olcusuz birakilmamasi gereken
durum icin: `uydurma_oneri` -- kimlikleri gercek, kisiti delmeyen, ama
politikanin HIC URETMEDIGI bir oneri satiri.

DENETCININ CANLI OLDUGU NASIL BILINIR. Bu dosya tek basina bir sey
kanitlamaz: hicbir seyi yakalamayan bir denetci de butun temiz vakalari
gecirir. Kanit harness/mutasyon.py'de -- temiz ciktidan uretilmis bozuk
varyantlarin her biri BEKLENEN tipte bir bulgu uretmek ZORUNDA. M5'in
"temizlik penceresi bos, rejim olu" ve M6'nin "ortusme teshisi olu"
kontrolleriyle ayni disiplin.

SAYI ESLESTIRME. Metindeki sayi ile defterdeki sayi birebir ayni olmak
zorunda degil: brifing yuvarlanmis yazar. Kabul kurali iki yoldan biri:
bagil tolerans (`harness.sayi_toleransi_bagil`) ya da defterdeki degerin
`harness.yuvarlama_basamaklari`ndan birine yuvarlanmasiyla esitlik. Ikisi
de config'te ve ikisi de TUNING.md'de satiri olan knob'lar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import yaml

from agent import narrative as nv
from agent import tools as at

# Kimlik bicimi: harf oneki + rakam ("ECZ0007", "SKU0042", "LOT000123").
# M1'in urettigi kimlik semasi; onekler config'ten degil VERIDEN turetilir
# (`kimlik_desenleri`), boylece sema degisirse denetci sessizce korlesmez.
_KIMLIK = re.compile(r"\b([A-Z]{2,6})(\d{2,})\b")
# Sayi belirteci. Isaret DAHIL: "-1453,01" pozitif okunsaydi negatif marj
# farklari sistematik olarak "uydurma" cikardi.
_SAYI = re.compile(r"(?<![A-Za-z0-9.,])-?\d+(?:[.,]\d+)*")
# Markdown gurultusu: baslik isaretleri, madde imleri, numarali madde.
_MADDE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|#{1,6}\s+)")
# Kisit satirlarinda rejim atfi: "- [sok] SKU0001: ...".
_REJIM_ONEKI = re.compile(r"^\s*[-*+]?\s*\[([^\]]+)\]")
# Binlik ayracin grup uzunlugu. Hem TR hem EN bicimde ucer basamak; bicim
# sabiti. Kullanildigi yer ve neden gerekli oldugu `_okuma`da.
BINLIK_GRUBU = 3


@dataclass(frozen=True)
class Bulgu:
    tip: str
    bolum: str
    kanit: str

    def __str__(self) -> str:
        return f"{self.tip} [{self.bolum}] {self.kanit}"


BULGU_TIPLERI: tuple[str, ...] = (
    "bicim_ihlali", "hallusinasyon", "kisit_ihlali", "senaryo_karismasi",
    "sayi_uydurma", "uydurma_oneri")


# --------------------------------------------------------------------------
# metin ayristirma
# --------------------------------------------------------------------------
def oneri_blogu(metin: str) -> tuple[list[dict] | None, str]:
    """Fenced ```oneri blogunu ayristirir. (kalemler, hata) doner."""
    desen = re.compile(r"```" + nv.ONERI_BLOGU + r"\s*\n(.*?)```", re.S)
    eslesme = desen.search(metin)
    if eslesme is None:
        return None, f"'{nv.ONERI_BLOGU}' blogu yok"
    govde = eslesme.group(1)
    try:
        cozulen = yaml.safe_load(govde)
    except yaml.YAMLError as hata:
        return None, f"oneri blogu YAML olarak cozulemedi: {hata}"
    if cozulen is None:
        return [], ""
    if not isinstance(cozulen, list):
        return None, "oneri blogu bir liste degil"
    if not all(isinstance(k, dict) for k in cozulen):
        return None, "oneri blogunda sozluk olmayan kalem var"
    return cozulen, ""


def oneri_blogunu_cikar(metin: str) -> str:
    """`oneri` blogu yapisal olarak ayrica denetleniyor; iki kez sayilmasin."""
    return re.sub(r"```" + nv.ONERI_BLOGU + r"\s*\n.*?```", "", metin, flags=re.S)


def satiri_temizle(satir: str) -> str:
    """Sayi taramasindan once TEK SATIRDAN cikarilanlar.

    - kimlikler: "ECZ0007" icindeki 0007 bir olgu degil, kimligin parcasi.
    - madde imleri: "1." bir sayi degil, liste isareti.

    Bolum basliklari BURADA silinmez; onlari `bolumlere_ayir` tuketiyor.
    Once temizleyip sonra bolumlere ayirmak baslik satirlarini yok ederdi ve
    butun metin tek bir GENEL bolum gibi okunurdu -- senaryo karistirma
    denetcisi de o anda olurdu.
    """
    return _MADDE.sub("", _KIMLIK.sub(" ", satir))


def bolumlere_ayir(metin: str, rejimler: list[str]) -> list[tuple[str, str]]:
    """[(etiket, satir)] -- her satirin hangi rejime ait oldugu.

    Etiket, `## Rejim: X` basligindan gelir; satirin basinda `[X]` varsa o
    ONCELIKLIDIR (kisit notlari bolumu rejim basina satir tasiyor). Hicbiri
    yoksa GENEL.
    """
    cikti: list[tuple[str, str]] = []
    etiket = at.GENEL
    for satir in metin.splitlines():
        if satir.startswith(nv.REJIM_BASLIGI):
            etiket = satir[len(nv.REJIM_BASLIGI):].strip()
            continue
        if satir.startswith("## "):
            etiket = at.GENEL
            continue
        yerel = _REJIM_ONEKI.match(satir)
        if yerel and yerel.group(1).strip() in rejimler:
            cikti.append((yerel.group(1).strip(), satir))
            continue
        cikti.append((etiket, satir))
    return cikti


def sayi_adaylari(belirtec: str) -> list[float]:
    """Bir belirtecin GECERLI sayisal okumalari.

    Iki bicim de kabul edilir: Turkce (nokta binlik, virgul ondalik) ve
    Ingilizce (virgul binlik, nokta ondalik). Model hangisini yazarsa
    yazsin okunabilsin diye.

    AMA BINLIK GRUBU UCER BASAMAK OLMAK ZORUNDA. Ilk uygulamada bu kural
    yoktu ve "0,38" belirteci Ingilizce okunusla "038" = 38 diye de
    okunuyordu; 38 defterde bulununca (bir adet, bir vade) sahte esitlik
    cikiyordu. Sonuc olculdu: `sayi_bozma` mutasyonu ikinci bir seed'de
    "sayi uydurma" yerine "senaryo karismasi" olarak siniflaniyordu --
    denetci bagirdi ama YANLIS ISIMLE. Gecersiz gruplamayi elemek
    belirsizligi gercekten belirsiz olan yerlere ("2.629") sinirliyor.
    """
    metin = belirtec.strip()
    isaret = -1.0 if metin.startswith("-") else 1.0
    govde = metin.lstrip("+-")
    adaylar = []
    for binlik, ondalik in ((".", ","), (",", ".")):
        deger = _okuma(govde, binlik, ondalik)
        if deger is not None:
            adaylar.append(isaret * deger)
    return adaylar


def _okuma(govde: str, binlik: str, ondalik: str) -> float | None:
    if govde.count(ondalik) > 1:
        return None
    tam, _, kesir = govde.partition(ondalik)
    if binlik in kesir or (kesir and not kesir.isdigit()):
        return None
    parcalar = tam.split(binlik)
    # Binlik grubu UCER basamaktir; bicim sabiti, knob degil. Sart olmadan
    # "0,38" Ingilizce okunusla "038" = 38 diye de okunuyor ve 38 baska bir
    # rejimin defterinde bulunca mutasyon `sayi_uydurma` yerine
    # `senaryo_karismasi` diye siniflaniyordu (reports/m7.md 6.2).
    if len(parcalar) > 1 and (not parcalar[0] or len(parcalar[0]) > BINLIK_GRUBU
                              or any(len(p) != BINLIK_GRUBU for p in parcalar[1:])):
        return None
    duz = tam.replace(binlik, "")
    if not duz.isdigit():
        return None
    return float(f"{duz}.{kesir}" if kesir else duz)


def _esitse(deger: float, kume: set[float], tolerans: float,
            basamaklar: list[int]) -> bool:
    """Iki yoldan biri yeterli: BAGIL tolerans ya da mesru bir yuvarlama.

    Tolerans bilerek MUTLAK bir taban tasimiyor. Tasisaydi ("en az +-0,005")
    kucuk sayilar -- kabul olasiliklari, MF oranlari -- bir kor bolge
    olusturur ve `mutasyon_sapmasi` kadar bozulmus bir olasilik degeri
    tolerans icinde kalirdi. Goruntuleme yuvarlamasini zaten ikinci kural
    karsiliyor.
    """
    for s in kume:
        if abs(deger - s) <= tolerans * abs(s):
            return True
        if any(round(s, b) == deger for b in basamaklar):
            return True
    return False


# --------------------------------------------------------------------------
# denetciler
# --------------------------------------------------------------------------
def denetle_bicim(metin: str, rejimler: list[str]) -> list[Bulgu]:
    bulgular = []
    if not metin.lstrip().startswith(nv.BASLIK_ONEKI):
        bulgular.append(Bulgu("bicim_ihlali", at.GENEL,
                              f"baslik '{nv.BASLIK_ONEKI}' ile baslamiyor"))
    for baslik in (nv.ECZANE_BASLIGI, nv.KISIT_BASLIGI, nv.ONERI_BASLIGI):
        if baslik not in metin:
            bulgular.append(Bulgu("bicim_ihlali", at.GENEL, f"eksik bolum: {baslik}"))
    for ad in rejimler:
        if f"{nv.REJIM_BASLIGI}{ad}" not in metin:
            bulgular.append(Bulgu("bicim_ihlali", ad,
                                  f"rejim bolumu eksik: {ad} (D3: tek rejim one "
                                  f"cikarilamaz, hepsi yazilir)"))
    kalemler, hata = oneri_blogu(metin)
    if kalemler is None:
        bulgular.append(Bulgu("bicim_ihlali", at.GENEL, hata))
        return bulgular
    for i, kalem in enumerate(kalemler):
        eksik = [a for a in nv.ONERI_ALANLARI if a not in kalem]
        if eksik:
            bulgular.append(Bulgu("bicim_ihlali", str(kalem.get("senaryo", at.GENEL)),
                                  f"oneri #{i + 1} eksik alan: {eksik}"))
    return bulgular


def kimlik_desenleri(b: at.AjanBaglami) -> dict[str, set[str]]:
    """Onek -> gecerli kimlikler. Onekler VERIDEN turetilir, sabit degil."""
    kumeler: dict[str, set[str]] = {}
    for kimlikler in (b.eczaneler, b.urunler, b.lotlar):
        for kimlik in kimlikler:
            eslesme = _KIMLIK.match(kimlik)
            if eslesme:
                kumeler.setdefault(eslesme.group(1), set()).add(kimlik)
    return kumeler


def denetle_hallusinasyon(metin: str, b: at.AjanBaglami,
                          rejimler: list[str]) -> list[Bulgu]:
    """Metinde ve oneri blogunda gecen, baglamda BULUNMAYAN kimlikler."""
    kumeler = kimlik_desenleri(b)
    bulgular, gorulen = [], set()
    for etiket, satir in bolumlere_ayir(metin, rejimler):
        for onek, rakam in _KIMLIK.findall(satir):
            kimlik = f"{onek}{rakam}"
            if onek not in kumeler or kimlik in kumeler[onek]:
                continue
            if kimlik in gorulen:
                continue
            gorulen.add(kimlik)
            bulgular.append(Bulgu("hallusinasyon", etiket,
                                  f"baglamda olmayan kimlik: {kimlik}"))
    return bulgular


def _satir_bul(b: at.AjanBaglami, rejim: str, eczane_id: str,
               sku_id: str) -> dict | None:
    for satir in b.teklifler.get((rejim, eczane_id), []):
        if satir["sku_id"] == sku_id:
            return satir
    return None


def _veto_bul(b: at.AjanBaglami, rejim: str, eczane_id: str,
              sku_id: str) -> dict | None:
    for satir in b.vetolar.get((rejim, eczane_id), []):
        if satir["sku_id"] == sku_id:
            return satir
    return None


# Onerinin gercek satirla karsilastirilan alanlari. Lot ayri ele alinir
# (kimligi metin, digerleri sayi).
KARSILASTIRILAN = (("mf_orani", "mf_orani"), ("vade_gun", "vade_gun"),
                   ("adet", "adet"), ("bedava_adet", "bedava_adet"))


def denetle_oneriler(metin: str, b: at.AjanBaglami, veto_renkleri: list[str],
                     tolerans: float, basamaklar: list[int]) -> list[Bulgu]:
    """Oneri kaydinin satir satir denetimi.

    Sirasiyla sorulan sorular ve urettikleri bulgu tipi:
      1. Kimlikler var mi                      -> hallusinasyon
      2. Urun regulasyon vetosunda mi          -> kisit_ihlali (en agir)
      3. Satir kisit katmani tarafindan vetolanmis mi -> kisit_ihlali
      4. Politika bu satira teklif verdi mi    -> uydurma_oneri
      5. MF kanali kapaliyken MF verilmis mi   -> kisit_ihlali
      6. Alanlar gercek degerlerle ayni mi     -> senaryo_karismasi | sayi_uydurma
    """
    kalemler, _ = oneri_blogu(metin)
    if not kalemler:
        return []
    bulgular = []
    for kalem in kalemler:
        rejim = str(kalem.get("senaryo", ""))
        eczane_id = str(kalem.get("eczane_id", ""))
        sku_id = str(kalem.get("sku_id", ""))
        if rejim not in b.rejim_parametreleri:
            bulgular.append(Bulgu("bicim_ihlali", rejim or at.GENEL,
                                  f"oneri tanimsiz senaryo tasiyor: {rejim}"))
            continue
        if eczane_id not in b.eczaneler or sku_id not in b.urunler:
            bulgular.append(Bulgu("hallusinasyon", rejim,
                                  f"oneride olmayan kimlik: {eczane_id}/{sku_id}"))
            continue
        renk = b.urunler[sku_id].get("recete_rengi")
        if renk in veto_renkleri:
            bulgular.append(Bulgu(
                "kisit_ihlali", rejim,
                f"{sku_id} recete rengi {renk}: promosyon yasak, hicbir "
                f"kosulda onerilemez (D6)"))
            continue
        vetolu = _veto_bul(b, rejim, eczane_id, sku_id)
        gercek = _satir_bul(b, rejim, eczane_id, sku_id)
        if gercek is None:
            if vetolu is not None:
                bulgular.append(Bulgu(
                    "kisit_ihlali", rejim,
                    f"{eczane_id}/{sku_id} kisit katmani tarafindan vetolandi "
                    f"({vetolu['veto_sebebi']}) ama onerildi"))
            else:
                bulgular.append(Bulgu(
                    "uydurma_oneri", rejim,
                    f"{eczane_id}/{sku_id} icin {rejim} rejiminde politika "
                    f"teklif URETMEDI"))
            continue
        bulgular += _oneri_alanlari(kalem, gercek, b, rejim, eczane_id, sku_id,
                                    tolerans, basamaklar)
    return bulgular


def _oneri_alanlari(kalem: dict, gercek: dict, b: at.AjanBaglami, rejim: str,
                    eczane_id: str, sku_id: str, tolerans: float,
                    basamaklar: list[int]) -> list[Bulgu]:
    bulgular = []
    if str(kalem.get("lot_id")) != gercek["lot_id"]:
        bulgular.append(Bulgu("kisit_ihlali", rejim,
                              f"{sku_id} lot referansi yanlis: onerilen "
                              f"{kalem.get('lot_id')}, tahsis edilen "
                              f"{gercek['lot_id']}"))
    mf = _sayi_oku(kalem.get("mf_orani"))
    if mf is not None and mf > 0 and not gercek["mf_kanali_acik"]:
        bulgular.append(Bulgu("kisit_ihlali", rejim,
                              f"{sku_id} satirinda MF kanali kapali "
                              f"(SGK kapsami) ama {mf} MF onerildi"))
    for alan, kaynak in KARSILASTIRILAN:
        deger = _sayi_oku(kalem.get(alan))
        if deger is None:
            continue
        if _esitse(deger, {float(gercek[kaynak])}, tolerans, basamaklar):
            continue
        digeri = _baska_rejimde(b, eczane_id, sku_id, kaynak, deger, rejim,
                                tolerans, basamaklar)
        if digeri:
            bulgular.append(Bulgu(
                "senaryo_karismasi", rejim,
                f"{sku_id}.{alan} = {deger}: {rejim} rejiminde "
                f"{gercek[kaynak]} olmali, bu deger '{digeri}' rejiminin"))
        else:
            bulgular.append(Bulgu(
                "sayi_uydurma", rejim,
                f"{sku_id}.{alan} = {deger}: gercek deger {gercek[kaynak]}"))
    return bulgular


def _baska_rejimde(b: at.AjanBaglami, eczane_id: str, sku_id: str, alan: str,
                   deger: float, haric: str, tolerans: float,
                   basamaklar: list[int]) -> str | None:
    for rejim in b.rejimler:
        if rejim == haric:
            continue
        satir = _satir_bul(b, rejim, eczane_id, sku_id)
        if satir is not None and _esitse(deger, {float(satir[alan])},
                                         tolerans, basamaklar):
            return rejim
    return None


def _sayi_oku(deger) -> float | None:
    if isinstance(deger, bool) or deger is None:
        return None
    if isinstance(deger, (int, float)):
        return float(deger)
    adaylar = sayi_adaylari(str(deger).strip())
    return adaylar[0] if adaylar else None


def denetle_sayilar(metin: str, defter: at.SayiDefteri, rejimler: list[str],
                    tolerans: float, basamaklar: list[int],
                    yoksayma_tavani: float) -> list[Bulgu]:
    """Serbest metindeki her sayi defterde var mi.

    Bulunamadiginda tip AYRIMI yapilir: sayi BASKA bir rejimin defterinde
    varsa bu bir karistirmadir (`senaryo_karismasi`), hicbir yerde yoksa
    uydurmadir (`sayi_uydurma`). Ayrim onemli cunku ikisinin tedavisi farkli:
    biri bicim/dikkat sorunu, digeri olgu sorunu.
    """
    bulgular = []
    for etiket, ham in bolumlere_ayir(oneri_blogunu_cikar(metin), rejimler):
        izinli = defter.izinli(etiket)
        for belirtec in _SAYI.findall(satiri_temizle(ham)):
            adaylar = sayi_adaylari(belirtec)
            if not adaylar:
                continue
            if any(_esitse(v, izinli, tolerans, basamaklar) for v in adaylar):
                continue
            if all(float(v).is_integer() and abs(v) <= yoksayma_tavani
                   for v in adaylar):
                continue
            digeri = _sayinin_rejimi(adaylar, defter, rejimler, etiket,
                                     tolerans, basamaklar)
            if digeri:
                bulgular.append(Bulgu(
                    "senaryo_karismasi", etiket,
                    f"'{belirtec}' bu bolumde yok ama '{digeri}' rejiminin "
                    f"defterinde var"))
            else:
                bulgular.append(Bulgu(
                    "sayi_uydurma", etiket,
                    f"'{belirtec}' modele verilen hicbir olguda yok"))
    return bulgular


def _sayinin_rejimi(adaylar: list[float], defter: at.SayiDefteri,
                    rejimler: list[str], haric: str, tolerans: float,
                    basamaklar: list[int]) -> str | None:
    for rejim in rejimler:
        if rejim == haric:
            continue
        if any(_esitse(v, defter.kume(rejim), tolerans, basamaklar)
               for v in adaylar):
            return rejim
    return None


# --------------------------------------------------------------------------
# tek giris noktasi
# --------------------------------------------------------------------------
def denetle(cfg, metin: str, b: at.AjanBaglami,
            defter: at.SayiDefteri) -> list[Bulgu]:
    """Butun denetciler. Sira sabit; cikti tekrar uretilebilir."""
    h = cfg.harness
    rejimler = b.rejimler
    bulgular = denetle_bicim(metin, rejimler)
    bulgular += denetle_hallusinasyon(metin, b, rejimler)
    bulgular += denetle_oneriler(metin, b, list(cfg.politika.kisit.recete_rengi_vetosu),
                                 h.sayi_toleransi_bagil, h.yuvarlama_basamaklari)
    bulgular += denetle_sayilar(metin, defter, rejimler, h.sayi_toleransi_bagil,
                                h.yuvarlama_basamaklari,
                                h.yoksayilan_tamsayi_ust_siniri)
    return bulgular


def tip_sayimi(bulgular: list[Bulgu]) -> dict[str, int]:
    return {tip: sum(1 for b in bulgular if b.tip == tip) for tip in BULGU_TIPLERI}
