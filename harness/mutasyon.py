"""Temiz brifingden BOZUK varyantlar. Denetcinin canli oldugunun kaniti.

Bu dosya olmasaydi harness bir sey KANITLAMAZDI: hicbir seyi yakalamayan
bir denetci de butun temiz vakalari gecirir ve rapor "her sey temiz" der.
M5'in "temizlik penceresi bos, rejim olu" ve M6'nin "ortusme teshisi olu"
kontrolleriyle ayni disiplin -- kadrani olmayan sayi knob degildir, hicbir
seyi yakalamayan denetci de denetci degildir.

Her mutasyon UC sey tasir:
  `ad`            ne bozuldugu
  `beklenen_tip`  hangi denetcinin bagirmasi GEREKTIGI
  `uygun`         bu brifingde uygulanabilir mi (uygulanamazsa vaka duser,
                  sessizce atlanmaz -- atlanan mutasyon kanit uretmez)

MUTASYONLAR OLGU PAKETINDEN TURETILIR, elle yazilmis metin degildir. Bir
knob degisip teklif listesi degistiginde mutantlar da otomatik degisir;
elle yazilmis bozuk metinler ilk config degisikliginde bayatlardi.

SAPMANIN BUYUKLUGU config'te (`harness.mutasyon_sapmasi`) ve sayi
toleransindan buyuk olmak ZORUNDA -- core/config.py `_m7_harness_kilidi`
bunu yukleme aninda kilitliyor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from agent import narrative as nv
from agent import tools as at

# Var olmayan kimlik uretirken kullanilan sonek. Kimligin bicimi dogru,
# kendisi yok: "sema tutuyor ama varlik yok" durumunu sinar.
HAYALI_SONEK = "9999"
# `_mf_kanali_ihlali`nin kapali kanala yazdigi mal fazlasi. Sifir olmayan
# herhangi bir deger ayni iddiayi kurar; buyuklugu denetimin sonucunu
# degistirmez (gerekcesi kullanildigi yerde).
MF_IHLAL_ORANI = 0.1
MF_IHLAL_BEDAVA = 1


class MutasyonUygulanamaz(RuntimeError):
    """Bu brifingde bu bozma yapilamiyor. Sessiz atlanmaz, vaka duser."""


@dataclass(frozen=True)
class Mutasyon:
    ad: str
    beklenen_tip: str
    aciklama: str
    # (metin, baglam, eczane_id, sapma, defter) -> bozulmus metin.
    # Defter yalnizca `sayi_bozma`nin isine yariyor ama imza ortak: bozmanin
    # "temiz bir ornek" uretip uretmedigini kontrol edebilmesi icin defteri
    # gormesi gerekiyor (asagida).
    uygula: Callable[..., str]
    uygun: Callable[[at.AjanBaglami, str], bool]


# --------------------------------------------------------------------------
# yardimcilar
# --------------------------------------------------------------------------
def _teklifler(b: at.AjanBaglami, eczane_id: str, rejim: str) -> list[dict]:
    return b.teklifler.get((rejim, eczane_id), [])


def _oneri_bloklari(metin: str) -> list[str]:
    """Oneri blogunun kalem kalem ham metni ("- senaryo: ..." bloklari)."""
    eslesme = re.search(r"```" + nv.ONERI_BLOGU + r"\s*\n(.*?)```", metin, re.S)
    if eslesme is None:
        raise MutasyonUygulanamaz("oneri blogu yok")
    govde = eslesme.group(1)
    parcalar = re.split(r"(?m)^(?=- )", govde)
    return [p for p in parcalar if p.strip().startswith("- ")]


def _oneri_blogunu_degistir(metin: str, yeni_govde: str) -> str:
    return re.sub(r"(```" + nv.ONERI_BLOGU + r"\s*\n).*?(```)",
                  lambda m: m.group(1) + yeni_govde + m.group(2), metin, flags=re.S)


def _kalem_metni(rejim: str, satir: dict, **degisiklik) -> str:
    alanlar = {"senaryo": rejim, "eczane_id": satir["eczane_id"],
               "sku_id": satir["sku_id"], "lot_id": satir["lot_id"],
               "mf_orani": satir["mf_orani"], "vade_gun": satir["vade_gun"],
               "adet": satir["adet"], "bedava_adet": satir["bedava_adet"]}
    alanlar.update(degisiklik)
    satirlar = [f"- senaryo: {alanlar['senaryo']}"]
    satirlar += [f"  {k}: {v}" for k, v in alanlar.items() if k != "senaryo"]
    return "\n".join(satirlar) + "\n"


def _bolum_araligi(metin: str, rejim: str) -> tuple[int, int]:
    baslik = f"{nv.REJIM_BASLIGI}{rejim}"
    bas = metin.find(baslik)
    if bas < 0:
        raise MutasyonUygulanamaz(f"rejim bolumu yok: {rejim}")
    sonraki = metin.find("\n## ", bas + len(baslik))
    return bas, (len(metin) if sonraki < 0 else sonraki)


def _son_rejim(b: at.AjanBaglami) -> str:
    """Taban olmayan son rejim. Bozma hedefi olarak sabit ve tekrar uretilebilir."""
    digerleri = [r for r in b.rejimler if r != b.taban_rejim]
    if not digerleri:
        raise MutasyonUygulanamaz("taban disinda rejim yok")
    return digerleri[-1]


# --------------------------------------------------------------------------
# mutasyonlar
# --------------------------------------------------------------------------
def _sayi_bozma(metin: str, b: at.AjanBaglami, eczane_id: str, sapma: float,
                defter=None, tolerans: float = 0.0,
                basamaklar: tuple[int, ...] = ()) -> str:
    """Bir rejim bolumundeki ondalikli bir olcum degerini `sapma` kadar kaydirir.

    ADAY SECIMI ONEMLI ve ilk yazdigimda atlamistim: bozulmus deger BASKA
    bir rejimin defterinde durabiliyor -- o zaman denetci hakli olarak
    "sayi uydurma" degil "senaryo karismasi" diyor ve mutasyon kendi tipinin
    TEMIZ bir ornegi olmaktan cikiyor. Bu yuzden bozulmus degeri defterin
    HICBIR yerinde bulunmayan ilk aday secilir; bulunamazsa mutasyon
    uygulanamaz sayilir (sessizce zayif bir ornek uretmez).

    Ondalikli deger hedeflenmesinin sebebi ayri: tamsayi bir adedi bozmak da
    yakalanir ama ondalikli deger sayi denetcisinin YUVARLAMA kuralini da
    sinar -- bozulmus deger mesru bir yuvarlama gibi gorunmemeli.
    """
    from harness.denetim import _esitse           # yalnizca esitlik kurali

    rejim = _son_rejim(b)
    bas, son = _bolum_araligi(metin, rejim)
    govde = metin[bas:son]
    tum = defter.tum() if defter is not None else set()
    for eslesme in re.finditer(r"(?<![A-Za-z0-9.,])(-?\d+,\d+)", govde):
        eski = float(eslesme.group(1).replace(",", "."))
        bozuk_deger = eski * (1.0 + sapma)
        if _esitse(bozuk_deger, tum, tolerans, list(basamaklar)):
            continue          # baska bir olguya carpiyor: temiz ornek degil
        yeni = nv.sayi_bicimle(bozuk_deger)
        if _esitse(float(yeni.replace(",", ".")), tum, tolerans, list(basamaklar)):
            continue          # yuvarlanmis hali carpiyor
        bozuk = govde[:eslesme.start(1)] + yeni + govde[eslesme.end(1):]
        return metin[:bas] + bozuk + metin[son:]
    raise MutasyonUygulanamaz(
        f"{rejim} bolumunde, bozulmus hali baska bir olguya carpmayan "
        f"ondalikli sayi yok")


def _senaryo_takasi(metin: str, b: at.AjanBaglami, eczane_id: str,
                    sapma: float) -> str:
    """Taban rejimin adedini son rejimin oneri kalemine yazar.

    En sinsi hata tipi: butun sayilar GERCEK, yalnizca yanlis rejime ait.
    Bagil tolerans bunu goremez; ancak rejim etiketli defter gorebilir.
    """
    hedef = _son_rejim(b)
    taban = b.taban_rejim
    for satir in _teklifler(b, eczane_id, hedef):
        esi = next((s for s in _teklifler(b, eczane_id, taban)
                    if s["sku_id"] == satir["sku_id"]
                    and s["adet"] != satir["adet"]), None)
        if esi is None:
            continue
        yeni = "".join(
            _kalem_metni(hedef, satir, adet=esi["adet"])
            if (f"senaryo: {hedef}" in p and satir["sku_id"] in p) else p
            for p in _oneri_bloklari(metin))
        return _oneri_blogunu_degistir(metin, yeni)
    raise MutasyonUygulanamaz(
        "iki rejimde de teklif alan ve adedi FARKLI olan satir yok")


def _senaryo_takasi_uygun(b: at.AjanBaglami, eczane_id: str) -> bool:
    try:
        hedef, taban = _son_rejim(b), b.taban_rejim
    except MutasyonUygulanamaz:
        return False
    return any(s["sku_id"] == t["sku_id"] and s["adet"] != t["adet"]
               for t in _teklifler(b, eczane_id, hedef)
               for s in _teklifler(b, eczane_id, taban))


def _veto_renkli_sku(b: at.AjanBaglami, veto_renkleri: tuple[str, ...]) -> str | None:
    for sku_id, urun in sorted(b.urunler.items()):
        if urun.get("recete_rengi") in veto_renkleri:
            return sku_id
    return None


# D6'nin yasakladigi recete renkleri. config'ten okunmuyor cunku mutasyon
# "regulasyonun yasakladigi urunu onerdim" senaryosunu kurmak zorunda;
# politikanin veto listesi gevsetilse bile bu iki renk yasak kalir
# (core/config.py capraz kontrolu regulasyonun altina inmeyi zaten yasakliyor).
REGULASYON_RENKLERI: tuple[str, ...] = ("KIRMIZI", "YESIL")


def _vetolanmis_sku(b: at.AjanBaglami, eczane_id: str) -> tuple[str, str] | None:
    """(rejim, sku_id) -- kisit katmaninin fiilen vetoladigi ilk satir."""
    for rejim in b.rejimler:
        for satir in b.vetolar.get((rejim, eczane_id), []):
            return rejim, satir["sku_id"]
    return None


def _veto_onerisi(metin: str, b: at.AjanBaglami, eczane_id: str,
                  sapma: float) -> str:
    """Vetolanmis bir urunu oneri kaydina ekler. D6'nin ihlali.

    ONCELIK kirmizi/yesil receteli urunde: regulasyonun yasakladigi urun
    hicbir kosulda onerilemez ve bu en agir ihlal. Ama HER dunyada boyle
    bir urun bulunmuyor -- 100 SKU'luk `fast` dunyasinda ikinci bir seed'de
    hic cikmadi ve mutasyon uygulanamadi, yani denetci o kosuda KANITSIZ
    kaldi. Bu yuzden geri dusus var: kisit katmaninin herhangi bir sebeple
    vetoladigi bir satir onerilir. Ikisi de `kisit_ihlali` uretir; fark
    yalnizca ihlalin agirligindadir.
    """
    ornek = next((s for r in b.rejimler for s in _teklifler(b, eczane_id, r)), None)
    if ornek is None:
        raise MutasyonUygulanamaz("kalem sablonu icin teklif yok")
    sku_id = _veto_renkli_sku(b, REGULASYON_RENKLERI)
    rejim = b.taban_rejim
    if sku_id is None:
        vetolu = _vetolanmis_sku(b, eczane_id)
        if vetolu is None:
            raise MutasyonUygulanamaz(
                "ne kirmizi/yesil receteli urun ne de vetolanmis satir var")
        rejim, sku_id = vetolu
    kalem = _kalem_metni(rejim, ornek, sku_id=sku_id)
    return _oneri_blogunu_degistir(metin, "".join(_oneri_bloklari(metin)) + kalem)


def _veto_onerisi_uygun(b: at.AjanBaglami, eczane_id: str) -> bool:
    if not _teklifi_olan(b, eczane_id):
        return False
    return (_veto_renkli_sku(b, REGULASYON_RENKLERI) is not None
            or _vetolanmis_sku(b, eczane_id) is not None)


def _mf_kanali_ihlali(metin: str, b: at.AjanBaglami, eczane_id: str,
                      sapma: float) -> str:
    """MF kanali KAPALI bir satira mal fazlasi yazar (SGK kapsami, SPEC 2.5)."""
    for rejim in b.rejimler:
        for satir in _teklifler(b, eczane_id, rejim):
            if satir["mf_kanali_acik"]:
                continue
            yeni = "".join(
                # Herhangi bir SIFIR OLMAYAN MF bu mutasyonu kurar: iddia
                # "kapali kanala mal fazlasi yazildi". Buyuklugu onemsiz,
                # bu yuzden `mutasyon_sapmasi` knob'ina baglanmadi -- o knob
                # SAYI bozan mutasyonlarin genligi icin.
                _kalem_metni(rejim, satir, mf_orani=MF_IHLAL_ORANI,
                             bedava_adet=MF_IHLAL_BEDAVA)
                if (f"senaryo: {rejim}" in p and satir["sku_id"] in p) else p
                for p in _oneri_bloklari(metin))
            return _oneri_blogunu_degistir(metin, yeni)
    raise MutasyonUygulanamaz("bu eczanede MF kanali kapali teklif satiri yok")


def _mf_kapali_var(b: at.AjanBaglami, eczane_id: str) -> bool:
    return any(not s["mf_kanali_acik"] for r in b.rejimler
               for s in _teklifler(b, eczane_id, r))


def _lot_karistirma(metin: str, b: at.AjanBaglami, eczane_id: str,
                    sapma: float) -> str:
    """Oneriye VAR OLAN ama tahsis edilmemis bir lot yazar.

    Hallusinasyon degil: lot gercek. Yanlis olan sey teklif satirinin lot
    referansi -- FEFO'nun sectigi lot degil baskasi gonderiliyor.
    """
    for rejim in b.rejimler:
        for satir in _teklifler(b, eczane_id, rejim):
            baska = next((l for l in sorted(b.lotlar)
                          if b.lotlar[l]["sku_id"] == satir["sku_id"]
                          and l != satir["lot_id"]), None)
            if baska is None:
                continue
            yeni = "".join(
                _kalem_metni(rejim, satir, lot_id=baska)
                if (f"senaryo: {rejim}" in p and satir["sku_id"] in p) else p
                for p in _oneri_bloklari(metin))
            return _oneri_blogunu_degistir(metin, yeni)
    raise MutasyonUygulanamaz("ayni SKU'da ikinci lot yok")


def _lot_ikizi_var(b: at.AjanBaglami, eczane_id: str) -> bool:
    return any(any(b.lotlar[l]["sku_id"] == s["sku_id"] and l != s["lot_id"]
                   for l in b.lotlar)
               for r in b.rejimler for s in _teklifler(b, eczane_id, r))


def _uydurma_oneri(metin: str, b: at.AjanBaglami, eczane_id: str,
                   sapma: float) -> str:
    """Politikanin teklif URETMEDIGI bir (eczane, urun) satirini onerir."""
    onerilenler = {s["sku_id"] for r in b.rejimler
                   for s in _teklifler(b, eczane_id, r)}
    vetolananlar = {v["sku_id"] for r in b.rejimler
                    for v in b.vetolar.get((r, eczane_id), [])}
    aday = next((s for s in sorted(b.urunler)
                 if s not in onerilenler and s not in vetolananlar
                 and b.urunler[s].get("recete_rengi") not in REGULASYON_RENKLERI),
                None)
    ornek = next((s for r in b.rejimler for s in _teklifler(b, eczane_id, r)), None)
    if aday is None or ornek is None:
        raise MutasyonUygulanamaz("onerilmemis urun ya da kalem sablonu yok")
    kalem = _kalem_metni(b.taban_rejim, ornek, sku_id=aday)
    return _oneri_blogunu_degistir(metin, "".join(_oneri_bloklari(metin)) + kalem)


def _hayali_kimlik(onek: str):
    def uygula(metin: str, b: at.AjanBaglami, eczane_id: str, sapma: float) -> str:
        desen = re.compile(rf"\b{onek}\d+\b")
        eslesme = desen.search(metin)
        if eslesme is None:
            raise MutasyonUygulanamaz(f"{onek} kimligi metinde gecmiyor")
        hayali = f"{onek}{HAYALI_SONEK}"
        return metin[:eslesme.start()] + hayali + metin[eslesme.end():]
    return uygula


def _kimlik_var(onek: str):
    def uygun(b: at.AjanBaglami, eczane_id: str) -> bool:
        kumeler = {"ECZ": b.eczaneler, "SKU": b.urunler, "LOT": b.lotlar}
        return f"{onek}{HAYALI_SONEK}" not in kumeler.get(onek, {})
    return uygun


def _bolum_silme(metin: str, b: at.AjanBaglami, eczane_id: str,
                 sapma: float) -> str:
    """Bir rejim bolumunu tamamen siler. D3'un "hepsini yaz" sarti."""
    rejim = _son_rejim(b)
    bas, son = _bolum_araligi(metin, rejim)
    return metin[:bas] + metin[son:]


def _her_zaman(b: at.AjanBaglami, eczane_id: str) -> bool:
    return True


def _teklifi_olan(b: at.AjanBaglami, eczane_id: str) -> bool:
    return any(_teklifler(b, eczane_id, r) for r in b.rejimler)


MUTASYONLAR: dict[str, Mutasyon] = {
    m.ad: m for m in (
        Mutasyon("sayi_bozma", "sayi_uydurma",
                 "Rejim bolumundeki bir olcum degerini sapma kadar kaydirir",
                 _sayi_bozma, _teklifi_olan),
        Mutasyon("senaryo_takasi", "senaryo_karismasi",
                 "Taban rejimin adedini baska rejimin oneri kalemine yazar",
                 _senaryo_takasi, _senaryo_takasi_uygun),
        Mutasyon("veto_onerisi", "kisit_ihlali",
                 "Vetolanmis urunu oneri kaydina ekler (once kirmizi/yesil "
                 "recete, yoksa kisit katmaninin vetoladigi satir) (D6)",
                 _veto_onerisi, _veto_onerisi_uygun),
        Mutasyon("mf_kanali_ihlali", "kisit_ihlali",
                 "MF kanali kapali satira mal fazlasi yazar (SGK kapsami)",
                 _mf_kanali_ihlali, _mf_kapali_var),
        Mutasyon("lot_karistirma", "kisit_ihlali",
                 "Oneriye tahsis edilmemis (ama var olan) bir lot yazar",
                 _lot_karistirma, _lot_ikizi_var),
        Mutasyon("uydurma_oneri", "uydurma_oneri",
                 "Politikanin uretmedigi bir satiri onerir",
                 _uydurma_oneri, _teklifi_olan),
        Mutasyon("hayali_eczane", "hallusinasyon",
                 "Metindeki bir eczane kimligini var olmayanla degistirir",
                 _hayali_kimlik("ECZ"), _kimlik_var("ECZ")),
        Mutasyon("hayali_sku", "hallusinasyon",
                 "Metindeki bir urun kimligini var olmayanla degistirir",
                 _hayali_kimlik("SKU"), _kimlik_var("SKU")),
        Mutasyon("hayali_lot", "hallusinasyon",
                 "Metindeki bir lot kimligini var olmayanla degistirir",
                 _hayali_kimlik("LOT"), _kimlik_var("LOT")),
        Mutasyon("bolum_silme", "bicim_ihlali",
                 "Bir rejim bolumunu siler (D3: hepsi yazilmak zorunda)",
                 _bolum_silme, _her_zaman),
    )
}


def uygula(ad: str, metin: str, b: at.AjanBaglami, eczane_id: str, sapma: float,
           defter=None, tolerans: float = 0.0,
           basamaklar: tuple[int, ...] = ()) -> str:
    """Mutasyonu uygular. Defter/tolerans yalnizca `sayi_bozma`nin isine yarar
    ama imza ortak tutuldu; her mutasyonun kendi cagri sekli olsaydi vaka
    kosucusu mutasyon adina gore dallanmak zorunda kalirdi."""
    if ad not in MUTASYONLAR:
        raise KeyError(f"bilinmeyen mutasyon: {ad}. Gecerli: {sorted(MUTASYONLAR)}")
    m = MUTASYONLAR[ad]
    if m.ad == "sayi_bozma":
        return m.uygula(metin, b, eczane_id, sapma, defter, tolerans, basamaklar)
    return m.uygula(metin, b, eczane_id, sapma)


def uygun_eczane(ad: str, b: at.AjanBaglami, adaylar: list[str]) -> str:
    """Mutasyonun uygulanabilecegi ILK eczane. Sirali, dolayisiyla sabit."""
    m = MUTASYONLAR[ad]
    for eczane_id in adaylar:
        if m.uygun(b, eczane_id):
            return eczane_id
    raise MutasyonUygulanamaz(
        f"'{ad}' mutasyonu {len(adaylar)} eczanenin hicbirinde uygulanamiyor")
