"""M7 regresyon kosusu. SPEC 3: "harness/run.py Regresyon kosusu".

    uv run python -m harness.run --kosu fast
    uv run python -m harness.run --kosu full --vaka mutant_veto_onerisi
    uv run python -m harness.run --kosu fast --kayit-uret     (fixture yenile)
    uv run python -m harness.run --kosu fast --canli --kaydet (gercek API)

CIKIS KRITERI BURADA OLCULUR. Kosu iki soruyu birden sorar:

  1. Temiz brifing sifir bulgu veriyor mu   -> yanlis alarm yok
  2. Her mutant BEKLENEN tipte bulgu veriyor mu -> denetci canli

Ikisinden biri kalirsa kosu duser (exit 1). "Butun vakalar temiz" ciktisi
tek basina bir sey soylemez; anlamli olan iki sorunun birlikte cevabidir.

AG YOK. Varsayilan yol (`ajan.istemci: kayitli|sablon`) API anahtari ya da
`anthropic` paketi istemez. Canli kosu ayri bir bayrak ve kendi kaydini
yazar; regresyon o kaydin uzerinde doner.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from agent import client as ac
from agent import narrative as nv
from agent import scenario as sc
from agent import tools as at
from core.config import Config, config_yukle
from core.io import VERI_DIZINI, Kosu
from harness import denetim as dn
from harness import mutasyon as mt

KOK = Path(__file__).resolve().parent.parent
VAKA_DOSYASI = KOK / "harness" / "cases.yaml"
# Dusen vaka basina ekrana basilan bulgu sayisi. Ekran sabiti, knob degil:
# denetimin sonucunu degil ciktinin uzunlugunu belirler.
BASILAN_BULGU = 5


# --------------------------------------------------------------------------
# vakalar
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Vaka:
    ad: str
    tip: str                      # temiz | mutant
    kaynak: str                   # sablon | kayitli | anthropic
    eczane: str = "otomatik"
    mutasyon: str | None = None
    kayit: str | None = None
    beklenen: dict[str, int] = field(default_factory=dict)
    aciklama: str = ""


def vakalari_yukle(yol: Path = VAKA_DOSYASI) -> list[Vaka]:
    govde = yaml.safe_load(yol.read_text(encoding="utf-8"))
    return [Vaka(**v) for v in govde["vakalar"]]


@dataclass
class VakaSonucu:
    vaka: Vaka
    eczane_id: str
    gecti: bool
    olcum: str
    bulgular: list[dn.Bulgu] = field(default_factory=list)
    metin: str = ""
    atlandi: bool = False


# --------------------------------------------------------------------------
# baglam kurulumu
# --------------------------------------------------------------------------
@dataclass
class HarnessBaglami:
    cfg: Config
    kosu: sc.SenaryoKosusu
    baglam: at.AjanBaglami
    adaylar: list[str]


def baglam_hazirla(cfg: Config, kosu_adi: str, kok: Path | None = None,
                   m4=None) -> HarnessBaglami:
    """M4 hattini kurar, senaryolari kosar, ajan baglamini uretir.

    `m4` disaridan verilebilir: scripts/verify_m7.py ve experiments/run.py
    ayni hatti iki kez kosmasin (M5/M6'nin M4'u yeniden kullanma disiplini).
    """
    from experiments.run import m4_boru_hatti      # gec import: dongu kirilir

    if m4 is None:
        m4 = m4_boru_hatti(cfg, kosu_adi, kok or VERI_DIZINI)
    senaryo = sc.senaryolari_kos(cfg, m4)
    baglam = at.baglam_kur(cfg, senaryo)
    return HarnessBaglami(cfg=cfg, kosu=senaryo, baglam=baglam,
                          adaylar=baglam.teklif_veren_eczaneler())


def kayit_yolu(cfg: Config, ad: str) -> Path:
    return KOK / cfg.ajan.kayit_dizini / ad.format(profil=cfg.profil.ad)


# --------------------------------------------------------------------------
# tek vaka
# --------------------------------------------------------------------------
def vaka_kos(hb: HarnessBaglami, vaka: Vaka) -> VakaSonucu:
    cfg, b = hb.cfg, hb.baglam
    try:
        eczane_id = _eczane_sec(hb, vaka)
    except mt.MutasyonUygulanamaz as hata:
        return VakaSonucu(vaka, "-", False, f"eczane secilemedi: {hata}")

    try:
        cikti = _brifing(hb, vaka, eczane_id)
    except ac.IstemciHatasi as hata:
        return VakaSonucu(vaka, eczane_id, False, f"istemci hatasi: {hata}",
                          atlandi=True)

    metin = cikti.metin
    if vaka.mutasyon:
        try:
            metin = mt.uygula(vaka.mutasyon, metin, b, eczane_id,
                              cfg.harness.mutasyon_sapmasi, cikti.defter,
                              cfg.harness.sayi_toleransi_bagil,
                              tuple(cfg.harness.yuvarlama_basamaklari))
        except mt.MutasyonUygulanamaz as hata:
            return VakaSonucu(vaka, eczane_id, False,
                              f"mutasyon uygulanamadi: {hata}")

    bulgular = dn.denetle(cfg, metin, b, cikti.defter)
    sayim = dn.tip_sayimi(bulgular)
    gecti, olcum = _degerlendir(cfg, vaka, bulgular, sayim)
    return VakaSonucu(vaka, eczane_id, gecti, olcum, bulgular, metin)


def _eczane_sec(hb: HarnessBaglami, vaka: Vaka) -> str:
    if vaka.eczane != "otomatik":
        return vaka.eczane
    if not hb.adaylar:
        raise mt.MutasyonUygulanamaz("hicbir eczanede teklif yok")
    if vaka.mutasyon:
        return mt.uygun_eczane(vaka.mutasyon, hb.baglam, hb.adaylar)
    return hb.adaylar[0]


def _brifing(hb: HarnessBaglami, vaka: Vaka, eczane_id: str) -> nv.BrifingCiktisi:
    cfg = hb.cfg
    if vaka.kaynak == "sablon":
        return nv.brifing_uret(cfg, hb.baglam, eczane_id, istemci=None)
    if vaka.kaynak == "kayitli":
        if not vaka.kayit:
            raise ac.IstemciHatasi(f"{vaka.ad}: kayitli vaka 'kayit' alani ister")
        istemci = ac.KayitliIstemci(kayit_yolu(cfg, vaka.kayit))
        return nv.brifing_uret(cfg, hb.baglam, eczane_id, istemci=istemci)
    istemci = ac.AnthropicIstemcisi(cfg)
    cikti = nv.brifing_uret(cfg, hb.baglam, eczane_id, istemci=istemci)
    if vaka.kayit:
        istemci.kaydet(kayit_yolu(cfg, vaka.kayit),
                       {"vaka": vaka.ad, "eczane_id": eczane_id,
                        "origin": hb.kosu.t, "profil": cfg.profil.ad})
    return cikti


def _degerlendir(cfg: Config, vaka: Vaka, bulgular: list[dn.Bulgu],
                 sayim: dict[str, int]) -> tuple[bool, str]:
    if vaka.tip == "temiz":
        tavan = cfg.harness.temiz_vaka_bulgu_tavani
        gecti = len(bulgular) <= tavan
        ozet = ", ".join(f"{t}={n}" for t, n in sayim.items() if n) or "bulgu yok"
        return gecti, f"{len(bulgular)} bulgu (tavan {tavan}) | {ozet}"
    eksik = [f"{tip}>={n} (gelen {sayim.get(tip, 0)})"
             for tip, n in vaka.beklenen.items() if sayim.get(tip, 0) < n]
    ozet = ", ".join(f"{t}={n}" for t, n in sayim.items() if n) or "bulgu yok"
    if eksik:
        return False, f"YAKALANMADI: {eksik} | gelen: {ozet}"
    return True, f"yakalandi | {ozet}"


# --------------------------------------------------------------------------
# kosu
# --------------------------------------------------------------------------
@dataclass
class HarnessSonucu:
    sonuclar: list[VakaSonucu]
    sure_sn: float

    @property
    def kalan(self) -> list[VakaSonucu]:
        return [s for s in self.sonuclar if not s.gecti]

    @property
    def gecen(self) -> int:
        return sum(1 for s in self.sonuclar if s.gecti)


def harness_kos(hb: HarnessBaglami, vakalar: list[Vaka]) -> HarnessSonucu:
    t0 = time.perf_counter()
    sonuclar = [vaka_kos(hb, v) for v in vakalar]
    return HarnessSonucu(sonuclar=sonuclar, sure_sn=round(time.perf_counter() - t0, 2))


def duz_metrikler(sonuc: HarnessSonucu) -> dict[str, float]:
    """Sweep tablosuna giren duz harness metrikleri."""
    duz: dict[str, float] = {
        "m7.harness.vaka_sayisi": float(len(sonuc.sonuclar)),
        "m7.harness.gecen": float(sonuc.gecen),
        "m7.harness.kalan": float(len(sonuc.kalan)),
    }
    temiz = [s for s in sonuc.sonuclar if s.vaka.tip == "temiz"]
    mutant = [s for s in sonuc.sonuclar if s.vaka.tip == "mutant"]
    duz["m7.harness.temiz_bulgu"] = float(sum(len(s.bulgular) for s in temiz))
    duz["m7.harness.mutant_yakalanan"] = float(sum(1 for s in mutant if s.gecti))
    duz["m7.harness.mutant_sayisi"] = float(len(mutant))
    for tip in dn.BULGU_TIPLERI:
        duz[f"m7.harness.bulgu.{tip}"] = float(
            sum(dn.tip_sayimi(s.bulgular).get(tip, 0) for s in sonuc.sonuclar))
    return duz


# --------------------------------------------------------------------------
# kayit uretimi
# --------------------------------------------------------------------------
def kayit_uret(hb: HarnessBaglami, eczane_id: str, yol: Path) -> Path:
    """Sablon ciktisini KAYIT bicimine cevirir (arac cagrilariyla birlikte).

    Amaci acik: canli API olmadan da oynatma yolunun -- arac dongusu, arac
    sonuclarinin yeniden hesaplanmasi, defterin araclardan kurulmasi --
    fiilen kosulmasi. Metin modelden gelmedigi icin bu bir MODEL testi
    degildir ve rapor bunu boyle yaziyor.
    """
    cfg = hb.cfg
    brifing = nv.brifing_kur(cfg, hb.baglam, eczane_id)
    metin = nv.sablon_metni(brifing)
    turlar = [
        {"metin": "", "durdurma_sebebi": "tool_use", "araclar": [
            {"kimlik": "arac_1", "ad": "senaryo_karsilastir",
             "girdi": {"eczane_id": eczane_id}}]},
        {"metin": "", "durdurma_sebebi": "tool_use", "araclar": [
            {"kimlik": f"arac_{2 + i}", "ad": "teklif_listesi",
             "girdi": {"eczane_id": eczane_id, "rejim": rejim}}
            for i, rejim in enumerate(hb.baglam.rejimler)]},
        {"metin": metin, "durdurma_sebebi": "end_turn", "araclar": []},
    ]
    return ac.kayit_yaz(
        yol,
        istem={"sistem": nv.SISTEM_ISTEMI, "kullanici": nv.kullanici_istemi(brifing)},
        turlar=turlar,
        ustveri={"uretim": "sablon", "eczane_id": eczane_id,
                 "origin": hb.kosu.t, "profil": cfg.profil.ad,
                 "config_hash": cfg.hash()})


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kosu", default="fast", help="data/<kosu> altindaki dunya")
    ap.add_argument("--profil", default=None,
                    help="varsayilan: kosu manifestindeki profil")
    ap.add_argument("--vaka", default=None, help="tek vaka adi")
    ap.add_argument("--kayit-uret", action="store_true",
                    help="kayitli vakalarin fixture'ini sablondan yeniden uret")
    ap.add_argument("--canli", action="store_true",
                    help="temiz vakayi gercek API ile kos (anthropic paketi gerekir)")
    ap.add_argument("--kaydet", default=None,
                    help="--canli ile: konusmayi bu ada kaydet")
    ap.add_argument("--metin", action="store_true", help="brifing metnini bas")
    args = ap.parse_args()

    manifest = Kosu(args.kosu).manifest_oku()
    profil = args.profil or manifest["profil"]
    cfg = config_yukle(profil)
    # `dunya_hash` M1 doneminde uretilmis kosu manifestlerinde bulunmayabilir;
    # varsa basilir, yoksa kosu durmaz (bu alan M7'nin urettigi bir sey degil).
    print(f"kosu={args.kosu} profil={profil} config_hash={cfg.hash()} "
          f"dunya_hash={manifest.get('dunya_hash', '(manifestte yok)')}")

    hb = baglam_hazirla(cfg, args.kosu)
    print(f"senaryo: origin={hb.kosu.t} politika={hb.kosu.politika} "
          f"rejimler={hb.kosu.rejim_adlari} | teklif veren eczane="
          f"{len(hb.adaylar)}")

    if args.kayit_uret:
        for vaka in vakalari_yukle():
            if vaka.kaynak != "kayitli" or not vaka.kayit:
                continue
            eczane_id = _eczane_sec(hb, vaka)
            yol = kayit_uret(hb, eczane_id, kayit_yolu(cfg, vaka.kayit))
            print(f"kayit yazildi: {yol.relative_to(KOK)} (eczane {eczane_id})")
        return

    vakalar = vakalari_yukle()
    if args.canli:
        vakalar = [Vaka(ad="canli_llm", tip="temiz", kaynak="anthropic",
                        kayit=args.kaydet)]
    if args.vaka:
        vakalar = [v for v in vakalar if v.ad == args.vaka]
        if not vakalar:
            raise SystemExit(f"vaka bulunamadi: {args.vaka}")

    sonuc = harness_kos(hb, vakalar)

    print("\n" + "=" * 132)
    print(f"{'DURUM':<8}{'VAKA':<28}{'TIP':<8}{'ECZANE':<10}OLCUM")
    print("-" * 132)
    for s in sonuc.sonuclar:
        durum = "GECTI" if s.gecti else ("ATLANDI" if s.atlandi else "KALDI")
        print(f"{durum:<8}{s.vaka.ad:<28}{s.vaka.tip:<8}{s.eczane_id:<10}{s.olcum}")
    print("=" * 132)

    for s in sonuc.kalan:
        # Dusen vaka basina basilan bulgu sayisi. Ekran sabiti: tam liste
        # `--metin` ile ya da dondurulen `sonuc`ta zaten var.
        for b in s.bulgular[:BASILAN_BULGU]:
            print(f"  [{s.vaka.ad}] {b}")
    if args.metin:
        for s in sonuc.sonuclar:
            print(f"\n--- {s.vaka.ad} ---\n{s.metin}")

    duz = duz_metrikler(sonuc)
    print(f"\nsure={sonuc.sure_sn} sn | metrikler: "
          f"{json.dumps({k: v for k, v in duz.items() if v}, ensure_ascii=False)}")
    print(f"SONUC: {sonuc.gecen}/{len(sonuc.sonuclar)} vaka gecti"
          + (f" | KALAN: {[s.vaka.ad for s in sonuc.kalan]}" if sonuc.kalan else ""))
    sys.exit(1 if sonuc.kalan else 0)


if __name__ == "__main__":
    main()
