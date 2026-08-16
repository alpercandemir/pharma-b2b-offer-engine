"""LLM tasima katmani ve DETERMINIZM cozumu.

CLAUDE.md 5: "Her kosu seed'li ve tekrar uretilebilir. Iki kez calistirinca
ayni sonuc." Bir dil modeli bunu garanti etmez -- `sicaklik = 0` bile
saglayicinin altyapisinda birebir tekrar vaat etmiyor. Bu yuzden M7'nin
regresyonu CANLI API'YA BAGLI DEGIL:

    anthropic  : gercek cagri. Konusmanin tamamini (mesajlar, arac
                 cagrilari, arac girdileri, model metni) diske yazar.
    kayitli    : o kaydi oynatir. Ag yok, anahtar yok, ayni cikti.
    sablon     : LLM hic yok; brifing olgu paketinden deterministik
                 sablonla uretilir. Denetcilerin TEMIZ referansi budur.

ARAC CAGRILARI OYNATILIRKEN YENIDEN CALISTIRILIR. Kayitta modelin hangi
araci hangi girdiyle cagirdigi durur; arac SONUCU kayittan okunmaz, o anda
`agent/tools.py` tarafindan yeniden hesaplanir. Sebep: harness'in sayi
defteri araclarin GERCEK ciktisindan kurulmali. Sonuc da kayittan gelseydi
bozuk bir kayit hem metni hem defteri ayni yonde kaydirir ve "sayi uydurma"
denetimi kendi kendini onaylardi.

Bu dosya `agent/tools.py`in aksine karar katmanina bakmaz ama ona da
ihtiyac duymaz: aldigi tek sey mesaj listesi ve arac semalaridir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Kayit bicimi surumu. Kayit semasi degisirse eski kayitlar sessizce yanlis
# oynatilmasin diye dosyaya yazilir ve okurken denetlenir.
KAYIT_SURUMU = 1


@dataclass
class AracCagrisi:
    kimlik: str
    ad: str
    girdi: dict


@dataclass
class ModelTuru:
    """Modelin tek bir cevabi: metin blogu + istedigi arac cagrilari."""

    metin: str
    araclar: list[AracCagrisi] = field(default_factory=list)
    durdurma_sebebi: str = "end_turn"


class IstemciHatasi(RuntimeError):
    pass


# --------------------------------------------------------------------------
# kayitli oynatici
# --------------------------------------------------------------------------
class KayitliIstemci:
    """Diskteki konusmayi sirayla oynatir.

    `sistem`/`kullanici` metinleri kayitla KARSILASTIRILIR: prompt degistigi
    halde eski kayit oynatilirsa harness "eski cevabi yeni soruya" test
    etmis olurdu. Uyusmazlik sessiz gecmez.
    """

    def __init__(self, yol: Path, prompt_denetimi: bool = True) -> None:
        self.yol = Path(yol)
        if not self.yol.exists():
            raise IstemciHatasi(
                f"kayit bulunamadi: {self.yol}. Once `--canli --kaydet` ile "
                f"uretin ya da `ajan.istemci: sablon` kullanin.")
        self.kayit = json.loads(self.yol.read_text(encoding="utf-8"))
        if self.kayit.get("surum") != KAYIT_SURUMU:
            raise IstemciHatasi(
                f"kayit surumu uyusmuyor: {self.kayit.get('surum')} != {KAYIT_SURUMU}")
        self.prompt_denetimi = prompt_denetimi
        self._sira = 0

    def konus(self, sistem: str, mesajlar: list[dict], araclar: list[dict]) -> ModelTuru:
        if self._sira == 0 and self.prompt_denetimi:
            self._prompt_dogrula(sistem, mesajlar)
        turlar = self.kayit["turlar"]
        if self._sira >= len(turlar):
            raise IstemciHatasi(
                f"kayit tukendi ({len(turlar)} tur) ama model bir tur daha "
                f"cagrildi: {self.yol}")
        tur = turlar[self._sira]
        self._sira += 1
        return ModelTuru(
            metin=tur.get("metin", ""),
            araclar=[AracCagrisi(a["kimlik"], a["ad"], a["girdi"])
                     for a in tur.get("araclar", [])],
            durdurma_sebebi=tur.get("durdurma_sebebi", "end_turn"))

    def _prompt_dogrula(self, sistem: str, mesajlar: list[dict]) -> None:
        beklenen = self.kayit.get("istem", {})
        ilk = mesajlar[0]["content"] if mesajlar else ""
        if beklenen.get("sistem") != sistem or beklenen.get("kullanici") != ilk:
            raise IstemciHatasi(
                f"kayittaki istem guncel istemle ayni degil ({self.yol}). "
                f"Prompt degistiyse kayit yenilenmeli: eski cevabi yeni soruya "
                f"karsi test etmek regresyonu anlamsiz kilar.")


# --------------------------------------------------------------------------
# canli istemci
# --------------------------------------------------------------------------
class AnthropicIstemcisi:
    """Gercek API. `anthropic` paketi ve API anahtari gerektirir.

    Import GEC yapiliyor: temel kurulumda `anthropic` yok (pyproject'te
    istege bagli `llm` grubunda) ve harness'in kayitli modu onsuz calismali.
    """

    def __init__(self, cfg) -> None:
        try:
            import anthropic
        except ModuleNotFoundError as hata:
            raise IstemciHatasi(
                "anthropic paketi kurulu degil. `uv sync --extra llm` ile "
                "kurun; kayitli/sablon modlari bu pakete ihtiyac duymaz."
            ) from hata
        self.cfg = cfg
        self.istemci = anthropic.Anthropic()
        self.turlar: list[dict] = []
        self.istem: dict = {}

    def konus(self, sistem: str, mesajlar: list[dict], araclar: list[dict]) -> ModelTuru:
        if not self.turlar:
            self.istem = {"sistem": sistem,
                          "kullanici": mesajlar[0]["content"] if mesajlar else ""}
        cevap = self.istemci.messages.create(
            model=self.cfg.ajan.model,
            max_tokens=self.cfg.ajan.azami_token,
            temperature=self.cfg.ajan.sicaklik,
            system=sistem,
            tools=araclar,
            messages=mesajlar,
        )
        metin = "".join(b.text for b in cevap.content if b.type == "text")
        cagrilar = [AracCagrisi(b.id, b.name, dict(b.input))
                    for b in cevap.content if b.type == "tool_use"]
        self.turlar.append({
            "metin": metin,
            "araclar": [{"kimlik": c.kimlik, "ad": c.ad, "girdi": c.girdi}
                        for c in cagrilar],
            "durdurma_sebebi": cevap.stop_reason,
        })
        return ModelTuru(metin=metin, araclar=cagrilar,
                         durdurma_sebebi=cevap.stop_reason or "end_turn")

    def kaydet(self, yol: Path, ustveri: dict | None = None) -> Path:
        yol = Path(yol)
        yol.parent.mkdir(parents=True, exist_ok=True)
        govde = {"surum": KAYIT_SURUMU, "model": self.cfg.ajan.model,
                 "sicaklik": self.cfg.ajan.sicaklik,
                 "ustveri": ustveri or {}, "istem": self.istem,
                 "turlar": self.turlar}
        yol.write_text(json.dumps(govde, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        return yol


def kayit_yaz(yol: Path, istem: dict, turlar: list[dict],
              ustveri: dict | None = None, model: str = "sablon",
              sicaklik: float = 0.0) -> Path:
    """Sablon ciktisini kayit bicimine cevirir.

    Sablon uretici de bir "konusma" gibi kaydedilir ki harness tek bir yoldan
    (kayitli oynatici) beslensin: temiz vaka ile mutant vakalar ayni bicimden
    okunur, denetciler iki farkli giris yolu gormez.
    """
    yol = Path(yol)
    yol.parent.mkdir(parents=True, exist_ok=True)
    govde = {"surum": KAYIT_SURUMU, "model": model, "sicaklik": sicaklik,
             "ustveri": ustveri or {}, "istem": istem, "turlar": turlar}
    yol.write_text(json.dumps(govde, indent=2, ensure_ascii=False), encoding="utf-8")
    return yol


def istemci_kur(cfg, kayit_yolu: Path | None = None):
    """`ajan.istemci` knob'ina gore istemci. `sablon` icin None doner."""
    if cfg.ajan.istemci == "sablon":
        return None
    if cfg.ajan.istemci == "kayitli":
        if kayit_yolu is None:
            raise IstemciHatasi("kayitli istemci icin kayit yolu gerekli")
        return KayitliIstemci(kayit_yolu)
    return AnthropicIstemcisi(cfg)
