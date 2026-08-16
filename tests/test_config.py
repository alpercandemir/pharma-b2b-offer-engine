"""Config semasinin sihirli sayi yasagini gercekten zorladigini test eder."""

from __future__ import annotations

import shutil

import pytest
import yaml
from pydantic import ValidationError

from core.config import CONFIG_DIR, load_config


def test_profil_yuklenir():
    cfg = load_config("fast")
    assert cfg.profil.eczane_sayisi > 0
    assert len(cfg.urun.kategoriler) == len(cfg.eczane.kategori_egilimi.tablo)


def test_config_hash_kararlı():
    assert load_config("fast").hash() == load_config("fast").hash()
    assert load_config("fast").hash() != load_config("full").hash()


def test_eksik_knob_hata_verir(tmp_path):
    """Bir knob YAML'dan silinirse sessizce varsayilana dusmemeli."""
    hedef = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, hedef)
    yol = hedef / "sim.yaml"
    icerik = yaml.safe_load(yol.read_text(encoding="utf-8"))
    del icerik["talep"]["dagilim"]["sifir_sisirme"]
    yol.write_text(yaml.safe_dump(icerik), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config("fast", config_dir=hedef)


def test_fazla_knob_hata_verir(tmp_path):
    """Yanlis yazilmis anahtar sessizce yok sayilmamali."""
    hedef = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, hedef)
    yol = hedef / "sim.yaml"
    icerik = yaml.safe_load(yol.read_text(encoding="utf-8"))
    icerik["talep"]["dagilim"]["sifir_sisirmee"] = 0.3      # yazim hatasi
    yol.write_text(yaml.safe_dump(icerik), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config("fast", config_dir=hedef)


def test_gecersiz_knob_yolu_hata_verir():
    with pytest.raises(KeyError):
        load_config("fast", gecersiz_kilma={"talep.dagilim.olmayan_knob": 1})


def test_promosyon_serbest_kurali_uygulanir():
    """D6 zemini: kirmizi/yesil recete hicbir kosulda promosyona acik olmamali."""
    from core.rng import SeedBank
    from sim.products import urun_evreni_kur

    cfg = load_config("fast")
    urunler, _ = urun_evreni_kur(cfg, SeedBank(cfg.profil.temel_seed))
    veto = set(cfg.urun.promosyon_serbest_kurali.recete_rengi_vetosu)
    vetolu = urunler.filter(urunler["recete_rengi"].is_in(list(veto)))
    assert not vetolu["promosyon_serbest"].any()


def test_tuning_md_her_knobu_kapsiyor():
    """SPEC 5b.1: config'e giren her knob TUNING.md'de hesap vermeli.

    Kontrol yaprak adi uzerinden yapilir (tam yol veya sade ad). Bir knob
    eklenip TUNING.md guncellenmezse bu test duser -- kural boylece iyi
    niyete kalmaz.
    """
    from pathlib import Path

    from core.config import load_config

    cfg = load_config("full")
    metin = (Path(__file__).resolve().parent.parent / "TUNING.md").read_text(encoding="utf-8")

    def yollar(dugum, on=""):
        if isinstance(dugum, dict):
            for k, v in dugum.items():
                yield from yollar(v, f"{on}.{k}" if on else k)
        elif isinstance(dugum, list) and dugum and isinstance(dugum[0], dict):
            for k in dugum[0]:
                yield f"{on}[].{k}"
        else:
            yield on

    eksik = []
    for yol in yollar(cfg.model_dump(mode="json")):
        yaprak = yol.split(".")[-1]
        if yaprak not in metin and yol not in metin:
            eksik.append(yol)
    assert not eksik, f"TUNING.md'de hesap verilmeyen knob'lar: {sorted(eksik)}"
