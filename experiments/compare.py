"""Iki kosuyu yan yana koy, farki ve istatistiksel anlamliligini ver. SPEC 5b.2.

    python -m experiments.compare --a fast_53a94904_1 --b fast_9c11aa02_1
    python -m experiments.compare --sweep _sweep_...  --a 4 --b 8

Iki karsilastirma rejimi var ve hangisinin gecerli oldugu VERI TARAFINDAN
belirlenir:

  ESLESMIS (paired) : iki kosunun dunya hash'i ayni. Yalnizca model/feature
      knob'u degismis demektir; ayni satirlar ustunde eslesmis bootstrap
      yapilir. Fark varyansi kucuktur, kucuk farklar bile ayirt edilir.
  ESLESMEMIS        : dunyalar farkli (sim knob'u veya seed degismis).
      Satir eslemesi yoktur; fark yalnizca seed'ler arasi dagilimla
      degerlendirilebilir. Tek seed varsa "anlamlilik yok" denir - uydurma
      guven araligi uretilmez.

Bu ayrim onemli: iki farkli dunyanin metriklerini eslesmis sanip dar bir
guven araligi vermek, M6'da OPE'nin yapacagi hatanin kucuk bir provasidir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from core.config import load_config
from eval import metrics as mt
from experiments.run import KOSU_DIZINI

# Eslesmis bootstrap'ta karsilastirilan olcutler: (fonksiyon, hangi kolon).
# Ayirt etme metrikleri HAM SKOR uzerinden olculur (metrikler.json ile ayni
# tanim); kalibrasyon metrikleri olasilik uzerinden. Kova kalibrasyonu
# siralamayi kabalastirdigi icin ikisini karistirmak AUC'yi olduren bir
# tutarsizlik yaratirdi.
OLCUTLER = {
    "auc": (mt.guvenli_auc, "skor"),
    "pr_auc": (mt.guvenli_pr_auc, "skor"),
    "brier": (mt.brier, "olasilik"),
}


def _oku(kosu_id: str, kok: Path) -> dict:
    yol = kok / kosu_id / "metrikler.json"
    if not yol.is_file():
        raise SystemExit(f"kosu bulunamadi: {yol}")
    return json.loads(yol.read_text(encoding="utf-8"))


def duz_karsilastir(a: dict, b: dict) -> pl.DataFrame:
    anahtarlar = sorted(set(a["duz"]) | set(b["duz"]))
    satirlar = []
    for k in anahtarlar:
        va, vb = a["duz"].get(k), b["duz"].get(k)
        if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            continue
        satirlar.append({
            "metrik": k, "A": round(float(va), 4), "B": round(float(vb), 4),
            "B-A": round(float(vb) - float(va), 4),
            "%": round(100 * (float(vb) - float(va)) / abs(float(va)), 1) if va else None,
        })
    return pl.DataFrame(satirlar)


def eslesmis_bootstrap(a_dizin: Path, b_dizin: Path, tahminciler: list[str],
                       tekrar: int, seed: int) -> pl.DataFrame:
    """Ayni dunyada kosmus iki kosuyu satir bazinda karsilastirir."""
    ta = pl.read_parquet(a_dizin / "tahminler.parquet")
    tb = pl.read_parquet(b_dizin / "tahminler.parquet")
    anahtar = ["eczane_id", "sku_id", "origin"]
    birlesik = ta.join(tb, on=anahtar, how="inner", suffix="_b")
    birlesik = birlesik.filter(pl.col("olcume_dahil"))
    if birlesik.height == 0:
        raise SystemExit("eslesen olcum satiri yok")
    # Karar ufkundaki gercek olay etiketi tahminler.parquet'te hazir.
    y = birlesik["gercek_karar_olayi"].to_numpy().astype(int)
    # Blok bootstrap: bagimsizlik birimi satir degil hucre (bkz. eval/metrics.py).
    hucre = np.unique((birlesik["eczane_id"] + "|" + birlesik["sku_id"]).to_numpy(),
                      return_inverse=True)[1]
    satirlar = []
    for ad in tahminciler:
        for olcut_adi, (olcut, kolon) in OLCUTLER.items():
            sa, sb = f"{ad}_{kolon}", f"{ad}_{kolon}_b"
            if sa not in birlesik.columns or sb not in birlesik.columns:
                continue
            pa, pb = birlesik[sa].to_numpy(), birlesik[sb].to_numpy()
            fark, alt, ust = mt.bootstrap_farki(y, pb, pa, olcut, tekrar, seed,
                                                grup=hucre)
            satirlar.append({
                "tahminci": ad, "olcut": olcut_adi,
                "A": round(olcut(y, pa), 4), "B": round(olcut(y, pb), 4),
                "B-A": round(fark, 4), "%2.5": round(alt, 4), "%97.5": round(ust, 4),
                "anlamli": bool(np.isfinite(alt) and (alt > 0) == (ust > 0)),
            })
    return pl.DataFrame(satirlar)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="kosu_id (veya --sweep ile deger)")
    ap.add_argument("--b", required=True)
    ap.add_argument("--sweep", default=None,
                    help="sweep dizini; --a/--b o taramadaki knob degerleridir")
    ap.add_argument("--metrik-filtre", default=None, help="metrik adinda gecen metin")
    args = ap.parse_args()

    cfg = load_config("fast")
    d = cfg.tukenme.degerlendirme

    if args.sweep:
        _sweep_karsilastir(args)
        return

    a, b = _oku(args.a, KOSU_DIZINI), _oku(args.b, KOSU_DIZINI)
    print(f"A = {args.a}  (config {a['config_hash']}, dunya {a['dunya_hash']}, "
          f"seed {a['temel_seed']}, knob {a['knob']})")
    print(f"B = {args.b}  (config {b['config_hash']}, dunya {b['dunya_hash']}, "
          f"seed {b['temel_seed']}, knob {b['knob']})")

    tablo = duz_karsilastir(a, b)
    if args.metrik_filtre:
        tablo = tablo.filter(pl.col("metrik").str.contains(args.metrik_filtre))
    with pl.Config(tbl_rows=200, tbl_width_chars=140):
        print(tablo)

    ayni_dunya = a["dunya_hash"] == b["dunya_hash"] and a["temel_seed"] == b["temel_seed"]
    if not ayni_dunya:
        print("\nDunyalar FARKLI (sim knob'u veya seed degismis). Satir eslemesi yok, "
              "eslesmis bootstrap yapilmiyor.\nAnlamlilik icin ayni knob degerini "
              "birden cok seed'le kosturun: python -m experiments.sweep ...")
        return
    a_dizin, b_dizin = KOSU_DIZINI / args.a, KOSU_DIZINI / args.b
    if not (a_dizin / "tahminler.parquet").is_file():
        print("\ntahminler.parquet yok (--tahmin-yazma ile kosulmus); "
              "eslesmis bootstrap atlaniyor.")
        return
    print("\nDunyalar AYNI -> eslesmis blok bootstrap (hucre bazinda):")
    with pl.Config(tbl_rows=60, tbl_width_chars=140):
        print(eslesmis_bootstrap(a_dizin, b_dizin, sorted(a["tahminciler"]),
                                 d.bootstrap_orneklem, d.bootstrap_seed))


def _sweep_karsilastir(args) -> None:
    """Bir taramanin iki knob degerini seed'ler arasi dagilimla karsilastirir."""
    dizin = KOSU_DIZINI / args.sweep
    # `deger` sayisal degerlerde i64 olarak geri okunuyor; komut satirindan
    # gelen deger her zaman metin. Karsilastirma metin uzerinden yapilir.
    tablo = pl.read_csv(dizin / "tablo.csv").with_columns(
        pl.col("deger").cast(pl.Utf8))
    a = tablo.filter(pl.col("deger") == args.a)
    b = tablo.filter(pl.col("deger") == args.b)
    if a.height == 0 or b.height == 0:
        raise SystemExit(f"deger bulunamadi. Mevcut: {tablo['deger'].unique().to_list()}")
    metrikler = [k for k in tablo.columns if k not in ("deger", "seed")]
    satirlar = []
    for m in metrikler:
        va = a[m].to_numpy().astype(float)
        vb = b[m].to_numpy().astype(float)
        va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
        if va.size == 0 or vb.size == 0:
            continue
        fark = float(vb.mean() - va.mean())
        # Seed'ler bagimsiz kosular: Welch tarzi standart hata.
        sh = float(np.sqrt(va.var(ddof=1) / va.size + vb.var(ddof=1) / vb.size)) \
            if min(va.size, vb.size) > 1 else float("nan")
        satirlar.append({
            "metrik": m, "A_ort": round(float(va.mean()), 4),
            "B_ort": round(float(vb.mean()), 4), "B-A": round(fark, 4),
            "std_hata": round(sh, 4) if np.isfinite(sh) else None,
            "|z|": round(abs(fark / sh), 2) if np.isfinite(sh) and sh > 0 else None,
        })
    cikti = pl.DataFrame(satirlar)
    if args.metrik_filtre:
        cikti = cikti.filter(pl.col("metrik").str.contains(args.metrik_filtre))
    print(f"sweep={args.sweep}  A={args.a} (n={a.height} seed)  B={args.b} (n={b.height} seed)")
    print("|z| > 2 kabaca %95 guven; seed sayisi kucukse bu kaba bir olcudur.")
    with pl.Config(tbl_rows=200, tbl_width_chars=140):
        print(cikti)


if __name__ == "__main__":
    main()
