# Data dictionary

What the simulator produces, table by table, with real rows from the committed sample.

`data/fast/` is checked into the repository (~400 KB) so you can inspect the shape of the world
without installing anything. It holds a complete, self-consistent world: **60 pharmacies × 100 SKUs ×
104 weeks**, generated from base seed `20260812`.

> **Do not quote numbers from `fast`.** Every figure in [`reports/`](reports/) comes from the `full`
> profile (200 × 300 × 104). The `fast` profile deliberately does not satisfy M1's exit criterion — it
> exists for sweeps, tests and fast iteration. Running `verify_m1 --kosu fast` will not reproduce the
> reported results, and that is expected, not a failure.

To generate the reference world:

```bash
uv run python -m scripts.generate_world --profil full --kosu full
```

---

## The split that matters: `observable/` vs `ground_truth/`

This is the central design idea of the project, and it is enforced by directory layout.

| | `observable/` | `ground_truth/` |
|---|---|---|
| Represents | What a real wholesaler would actually have in its systems | What is true in the world |
| Who may read it | Feature builders, models, policies | **Measurement code only** — `eval/oracle.py` |
| Contains | Orders placed with us, our shipments, lots, prices, published events | Real consumption, latent share-of-wallet, competitor orders, true depletion weeks |

The wholesaler only sees the share of demand that comes to *it*. A pharmacy works with three or four
distributors, so orders we receive are a censored view of real consumption. In this sample we see
**35% of unit demand** (135,118 units to us, 246,288 to competitors).
`share_of_wallet` — the fraction of a pharmacy's buying we capture — is
**latent by construction**: it lives in `ground_truth/` and no model may read it.

That is what makes the project's central question answerable. Off-policy evaluation normally cannot be
audited, because the counterfactual is unknown. Here it is known, so estimator error becomes a
*measurement* rather than an argument.

Every feature builder is point-in-time correct and covered by leakage guards in
`tests/test_features.py`.

## Loading

```python
import polars as pl

orders = pl.read_parquet("data/fast/observable/siparisler.parquet")
products = pl.read_parquet("data/fast/observable/urunler.parquet")
```

`data/fast/manifest.json` records the profile, both config hashes, the base seed, the per-subsystem seed
derivations, and row counts for all 19 tables.

## Keys and grain

| Entity | Key | Sample values |
|---|---|---|
| Pharmacy | `eczane_id` | `ECZ0000` … `ECZ0059` |
| Product | `sku_id` | `SKU0000` … `SKU0099` |
| Batch | `lot_id` | `LOT000000` … |
| Event | `olay_id` | `OLY0000` … |
| Time | `hafta` | Integer week index `0`–`103`; `takvim` maps it to a calendar date |

Stock is tracked at **lot** grain, not SKU grain, because expiry is a property of the batch (SPEC §2.5).
Allocation runs FEFO — first expired, first out — and every shipment line carries the lot it was served
from, so a unit stays traceable from warehouse intake to pharmacy shelf to return or disposal.

Week `0` is `2024-08-05`, a Monday. All dates are week-start Mondays.

---

# `observable/` — what the business can see

## `urunler.parquet` — product master

100 rows × 18 columns. One row per SKU, static across the run.

| Column | Type | Meaning |
|---|---|---|
| `sku_id` | str | Product identifier |
| `kategori_kod` | str | `J01` `R05` `R06` `N02` `A02` `C09` `A10` `J07` `DERMO` `TEG` `MEDIKAL` |
| `atc_kodu` | str | 5-level ATC code, e.g. `J01CA04`. **Null for non-drug categories** (TEG, dermocosmetics, medical) |
| `etken_madde` | str | Active ingredient (INN) — the equivalence group for brand-switch offers |
| `urun_tipi` | str | `RX` / `OTC` / `TEG` / `DERMOKOZMETIK` / `MEDIKAL` |
| `recete_rengi` | str | Prescription colour. `KIRMIZI` (narcotic) and `YESIL` (psychotropic) are a **hard veto** on any promotion (D6) |
| `sgk_geri_odeme` | bool | On the social security reimbursement list — restricts discount freedom |
| `titck_tedarik_guclugu` | bool | On the regulator's supply-difficulty list — an allocation problem, not a campaign one |
| `promosyon_serbest` | bool | Derived: may this product be promoted at all? Where campaign logic actually lives |
| `psf` / `dsf` | float | Retail price / wholesaler price |
| `kdv_orani` | float | VAT rate |
| `depo_kar_marji` / `eczane_kar_marji` | float | Wholesaler / pharmacy margin, tiered by price band |
| `soguk_zincir` | bool | Cold chain — drives minimum order size and a narrower expiry tolerance |
| `koli_ici_adet` | int | Units per case; free-goods ratios round to case multiples |
| `birim_hacim` | float | Unit volume |
| `its_serilestirilmis` | bool | Serialized in the national drug tracking system |

```
| `sku_id`                | `SKU0000`    |
| `kategori_kod`          | `TEG`        |
| `atc_kodu`              | `None`       |
| `etken_madde`           | `TEG-INN-11` |
| `urun_tipi`             | `TEG`        |
| `recete_rengi`          | `NORMAL`     |
| `sgk_geri_odeme`        | `False`      |
| `titck_tedarik_guclugu` | `False`      |
| `promosyon_serbest`     | `True`       |
| `psf`                   | `91.95`      |
| `dsf`                   | `69.89`      |
| `kdv_orani`             | `0.1`        |
| `depo_kar_marji`        | `0.07`       |
| `eczane_kar_marji`      | `0.24`       |
| `soguk_zincir`          | `False`      |
| `koli_ici_adet`         | `1`          |
| `birim_hacim`           | `0.22`       |
| `its_serilestirilmis`   | `False`      |
```

## `eczaneler.parquet` — pharmacy master

60 rows × 14 columns. One row per pharmacy, static across the run.

| Column | Type | Meaning |
|---|---|---|
| `eczane_id` | str | Pharmacy identifier |
| `il` / `ilce` / `semt` | str | Province / district / neighbourhood |
| `hastane_yakinligi_km` | float | Distance to the nearest hospital — **the strongest driver of prescription mix** |
| `semt_sosyoekonomik_index` | float | Neighbourhood affluence; drives dermocosmetics and supplement share |
| `turizm_bolgesi` | bool | Tourist region — 2–3× summer revenue swing |
| `nobet_rotasyon_gun` / `_ofset` | int | On-call rotation period and offset; acute demand spikes on duty nights |
| `aylik_ciro_bandi` | str | Revenue band `S` / `M` / `L` / `XL` |
| `aylik_recete_adedi` | int | Monthly prescription count |
| `vade_riski_skoru` | float | Credit risk score |
| `dbs_limiti` | float | Direct debit system credit limit — feeds the constraint layer |
| `sgk_recete_orani` | float | Reimbursed vs. out-of-pocket mix |

```
| `eczane_id`                | `ECZ0000`            |
| `il`                       | `Konya`              |
| `ilce`                     | `Konya-ILCE1`        |
| `semt`                     | `Konya-ILCE1-SEMT4`  |
| `hastane_yakinligi_km`     | `2.029`              |
| `semt_sosyoekonomik_index` | `0.563`              |
| `turizm_bolgesi`           | `False`              |
| `nobet_rotasyon_gun`       | `7`                  |
| `nobet_rotasyon_ofset`     | `4`                  |
| `aylik_ciro_bandi`         | `XL`                 |
| `aylik_recete_adedi`       | `2629`               |
| `vade_riski_skoru`         | `0.4485`             |
| `dbs_limiti`               | `2113390.27`         |
| `sgk_recete_orani`         | `0.9217`             |
```

Note what is **absent**: no `share_of_wallet`, no stockpiling tendency, no expiry tolerance. Those are
latent and live in `ground_truth/latent_eczane.parquet`.

## `siparisler.parquet` — orders placed with us

4,568 rows. Grain: (week, pharmacy, SKU). This is the primary training signal — and it is censored:
only the portion of demand that came to us appears here.

| Column | Type | Meaning |
|---|---|---|
| `hafta` | int | Week index |
| `eczane_id` / `sku_id` | str | Who ordered what |
| `talep_adet` | int | Units requested |
| `karsilanan_adet` | int | Units actually served |
| `miad_kisiti_nedeniyle_verilemeyen` | int | Units we had in stock but could not ship because the pharmacy rejected that expiry date |
| `hafta_basi_tarih` | date | Week-start date |

| hafta | eczane_id | sku_id  | talep_adet | karsilanan_adet | miad_kisiti_nedeniyle_verilemeyen | hafta_basi_tarih |
|-------|-----------|---------|------------|-----------------|-----------------------------------|------------------|
| 0     | ECZ0000   | SKU0048 | 18         | 18              | 0                                 | 2024-08-05       |
| 0     | ECZ0000   | SKU0056 | 25         | 25              | 0                                 | 2024-08-05       |
| 0     | ECZ0001   | SKU0032 | 1          | 1               | 0                                 | 2024-08-05       |
| 0     | ECZ0001   | SKU0045 | 1          | 1               | 0                                 | 2024-08-05       |

The third column is the one to notice: `miad_kisiti_nedeniyle_verilemeyen` separates "no stock" from
"stock existed but was too close to expiry for this pharmacist." Those are different failures with
different fixes, and collapsing them would hide the entire M5 problem.

## `sevkiyat_satirlari.parquet` — shipment lines

4,548 rows. Grain: (week, pharmacy, SKU, **lot**). Every shipment names the batch it came from.

| Column | Type | Meaning |
|---|---|---|
| `hafta` | int | Week index |
| `eczane_id` / `sku_id` / `lot_id` | str | Recipient, product, source batch |
| `adet` | int | Units shipped |
| `kalan_raf_omru_gun` | int | Remaining shelf life **at time of shipment**, in days |

| hafta | eczane_id | sku_id  | lot_id    | adet | kalan_raf_omru_gun |
|-------|-----------|---------|-----------|------|--------------------|
| 0     | ECZ0000   | SKU0048 | LOT000048 | 18   | 314                |
| 0     | ECZ0000   | SKU0056 | LOT000056 | 25   | 441                |
| 0     | ECZ0001   | SKU0032 | LOT000032 | 1    | 616                |
| 0     | ECZ0001   | SKU0045 | LOT000045 | 1    | 628                |

## `stok_lotlari.parquet` — batch intake

873 rows. One row per lot received into the warehouse.

| Column | Type | Meaning |
|---|---|---|
| `lot_id` / `sku_id` | str | Batch and product |
| `giris_haftasi` | int | Intake week |
| `miad_gun_indeksi` | int | Expiry as a day offset from the run start |
| `miad_tarihi` | date | Expiry date |
| `adet_giris` | int | Units received |
| `birim_maliyet` | float | Unit cost |

| lot_id    | sku_id  | giris_haftasi | miad_gun_indeksi | miad_tarihi | adet_giris | birim_maliyet |
|-----------|---------|---------------|------------------|-------------|------------|---------------|
| LOT000000 | SKU0000 | 0             | 1013             | 2027-05-15  | 1965       | 65.87         |
| LOT000001 | SKU0001 | 0             | 352              | 2025-07-23  | 12         | 98.69         |
| LOT000002 | SKU0002 | 0             | 560              | 2026-02-16  | 12         | 15.71         |
| LOT000003 | SKU0003 | 0             | 389              | 2025-08-29  | 12         | 328.68        |

`raf_omru_kalan_gun` is never stored as a column — it is derived point-in-time from `miad_tarihi` and
the run date, because a shelf-life value computed at the wrong moment is leakage.

## `iadeler.parquet` — returns

713 rows. Goods the pharmacy could not sell and sent back.

| Column | Type | Meaning |
|---|---|---|
| `hafta` | int | Week of return |
| `eczane_id` / `sku_id` | str | Who returned what |
| `iade_adet` | int | Units returned |
| `depoya_donen_adet` | int | Units actually resalable on arrival — **always lower** |
| `kalan_raf_omru_gun` | float | Remaining shelf life at return |
| `kredi_tutari` | float | Credit issued to the pharmacy |

| hafta | eczane_id | sku_id  | iade_adet | depoya_donen_adet | kalan_raf_omru_gun | kredi_tutari |
|-------|-----------|---------|-----------|-------------------|--------------------|--------------|
| 1     | ECZ0045   | SKU0051 | 10        | 7                 | 60.0               | 2063.82      |
| 2     | ECZ0014   | SKU0076 | 4         | 3                 | 33.0               | 364.18       |
| 2     | ECZ0015   | SKU0099 | 134       | 94                | 30.0               | 6927.42      |
| 2     | ECZ0038   | SKU0089 | 8         | 6                 | 25.0               | 628.37       |

The gap between `iade_adet` and `depoya_donen_adet` is where blind discounting quietly destroys value:
pushing short-dated stock does not remove the loss, it *transfers* it and adds handling cost on top.
M5 measures exactly this.

## `imhalar.parquet` — disposals

844 rows.

| Column | Type | Meaning |
|---|---|---|
| `hafta` | int | Week of disposal |
| `sku_id` / `lot_id` | str | Product and batch. **`lot_id` is null for pharmacy-origin returns** — once goods leave the warehouse the batch link is lost |
| `adet` | int | Units destroyed |
| `imha_maliyeti` | float | Disposal cost |
| `kaynak` | str | `depo_miad` (expired in our warehouse), `eczane_iadesi:eczane_miad` (returned, expired), `eczane_iadesi:cesitten_cikarma` (returned, delisted) |

| hafta | sku_id  | lot_id | adet | imha_maliyeti | kaynak                         |
|-------|---------|--------|------|---------------|--------------------------------|
| 1     | SKU0051 | null   | 7    | 206.38        | eczane_iadesi:cesitten_cikarma |
| 2     | SKU0076 | null   | 3    | 36.42         | eczane_iadesi:eczane_miad      |
| 2     | SKU0099 | null   | 94   | 692.74        | eczane_iadesi:eczane_miad      |
| 2     | SKU0089 | null   | 6    | 62.84         | eczane_iadesi:eczane_miad      |

## `olaylar.parquet` — regime events, as published

26 rows. The discrete shocks that make this domain interesting (SPEC §2.4).

| Column | Type | Meaning |
|---|---|---|
| `olay_id` | str | Event identifier |
| `tip` | str | `REFERANS_KUR_GUNCELLEME` (reference FX update), `SGK_LISTE_GUNCELLEME`, `TITCK_GERI_CEKME` (recall), `TEDARIK_KRIZI` (supply crisis), `EPIDEMI_DALGASI` (epidemic wave) |
| `kapsam` | str | `GLOBAL` / `KATEGORI_AKUT` / `SKU` |
| `hedef` | str | The affected SKU or category |
| `yururluk_hafta` / `bitis_hafta` | int | Effective and end week |
| `gorunur_hafta` | int | **The week the event becomes knowable.** Features may only read events where `gorunur_hafta <= t` |

| olay_id | tip                     | kapsam | hedef  | yururluk_hafta | bitis_hafta | gorunur_hafta |
|---------|-------------------------|--------|--------|----------------|-------------|---------------|
| OLY0000 | REFERANS_KUR_GUNCELLEME | GLOBAL | GLOBAL | 14             | 15          | 14            |
| OLY0001 | REFERANS_KUR_GUNCELLEME | GLOBAL | GLOBAL | 31             | 32          | 31            |
| OLY0002 | REFERANS_KUR_GUNCELLEME | GLOBAL | GLOBAL | 56             | 57          | 56            |
| OLY0003 | REFERANS_KUR_GUNCELLEME | GLOBAL | GLOBAL | 80             | 81          | 80            |

The separation of `gorunur_hafta` from `yururluk_hafta` is the whole point. Anticipation is *behavioural* —
pharmacies stockpile weeks before an announcement because they expect it — while the announcement itself
arrives later. A model that reads the effective week instead of the visible week gets to see the future,
which is why the anticipation window is in `ground_truth/`, not here.

## `urun_fiyat_haftalik.parquet` — weekly prices

10,400 rows = 100 SKUs × 104 weeks. Prices move when the reference rate updates.

| hafta | sku_id  | psf    | dsf    |
|-------|---------|--------|--------|
| 0     | SKU0000 | 91.95  | 69.89  |
| 0     | SKU0001 | 137.17 | 104.25 |
| 0     | SKU0002 | 23.19  | 17.39  |
| 0     | SKU0003 | 401.75 | 337.47 |

## `depo_stok_haftalik.parquet` — our warehouse stock

10,400 rows = 100 SKUs × 104 weeks. Units on hand at week start, aggregated across lots.

| hafta | sku_id  | eldeki_adet |
|-------|---------|-------------|
| 0     | SKU0000 | 3125        |
| 0     | SKU0001 | 12          |
| 0     | SKU0002 | 12          |
| 0     | SKU0003 | 21          |

## `makro_haftalik.parquet` — macro series

104 rows. `referans_avro_kuru` is the regulator's reference euro rate — it lags the market rate and
updates in steps, which is the actual driver of stockpiling behaviour (D4). `fiyat_endeksi` is the
resulting price index.

| hafta | referans_avro_kuru | fiyat_endeksi |
|-------|--------------------|---------------|
| 0     | 38.0               | 1.0           |
| 1     | 38.0               | 1.0           |
| 2     | 38.0               | 1.0           |
| 3     | 38.0               | 1.0           |

## `takvim.parquet` — calendar

104 rows. Week index to date, plus the seasonal flags the simulator uses.

| hafta | hafta_basi_tarih | yil  | ay | ramazan_payi | yil_sonu_stoklama |
|-------|------------------|------|----|--------------|-------------------|
| 0     | 2024-08-05       | 2024 | 8  | 0.0          | false             |
| 1     | 2024-08-12       | 2024 | 8  | 0.0          | false             |
| 2     | 2024-08-19       | 2024 | 8  | 0.0          | false             |
| 3     | 2024-08-26       | 2024 | 8  | 0.0          | false             |

---

# `ground_truth/` — what is actually true

**Measurement code only.** Any model or feature builder reading these files is leakage by definition.

## `hucre_haftalik.parquet` — the real world, cell by week

107,536 rows — the relevant slice of the full 624,000-cell grid. The counterfactual ledger.

| Column | Type | Meaning |
|---|---|---|
| `hafta` / `eczane_id` / `sku_id` | | Cell identity |
| `gercek_tuketim` | int | Real consumption — patients served, not units ordered from us |
| `gercek_eczane_stogu` | int | Real on-hand stock at the pharmacy |
| `karsilanmayan_hasta_talebi` | int | Patient demand lost to the pharmacy's own stockout |
| `cesitte_var` | bool | Whether the SKU is in that pharmacy's assortment at all |
| `latent_tuketim_hizi` | float | The true consumption rate the depletion model tries to infer |

| hafta | eczane_id | sku_id  | gercek_tuketim | gercek_eczane_stogu | karsilanmayan_hasta_talebi | cesitte_var | latent_tuketim_hizi |
|-------|-----------|---------|----------------|---------------------|----------------------------|-------------|---------------------|
| 0     | ECZ0000   | SKU0000 | 14             | 138                 | 0                          | true        | 60.636              |
| 0     | ECZ0000   | SKU0008 | 1              | 2                   | 0                          | true        | 1.031               |
| 0     | ECZ0000   | SKU0009 | 0              | 0                   | 0                          | false       | 1.459               |
| 0     | ECZ0000   | SKU0018 | 0              | 2                   | 0                          | true        | 0.672               |

Most cells are zero in most weeks — **70% of rows here, and still 66% once you restrict to cells the
pharmacy actually stocks**. That intermittency is deliberate: pharmaceutical distribution really does
look like this, and a simulator that smooths it away would make M2 trivially easy and teach nothing.

## `latent_eczane.parquet` — pharmacy persona

60 rows × 18 columns. Everything a model is not allowed to know about its customers.

| Column | Meaning |
|---|---|
| `share_of_wallet` | Fraction of this pharmacy's buying we capture. **The single largest uncertainty in the system** |
| `stokculuk_egilimi` | Stockpiling tendency — reaction coefficient to expected price increases |
| `miad_toleransi_gun` | Minimum remaining shelf life this pharmacist will accept |
| `kapsama_hafta` | Target stock coverage in weeks |
| `gozden_gecirme_periyodu` | Review period of the pharmacy's own (s, S) ordering policy |
| `latent_buyukluk` | Size multiplier |
| `latent_affinite_*` | Per-category affinity: `J01` `R05` `R06` `N02` `A02` `C09` `A10` `J07` `DERMO` `TEG` `MEDIKAL` |

```
| `eczane_id`               | `ECZ0000` |
| `share_of_wallet`         | `0.65686` |
| `stokculuk_egilimi`       | `0.48193` |
| `miad_toleransi_gun`      | `233.15`  |
| `kapsama_hafta`           | `2.387`   |
| `gozden_gecirme_periyodu` | `2`       |
| `latent_buyukluk`         | `1.87763` |
| `latent_affinite_J01`     | `1.29826` |
| `latent_affinite_A10`     | `2.80422` |
| `latent_affinite_DERMO`   | `0.31074` |
```

## `sow_haftalik.parquet` — share of wallet over time

6,240 rows = 60 pharmacies × 104 weeks. SOW is not static: unfilled orders erode it permanently, returns
carry their own penalty, and it recovers slowly through mean reversion. This is the mechanism behind
M6's short-horizon vs. long-horizon contrast — aggressive discounting looks profitable at 4 weeks
partly because SOW damage has not surfaced yet.

| hafta | eczane_id | share_of_wallet |
|-------|-----------|-----------------|
| 0     | ECZ0000   | 0.66686         |
| 0     | ECZ0001   | 0.36041         |
| 0     | ECZ0002   | 0.42605         |
| 0     | ECZ0003   | 0.88233         |

## `rakip_siparisleri.parquet` — competitor orders

8,588 rows. The orders that went to other distributors — the demand we never saw. Comparing this against
`siparisler` is how "observed share" is computed, and it is the reason a model trained on our orders
alone is systematically blind.

| hafta | eczane_id | sku_id  | rakip_siparis_adedi |
|-------|-----------|---------|---------------------|
| 0     | ECZ0001   | SKU0014 | 1                   |
| 0     | ECZ0001   | SKU0040 | 1                   |
| 0     | ECZ0001   | SKU0076 | 2                   |
| 0     | ECZ0001   | SKU0082 | 1                   |

## `tukenme_olaylari.parquet` — true depletion events

2,602 rows. The week each (pharmacy, SKU) pair actually hit zero. M2's hazard model is scored against
this, and it is also how the naive "did they buy in the last 30 days?" rule is shown to be worse.

| gercek_tukenme_haftasi | eczane_id | sku_id  |
|------------------------|-----------|---------|
| 0                      | ECZ0000   | SKU0048 |
| 0                      | ECZ0000   | SKU0056 |
| 0                      | ECZ0002   | SKU0005 |
| 0                      | ECZ0002   | SKU0019 |

## `olaylar_gercek.parquet` — events with their hidden anticipation

26 rows. Same as `observable/olaylar.parquet` plus two columns that must never be visible:

| Column | Meaning |
|---|---|
| `antisipasyon_baslangic_hafta` | When pharmacies *began* reacting to the expected event |
| `antisipasyon_siddeti` | Strength of the anticipatory response |

| olay_id | tip                     | yururluk_hafta | gorunur_hafta | antisipasyon_baslangic_hafta | antisipasyon_siddeti |
|---------|-------------------------|----------------|---------------|------------------------------|----------------------|
| OLY0000 | REFERANS_KUR_GUNCELLEME | 14             | 14            | 9                            | 1.0                  |
| OLY0001 | REFERANS_KUR_GUNCELLEME | 31             | 31            | 25                           | 1.0                  |
| OLY0002 | REFERANS_KUR_GUNCELLEME | 56             | 56            | 52                           | 1.0                  |
| OLY0003 | REFERANS_KUR_GUNCELLEME | 80             | 80            | 76                           | 1.0                  |

Event `OLY0000` takes effect in week 14, but pharmacies started stockpiling in week 9. The model must
infer that five-week ramp from order patterns alone — it cannot look it up.

## `latent_urun.parquet` — product popularity

100 rows. The true popularity weight driving baseline demand. Long-tailed: in this sample the top SKU is
about 32× the median.

| sku_id  | latent_populerlik |
|---------|-------------------|
| SKU0000 | 14.788            |
| SKU0001 | 0.130             |
| SKU0002 | 0.121             |
| SKU0003 | 0.535             |

---

## Regenerating

Both profiles are reproducible from seed; running twice gives byte-identical output.

```bash
uv run python -m scripts.generate_world --profil fast --kosu fast
```

```bash
uv run python -m scripts.generate_world --profil full --kosu full
```

Override any knob without editing config, for example a denser world:

```bash
uv run python -m scripts.generate_world --profil fast --kosu deneme --knob profil.eczane_sayisi=120
```

Profiles live in [`config/profiles/`](config/profiles/); every knob is catalogued in
[`TUNING.md`](TUNING.md).
