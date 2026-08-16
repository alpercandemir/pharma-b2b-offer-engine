# Kalibrasyon defteri

README'nin "Kalibrasyon disiplini" bölümü: **tahminini yazmadan koşturma.** Öğrenme tahminin tuttuğu yerde değil, tutmadığı yerde oluyor.

Kullanım:
1. Egzersizi oku, komutu **henüz koşturma**
2. "Tahmin" satırlarını doldur — yön yetmez, **sayı yaz**
3. Komutu koştur
4. "Gerçek" satırlarını doldur
5. Ayrışma varsa "Neden yanıldım"ı yaz — asıl değerli kısım bu

Metriklerin tanımı `TUNING.md` → "Metrik sözlüğü" bölümünde.

---

## M1 / Egzersiz 1 — `sim.ikmal.minimum_parti_adet`: 12 → 40

**Bağlam.** Depomuz bir SKU'yu ikmal ederken en az bu kadar adet alıyor. `reports/m1.md` §3.3'te ölçüldü: imhanın çoğu kısa miatlı partiden değil, minimum parti büyüklüğünün yavaş SKU'da yarattığı zorunlu fazla stoktan geliyor. Şimdi o mekanizmayı doğrudan üçe katlıyoruz.

**Referans (varsayılan `minimum_parti_adet = 12`, `profil=full`, 3 seed):**

| metrik | değer |
|---|---|
| `imha_orani` | 0.122 |
| `iade_orani` | 0.114 |
| `karsilama_orani` | 0.924 |
| `gozlenen_pay` | 0.339 |
| `miad_reddi_orani` | 0.052 |
| `siparis_satiri` | 44.474 |

**Sorular — 40'ta ne olur?**

| metrik | tahminim | gerçek | fark |
|---|---|---|---|
| `imha_orani` | | | |
| `iade_orani` | | | |
| `karsilama_orani` | | | |
| `gozlenen_pay` | | | |

**Ek soru (sayı değil, mekanizma):** İmha artışının ne kadarı **hacim** kaynaklı (daha çok mal aldık), ne kadarı **kompozisyon** kaynaklı (imha yavaş SKU'larda yoğunlaştı)? Hangisini beklersin?
> Tahminim:

**Zor soru:** `karsilama_orani` yükselir mi düşer mi? İki karşıt kuvvet var — daha çok stok tutmak karşılamayı artırmalı, ama yaşlanan stok miad reddine takılıp karşılamayı düşürmeli. Hangisi baskın?
> Tahminim:

**Komut:**
```bash
uv run python -m scripts.verify_m1 --sadece-tarama --knob-taramasi sim.ikmal.minimum_parti_adet --degerler 1,12,40 --seeds 3
```

Kompozisyon sorusu için:
```bash
uv run python -m scripts.generate_world --profil full --kosu ex1 --knob sim.ikmal.minimum_parti_adet=40
uv run python -c "
import polars as pl
i=pl.read_parquet('data/ex1/observable/imhalar.parquet')
h=pl.read_parquet('data/ex1/ground_truth/hucre_haftalik.parquet')
hacim=h.group_by('sku_id').agg(pl.col('gercek_tuketim').sum().alias('hacim'))
d=i.group_by('sku_id').agg(pl.col('adet').sum().alias('imha')).join(hacim,on='sku_id')
d=d.with_columns(pl.col('hacim').qcut(4,labels=['Q1_yavas','Q2','Q3','Q4_hizli']).alias('dilim'))
print(d.group_by('dilim').agg(pl.col('imha').sum()).sort('dilim'))"
```

**Gerçek:**

**Neden yanıldım / tuttu:**

---

## M1 / Egzersiz 2 — `sim.envanter.emniyet_z_katsayisi`: 1.3 → 2.2

**Bağlam.** Bu **eczanenin** emniyet stoğu katsayısı, bizim değil. Eczane daha temkinli hale geliyor. `reports/m1.md` §3.2'de bu parametrenin doğru birimde olmasının eczane servis kalitesini nasıl düzelttiğini gördük (stok-sıfır %32 → %2.9). Şimdi daha da yükseltiyoruz.

**Referans (varsayılan `1.30`):**

| metrik | değer |
|---|---|
| `eczane_kayip_talep_orani` | 0.116 |
| `kapsama_hf` (eczane stoğu) | 9.5 |
| `siparis_satiri` | 44.474 |
| `karsilama_orani` (bizim) | 0.924 |
| `imha_orani` (bizim) | 0.122 |

> **Uyarı — bu egzersiz artık bir tuzak içeriyor.** `azami_kapsama_hafta = 6.0` tavanı devrede; emniyet stoğunu büyütmek tavana çarptığı andan sonra etkisiz kalabilir. Tahminini yaparken bunu hesaba kat.

**Sorular — 2.2'de ne olur?**

| metrik | tahminim | gerçek | fark |
|---|---|---|---|
| `eczane_kayip_talep_orani` | | | |
| `kapsama_hf` | | | |
| `siparis_satiri` | | | |
| `imha_orani` | | | |

**Asıl soru — kamçı (bullwhip):** Eczane daha çok tampon tutunca **bizim** karşılama oranımıza ne olur? Üç senaryo var, hangisi?

- (A) Yükselir — eczane daha öngörülebilir sipariş verir
- (B) Düşer — eczanenin tepe siparişleri bizim deponun emniyet stoğunu aşar
- (C) Değişmez — iki etki birbirini götürür

> Tahminim: ___  Gerekçem:

**Komut:**
```bash
uv run python -m scripts.verify_m1 --sadece-tarama --knob-taramasi sim.envanter.emniyet_z_katsayisi --degerler 0.6,1.3,2.2 --seeds 3
```

**Gerçek:**

**Neden yanıldım / tuttu:**

---

## M1 / Egzersiz 3 — 2D: miad toleransı × depo kapsaması (**en öğretici olan**)

**Bağlam.** SPEC §5, M5 için "en öğretici sweep" olarak `clearance.trigger_days` × `disposal_cost_per_unit` ikilisini gösteriyor. Bunun M1'deki muadili şu ikili:

- `eczane.latent_miad_toleransi.taban_gun_ort` — eczacının direnci (talep tarafı)
- `sim.ikmal.hedef_kapsama_hafta` — bizim stok derinliğimiz (arz tarafı)

Her ikisinin **tek başına** etkisi `reports/m1.md` §2'de ölçüldü. Soru bunların **etkileşimi**.

**Bilinen tek boyutlu etkiler:**

| knob | değer aralığı | `imha_orani` | `karsilama_orani` |
|---|---|---|---|
| `miad_toleransi.taban_gun_ort` | 60 → 260 | 0.113 → 0.224 | 0.970 → 0.681 |
| `ikmal.hedef_kapsama_hafta` | 4 → 14 | 0.129 → 0.167 | 0.820 → 0.954 |

**Soru 1 — etkileşim var mı?** `hedef_kapsama_hafta`'yı 4'ten 14'e çıkarmanın `imha_orani` üzerindeki etkisi, `miad_toleransi` düşükken (60 gün) ve yüksekken (260 gün) **aynı büyüklükte mi**?

- (A) Bağımsız — iki etki toplanır, kutuların hepsinde aynı artış
- (B) Çarpımsal — yüksek toleransta kapsama artışı imhayı çok daha fazla artırır
- (C) Doyumlu — yüksek toleransta imha zaten tavanda, kapsama artışı bir şey eklemez

> Tahminim: ___  Gerekçem:

**Soru 2 — sayı.** 3×3 kutunun **en kötü köşesindeki** (`tolerans=260`, `kapsama=14`) `imha_orani` ve `iade_orani` ne olur?
> Tahminim:

**Soru 3 — en iyi karşılama.** 9 kutu içinde en yüksek `karsilama_orani` hangi kombinasyonda çıkar ve kaç olur? (Dikkat: bu bir tuzak sorusu olabilir.)
> Tahminim:

**Komut** (9 kutu × 3 seed = 27 koşu, ~2 dk):
```bash
for kapsama in 4 8 14; do
  uv run python -m scripts.verify_m1 --sadece-tarama \
    --knob-taramasi eczane.latent_miad_toleransi.taban_gun_ort \
    --degerler 60,150,260 --seeds 3 \
    --sabit sim.ikmal.hedef_kapsama_hafta=$kapsama
done
```

**Gerçek (3×3 tablo — `imha_orani` / `iade_orani` / `karsilama_orani`):**

| tolerans \ kapsama | 4 | 8 | 14 |
|---|---|---|---|
| 60 | | | |
| 150 | | | |
| 260 | | | |

**Neden yanıldım / tuttu:**

**M5'e taşınan çıkarım:** *(bu ikilinin etkileşim şekli, M5'te salvage eğrisi ile temizlik eşiğinin etkileşimini tahmin etmene yarayacak — buraya not düş)*

---

## M1 / Egzersiz 4 — `sim.envanter.azami_kapsama_hafta`: 6 → tavansız

**Bağlam.** `reports/m1.md` §3.7: bu tavan olmadan eczaneler 50+ haftalık stoğa çıkıyordu. Sebep kapalı bir geri besleme döngüsü — stoksuzluk satışı sansürler → mal gelince satış sıçrar → varyans tahmini patlar → hedef stok patlar. Tavan bu döngüyü kesiyor.

**Referans (varsayılan `6.0`):** `kapsama_hf` = 9.5, `iade_orani` = 0.114, `imha_orani` = 0.122, `eczane_kayip_talep_orani` = 0.116

**Soru 1.** Tavanı 4'e indirince `kapsama_hf` ne olur? Ya 10'a çıkarınca?
> Tahminim: 4 → ___ , 10 → ___

**Soru 2 — asıl soru.** Tavanı 6'dan 10'a çıkarmak, 4'ten 6'ya çıkarmakla **aynı büyüklükte** bir etki yapar mı? (İpucu: tavan ne zaman bağlar?)
> Tahminim:

**Soru 3.** Tavan gevşeyince `kur_zirve` (kur öncesi sipariş sıçraması) artar mı azalır mı? Neden?
> Tahminim:

**Komut:**
```bash
uv run python -m scripts.verify_m1 --sadece-tarama --knob-taramasi sim.envanter.azami_kapsama_hafta --degerler 4,6,10 --seeds 3
```

**Gerçek:**

**Neden yanıldım / tuttu:**

---

## Şablon — sonraki egzersizler için

```markdown
## M<N> / <knob>: <a> → <b>
Bağlam:
Referans (varsayılan): metrik1=..., metrik2=...
Tahmin: metrik1 ~..., metrik2 ~..., metrik3 ~...
Komut: uv run python -m ...
Gerçek: metrik1=..., metrik2=..., metrik3=...
Neden yanıldım:
```

---

## M2 / Egzersiz 1 — `feature.stok.tavan_kapsama_hafta`: 30 → 2

**Bağlam.** Defter (ledger) stok tahmini `stok += sevkiyat − tahmini_tüketim` ile ilerliyor ve her hafta `hız × tavan_kapsama_hafta` değerinde kırpılıyor. Gerekçe makul görünüyordu: eczane order-up-to çalışır, elinde 6 haftalıktan fazla mal olmaz (`sim.envanter.azami_kapsama_hafta = 6`), defter de bu tavanı aşmamalı.

Şimdi tavanı **eczanenin gerçek tavanına yaklaştırıyoruz** (30 → 2 hafta). Sezgi: daha doğru bir varsayım, daha iyi bir tahmin.

> ⚠️ **Önce tahmin et.** Bu egzersizin cevabı `reports/m2.md` §7.1'de yazılı — tahminini yazmadan oraya bakma.

**Referans (varsayılan `30`, `profil=fast`, 5 seed):**

| metrik | değer |
|---|---|
| `defter.auc` | 0.549 |
| `hazard.auc` | 0.525 |
| `hazard.mae_gun` | 20.4 |
| `kural_ikili.auc` | 0.494 |

**Sorular — `2`'de ne olur?**

| metrik | tahminim | gerçek | fark |
|---|---|---|---|
| `defter.auc` | | | |
| `hazard.auc` | | | |
| `hazard.mae_gun` | | | |

**Asıl soru (mekanizma):** Defterin stok tahmini eczanenin **gerçek** stoğunun ne kadarını görüyor? (İpucu: `reports/m2.md` §3.2'de bir çarpan var.) Bu çarpanı hesaba katarsan, eczanenin 6 haftalık gerçek tavanı *defterin birimlerinde* kaç haftaya denk gelir?
> Tahminim: ___ hafta

**İkinci soru:** `hazard.auc`, `defter.auc` kadar oynar mı? Neden?
> Tahminim:

**Komut:**
```bash
uv run python -m experiments.sweep --knob feature.stok.tavan_kapsama_hafta \
    --values 2,8,30,100 --seeds 5 --sabit tukenme.degerlendirme.oracle_teshisi=false
```

**Gerçek:**

**Neden yanıldım / tuttu:**

---

## M2 / Egzersiz 2 — `eczane.latent_share_of_wallet.beta_b`: 2.4 → 5.0

**Bağlam.** Bu bir **simülatör** knob'u: büyüdükçe eczanelerin bizden aldığı pay küçülür, yani dünyanın gördüğümüz kısmı daralır. M1'de bu "modelin ne kadar kör olduğu" kadranıydı (bkz. `TUNING.md` A2). Şimdi körlüğün M2 metriklerine bedelini ölçüyoruz.

**Referans (varsayılan `2.4`, `profil=fast`, 5 seed):**

| metrik | değer |
|---|---|
| `hazard.auc` | 0.525 |
| `hazard.kalibrasyon_hatasi` | 0.092 |
| `hazard.mae_gun` | 20.4 |
| `teshis_oracle_ozellik.auc` (bilgi tavanı) | 0.739 |
| `panel.gercek_tukenme_taban_orani` | 0.081 |

**Sorular — `5.0`'da (dünya körleşiyor) ne olur?**

| metrik | tahminim | gerçek | fark |
|---|---|---|---|
| `hazard.auc` | | | |
| `hazard.kalibrasyon_hatasi` | | | |
| `hazard.mae_gun` | | | |
| `teshis_oracle_ozellik.auc` | | | |

> ⚠️ **Önce tahmin et.** Cevap `reports/m2.md` §7.2'de.

**Tuzak soru:** Yukarıdaki dört metrikten **biri iyileşecek.** Hangisi ve neden? (İpucu: model bir olayın olasılığını üretiyor, biz onu **başka** bir olayla ölçüyoruz. İki olayın taban oranları nasıl hareket eder?)
> Tahminim:

**Sağlamlık sorusu:** `teshis_oracle_ozellik.auc` (gerçek stok / gerçek hızla kurulan tavan) oynamalı mı? Oynarsa bu neyin işareti olur?
> Tahminim:

**Komut:**
```bash
uv run python -m experiments.sweep --knob eczane.latent_share_of_wallet.beta_b \
    --values 1.0,2.4,5.0 --seeds 5
```

**Gerçek:**

**Neden yanıldım / tuttu:**

---

## M2 / Egzersiz 3 — Körlüğün iki katmanı (**en öğretici olan**)

**Bağlam.** Hazard modeli gerçek tükenmede AUC **0.532** yapıyor. Gerçek stok ve gerçek hızla kurulan bilgi tavanı **0.733**. Arada **0.201**'lik bir boşluk var ve bu boşluk iki farklı körlükten geliyor:

- **Özellik körlüğü:** stok ve hızı yalnızca *kendi* siparişlerimizden çıkarabiliyoruz. Eczaneye giren malın ~%60'ı rakip depolardan geliyor ve görünmüyor.
- **Etiket körlüğü:** "gerçek tükenme" diye bir etiketimiz yok; model "bize sipariş geldi mi" ile eğitiliyor. Bu, tükenmenin hem seyreltilmiş hem kaydırılmış bir izdüşümü.

> ⚠️ **Önce tahmin et.** Cevap `reports/m2.md` §3.3'te sayıyla duruyor — üç soruyu da yazmadan oraya bakma.

**Soru 1 — pay.** 0.201'lik boşluğun yüzde kaçı özellik körlüğünden, yüzde kaçı etiket körlüğünden geliyor?
> Tahminim: özellik ___ % / etiket ___ %  Gerekçem:

**Soru 2 — sayı.** Aynı özelliklerle ama **gerçek tükenme etiketiyle** eğitilmiş bir hazard modeli hangi AUC'yi yapar? (0.532 ile 0.733 arasında bir yerde.)
> Tahminim:

**Soru 3 — karar.** Cevaba göre, M2'yi iyileştirmek için bir haftan olsa hangisine harcarsın?
- (A) Daha fazla feature mühendisliği (rakip akışını tahmin eden bir model, daha iyi stok defteri)
- (B) Daha büyük/derin model, hiperparametre araması
- (C) Etiketi değiştirmenin bir yolunu bulmak (farklı hedef, çoklu görev, teklif logu)

> Tahminim: ___  Gerekçem:

**Komut:**
```bash
uv run python -m scripts.verify_m2 --kosu full --hizli
```
Çıktıda "ikili karşılaştırmalar" tablosuna bak: `teshis_oracle_etiket` satırı etiket körlüğünü, `teshis_oracle_ozellik` satırı özellik körlüğünü verir.

**Gerçek:**

**Neden yanıldım / tuttu:**

**M4'e taşınan çıkarım:** *(cevap (C) çıktıysa, M4'te teklif logu doğduğunda hangi yeni etiketin kullanılabileceğini buraya not düş — o etiket de tükenme değil, o da yanlı olacak; hangi yönde?)*

---

## M3 / Egzersiz 1 — `politika.aday.miad_baskisi_agirligi`: 0.3 → 3.0

**Bağlam.** Miad baskısı, depoda kısa miatlı lotu olan SKU'yu aday havuzunda öne çeker: `skor *= (1 + ağırlık × baskı)`. Baskı = kısa miatlı adet / eldeki adet, eşik `miad_baskisi_esik_gun = 180`. Kısıt katmanı ise `asgari_kalan_raf_omru_gun = 120` altındaki lotu **vetolar**.

Şimdi temizlik güdüsünü on kat artırıyoruz.

**Referans (varsayılan `miad_baskisi_agirligi = 0.3`, `profil=full`, 5 seed):**

| metrik | değer |
|---|---|
| `aday.hibrit.recall` | 0.398 |
| `aday.hibrit.yeni_recall` | 0.063 |
| `kisit.veto_orani` | 0.322 |
| `kisit.veto_raf_omru` | 0.125 |
| `kisit.liste_recall` | 0.119 |

**Sorular — ağırlık 3.0'da ne olur?**

| metrik | tahminim | gerçek | fark |
|---|---|---|---|
| `aday.hibrit.recall` | | | |
| `aday.hibrit.yeni_recall` | | | |
| `kisit.veto_raf_omru` | | | |
| `kisit.liste_recall` | | | |

**Zor soru (sayı değil, mekanizma):** `yeni_recall` yükselir mi düşer mi? İki karşıt kuvvet var — miad baskısı hedefle ilgisiz bir sinyal (düşürmeli), ama aynı zamanda tekrar/popülerlik sırasını bozuyor (yükseltmeli). Hangisi baskın?
> Tahminim:

**En zor soru:** `veto_raf_omru` ile `miad_baskisi_agirligi` aynı yönde hareket ederse bu ne anlama gelir — kısıt katmanı mı fazla katı, aday üretimi mi yanlış yere bakıyor? Düzeltmek için hangi knob'a dokunurdun?
> Tahminim:

**Komut:**
```bash
uv run python -m experiments.sweep --knob politika.aday.miad_baskisi_agirligi --values 0,0.3,1.0,3.0 --seeds 5 --asama m3 --profil full
```

**Gerçek:**

**Neden yanıldım / tuttu:**

---

## M3 / Egzersiz 2 — `politika.kisit.asgari_kalan_raf_omru_gun`: 120 → 240 (**en öğretici olan**)

**Bağlam.** Kısa miatlı malı eczaneye yıkmak zararı transfer eder (SPEC §2.5). Kısıt katmanı, teklif edilen lotun en az `asgari_kalan_raf_omru_gun` kadar kalan raf ömrü olmasını şart koşuyor. Eşiği iki katına çıkarıyoruz: daha katı bir kural, daha az teklif.

**Referans (varsayılan `120`, `profil=full`, 5 seed):**

| metrik | değer |
|---|---|
| `kisit.veto_orani` | 0.322 |
| `kisit.veto_raf_omru` | 0.125 |
| `kisit.havuz_recall` | 0.398 |
| `kisit.veto_sonrasi_recall` | 0.298 |
| **`kisit.liste_recall`** | **0.119** |

**Soru 1 — yön.** `240`'ta `kisit.liste_recall` hangi yöne gider?
> Tahminim: ☐ düşer  ☐ artar  ☐ değişmez   Gerekçem:

**Soru 2 — sayı.**

| metrik | tahminim | gerçek |
|---|---|---|
| `kisit.veto_orani` | | |
| `kisit.veto_sonrasi_recall` | | |
| `kisit.liste_recall` | | |

**Soru 3 — karar.** Diyelim ki bu kısıtı sıkılaştırmanın maliyetini patronuna raporlayacaksın. Hangi sütunu gösterirsin, neden?
- (A) `kisit.liste_recall` — sahaya çıkan listenin recall'u, sonuçta önemli olan bu
- (B) `kisit.veto_sonrasi_recall` — kısıtın doğrudan kestiği yer
- (C) `kisit.veto_orani` — kaç satır vetolandı

> Tahminim: ___  Gerekçem:

> ⚠️ **Bu egzersizin bütün değeri Soru 1 ile Soru 3 arasında.** Sayıyı görmeden yaz. Cevap `reports/m3.md` §7.2'de.

**Komut:**
```bash
uv run python -m experiments.sweep --knob politika.kisit.asgari_kalan_raf_omru_gun --values 0,60,120,240 --seeds 5 --asama m3 --profil full
```

Maskelenmeyi kendi gözünle görmek için frekans tavanını gevşetip aynı taramayı tekrarla:
```bash
uv run python -m experiments.sweep --knob politika.kisit.asgari_kalan_raf_omru_gun --values 0,60,120,240 --seeds 5 --asama m3 --profil full --sabit politika.kisit.eczane_haftalik_teklif_tavani=50
```

**Gerçek:**

**Neden yanıldım / tuttu:**

**M4/M5'e taşınan çıkarım:** *(bir kısıtın bedelini yanlış sütundan okumak, M6'da "offline tahmin +%12 dedi, gerçek −%3 çıktı" hikâyesinin küçük provası. Buraya not düş: hangi metriği hangi kısıt doyurmuş olabilir?)*

---

## M4 / Egzersiz 1 — `tepki.duyarlilik.heterojenlik_carpani`: 1.0 → 0

**Bağlam.** Bu knob, simülatörde eczane düzeyindeki bütün teklif-duyarlılığı terimlerini tek çarpanla ölçekliyor. `0` yapıldığında **her eczane MF ve vadeye birebir aynı tepkiyi veriyor** (`mf_duyarliligi = vade_duyarliligi = 1`). Gözlemlenebilir sürücüler (sosyoekonomik, ölçek, vade riski, DBS) da latent sürücüler (share_of_wallet, stokçuluk) da devre dışı kalıyor.

CLAUDE.md §7 "uplift heterojen olacak: segmentler tekliflere farklı tepki verecek" diyor. Bu egzersiz o cümlenin **marj farkı üzerindeki karşılığını** ölçüyor.

**Referans (varsayılan `heterojenlik_carpani = 1.0`, `profil=full`, 3 seed):**

| metrik | değer |
|---|---|
| `m4.oracle_marj_farki_tl` (amaç fonksiyonunun tek başına farkı) | 3.371 |
| `m4.marj_farki_tl` (fiilen elde edilen) | 4.096 |
| `m4.heterojenlik.cate_sapmasi` | 0.171 |
| `m4.cate.sira_kor_x` | 0.576 |
| `m4.propensity.artimsal_marj` | 47.940 |

**Sorular — `0`'da ne olur?**

| metrik | tahminim | gerçek | fark |
|---|---|---|---|
| `m4.oracle_marj_farki_tl` | | | |
| `m4.heterojenlik.cate_sapmasi` | | | |
| `m4.cate.sira_kor_x` | | | |
| `m4.propensity.artimsal_marj` | | | |

**Asıl soru (sayı değil, mekanizma):** Bütün eczaneler tekliflere aynı tepkiyi verirse propensity-based politika ile uplift-based politika **aynı** politika haline gelir mi? Gelmezse, geriye hangi heterojenlik kaynağı kalıyor?
> Tahminim:

**Komut:**
```bash
uv run python -m experiments.sweep --knob tepki.duyarlilik.heterojenlik_carpani \
       --values 0,0.5,1.0,2.0 --seeds 3 --asama m4 --profil full
```

> **Not:** Bu egzersizin cevabı `reports/m4.md` §7.3'te — ama **önce tahmininizi yazın.** Ben yanıldım: "marj farkı sıfıra iner" bekliyordum.

---

## M4 / Egzersiz 2 — `politika.kisit.eczane_haftalik_teklif_tavani`: 5 → 1

**Bağlam.** Eczane başına haftalık teklif slotu beşten bire iniyor. Aksiyon seçimi aynı aday havuzundan artık yalnızca **bir** satır seçebiliyor.

**Referans (varsayılan `tavan = 5`, `profil=full`, 3 seed):**

| metrik | değer |
|---|---|
| `m4.uplift_x.teklif_sayisi` | 2.987 |
| `m4.uplift_x.artimsal_marj` | 52.040 |
| `m4.oracle_marj_farki_tl` | 3.371 |
| `m4.heterojenlik.farkli_karar_orani` | 0.238 |

**Sorular — `1`'de ne olur?**

| metrik | tahminim | gerçek | fark |
|---|---|---|---|
| `m4.uplift_x.artimsal_marj` | | | |
| `m4.oracle_marj_farki_tl` | | | |
| `m4.heterojenlik.farkli_karar_orani` | | | |

**Zor soru:** Teklif sayısı beşte bire iniyor. Uplift'in propensity'ye üstünlüğü (`oracle_marj_farki_tl`) **artar mı azalır mı**? İki karşıt kuvvet var: (a) az slot = az hata payı, hedefleme daha önemli; (b) az teklif = az toplam marj, farkın mutlak büyüklüğü küçülmeli.
> Tahminim (yön ve yaklaşık büyüklük):

**Bonus soru:** `farkli_karar_orani` (iki politikanın ayrıştığı satır oranı) hangi yöne gider? Bu oranla marj farkı **aynı yönde mi** hareket eder?
> Tahminim:

**Komut:**
```bash
uv run python -m experiments.sweep --knob politika.kisit.eczane_haftalik_teklif_tavani \
       --values 1,3,5,10 --seeds 3 --asama m4 --profil full
```

---

## M5 / Egzersiz 1 — `tahsis.temizlik.tetik_gun`: 120 → 60

**Bağlam.** Temizlik tetiği, bir lotun **değerinin** düşmeye başladığı kalan raf ömrüdür. `120` iken 120 günden az kalmış her lot "temizlik penceresinde" sayılır ve devam değeri (salvage) azalmaya başlar; `60`'a çekilince pencere yarıya iner.

Salvage'in **işaret değiştirdiği** nokta kapalı formda:

```
kalan_gun* = tetik_gun × imha / (normal + imha)
           = tetik_gun × 0.08 / (0.05×0.85 + 0.08)   ≈ tetik_gun × 0.653
```

Yani `120`'de ~78 gün, `60`'ta ~39 gün. Bu eşiğin altındaki lotlarda gölge fiyat negatiftir ve LP mal çıkarmak için marjdan taviz verir.

**Referans (`tetik_gun = 120`, `profil=full`, `miad_hizlandirma_gun=60`, 3 origin):**

| metrik | değer |
|---|---|
| `m5.hedefli_temizlik.teklif_sayisi_temizlik` | 375 |
| `m5.hedefli_temizlik.ortalama_mf_temizlik` | 0.0061 (normal koşuldaki MF: 0.0018) |
| `m5.b.imha_temizlik_hedefli_farki` | −768 adet |
| `m5.b.iade_hedefli_farki` | +587 adet |
| `m5.b.net_marj_hedefli_farki` | +29.942 TL |
| `m5.golge.hedefli_temizlik.negatif_golge_lot_orani` | 0.098 |

**Sorular — `60`'ta ne olur?**

| metrik | tahminim | gerçek | fark |
|---|---|---|---|
| `m5.hedefli_temizlik.teklif_sayisi_temizlik` | | | |
| `m5.b.imha_temizlik_hedefli_farki` | | | |
| `m5.b.iade_hedefli_farki` | | | |
| `m5.b.net_marj_hedefli_farki` | | | |

**Asıl soru (SPEC §5'in "en öğretici sweep"i).** SPEC şunu iddia ediyor: *"`trigger_days` çok erken → gereksiz marj bırakılıyor; çok geç → imha patlıyor. Arada bir optimum var."* Bu dünyada **optimum var mı?** Varsa 60 ile 180 arasında mı, yoksa net marj tetiğe **monoton** mu tepki veriyor?
> Tahminim:

**Zor soru.** Tetik küçülünce temizlik teklifi sayısı düşer ama **kalanların her biri daha derin MF** taşır (çünkü pencereye giren lotlar daha kısa miatlıdır ve gölge fiyatları daha negatiftir). `ortalama_mf_temizlik` hangi yöne gider — teklif sayısıyla **aynı** yönde mi, ters mi?
> Tahminim:

**Komut:**
```bash
uv run python -m experiments.sweep --knob tahsis.temizlik.tetik_gun --values 60,90,120,180 --seeds 3 --asama m5 --profil full --sabit tahsis.senaryo.miad_hizlandirma_gun=60
```

---

## M5 / Egzersiz 2 — `tahsis.temizlik.normal_realizasyon_orani`: 0.85 → 0.30

**Bağlam.** Bu knob LP'ye şunu söyler: *"teklif etmeyip depoda bıraktığın bir adet, ileride organik talep tarafından normal marjla satılma olasılığı bu kadardır."* Yani **stoğun fırsat maliyetidir.** LP bir adedi bugün satmak için, o adedin devam değerinden (`dsf × depo_marji × bu`) daha fazlasını kazanmak zorundadır.

`ranking_only` bu maliyeti **hiç** ödemez — stoğu bedava sayar. LP'nin `ranking_only`den daha az teklif çıkarmasının tek sebebi budur.

**Referans (`0.85`, `profil=full`, `kit_stok_carpani=0.25`, 3 origin):**

| metrik | `ranking_only` | `lp` |
|---|---|---|
| talep (adet) | 19.700 | 12.913 |
| karşılanmayan (adet) | 3.985 | 2.106 |
| stockout (teklif) | 280,9 | 178,2 |
| brüt marj (TL) | 194.149 | 186.179 |
| net marj (TL) | −239.459 | −255.177 |

**Sorular — `0.30`'da ne olur?**

| metrik | tahminim | gerçek | fark |
|---|---|---|---|
| `m5.lp.talep_adet` | | | |
| `m5.lp.karsilanmayan_adet` | | | |
| `m5.lp.brut_marj` | | | |
| `m5.a.net_marj_farki` (LP − ranking_only) | | | |

**Asıl soru.** Varsayılan `0.85`'te LP, `ranking_only`den **daha az** brüt marj üretiyor ama karşılanmayan talebi yarıya indiriyor. `0.30`'da LP'nin stok tutma isteği azalacak ve daha çok teklif çıkaracak. Peki **her iki metrikte birden** `ranking_only`yi geçebilir mi, yoksa stockout ile marj arasında sabit bir takas mı var?
> Tahminim:

**Zor soru.** Bu dünyada depodaki stoğun önemli bir kısmı zaten imha ediliyor (`m5.lp.imha_adet` ≈ 72.000 adet / 3 origin). Yani **gerçek** realizasyon oranı 0.85'ten belirgin biçimde düşük. LP'nin inancını gerçeğe yaklaştırmak (0.85 → 0.30) net marjı **artırır mı**? Artırıyorsa, varsayılan neden 0.85'te bırakıldı sorusunun cevabı ne olmalı?
> Tahminim:

**Komut:**
```bash
uv run python -m experiments.sweep --knob tahsis.temizlik.normal_realizasyon_orani --values 0.30,0.60,0.85,1.00 --seeds 3 --asama m5 --profil full
```

---

## M6 / Egzersiz 1 — `ope.tahminci.kirpma_esigi`: 20 → 2 (**varyans–yanlılık takasının kendisi**)

**Bağlam.** IPS'in ağırlığı `w = 1[a = π(x)] / π_log(a|x)`. Küçük propensity'li tek bir satır tahmini tek başına taşıyabilir; `kirpma_esigi` bu ağırlığa tavan koyar. Tavan **varyansı kırar** ve **yanlılık ekler** — M6'nın merkezindeki takas budur ve sentetik dünyada ikisi de ayrı ayrı ölçülebiliyor (oracle elimizde).

Kırpmanın yanlılığı rassal değil **yönlü**: tavan yalnızca büyük ağırlıkları keser, dolayısıyla pozitif ödül rejiminde katkıları hep aşağı çeker (`tests/test_ope.py::test_kirpma_kutle_siliyor_ve_yanlilik_yonu_belli`).

**Sorular — `2`'ye indirince ne olur?**

| metrik | tahminim | gerçek | fark |
|---|---|---|---|
| `m6.teshis.uplift_x.kirpilan_kutle_orani` | | | |
| `m6.teshis.uplift_x.ess_orani` | | | |
| `m6.ayristirma.uplift_x.kirpma` | | | |
| `m6.sapma_sd.uplift_x.ips_sapma` (8 bağımsız kayıt koşusunun sd'si) | | | |
| `m6.denetim.uplift_x.ips.sapma_yuzde` | | | |

**Asıl soru.** Kırpma varyansı düşürüp yanlılığı artırıyorsa, **toplam hatayı** (|sapma|) minimize eden bir iç optimum olmalı. Bu dünyada var mı? Varsa `2` ile `100` arasında mı?
> Tahminim:

**Zor soru.** `ips_kirpmasiz` her koşuda ayrıca hesaplanıyor. Kırpma eşiğini `10^6`'ya çıkarınca `ips` ile `ips_kirpmasiz` **çakışmalı**. Çakışıyorsa kırpma kalemi sıfırdır — peki o zaman `sd_IPS` neden hâlâ `sd_DR`'den büyük olabilir?
> Tahminim:

**Komut:**
```bash
uv run python -m experiments.sweep --knob ope.tahminci.kirpma_esigi --values 2,5,20,100,1000000 --seeds 5 --profil fast --asama m4,m6 --sabit ope.rollout.ufuk_hafta=1 --sabit 'ope.rollout.raporlanan_ufuklar=[1]' --sabit ope.rollout.teklif_penceresi_hafta=1 --sabit 'ope.rollout.politikalar=["teklif_yok"]'
```

---

## M6 / Egzersiz 2 — `uplift.kayit.kesif_orani`: 0.25 → 0.05 (**CLAUDE.md'nin talimatı**)

**Bağlam.** Milestone talimatı açık: *"Offline ile online şüpheli derecede uyuşuyorsa exploration fazla geniş demektir — daralt ve tekrar koş."* Keşif oranı kayıt politikasının her izinli kola verdiği taban olasılıktır (`policy/bandit.py`); D7'nin overlap sigortasıdır.

Geniş keşif OPE'yi **kolaylaştırır**: her kol loglarda görülür, örtüşme ihlali azalır, ağırlıklar küçük kalır. Ama geniş keşif **canlıda pahalıdır** — rassal aksiyon marj yakar. Gerçek sistemlerde keşif dardır ve OPE tam da bu yüzden zordur.

**Sorular — `0.05`'e indirince ne olur?**

| metrik | tahminim | gerçek | fark |
|---|---|---|---|
| `m6.teshis.uplift_x.ortusme_ihlali_orani` | | | |
| `m6.teshis.uplift_x.agirlik_azami` | | | |
| `m6.teshis.uplift_x.ess_orani` | | | |
| `m6.denetim.uplift_x.ips.sapma_yuzde` | | | |
| `m6.ayristirma.uplift_x.ortusme_kor_deger_payi` | | | |

**Asıl soru.** Keşif daralınca tahmincinin sapması **büyümeli**. Peki hangi kalem büyür — `varyans` mı, `kirpma` mı? İkisi aynı sebebin (küçük propensity) iki yüzü; ayrışma nerede görünür?
> Tahminim:

**Zor soru.** Keşif daraldıkça `dogrudan` (sadece sonuç modeli) tahminci **etkilenmez** — önem ağırlığı kullanmıyor. O zaman dar keşifte en güvenilir tahminci `dogrudan` mı olur? Cevap "hayır" ise, `dogrudan`ın hangi hatası önem ağırlığının varyansından daha kötüdür?
> Tahminim:

**Komut:**
```bash
uv run python -m experiments.sweep --knob uplift.kayit.kesif_orani --values 0.05,0.10,0.25,0.50 --seeds 5 --profil fast --asama m4,m6
```

---

## M6 / Egzersiz 3 — `ope.rollout.ufuk_hafta`: 4 → 52 (**SPEC §5'in en öğretici anı — ve benim yanıldığım yer**)

**Bağlam.** SPEC §5 açıkça iddia ediyor: *"Kısa ufukta agresif iskonto kazanır, uzun ufukta kaybeder. Bu kontrastı görmek POC'un en öğretici anı."*

Kurulum bu iddiayı sınayacak biçimde yapıldı: teklif yalnızca **ilk 4 hafta** veriliyor (`teklif_penceresi_hafta = 4`), ölçüm penceresi 4 / 13 / 26 / 52 hafta olarak değiştiriliyor. Yani **aynı müdahale, farklı ölçüm ufku**.

İki agresif varyant var ve ayrı olmaları şart (`experiments/run.py::agresif_politika`):
- `agresif` — en derin MF + en uzun vade. MF bir **mal** maliyetidir, teklif anında peşin ödenir.
- `agresif_vade` — MF yok, yalnızca en uzun vade. Vade bir **fonlama** maliyetidir; günlük ve küçük.

**Sorular — ufuk 4'ten 52'ye çıkınca `artimsal_yuzde_son` ne olur?**

| politika | tahminim @4 | tahminim @52 | gerçek @4 | gerçek @52 |
|---|---|---|---|---|
| `uplift_x` | | | | |
| `agresif` | | | | |
| `agresif_vade` | | | | |

**Asıl soru.** SPEC'in iddiası bu dünyada **doğrulanıyor mu**? Yani agresif politika kısa ufukta pozitif, uzun ufukta negatif mi çıkıyor? Çıkmıyorsa, mekanizmanın hangi parçası eksik?
> Tahminim:

**Zor soru — ölçünün kendisi.** Net marj sevk anında yazılıyor; iade ve imha ancak gerçekleştiklerinde düşülüyor. Ufuk kesildiğinde teklif veren politikanın gönderdiği malın bir kısmı hâlâ eczane rafında: marjı yazılmış, akıbeti belli değil (`m6.gecikmeli.<politika>.terminal_riskli_pay@<ufuk>`). Bu yanlılık **hangi yönde** çalışır ve ufuk uzadıkça büyür mü, küçülür mü?
> Tahminim:

**Komut:**
```bash
uv run python -m experiments.sweep --knob ope.rollout.ufuk_hafta --values 4,13,26,52 --seeds 8 --profil fast --asama m4,m6 \
  --sabit ope.rollout.teklif_penceresi_hafta=4 --sabit 'ope.rollout.raporlanan_ufuklar=[4]' \
  --sabit 'ope.rollout.politikalar=["teklif_yok","uplift_x","agresif","agresif_vade"]'
```

---

## M6 / Egzersiz 4 — `ope.propensity.sicaklik`: 1.0 → 2.0 (**benim yanıldığım tahmin, kayda geçmiş hâliyle**)

**Bağlam.** Sıcaklık, loglanan propensity'ye `p' ∝ p^(1/T)` uygular. `T > 1` dağılımı **düzleştirir**. Bu bir tuning kadranı değil, **kontrollü bir bozma**: kalibrasyon hatasının yönünü bilerek verip IPS'in nereye kaydığını görmek için var.

**Benim ilk tahminim (yanlış çıktı, `scripts/verify_m6.py`'nin ilk hâlinde kriter olarak yazılmıştı):**
> "Düzleştirme küçük propensity'leri büyütür; ağırlık `w = 1/π` olduğu için ağırlıklar küçülür ve IPS **aşağı** kayar."

**Gerçek:** 8 politikanın 7'sinde propensity kalemi **pozitif** çıktı. Sebep tek satırda: hedef politikaların seçimlerinin %71–100'ü **kol 0** (teklif yok) ve kol 0 olasılık kütlesinin büyük kısmını taşıyor. Düzleştirme *büyük* olasılıkları küçültür — yani kol 0'ın propensity'si **düşer**, ağırlık **büyür**, IPS **yukarı** kayar.

Doğru ifade yön içermez, ölçüme bağlıdır:
```
işaret(propensity kalemi) = −işaret( E[π_kullanılan − π_log] )   eşleşen satırlar üzerinde
```

**Sorular — `0.5`'e (keskinleştirme) indirince ne olur?**

| metrik | tahminim | gerçek | fark |
|---|---|---|---|
| `m6.propensity.kalibrasyon_hatasi` | | | |
| `m6.propensity.log_orani` | | | |
| `m6.ayristirma.uplift_x.propensity` | | | |
| `m6.denetim.uplift_x.ips.sapma_yuzde` | | | |

**Asıl soru.** `T = 0.5` keskinleştirme, `T = 2.0` düzleştirme. Propensity kaleminin işareti ikisinde **ters** mi olur? Yukarıdaki formüle göre evet olmalı — ölç ve doğrula.
> Tahminim:

**Zor soru.** `ope.propensity.kaynak = tahmin` (propensity bir modelle yeniden kestiriliyor) ile `sicaklik = 2.0` arasında hangisi daha büyük sapma üretir? Biri **sistematik** bir bozma, diğeri **öğrenilmiş** bir yaklaşım hatası — hangisinin daha zararlı olduğunu ne belirler?
> Tahminim:

**Komut:**
```bash
uv run python -m experiments.sweep --knob ope.propensity.sicaklik --values 0.5,0.8,1.0,1.5,2.0 --seeds 5 --profil fast --asama m4,m6 --sabit ope.rollout.ufuk_hafta=1 --sabit 'ope.rollout.raporlanan_ufuklar=[1]' --sabit ope.rollout.teklif_penceresi_hafta=1 --sabit 'ope.rollout.politikalar=["teklif_yok"]'
```

---

## M7 / Egzersiz 1 — `senaryo.ikame_ufku_hafta`: 12 → 4

**Bağlam.** Bu knob, "depodaki bir adet satılmayıp beklenirse yeniden fiyatlanmış olarak satılabilir mi" ufkunu belirliyor. Erteleme kazancı `pay = kırp(1 − güncelleme_beklentisi_hafta / ikame_ufku, 0, 1)` ile ölçekleniyor. `4`e indirildiğinde `yuksek` rejimin güncellemesi (8 hafta) **ufkun dışında** kalıyor ve o rejimde erteleme kalemi tam olarak sıfırlanıyor.

**Referans (varsayılan `12`, `profil=fast`, 4 seed):**

| metrik | değer |
|---|---|
| `m7.senaryo.yuksek.erteleme_tl_adet` | 2,79 |
| `m7.senaryo.yuksek.artimsal_marj` | 3.003 |
| `m7.senaryo.sok.teklif_sayisi` | 206 |
| `m7.fark.sok.kol_degisen` | 425 |

**Sorular — `4`'te ne olur?**

| metrik | tahminim | gerçek | fark |
|---|---|---|---|
| `m7.senaryo.yuksek.erteleme_tl_adet` | | | |
| `m7.senaryo.yuksek.artimsal_marj` | | | |
| `m7.senaryo.sok.teklif_sayisi` | | | |
| `m7.fark.yuksek.kol_degisen` | | | |

**Ek soru (sayı değil, mekanizma).** `yuksek` rejimin erteleme kalemi sıfırlanınca o rejim **tabana mı yaklaşır**, yoksa tabandan **daha da uzaklaşır** mı? İki karşıt kuvvet var: erteleme kanalı ölüyor (tabana yaklaştırır) ama antisipasyon ve fonlama kanalları çalışmaya devam ediyor (uzaklaştırır). Hangisi baskın?
> Tahminim:

**Zor soru.** `sok` rejiminin `artimsal_marj`ı `4`te mi `26`da mı yüksek çıkar? Cevabı yazmadan önce şunu düşün: erteleme kalemi **kol 0'ın (teklif yok) marjına da** uygulanıyor ve o marj negatife dönebilir. Artımsal marj `p(a)·m(a) − p(0)·m(0)` olduğuna göre, çıkarılan terim negatifleşince fark ne yönde hareket eder?
> Tahminim:

**Komut:**
```bash
uv run python -m experiments.sweep --knob senaryo.ikame_ufku_hafta --values 4,12,26 --seeds 4 --profil fast --asama m7
```

---

## M7 / Egzersiz 2 — `harness.sayi_toleransi_bagil`: 0.005 → 0.20

**Bağlam.** Sayı denetçisinin eşleşme toleransı. `mutasyon_sapmasi = 0.25` olduğu için `0.20` hâlâ yükleme kilidini geçer (`tolerans < sapma`) ama arada yalnızca 5 puan kalır.

**Referans (varsayılan `0.005`, `profil=fast`):**

| metrik | değer |
|---|---|
| `m7.harness.temiz_bulgu` | 0 |
| `m7.harness.mutant_yakalanan` | 10 |
| `m7.harness.bulgu.sayi_uydurma` | 3 |

**Sorular — `0.20`'de ne olur?**

| metrik | tahminim | gerçek | fark |
|---|---|---|---|
| `m7.harness.temiz_bulgu` | | | |
| `m7.harness.mutant_yakalanan` | | | |
| `m7.harness.bulgu.sayi_uydurma` | | | |

**Asıl soru.** `mutant_sayi_bozma` vakası hâlâ **geçer** mi? Dikkat: mutasyon üreteci, bozulmuş değeri defterin hiçbir yerinde bulunmayan bir aday seçecek şekilde yazıldı — ve "bulunmama" testi **aynı toleransı** kullanıyor. Tolerans genişlediğinde uygun aday bulmak zorlaşır. Vakanın düşme sebebi "yakalanmadı" mı olur, "mutasyon uygulanamadı" mı?
> Tahminim:

**Zor soru.** `0.30`'a çıkarılırsa (yani `mutasyon_sapmasi`'nın üstüne) ne olur? Cevap "harness düşer" değil — **koşu hiç başlamaz**. Neden? Bu kilidin hangi hata sınıfını kapattığını bir cümleyle yaz.
> Tahminim:

**Komut:**
```bash
uv run python -m experiments.sweep --knob harness.sayi_toleransi_bagil --values 0.005,0.05,0.20 --seeds 2 --profil fast --asama m7
```

---

## M7 / Egzersiz 3 — `ajan.brifing_teklif_sayisi`: 5 → 20

**Bağlam.** Olgu paketine rejim başına kaç teklif satırı girdiğini belirliyor. Modele daha çok olgu vermek "daha iyi brifing" gibi görünüyor — ama her sayı bir uydurma fırsatı ve her satır istemde yer kaplıyor.

**Sorular — 20'ye çıkarınca ne olur?**

| metrik | tahminim | gerçek | fark |
|---|---|---|---|
| `m7.harness.temiz_bulgu` | | | |
| `m7.harness.mutant_yakalanan` | | | |
| brifing metninin karakter sayısı | | | |

**Asıl soru (bu koşuda ölçülemeyen).** Denetçi tarafında bir şey değişmeyeceğini tahmin etmek kolay — şablon üreteci uydurmuyor. Asıl soru **canlı model** için: daha büyük bir olgu paketi uydurma oranını düşürür mü, yükseltir mi? İki karşıt kuvvet: (a) modelin ihtiyaç duyduğu sayı elinde olur, uydurmasına gerek kalmaz; (b) daha çok sayı, daha çok karıştırma fırsatı. **Bu koşuda ölçülemedi** (canlı API çalıştırılmadı) ve `reports/m7.md` §8 borç #1'de yazıyor.
> Tahminim:

**Komut:**
```bash
uv run python -m harness.run --kosu fast --metin | tail -40
uv run python -m experiments.sweep --knob ajan.brifing_teklif_sayisi --values 5,20 --seeds 2 --profil fast --asama m7
```

---

## M7 / Egzersiz 4 — post-kabul sweep oturumu (tahminler koşudan ÖNCE yazıldı)

### 4a. 1D: `senaryo.ikame_ufku_hafta` = 4, 8, 12, 26 (5 seed)

Bilinen taban (TUNING M7-A1, 3 değer × 4 seed): `4 / 12 / 26`.
Yeni olan: **8** noktası ve seed sayısının 4→5 çıkması.

`8`in özel olmasının sebebi aritmetik: `yuksek` rejiminin
`guncelleme_beklentisi_hafta = 8.0` ve pay `kırp(1 − 8/ufuk, 0, 1)`.
Ufuk tam 8 olduğunda pay **tam sıfır**.

| soru | tahminim |
|---|---|
| `m7.senaryo.yuksek.erteleme_tl_adet` @ ufuk=8 | **tam 0,00**, seed sapması da 0 (sınır noktası, `4`teki ile aynı) |
| `m7.fark.yuksek.kol_degisen` @ 8 vs @ 4 | ayırt edilemez (~130–145 bandı, t < 1) — erteleme kanalı ikisinde de ölü, geriye antisipasyon+fonlama kalıyor |
| `m7.senaryo.sok.erteleme_tl_adet` sırası | 4 < 8 < 12 < 26, monoton artan ve doyuma giden (pay 0,50 → 0,75 → 0,83 → 0,92) |
| `sok kol_degisen`, 12 vs 26, **5 seed'de** | hâlâ **ayırt edilemez**. TUNING 4 seed'de t≈0,1 demişti; 5 seed hata payını yalnızca ~%11 daraltır, t≈0,1'i anlamlı yapmaz |
| `yuksek kol_degisen`, 12 vs 26 | ayrık kalır (TUNING'de 192,0 → 291,5, SH ≤ 10,5) |

### 4b. 2D: `senaryo.ikame_ufku_hafta` × `politika.kisit.asgari_kalan_raf_omru_gun`

**Etkileşim var mı, bağımsız mı — tahminim: ETKİLEŞİM VAR, ve çarpımsal.**

Gerekçe: erteleme kaleminin toplam etkisi iki çarpanın çarpımı —
(a) kalemin büyüklüğü, ufka bağlı: `kırp(1 − beklenti/ufuk, 0, 1)`,
(b) kalemin kaç satıra uygulandığı, kapıya bağlı:
`kalan_gun − beklenti×7 ≥ asgari_raf_omru`.
`asgari` yükseldikçe (b) küçülür. Kalem büyüse bile uygulanacak satır
kalmazsa ufkun etkisi **sönümlenir**.

| soru | tahminim |
|---|---|
| heatmap şekli | köşegen değil, **sağ üstte sönen** bir yüzey: yüksek `asgari`de ufuk satırı düzleşir |
| `m7.senaryo.<rejim>.bekleyemeyen_pay`, `asgari` boyunca | monoton artan; `asgari` yeterince yüksekte `sok`ta bile %3,8'den yukarı fırlar |
| ayrılabilirlik | toplamsal (additive) model **kalıntı bırakır**; en büyük kalıntı düşük-ufuk × yüksek-asgari köşesinde değil, **yüksek-ufuk × yüksek-asgari** köşesinde olur |
| yanılma senaryom | `asgari`nin makul aralığında kapı zaten bağlamıyorsa (dünyanın lot yaşı dağılımı yüzünden — TUNING M7-A2 "kapının fiilen bağlayıp bağlamadığını **dünya** belirliyor") 2D yüzey düz çıkar ve **bağımsız görünür**. O zaman bulgu "etkileşim yok" değil, "bu dünyada kapı bağlamıyor" olur |

### 4c. compare: sweep'in iki ucu (`4` vs `26`)

Dünya hash'i knob değiştiği için **farklı** → `compare.py` eşleşmemiş rejime
düşer. Tahminim: `m7.senaryo.yuksek.*` sütunlarında fark anlamlı,
`m7.senaryo.sok.teklif_sayisi`de anlamsız çıkar.

### 4d. Sonuç (koşu sonrası doldurma)

**4a — 1D.** Beş tahminin beşi de tuttu:

| tahmin | gerçek | sonuç |
|---|---|---|
| `yuksek.erteleme` @8 = tam 0, sapma 0 | `0,00 ± 0,00` | ✅ sınır aritmetiği doğrulandı |
| `yuksek kol_degisen` @4 ≡ @8 | `133,60 ± 18,27` her ikisinde de **birebir aynı** | ✅ (yalnızca "ayırt edilemez" değil, **özdeş** — beklediğimden güçlü) |
| `sok.erteleme` monoton doyan | `15,35 → 23,02 → 25,58 → 28,33` | ✅ |
| `sok kol_degisen` 12 vs 26 hâlâ ayırt edilemez | `428,0` vs `430,4`, SH ≈ 12,4, z = 0,19 | ✅ |
| `yuksek kol_degisen` 12 vs 26 ayrık | `190,4` vs `293,8`, SH ≈ 5,6 | ✅ |

**4b — 2D. Tahminim TUTMADI, ve yanılma senaryosu gerçekleşti.**
Etkileşim çarpımsal olacak dedim; iki yönlü ANOVA (5 tekrar/hücre, df=9,64)
**dört metrikte de etkileşimi reddetti**: `p = 0,89–0,99`, etkileşimin SS payı
`≤ %2,2`. Dahası `asgari`nin **ana etkisi de** anlamsız
(`kol_degisen`: F=0,92 p=0,44). Yani yanıldığım yer "çarpımsal mı toplamsal mı"
değil, bir alt katman: **bu dünyada kapı zaten bağlamıyor.**
`asgari`yi 60'tan 240'a çıkarmak `bekleyemeyen_teklif_pay`ı monoton bile
hareket ettirmiyor (`0,111 → 0,073 → 0,079 → 0,135` — sıralama gürültüde).
TUNING M7-A2'nin "kapının fiilen bağlayıp bağlamadığını **dünya** belirliyor"
cümlesi bu koşuda **ölçülmüş** oldu: `fast` profilinin lot yaşı dağılımı
kapıyı 240 günde bile kırmıyor. Etkileşim testi için önce kapıyı bağlayan bir
dünya (`lot.raf_omru.*`) gerekiyor — knob çifti yanlış değil, **rejim** yanlış.

**4c — compare.** Tahminim tuttu: `yuksek.*` sütunları anlamlı
(`artimsal_marj` z=7,3; `erteleme` z=15,5; `bedava_adet` z=17,2),
`sok.teklif_sayisi` sınırda (z=2,21 — "anlamsız" dedim, **kıl payı yanlış**).
