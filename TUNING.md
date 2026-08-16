# TUNING.md

Bu POC'un knob kataloğu. SPEC §5b.1 formatı.

**Kural (SPEC §5b.1):** Bir knob `config/` altına girdiyse burada satırı olmak zorunda. Satırı yoksa knob değil, sihirli sayıdır ve koddan çıkarılır.

---

## Kapsam ve okuma kılavuzu

M1 sonunda config'te **169 skaler knob + 5 yapılandırılmış tablo bloğu** var. Hepsi burada hesap veriyor, ama üç farklı yoğunlukta:

| Bölüm | Ne içerir | Neden bu yoğunluk |
|---|---|---|
| **A** | 12 adet **tam satır** (8 alan) | Ölçülmüş, birinci derece etkili knob'lar. Her satırdaki sayı 3 seed'li süpürmeden gelir. |
| **B** | 21 adet **aile satırı** | Birlikte hareket eden parametre grupları. Her ailenin *birincil kadranı* adlandırılmış, yardımcı üyeleri (sigma / min / max / ağırlık) listelenmiş. |
| **C** | 5 adet **blok satırı** | Domain veri tabloları (kategoriler, iller, marj kademeleri, olay kataloğu, ramazan pencereleri). Bunlar tuning kadranı değil, dünya tanımıdır. |

**Neden A/B/C ayrımı:** 169 knob'un her birine 8 alanlık satır yazmak, `..._sigma: bir log-normalin standart sapması` gibi bilgi taşımayan 130 satır üretirdi — CLAUDE.md §3'ün açıkça yasakladığı şey. Bunun yerine her knob, mekanizmasını gerçekten paylaştığı ailenin satırında hesap veriyor. **Bu bir kısaltmadır ve kabulünüze sunuluyor;** aileleri tek tek açmamı isterseniz açarım.

**Teşhis komutu formatı.** Hepsi aynı koşucuyu kullanır:

```bash
uv run python -m scripts.verify_m1 --sadece-tarama --knob-taramasi <yol> --degerler a,b,c --seeds 3
```

`experiments/sweep.py` **M1'de yoktur** (SPEC §5b.2 gereği M2'de doğar). Yukarıdaki komut onun M1'deki muadili: aynı çok-seed karşılaştırmalı koşu, aynı metrik seti.

**Metrik sözlüğü** (tarama çıktısındaki kolonlar):

| Metrik | Tanım |
|---|---|
| `grid_sifir` / `sifir_orani_aktif` | Tüm (eczane, SKU, hafta) hücrelerinde / sadece çeşitte olan hücrelerde sıfır talep oranı |
| `siparis_satiri` | Bize gelen sipariş satırı sayısı — M2'nin eğitim veri hacmi |
| `siparis_p90_hucre` | Bir hücreden bize gelen sipariş sayısının p90'ı — M2'nin hücre başına sinyal derinliği |
| `karsilama_orani` | Bize gelen talebin karşıladığımız payı |
| `miad_reddi_orani` | Talebin, stok VAR ama eczacı o miadı kabul etmediği için karşılanamayan payı |
| `imha_orani` | İmha / (sevk + imha) |
| `eczane_kayip_talep_orani` | Eczanenin stoksuzluğu yüzünden kaybettiği hasta talebi payı |
| `gozlenen_pay` | Bize gelen talep / eczanenin toplam siparişi — **modelin dünyanın ne kadarını gördüğü** |
| `iade_orani` | Eczanenin satamayıp iade ettiği adet / bizim sevkiyatımız |
| `kapsama_hf` | Eczane stoğunun hafta cinsinden kapsaması (toplam stok / haftalık tüketim) |
| `kur_zirve` | Referans kur olayı öncesi sipariş hacminin taban seviyeye oranı (D4'ün ölçüsü) |

Aşağıdaki tüm sayılar `profil=full` (200×300×104), 3 seed ortalamasıdır.

---

# A. Tam satırlar — ölçülmüş, birinci derece knob'lar

### A1. `eczane.latent_miad_toleransi.taban_gun_ort`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Eczacının kabul edeceği minimum kalan raf ömrünün (gün) ortalaması. Kategori çarpanıyla ölçeklenir. FEFO kuyruğunda bu eşiğin altındaki lot, sırada en önde olsa bile o eczaneye verilemez. |
| **Varsayılan / aralık** | `150.0`, makul aralık `60–260` |
| **Artırınca** (60 → 260) | `karsilama_orani` 0.970 → 0.681, `miad_reddi_orani` 0.014 → 0.251, `imha_orani` 0.113 → 0.224, `iade_orani` 0.118 → 0.188, `gozlenen_pay` 0.357 → 0.262, `siparis_satiri` 46.161 → 36.859 |
| **Azaltınca** | 60 günde miad kısıtı fiilen kalkar (`miad_reddi_orani` %1.4), dünya M5'in çözeceği problemi kaybeder |
| **Yanlış ayarın belirtisi** | *Çok yüksek:* karşılama %70'in altına iner, elde stok varken sipariş karşılanamaz. *Çok düşük:* `miad_reddi_orani` < %2 — `miad_toleransi` ölü bir alandır, M5'in hedefli temizlik problemi anlamsızlaşır |
| **Etkileşim** | `lot.giris.kisa_miatli_parti_olasiligi` ile **çarpımsal**: kısa miatlı parti ancak eczacının eşiğinin altındaysa zarar verir. `lot.tahsis.miad_toleransi_uygulanir=false` knob'u tamamen devre dışı bırakır (karşılama 0.924 → 0.987, miad reddi 0.052 → 0.000, imha 0.122 → 0.102) |
| **Teşhis** | `uv run python -m scripts.verify_m1 --sadece-tarama --knob-taramasi eczane.latent_miad_toleransi.taban_gun_ort --degerler 60,150,260 --seeds 3` |

> **M1'in en güçlü knob'u budur.** Tek başına karşılama oranını 29 puan, miad reddini 24 puan oynatıyor.

---

### A2. `eczane.latent_share_of_wallet.beta_b`

| Alan | İçerik |
|---|---|
| **Ne yapar** | `share_of_wallet` Beta(a,b) dağılımının ikinci şekil parametresi. Büyüdükçe eczanelerin bizden aldığı pay küçülür (multi-homing derinleşir). SOW **latent'tir** — model asla göremez. |
| **Varsayılan / aralık** | `2.4` (başlangıç ortalama SOW ≈ 0.42), makul aralık `1.0–5.0` |
| **Artırınca** (1.0 → 5.0) | `gozlenen_pay` 0.506 → 0.206, `siparis_satiri` 68.248 → 26.439, `siparis_p90_hucre` 13.3 → 6.0. **Gerçek tüketim ve eczane kapsaması değişmez** (9.3 → 9.5 hafta) — sadece görünürlük daralır |
| **Azaltınca** | Dünya şeffaflaşır; 1.0'da eczane siparişlerinin yarısı bize geliyor |
| **Yanlış ayarın belirtisi** | *Çok düşük:* M2'nin tükenme MAE'si şüpheli derecede iyi çıkar — model neredeyse tam bilgiye sahiptir, D2'nin öğreteceği şey kaybolur. *Çok yüksek:* `siparis_p90_hucre` 6'nın altına iner, hazard modeli için hücre başına olay yetersiz kalır |
| **Etkileşim** | `iade_orani` 0.077 → 0.204 yükselir — payımız küçülünce sevk tabanı küçülür ama eczanenin elinde biriken satılamaz stok küçülmez. Bu oranın paydası bizim sevkiyatımızdır, dikkat |
| **Teşhis** | `uv run python -m scripts.verify_m1 --sadece-tarama --knob-taramasi eczane.latent_share_of_wallet.beta_b --degerler 1.0,2.4,5.0 --seeds 3` |

> **SPEC §5'teki "share_of_wallet gözlemlenebilirliği — açık/kapalı" knob'unun M1'deki karşılığı budur.** M1'de SOW her hâlükârda kapalıdır (ground_truth'ta durur); bu knob kapalılığın *ne kadar acıttığını* ayarlar. Açık/kapalı anahtarı feature builder'a ait olduğu için M2'de doğacak.

---

### A3. `sim.ikmal.hedef_kapsama_hafta`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Bizim deponun hedef stok kapsaması (hafta). İkmal miktarı = EWMA haftalık çıkış × bu + emniyet stoğu − eldeki. |
| **Varsayılan / aralık** | `8.0`, makul aralık `4–14` |
| **Artırınca** (4 → 14) | `karsilama_orani` 0.820 → 0.954, `gozlenen_pay` 0.308 → 0.355, ama `imha_orani` 0.129 → 0.167. Servis ile israf arasındaki doğrudan takas |
| **Azaltınca** | Karşılama %82'ye iner, sipariş rakibe kayar, sipariş verisi stoksuzlukla sansürlenir |
| **Yanlış ayarın belirtisi** | *Çok yüksek:* imha oranı %16'yı aşar ve imhaların çoğu kısa miatlı partiden değil, **normal partinin yavaş SKU'da yaşlanmasından** gelir. *Çok düşük:* karşılama %85'in altına iner |
| **Etkileşim** | `ikmal.minimum_parti_adet` ile birlikte yavaş SKU'larda zorunlu fazla stok yaratır — depo tarafı imhasının asıl kaynağı bu ikilidir |
| **Teşhis** | `uv run python -m scripts.verify_m1 --sadece-tarama --knob-taramasi sim.ikmal.hedef_kapsama_hafta --degerler 4,8,14 --seeds 3` |

> **M5'in ön izlemesi.** "Stok kıt kaynak mı, yükümlülük mü" sorusunun M1'deki tek kadranı; M5'te aynı takas LP'nin shadow price işaretine dönüşecek.

---

### A4. `sim.envanter.antisipasyon_kapsama_kazanci`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Rejim beklentisi altında kapsama hedefinin şişme katsayısı: `min(1 + stokculuk_egilimi × antisipasyon_şiddeti × bu, antisipasyon_azami_carpan)`. D4'ün davranışsal motoru. |
| **Varsayılan / aralık** | `5.0`, makul aralık `0–10` |
| **Artırınca** (0 → 10) | Referans kur öncesi sipariş zirvesi **1.14x → 2.16x**; eczane kapsaması 7.2 → 10.0 hafta; `eczane_kayip_talep_orani` 0.146 → 0.107 (eczaneler daha çok tampon tutuyor); ama `imha_orani` 0.103 → 0.151 ve `iade_orani` 0.094 → 0.123 |
| **Azaltınca** | 0'da olay katmanının sipariş kanalı kapanır: kur zirvesi 1.14x'e iner, D4 dünyada fiilen yok olur |
| **Yanlış ayarın belirtisi** | *Çok düşük:* olay etüdü grafiğinde `REFERANS_KUR` panelinde sipariş eğrisi tüketim eğrisiyle çakışır — doğrulama scripti bu kontrolden kalar (eşik 1.5x). *Çok yüksek:* eczane kapsaması 12 haftayı aşar ve "eczane stoğu makul kapsamada" kontrolü kalır |
| **Etkileşim** | **`antisipasyon_azami_carpan` ile birlikte okunmalı** — tavan bağladığı andan sonra bu knob'u artırmak neredeyse etkisizdir (5.0 → 10.0 zirveyi sadece 2.11x → 2.16x oynatıyor). `eczane.latent_stokculuk.log_sigma` kimin ne kadar tepki verdiğini belirler |
| **Teşhis** | `uv run python -m scripts.verify_m1 --sadece-tarama --knob-taramasi sim.envanter.antisipasyon_kapsama_kazanci --degerler 0,5,10 --seeds 3` |

> **Bu knob'un maliyeti var ve ölçüldü:** stoklama eczanenin kayıp hasta talebini düşürüyor ama bizim imhamızı ve iademizi yükseltiyor. Kamçı etkisinin (bullwhip) dünyadaki karşılığı budur.

---

### A5. `sim.envanter.azami_kapsama_hafta`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Eczanenin max-stok tavanı: `hedef ≤ ewma × bu × antisipasyon_çarpanı + minimum_sipariş`. Varyans güdümlü emniyet stoğunun patlamasını sınırlar. |
| **Varsayılan / aralık** | `6.0`, makul aralık `4–10` |
| **Artırınca** (4 → 10) | Eczane kapsaması 7.9 → 9.7 hafta, `eczane_kayip_talep_orani` 0.155 → 0.110, kur zirvesi 2.43x → 2.02x (tavan gevşeyince stoklama daha az göze çarpıyor) |
| **Azaltınca** | Eczane daha ince çalışır, kayıp hasta talebi artar; 4'te `siparis_satiri` 50.244'e çıkar (daha sık, daha küçük sipariş) |
| **Yanlış ayarın belirtisi** | *Tavan yokken* (bu knob eklenmeden önce ölçülen hal) hücreler **50+ haftalık** stoğa çıkıyordu: stoksuzluk satışı sansürler → mal gelince satış sıçrar → varyans tahmini patlar → hedef stok patlar. Semptom: `kapsama_hf` 15'i aşar, `iade_orani` %25'i geçer |
| **Etkileşim** | `emniyet_z_katsayisi` ve `talep_varyans_ewma_alfa` bu tavanın frenlediği mekanizmayı üretir. `antisipasyon_azami_carpan` tavanın stoklama sırasında ne kadar gevşediğini belirler |
| **Teşhis** | `uv run python -m scripts.verify_m1 --sadece-tarama --knob-taramasi sim.envanter.azami_kapsama_hafta --degerler 4,6,10 --seeds 3` |

---

### A6. `sim.envanter.antisipasyon_azami_carpan`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Stoklama çarpanının üst sınırı. Hem sipariş hedefini hem max-stok tavanını aynı anda gevşetir; bu sınır olmadan tavan tam da gerektiği anda bağlamaz hale gelir. |
| **Varsayılan / aralık** | `3.0`, makul aralık `1.5–5.0` |
| **Artırınca** | Stoklama serbestleşir: kur zirvesi büyür ama eczane kapsaması ve imha da büyür. `kazanc=5, carpan=4` ölçümü: kapsama 10.4 hafta, iade 0.141, imha 0.146 (carpan=3'te sırasıyla 9.7 / 0.119 / 0.128) |
| **Azaltınca** | 1.0'da stoklama tamamen kapanır (`antisipasyon_kapsama_kazanci=0` ile aynı etki) |
| **Yanlış ayarın belirtisi** | Sınır `kazanc × medyan stokculuk`'un altına inerse neredeyse tüm eczaneler sınıra yapışır ve **stoklama heterojenliği kaybolur** — M4'te uplift farkının bir kaynağı ölür. Medyan `stokculuk_egilimi` ≈ 0.70, yani `kazanc × 0.70 < carpan` tutulmalı |
| **Etkileşim** | A4 ile birlikte tek bir mekanizma; ikisi ayrı süpürülürse yanıltıcı sonuç verir (2D süpürün) |
| **Teşhis** | `uv run python -m scripts.verify_m1 --sadece-tarama --knob-taramasi sim.envanter.antisipasyon_azami_carpan --degerler 1.5,3.0,5.0 --seeds 3 --sabit sim.envanter.antisipasyon_kapsama_kazanci=5.0` |

---

### A7. `sim.iade.degerlendirme_esigi_gun`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Eczacı sadece kalan raf ömrü bu pencereye giren stoğu iade kararı için değerlendirir. İki yıl miatlı stoğa bakıp karar vermez. |
| **Varsayılan / aralık** | `60`, makul aralık `30–120` |
| **Artırınca** (30 → 120) | `iade_orani` 0.113 → 0.145, `imha_orani` 0.123 → 0.165, `karsilama_orani` 0.931 → 0.911, `gozlenen_pay` 0.353 → 0.322 |
| **Azaltınca** | İade daha geç ve daha az olur; 30 günde bile `iade_orani` %11'de kalır — çünkü sürücü eşik değil, **olay sonrası talebi çöken yapısal olarak satılamaz stok** |
| **Yanlış ayarın belirtisi** | Pencereyi daraltmak iade oranını %10'un altına indirmiyorsa sorun eşikte değil, eczane stok seviyesindedir: `azami_kapsama_hafta` ve `antisipasyon_*` knob'larına bakın |
| **Etkileşim** | `eczaci_guvenlik_marji_gun` ile aynı formülün iki ucu; `envanter.azami_kapsama_hafta` iade edilecek fazlanın oluşup oluşmayacağını belirler |
| **Teşhis** | `uv run python -m scripts.verify_m1 --sadece-tarama --knob-taramasi sim.iade.degerlendirme_esigi_gun --degerler 30,60,120 --seeds 3` |

---

### A8. `lot.giris.kisa_miatli_parti_olasiligi`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Bize giren bir partinin kısa miatlı olma olasılığı. Bu partiler `kisa_miatli_kalan_gun_ort` civarında raf ömrüyle gelir ve çoğu eczanenin kabul eşiğinin altında kalır. |
| **Varsayılan / aralık** | `0.03`, makul aralık `0.0–0.12` |
| **Artırınca** (0 → 0.12) | `imha_orani` 0.109 → 0.208, `iade_orani` 0.099 → 0.190, `miad_reddi_orani` 0.037 → 0.094, `karsilama_orani` 0.949 → 0.877, `gozlenen_pay` 0.368 → 0.291 |
| **Azaltınca** | 0'da bile `imha_orani` **0.109**'da kalır — imhanın çoğu bu knob'dan gelmiyor; `ikmal.hedef_kapsama_hafta` × `ikmal.minimum_parti_adet` ikilisinin yavaş SKU'da yarattığı fazla stoktan ve eczane iadesinden geliyor |
| **Yanlış ayarın belirtisi** | *Çok yüksek:* karşılama %88'in altına iner ve düşüşün yarısından fazlası miad reddidir — kısa miatlı stok FEFO kuyruğunu tıkar |
| **Etkileşim** | A1 ile **çarpımsal**. M5'in stres senaryosu bu ikisini birlikte yükseltmek olacak |
| **Teşhis** | `uv run python -m scripts.verify_m1 --sadece-tarama --knob-taramasi lot.giris.kisa_miatli_parti_olasiligi --degerler 0,0.03,0.12 --seeds 3` |

---

### A9. `sim.talep.dagilim.sifir_sisirme`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Yapısal olmayan ek sıfır olasılığı. Çeşitte olan bir hücrede bile o hafta hiç talep gelmeme olasılığı. Intermittent talebin ana kadranı. |
| **Varsayılan / aralık** | `0.34`, makul aralık `0.15–0.55` |
| **Artırınca** (0.15 → 0.55) | `aktif_sifir_orani` 0.574 → 0.775, `siparis_satiri` 54.571 → 33.177, `siparis_p90_hucre` 12 → 8, eczane kapsaması 7.9 → 12.1 hafta (seyrek talep tampon gerektirir), `iade_orani` 0.088 → 0.166 |
| **Azaltınca** | Seri yoğunlaşır, Croston/hazard tipi modeller kolaylaşır |
| **Yanlış ayarın belirtisi** | *Çok düşük:* M2'de basit "son N günde aldı mı" kuralı hazard modeline yaklaşır — D2'nin ölçmek istediği fark kapanır. *Çok yüksek:* `siparis_p90_hucre` 8'in altına iner ve eczane kapsaması 12 haftayı aşarak doğrulama kontrolünü düşürür |
| **Etkileşim** | `negbin_shape` ile birlikte seyrekliğin *tipini* belirler: bu knob sıfır sayısını, negbin_shape sıfır-olmayanların büyüklük kuyruğunu kontrol eder |
| **Teşhis** | `uv run python -m scripts.verify_m1 --sadece-tarama --knob-taramasi sim.talep.dagilim.sifir_sisirme --degerler 0.15,0.34,0.55 --seeds 3` |

---

### A10. `sim.talep.dagilim.negbin_shape`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Gamma-Poisson karışımının shape'i. Küçüldükçe aşırı yayılım artar: aynı ortalamada çok daha ağır kuyruk. Var/Ort = 1 + ort/shape. |
| **Varsayılan / aralık** | `0.65`, makul aralık `0.35–1.5` (∞ = saf Poisson) |
| **Artırınca** (0.35 → 1.5, Poisson'a yaklaşır) | `eczane_kayip_talep_orani` **0.182 → 0.065**, eczane kapsaması 11.4 → 7.8 hafta, `siparis_satiri` 37.815 → 53.373, `iade_orani` 0.154 → 0.086 |
| **Azaltınca** | Talep patlamaları emniyet stoğunu aşar; eczane hasta talebinin %18'ini kaybeder ve fazla tampon tutar |
| **Yanlış ayarın belirtisi** | *Çok yüksek (Poisson):* talep tahmininde MAE şüpheli derecede düşük çıkar; gerçek dağıtım verisinde asla görülmeyen bir düzenlilik. *Çok düşük:* eczane kayıp talebi %18'i aşar |
| **Etkileşim** | `emniyet_z_katsayisi` bunun karşı ağırlığıdır, `azami_kapsama_hafta` ise karşı ağırlığın patlamasını frenler. **Üçünü birlikte düşünün** |
| **Teşhis** | `uv run python -m scripts.verify_m1 --sadece-tarama --knob-taramasi sim.talep.dagilim.negbin_shape --degerler 0.35,0.65,1.5 --seeds 3` |

---

### A11. `sim.talep.cesitlendirme.haftalik_churn_orani`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Stoğu tükenmiş çeşitteki hücrelerin haftalık listeden çıkma olasılığı. Çıkan hücre sayısı kadar yeni hücre (affinite ağırlıklı) eklenir — çeşit **seviyesi** durgun kalır, **kompozisyonu** değişir. |
| **Varsayılan / aralık** | `0.050`, makul aralık `0.0–0.15` |
| **Artırınca** | Geçmiş sipariş kalıbı daha hızlı bayatlar; M2'nin point-in-time doğruluğu daha sert sınanır |
| **Azaltınca** | 0'da çeşit sabitlenir, "eczane bu SKU'yu bıraktı" vakası hiç oluşmaz ve M2'nin tükenme modeli hiç yanlış pozitif üretmez — şüpheli derecede iyi metrik |
| **Yanlış ayarın belirtisi** | Doğrulama scriptindeki **"Çeşit oranı ufuk boyunca durgun"** kontrolü kalırsa ekleme/çıkarma dengesi bozulmuş demektir. Bu gerçek bir hataydı: ayrı olasılıklarla çeşit oranı 104 haftada %15.8'den %32.8'e kaymıştı |
| **Etkileşim** | `cesitlendirme.taban_oran` seviyeyi, bu knob devir hızını belirler. Çıkışın stoksuz hücreyle sınırlanması `iade_orani`'nı doğrudan düşürür (bu koşul olmadan iadelerin %67'si buradan geliyordu) |
| **Teşhis** | `uv run python -m scripts.verify_m1 --sadece-tarama --knob-taramasi sim.talep.cesitlendirme.haftalik_churn_orani --degerler 0.0,0.05,0.15 --seeds 3` |

---

### A12. `lot.tahsis.miad_toleransi_uygulanir`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Anahtar. `false` iken FEFO kördür: en erken miatlı lot, eczacının kabul edip etmeyeceğine bakılmaksızın verilir. |
| **Varsayılan / aralık** | `true`, aralık `{true, false}` |
| **`false` yapınca** | `karsilama_orani` 0.924 → 0.987, `miad_reddi_orani` 0.052 → 0.000, `imha_orani` 0.122 → 0.102, ama `iade_orani` 0.114 → **0.134 yükselir** |
| **`true` tutunca** | SPEC §2.5'in "alıcı tarafı direnci" mekanizması sipariş anında devrede olur |
| **Yanlış ayarın belirtisi** | `false` bırakılırsa M5'in "kısa miatlı stoğu eczaneye yıkmak zararı transfer eder" cümlesi dünyada karşılıksız kalır |
| **Etkileşim** | **Ölçümdeki asıl ders burada:** toleransı kapatınca kısa miatlı mal eczaneye gidiyor, karşılama yükseliyor, depo imhası düşüyor — ama zarar yok olmuyor, **eczaneye transfer oluyor** (iade %14'e çıkıyor). SPEC §2.5'in tam olarak anlattığı şey, sayıyla |
| **Teşhis** | `uv run python -m scripts.verify_m1 --sadece-tarama --knob-taramasi lot.tahsis.miad_toleransi_uygulanir --degerler true,false --seeds 3` |

---

# B. Aile satırları

Her satırda: **birincil kadran** (etkiyi taşıyan knob) ve **aile üyeleri** (şekil/sınır parametreleri). Aralıklar `full` profilinde denenmiş, ölçüm A bölümündeki kadar derin değil — o yüzden burada yön ve belirti var, sweep sayısı yok.

| # | Aile / birincil kadran | Ne yapar (mekanizma) | Varsayılan → aralık | Artırınca / azaltınca | Yanlış ayarın belirtisi | Etkileşim | Teşhis |
|---|---|---|---|---|---|---|---|
| B1 | `profil.*`<br>**kadran:** `eczane_sayisi`, `sku_sayisi`, `hafta_sayisi`<br>üyeler: `ad`, `baslangic_tarihi`, `temel_seed` | Dünyanın boyutu. `full` çıkış kriterini, `fast` sweep hızını hedefler. `temel_seed` tüm rastgeleliğin kökü. | `200/300/104` → `fast: 40/60/52` | Büyütünce koşu süresi ~lineer artar (full = 1.2 sn); küçültünce olay sayısı düşer ve olay etüdü gürültülenir | `fast` profilinde 52 haftada bazı olay tipleri hiç oluşmayabilir; `hafta_sayisi < 60` ise olaya dayalı kontroller güvenilmez | `profil.*` değişince config_hash değişir; farklı profil koşuları karşılaştırılamaz | `uv run python -m scripts.generate_world --profil fast` |
| B2 | `urun.evren.populerlik_log_sigma`<br>üyeler: — | SKU hacim dağılımının log-normal sigması. Uzun kuyruğun tek kadranı. | `1.15` → `0.6–1.8` | Artırınca üst %10 SKU hacim payı büyür (ölçüm: %56.6); azaltınca SKU'lar birbirine benzer | *Çok düşük:* doğrulama scriptindeki "uzun kuyruk" kontrolü kalır (< %45); SKU'lar ayırt edilemez hale gelir, M3'ün aday üretimi anlamını yitirir | `cesitlendirme.populerlik_agirligi` ile birlikte hangi SKU'nun hangi eczanede bulunduğunu belirler | `--knob-taramasi urun.evren.populerlik_log_sigma --degerler 0.6,1.15,1.8` |
| B3 | `urun.evren.*` (lojistik/vergi)<br>**kadran:** `koli_ici_adet_secenekleri`+`_agirliklari`<br>üyeler: `birim_hacim_log_ort/sigma`, `kdv_orani`, `its_olasiligi_rx`, `its_olasiligi_rx_disi` | Koli içi adet **M1'de sipariş miktarını yuvarlamaz** — SPEC §2.1'de koli katı kuralı MF oranları içindir, M4'te devreye girer. Hacim/İTS/KDV M1'de üretilir, M3–M5'te kısıt ve marj hesabını besler. | koli `[1..30]`, kdv `0.10` | M1 metriklerinde etkisiz; M4'te MF granülerliğini, M5'te lojistik maliyeti belirler | M4'te MF oranı koli katına yuvarlanınca aksiyon uzayı beklenmedik biçimde kabalaşırsa buraya bakılır | `koli_ici_adet` M4'ün aksiyon uzayı granülerliğini doğrudan belirler | `uv run python -c "import polars as pl;print(pl.read_parquet('data/full/observable/urunler.parquet')['koli_ici_adet'].value_counts())"` |
| B4 | `urun.promosyon_serbest_kurali.*`<br>**kadran:** `recete_rengi_vetosu`<br>üyeler: `urun_tipi_serbest`, `rx_sgk_disi_serbest`, `rx_sgk_kapsaminda_serbest` | SPEC §2.1'in "genelde serbest" ifadesini **açık kurala** çevirir. `promosyon_serbest` bu tablodan türetilir; D6'nın hard veto listesinin kaynağıdır. | veto `[KIRMIZI, YESIL]` | `recete_rengi_vetosu` boşaltılırsa kırmızı/yeşil reçeteli ürün promosyona açılır — **D6 ihlali** | M3'te kırmızı reçeteli ürün öneri listesinde çıkıyorsa önce buraya bakılır | `kategoriler[].recete_rengi_dagilimi` bu bayrakların ne kadar SKU'yu kapsadığını belirler | `uv run pytest tests/test_config.py::test_promosyon_serbest_kurali_uygulanir` |
| B5 | `eczane.konum.*`<br>**kadran:** `mesafe_olcegi_km`<br>üyeler: `hastane_mesafesi_log_ort/sigma` | Hastane yakınlık skoru = `exp(−km / mesafe_olcegi_km)`. SPEC §2.2'nin "reçete miksini belirleyen en güçlü feature"ının ölçek parametresi. | `1.5 km` → `0.5–4.0` | Artırınca yakınlık skoru tüm eczanelerde 1'e yaklaşır, hastane etkisi düzleşir; azaltınca sadece hastane bitişiği eczaneler ayrışır | *Çok büyük:* kategori affinite tablosundaki `hastane_kats` katsayıları etkisizleşir, C09/A10 ile DERMO/TEG arasındaki kontrast kaybolur | `kategori_egilimi.tablo.*.hastane_kats` ile çarpımsal | `--knob-taramasi eczane.konum.mesafe_olcegi_km --degerler 0.7,1.5,3.0` |
| B6 | `eczane.olcek.*`<br>**kadran:** `buyukluk_log_sigma`<br>üyeler: `aylik_recete_taban`, `ciro_bandi_sinirlari` | Eczane büyüklüğü heterojenliği. Talep ölçeğini ve çeşit genişliğini birlikte sürükler; `aylik_recete_adedi` ve ciro bandı bunun gözlemlenebilir izdüşümleridir. | `0.55` → `0.3–0.9` | Artırınca birkaç büyük eczane hacmin çoğunu alır (eczane tarafında uzun kuyruk); azaltınca eczaneler benzeşir | *Çok düşük:* M3/M4'te segment heterojenliği kaybolur, uplift farkı ölçülemez | `cesitlendirme.buyukluk_agirligi` ile birlikte büyük eczanenin çeşit avantajını belirler | `--knob-taramasi eczane.olcek.buyukluk_log_sigma --degerler 0.3,0.55,0.9` |
| B7 | `eczane.nobet.*`<br>**kadran:** `rotasyon_periyodu_gun_secenekleri`<br>üyeler: `rotasyon_periyodu_agirliklari` | Nöbet rotasyonu 7 günün katı değilse nöbet günü haftadan haftaya kayar ve akut kategorilerde haftalık gerçek dalgalanma yaratır. Hepsi 7 olursa etki sabit seviyeye dönüşür ve öğretici olmaktan çıkar. | `[6,7,8,9,10]` | Seçenekleri 7'ye daraltmak haftalık nöbet varyasyonunu sıfırlar | Akut kategorilerde açıklanamayan haftalık varyans beklenenden düşükse buraya bakılır | `talep.nobet.akut_carpani` şiddeti, bu knob zamanlamayı verir | `uv run python -c "import polars as pl;print(pl.read_parquet('data/full/observable/eczaneler.parquet')['nobet_rotasyon_gun'].value_counts())"` |
| B8 | `eczane.kredi.*`<br>**kadran:** `dbs_limiti_carpani_ort`<br>üyeler: `vade_riski_ort/sigma`, `ortalama_recete_tutari`, `dbs_limiti_carpani_sigma` | DBS limiti = aylık reçete adedi × ortalama reçete tutarı × çarpan. M1'de **sadece üretilir**, hiçbir davranışı etkilemez; M3'ün kredi limiti vetosunu besleyecek. | çarpan `2.4` → `1.0–5.0` | M1 metriklerinde etkisiz | M3'te "kredi limiti aşan teklif üretilmiyor" testi hiç bağlamıyorsa (hiç teklif elenmiyorsa) çarpan çok yüksektir | `vade_riski_skoru` ile birlikte M3'ün risk katmanını oluşturur | `uv run python -c "import polars as pl;print(pl.read_parquet('data/full/observable/eczaneler.parquet')['dbs_limiti'].describe())"` |
| B9 | `eczane.recete_karmasi.*`<br>**kadran:** `sgk_recete_orani_ort`<br>üyeler: `sgk_recete_orani_sigma` | Eczanenin SGK reçetesi / serbest satış karması. M1'de üretilir, M3–M4'te SGK kısıtlı ürünlerde aksiyon uzayını daraltacak. | `0.72` → `0.4–0.9` | M1 metriklerinde etkisiz | — | `urun.kategoriler[].sgk_olasiligi` ile birlikte SGK kısıtının kapsamını belirler | `uv run python -c "import polars as pl;print(pl.read_parquet('data/full/observable/eczaneler.parquet')['sgk_recete_orani'].mean())"` |
| B10 | `eczane.latent_share_of_wallet.*` (A2 dışı)<br>**kadran:** `beta_a`<br>üyeler: `min`, `max` | Beta dağılımının ilk şekil parametresi ve kırpma sınırları. `beta_a`/`beta_b` birlikte SOW'un ortalamasını ve çarpıklığını verir; `min`/`max` uç eczaneleri sınırlar. | `a=1.6, min=0.05, max=0.92` | `beta_a` artınca ortalama SOW yükselir; `min` yükseltmek "bizden hiç almayan eczane" vakasını yok eder | `min` 0.2'nin üstüne çıkarılırsa cold-start (bize hiç sipariş vermemiş eczane) vakası dünyadan silinir — M3'ün cold start problemi kaybolur | A2 (`beta_b`) ile birlikte tek bir dağılım tanımlar | `--knob-taramasi eczane.latent_share_of_wallet.beta_a --degerler 1.0,1.6,3.0` |
| B11 | `eczane.latent_stokculuk.*`<br>**kadran:** `log_sigma`<br>üyeler: `log_ort`, `ust_sinir` | Rejim beklentisine tepki katsayısının dağılımı. **Uplift heterojenliğinin M1'deki taşıyıcısı:** aynı olayda üst çeyrek stokçu eczane 4.2x, ortalama eczane 2.7x sipariş artışı gösteriyor. | `log_ort=-0.35, log_sigma=0.70` | `log_sigma` artınca segmentler arası tepki farkı büyür; 0'a indirince tüm eczaneler aynı tepkiyi verir | *`log_sigma`=0:* M4'te propensity-based ile uplift-based politika **aynı sonucu verir** — PROMPTS.md M4'ün "kod yazmayı bırak, bana söyle" durumu. Heterojenliğin bu ailedeki kaynağı budur | `envanter.antisipasyon_kapsama_kazanci` şiddeti çarpar; `olay.olaylar[].antisipasyon_siddeti` tetikleyiciyi verir | `--knob-taramasi eczane.latent_stokculuk.log_sigma --degerler 0.01,0.70,1.20` |
| B12 | `eczane.latent_miad_toleransi.*` (A1 dışı)<br>**kadran:** `taban_gun_sigma`<br>üyeler: `min_gun`, `max_gun` | Miad toleransının eczaneler arası yayılımı. Sigma, "hangi eczane kısa miatlıyı kabul eder" ayrımını üretir — M5'in hedefli temizliğinin öğrenilebilir olması buna bağlı. | `sigma=55, [45, 420]` | Sigma 0'a inince tüm eczaneler aynı eşiği uygular; hedefleme problemi kalmaz, sadece bir global eşik kalır | *Sigma çok düşük:* M5'te "hedefli temizlik" ile "kör iskonto" arasındaki fark kapanır | A1 (ortalama) ile birlikte dağılımı tanımlar; `kategoriler[].miad_toleransi_carpani` kategori kırılımını ekler | `--knob-taramasi eczane.latent_miad_toleransi.taban_gun_sigma --degerler 10,55,110` |
| B13 | `eczane.latent_siparis_davranisi.gozden_gecirme_*`<br>**kadran:** `gozden_gecirme_periyodu_agirliklari`<br>üyeler: `_secenekleri`, `kapsama_hafta_sigma`, `kapsama_hafta_min` | Eczanenin sipariş ritmi. Periyot dağılımı ağırlıkları uzun periyotlara kayarsa siparişler seyrekleşir ve emniyet stoğu risk penceresi (√(L+periyot)) üzerinden büyür. | `[1,2,3,4]` ağırlık `[.46,.30,.15,.09]` | Uzun periyotlara kaydırınca sipariş satırı azalır, satır başına adet artar | Sipariş serisinde eczane bazlı sabit periyot desenleri fazla belirginse model bunu tüketim sinyali sanır — M2'de leakage değil ama sahte sinyal | `kapsama_hafta_ort` (B13) ve `emniyet_z_katsayisi` (B20) ile üçlü | `uv run python -c "import polars as pl;o=pl.read_parquet('data/full/observable/siparisler.parquet');print(o.group_by('eczane_id').len()['len'].describe())"` |
| B14 | `eczane.kategori_egilimi.gurultu_sigma`<br>üyeler: `tablo` → bkz. C5 | Eczane–kategori affinitesinin **açıklanamayan** kısmı. Sıfır olsaydı çeşit ve talep tamamen gözlemlenebilir özniteliklerden türerdi ve M3'ün aday üretimi trivialleşirdi. | `0.45` → `0.1–1.0` | Artırınca affinite gözlemlenebilir özniteliklerden koparılır (M3 zorlaşır); azaltınca hastane/sosyoekonomik feature'ları neredeyse tam açıklayıcı olur | *Çok düşük:* M3'te CF ile basit attribute-based kural aynı sonucu verir — collaborative filtering'in katkısı ölçülemez | `kategori_egilimi.tablo` ile toplam affinite varyansını paylaşır | `--knob-taramasi eczane.kategori_egilimi.gurultu_sigma --degerler 0.1,0.45,0.9` |
| B15 | `sim.takvim.yil_sonu_*`<br>**kadran:** `yil_sonu_stoklama_yogunlugu`<br>üyeler: `yil_sonu_stoklama_aylari` | SPEC §2.3'ün "Aralık: yıl sonu SGK/katılım payı değişim beklentisiyle stoklama" maddesi. Olay değil, **takvim kaynaklı antisipasyon**: tüketimi değil siparişi etkiler. | `0.55`, aylar `[12]` | Artırınca Aralık'ta sipariş sıçraması büyür; 0'da yıl sonu deseni kaybolur | Aralık sipariş sıçraması varken tüketim düz değilse mekanizma yanlış kanala bağlanmış demektir | `envanter.antisipasyon_kapsama_kazanci` şiddeti çarpar (aynı kanal) | `--knob-taramasi sim.takvim.yil_sonu_stoklama_yogunlugu --degerler 0,0.55,1.0` |
| B16 | `sim.talep.cesitlendirme.*` (A7 dışı)<br>**kadran:** `affinite_agirligi`<br>üyeler: `populerlik_agirligi`, `buyukluk_agirligi`, `yeni_cesit_deneme_adedi` (churn için bkz. A11) | Çeşidin ne kadarının affiniteden, ne kadarının popülerlik ve büyüklükten geldiği; çeşidin zaman içinde kayması (churn). Churn, geçmiş sipariş kalıbını eskitir — M2'nin point-in-time doğruluğunu gerçekten test eden şey budur. | `affinite 1.30` | `affinite_agirligi` artınca çeşit gözlemlenebilir özniteliklerle daha iyi açıklanır. `yeni_cesit_deneme_adedi` yeni hücrenin ilk sipariş büyüklüğünü belirler — koli içi adetten türetmek YANLIŞTIR (sipariş adet bazında verilir; 30'luk kolisi olan SKU'ya haftada 30 adetlik talep atfedilirdi) | `yeni_cesit_deneme_adedi` çok yüksekse yeni hücreler hemen ölü stok biriktirir ve iade oranı şişer | A9/A11 ile aynı çeşit mekanizmasını paylaşır | `--knob-taramasi sim.talep.cesitlendirme.yeni_cesit_deneme_adedi --degerler 1,3,8` |
| B17 | `sim.talep.yogunluk.*`<br>**kadran:** `hucre_gurultu_shape`<br>üyeler: `taban_adet_hafta` | Hücre bazlı gamma gürültüsü: aynı eczane–kategori affinitesine sahip iki hücrenin bile farklı hızda olmasını sağlar. Shape küçüldükçe heterojenlik artar. | `shape=0.90, taban=1.35` | Shape artınca hücreler benzeşir (dünya kolaylaşır); azaltınca birkaç hücre hacmin çoğunu alır | *Shape çok yüksek:* hücre hızı tamamen (popülerlik × affinite × büyüklük) ile belirlenir — M2 latent hızı feature'lardan doğrudan çözer, MAE şüpheli derecede düşer | `urun.evren.populerlik_log_sigma` ve `olcek.buyukluk_log_sigma` ile toplam hücre varyansını paylaşır | `--knob-taramasi sim.talep.yogunluk.hucre_gurultu_shape --degerler 0.4,0.9,3.0` |
| B18 | `sim.talep.dagilim` şok üyeleri<br>**kadran:** `kategori_hafta_soku_sigma`<br>üyeler: `eczane_hafta_soku_sigma` | Kategori×hafta ve eczane×hafta **korele** şoklar. Bağımsız gürültüden farkı: model bunları hücre bazında ortalamayla temizleyemez, ortak faktör olarak modellemek zorunda. | `0.22 / 0.18` → `0.05–0.5` | Artırınca haftalık seriler daha oynak, olay etüdü gürültülenir; azaltınca dünya fazla düzenli | *Çok düşük:* olay etkileri gürültüsüz zeminde net görünür — gerçek veride asla olmayan bir netlik. *Çok yüksek:* olay etüdü kontrolü kalır | Olay kataloğundaki `tuketim_carpani` ile sinyal/gürültü oranını belirler | `--knob-taramasi sim.talep.dagilim.kategori_hafta_soku_sigma --degerler 0.05,0.22,0.45` |
| B19 | `sim.talep.nobet` + `sim.talep.turizm`<br>**kadran:** `turizm.zirve_carpani`<br>üyeler: `nobet.akut_carpani`, `turizm.zirve_aylari`, `omuz_carpani`, `omuz_aylari` | SPEC §2.2/§2.3'ün turizm (yazın 2–3x ciro) ve nöbet (akut talep spike) mekanizmaları. Turizm eczane özniteliğine, nöbet rotasyona bağlı. | turizm `2.40`, nöbet `1.85` | Artırınca turizm bölgesi eczaneleri yazın belirgin ayrışır; azaltınca `turizm_bolgesi` feature'ı ölür | Turizm bölgesi eczanelerinin yaz/kış oranı 1.2'nin altındaysa feature öngörü gücü taşımaz | `iller[].turizm_olasiligi` kaç eczanenin etkilendiğini belirler | `--knob-taramasi sim.talep.turizm.zirve_carpani --degerler 1.2,2.4,3.5` |
| B20 | `sim.envanter` kalan üyeler<br>**kadran:** `talep_ewma_alfa`<br>üyeler: `tedarik_suresi_hafta`, `emniyet_stogu_hafta`, `emniyet_z_katsayisi`, `talep_varyans_ewma_alfa`, `minimum_siparis_adedi`, `soguk_zincir_minimum_siparis_adedi`, `baslangic_kapsama_hafta`, `stoksuzlukta_acil_gozden_gecirme`, `antisipasyon_siparis_esigi` (tavan için bkz. A5/A6) | Eczanenin **kendi** talep tahmininin hafızası. Kritik nokta: bu EWMA **satışla** beslenir, talep ile değil — stoksuzlukta sansürlenir. Bu, sipariş serisinin tüketimin bozuk bir izdüşümü olmasının ana kaynaklarından biridir. | `alfa=0.25` → `0.05–0.6` | Artırınca eczane son haftaya aşırı tepki verir, sipariş oynaklığı büyür; azaltınca mevsimsel dönüşlere geç uyum sağlar | *Alfa çok yüksek:* sipariş serisi tüketimin gecikmeli kopyasına dönüşür ve D2'nin zorluğu azalır. *`soguk_zincir_minimum_siparis_adedi`* M3'ün min sipariş testinin dünya tarafındaki karşılığıdır; 1'e indirilirse test bağlamaz | A5 (max-stok tavanı) bu ailenin ürettiği patlamayı frenler; `emniyet_z_katsayisi` ile `talep_varyans_ewma_alfa` birlikte varyans tahminini besler | `--knob-taramasi sim.envanter.talep_ewma_alfa --degerler 0.1,0.25,0.5` |
| B21 | `sim.ikmal` kalan üyeler + `sim.tedarikci_secimi` kalan üyeler<br>**kadran:** `ikmal.minimum_parti_adet`<br>üyeler: `ikmal.periyot_hafta`, `emniyet_z_katsayisi`, `baslangic_kapsama_hafta`, `siparis_gurultusu_sigma`; `tedarikci_secimi.siparis_gurultusu`, `karsilanamayan_siparis_rakibe_gider`, `stoksuzluk_sow_cezasi`, `sow_toparlanma_hizi`, `sow_rassal_yuruyus_sigma` | `minimum_parti_adet` yavaş SKU'da **zorunlu fazla stok** yaratır: haftada 0.2 satan bir üründen 12 adet almak 60 haftalık kapsama demektir. İmhanın endojen kaynağı budur (A8'e bakınız). `sow_rassal_yuruyus_sigma` SOW'u zamanla kaydırır — geçmişten hesaplanan pay tahmini bayatlar. | `minimum_parti_adet=12` → `1–40` | Artırınca imha oranı yükselir ve uzun kuyruk SKU'larında yoğunlaşır; 1'e indirince imha büyük ölçüde `kisa_miatli_parti_olasiligi`'na iner | *Çok yüksek:* imha oranı %15'i aşar ve imhaların çoğu düşük hacimli SKU'da toplanır. *`karsilanamayan_siparis_rakibe_gider=false`:* eczane stoksuzluğumuzda hiç mal alamaz — gerçekçi değil ve eczane kayıp talebini yapay olarak şişirir | A3 (`hedef_kapsama_hafta`) ile çarpımsal; ikisi birlikte DEPO tarafı imhasının ana belirleyicisi | `--knob-taramasi sim.ikmal.minimum_parti_adet --degerler 1,12,40` |

| B22 | `sim.iade.*` (A7 dışı)<br>**kadran:** `depoya_iade_orani`<br>üyeler: `eczaci_guvenlik_marji_gun`, `kredi_orani`, `sow_cezasi`, `cesitten_cikarmada_iade` | İade ekonomisi. `depoya_iade_orani` iadenin bize fiziken dönen payı (kalanı eczanede zayi); dönen mal satılamaz — imha maliyeti + `kredi_orani × DSF` kredi olarak iki kez maliyet yazar. `sow_cezasi` iadenin ilişkiye maliyetidir: eczaneye satamayacağı mal göndermişiz demektir. | `depoya_iade_orani=0.70`, `kredi_orani=0.80`, `sow_cezasi=0.08`, `marj=14 gün` | `depoya_iade_orani` artınca imha ve kredi maliyeti bize kayar, eczanenin zayiatı azalır; toplam iade adedi değişmez | **Bu blok tamamen doğrulanmamıştır** (SPEC §8: "miad iade mekanizmasının fiili işleyişi, kredi oranı, zaman penceresi"). Mutlak TL sonuçları değil, politikalar arası oransal farkı okuyun | A7 formülün diğer ucu; `envanter.azami_kapsama_hafta` iade edilecek fazlanın oluşup oluşmayacağını belirler | `--knob-taramasi sim.iade.depoya_iade_orani --degerler 0.3,0.7,1.0` |
| B23 | `sim.envanter.eczane_lot_bolme_sayisi` | Eczane stoğu hücre başına kaç miad kovasında tutulur. Depo tarafındaki lot izlemesinin eczane tarafındaki vektörize karşılığı. Kova dolu iken gelen sevkiyat miadı en yakın kovaya adet-ağırlıklı ortalamayla birleştirilir. | `4`, makul aralık `2–8` | Artırınca hücre içi miad dağılımı daha sadık temsil edilir (birleştirme kaybı azalır), maliyeti bellek: `[K, 200, 300]` iki dizi | *Çok düşük (1–2):* birleştirme tüm stoğu tek bir ortalama miada indirger; kısa miatlı ve taze mal ayırt edilemez, iade zamanlaması bozulur | `iade.*` bloğunun çözünürlüğünü belirler | `--knob-taramasi sim.envanter.eczane_lot_bolme_sayisi --degerler 2,4,8` |

---

# C. Blok satırları — domain veri tabloları

Bunlar tuning kadranı değil, **dünyanın tanımıdır.** Değiştirilebilir olmaları gerekiyor (SPEC §8), ama süpürülmeleri anlamlı değil. Her biri için: ne tanımlar, hangi alanı değiştirmek neyi bozar, nasıl bakılır.

| # | Blok | İçerik ve etkisi | Değiştirilmesi riskli alan | Teşhis |
|---|---|---|---|---|
| C1 | `urun.kategoriler[]` (11 kayıt × 17 alan) | Kategori evreni: pay, ürün tipi, akut/kronik bayrağı, 12 aylık mevsimsellik profili, ramazan çarpanı, miad toleransı çarpanı, fiyat dağılımı, reçete rengi dağılımı, SGK ve tedarik güçlüğü olasılıkları. `mevsimsellik` vektörü ve `kronik` bayrağı doğrulama scriptindeki "mevsimsel/kronik CV kontrastı" kontrolünün doğrudan kaynağıdır (ölçüm: 2.87, eşik 1.8). | `kronik: true` olan kategorinin `mevsimsellik` vektörünü düzlükten çıkarmak kontrastı bozar ve kontrol kalır. `pay` toplamı 1.0 olmak zorunda (config doğrulaması bunu zorluyor). `recete_rengi_dagilimi` kırmızı/yeşil payını sıfırlarsa M3'ün veto testi bağlamaz. | `uv run python -m scripts.verify_m1 --kosu full` → "Mevsimsellik" satırı ve `reports/figures/m1/mevsimsellik.png` |
| C2 | `olay.olaylar[]` (5 kayıt × 13 alan) | Rejim olay kataloğu. Her tip için sıklık aralığı (`min/max_ara_hafta`), antisipasyon penceresi, şiddet, süre, kalıcı seviye kayması, etkilenen SKU oranı, ikmal bloklama. **Kanal ayrımı burada tanımlanır:** `REFERANS_KUR` yalnızca antisipasyon (sipariş) kanalını, `EPIDEMI` yalnızca tüketim kanalını kullanır. | `REFERANS_KUR_GUNCELLEME.tuketim_carpani`'nı 1.0'dan farklı yapmak **D4'ü ihlal eder** — kur beklentisi tüketimi değil siparişi hareket ettirir. `min/max_ara_hafta`'yı sabitlemek (min=max) olayları tam periyodik yapar ve "ne kadar gecikti" sinyalini trivialleştirir. | `uv run python -m scripts.verify_m1 --kosu full` → "Olay etkisi" satırı ve `reports/figures/m1/olay_etkisi.png` |
| C3 | `eczane.iller[]` (10 kayıt × 5 alan) | Coğrafya: il payları, turizm olasılığı, sosyoekonomik indeks ortalaması ve sapması. Turizm ve sosyoekonomik indeks, kategori affinite tablosu üzerinden DERMO/TEG payını sürükler. | `pay` toplamı 1.0 olmak zorunda. Tüm illerin `turizm_olasiligi`'nı 0 yapmak `turizm_bolgesi` feature'ını öldürür. | `uv run python -c "import polars as pl;e=pl.read_parquet('data/full/observable/eczaneler.parquet');print(e.group_by('il').agg(pl.col('turizm_bolgesi').mean()))"` |
| C4 | `urun.marj_kademeleri[]` (5 kayıt) | Fiyat bandına göre kademeli depo/eczane kâr marjı. DSF = PSF × (1 − eczane marjı); lot birim maliyeti = DSF × (1 − depo marjı) × pazarlık gürültüsü. **M1'de hiçbir davranışı etkilemez** — M4'ün beklenen marj hesabının ve M5'in salvage value'sunun tabanıdır. | Son kademenin `psf_ust_siniri` **null** olmak zorunda (config doğrulaması zorluyor). Değerler SPEC §8'de doğrulanacaklar listesinde: **placeholder'dır.** | `uv run python -c "import polars as pl;u=pl.read_parquet('data/full/observable/urunler.parquet');print(u.group_by('depo_kar_marji').agg(pl.len(),pl.col('psf').mean()).sort('depo_kar_marji'))"` |
| C5 | `eczane.kategori_egilimi.tablo` (11 kategori × 4 katsayı) | Eczane özniteliklerinin kategori affinitesine etkisi: `log_affinite = taban + hastane_kats × yakınlık + sosyo_kats × (sosyo−0.5)×2 + turizm_kats × turizm + N(0, gurultu_sigma)`. SPEC §2.2'nin "hastane yanındaki eczane ile AVM içindeki eczanenin ATC dağılımı tamamen farklı" iddiasının tek uygulama yeri. | Tablo, `urun.kategoriler[]` ile birebir örtüşmek zorunda (config çapraz doğrulaması zorluyor). Tüm katsayıları 0 yapmak affiniteyi saf gürültüye indirger — M3'ün cold start'ı için gözlemlenebilir sinyal kalmaz. | `uv run python -c "import polars as pl;print(pl.read_parquet('data/full/ground_truth/latent_eczane.parquet').select(['latent_affinite_C09','latent_affinite_DERMO']).describe())"` |

*(`sim.takvim.ramazan_pencereleri[]` — 4 kayıt — takvim verisidir, B15 satırında hesap veriyor.)*

---

## M1'de doğmayan, sonraki milestone'lara ait knob'lar

SPEC §5'te listelenen aşağıdaki knob'lar **kasıtlı olarak yoktur** (CLAUDE.md §1: ileriye dönük dosya/iskelet yasağı):

| Knob | Hangi milestone | Neden şimdi değil |
|---|---|---|
| `policy.depletion_threshold_days` | M2/M4 | Politika katmanı yok |
| Exploration oranı, Thompson posterior genişliği | M4 | Bandit ve propensity logging M4'te doğuyor |
| Marj tabanı, vade maliyeti, frekans tavanı | M4 | Aksiyon uzayı henüz yok |
| `clearance.trigger_days`, `salvage_curve`, `safety_factor`, `pharmacist_margin_days` | M5 | Salvage value ve temizlik rejimi tahsis katmanının parçası |
| `disposal_cost_per_unit` | **kısmen var** | M1'de `lot.maliyet.imha_birim_maliyeti_dsf_orani` olarak var (imha maliyetini ölçebilmek için); M5'te salvage eğrisinin girdisi olacak |
| OPE estimator seçimi, clipping eşiği, değerlendirme ufku | M6 | Eval katmanı yok |
| `share_of_wallet` gözlemlenebilirlik anahtarı | M2 | Feature builder'a ait; M1'de SOW her hâlükârda ground_truth'ta (bkz. A2) |

---

# D. Tam envanter — A/B/C satırlarında adı geçmeyen knob'lar

A/B/C satırları mekanizmayı aile düzeyinde anlatıyor; bu tablo **hiçbir knob adının hesapsız kalmadığını** garanti eder. `tests/test_config.py::test_tuning_md_her_knobu_kapsiyor` bu bölümü mekanik olarak zorlar: config'e knob eklenip burası güncellenmezse test düşer.

| Knob | Ailesi | Rolü |
|---|---|---|
| `urun.evren.koli_ici_adet_agirliklari` | B3 | Koli içi adet seçeneklerinin olasılık ağırlıkları. M1'de siparişi etkilemez; M4'te MF granülerlik dağılımını belirler |
| `urun.evren.birim_hacim_log_sigma` | B3 | Birim hacim log-normal yayılımı. M5'te lojistik maliyet farklılaşmasının kaynağı |
| `urun.marj_kademeleri[].depo_marji` | C4 | Kademedeki depo kâr marjı. Lot birim maliyeti = DSF × (1 − bu) × pazarlık gürültüsü |
| `urun.marj_kademeleri[].eczane_marji` | C4 | Kademedeki eczane kâr marjı. DSF = PSF × (1 − bu) |
| `urun.kategoriler[].atc_prefix` | C1 | ATC kodu öneki; `null` ise kategori ilaç dışıdır (DERMO/TEG/MEDIKAL) ve ATC kodu üretilmez |
| `urun.kategoriler[].urun_tipi_dagilimi` | C1 | Kategori içindeki RX/OTC/TEG/... karışımı. `promosyon_serbest` türetmesinin girdisi (B4) |
| `urun.kategoriler[].soguk_zincir_olasiligi` | C1 | Soğuk zincir bayrağı olasılığı. Raf ömrünü `soguk_zincir_carpani` ile kısaltır, min siparişi yükseltir |
| `urun.kategoriler[].ramazan_carpani` | C1 | Ramazan penceresinde talep çarpanı. Hafta içindeki ramazan gün payıyla ölçeklenir |
| `urun.kategoriler[].etken_madde_orani` | C1 | Kategorideki INN sayısı / SKU sayısı. Eşdeğer grup büyüklüğünü belirler; geri çekmede ikame kayışının paydası |
| `urun.kategoriler[].fiyat_log_ort` | C1 | PSF log-normal medyanı (TL). **Doğrulanmamış** (SPEC §8) |
| `urun.kategoriler[].fiyat_log_sigma` | C1 | PSF log-normal yayılımı. Marj kademelerine dağılımı belirler |
| `urun.kategoriler[].tedarik_guclugu_olasiligi` | C1 | `titck_tedarik_guclugu` bayrağı olasılığı. M1'de üretilir, M3 kısıt katmanını besleyecek |
| `eczane.iller[].sosyoekonomik_ort` | C3 | İlin semt sosyoekonomik indeks ortalaması. DERMO/TEG affinitesini sürükler |
| `eczane.iller[].sosyoekonomik_sigma` | C3 | İl içi sosyoekonomik yayılım. Aynı ildeki eczaneleri ayrıştırır |
| `eczane.cografya.ilce_sayisi_il_basina` | B5 | Sentetik ilçe sayısı. M3'te coğrafi kümeleme granülerliği |
| `eczane.cografya.semt_sayisi_ilce_basina` | B5 | Sentetik semt sayısı. Aynı amaç, bir alt düzey |
| `eczane.konum.hastane_mesafesi_log_sigma` | B5 | Hastane mesafesi log-normal yayılımı. Yakınlık skorunun ayrıştırıcılığını belirler |
| `eczane.kredi.vade_riski_sigma` | B8 | Vade riski skoru yayılımı. M3'ün risk katmanında eczaneleri ayrıştırır |
| `eczane.kredi.dbs_limiti_carpani_min` | B8 | DBS çarpanının alt sınırı; negatif/sıfır limit oluşmasını engeller |
| `eczane.latent_siparis_davranisi.gozden_gecirme_periyodu_secenekleri` | B13 | Sipariş gözden geçirme periyodu seçenekleri (hafta). Emniyet stoğu risk penceresi √(L + periyot) buradan gelir |
| `sim.takvim.ramazan_pencereleri[].bitis` | B15 | Ramazan penceresinin bitiş tarihi (başlangıçla birlikte tek bir aralık tanımlar) |
| `sim.tedarikci_secimi.stoksuzluk_sow_cezasi` | B21 | Karşılayamama oranıyla ölçekli SOW cezası. 0'da SOW ortalamaya dönüşle sabit kalır; büyütmek payı ufuk boyunca sistematik olarak aşağı çeker. Teşhis: `--knob-taramasi sim.tedarikci_secimi.stoksuzluk_sow_cezasi --degerler 0,0.05,0.15` |
| `olay.referans_kur.baslangic_deger` | C2 | Referans avro kurunun başlangıç seviyesi (TL). **Doğrulanmamış** (SPEC §8) |
| `olay.referans_kur.guncelleme_artis_ort` | C2 | Her güncellemedeki oransal artışın ortalaması. Fiyat endeksini ve dolayısıyla marjı sürükler |
| `olay.referans_kur.guncelleme_artis_sigma` | C2 | Artış oranının yayılımı. Beklentinin belirsizliğini üretir |
| `olay.referans_kur.fiyat_gecis_katsayisi` | C2 | Kur artışının fiyata yansıma oranı (pass-through). 0'da fiyat sabit kalır, stoklama güdüsü ekonomik temelini kaybeder |
| `olay.referans_kur.fiyat_gecis_gecikme_hafta` | C2 | Fiyata yansımanın gecikmesi. Antisipasyon penceresi ile birlikte D4'ün zamanlamasını kurar |
| `olay.olaylar[].min_ara_hafta` | C2 | Olay tipinin minimum tekrar aralığı. `max_ara_hafta` ile eşitlenirse olay tam periyodik olur ve "ne kadar gecikti" sinyali trivialleşir (D4 ihlali) |
| `olay.olaylar[].antisipasyon_hafta_min` | C2 | Antisipasyon penceresinin alt sınırı (hafta) |
| `olay.olaylar[].antisipasyon_hafta_max` | C2 | Antisipasyon penceresinin üst sınırı. SPEC §2.4 kur için 2–6 hafta diyor |
| `olay.olaylar[].sure_hafta_min` | C2 | Olayın etki süresinin alt sınırı |
| `olay.olaylar[].sure_hafta_max` | C2 | Olayın etki süresinin üst sınırı. Epidemi için SPEC §2.4 3–8 hafta diyor |
| `olay.olaylar[].kalici_seviye_kaymasi` | C2 | Olaydan sonra kalıcı talep seviyesi kayması (SGK listeye giriş/çıkış). 0 = kalıcı etki yok |
| `olay.olaylar[].kalici_kayma_yukari_olasiligi` | C2 | Kalıcı kaymanın yukarı yönlü olma olasılığı. 0.5 = listeye giriş ve çıkış dengeli |
| `olay.olaylar[].etkilenen_sku_orani` | C2 | SKU kapsamlı olaylarda etkilenen SKU payı. Büyütmek olayı portföy geneline yayar |
| `olay.olaylar[].ikmal_bloklar` | C2 | Olay bizim ikmalimizi durduruyor mu (tedarik krizi, geri çekme). M5'in kıt stok senaryosunun M1'deki kaynağı |
| `olay.ikame.geri_cekmede_ikame_orani` | C2 | Geri çekilen ürünün kaybolan talebinin aynı etken maddedeki ürünlere kayan payı. 0'da geri çekme saf talep kaybı olur; M3'ün eşdeğer grup mantığı için sinyal kalmaz |
| `lot.raf_omru.toplam_gun_log_ort` | B24 | Üretimden itibaren toplam raf ömrü log-normal medyanı (ln gün) |
| `lot.raf_omru.toplam_gun_log_sigma` | B24 | Raf ömrü yayılımı. Kısa ve uzun miatlı ürünlerin ayrışmasını belirler |
| `lot.raf_omru.soguk_zincir_carpani` | B24 | Soğuk zincir ürünlerde raf ömrü çarpanı (< 1). SPEC §2.5 "soğuk zincirde miad daha kritik" |
| `lot.giris.tuketilmis_oran_ort` | B24 | Parti bize girerken raf ömrünün tüketilmiş ortalama payı. Büyütmek tüm stoğu yaşlandırır |
| `lot.giris.tuketilmis_oran_sigma` | B24 | Aynı büyüklüğün yayılımı |
| `lot.giris.tuketilmis_oran_ust_sinir` | B24 | Tüketilmiş oranın üst sınırı; raf ömrünün tamamen tükenmiş partisi girmesin diye |
| `lot.giris.kisa_miatli_kalan_gun_sigma` | A8 | Kısa miatlı partinin kalan gün yayılımı |
| `lot.giris.kisa_miatli_min_gun` | A8 | Kısa miatlı partinin kalan gün alt sınırı |
| `lot.maliyet.parti_pazarligi_sigma` | B24 | Lot birim maliyetindeki parti pazarlığı gürültüsü. M4/M5'te aynı SKU'nun lotları arasında maliyet farkı yaratır |
| `lot.tahsis.fefo_aktif` | A12 | `false` iken tahsis LEFO çalışır (en geç miatlı önce). M5'in FEFO ablasyonu için; M1'de `true` |

*(`lot.raf_omru.*`, `lot.giris.tuketilmis_oran_*` ve `lot.maliyet.parti_pazarligi_sigma` ortak bir aile oluşturur — aşağıda B24 olarak anılır; A8 bu ailenin ölçülmüş kadranıdır.)*

---
---

# M2 knob'ları — tükenme modeli ve feature katmanı

M2 iki yeni config dosyası getirdi: `config/features.yaml` (13 knob) ve `config/depletion.yaml` (18 knob). Toplam **31 yeni skaler knob**. M1'deki A/B ayrımı burada da geçerli; M2'de knob sayısı az olduğu için **hepsi tam satır ya da adlandırılmış aile satırı** olarak hesap veriyor, "blok" yok.

## M2 metrik sözlüğü

Bütün sayılar `experiments/sweep.py` çıktısındandır. Ölçüm hedefi **oracle'ın gerçek stok sıfırlanma haftasıdır**; model bunu asla görmez.

| Metrik | Tanım |
|---|---|
| `hazard.auc` | Hazard modelinin **gerçek tükenmede** (karar ufkunda) AUC'si. Ana metrik. |
| `kural_ikili.auc` | "Son N günde aldı mı" kuralının aynı metrikte karşılığı. Referans noktası: 0.500 = bilgi yok |
| `defter.auc` | `stok_tahmini / hız_tahmini` defterinin AUC'si (öğrenme yok, muhasebe var) |
| `teshis_oracle_etiket.auc` | Gerçek tükenme etiketiyle eğitilmiş aynı hazard. **Etiket körlüğünün** tavanı |
| `teshis_oracle_ozellik.auc` | Gerçek stok / gerçek hız. **Özellik körlüğünün** tavanı, bilgi tavanı |
| `hazard.auc_gozlemlenebilir` | Modelin *eğitildiği* soruda ("bize sipariş gelir mi") AUC'si |
| `hazard.mae_gun` | Kırpılmış tükenme süresi MAE'si (gün), tüm canlı hücreler |
| `sabit.mae_gun` | Sabit tahmincinin MAE'si. **MAE'nin tabanı** — bunu geçmeyen model MAE'de bir şey öğrenmemiştir |
| `hazard.kalibrasyon_hatasi` | Beklenen kalibrasyon hatası (ECE), gerçek tükenmeye karşı |
| `panel.olcum_satiri` | Ölçüme giren (hücre, origin) satırı sayısı |

**Teşhis komutu formatı** (M1'den farklı — artık `experiments/` var):

```bash
uv run python -m experiments.sweep --knob <yol> --values a,b,c --seeds 5
```

Aksi belirtilmedikçe aşağıdaki sayılar `profil=fast` (60×100×104), **5 seed ortalamasıdır**; `hazard.auc`'nin seed'ler arası standart sapması ~0.025'tir, yani **0.025'ten küçük farklar gürültüdür ve öyle işaretlenmiştir.** `full` ile teyit edilenler ayrıca belirtilir.

---

## M2-A1. `feature.stok.tavan_kapsama_hafta`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Defter (ledger) stok tahmininin üst sınırı, hafta cinsi kapsama olarak. Defter `stok += sevkiyat − tahmini tüketim` ile ilerler ve her hafta `hız × bu` değerinde kırpılır. |
| **Varsayılan / aralık** | `30.0`, makul aralık `8–100`. **Varsayılan 8'den 30'a ölçümle taşındı** (aşağıda). |
| **Artırınca** | `defter.auc` 0.506 (2) → 0.520 (8) → **0.549 (30)** → 0.549 (100) → 0.549 (300). 30'dan sonra doyuyor: tavan artık bağlamıyor. `hazard.auc` **değişmiyor** (0.521–0.527 bandında, sd 0.025) |
| **Azaltınca** | Defter ayırt etme gücünü kaybeder; 2'de AUC 0.506 = bilgi yok. Stok tahmininin üst ucundaki sıralama bilgisi beraberliğe dönüşür |
| **Yanlış ayarın belirtisi** | *Çok düşük:* `defter.auc` 0.51'in altına iner ama `defter_stok` dağılımı tavanda yığılır — `python -c "...describe()"` ile histogramın sağ ucundaki çubuk ele verir. *Çok yüksek:* etki yok, sadece ölü knob |
| **Etkileşim** | `feature.hiz.pencereler_hafta`'nın en uzunu (defterin referans hızı) ve `stok.varsayilan_gozlenen_pay` ile aynı formülde. Tavan `hız × pay_telafisi × bu` olduğu için telafi katsayısı tavanı da ölçekler |
| **Teşhis** | `uv run python -m experiments.sweep --knob feature.stok.tavan_kapsama_hafta --values 2,8,30,100 --seeds 5` |

> **Bu knob'un hikâyesi M2'nin en öğretici parçası.** İlk gerekçe şuydu: "eczane order-up-to çalışır, elinde 6 haftalıktan fazla mal olmaz; defter de bu tavanı aşmasın." Ölçüm bunu çürüttü. Sebep: defter **seyreltilmiş** akışla besleniyor. Eczanenin gerçek 6 haftalık tavanı bizim birimimizde ~6 × 0.4 ≈ 2.4 haftaya denk geliyor — ve tam o civar en kötü ayar. `full` profilinde eşleşmiş blok bootstrap (hücre bazında): 8 → 30 farkı **+0.015 AUC [%95: +0.005, +0.024]**, anlamlı. Doğru varsayımı yanlış birimde uygulamak, varsayımı hiç uygulamamaktan kötü.

---

## M2-A2. `tukenme.hedef.karar_ufku_hafta`

| Alan | İçerik |
|---|---|
| **Ne yapar** | "Önümüzdeki kaç hafta içinde tükenir" sorusunun ufku. AUC / PR-AUC / kalibrasyon bu ufukta ölçülür; M4'te teklif kararı bu olasılığa bağlanacak. |
| **Varsayılan / aralık** | `4`, makul aralık `2–8` (üst sınır `ufuk_hafta`) |
| **Artırınca** | Taban oran büyür: 0.041 (2 hafta) → 0.081 (4) → 0.159 (8). `hazard.pr_auc` 0.047 → 0.091 → 0.181 — **PR-AUC'nin artması modelin iyileşmesi değil, tabanın yükselmesidir**, karşılaştırırken taban oranla birlikte okunmalı. `hazard.auc` 0.528 → 0.525 → 0.535 (fark gürültü içinde). `hazard.kalibrasyon_hatasi` 0.050 → 0.092 → 0.155 kötüleşir |
| **Azaltınca** | Olay seyrekleşir, ölçüm gürültülenir; 2 haftada 5.450 satırlık test kümesinde ~225 olay kalır |
| **Yanlış ayarın belirtisi** | *Çok uzun:* kalibrasyon hatası taban oranla birlikte büyür — model "yakında tükenir" diyemediği için değil, gözlemlenebilir etiketin oranı gerçek tükenme oranından hızlı büyüdüğü için. *Çok kısa:* PR-AUC 0.05'in altına iner (2 haftada 0.047), sıralama metrikleri oynak olur |
| **Etkileşim** | `ufuk_hafta` üst sınırdır; `sinir_tamponu_hafta` ufka bağlıdır. M4'te teklif frekans tavanıyla birlikte hareket edecek |
| **Teşhis** | `uv run python -m experiments.sweep --knob tukenme.hedef.karar_ufku_hafta --values 2,4,8 --seeds 5` |

---

## M2-A3. `tukenme.hedef.ufuk_hafta`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Ayrık zamanlı hazard'ın ufku: model `h(k)`'yı `k = 1..bu` için öğrenir, tahmin edilen tükenme süresi bu ufukta kırpılır (restricted mean). Aynı zamanda kişi-periyot tablosunun satır çarpanıdır. |
| **Varsayılan / aralık** | `12`, makul aralık `6–20` |
| **Artırınca** | `hazard.mae_gun` 6.2 (6 hafta) → 20.4 (12) → 43.8 (20) — ama `sabit.mae_gun` de 2.3 → 9.2 → 27.5. **Oran** (hazard/sabit) 2.7 → 2.2 → 1.6: uzun ufukta sabit tahminciyi geçmek zorlaşır, çünkü kırpma kütlesi büyür. `hazard.auc` 0.549 → 0.533 → 0.528 |
| **Azaltınca** | Eğitim satırı azalır (kişi-periyot ufukla çarpımsal), koşu hızlanır; 6 haftada ayırt etme gücü en yüksek çünkü kısa vadeli tükenme daha öngörülebilir |
| **Yanlış ayarın belirtisi** | *Çok uzun:* MAE tablosunda `sabit` her şeyi geçer ve MAE metriği bilgi taşımaz olur; kırpma kütlesi %85'i aşar. *Çok kısa:* tükenme olaylarının çoğu ufkun dışında kalır, hazard eğrisinin kuyruğu ölçülemez |
| **Etkileşim** | `sinir_tamponu_hafta` bundan küçük olamaz (config doğrulaması hata verir — ölçüldü, `--values 20` denemesi tam bu yüzden düştü). `model.azami_egitim_satiri` ile çarpımsal |
| **Teşhis** | `uv run python -m experiments.sweep --knob tukenme.hedef.ufuk_hafta --values 6,12,20 --seeds 5 --sabit tukenme.hedef.sinir_tamponu_hafta=20` |

---

## M2-A4. `tukenme.taban_kural.son_n_gun`

| Alan | İçerik |
|---|---|
| **Ne yapar** | D2'nin karşısına konan naif kuralın eşiği: "son N günde aldıysa stoğu vardır". `kural_ikili` tahmincisi bu eşiği literal olarak uygular. |
| **Varsayılan / aralık** | `30`, makul aralık `7–90` |
| **Artırınca** | `kural_ikili.auc`: N=7 → 0.497, N=30 → 0.494, N=60 → 0.501, N=90 → 0.497 (5 seed). **Hiçbir N'de 0.50'den anlamlı biçimde ayrılmıyor.** `kural_ikili.mae_gun` 74.5 → 71.5 → 64.3 → 54.7 "düzelir" ama bu bir öğrenme değil: N büyüdükçe kuralın süre tahmini kırpma kütlesine yaklaşır |
| **Azaltınca** | Aynı şey ters yönde. Kural her N'de aynı: bilgi yok |
| **Yanlış ayarın belirtisi** | Bu knob'un "doğru" ayarı yok — **ölçümün konusu knob değil, kuralın kendisi.** Eğer bir gün `kural_ikili.auc` 0.55'in üstüne çıkarsa simülatörde sipariş ritmi tükenmeyle mekanik olarak eşleşmeye başlamış demektir (dünya kolaylaşmıştır) |
| **Etkileşim** | Yok; diğer tahmincileri etkilemez (sweep'te `hazard.auc` dört değerde de 0.5251 — birebir aynı, kuralın izole olduğunun kanıtı) |
| **Teşhis** | `uv run python -m experiments.sweep --knob tukenme.taban_kural.son_n_gun --values 7,30,60,90 --seeds 5` |

---

## M2-A5. `eczane.latent_share_of_wallet.beta_b` (M1 knob'u, M2 etkisi)

M1'de bu knob "dünyanın ne kadarını görüyoruz" kadranıydı (bkz. A2). M2'de **ölçülebilir bir bedeli** var:

| `beta_b` | görünürlük | `hazard.auc` | `hazard.mae_gun` | `hazard.kalibrasyon_hatasi` | `teshis_oracle_ozellik.auc` | `teshis_oracle_etiket.auc` |
|---|---|---|---|---|---|---|
| 1.0 | en geniş | 0.548 | 25.3 | 0.146 | 0.737 | 0.594 |
| **2.4** | varsayılan | **0.525** | **20.4** | **0.092** | **0.739** | **0.583** |
| 5.0 | en dar | 0.531 | 15.7 | 0.057 | 0.743 | 0.575 |

Üç okuma:

1. **`teshis_oracle_ozellik.auc` sabit (0.737 / 0.739 / 0.743).** Gerçek stok ve gerçek hızla kurulan tavan, bizim görünürlüğümüzden bağımsızdır — olması gerektiği gibi. Bu satır bir **sağlamlık kontrolüdür**: oynasaydı ölçüm düzeneğinde bir sızıntı olurdu.
2. **`teshis_oracle_etiket.auc` görünürlükle düşüyor** (0.594 → 0.575): doğru etiketiniz olsa bile, daha az veri gören özellikler daha az iş görür.
3. **Kalibrasyon hatası dünya körleştikçe *iyileşiyor* (0.146 → 0.057).** Sezgiye aykırı ve mekanizması net: hata, "bize sipariş" oranı ile gerçek tükenme oranı arasındaki farktan geliyor. Pay küçüldükçe bize gelen sipariş seyrekleşiyor ve gözlemlenebilir etiketin oranı gerçek tükenme oranına yaklaşıyor. **Model iyileşmiyor, sadece iki yanlışlık birbirini götürüyor.** M6'nın "offline tahmin neden yalan söyler" sorusunun küçük bir provası.

**Teşhis:** `uv run python -m experiments.sweep --knob eczane.latent_share_of_wallet.beta_b --values 1.0,2.4,5.0 --seeds 5`

---

## M2-A6. `tukenme.model.azami_yaprak`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Hazard modelindeki her ağacın yaprak sayısı üst sınırı. Model kapasitesinin ana kadranı. |
| **Varsayılan / aralık** | `31`, makul aralık `15–63` |
| **Artırınca** | `hazard.auc` 0.524 (15) → 0.525 (31) → 0.528 (63); `hazard.mae_gun` 21.2 → 20.4 → 19.0; `kalibrasyon_hatasi` 0.100 → 0.092 → 0.080. Yön tutarlı ama AUC farkı seed sapmasının (0.03) altında — **"iyileşiyor" demek için 5 seed yetmiyor.** Dikkat çeken tek şey `auc_gozlemlenebilir`in ters yönde gitmesi (0.629 → 0.627 → 0.624): kapasite arttıkça model eğitildiği soruda değil, sorulan soruda iyileşiyor |
| **Azaltınca** | Model doğrusallaşır; 15 yaprakta kalibrasyon belirgin biçimde kötüleşir |
| **Yanlış ayarın belirtisi** | *Çok yüksek:* eğitim etiketinde (`auc_gozlemlenebilir`) AUC yükselirken gerçek tükenmede (`auc`) düşer (63'te henüz olmadı) — yanlış soruya daha iyi uyum. Bu iki sütunun ayrışması aşırı uyumun M2'deki imzasıdır |
| **Etkileşim** | `min_yaprak_ornegi` ve `ogrenme_orani` ile klasik kapasite üçlüsü; `azami_agac` sabitken derinleşmek erken durdurma olmadığı için doğrudan aşırı uyuma gider |
| **Karar** | Varsayılan **31'de bırakıldı.** 63'e çıkarmanın seed'ler arası anlamlılığı: AUC \|z\|=0.15, PR-AUC \|z\|=0.13, kalibrasyon \|z\|=1.12, MAE \|z\|=1.99. Yalnızca MAE sınırda kıpırdıyor; başlık metriği kıpırdamıyor. Gürültüye göre varsayılan değiştirmemek, ikinci bir test penceresi seçimi yapmamak için |
| **Teşhis** | `uv run python -m experiments.sweep --knob tukenme.model.azami_yaprak --values 15,31,63 --seeds 5` |

---

## M2-A7. `feature.stok.varsayilan_gozlenen_pay`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Defterin akışını "gerçek" akışa çevirme katsayısı: sevkiyat ve tahmini tüketim bu değere bölünür. 1.0 = telafi yok. |
| **Varsayılan / aralık** | `1.0`, makul aralık `0.2–1.0` |
| **Artırınca / azaltınca** | `defter_stok` seviyesi 1/pay ile ölçeklenir (0.4'te 2.5 kat), ama **`defter_tukenme_hafta` değişmez** — pay ve payda aynı katsayıyı taşır, sadeleşir. Ölçüm (`full`, 291.440 panel satırı): `defter_stok` medyan oranı tam **2.50** (= 1/0.4), `defter_tukenme_hafta` satırların **%98.07'sinde birebir aynı.** Değişen %1.93, `min_hiz` tabanının altındaki hücreler: telafi onları ölçülebilir hale getirdiği için süreleri tavandan gerçek bir değere düşüyor (değişenlerde ortalama fark 10.2 hafta). `defter.auc` 0.5489 → 0.5485, `hazard.auc` 0.525 → 0.523 (ikisi de gürültü) |
| **Yanlış ayarın belirtisi** | **`defter_tukenme_hafta` bu knob'la anlamlı biçimde oynuyorsa oran sadeleşmesi kırılmıştır** — ya stok telafi ediliyor da hız edilmiyordur, ya `min_hiz` tabanı çok yüksektir. `tests/test_features.py::test_defter_orani_gozlenen_pay_telafisinden_bagimsiz` bunu koruyor |
| **Etkileşim** | `hiz.min_hiz` (kırpma sadeleşmeyi bozan tek yer), `stok.tavan_kapsama_hafta` (tavan da aynı katsayıyla ölçeklenir), `son_siparis_tukenme_hafta` özelliği (o **sadeleşmez**: sipariş miktarı seyreltilmemiştir, hız seyreltilmiştir — ölçüm: ortalama süre 7.07 → 2.39 hafta) |
| **Teşhis** | `uv run python -m experiments.sweep --knob feature.stok.varsayilan_gozlenen_pay --values 0.4,1.0 --seeds 5` |

> Bu knob'un işi metriği iyileştirmek değil, **bir özdeşliği görünür kılmak:** mutlak stok ve mutlak hız gözlemlenemezken, ikisinin oranı olan tükenme süresi gözlemlenebilir. M2'nin bütün çıkarım yapısı buna dayanıyor (`reports/m2.md` §3.2).

---

## M2-A8. `feature.hiz.havuzlama_gucu`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Hücre hızının eczane × kategori ortalamasına çekilme gücü: `ağırlık = n / (n + bu)`, n = hücrenin sipariş sayısı. Uzun kuyrukta hücre başına 1–2 sipariş var; havuzlama tek siparişe bağlı hız tahminini yumuşatır. |
| **Varsayılan / aralık** | `3.0`, makul aralık `0–12` |
| **Artırınca** | `hazard.auc` 0.523 (0) → 0.525 (3) → 0.527 (12). Yön beklenen yönde ama **üç değer de seed sapmasının (0.03) içinde**: havuzlamanın bu dünyada ölçülebilir bir katkısı yok |
| **Azaltınca** | 0'da havuzlama tamamen kapalı; `hiz_havuzlu` özelliği `hiz_akis_52h` ile aynı sütuna dönüşür |
| **Yanlış ayarın belirtisi** | Ölçülemez bir knob olması kendi başına bir bulgu: model zaten `hiz_akis_*` ailesinin dört penceresini birden görüyor, havuzlanmış sürüm ek bilgi taşımıyor. `defter.auc`'nin bu knob'dan hiç etkilenmemesi (üç değerde de 0.5489) doğru bağlandığının kanıtı — defter havuzlanmamış hızı kullanır |
| **Etkileşim** | `hiz.pencereler_hafta` (havuzlama en uzun pencere üzerinde kurulur), `panel.min_siparis_sayisi` (aday evreni daraldıkça n büyür, havuzlama zayıflar) |
| **Teşhis** | `uv run python -m experiments.sweep --knob feature.hiz.havuzlama_gucu --values 0,3,12 --seeds 5` |

---

## M2-A9. `feature.hiz.varsayilan_dongu_hafta`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Sipariş miktarından hız çıkarımının varsayılan döngü uzunluğu: `hiz_miktar = ortalama_siparis_adedi / bu`. Eczane order-up-to çalıştığı için bir sipariş yaklaşık bu kadar haftalık ihtiyacı kapatır. |
| **Varsayılan / aralık** | `4.0`, makul aralık `2–8` |
| **Artırınca / azaltınca** | `hazard.auc` 0.524 / 0.525 / 0.524 (2 / 4 / 8), `defter.auc` üçünde de 0.5489 — **etki yok, ve olmaması gerekiyordu.** Bu katsayı global bir ölçek: `hiz_miktar`ın hücreler arası sıralamasını değiştirmez, ağaç modeli ölçekten bağımsızdır. Tek gerçek etkisi `gozlenen_pay_tabani` kırpmasının nereye düştüğüdür |
| **Yanlış ayarın belirtisi** | Doğru değeri `full` üzerinde ölçüldü: `hiz_miktar / gerçek tüketim` medyan oranı **2.11**, yani gerçek döngü ~8.4 hafta. 4.0 iki kat yanlış — **ve hiçbir metriği bozmuyor.** Bunu bilmeden "önce bu katsayıyı düzelteyim" demek, hiçbir şey değiştirmeyen bir işe gün harcamaktır |
| **Etkileşim** | `gozlenen_pay_tabani` (pay tahmini `hiz_akis / hiz_miktar` olduğu için bu katsayı payın *seviyesini* kaydırır, sıralamasını değil) |
| **Teşhis** | `uv run python -m experiments.sweep --knob feature.hiz.varsayilan_dongu_hafta --values 2,4,8 --seeds 5` |

---

## M2-A10. `feature.panel.origin_araligi_hafta`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Panelin origin (tahmin anı) adımı. 1 = her hafta bir origin, 4 = dört haftada bir. Eğitim satırı sayısını doğrudan belirler. |
| **Varsayılan / aralık** | `2`, makul aralık `1–4` |
| **Artırınca** | Satır sayısı düşer, koşu hızlanır; `hazard.auc` 0.524 (1) → 0.525 (2) → 0.517 (4) — fark seed sapmasının içinde. **İki kat veri, ölçülebilir sıfır kazanç** |
| **Azaltınca** | Satır sayısı iki katına çıkar ama bağımsız bilgi artmaz: ufuk 12 haftayken 1 hafta aralıklı origin'ler aynı hücrenin %92 örtüşen pencerelerini üretir. Örtüşme, standart hataları da yanıltıcı biçimde daraltır |
| **Yanlış ayarın belirtisi** | *Çok sık:* eğitim süresi ve bellek büyür, metrik sabit kalır; eşleşmiş bootstrap aralıkları gerçekte olduğundan dar görünür (satırlar bağımsız değil) |
| **Etkileşim** | `hedef.ufuk_hafta` (örtüşme oranı = 1 − aralık/ufuk), `model.azami_egitim_satiri` (tavan bağlarsa aralığı düşürmek hiçbir şey eklemez) |
| **Teşhis** | `uv run python -m experiments.sweep --knob feature.panel.origin_araligi_hafta --values 1,2,4 --seeds 5` |

---

## M2-B. Aile satırları

Aşağıdakiler tek tek süpürülmedi; mekanizmalarını paylaştıkları aile altında hesap veriyorlar. Her ailenin **birincil kadranı** yukarıda A bölümünde ölçülmüştür.

| # | Aile / üyeler | Ne yapar (mekanizma) | Varsayılan → aralık | Yanlış ayarın belirtisi | Teşhis |
|---|---|---|---|---|---|
| M2-B1 | `feature.panel.ilk_origin_hafta`, `aday_pencere_hafta`, `min_siparis_sayisi` | **Aday evreninin tanımı.** Bir hücre panele girmek için son `aday_pencere_hafta` içinde bize en az `min_siparis_sayisi` sipariş vermiş olmalı; ilk `ilk_origin_hafta` ısınma için harcanır. Hiç görmediğimiz hücre için hız tahmini yoktur — bu bir modelleme tercihi değil, veri gerçeği | `26` / `52` / `1` → ısınma `13–52` | *Pencere çok dar veya min çok yüksek:* panel küçülür ve **hayatta kalan hücreler seçilmiş olur** (sık sipariş verenler); metrik iyileşmiş görünür ama kapsama düşer. `panel.olcum_satiri`'nı her zaman metrikle birlikte oku. *Isınma çok kısa:* ilk origin'lerde `hiz_akis_52h` yarım pencereden hesaplanır | `--knob feature.panel.aday_pencere_hafta --values 26,52,104` |
| M2-B2 | `feature.panel.egitim_orani`, `tukenme.hedef.sinir_tamponu_hafta` | **Zaman bölmesinin yeri ve tamponu.** Origin'lerin ilk `egitim_orani` payı eğitim, kalanı test; arada `sinir_tamponu_hafta` boşluk var çünkü eğitim etiketi ufuk kadar ileri bakıyor | `0.65` / `12` → oran `0.5–0.8` | *Tampon < ufuk:* config doğrulaması koşuyu düşürür (denendi: `ufuk=20, tampon=12` → `ValidationError`). *Oran çok yüksek:* test penceresi tek mevsime sıkışır, mevsimsel etkiler metrikle karışır | `uv run python -m scripts.verify_m2 --kosu full` → "Zaman bölmesi" satırı |
| M2-B3 | `feature.hiz.pencereler_hafta`, `ewma_alfa` | **Hız tahmininin zaman ölçeği.** Dört pencere birden özellik olarak verilir, ağırlığı model kurar; EWMA pencere ortalamasının sürekli muadili. Defterin referans hızı **en uzun penceredir** | `[4,13,26,52]` / `0.20` | *En uzun pencere kısaltılırsa:* defter gürültülenir ve `defter.auc` düşer — pencere listesini değiştirmek defteri de değiştirir, bu bağ kolay gözden kaçar. *Alfa çok yüksek:* `hiz_ewma` son haftanın kopyası olur | `--knob feature.hiz.ewma_alfa --values 0.05,0.2,0.5` |
| M2-B4 | `feature.hiz.min_hiz`, `gozlenen_pay_tabani` | **Sıfıra bölme korumaları.** `min_hiz` altındaki hız "ölçemedik" demektir ve tükenme süresi tavana dayanır; `gozlenen_pay_tabani` pay tahmininin alt sınırı | `0.02` / `0.10` | *`min_hiz` çok yüksek:* yavaş hücrelerin tamamı ufuk değerinde yığılır ve A7'deki oran sadeleşmesi bozulur (test kırmızıya döner). *`gozlenen_pay_tabani` çok yüksek:* `hiz_duzeltilmis` özelliği `hiz_akis`ın sabit katı olur, bilgi taşımaz | `uv run pytest tests/test_features.py::test_defter_orani_gozlenen_pay_telafisinden_bagimsiz` |
| M2-B5 | `feature.stok.baslangic_kapsama_hafta` | Geçmişin başında eczanede varsayılan stok. Defter buradan başlar; `0.0` "hiçbir şey bilmiyoruz" demektir | `0.0` → `0–3` | Isınma penceresi (B1) yeterince uzunsa etkisi kaybolur. Etkisi kaybolmuyorsa ısınma çok kısadır | `--knob feature.stok.baslangic_kapsama_hafta --values 0,1,3` |
| M2-B6 | `tukenme.model.ogrenme_orani`, `azami_agac`, `min_yaprak_ornegi`, `l2_duzenlilestirme`, `ozellik_orani` | **Klasik GBDT kapasite ailesi.** Birincil kadran A6'da (`azami_yaprak`). `ogrenme_orani × azami_agac` toplam uyum bütçesidir | `0.06` / `400` / `80` / `1.0` / `0.80` | *Bütçe çok büyük:* `auc_gozlemlenebilir` yükselirken `auc` düşer (yanlış soruya aşırı uyum). *`min_yaprak_ornegi` çok küçük:* uzun kuyruk hücreleri kendi yapraklarını alır, kalibrasyon bozulur | `--knob tukenme.model.ogrenme_orani --values 0.03,0.06,0.12` |
| M2-B7 | `tukenme.model.erken_durdurma`, `dogrulama_orani`, `sabir` | **Erken durdurma ve bilinçli kapalılığı.** `false` çünkü sklearn'un iç doğrulama bölmesi rassaldır: aynı hücrenin kişi-periyot satırları eğitim ve doğrulama arasında bölünür, doğrulama iyimser çıkar ve durdurma geç olur. Kapalıyken ağaç sayısı sabittir ve koşu tam determinist olur | `false` / `0.10` / `25` | *Açıkken:* iki koşu farklı ağaç sayısında durabilir; `tests/test_depletion.py::test_kosu_tekrar_uretilebilir` bunu yakalar. Açmak isteniyorsa bölme hücre bazında gruplanmalı (M2 borcu) | `uv run pytest tests/test_depletion.py::test_kosu_tekrar_uretilebilir` |
| M2-B8 | `tukenme.model.seed`, `azami_egitim_satiri` | **Tekrar üretilebilirlik ve satır tavanı.** Tavan aşılırsa **(hücre, origin) çiftleri** seed'li olarak alt örneklenir; periyotlar bölünmez, yoksa sansür yapısı bozulur. `full`'de kişi-periyot tablosu 1.43M satır, tavan 1.5M — şu an bağlamıyor | `11` / `1500000` | *Tavan bağlıyorsa:* `panel.hazard_egitim_periyot_satiri` tavana yapışır; ufku veya origin aralığını değiştirmek etkisiz kalır | `uv run python -c "import json;print(json.load(open('experiments/runs/m2_full/metrikler.json'))['panel']['hazard_egitim_periyot_satiri'])"` |
| M2-B9 | `tukenme.taban_kural.n_gun_adaylari` | Ayarlı kuralın (`kural`) N'ini eğitim döneminde bu adaylar arasından AUC'ye göre seçer. Kurala en iyi şansı vermek için var: sabit 30 ile karşılaştırmak kuralı haksız yere zayıf gösterirdi | `[7,14,21,30,45,60,90]` | Seçilen N hep listenin ucundaysa (şu an hep 7 = alt uç) liste yanlış aralıktadır. Şu anki durum bunu söylüyor: **kural gözlemlenebilir etikette en kısa pencereyi seviyor, gerçek tükenmede yine 0.49** | `metrikler.json` → `panel.kural_secilen_n_gun` |
| M2-B10 | `tukenme.degerlendirme.kalibrasyon_kova_sayisi`, `bootstrap_orneklem`, `bootstrap_seed` | **Ölçüm aletinin ayarları.** Kova sayısı kalibrasyon eğrisinin çözünürlüğü; bootstrap tekrarı fark aralıklarının gürültüsü. Bootstrap **hücre bazında blok** çalışır (örtüşen origin'ler bağımsız değil) — bu bir knob değil, `bootstrap_farki(grup=...)`'in çağrı biçimi. Bunlar politikayı değil, ölçümü ayarlar | `10` / `300` / `7` | *Kova çok fazla:* kova başına olay sayısı düşer, ECE yukarı yanlı çıkar. *Bootstrap çok az:* aralıklar koşudan koşuya oynar (seed sabit olduğu için tekrar üretilebilir ama kararsız) | `uv run python -m scripts.verify_m2 --kosu full` → ikili karşılaştırma tablosu |
| M2-B11 | `tukenme.degerlendirme.oracle_teshisi` | **Teşhis anahtarı.** Açıkken iki tavan ölçülür: gerçek etiketle eğitilmiş hazard ve gerçek stok/hızla kurulmuş kapsama. Teslim edilen model değil, **körlüğün bedelini ayrıştıran ölçüm.** Kapatmak koşuyu ~%50 hızlandırır | `true` | *Kapalıyken:* "sızıntı kokusu" kontrolü bağlamaz (tavan bilinmez) ve `reports/m2.md` §3.3'teki ayrıştırma üretilemez. Sweep'lerde bilerek kapatılıyor | `--sabit tukenme.degerlendirme.oracle_teshisi=false` |

---

## M2'de doğmayan, sonraki milestone'lara ait knob'lar

`policy.depletion_threshold_days` — SPEC §5'in "kaç gün kala teklif ver" knob'u — **M2'de hâlâ yok.** M2 bir olasılık ve bir süre üretir; eşiği karar katmanı koyar ve karar katmanı M4'te doğar. Eşiği burada tanımlamak, tükenme modelini politika varsayımına bağlamak olurdu.

`reports/m1.md` §3.2'deki uyarı hâlâ geçerli: o eşik **sabit bir gün sayısı olarak parametrize edilmemeli.** Yavaş ve hızlı hücrelerde "30 gün kala" farklı anlamlara gelir. M2'nin çıktısı bu yüzden gün değil, **olasılık** olarak da veriliyor: `P(karar_ufku içinde tükenir)` hücre hızından bağımsız bir ölçektir.

---
---

# M3 knob'ları — aday üretimi ve kısıt katmanı

M3 tek yeni config dosyası getirdi: `config/policy.yaml`, iki blok — `aday` (geri getirme, 30 knob) ve `kisit` (hard veto, 13 knob). Toplam **43 yeni knob**; 9'u tam satır (ölçümlü), kalanı 11 aile satırında hesap veriyor.

Blokların ayrı olması tasarımın kendisidir: **aday üretimi kısıtlardan habersiz çalışır** (D6). Kısıt katmanının ML skoru üzerinde veto yetkisi olduğunu ölçebilmenin tek yolu, skorun vetodan habersiz üretilmesidir. `tests/test_constraints.py::test_havuz_kisittan_habersiz` bu ayrımı koruyor: `kisit.*` knob'larından hiçbiri aday havuzunu değiştiremez.

## M3 metrik sözlüğü

Bütün sayılar `experiments/sweep.py` çıktısındandır. Aksi belirtilmedikçe **`profil=full`, 5 seed, 3 origin ortalaması**. `aday.hibrit.recall`'ün seed'ler arası standart sapması ~**0.016**; bundan küçük farklar gürültüdür ve öyle işaretlenmiştir.

| Metrik | Tanım |
|---|---|
| `aday.<uretici>.recall` | Origin sonrası `ufuk_hafta` içinde **bize gelen** siparişlerin ne kadarı havuzun ilk K'sında. Ana metrik |
| `aday.<uretici>.yeni_recall` | Aynı metrik, **yalnızca yeni hücrelerde** (origin'e kadar hiç sipariş görmemiş eczane×SKU). Cross-sell'in asıl ölçüsü |
| `aday.hibrit.oracle_recall` | Aynı metrik, hedef **gerçek tüketim** (ground_truth). Tedarikçiden bağımsız gerçek ihtiyaç; körlüğün tavanı |
| `aday.<uretici>.kapsama` | Havuzun değdiği farklı SKU oranı. Kişiselleştirme ≠ katalog çeşitliliği |
| `aday.hibrit.soguk_eczane_recall` | En az geçmişli `soguk_dilim` kadar eczanede recall |
| `aday.yeni_hedef_orani` | Hedef kümesinin yeni hücre payı. Havuz kompozisyonu bununla kıyaslanır |
| `kisit.havuz_recall` | Veto **öncesi** havuz recall'u |
| `kisit.veto_sonrasi_recall` | Veto **sonrası**, frekans tavanı **öncesi** |
| `kisit.liste_recall` | Fiilen sahaya çıkan öneri listesinin recall'u |
| `kisit.veto_orani` | Aday satırlarının vetolanma oranı |
| `kisit.ust_dilim_veto_orani` | **En yüksek skorlu %10'un** vetolanma oranı. D6'nın ölçüsü: kısıt katmanı ML'in en emin olduğu yeri kesiyor mu |
| `kisit.veto_<sebep>` | Sebep başına veto oranı (satırlar birden fazla sebeple bağlanabilir) |
| `ihlal.*` | **Sıfır olmak zorunda.** Her koşuda üretilir; sweep tablosunda sıfırdan farklıysa `experiments/sweep.py` bağırır |

**Teşhis komutu formatı:**

```bash
uv run python -m experiments.sweep --knob <yol> --values a,b,c --seeds 5 --asama m3 --profil full
```

`--asama m3` M2'nin hazard eğitimini atlar (koşu 60 sn → 3.5 sn). M2 knob'ları için `--asama m2`, ikisini birden etkileyenler için varsayılan (`m2,m3`).

---

## M3-A1. `politika.aday.havuz_boyutu_k`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Eczane başına aday havuzunun boyutu. Hibrit skora göre ilk K satır kısıt katmanına gider. |
| **Varsayılan / aralık** | `20`, makul aralık `10–50` |
| **Artırınca** | `aday.hibrit.recall` 0.110 (K=5) → 0.215 (10) → **0.398 (20)** → 0.660 (50); `yeni_recall` 0.009 → 0.278. Katalog kapsaması 0.18 → 0.59. `veto_orani` **düşer** (0.454 → 0.240): havuz büyüdükçe kuyruğa daha az kısıtlı satırlar giriyor |
| **Azaltınca** | Recall K ile neredeyse doğrusal düşer. K=5'te sistem "her eczaneye en çok aldığı 5 şey" listesine iner |
| **Yanlış ayarın belirtisi** | **Kritik ve kolayca kaçırılır:** K=20'nin üstünde `kisit.liste_recall` **artmıyor** (20 ve 50'de ikisi de 0.119) çünkü `kisit.eczane_haftalik_teklif_tavani` (5) bağlıyor. `kisit.liste_satiri` `eczane_sayısı × tavan` değerine yapışmışsa havuzu büyütmek boşa iştir. Bu ikisi **birlikte** ayarlanmalı |
| **Etkileşim** | `kisit.eczane_haftalik_teklif_tavani` (yukarıdaki tavan), `karisim_agirliklari` (K büyüdükçe karışım farkları silinir — K=50'de her üretici hedefin çoğunu yakalar) |
| **Teşhis** | `uv run python -m experiments.sweep --knob politika.aday.havuz_boyutu_k --values 5,10,20,50 --seeds 5 --asama m3 --profil full` |

---

## M3-A2. `politika.kisit.eczane_haftalik_teklif_tavani`

| Alan | İçerik |
|---|---|
| **Ne yapar** | SPEC §5'in "frekans tavanı". Vetodan **sonra**, eczane başına skor sırasına göre en iyi bu kadar satır listede kalır. Veto değil **budama**: satır kısıtı ihlal ettiği için değil, sıraya girmediği için düşer. |
| **Varsayılan / aralık** | `5`, makul aralık `1–20` |
| **Artırınca** | `kisit.liste_recall` 0.025 (1) → 0.073 (3) → 0.119 (5) → 0.298 (20) — neredeyse doğrusal. `aday.*` metriklerinin **hiçbiri değişmez** (aday üretimi kısıttan habersiz, D6) |
| **Azaltınca** | Liste kısalır, recall doğrusal düşer. Eczane başına 1 teklifte sistem yıllık kampanya takvimi olur, haftalık öneri motoru olmaktan çıkar |
| **Yanlış ayarın belirtisi** | *Çok düşük:* **diğer kısıtların bedelini maskeler.** Ölçüldü: `asgari_kalan_raf_omru_gun` 0 → 240 yapıldığında `veto_sonrasi_recall` 0.331 → 0.286 düşüyor ama `liste_recall` 0.114 → 0.121'e **çıkıyor** — vetolanan satırın yerini bir sonraki aday alıyor ve tavan zaten bağlayıcı olduğu için liste boyu değişmiyor. Kısıt sıkılaştırmasının etkisini `liste_recall`'a bakarak ölçerseniz "kısıt bedava" sonucuna varırsınız; doğru sütun `veto_sonrasi_recall`. *Çok yüksek:* saha kapasitesini aşan liste; M4'te teklif başına marj düşerken toplam MF maliyeti artar |
| **Etkileşim** | `aday.havuz_boyutu_k` (A1'deki doygunluk), bütün `kisit.veto_*` sebepleri (maskeleme) |
| **Teşhis** | `uv run python -m experiments.sweep --knob politika.kisit.eczane_haftalik_teklif_tavani --values 1,3,5,20 --seeds 5 --asama m3 --profil full` |

---

## M3-A3. `politika.aday.karisim_agirliklari.tekrar`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Hibrit karışımda "hücrenin kendi geçmişi" üreticisinin ağırlığı. Üreticiler eczane içi yüzdelik sıraya çevrilip bu ağırlıklarla toplanır. |
| **Varsayılan / aralık** | `0.50`, makul aralık `0–2`. **Varsayılan 1.0'dan 0.5'e ölçümle taşındı** (aşağıda) |
| **Artırınca** | Bir **cephe** boyunca hareket, optimum yok: `recall` 0.236 (0) → 0.348 (0.25) → **0.398 (0.5)** → 0.441 (1.0) → 0.459 (2.0); aynı sırada `yeni_recall` 0.155 → 0.095 → **0.063** → 0.025 → 0.011. Toplam recall ile yeni hücre recall'u **ters yönde** |
| **Azaltınca** | Havuz cross-sell'e kayar; `kapsama` düşer (2.0'da 0.71, 0'da 0.24 — sezgiye aykırı: tekrar üreticisi eczane içinde çeşitsiz ama KATALOG düzeyinde en geniş kapsamaya sahip, çünkü her eczanenin idiyosinkratik sepeti farklı) |
| **Yanlış ayarın belirtisi** | *Çok yüksek:* `yeni_recall` 0.02'nin altına iner ve havuz fiilen "tekrar" tabanına döner — CF, sepet ve cold start üreticileri koşuyor ama teslim edilen havuza girmiyorlar. Kontrol: havuzun `yeni_hucre` payı. *Çok düşük:* zaten haftalık alınan ürünler havuzdan düşer, tükenme kaynaklı ikmal teklifleri kaçar (D2 ihlali değil ama D2'nin işini yapacak aday kalmaz) |
| **Etkileşim** | Diğer dört `karisim_agirliklari` üyesi (toplam normalize edilir, mutlak değil **oran** önemlidir), `havuz_boyutu_k` |
| **Karar** | **0.5'e taşındı.** Seçim kuralı optimizasyon değil, gözlemlenebilir bir **eşitleme**: havuzun yeni hücre payı hedefin yeni hücre payına eşit olsun. Ölçüm (`full`, origin 99): hedefte yeni hücre payı **0.168**; havuzdaki pay ağırlık 0 → 0.62, 0.25 → 0.34, **0.5 → 0.19**, 1.0 → 0.09, 2.0 → 0.03. Bedeli toplam recall 0.450 → 0.418, kazancı yeni hücre recall'u 0.013 → 0.040. **Dürüstlük notu:** bu seçim ölçüme bakıldıktan sonra yapıldı ve M3'ün bir amaç fonksiyonu yok — doğru ağırlığı ancak marj ölçebilen M4 seçebilir |
| **Teşhis** | `uv run python -m experiments.sweep --knob politika.aday.karisim_agirliklari.tekrar --values 0,0.25,0.5,1.0,2.0 --seeds 5 --asama m3 --profil full` |

---

## M3-A4. `politika.aday.miad_baskisi_agirligi`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Depoda kısa miatlı lotu olan SKU'yu havuzda öne çeker: `skor *= (1 + bu × baskı)`, baskı = kısa miatlı adet / eldeki adet. SPEC §2.5'in temizlik güdüsünün aday tarafındaki tek izi. |
| **Varsayılan / aralık** | `0.30`, makul aralık `0–1` |
| **Artırınca** | **Kendini yiyor.** `recall` 0.429 (0) → 0.398 (0.3) → 0.292 (1.0) → 0.178 (3.0). Sebebi doğrudan ölçülüyor: `kisit.veto_raf_omru` 0.046 → 0.125 → 0.216 → **0.303**. Baskı, kısa miatlı SKU'ları havuzun tepesine taşıyor; kısıt katmanı da tam onları `asgari_kalan_raf_omru_gun` ile kesiyor. Havuz kendi vetolanacağı satırlarla doluyor. Yan etki: `yeni_recall` **artıyor** (0.054 → 0.098) çünkü miad baskısı popülerlik/tekrar sırasını bozuyor |
| **Azaltınca** | 0'da temizlik güdüsü tamamen kapanır; M5'in hedefli temizlik politikası için aday üretmez |
| **Yanlış ayarın belirtisi** | `veto_raf_omru` ile `miad_baskisi_agirligi` birlikte tırmanıyorsa ayar yanlıştır: iki katman birbirine ters çalışıyor. Doğru çözüm ağırlığı düşürmek değil, **eşikleri hizalamak** — `aday.miad_baskisi_esik_gun` (180) `kisit.asgari_kalan_raf_omru_gun`'un (120) üstünde kaldığı sürece "baskı altında ama teklif edilebilir" bandı vardır; ikisi kesişirse baskı yalnızca vetolanacak satırları öne çeker |
| **Etkileşim** | `miad_baskisi_esik_gun`, `kisit.asgari_kalan_raf_omru_gun` (yukarıdaki hizalama), `kisit.soguk_zincir_raf_omru_carpani` |
| **Sınır** | **Bu knob'un veto yetkisi yoktur.** 1000'e çıkarılsa bile kırmızı/yeşil reçeteli ürün listeye giremez (SPEC §2.5; `tests/test_constraints.py::test_miad_baskisi_promosyon_vetosunu_asmiyor` ve `scripts/verify_m3.py` stres koşusu) |
| **Teşhis** | `uv run python -m experiments.sweep --knob politika.aday.miad_baskisi_agirligi --values 0,0.3,1.0,3.0 --seeds 5 --asama m3 --profil full` |

---

## M3-A5. `politika.kisit.asgari_kalan_raf_omru_gun`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Teklif edilen lotun en az bu kadar kalan raf ömrü olmalı. FEFO sırasında bu eşiği geçen ilk lot seçilir; hiçbiri geçmiyorsa satır `raf_omru` sebebiyle vetolanır. Kısa miatlı malı eczaneye yıkmak zararı **transfer eder** (SPEC §2.5; M1'de iade oranı ölçüldü). |
| **Varsayılan / aralık** | `120`, makul aralık `60–240` |
| **Artırınca** | `veto_orani` 0.197 (0) → 0.246 (60) → 0.322 (120) → 0.363 (240); `veto_sonrasi_recall` 0.331 → 0.317 → 0.298 → 0.286. Yani her 60 günlük sıkılaştırma ~0.014 recall'a mal oluyor |
| **Azaltınca** | 0'da raf ömrü vetosu tamamen kapanır ve teklif edilen lotun miadına bakılmaz. M1'in ölçtüğü iade mekanizması bu durumda doğrudan devreye girer |
| **Yanlış ayarın belirtisi** | *Çok yüksek:* `veto_raf_omru` toplam vetonun yarısını geçer ve havuzun tepesi boşalır; `ust_dilim_veto_orani` 0.6'nın üstüne çıkar. *Çok düşük:* M5'te iade ve imha kalemleri şişer — M3 bunu **göremez**, çünkü teklifin kabul edilip edilmediğini M4, sonucunu M5 ölçüyor. **Bu knob'un gerçek bedeli M3'te ölçülemez;** burada yalnızca recall maliyeti görünür |
| **Etkileşim** | `soguk_zincir_raf_omru_carpani` (soğuk zincirde eşik bu katsayıyla çarpılır), `aday.miad_baskisi_esik_gun` (A4'teki hizalama), `eczane_haftalik_teklif_tavani` (**bedeli maskeler**, A2) |
| **Teşhis** | `uv run python -m experiments.sweep --knob politika.kisit.asgari_kalan_raf_omru_gun --values 0,60,120,240 --seeds 5 --asama m3 --profil full` |

---

## M3-A6. `politika.aday.hiz_telafi_katsayisi`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Teklif adedini hesaplarken akış hızını çarpan katsayı: `adet = tavan(hız × bu × teklif_kapsama_hafta)`. M2'nin ölçtüğü seyreltmeyi telafi eder — bize gelen akış, eczanenin gerçek tüketiminin `share_of_wallet` kadarlık parçasıdır (medyan oran 0.385, `reports/m2.md` §3.2). |
| **Varsayılan / aralık** | `1.0`, makul aralık `1–3`. Ölçülen "doğru" değer ~**2.6** (= 1/0.385) |
| **Artırınca** | Teklif adedi ve tutarı orantılı büyür. `kisit.veto_kredi_limiti` 0.0002 (1.0) → 0.0033 (2.5) → 0.0052 (5.0): **2.5 katına çıkıyor ama hâlâ binde beş.** `kisit.veto_emilim_tavani` 0.125 → 0.034 → **0.000**: emilim tavanı da aynı telafili hızla kurulduğu için oran sadeleşiyor; tavana yalnızca `adet ≥ 1` tabanının bağladığı çok yavaş hücreler takılıyordu, telafi onları da ölçülebilir hale getiriyor. Net: `veto_orani` 0.322 → 0.243 |
| **Azaltınca** | 1.0'ın altında teklif adedi gerçek ihtiyacın çok altına iner; soğuk zincir minimum sipariş kuralı daha sık devreye girer (yükseltme oranı artar) |
| **Yanlış ayarın belirtisi** | *Çok düşük (varsayılan hali):* teklif adetleri sistematik olarak eksik. M3 bunu **kendi metrikleriyle göremez** — recall adetten bağımsız. Belirti M5'te çıkar: teklif kabul edilir ama miktar tüketim hızının altında kalır, tükenme önlenmez. *Çok yüksek:* eczane emebileceğinden fazla mal alır, iade oranı yükselir |
| **Etkileşim** | `teklif_kapsama_hafta` (ikisi çarpımsal, tek bir "kaç haftalık ihtiyaç" bütçesi oluştururlar), `kisit.azami_kapsama_hafta` (aynı katsayıyla ölçekli — bu yüzden sadeleşiyor), `kisit.kredi_kullanim_tavani` |
| **Karar** | **1.0'da bırakıldı.** Doğru değerin ~2.6 olduğu ölçülü, ama M3'te değiştirmenin bir metriği yok: adedin doğruluğu ancak teklif kabulü (M4) ve stok sonucu (M5) ölçüldüğünde görünür. Bilerek yanlış ve **bilinerek** yanlış bırakıldı; `reports/m3.md` §8'de borç olarak kayıtlı |
| **Teşhis** | `uv run python -m experiments.sweep --knob politika.aday.hiz_telafi_katsayisi --values 1.0,2.5,5.0 --seeds 5 --asama m3 --profil full` |

---

## M3-A7. `politika.aday.yariomur_hafta`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Geri getirmede zaman azalımının yarı ömrü: `ağırlık = 0.5 ** (geçen_hafta / bu)`. CF matrisi, sepet kuralları ve tekrar skoru bu ağırlıkla beslenir. Çeşit (assortment) haftalık churn ile değişiyor; yarı ömür bu kaymayı ne kadar takip ettiğimizi belirler. |
| **Varsayılan / aralık** | `26`, makul aralık `6–104` |
| **Artırınca / azaltınca** | **Etki yok.** `recall` 0.3997 (6) → 0.3976 (26) → 0.3970 (104); `yeni_recall` 0.061 / 0.063 / 0.064. Üçü de seed sapmasının (0.016) çok içinde. Ölçüm dört mertebe aralığı tarıyor ve metrik kıpırdamıyor |
| **Yanlış ayarın belirtisi** | Ölçülemez bir knob olması kendi başına bir bulgudur: bu dünyada bir eczanenin çeşidi 52 hafta içinde radikal değişmiyor (`sim.talep.cesitlendirme.haftalik_churn_orani` düşük), bu yüzden "yakın geçmiş" ile "tüm geçmiş" hemen hemen aynı sıralamayı veriyor. Simülatörde churn oranı yükseltilirse bu knob canlanmalı — canlanmıyorsa azalım hesabı yanlış bağlanmıştır |
| **Etkileşim** | `pencere_hafta` (azalım pencere içinde çalışır; yarı ömür pencereden çok büyükse azalım fiilen kapalıdır), `sim.talep.cesitlendirme.haftalik_churn_orani` (dünya tarafı) |
| **Teşhis** | `uv run python -m experiments.sweep --knob politika.aday.yariomur_hafta --values 6,26,104 --seeds 5 --asama m3 --profil full` |

---

## M3-A8. `politika.kisit.soguk_zincir_min_siparis_adedi` ve `soguk_zincir_min_altinda`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Soğuk zincir ürünlerinde kargo penceresi ve taşıma maliyeti nedeniyle minimum sevkiyat adedi. Teklif adedi altında kalırsa `soguk_zincir_min_altinda` kararı uygulanır: `yukselt` (adet minimuma çıkarılır) veya `veto` (satır listeden çıkar). |
| **Varsayılan / aralık** | `5` ve `yukselt`; adet aralığı `1–50` |
| **Artırınca** | Yükseltilen satır oranı artar, yükseltilen satırların daha büyük kısmı `emilim_tavani`ne takılır. `full` varsayılanda: aday satırlarının **%6.5'i** yükseltiliyor, bunların bir kısmı hemen ardından emilim tavanıyla vetolanıyor (`tests/test_constraints.py::test_soguk_zincir_min_siparis_ihlal_edilmiyor` bu zincirin koptuğunu değil, **işlediğini** sınıyor) |
| **Azaltınca** | 1'de kural fiilen kapanır; soğuk zincir ürünü normal ürün gibi davranır |
| **Yanlış ayarın belirtisi** | *`yukselt` + yüksek minimum:* eczaneye emebileceğinden fazla soğuk zincir malı gitmesin diye emilim tavanı devreye girer ve soğuk zincir ürünleri listeden **sessizce** kaybolur; belirti `veto_emilim_tavani`nin soğuk zincir alt kümesinde yoğunlaşmasıdır. *`veto` modu:* aynı satırlar `soguk_zincir_min` sebebiyle düşer — sayı aynı, ama sebep tablosu doğruyu söyler. Tanı koyabilmek için `veto` modu daha dürüst, üretim için `yukselt` daha kullanışlı |
| **Etkileşim** | `kisit.azami_kapsama_hafta` (yükseltmeyi kesen tavan), `aday.hiz_telafi_katsayisi` (telafi büyüdükçe taban adet minimumun üstüne çıkar ve kural kendiliğinden bağlamaz olur), `sim.envanter.soguk_zincir_minimum_siparis_adedi` (dünyanın karşılığı — **ayrı knob**, aşağıya bakınız) |
| **Teşhis** | `uv run python -m experiments.run --profil full --asama m3 --ad _t --knob politika.kisit.soguk_zincir_min_altinda=veto` → `kisit.veto_soguk_zincir_min` |

> **Neden simülatörle ayrı knob.** `sim.envanter.soguk_zincir_minimum_siparis_adedi` dünyanın kuralı, bu knob politikanın kuralı **bildiği hali**. İkisini tek knob yapmak "politika dünyayı kusursuz biliyor" varsayımını koda gömerdi. Ayrı bırakıldıklarında ikisini farklı ayarlamak doğrudan bir hata modeli üretir: politika 5 sanıp dünya 10 istiyorsa teklif edilen adet sevk edilemez. M3'te ikisi de 5; M5'te bu ayrımın sonucu ölçülebilir hale gelecek.

---

## M3-A9. `politika.kisit.kredi_kullanim_tavani`

| Alan | İçerik |
|---|---|
| **Ne yapar** | DBS limitinin ne kadarına kadar risk alınır. Etkin tavan = `dbs_limiti × bu × (1 − vade_riski × vade_riski_cezasi)`. Kısıt **satır değil portföy** düzeyinde: eczanenin adayları skora göre gezilir, kümülatif tutar kalan limiti aşınca satır vetolanır (aşan satır atlanır, durulmaz — daha küçük bir teklif hâlâ sığabilir). |
| **Varsayılan / aralık** | `0.85`, makul aralık `0.5–1.0` |
| **Artırınca / azaltınca** | **Varsayılan dünyada bağlamıyor:** `kisit.veto_kredi_limiti` = 0.0002 (10.000 aday satırında ~2 satır). Sebebi ölçülü: DBS limiti aylık ciro tahmininin 2.4 katı (~1M TL), haftalık teklif portföyü ise onun binde biri. Kısıt ancak tavan **0.02**'ye indirildiğinde bağlıyor (o ayarda 1.457 satır vetolanıyor, en yüksek limit kullanımı 0.999) |
| **Yanlış ayarın belirtisi** | Kredi vetosunun **hiç** ateşlenmemesi tek başına hata değil — bu dünyada limit gerçekten geniş. Hata, bu durumda "kredi kısıtı çalışıyor" demektir: çalıştığı ancak stres altında görülebilir. `scripts/verify_m3.py` bu yüzden kontrolü daraltılmış tavanla koşar ve **vetonun ateşlendiğini ayrıca doğrular**; ateşlenmezse kontrolü GEÇMİŞ saymaz |
| **Etkileşim** | `vade_riski_cezasi` (riskli eczanede tavanı daraltır), `acik_bakiye_vade_hafta` (açık bakiye penceresi), `aday.hiz_telafi_katsayisi` ve `teklif_kapsama_hafta` (teklif tutarını belirleyen çift), `eczane_haftalik_teklif_tavani` (portföyün kaç satırdan oluştuğu) |
| **Teşhis** | `uv run python -m experiments.sweep --knob politika.kisit.kredi_kullanim_tavani --values 0.02,0.1,0.85 --seeds 3 --asama m3 --profil full` |

---

## M3-B. Aile satırları

Aşağıdakiler tek tek süpürülmedi; mekanizmalarını paylaştıkları aile altında hesap veriyorlar. Her ailenin **birincil kadranı** yukarıda A bölümünde ölçülmüştür.

| # | Aile / üyeler | Ne yapar (mekanizma) | Varsayılan → aralık | Yanlış ayarın belirtisi | Teşhis |
|---|---|---|---|---|---|
| M3-B1 | `aday.pencere_hafta` | **Geri getirmenin hafızası.** CF benzerliği, sepet kuralları ve akış hızı yalnızca son bu kadar haftadan beslenir. Yarı ömürden (A7) farkı: yarı ömür ağırlık verir, bu knob keser | `52` → `26–104` | *Çok kısa:* seyrek hücrelerde ortak eczane sayısı `benzerlik.min_ortak_eczane`in altına düşer ve CF matrisi boşalır — belirti `aday.cf.kapsama`nın çökmesi. *Çok uzun:* kapanmış çeşitler benzerlik üretmeye devam eder | `--knob politika.aday.pencere_hafta --values 26,52,104` |
| M3-B2 | `aday.benzerlik.min_ortak_eczane`, `kirpma`, `komsu_sku_sayisi` | **CF'in gürültü kontrolü.** Kosinüs benzerliği destek ile kırpılır (`sim × n/(n+kirpma)`), `min_ortak_eczane` altındaki çift sıfırlanır, her SKU'nun yalnızca en yakın `komsu_sku_sayisi` komşusu tutulur | `3` / `25.0` / `25` → `2–10` / `5–100` / `10–50` | *Kırpma çok düşük:* tek bir tesadüfi ortak alım yüksek kosinüs üretir ve uzun kuyruk SKU'ları herkese önerilir; belirti `aday.cf.kapsama`nın anormal yükselmesi ama `cf.recall`ün düşmesi. *Komşu sayısı çok düşük:* CF popüler SKU'lara çöker (şu an `cf.kapsama` = 0.116, zaten en dar üretici) | `--knob politika.aday.benzerlik.kirpma --values 5,25,100` |
| M3-B3 | `aday.sepet.pencere_hafta`, `min_destek`, `min_lift` | **Market basket kural madenciliği.** Bir "sepet" = (eczane, `pencere_hafta` bloğu). Kural ancak `min_destek` kadar birlikte görülmüş **ve** `min_lift` üstünde ise tutulur; skor = güven × ln(lift) | `4` / `5` / `1.20` → `1–8` / `3–20` / `1.05–2.0` | *Pencere 1 hafta:* eczane periyodik sipariş veriyor (gözden geçirme 1–4 hafta), tek haftalık sepet aynı siparişi böler ve birliktelik kaybolur — belirti `aday.sepet.recall`ün yarıya inmesi. *`min_lift` 1.0'ın altı:* bağımsız çiftler kural sayılır, sepet üreticisi popülerliğe dönüşür | `--knob politika.aday.sepet.pencere_hafta --values 1,4,8` |
| M3-B4 | `kisit.acik_bakiye_vade_hafta`, `vade_riski_cezasi` | **Kredi kısıtının iki yardımcı kadranı.** Açık bakiye vekili = son bu kadar haftadaki sevkiyat tutarı (gerçek cari hesap tablosu POC'ta yok); ceza riskli eczanede tavanı daraltır | `8` / `0.40` → `4–13` / `0–1` | *Vade penceresi çok uzun:* açık bakiye şişer, kredi vetosu yapay olarak bağlar ve büyük eczaneler cezalandırılır (onların sevkiyat hacmi de büyük). Belirti: veto oranının eczane büyüklüğüyle korelasyonu. *Ceza 0:* `vade_riski_skoru` ölü bir kolon olur | `--knob politika.kisit.acik_bakiye_vade_hafta --values 4,8,13 --sabit politika.kisit.kredi_kullanim_tavani=0.02` |
| M3-B5 | `kisit.azami_kapsama_hafta`, `depo_stok_yeterlilik_carpani` | **Fizibilite tavanları.** Teklif adedi eczanenin bu kadar haftalık ihtiyacını aşamaz; depoda teklif adedinin bu katı kadar mal olmalı | `12.0` / `1.0` → `6–26` / `1.0–3.0` | *Emilim tavanı çok dar:* yavaş hücrelere hiç teklif çıkmaz (`adet ≥ 1` tabanı tavanı aşar) — şu an vetoların %12.5'i bu. *Yeterlilik çarpanı > 1:* aynı lot birden fazla eczaneye söz verilmesin diye pay bırakır; **doğru çözümü bu değil, M5'in tahsis LP'sidir** | `--knob politika.kisit.azami_kapsama_hafta --values 6,12,26` |
| M3-B6 | `aday.soguk_start.komsu_eczane_sayisi`, `soguk_dilim`, `oznitelik_agirliklari.*` (`il`, `hastane_yakinligi`, `sosyoekonomik`, `turizm`, `olcek`, `sgk_recete_orani`, `nobet`) | **Cold start komşuluğu.** Öznitelikler standartlaştırılır, ağırlıklı Öklid mesafesi alınır (`il` kategorik: farklıysa ağırlık kadar mesafe), en yakın `komsu_eczane_sayisi` eczanenin alım oranı skor olur. `hastane_yakinligi` en yüksek ağırlıklı (SPEC §2.2: reçete miksini belirleyen en güçlü özellik). `soguk_dilim` **hiçbir adayı değiştirmez**, yalnızca metriği ikiye böler | `10` / `0.20` / tablo → `5–30` / `0.1–0.5` | *Komşu sayısı çok yüksek:* skor global popülerliğe yakınsar — belirti `soguk_start.recall`ün `populerlik.recall`e yaklaşması (şu an 0.178 vs 0.226, ayrık). *Ağırlıklar hepsi eşit:* `nobet` gibi zayıf sinyaller `hastane_yakinligi`yi bastırır. `soguk_dilim` mutlak eşik değil dilim, çünkü `full`da en az geçmişli eczanenin bile 13 sipariş satırı var ve mutlak eşik hiç bağlamıyordu (metrik NaN çıkıyordu) | `--knob politika.aday.soguk_start.komsu_eczane_sayisi --values 3,10,30` |
| M3-B7 | `aday.karisim_agirliklari.cf`, `sepet`, `soguk_start`, `populerlik` | **Karışımın diğer dört üyesi.** Birincil kadran A3'te (`tekrar`). Ağırlıklar toplama normalize edilir: mutlak değerleri değil **oranları** anlamlıdır | `0.80` / `0.60` / `0.40` / `0.20` → `0–2` | Bir üreticinin ağırlığı 0 yapıldığında hibrit metriği değişmiyorsa o üretici zaten diğerleriyle örtüşüktür. Ölçülmüş örtüşme: `populerlik` ve `cf` yeni hücre recall'unda çok yakın (0.182 vs 0.174) — CF bu dünyada popülerliğin ötesine az geçiyor | `--knob politika.aday.karisim_agirliklari.cf --values 0,0.8,2.0` |
| M3-B8 | `aday.teklif_kapsama_hafta` | **Teklif adedi bütçesi:** kaç haftalık ihtiyaç teklif edilir. `hiz_telafi_katsayisi` (A6) ile çarpımsal; ikisi tek bir bütçe oluşturur | `4.0` → `2–8` | Adet büyüdükçe önce emilim tavanı, sonra kredi limiti bağlar. Etkisi A6 ile aynı eksende; ikisini birlikte artırmak çarpımsaldır ve fark ettirmeden `azami_kapsama_hafta`yı aşar | `--knob politika.aday.teklif_kapsama_hafta --values 2,4,8` |
| M3-B9 | `aday.miad_baskisi_esik_gun` | Depodaki lot bu kadar günden az kalmışsa "baskı altında" sayılır. A4'ün eşiği | `180` → `90–365` | `kisit.asgari_kalan_raf_omru_gun`un (120) **üstünde** kalmalı; altına inerse baskı yalnızca zaten vetolanacak lotları öne çeker (A4'teki kendini yiyen rejim). Belirti: `veto_raf_omru` ile baskı ağırlığının birlikte tırmanması | `--knob politika.aday.miad_baskisi_esik_gun --values 90,180,365` |
| M3-B10 | `aday.degerlendirme.ufuk_hafta`, `origin_sayisi`, `k_degerleri` | **Ölçüm aletinin ayarları**, politikanın değil. Ufuk hem hedef penceresi hem origin adımıdır (pencereler örtüşmesin diye); `k_degerleri` recall@K eğrisinin ölçüldüğü noktalar | `4` / `3` / `[5,10,20,50]` → `2–8` / `1–6` | *Ufuk uzun:* hedef kümesi büyür ve recall mekanik olarak düşer — ufuk söylenmeden recall okunamaz. *Origin sayısı 1:* tek haftanın mevsimselliği metriğe karışır. Origin'ler koşunun sonundan geriye alındığı için `origin_sayisi × ufuk` ısınma penceresine çarparsa koşu açık hatayla düşer | `--knob politika.aday.degerlendirme.ufuk_hafta --values 2,4,8` |
| M3-B11 | `kisit.recete_rengi_vetosu`, `tedarik_guclugu_veto`, `sgk_kapsaminda_mf_serbest`, `soguk_zincir_raf_omru_carpani` | **Regülasyon tablosu.** Bunlar tuning kadranı **değil**, politikanın regülasyon bilgisidir. `recete_rengi_vetosu` config yüklemesinde denetlenir: `urun.promosyon_serbest_kurali.recete_rengi_vetosu`nu kapsamak zorunda, gevşetme denemesi `ValidationError` verir (D6 mekanik kilidi). `sgk_kapsaminda_mf_serbest=false` veto değil **kanal kısıtıdır**: teklif listede kalır, `mf_izinli=false` ile işaretlenir (SPEC §2.5: SGK'da temizlik MF yerine vade ile) | `[KIRMIZI, YESIL]` / `true` / `false` / `1.5` | Kırmızı/yeşil vetosunun ateşlenmemesi bu dünyada normaldir (katalogun %0.3'ü); doğrulama bu yüzden **stres altında** koşar. `tedarik_guclugu_veto` vetoların %8'ini taşıyor — tedarik güçlüğündeki ürünü kampanyayla daha hızlı tüketmek zarar (SPEC §2.1) | `uv run python -m scripts.verify_m3 --kosu full` → "Kirmizi/yesil recete oneri listesinde YOK (stres altinda)" satırı |

---

## M3'te doğmayan, sonraki milestone'lara ait knob'lar

- `policy.scorer.*` — beklenen marj, MF maliyeti, vade fonlama oranı. **M4.** M3 aday üretir ve veto koyar; hangi aksiyonla teklif edileceğini ve beklenen marjı M4 hesaplar.
- `clearance.*` (SPEC §5: `trigger_days`, `salvage_curve`, `safety_factor`, `pharmacist_margin_days`, `disposal_cost_per_unit`) — **M5.** M3'ün `miad_baskisi_agirligi`si temizlik *güdüsünün* aday tarafındaki izidir; salvage eğrisi ve `max_teklif_adedi` kuplajı tahsis katmanına ait.
- `policy.depletion_threshold_days` — hâlâ yok. M2 olasılık üretti, M3 aday havuzu üretti; eşiği koyan karar katmanı **M4**'te doğuyor.
- Exploration / propensity (`bandit.*`) — **M4/M6.** M3'ün havuzu deterministik; keşif ve propensity loglaması (D7) teklif seçimiyle birlikte doğar.

---

# M4 knob'ları — uplift (CATE), aksiyon uzayı ve marj

M4 iki yeni config dosyası getirdi ve `config/policy.yaml`'a iki blok ekledi:

| Dosya / blok | Knob | Ne | Kim okur |
|---|---|---|---|
| `config/response.yaml` → `tepki` | 24 | **Simülatörün** teklif tepkisi (uplift ground truth) | yalnızca `sim/response.py` |
| `config/uplift.yaml` → `uplift` | 19 | Kayıt politikası (D7) + T/X öğrenici + ölçüm | `policy/bandit.py`, `models/uplift.py` |
| `config/policy.yaml` → `politika.aksiyon` | 4 | **Aksiyon uzayı** (D1) | `policy/scorer.py` |
| `config/policy.yaml` → `politika.skor` | 5 | Marj aritmetiği | `policy/scorer.py` |

**Toplam 52 yeni knob.** `tepki.*` bloğu diğerlerinden kategorik olarak farklıdır: bunlar bir politika kadranı değil, **dünyanın zorluk ayarıdır** (CLAUDE.md §7). Politikayı iyileştirmek için `tepki.*` çevrilmez; çevrildiğinde ölçülen şey politikanın kalitesi değil, problemin kendisi değişir. Bu yüzden aşağıda ayrı bir bölümde duruyorlar.

**`tepki` neden `DUNYA_BOLUMLERI`nde değil.** M1 dünyası (sipariş/sevkiyat/stok tabloları) teklif kavramını hiç bilmiyor; tepki katmanı o dünyanın **üzerine**, origin anında kuruluyor ve dünyayı değiştirmiyor. Bu yüzden `dunya_hash` M1/M2/M3 ile birebir aynı kaldı (`9d6191c761d43e52`). M6'da kapalı döngü (rollout) gelince teklifler dünyayı değiştirecek ve bu blok `DUNYA_BOLUMLERI`ne **girmek zorunda** kalacak.

## M4 metrik sözlüğü

Bütün sayılar `profil=full`, 3 ölçüm origin'i (91, 95, 99), 50 eğitim origin'i (38–87). Marjlar **toplam TL**, ortalama değil.

| Metrik | Tanım |
|---|---|
| `m4.<politika>.artimsal_marj` | Politikanın gerçek beklenen marjı **eksi** "hiç teklif verme"nin marjı. Ana metrik. Sentetik dünyada karşı-olgusal elimizde olduğu için tahmin değil, **hesap** |
| `m4.<politika>.teklif_basina_artimsal` | Aynı büyüklüğün teklif başına hali. Politikalar farklı sayıda teklif verebilir |
| `m4.<politika>.yakilan_marj` | **Artımsal marjı NEGATİF olan tekliflerin toplamı.** "Propensity nerede marj yakıyor" sorusunun doğrudan cevabı |
| `m4.<politika>.negatif_teklif_orani` | Aynı kümenin teklif sayısına oranı |
| `m4.marj_farki_tl` | **Çıkış kriteri.** `uplift_x − propensity`, TL |
| `m4.marj_farki_alt/ust` | Eczane blok bootstrap'ı ile %95 aralık |
| `m4.oracle_marj_farki_tl` | `oracle_uplift − oracle_propensity`: **amaç fonksiyonunun tek başına** yarattığı fark, model hatası sıfırken. Yapısal büyüklük budur |
| `m4.tahmin_hatasinin_bedeli_tl` | İkisinin farkı: CATE tahmin hatasının yediği kısım |
| `m4.cate.pehe_t` / `pehe_x` | `sqrt(E[(τ̂ − τ)²])`, yalnızca **izinli** kollarda. Kapalı kolda CATE tanımsız |
| `m4.cate.sira_kor_t` / `sira_kor_x` | Spearman. Politika sırayla çalışır; seviye değil sıra önemli |
| `m4.auuc.<politika>` | Kazanç eğrisinin rassal sıraya göre alanı (Qini'nin marj karşılığı) |
| `m4.heterojenlik.cate_sapmasi` | Gerçek CATE'in (en iyi kol, olasılık ölçeği) standart sapması |
| `m4.heterojenlik.farkli_karar_orani` | İki politikanın ayrışan satır oranı |
| `m4.destek.propensity_min` | Loglanan en küçük propensity. **M6'nın IPS ağırlığının üst sınırı** = 1/bu |
| `m4.destek.kol_orneklemi_min` | En seyrek kolun eğitim örneklemi |
| `ihlal.m4_*` | **Sıfır olmak zorunda**; her koşuda üretilir, sweep tablosunda bağırır |

**Politikalar** (hepsi aynı aday kümesi ve aynı frekans tavanı altında):

| Politika | Amaç fonksiyonu | Ne temsil ediyor |
|---|---|---|
| `teklif_yok` | — | Referans nokta |
| `m3_sabit_kampanya` | M3 skor sırası, sabit aksiyon (en derin MF + taban vade) | Sahadaki "herkese aynı kampanya" |
| `propensity_ham` | `p̂(a)` | Ders kitabı response modeli: dönüşümü maksimize et |
| `propensity` | `p̂(a)·marj(a)`, taban = **0** | Marj farkındalı ama karşı-olgusalı yok sayan |
| `uplift_t` | `p̂(a)·marj(a) − p̂(0)·marj(0)` (T-öğrenici) | CATE politikası |
| `uplift_x` | aynısı, X-öğrenici | CATE politikası |
| `oracle_propensity` / `oracle_uplift` | aynı iki amaç, **gerçek** olasılıkla | Model hatası sıfırken tavan |

**Teşhis komutu formatı:**

```bash
uv run python -m experiments.sweep --knob <yol> --values a,b,c --seeds 3 --asama m4 --profil full
```

`--asama m4` M2/M3'ü atlar. M4 koşusu `full`da ~40 sn (kayıt 2 sn, eğitim 32 sn, ölçüm 1.5 sn).

---

## M4-A1. `tepki.duyarlilik.heterojenlik_carpani`  ⚠ dünya knob'u

| Alan | İçerik |
|---|---|
| **Ne yapar** | Eczane düzeyindeki bütün teklif-duyarlılığı terimlerini (katsayılar + log-normal sigmalar) tek çarpanla ölçekler. `0` → her eczane MF ve vadeye **aynı** tepkiyi verir (`mf_duyarliligi = vade_duyarliligi = 1`). Uplift heterojenliğinin ana kadranı |
| **Varsayılan / aralık** | `1.0`, makul aralık `0–2` |
| **Artırınca** | Gerçek CATE sd 0.099 (0) → 0.113 (0.5) → **0.171 (1.0)** → 0.282 (2.0); üst/alt dilim oranı 31x → 137x. Bütün politikaların artımsal marjı büyür (39.6k → 84.9k): daha duyarlı eczane daha çok satın alır. CATE **öğrenilebilirliği de artar**: `sira_kor_x` 0.514 → 0.659 |
| **Azaltınca** | `0`'da tasarlanmış heterojenlik tamamen kapanır |
| **BEKLENTİYLE GERÇEĞİN AYRIŞTIĞI YER** | **Marj farkı kapanmıyor.** `oracle_marj_farki_tl` = 3.562 (0) → 3.294 (0.5) → 3.371 (1.0) → 3.771 (2.0) TL. Yani eczane düzeyinde hiç heterojenlik olmasa bile propensity ile uplift politikası ayrı sonuç veriyor. Sebep §7.1'de: logit uzayında **sabit** bir teklif etkisi bile olasılık uzayında heterojen uplift üretir (satürasyon), ve `tepki.teklif.ihtiyac_etkilesimi` bu carpandan bağımsız çalışmaya devam eder. "Heterojenlik = eczane farkı" varsayımı yanlış |
| **Yanlış ayarın belirtisi** | `0` yapıldığında `sira_kor_x` düşer ama sıfırlanmaz — modelin öğrendiği şey artık eczane değil **ürün tipi ve ihtiyaç** heterojenliğidir. Sıfırlanıyorsa duyarlılık hesabı yanlış bağlanmıştır |
| **Etkileşim** | `tepki.teklif.ihtiyac_etkilesimi` (A2 — asıl sürücü), `tepki.duyarlilik.*` üyeleri (hepsi bu çarpanla ölçekli) |
| **Teşhis** | `uv run python -m experiments.sweep --knob tepki.duyarlilik.heterojenlik_carpani --values 0,0.5,1.0,2.0 --seeds 3 --asama m4 --profil full` |

---

## M4-A2. `tepki.teklif.ihtiyac_etkilesimi`  ⚠ dünya knob'u — **marj farkının asıl sürücüsü**

| Alan | İçerik |
|---|---|
| **Ne yapar** | Teklifin toplam etkisi `exp(bu × ihtiyac)` ile ölçeklenir. Katsayı **negatif**: stoğu bitmek üzere olan eczane zaten sipariş verecektir (D2), teklif ona boşa gider; stoğu olan eczane ancak bir tavizle öne çekilir. Yani teklif esnekliği acil ihtiyaçla **ters orantılı** |
| **Varsayılan / aralık** | `-1.60`, makul aralık `-3 – 0` |
| **Artırınca (0'a doğru)** | Bütün politikaların artımsal marjı büyür (30.4k → 83.9k): teklif her yerde işe yarar. CATE **daha kolay öğrenilir** (`sira_kor_x` 0.464 → 0.693): ihtiyaç latent, model onu yalnızca `tahmini_kapsama_hafta` ile yaklaşık görüyor, etkileşim zayıfladıkça bu körlük önemsizleşiyor |
| **Azaltınca (−3'e doğru)** | Kesin alıcı ile ikna edilebilir keskin biçimde ayrışır. `propensity.yakilan_marj` −5.5k → −9.7k, negatif teklif oranı 0.096 → 0.158 |
| **ÖLÇÜM — amaç fonksiyonunun tek başına farkı** | | 

| `ihtiyac_etkilesimi` | `oracle_marj_farki_tl` | artımsal marja oranı | `propensity.yakilan_marj` | `sira_kor_x` |
|---|---|---|---|---|
| `0` | 1.842 TL | **%1.7** | −5.525 | 0.693 |
| `−0.8` | 2.605 TL | **%3.1** | −7.508 | 0.638 |
| **`−1.6`** | **3.371 TL** | **%5.1** | −7.727 | 0.576 |
| `−3.0` | 4.017 TL | **%8.3** | −9.694 | 0.464 |

| Alan | İçerik |
|---|---|
| **Okuma** | Fark **monoton** ve mekanizma tek: propensity ile uplift'i ayıran şey "hangi eczane daha duyarlı" (A1 — etkisiz) değil, **teklif etkisinin taban olasılıkla ters ilişkili olması**. Bu terim sıfırlanınca fark %1.7'ye iner ama sıfırlanmaz: geriye satürasyon kalır (§7.1) |
| **Yanlış ayarın belirtisi** | *Sıfıra yakın:* uplift ile propensity fiilen aynı politika olur, `farkli_karar_orani` düşer (0.26 → 0.19) ve M4'ün karşılaştırması anlamını yitirir. *Çok negatif:* teklifin hiçbir işe yaramadığı bir dünya; `m3_sabit_kampanya` −42.9k'ya iner ve bütün politikalar "teklif verme"ye yakınsar |
| **Etkileşim** | `taban.ihtiyac_katsayisi` ve `ihtiyac_referans_hafta` (ihtiyaç tanımının kendisi), `duyarlilik.heterojenlik_carpani` (A1 — bağımsız eksen), `politika.kisit.eczane_haftalik_teklif_tavani` (A3) |
| **Teşhis** | `uv run python -m experiments.sweep --knob tepki.teklif.ihtiyac_etkilesimi --values 0,-0.8,-1.6,-3.0 --seeds 3 --asama m4 --profil full` |

---

## M4-A3. `politika.kisit.eczane_haftalik_teklif_tavani` (M3 knob'u, M4 anlamı)

| Alan | İçerik |
|---|---|
| **Ne yapar** | M3'te "liste ne kadar uzun" sorusuydu; M4'te **kıtlık kadranı**: eczane başına kaç slot var. Aksiyon seçimi bu tavan altında en yüksek kazançlı satırları seçiyor |
| **Varsayılan / aralık** | `5`, makul aralık `1–20` |
| **ÖLÇÜM — hedeflemenin değeri kıtlıkla ölçekleniyor** | | 

| tavan | `oracle_marj_farki_tl` | artımsala oranı | farklı karar oranı | `uplift_x` teklif sayısı |
|---|---|---|---|---|
| `1` | **5.313 TL** | **%15,5** | 0.059 | 599 |
| `3` | 4.425 TL | %7,6 | 0.152 | 1.795 |
| **`5`** | **3.371 TL** | **%5,1** | 0.238 | 2.987 |
| `10` | 1.908 TL | %2,6 | 0.353 | 5.697 |

| Alan | İçerik |
|---|---|
| **Okuma** | Tavan gevşedikçe hemen hemen her uygun satıra teklif çıkıyor, seçilecek bir şey kalmıyor ve iki politika yakınsıyor. **Uplift modellemenin değeri slot kıtlığıyla doğru orantılı.** Bu, M5'in kıt stok altındaki tahsis problemine doğrudan bağlanıyor: orada kıt kaynak slot değil stok olacak, ama mantık aynı |
| **Ters yönlü ilginç detay** | `farkli_karar_orani` tavanla **artıyor** (0.059 → 0.353) ama marj farkı **düşüyor**. Yani politikalar daha çok satırda ayrışıyor ama ayrıştıkları satırların değeri küçülüyor. Karar farkı oranını tek başına "politikalar ne kadar farklı" diye okumak yanlış olurdu |
| **Etkileşim** | `aday.havuz_boyutu_k` (kaç aday arasından seçiliyor), `skor.asgari_teklif_marji` (tavan gevşekken eleme eşiği bağlar) |
| **Teşhis** | `uv run python -m experiments.sweep --knob politika.kisit.eczane_haftalik_teklif_tavani --values 1,3,5,10 --seeds 3 --asama m4 --profil full` |

---

## M4-A4. `politika.skor.tedarikci_mf_destek_orani`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Bedava verilen malın birim maliyetinin ne kadarını üretici üstleniyor. `MF maliyeti = bedava × birim_maliyet × (1 − bu)`. Depo marjı %3–9 olduğu için MF'in ekonomik olup olmadığını **tek başına bu knob belirliyor** |
| **Varsayılan / aralık** | `0.50`, makul aralık `0–1`. **SPEC §8 doğrulama listesinde:** üretici→depo MF destek oranları teyit edilmeli |
| **ÖLÇÜM** | | 

| destek | `propensity` artımsal | `uplift_x` ortalama MF | `m3_sabit_kampanya` | `propensity_ham` | `oracle_marj_farki_tl` (oran) |
|---|---|---|---|---|---|
| `0.0` | 19.190 | 0.010 | **−237.100** | −337.400 | 3.080 (**%8,4**) |
| **`0.5`** | 47.940 | 0.033 | −32.800 | −165.900 | 3.371 (**%5,1**) |
| `0.9` | **132.100** | 0.056 | **+130.700** | −28.740 | 2.977 (**%1,9**) |

| Alan | İçerik |
|---|---|
| **Okuma** | Destek `0.9`da **"herkese 10+1" politikası kârlı hale geliyor** (−32.800 → +130.700). Yani kampanyanın mantıklı olup olmadığı bir hedefleme sorusu değil, önce bir **tedarik anlaşması** sorusu. Aynı sırada uplift'in göreli değeri **düşüyor** (%8,4 → %1,9): MF ucuzladıkça yanlış hedeflemenin bedeli azalıyor |
| **Yanlış ayarın belirtisi** | `m4.uplift_x.ortalama_mf` sıfıra yapışıyorsa MF ekonomik değildir; `m3_sabit_kampanya` artımsal marjı pozitife dönüyorsa kişiselleştirme gereksizleşmeye başlamıştır (taban politikayı geçmek zorlaşır) |
| **Etkileşim** | `aksiyon.mf_oranlari` (derinlik), `urun.marj_kademeleri` (marj bandı — pahalı üründe %3 marj, %10 MF hiçbir destekte ödenmez), `tepki.urun_tipi_mf_carpani` |
| **Teşhis** | `uv run python -m experiments.sweep --knob politika.skor.tedarikci_mf_destek_orani --values 0.0,0.5,0.9 --seeds 3 --asama m4 --profil full` |

---

## M4-A5. `uplift.kayit.kesif_orani` — D7'nin overlap sigortası

| Alan | İçerik |
|---|---|
| **Ne yapar** | Kayıt politikasının keşif payı: bu olasılıkla aksiyon **izinli kollar üzerinde düzgün** seçilir. Varlık sebebi CATE kalitesi değil, **overlap**: `eps > 0` olduğu sürece her izinli kolun propensity'si pozitiftir. `eps = 0` olsaydı bazı (satır, kol) çiftleri hiç gözlenmez, o bölgede karşı-olgusal tahmin ekstrapolasyon olurdu ve M6'nın IPS/DR tahmincisi tanımsız kalırdı. **Config yüklemesi `0`ı reddediyor** |
| **Varsayılan / aralık** | `0.25`, makul aralık `0.05–0.6` |
| **ÖLÇÜM — M4'te ölçülebilir bir faydası yok, bedelini M6 ödeyecek** | | 

| `kesif_orani` | `pehe_x` | `sira_kor_x` | `destek.propensity_min` | **azami IPS ağırlığı** | `oracle_marj_farki_tl` |
|---|---|---|---|---|---|
| `0.05` | 0.1053 | 0.594 | 0.0067 | **149** | 3.371 |
| **`0.25`** | 0.1059 | 0.576 | 0.0086 | **122** | 3.371 |
| `0.6` | 0.1052 | 0.587 | 0.0117 | **85** | 3.371 |

| Alan | İçerik |
|---|---|
| **Okuma** | PEHE ve sıra korelasyonu üç değerde de **aynı** (fark seed sapmasının içinde). Keşif oranı M4'ün metriklerinde görünmüyor; değiştirdiği tek şey `propensity_min` ve dolayısıyla M6'nın IPS ağırlığının üst sınırı (149 → 85). **Bu knob M4'te ölçülemez, M6'da belirleyici olur** — TUNING.md'nin "bu milestone'da ölçülemeyen knob" kategorisinin en temiz örneği |
| **Yanlış ayarın belirtisi** | *Çok düşük:* `destek.kol_orneklemi_min` düşer, seyrek kollar `min_kol_orneklemi` eşiğinin altına iner ve model kurulmaz (τ=0). *Çok yüksek:* kayıt politikası gerçekçiliğini kaybeder — sahada hiçbir sistem tekliflerinin %60'ını rastgele vermez, ve o veriyle eğitilen model gerçek log dağılımını temsil etmez |
| **Etkileşim** | `model.min_kol_orneklemi` (destek eşiği), `egitim.azami_origin_sayisi` (toplam örneklem), `aksiyon.mf_oranlari` (kol sayısı — keşif bütçesi kollara bölünüyor) |
| **Teşhis** | `uv run python -m experiments.sweep --knob uplift.kayit.kesif_orani --values 0.05,0.25,0.6 --seeds 3 --asama m4 --profil full` |

---

## M4-A6. `politika.skor.yillik_fonlama_orani` ve `tedarikci_vade_gun`

| Alan | İçerik |
|---|---|
| **Ne yapar** | Vade kaleminin fiyatı. `fonlama = ciro × yillik_oran × (musteri_vadesi − tedarikci_vadesi) / 365`. **İşareti değişebilir**: müşteri vadesi tedarikçi vademizin altındayken tedarikçi bizi finanse ediyordur ve kalem marja **eklenir** |
| **Varsayılan / aralık** | `0.30` ve `90`; oran için `0–0.6`, tedarikçi vadesi için `30–120`. **SPEC §8 doğrulama listesinde:** depo–üretici ödeme koşulları teyit edilmeli |
| **Neden birinci derece** | Yüksek enflasyon rejiminde vade **marjın kendisiyle aynı mertebede**: depo marjı %3–9 iken 30 günlük vade farkı ciroya oranla %2,5. D1'in aksiyon uzayına vadeyi koyması bu yüzden doğru bir karar; tek başına yüzde iskonto tartışması bu kaldıracı görmezdi |
| **ÖLÇÜM** | `uplift_x` ortalama vadesi: **94,4 gün** (oran 0) → **62,8** (0.30) → **60,7** (0.60). Yani `0`'da politika en uzun vadeyi dağıtıyor (kanal bedava), `0.6`'da taban vadeye yapışıyor (kanal kapalı). Varsayılan ikisinin arasında ve **vade fiilen bir karar değişkeni**. Aynı sırada `oracle_marj_farki_tl` 2.399 → 3.371 → 4.416: vade pahalılaştıkça yanlış hedeflemenin bedeli büyüyor |
| **Artırınca** | Vade uzatan kollar pahalılaşır; politikalar taban vadeye sıkışır. Vade kanalı kapandıkça aksiyon uzayı fiilen MF'e iner. `0.6`'da `m3_sabit_kampanya` **pozitife dönüyor** (+34.900): taban vade artık kârlı olduğu için "herkese teklif" bile değer üretiyor |
| **Azaltınca** | `0`'da vade bedava olur ve her politika en uzun vadeyi verir — kabul olasılığı artar, maliyet yoktur. Aksiyon uzayının vade boyutu tek yönlü bir "hediye"ye döner ve `m3_sabit_kampanya` −100.500'e iner |
| **Yanlış ayarın belirtisi** | `m4.<politika>.ortalama_vade` taban vadeye yapışmışsa vade kanalı ekonomik değildir; en uzun vadeye yapışmışsa fiyatlanmamıştır. Sağlıklı aralıkta ortalama ikisinin arasında olmalı (`full` varsayılanı: 63,1 gün, taban 60) |
| **Etkileşim** | `aksiyon.vade_gunleri` (grid), `aksiyon.taban_vade_gun` (karşı-olgusalın vadesi), `skor.temerrut_ceza_katsayisi` (vadeye en çok değer veren eczane aynı zamanda en riskli olandır — gerilim burada) |
| **Teşhis** | `uv run python -m experiments.sweep --knob politika.skor.yillik_fonlama_orani --values 0.0,0.30,0.60 --seeds 3 --asama m4 --profil full` |

<!-- M4-A-EK -->

---

## M4-A7. `politika.aksiyon.mf_oranlari` ve `vade_gunleri` — aksiyon uzayının kendisi (D1)

| Alan | İçerik |
|---|---|
| **Ne yapar** | Kolların tanımı. Kol sayısı = `1 + |mf_oranlari| × |vade_gunleri|`; kol 0 "teklif yok". D1 gereği burada **yüzde iskonto knob'u yoktur ve olmayacaktır** |
| **Varsayılan / aralık** | `[0.0, 0.02, 0.05, 0.10]` ve `[60, 90, 120]`; MF için `0–0.15`, vade için `taban_vade_gun` ve üstü |
| **MF üst sınırı marj kademesine bağlı** | Depo marjı `%3–9` (`urun.marj_kademeleri`). Tedarikçi desteği olmadan `%10` MF, cironun `%9.5`ini götürür ve pahalı ürünün (marj %3) bütün marjını yer. Bu yüzden derin MF **yalnızca ucuz ürün bandında** rasyoneldir ve model bunu `sku_dsf` / `sku_depo_kar_marji` üzerinden öğrenebilir |
| **Vade neden tek yönlü** | Grid'in tamamı taban vadeye eşit ya da daha uzun. `30` (taban altı) **denendi ve geri alındı**: net işletme sermayesi = müşteri vadesi − tedarikçi vadesi olduğu için vadeyi kısaltmak ciroya oranla ~%2.5 marj kazandırıyor ve bu kazanç hem seviyeye hem artıma **ortak** giriyor; sonuçta bütün politikalar herkese "vade 30" verip aynılaşıyordu (marj farkı −%4). Ölçüm `reports/m4.md` §7.1. Bir kampanya eczaneye standart şarttan kötü vade teklif etmez; vade 30 bir kampanya değil, yeniden pazarlıktır |
| **Yanlış ayarın belirtisi** | *MF gridi çok seyrek:* `m4.destek.kol_orneklemi_min` düşer, o kol destek dışı kalır ve `tau_a = 0` döner — politika o kolu hiç seçmez. `full` varsayılanında en seyrek kol **161 örnek** (`min_kol_orneklemi = 300`'ün altında): `mf=0.02` kolu satırların yalnızca %1.8'inde açık, çünkü `floor(adet × 0.02) ≥ 1` için 50 adet gerekiyor. *Grid çok geniş:* kol başına örneklem düşer, T-öğrenici gürültüye boğulur |
| **Etkileşim** | `koli_katina_yuvarla` ve `aday.teklif_kapsama_hafta` (bedava adedin tam sayı olması adet tabanına bağlı), `skor.tedarikci_mf_destek_orani` (MF'in maliyeti), `skor.tedarikci_vade_gun` (vadenin maliyeti) |
| **Teşhis** | `uv run python -m experiments.run --profil full --asama m4 --ad _t --knob politika.aksiyon.mf_oranlari=[0.0,0.05,0.10,0.20]` → `m4.destek.kol_orneklemi_min` ve `m4.uplift_x.ortalama_mf` |

---

## M4-A8. `politika.aksiyon.koli_katina_yuvarla` — M3'ün bıraktığı borç

| Alan | İçerik |
|---|---|
| **Ne yapar** | MF'li kolda teklif adedini koli katına yukarı yuvarlar. SPEC §2.1 "MF oranları koli katlarına yuvarlanır". İki ayrı kural birlikte çalışır: **(1)** bedava adet `floor(adet × mf)` ile tam sayıdır, sonuç 0 ise o kol **kapalıdır** ("7 adetlik satırda 10+1 anlamsız" — `reports/m3.md` §8 borcu); **(2)** yuvarlama emilim tavanını aşarsa **atlanır**, kanal kapanmaz |
| **Varsayılan / aralık** | `true` |
| **İlk uygulama kanalı kapatıyordu** | Yuvarlama koşulsuz yapıldığında (koli 30, teklif 4 adet → 30 adet) emilim tavanı deliniyor ve MF kolları satırların yalnızca **%27'sinde** açık kalıyordu; aksiyon uzayı fiilen tek boyuta (vade) iniyordu. Atlama kuralıyla kanal açık kalıyor, adet şişmiyor |
| **Yanlış ayarın belirtisi** | `m4.<politika>.ortalama_mf` sıfıra yapışıyorsa ya bedava adet tabanı bağlıyor ya da MF ekonomik değil. Ayrımı `mat.kapali_sebep` verir: `mf_bedava_sifir` (adet küçük) vs `mf_kanali_kapali` (SGK). `full` varsayılanı: 31.245 kol `mf_bedava_sifir`, 29.331 kol `mf_kanali_kapali` |
| **Etkileşim** | `aday.hiz_telafi_katsayisi` ve `aday.teklif_kapsama_hafta` — **teklif adedini belirleyen çift.** Adet büyüdükçe MF kanalı açılır. M3'ün ölçtüğü "doğru" telafi ~2.6, varsayılan hâlâ 1.0 (§M4-B4) |
| **Teşhis** | `uv run python -m experiments.run --profil full --asama m4 --ad _t --knob politika.aksiyon.koli_katina_yuvarla=false` |

---

## M4-B. Aile satırları

### M4-B1. `tepki.*` — simülatörün zorluk ayarı ⚠

Bu blok **politika kadranı değildir.** Çevirmek problemi değiştirir, çözümü iyileştirmez. Birincil kadranı A1 ve A2'de ölçüldü; kalanlar aynı mekanizmaların parçalarıdır.

| # | Üyeler | Ne yapar (mekanizma) | Varsayılan | Yanlış ayarın belirtisi |
|---|---|---|---|---|
| B1a | `taban.kesme`, `ihtiyac_katsayisi`, `ihtiyac_referans_hafta`, `cesit_disi_ihtiyac`, `sow_katsayisi`, `yeni_hucre_cezasi`, `fiyat_katsayisi`, `hucre_gurultu_sigma` | **Teklif yokken sipariş verme olasılığının logit'i** — propensity'nin kendisi. `ihtiyac = exp(−kapsama/referans)`, kapsama = gerçek stok / latent hız (ground truth). `sow_katsayisi` ve `hucre_gurultu_sigma` **latent**: model asla göremez, indirgenemez hata bunlardan | `-1.55` / `2.10` / `5.0` / `0.12` / `0.85` / `-0.80` / `-0.18` / `0.60` | Ortalama taban olasılık (`full`: 0.23, dilim 0–9: 0.028→0.592) 0.05'in altına inerse hiçbir politika ayrışamaz (herkes "ikna edilebilir"); 0.6'nın üstüne çıkarsa hepsi "kesin alıcı" olur ve teklifin işlevi kalmaz |
| B1b | `teklif.taban_etki`, `mf_taban_etkisi`, `mf_referans_orani`, `mf_azalan_us`, `vade_taban_etkisi`, `vade_azalan_us` | **Teklifin logit'e katkısı.** `taban_etki` bedelsiz kolun (MF yok, taban vade) etkisi: sadece görünür olmanın değeri. MF etkisi `(oran/referans)^azalan_us` ile azalan verimli | `0.30` / `0.90` / `0.05` / `0.70` / `0.45` / `0.80` | `taban_etki = 0` yapılırsa bedelsiz kol ölür ve "teklif ver ama bir şey verme" aksiyonu anlamsızlaşır; uplift politikasının en kârlı bandı kapanır. `mf_azalan_us = 1` yapılırsa derin MF'in marjinal etkisi sabitlenir ve MF gridi tepe noktasız olur |
| B1c | `duyarlilik.mf_log_sigma`, `mf_sosyoekonomik`, `mf_olcek`, `mf_sow`, `vade_log_sigma`, `vade_riski`, `vade_dbs`, `vade_stokculuk` | **Heterojenliğin sürücüleri.** Yarısı gözlemlenebilir (sosyoekonomik, ölçek, vade riski, DBS), yarısı **latent** (`sow`, `stokculuk`, log-normal gürültü). Gözlemlenebilir sürücü olmasaydı CATE öğrenilemezdi; latent sürücü olmasaydı model tavana çarpar ve "uplift modeli mükemmel" gibi yanlış bir sonuç çıkardı | `0.55` / `0.30` / `−0.35` / `−0.50` / `0.50` / `0.85` / `−0.40` / `0.30` | Latent katsayılar sıfırlanırsa `sira_kor_x` 0.9'un üstüne çıkar — **şüphelenilmesi gereken durum budur** (CLAUDE.md §7: metrik şüpheli derecede iyiyse önce sızıntı, sonra simülatör kolaylığı) |
| B1d | `urun_tipi_mf_carpani` (`RX`, `OTC`, `TEG`, `DERMOKOZMETIK`, `MEDIKAL`) | MF kanalının ürün tipine göre gücü. SPEC §2.1: kampanya mantığının asıl yaşadığı yer TEG/DERMOKOZMETİK/OTC. **Gözlemlenebilir** heterojenlik: model bunu `sku_urun_tipi` üzerinden öğrenebilir | `0.35` / `1.00` / `1.15` / `1.25` / `0.80` | Hepsi eşitlenirse ürün tarafı heterojenliği kapanır ve CATE yalnızca eczane + ihtiyaç sinyaline kalır. Config yüklemesi eksik tipi **reddeder** (sessizce 0 çarpan olmasın diye) |
| B1e | `miad.direnc_katsayisi` | **Alıcı tarafı direnci** (SPEC §2.5): teklif edilen lotun kalan raf ömrü eczacının toleransının altına düştükçe kabul düşer. Yalnızca teklif kollarında — organik siparişin lotunu biz seçmiyoruz | `-1.40` | Bu dünyada nadiren bağlıyor: kısıt katmanı zaten `asgari_kalan_raf_omru_gun = 120` uyguluyor. **M5'te temizlik rejimi o eşiği gevşetince asıl işlevini görecek** ve teklifin "miadı yaklaşan lotu ver" güdüsünü frenleyen tek terim bu olacak |
| B1f | `miktar.kabul_gurultu_sigma`, `asgari_kabul_orani`, `asiri_adet_direnci`, `kapsama_toleransi`, `asgari_esik_adet` | **Kabul edilen miktar.** İlk ikisi log-normal çarpanın gürültüsü (beklenen değeri kapalı formda hesaplanır — `beklenen_miktar_carpani`). Son üçü **aşırı adet direnci**: teklif eczanenin kapsama hedefini aştıkça kabul düşer | `0.35` / `0.25` / `-0.55` / `2.0` / `6.0` | **Bu terim olmadan marj metriği sömürülebilir.** Ölçüldü: direnç yokken `hiz_telafi_katsayisi` 1.0 → 4.0 yapılınca artımsal marj 13.5k → 35.8k'ya çıkıyordu ve tek fren kısıt katmanındaki emilim tavanıydı. `asgari_esik_adet` olmadan hiç tüketimi olmayan hücrede eşik sıfıra gider ve yeni çeşit açan teklifler yapısal olarak imkânsız olur |

### M4-B2. `uplift.model.*` ve `uplift.x_ogrenici.*` — öğrenicinin kendisi

| # | Üyeler | Ne yapar | Varsayılan | Yanlış ayarın belirtisi |
|---|---|---|---|---|
| B2a | `model.ogrenme_orani`, `azami_agac`, `azami_yaprak`, `min_yaprak_ornegi`, `l2_duzenlilestirme`, `ozellik_orani`, `erken_durdurma`, `dogrulama_orani`, `sabir`, `seed` | `HistGradientBoosting` hiperparametreleri; **13 sınıflandırıcı (μ) + 24 regresör (τ₀, τ₁)** aynı ayarı paylaşır. M2'nin `tukenme.model.*` bloğuyla aynı disiplin | M2 ile benzer, `azami_agac=220` | Ağaç sayısı büyüdükçe T-öğrenicinin **farkı** daha gürültülü olur (iki gürültülü tahminin çıkarması); X-öğrenici bundan daha az etkilenir. `pehe_t − pehe_x` farkının büyümesi bu aşırı uyumun belirtisidir |
| B2b | `model.min_kol_orneklemi` | Bir kolda bundan az gözlem varsa model **kurulmaz** ve `μ_a := μ_0` atanır → `τ_a = 0`. Uydurma yerine "bilmiyorum" | `300` | *Çok düşük:* seyrek kolda ekstrapolasyon yapan CATE, politikayı hiç gözlenmemiş kola sürükler ve M6'nın OPE'si orada tanımsız kalır. *Çok yüksek:* geçerli kollar sessizce ölür — belirti `m4.uplift_x.ortalama_mf`in düşmesi. `full` varsayılanında en seyrek kol 161 örnekle **eşiğin altında** ve kapalı |
| B2c | `x_ogrenici.egilim_kirpma_alt`, `egilim_kirpma_ust` | `g(x) = π_a/(π_0+π_a)` kırpma sınırları. Kırpmasız `g → 0/1` olduğunda birleştirme ağırlığı patlar | `0.05` / `0.95` | Aralık daraltılınca X-öğrenici T'ye yaklaşır (τ₀ ve τ₁ eşit ağırlık alır); genişletilince tek bir örneklem tarafına yapışır |

### M4-B5. `uplift.kayit.*` — kayıt politikasının şekli (D7)

Keşif payı A5'te ölçüldü; aşağıdakiler kayıt politikasının **confounding** yapısını belirliyor. Bunlar bilerek sıfır değil: gerçek loglar hep bir politikadan gelir ve o politika yüksek skorlu satırlara daha çok teklif verir. Confounding'i üretmek öğreticinin işini gerçekçi kılıyor.

| # | Üye | Ne yapar | Varsayılan | Yanlış ayarın belirtisi |
|---|---|---|---|---|
| B5a | `kayit.teklif_taban_olasiligi` | Teklif verme olasılığının tabanı; kontrol grubunun büyüklüğünü belirler | `0.42` | *Çok yüksek:* kontrol kolu küçülür, `μ₀` gürültülenir ve **bütün** CATE tahminleri bozulur (τ = μ_a − μ₀). *Çok düşük:* teklif kolları seyrekleşir, `destek.kol_orneklemi_min` `min_kol_orneklemi` eşiğinin altına iner |
| B5b | `kayit.skor_egilimi` | Teklif verme olasılığı aday skoruyla artar: `q = taban + bu × (yüzdelik_skor − 0.5)` | `0.50` | `0` yapılırsa confounding kalkar ve T-öğrenici ile X-öğrenici arasındaki fark daralır — X'in avantajı tam da bu dengesizlikten doğuyor. Sahadaki logları temsil etmez |
| B5c | `kayit.derin_mf_egilimi` | Eski saha kuralı: riskli/küçük eczaneye daha derin MF ve uzun vade. Kollara softmax ağırlığı verir | `0.90` | Kollar arası örneklem dengesizliğinin ana kaynağı. `0` yapılırsa kollar keşif dağılımına yakınsar, `destek.kol_orneklemi_min` yükselir ama veri gerçekçiliğini kaybeder |
| B5d | `kayit.seed` | Kayıt çekilişlerinin seed'i. Aşama adına gömülü (`kayit_politikasi_<seed>_<origin>`): değiştirmek loglanan aksiyonları değiştirir, **dünyanın çekiliş akışını değiştirmez** | `40230812` | Değiştirildiğinde `dunya_hash` sabit kalmalı; kalmıyorsa seed disiplini kırılmıştır (`core/rng.py`) |

### M4-B3. `uplift.egitim.*` ve `uplift.degerlendirme.*` — ölçüm aletinin ayarları

| # | Üyeler | Ne yapar | Varsayılan | Yanlış ayarın belirtisi |
|---|---|---|---|---|
| B3a | `egitim.ilk_origin_hafta`, `origin_araligi_hafta`, `azami_origin_sayisi`, `sinir_tamponu_hafta` | Kayıt koşusunun origin'leri. Ölçüm origin'lerinden **önce** ve tamponlu; örtüşme **config yüklemesinde reddedilir** (`_m4_zaman_kilidi`). Pencere taşarsa en geç origin'ler tutulur (ölçüme zaman olarak en yakın dönem) | `30` / `1` / `50` / `4` | Ardışık origin'ler büyük ölçüde aynı hücreleri içeriyor: etkin örneklem satır sayısından **küçük**. `azami_origin_sayisi`yi büyütmek `kol_orneklemi_min`i doğrusal artırır ama PEHE'yi orantılı iyileştirmez — bu, bağımsızlığın satırda değil hücrede olduğunun belirtisidir |
| B3b | `degerlendirme.bootstrap_orneklem`, `bootstrap_seed`, `qini_dilim_sayisi` | Eczane blok bootstrap'ı ve kazanç eğrisinin dilim sayısı. Bağımsızlık birimi **eczane**: aynı eczanenin satırları ortak frekans tavanını paylaşıyor | `300` / `20260812` / `10` | Satır bazlı bootstrap'a düşülürse aralık olduğundan **dar** çıkar ve anlamsız bir fark anlamlı görünür (M2/M3'te aynı tuzak ölçüldü) |

### M4-B4. `politika.skor.asgari_teklif_marji` ve M3'ten devralınan adet knob'ları

| # | Üye | Ne yapar | Varsayılan | Not |
|---|---|---|---|---|
| B4a | `skor.asgari_teklif_marji` | Bir satıra teklif çıkması için gereken asgari **beklenen artımsal marj** (TL). Uplift politikasının "teklif verme" kararı bu eşikten geçer. **SPEC §5b.4'ün `uplift.min_effect_threshold` dediği knob budur**; birimi olasılık değil TL, çünkü karar olasılıkta değil marjda veriliyor | `0.0` | Propensity politikası aynı eşiği **seviye** marja uyguladığı için neredeyse hiçbir satırı elemez — farkın kaynaklarından biri budur. Eşiği yükseltmek uplift'in teklif sayısını düşürür, teklif başına marjı yükseltir |
| B4b | `aday.hiz_telafi_katsayisi`, `aday.teklif_kapsama_hafta` (M3 knob'ları) | Teklif adedini belirleyen çift. M3 raporu telafi katsayısının "bilerek yanlış" (1.0, ölçülen doğru değer ~2.6) bırakıldığını ve **M4'te bir sonuç metriğine karşı ayarlanması** gerektiğini yazmıştı | `1.0` / `4.0` | **Bu borç M4'te de kapanmadı ve sebebi artık ölçülü.** Aşırı adet direnci eklenmeden önce telafiyi büyütmek marjı serbestçe artırıyordu (sahte optimum); direnç eklendikten sonra bile tepki fonksiyonu adedin **iade/imha** sonucunu görmüyor. Adedin doğru değeri ancak kapalı döngüde (M5 stok, M6 rollout) ölçülebilir. `reports/m4.md` §9 borç #3 |

---

## M4'te doğmayan, sonraki milestone'lara ait knob'lar

- `clearance.*` (SPEC §5) — **M5.** M4 miad direncini tepki tarafına koydu (`tepki.miad.direnc_katsayisi`) ama salvage eğrisi ve `max_teklif_adedi` kuplajı tahsis katmanına ait.
- Tahsis / LP knob'ları — **M5.** M4 hâlâ aynı lotu birden fazla eczaneye söz verebiliyor (M3'ten devralınan borç #2).
- OPE estimator seçimi, clipping eşiği, değerlendirme ufku (SPEC §5 "Eval tarafı") — **M6.** M4 karşı-olgusalı **biliyor**; M6'nın işi onu bilmeden tahmin etmek. `m4.destek.propensity_min` (0.0074 → azami IPS ağırlığı 136) o milestone'un varyans bütçesini şimdiden belirliyor.
- Thompson sampling posterior genişliği (`bandit.*`) — **M6.** M4'ün kayıt politikası sabit ve ε-karışımlı; posterior güncellemesi kapalı döngü gerektiriyor.

---

# M5 knob'ları — tahsis LP'si ve miad rejimi (D5 + D9)

**19 yeni knob:** `tahsis.lp` (5) + `tahsis.temizlik` (9) + `tahsis.senaryo` (2) + `tahsis.degerlendirme` (3). Hepsi `config/allocation.yaml`da.

Aşağıdaki bütün sayılar `profil=full`, `--asama m5`, **3 seed** ortalamasıdır. Referans koşu `experiments/runs/m5_full` (`config_hash=8e78eda7836a7fbf`, `dunya_hash=9d6191c761d43e52` — M1–M4 dünyasıyla birebir aynı). Seviye (level) sayıları tek dünyadan, **fark** sayıları 3 seed'den okunmalı: imha seviyesi dünyalar arasında 3 kat oynuyor, farklar oynamıyor.

**Bu bloktaki knob'ların üçü kategorik olarak farklı** ve ⚠ ile işaretli: `senaryo.*` bir politika kadranı değil, **soru kadranıdır** (D3'ün "kur tahmin edilmez, senaryolaştırılır" disiplininin stok tarafı). Çevirmek dünyayı değiştirmez, LP'ye sorulan soruyu değiştirir. `dunya_hash` sabit kalır — `tests/test_allocate.py::test_senaryo_dunyayi_degistirmiyor` bunu sınar.

**SPEC §5'in İngilizce knob adlarının karşılıkları:**

| SPEC §5 | bu repo |
|---|---|
| `clearance.trigger_days` | `tahsis.temizlik.tetik_gun` |
| `clearance.salvage_curve` | `tahsis.temizlik.deger_egrisi` |
| `clearance.safety_factor` | `tahsis.temizlik.guvenlik_katsayisi` |
| `clearance.pharmacist_margin_days` | `tahsis.temizlik.eczaci_marji_gun` |
| `disposal_cost_per_unit` | `tahsis.temizlik.imha_birim_maliyeti_dsf_orani` |
| "Tahsis hedefi: kısa vadeli marj vs SOW büyütme" | `tahsis.lp.sow_buyutme_agirligi` |

---

## M5-A1. `tahsis.temizlik.normal_realizasyon_orani` — **stoğun fırsat maliyeti**

| Alan | İçerik |
|---|---|
| **Ne yapar** | LP'ye "teklif etmeyip depoda bıraktığın bir adet ileride organik talep tarafından normal marjla satılır" olasılığını söyler. Lotun devam değeri `v = dsf × depo_kar_marji × bu`; LP bir adedi bugün satmak için artımsal kazancın `v`yi aşmasını ister. **`ranking_only` bu maliyeti hiç ödemez** — LP'nin ondan daha az teklif çıkarmasının tek sebebi budur |
| **Varsayılan / aralık** | `0.85`, makul aralık `0.2–1.0` |
| **Neden birinci derece** | LP ile ranking-only arasındaki bütün fark bu knob'ın üzerinde duruyor. `1.0`da LP fiilen stok biriktiricisi, `0.3`te fiilen ranking-only |

| `realizasyon` | LP talep (adet) | LP karşılanmayan | LP stockout | LP brüt marj | **(a) net marj farkı** | (a) karşılanmayan farkı | ort. gölge fiyat |
|---|---|---|---|---|---|---|---|
| `0.30` | 22.271 | 1.954 | 111,4 | 201.926 | **−6.653** | +1.068 | 2,2 |
| `0.60` | 18.657 | 1.711 | 105,8 | 187.276 | −17.785 | +826 | 4,5 |
| **`0.85`** | 15.283 | 1.432 | 94,0 | **169.451** | **−32.892** | +547 | **6,3** |
| `1.00` | 13.520 | 1.006 | 73,2 | 163.280 | −37.103 | +121 | 7,5 |

| Alan | İçerik |
|---|---|
| **Artırınca** | LP daha çok stok tutar: teklif hacmi düşer (22.271 → 13.520 adet), brüt marj düşer, ortalama gölge fiyat yükselir. **Kendi** karşılanmayan talebi de düşer — çünkü daha az söz veriyor |
| **Azaltınca** | LP `ranking_only`ye yakınsar; hacim ve brüt marj artar, stok tamponu erir |
| **Yanlış ayarın belirtisi** | *Çok yüksek:* `m5.lp.teklif_sayisi` doluyken `m5.lp.talep_adet` `ranking_only`nin yarısına iner — LP satır seçiyor ama adet kısıyor. Aynı anda `m5.lp.imha_adet` yüksek kalıyorsa inanç yanlış: tuttuğu stok satılmıyor, imha ediliyor. *Çok düşük:* gölge fiyat sıfıra yapışır, LP fiilen kıtlık fiyatlaması yapmaz |
| **Bu dünyada inanç YANLIŞ ve raporda öyle yazıyor** | `m5.lp.imha_adet` ≈ 72.000 adet / 3 origin. Depodaki stoğun kayda değer bir kısmı zaten imha ediliyor, yani gerçek realizasyon `0.85`in belirgin biçimde altında. Varsayılan yine de `0.85`te bırakıldı çünkü **bu knob'ı sonuç metriğine karşı fit etmek M5'in işi değil** — kalibre edilmesi gereken bir inanç olduğunu göstermek M5'in işi. `reports/m5.md` §5.1 |
| **Etkileşim** | `senaryo.kit_stok_carpani` (A2 — stok bolken fırsat maliyeti kurgusaldır), `temizlik.tetik_gun` (aynı formülün diğer ucu), `politika.skor.asgari_teklif_marji` (eşik rejim değerine uygulanıyor) |
| **Teşhis** | `uv run python -m experiments.sweep --knob tahsis.temizlik.normal_realizasyon_orani --values 0.30,0.60,0.85,1.00 --seeds 3 --asama m5 --profil full` |

---

## M5-A2. `tahsis.senaryo.kit_stok_carpani` ⚠ — **D5'in ne zaman geçerli olduğunu belirleyen kadran**

| Alan | İçerik |
|---|---|
| **Ne yapar** | Origin görünümündeki tahsis edilebilir lot adetlerini ölçekler. **Dünyayı değiştirmez** (`dunya_hash` sabit), LP'ye sorulan soruyu değiştirir: "stok bu kadar kıt olsaydı?" |
| **Varsayılan / aralık** | `1.0` (dokunma), makul aralık `0.05–1.0` |
| **Neden var** | Doğal dünyada aday talebi depo stoğunu yalnızca 2 SKU'da aşıyor (origin 99, `full`). D5'in "ranking tek başına 400 eczaneye aynı kıt SKU'yu önerir" iddiası o dünyada **görünmez**. Kıtlık üretmeden LP'nin ne yaptığı ölçülemez |

| `kit_stok_carpani` | ranking karş.mayan | LP karş.mayan | **fark** | ranking stockout | LP stockout | ranking brüt | LP brüt | (a) net marj farkı (±sd) |
|---|---|---|---|---|---|---|---|---|
| `1.0` (bol) | 885 | 1.432 | **+547** | 50,9 | 94,0 | 214.878 | 169.451 | −32.892 (±14.697) |
| `0.5` | 1.995 | 2.020 | +24 | 120,2 | 131,7 | 207.556 | 163.496 | −32.043 (±12.212) |
| `0.2` | 4.074 | 2.923 | **−1.151** | 241,1 | 196,7 | 184.565 | 155.073 | −21.437 (±5.670) |
| `0.1` (kıt) | 8.290 | 3.901 | **−4.389** | 452,8 | 297,4 | 127.949 | 117.370 | +2.773 (±12.679) |

| Alan | İçerik |
|---|---|
| **Okuma — işaret dönüyor** | Stok bolken LP **kaybediyor** (karşılanmayan +547, marj −32.892); kıtlaştıkça kazanmaya başlıyor ve `0.2`de karşılanmayan talebi 4.074 → 2.923'e indiriyor. **Tahsis katmanının değeri kıtlıkla ölçekleniyor** — M4'ün "hedeflemenin değeri slot kıtlığıyla ölçekleniyor" bulgusunun (M4-A3) stok tarafındaki tam karşılığı |
| **Dürüstlük notu** | Net marj farkının seed sapması 5.670–14.697 TL, ortalamanın kendisiyle aynı mertebede. **Sağlam olan sinyal karşılanmayan talep farkı**, net marj değil. `0.1`deki +2.773 TL tek başına okunamaz |
| **Yanlış ayarın belirtisi** | `1.0`da bırakılırsa `m5.a.karsilanmayan_farki` **pozitif** çıkar ve rapor "LP işe yaramıyor" der — doğru cevap "bu dünyada kıtlık yok"tur. Ayırt etme kuralı: `m5.lp.stockout_sayisi` ile `m5.ranking_only.stockout_sayisi` ikisi de düşükse senaryo bağlamıyordur |
| **Etkileşim** | `temizlik.normal_realizasyon_orani` (A1 — kıtlığın fiyatı), `lp.aday_lot_sayisi` (A6), `politika.kisit.eczane_haftalik_teklif_tavani` (diğer kıt kaynak) |
| **Teşhis** | `uv run python -m experiments.sweep --knob tahsis.senaryo.kit_stok_carpani --values 1.0,0.5,0.2,0.1 --seeds 3 --asama m5 --profil full` |

---

## M5-A3. `tahsis.temizlik.tetik_gun` — SPEC §5'in "en öğretici sweep"i

| Alan | İçerik |
|---|---|
| **Ne yapar** | Bir lotun **değerinin** düşmeye başladığı kalan raf ömrü (SPEC §5 `clearance.trigger_days`). Teklif edilebilirliği değil, **değeri** belirler — teklif edilebilirlik `temizlik.asgari_kalan_raf_omru_gun`da (B1) |
| **Varsayılan / aralık** | `120`, makul aralık `60–240`. Config yüklemesi `tetik_gun <= temizlik.asgari_kalan_raf_omru_gun` durumunu **reddeder** (pencere boş = rejim ölü) |
| **İşaret değişim noktası kapalı formda** | `kalan_gun* = tetik_gun × imha/(normal+imha)`. Varsayılanlarla `120 × 0.08/(0.05×0.85+0.08) ≈ 78 gün`. Bu eşiğin altında gölge fiyat negatiftir. `policy/allocate.py::isaret_esigi_gun` |

`miad_hizlandirma_gun=60` senaryosu, 3 seed:

| `tetik_gun` | temizlik teklifi | ort. MF (temizlik) | **imha farkı** (temizlik penceresi) | **iade farkı** | net marj farkı (±sd) | negatif gölge lot oranı |
|---|---|---|---|---|---|---|
| `60` | **0** | — | −11 | +103 | +14.844 (±17.484) | 0.064 |
| `90` | 143 | 0.004 | −396 | +248 | +22.079 (±19.729) | 0.077 |
| **`120`** | 269 | 0.006 | −501 | +401 | +22.239 (±20.451) | 0.089 |
| `180` | 506 | 0.016 | −719 | +680 | +23.186 (±23.057) | 0.112 |

| Alan | İçerik |
|---|---|
| **SPEC'in beklediği optimum BU DÜNYADA YOK** | SPEC §5: *"`trigger_days` çok erken → gereksiz marj bırakılıyor; çok geç → imha patlıyor. Arada bir optimum var."* Ölçüm **monoton**: 60'tan 180'e net marj farkı sürekli artıyor, tepe noktası yok. Sebebi `reports/m5.md` §6.2'de: imha işlem maliyeti (DSF'in %8'i) depo marjının (%3–9) **iki katı** mertebesinde, dolayısıyla bir adedi çıkarmanın kazancı MF tavizinin bedelinden neredeyse her zaman büyük. Optimumun doğması için ya imha maliyetinin düşmesi ya da iade cezasının ağırlaşması gerekir |
| **`60`'ta rejim ölü** | Temizlik teklifi **sıfır**: pencere (60 gün) ile teklif edilebilirlik tabanı (45 gün, soğuk zincirde 67,5) arasında pratikte kullanılabilir lot kalmıyor. Yine de +14.844 TL fark var — o fark temizlikten değil, **raf ömrü tabanının gevşemesinden** geliyor (45–120 gün arası lotlar açılıyor). İki etkiyi ayırmak isteyen `temizlik.asgari_kalan_raf_omru_gun`u normal tabana eşitlemeli |
| **Artırınca** | Daha çok lot pencereye girer, MF derinleşir (0.004 → 0.016), imha kazancı ve iade bedeli birlikte büyür |
| **Yanlış ayarın belirtisi** | *Çok düşük:* `m5.hedefli_temizlik.teklif_sayisi_temizlik = 0` — rejim ölü, `m5.golge.*.negatif_golge_lot_orani` düşük. *Çok yüksek:* uzun miatlı sağlıklı lotlar iskontoya girer; belirtisi `m5.b.iade_hedefli_farki`nin imha kazancını geçmesi |
| **Etkileşim** | `imha_birim_maliyeti_dsf_orani` (A4 — **SPEC bu ikisini birlikte süpürmeyi öneriyor**; ikisi de aynı kapalı formun içinde), `deger_egrisi` (B2), `normal_realizasyon_orani` (A1) |
| **Teşhis** | `uv run python -m experiments.sweep --knob tahsis.temizlik.tetik_gun --values 60,90,120,180 --seeds 3 --asama m5 --profil full --sabit tahsis.senaryo.miad_hizlandirma_gun=60` |

---

## M5-A4. `tahsis.temizlik.imha_birim_maliyeti_dsf_orani` — politikanın imha inancı

| Alan | İçerik |
|---|---|
| **Ne yapar** | SPEC §5'in `disposal_cost_per_unit`i, DSF oranı olarak. Salvage fonksiyonunun **alt sınırı**: miadı geçmiş bir adedin değeri `−dsf × bu` |
| **Varsayılan / aralık** | `0.08`, makul aralık `0.01–0.30`. **SPEC §8 doğrulama listesinde:** depo→üretici iade koşulları, kredi oranı ve İTS imha prosedürü teyit edilmeli |
| **Dünyanın karşılığı ayrı bir knob** | `lot.maliyet.imha_birim_maliyeti_dsf_orani` (`0.08`). Politikanın maliyeti *bilmesi* ile dünyanın onu *uygulaması* ayrı şeyler (M3-A4 disiplini). Varsayılanlar **kasıtlı olarak eşit** seçildi: önce inanç doğruyken temizliğin ne kazandırdığı ölçülsün, yanlış inancın bedeli ayrıca süpürülsün |

`miad_hizlandirma_gun=60` senaryosu, 3 seed (dünyanın gerçek imha maliyeti **sabit 0.08**):

| politikanın inancı | temizlik teklifi | ort. MF | imha farkı | iade farkı | net marj farkı |
|---|---|---|---|---|---|
| `0.02` (hafife alıyor) | 242 | 0.004 | −434 | +321 | +21.063 |
| **`0.08`** (doğru) | 269 | 0.006 | −501 | +401 | +22.239 |
| `0.20` (abartıyor) | 300 | 0.010 | −617 | +486 | **+23.859** |

| Alan | İçerik |
|---|---|
| **Okuma — doğru inanç en iyi sonucu vermiyor** | Politika imha maliyetini **abarttığında** (0.20, gerçeğin 2,5 katı) daha çok net marj üretiyor. Bu bir çelişki değil, A1'in belirtisi: `normal_realizasyon_orani = 0.85` LP'yi stok tutmaya itiyor ve imha maliyetini abartmak o eğilimi **telafi ediyor**. İki yanlış inanç birbirini kısmen götürüyor — kalibrasyon yaparken tek knob'a bakmanın neden yanıltıcı olduğunun temiz bir örneği |
| **Artırınca** | Salvage eğrisi aşağı kayar, işaret değişim noktası **sağa** kayar (daha uzun miatlı lotlar da yükümlülük sayılır), MF derinleşir |
| **Yanlış ayarın belirtisi** | `0`a çekilirse temizlik rejimi fiilen kapanır: salvage hiç negatife dönmez, `negatif_golge_lot_orani = 0` olur ve D9 uygulanmamış olur (`verify_m5` bu durumu **kriter olarak** yakalar) |
| **Etkileşim** | `tetik_gun` (A3 — aynı kapalı form), `normal_realizasyon_orani` (A1 — yukarıdaki telafi), `sim.iade.kredi_orani` (dünyanın iade tarafı) |
| **Teşhis** | `uv run python -m experiments.sweep --knob tahsis.temizlik.imha_birim_maliyeti_dsf_orani --values 0.02,0.08,0.20 --seeds 3 --asama m5 --profil full --sabit tahsis.senaryo.miad_hizlandirma_gun=60` |

---

## M5-A5. `tahsis.temizlik.guvenlik_katsayisi` — M2 kuplajının güvenlik payı

| Alan | İçerik |
|---|---|
| **Ne yapar** | SPEC §2.5 kuplajının çarpanı (SPEC §5 `clearance.safety_factor`): `max_teklif_adedi = tuketim_hizi × (kalan_gun − eczaci_marji) × bu`. Tüketim hızı **M2'nin çıktısı** (`OriginGorunumu.hiz_tahmini`); sabit bir tavan yok |
| **Varsayılan / aralık** | `0.70`, makul aralık `0.3–1.0` |

`miad_hizlandirma_gun=60`, 3 seed:

| `guvenlik_katsayisi` | temizlik teklifi | imha farkı | iade farkı | net marj farkı |
|---|---|---|---|---|
| `0.30` | **0** | 0 | −5 | +5.734 |
| **`0.70`** | 269 | −501 | +401 | +22.239 |
| `1.00` | 373 | −664 | +586 | +27.347 |

| Alan | İçerik |
|---|---|
| **Okuma** | `0.30`ta kuplaj o kadar sıkı ki temizlik penceresindeki **hiçbir** (eczane, lot) çifti aday kalmıyor — SPEC §2.5'in "sonuç sıfır veya negatifse o eczane o lot için aday değildir" hükmü tüm pencereyi kapatıyor. Kısıtın gerçekten bağladığının kanıtı: `verify_m5` bu kuplajın `full`da **2.130 (satır, lot, kol) üçlüsü** elediğini, 2.115'inin temizlik penceresinde olduğunu raporluyor |
| **Artırınca** | Daha büyük teklif adedi → daha çok imha kazancı **ve** daha çok iade. `1.0`da güvenlik payı tamamen kalkar: eczacının satabileceğinin tamamı teklif edilir |
| **Yanlış ayarın belirtisi** | *Çok düşük:* `teklif_sayisi_temizlik = 0`, rejim ölü. *Çok yüksek:* `m5.b.iade_hedefli_farki` `kor_iskonto`nunkine yaklaşır — hedefleme fiilen kör iskontoya dönüşür. Ayırt etme oranı: `iade_hedefli_farki / iade_kor_farki` (varsayılanda %37) |
| **Bu knob M2'ye bağlı ve M2 borcunu taşıyor** | Hız tahmini `hiz_telafi_katsayisi = 1.0` ile ölçekleniyor ve M2 ölçümü doğru değerin ~2,6 olduğunu söylüyor (`reports/m2.md` §3.2). Yani kuplaj bugün **gerçek hızın ~%38'i** üzerinden hesaplanıyor ve `guvenlik_katsayisi` bu sistematik küçüklüğü de emiyor. İkisini birlikte süpürmeden tek başına yorumlamak yanlış |
| **Etkileşim** | `eczaci_marji_gun` (B1 — aynı formül), `aday.hiz_telafi_katsayisi` (M2 borcu), `politika.kisit.azami_kapsama_hafta` (normal emilim tavanı) |
| **Teşhis** | `uv run python -m experiments.sweep --knob tahsis.temizlik.guvenlik_katsayisi --values 0.3,0.7,1.0 --seeds 3 --asama m5 --profil full --sabit tahsis.senaryo.miad_hizlandirma_gun=60` |

---

## M5-A6. `tahsis.lp.aday_lot_sayisi` — LP'ye kaç lot seçeneği verilir

| Alan | İçerik |
|---|---|
| **Ne yapar** | Bir aday satırı için LP'ye açılan lot sütunu sayısı (FEFO sırasında ilk N). `1` = "lotu FEFO seçsin, LP yalnızca kimin alacağına karar versin"; `>1` = lot seçimi de LP'nin kararı |
| **Varsayılan / aralık** | `3`, makul aralık `1–6` |

`kit_stok_carpani=0.25`, 3 seed:

| `aday_lot_sayisi` | LP karşılanmayan | LP stockout | LP brüt marj | (a) net marj farkı | kolon sayısı | bütünlük açığı (TL) |
|---|---|---|---|---|---|---|
| **`1`** | **1.160** | 106,1 | 155.447 | **−12.880** | 33.742 | 151,3 |
| `3` | 2.923 | 196,7 | 155.073 | −21.436 | 71.303 | 67,8 |
| `6` | 3.010 | 206,3 | 154.475 | −23.038 | 79.407 | 51,1 |

| Alan | İçerik |
|---|---|
| **Beklentinin tersi ve sebebi öğretici** | Daha çok lot seçeneği LP'yi **kötüleştiriyor**: karşılanmayan talep 1.160 → 3.010'a çıkıyor. Sebep: LP beklenen değere göre planlıyor ve serbestlik arttıkça **en ucuz fırsat maliyetli lotları tam sınırına kadar tüketiyor**; gerçekleşen kabul o sınırın etrafında oynadığında taşan kısım karşılanamıyor. `1`de FEFO zorunlu olduğu için talep lotlara doğal olarak yayılıyor ve tampon kendiliğinden oluşuyor. **LP'nin tampon kavramı yok** — `reports/m5.md` §6.3, borç #2 |
| **Artırınca** | LP gevşetmesi tam sayılı çözüme yaklaşır (bütünlük açığı 151 → 51 TL) ve kolon sayısı 2,4 katına çıkar; çözüm süresi büyür |
| **Yanlış ayarın belirtisi** | `m5.lp.karsilanmayan_adet` `ranking_only`ninkini geçiyorsa LP planı sınıra dayamış demektir. Teşhis: `m5.lp.stockout_sayisi / m5.lp.teklif_sayisi` oranı `ranking_only`ninkinden büyükse |
| **Etkileşim** | `senaryo.kit_stok_carpani` (A2), `butunluk_yuvarlamasi` (B3), `degerlendirme.ornek_sayisi` (gürültünün ölçüldüğü tekrar sayısı) |
| **Teşhis** | `uv run python -m experiments.sweep --knob tahsis.lp.aday_lot_sayisi --values 1,3,6 --seeds 3 --asama m5 --profil full --sabit tahsis.senaryo.kit_stok_carpani=0.25` |

---

## M5-A7. `tahsis.lp.sow_buyutme_agirligi` — tahsis hedefi (SPEC §5)

| Alan | İçerik |
|---|---|
| **Ne yapar** | SPEC §5'in "tahsis hedefi: kısa vadeli marj vs share-of-wallet büyütme" kadranı. Amaç fonksiyonuna `bu × beklenen_ciro × yeni_hucre` primi ekler. `share_of_wallet` **latent** olduğu için prim gözlemlenebilir bir vekile (yeni hücre) bağlanır — doğrudan SOW'a bağlanamaz |
| **Varsayılan / aralık** | `0.0` (saf marj), makul aralık `0–0.10` |

`kit_stok_carpani=0.25`, 3 seed:

| `sow_buyutme_agirligi` | LP teklif | LP brüt marj | LP karşılanmayan | (a) net marj farkı |
|---|---|---|---|---|
| **`0.0`** | 2.947 | 155.073 | 2.923 | −21.436 |
| `0.02` | 2.970 | 159.590 | 2.781 | **−18.033** |
| `0.05` | 2.970 | 156.054 | **2.636** | −19.033 |

| Alan | İçerik |
|---|---|
| **Okuma — bu milestone'da ölçülemiyor** | Etki seed sapmasının içinde ve **beklenen yön görünmüyor**: prim büyüdükçe kısa vadeli marjın düşmesi beklenirdi, düşmüyor. Sebep yapısal: **M5 tek dönemlik.** SOW büyütmenin getirisi tanımı gereği *gelecek* dönemlerde; M5'in ölçüm ufkunda yalnızca maliyeti görünür, faydası görünmez. Bu knob'ın gerçek kadranı **M6'nın kapalı döngüsüdür** ve orada `sim.tedarikci_secimi.sow_toparlanma_hizi` ile birlikte okunmalı |
| **Yanlış ayarın belirtisi** | Prim büyüdükçe `m5.lp.brut_marj` düşmüyorsa prim fiilen bağlamıyordur (yeni hücre oranı düşük). Kontrol: `m4` özellik tablosundaki `yeni_hucre` oranı |
| **Etkileşim** | `aday.karisim_agirliklari.tekrar` (M3 — havuzdaki yeni hücre payını belirleyen), `sim.tedarikci_secimi.sow_toparlanma_hizi` (M6) |
| **Teşhis** | `uv run python -m experiments.sweep --knob tahsis.lp.sow_buyutme_agirligi --values 0.0,0.02,0.05 --seeds 3 --asama m5 --profil full --sabit tahsis.senaryo.kit_stok_carpani=0.25` |

---

## M5-B. Aile satırları

### M5-B1. `tahsis.temizlik.eczaci_marji_gun` ve `asgari_kalan_raf_omru_gun` — kuplajın iki ucu

| # | Üye | Ne yapar (mekanizma) | Varsayılan | Yanlış ayarın belirtisi |
|---|---|---|---|---|
| B1a | `eczaci_marji_gun` | SPEC §5 `clearance.pharmacist_margin_days`. Kuplaj formülünden düşülür: eczacı miadın son gününe kadar satmayı planlamaz. **Politikanın inancı**; dünyanın karşılığı `sim.iade.eczaci_guvenlik_marji_gun = 14` ve ikisi ayrı knob. Varsayılan `30` bilerek dünyadan **muhafazakâr** seçildi | `30.0` | *Çok yüksek:* kısa miatlı lotlarda tavan sıfıra iner ve rejim ölür (belirti: `teklif_sayisi_temizlik = 0`). *Çok düşük:* dünyanın 14 günlük marjının altına inerse teslim edilen mal eczanede yaşlanır; belirti `m5.hedefli_temizlik.iade_adet`in `kor_iskonto`nunkine yaklaşması |
| B1b | `asgari_kalan_raf_omru_gun` | Temizlik rejiminde **teklif edilebilirlik** tabanı. Normal rejimde `politika.kisit.asgari_kalan_raf_omru_gun` (120) bağlar; temizlikte bu (45). Soğuk zincirde ikisi de `soguk_zincir_raf_omru_carpani` ile çarpılır (D6) | `45.0` | Config yüklemesi normal tabandan **büyük** değeri reddeder: temizlik bir gevşetmedir, sıkılaştırma değil (`_m5_miad_kilidi`). Sıfır da olamaz — hiçbir rejimde miadı dolmak üzere olan mal eczaneye yıkılmaz (SPEC §2.5). *Çok gevşek:* `m5.hedefli_temizlik.ortalama_teslim_raf_omru` düşer ve iade patlar |

### M5-B2. `tahsis.temizlik.deger_egrisi`, `egri_ussu`, `basamak_esigi` — salvage'ın şekli

SPEC §5 `clearance.salvage_curve`. Üçü birlikte **tek bir eğriyi** tanımlar; uç noktaları (tetikte `+normal`, miad sonrası `−imha`) eğri tipinden bağımsızdır — `tests/test_allocate.py::test_salvage_egri_tipleri_uc_noktalarda_ayni` bunu sınar. Değiştirdiği şey **aradaki yol**, yani işaret değişiminin nerede olduğu.

| # | Üye | Ne yapar | Varsayılan | Yanlış ayarın belirtisi |
|---|---|---|---|---|
| B2a | `deger_egrisi` | `lineer` / `eksponansiyel` / `basamakli`. `u = kalan_gun/tetik` için sırasıyla `u`, `u^egri_ussu`, `u >= basamak_esigi` | `lineer` | `basamakli`da gölge fiyat iki değer arasında zıplar ve LP eşiğin iki yanındaki lotları **tamamen farklı** muamele eder; belirti `m5.hedefli_temizlik.ortalama_mf_temizlik`in bimodal dağılması |
| B2b | `egri_ussu` | `eksponansiyel` eğrinin üssü. `>1` değeri **geç** düşürür (son ana kadar normal marj beklenir), `<1` erken düşürür | `2.0` | `>1`de işaret değişimi miada yaklaşır: `negatif_golge_lot_orani` düşer, temizlik geç başlar ve imha kazancı küçülür |
| B2c | `basamak_esigi` | `basamakli` eğrinin `u` cinsinden eşiği. Eğri lineer iken **okunmaz** ama config'te durur: eğri tipini değiştirmek tek satırlık olsun | `0.50` | Eğri lineer/eksponansiyelken bu değeri çevirmek hiçbir metriği oynatmıyorsa bu **beklenen** davranıştır, hata değil |

### M5-B3. `tahsis.lp.*` — çözücünün ve teslim biçiminin ayarları

| # | Üye | Ne yapar | Varsayılan | Not |
|---|---|---|---|---|
| B3a | `cozucu_zaman_siniri_sn` | HiGHS süre sınırı. Aşılırsa koşu **düşer** (sessizce kötü bir çözüme razı olunmaz): çözülmemiş bir LP'nin metrikleri okunmamalı | `300` | `full`da tipik LP 71.303 sütun × 3.700 kısıt, 3 origin toplamı **1,7 sn**. Sınır bir güvenlik ağı, tuning kadranı değil |
| B3b | `butunluk_yuvarlamasi` | LP gevşetmesini aynı kısıtlar altında tam sayılı politikaya çevirir (azalan LP ağırlığı sırasında açgözlü geçiş). `false` yapılırsa **teslim edilebilir politika üretilmez** — yalnızca bütünlük açığını ölçmek için | `true` | Açık `full`da %0,00–%0,56 (`lp` 55.339 → 55.338 TL). Yani LP gevşetmesi bu problemde neredeyse tam sayılı çıkıyor; **LP değeri bir ÜST SINIRDIR** ve rapor onu teslim edilen politika sanmamalı |
| B3c | `kredi_kisiti` | DBS limitini LP'nin kısıtı yapar (D6: kısıt katmanı optimizasyonun **içinde** de veto yetkisini korur). `false` yapılırsa limit yalnızca seçim sonrası kontrol edilir (M4 davranışı) ve LP fizibil olmayan bir plan önerebilir | `true` | Bu dünyada kredi bağlamıyor (M3/M4'te de ölçüldü); knob **vetonun bedelini ölçmek** için bilerek kapatılabilir bırakıldı |

### M5-B4. `tahsis.senaryo.miad_hizlandirma_gun` ⚠ — miad senaryosunun kadranı

| Alan | İçerik |
|---|---|
| **Ne yapar** | Origin görünümündeki bütün lotların kalan raf ömründen bu kadar gün düşer. Dünyayı değiştirmez (`dunya_hash` sabit); "depo bu kadar yaşlanmış olsaydı" sorusunu sorar |
| **Varsayılan / aralık** | `0`, makul aralık `0–120` |
| **Ölçüm** | `0` → `60` → `120`: `m5.lp.imha_adet_temizlik` 42.715 → 66.392 → ~10.400 (fast) mertebesinde büyür. **`120` ve üstünde temizlik fiilen imkânsızlaşır**: kuplaj tavanı `(kalan_gun − 30) × hız × 0.7` olduğu için 60 günün altına inen lotlarda aday kalmaz. Bu bir hata değil, ölçülmüş bir sonuç: **temizlik erken başlamak zorunda; lot 60 güne indiğinde kurtarılamaz** |
| **Yanlış ayarın belirtisi** | Çok büyük değerlerde `teklif_sayisi_temizlik` düşerken `imha_adet_temizlik` patlıyorsa senaryo rejimi test etmiyor, sadece dünyayı kurtarılamaz hale getiriyordur |
| **Teşhis** | `uv run python -m experiments.sweep --knob tahsis.senaryo.miad_hizlandirma_gun --values 0,60,120 --seeds 3 --asama m5 --profil full` |

### M5-B5. `tahsis.degerlendirme.*` — ölçüm aletinin ayarları

| # | Üye | Ne yapar | Varsayılan | Yanlış ayarın belirtisi |
|---|---|---|---|---|
| B5a | `ornek_sayisi` | Karşılama simülasyonunun tekrar sayısı. Politika **tahmin edilen** olasılıkla plan yapar, ölçüm **gerçek** olasılıkla örnekler; stockout tam olarak bu farktan doğar ve tek örnekte gürültüdür | `20` | Düşürülürse `m5.*.stockout_sayisi` seed'e göre 2–3 kat oynar. Ölçünün kendisi, dünyanın değil |
| B5b | `ornek_seed` | Örnekleme seed'i. Aşama adına gömülü (`karsilama_<origin>_<politika>_<tekrar>`, `core/rng.py`): değiştirmek örnekleri değiştirir, **dünyanın çekiliş akışını değiştirmez** | `50260812` | Değiştirildiğinde `dunya_hash` sabit kalmalı; kalmıyorsa seed disiplini kırılmıştır |
| B5c | `organik_cikis_pencere_hafta` | Depoda kalan adedin miadına kadar organik talep tarafından tüketilip tüketilemeyeceği, SKU'nun son bu kadar haftalık **gözlemlenen** sevk hızından projelendirilir | `26` | Kısaltılırsa projeksiyon gürültülenir ve `imha_adet` seviyesi kayar; **politikalar arasında aynı uygulandığı için farkları yanlılamaz**, yalnızca seviyeyi kaydırır. Seviyeyi okurken bu varsayımın farkında olunmalı |

---

## M5'te doğmayan, sonraki milestone'lara ait knob'lar

- **Stok tamponu / servis seviyesi knob'ı** — **M6.** LP beklenen değere göre planlıyor ve tampon kavramı yok (A6). Bir `hizmet_seviyesi` knob'ı (`z × sd(çekiliş)` kadar rezerv) stockout ile marj arasındaki takası açık bir kadran haline getirir. **Yapmadım** çünkü SPEC M5'in kapsamı LP + gölge fiyat + miad rejimi; tampon ayrı bir tasarım kararı.
- **Akış duyarlı salvage** — **M6.** SPEC §2.5 salvage'ı yalnızca `kalan_gun`un fonksiyonu tanımlıyor; gerçek imha riski `stok / akış` oranına da bağlı (`reports/m5.md` §8, borç #1). D9'un "aynı LP, işaret değişimi" iddiasını bozmadan eklenebilir ama **D9'un formülünü değiştirmek onay ister** (CLAUDE.md §4).
- OPE estimator seçimi, clipping eşiği, değerlendirme ufku — **M6.** M5 gerçek kabul olasılığını **biliyor**; M6'nın işi onu bilmeden tahmin etmek.

---

# M6 knob'ları — off-policy değerlendirme ve kapalı döngü

**21 yeni knob:** `ope.kayit` (2) + `ope.tahminci` (2) + `ope.propensity` (5 + model bloğu) + `ope.ortusme` (2) + `ope.rollout` (7) + `ope.degerlendirme` (2). Hepsi `config/ope.yaml`da. Artı **bir eski knob'ın yeni anlamı**: `uplift.kayit.kesif_orani` M4'te doğdu, bedelini M6 ödüyor (M6-A3).

**Bu bloktaki knob'ların hiçbiri dünyayı değiştirmez.** `DUNYA_BOLUMLERI` dışındalar: `dunya_hash` sabit kalır. Hepsi ya "aynı loglardan hangi sayıyı çıkarıyorsun" (tahminci tarafı) ya da "o kararı kaç hafta takip ediyorsun" (rollout tarafı) sorusunun kadranı.

**Ölçüm tabanı.** Aşağıdaki offline tabloların hepsi `profil=fast`, `--asama m4,m6`, **5 seed**, hedef politika `uplift_x`, referans `oracle = 17,0278 TL/satır`. Offline tahminciler **satır başına TL** üretir (toplam değil): kayıt tekrar sayısı değişince toplam değişir, satır başına değer değişmez ve oracle karşılaştırması ancak aynı normalizasyonda anlamlıdır.

**Kapalı döngü tablolarında dikkat:** TL metriklerinin seed sapması ortalamalarının mertebesinde. Adet metrikleri (kanibalizm, iade, terminal stok) sağlam, TL metrikleri değil — `reports/m6.md` §7'de ayrı ayrı işaretlendi. M5'in "adet sağlam, TL değil" uyarısı M6'da **daha da** geçerli çünkü rollout 52 haftalık birikimli bir büyüklük üretiyor.

## M6 metrik sözlüğü

| metrik | ne ölçer |
|---|---|
| `m6.offline.<pol>.{ips,ips_kirpmasiz,snips,dogrudan,dr}` | tahmincinin `V(π)` tahmini, TL/satır |
| `m6.offline.<pol>.oracle` | **gerçek** `V(π)` — sentetik dünyanın lüksü, `eval/oracle.py` |
| `m6.denetim.<pol>.<tahminci>.sapma_yuzde` | `(tahmin − oracle) / |oracle|` |
| `m6.ayristirma.<pol>.{varyans,kirpma,propensity}` | IPS sapmasının üç kalemi; **toplamları sapmaya tam eşit** |
| `m6.ayristirma.<pol>.artik` | özdeşlik artığı. Sıfırdan farklıysa ayrıştırma yalan söylüyor |
| `m6.teshis.<pol>.ess_orani` | etkin örneklem / n. Ağırlıkların kaç satıra yığıldığı |
| `m6.teshis.<pol>.agirlik_azami` | `max 1/π`. IPS varyansının üst sınırı |
| `m6.teshis.<pol>.kirpilan_kutle_orani` | kırpmanın sildiği ağırlık kütlesi |
| `m6.teshis.<pol>.ortusme_ihlali_orani` | hedefin seçtiği kolun loglanan olasılığı eşiğin altında olan satır oranı |
| `m6.sapma_sd.<pol>.<tahminci>_sapma` | **bağımsız kayıt koşuları** arasındaki sd — bootstrap değil, gerçek tekrar |
| `m6.propensity.{kalibrasyon_hatasi,ortalama_mutlak_hata,log_orani}` | kullanılan propensity'nin gerçeğinden sapması |
| `m6.online.<pol>.artimsal@<ufuk>` | kapalı döngüde `teklif_yok`a göre birikimli net marj |
| `m6.gecikmeli.<pol>.*` | kanibalizm / iade / SOW kanallarının büyüklüğü |
| `m6.gecikmeli.<pol>.terminal_riskli_pay@<ufuk>` | **ölçünün kendi yanlılığı**: ufuk kesildiğinde rafta duran fazla malın artımsala oranı |
| `m6.kopru.<pol>.<tahminci>@<ufuk>.ufuk_kalemi` | online − offline (TL). M6'nın cevapladığı sorunun kendisi |
| `m6.ozdeslik.ozdeslik_sapma_yuzde` | hedef = kayıt politikası iken IPS ile oracle farkı. **Ön koşul** |

---

## M6-A1. `ope.tahminci.kirpma_esigi` — varyans–yanlılık takasının kendisi

| Alan | İçerik |
|---|---|
| **Ne yapar** | Önem ağırlığı `w = 1[a = π(x)] / π_log(a|x)`'ye tavan koyar. Küçük propensity'li tek bir satır IPS'i tek başına taşıyabilir; tavan bunu keser |
| **Varsayılan / aralık** | `20`, makul aralık `5–100` |
| **Mekanizma** | Kırpmanın yanlılığı **rassal değil yönlü**: tavan yalnızca büyük ağırlıkları keser, dolayısıyla pozitif ödül rejiminde katkıları hep aşağı çeker. `tests/test_ope.py::test_kirpma_kutle_siliyor_ve_yanlilik_yonu_belli` bu yönü sınıyor |

`oracle = 17,03 TL/satır`, `uplift_x`, 5 seed:

| `kirpma_esigi` | IPS | **IPS sapma %** | `kirpma` kalemi | kırpılan kütle | ESS/n | **sd(IPS)** 8 bağımsız koşu |
|---|---|---|---|---|---|---|
| `2` | 7,62 | **−54,8** | −11,62 | %29,5 | 0,416 | **1,39** |
| `5` | 11,85 | −30,0 | −7,40 | %17,3 | 0,352 | 2,86 |
| **`20`** | 18,13 | **+6,6** | −1,12 | %2,8 | 0,195 | 6,60 |
| `100` | 19,25 | +13,2 | 0,00 | %0,0 | 0,158 | 7,18 |
| `10⁶` | 19,25 | +13,2 | 0,00 | %0,0 | 7,18 | 7,18 |

| Alan | İçerik |
|---|---|
| **Artırınca** | Yanlılık kaybolur (`kirpma` kalemi → 0) ama varyans patlar: sd 1,39 → 7,18 (**5 kat**). `100`ün üstünde hiçbir şey değişmiyor çünkü bu dünyada `max 1/π = 51,3` |
| **Azaltınca** | Varyans çöker, yanlılık büyür ve **hep aynı yönde**: `2`de tahmin oracle'ın yarısına iner (−%54,8) |
| **İç optimum VAR ve varsayılan ona yakın** | \|sapma\| `2`de %54,8, `20`de %6,6, `10⁶`da %13,2 → minimum `20` ile `100` arasında. Bu, kırpmanın "her zaman kötü" ya da "her zaman iyi" olmadığının ölçülmüş hâli |
| **Yanlış ayarın belirtisi** | *Çok düşük:* `kirpilan_kutle_orani > %20` ve IPS sistematik olarak `snips`in altında. *Çok yüksek:* `ess_orani < 0,15` ve `sd(IPS)` `sd(DR)`nin iki katı — tahmin tek tek satırların üstünde duruyor |
| **Etkileşim** | `dr_kirpma_esigi` (ayrı knob, aşağıdaki tabloda DR sütununun neden sabit kaldığını açıklıyor), `uplift.kayit.kesif_orani` (A3 — dar keşif ağırlıkları büyütür, aynı tavan daha çok keser), `ope.ortusme.esik` (`1/esik` bir satırın taşıyabileceği azami ağırlık) |
| **Teşhis** | `uv run python -m experiments.sweep --knob ope.tahminci.kirpma_esigi --values 2,5,20,100,1000000 --seeds 5 --profil fast --asama m4,m6 --sabit ope.rollout.ufuk_hafta=1 --sabit 'ope.rollout.raporlanan_ufuklar=[1]' --sabit ope.rollout.teklif_penceresi_hafta=1 --sabit 'ope.rollout.politikalar=["teklif_yok"]'` |

**DR sütunu bu taramada sabit (18,83) ve bu bir hata değil:** DR kendi tavanını (`dr_kirpma_esigi`) kullanıyor. İki eşiğin ayrı knob olmasının sebebi tam olarak bu — DR'nin ağırlığı yalnızca **artığı** ölçekler, IPS'inki ödülün tamamını; aynı sayı ikisinde farklı anlam taşır.

---

## M6-A2. `ope.propensity.kaynak` ve `sicaklik` — D7'nin bedeli, sayıyla

| Alan | İçerik |
|---|---|
| **Ne yapar** | `kaynak = loglanan` D7'nin vaadidir: `π(a|x)` gösterim anında yazıldı. `tahmin` sahada en sık görülen arızayı kurar: log tutulmamış, propensity veriden kestiriliyor. `sicaklik` ise **kontrollü bir bozma** — `π' ∝ π^(1/T)`, `T>1` düzleştirir, `T<1` keskinleştirir |
| **Varsayılan / aralık** | `loglanan`, `1.0`. Sıcaklık makul aralık `0.5–2.0` |

**(a) Loglamanın değeri — `kaynak` taraması (5 seed):**

| `kaynak` | IPS | **IPS sapma %** | DR | **DR sapma %** | ECE | `log_oran` | `propensity` kalemi |
|---|---|---|---|---|---|---|---|
| **`loglanan`** | 18,13 | **+6,6** | 18,83 | +10,4 | **0,000** | **0,000** | **0,000** |
| `tahmin` | 12,91 | **−23,6** | 18,57 | **+9,3** | 0,048 | +0,204 | −5,22 |

**D7'nin bedeli 30 puan.** Propensity'yi loglamak yerine kestirmek IPS'in hatasını +%6,6'dan −%23,6'ya taşıyor. **DR neredeyse hiç kıpırdamıyor** (+%10,4 → +%9,3) — "çift sağlamlık"ın ne demek olduğunun sayısal karşılığı: propensity bozulduğunda sonuç modeli tahmini ayakta tutuyor. `π` doğruyken `propensity` kalemi **tam sıfır**; bu bir yaklaşıklık değil, ayrıştırma merdiveninin üçüncü basamağının özdeşliği.

**(b) Kalibrasyon hatasının YÖNÜ, büyüklüğünden önemli — `sicaklik` taraması (5 seed):**

| `sicaklik` | `propensity` kalemi | ECE | `log_oran` | `max 1/π` | ESS/n | IPS sapma % | sd(IPS) |
|---|---|---|---|---|---|---|---|
| `0.5` (keskinleştir) | **+3,72** | 0,165 | −0,236 | **200,0** | 0,104 | **+28,0** | 9,80 |
| `0.8` | +0,63 | 0,058 | −0,021 | 114,2 | 0,152 | +10,4 | 7,22 |
| **`1.0`** (doğru) | **0,00** | **0,000** | **0,000** | 51,3 | 0,195 | +6,6 | 6,60 |
| `1.5` | −0,03 | 0,091 | −0,042 | 24,1 | 0,288 | +6,2 | 5,99 |
| `2.0` (düzleştir) | +0,57 | 0,132 | −0,097 | 19,1 | 0,337 | +9,5 | 5,83 |

**Asimetri bu tablonun asıl bulgusu.** `0.5` ile `2.0`ın kalibrasyon hatası neredeyse aynı (0,165 vs 0,132) ama sapmaları **6,5 kat** farklı (+3,72 vs +0,57). Sebep tek satırda: keskinleştirme küçük propensity'leri sıfıra doğru iter, `1/π` orada patlar (`max 1/π` 51 → 200). Düzleştirme onları büyütür ve ağırlıklar küçülür. **Aynı büyüklükte bir kalibrasyon hatası, yönüne göre bambaşka bir bedel ödetiyor.**

| Alan | İçerik |
|---|---|
| **İşaretin kuralı** | `işaret(propensity kalemi) = −işaret( E[π_kullanılan − π_log] )`, **eşleşen satırlar üzerinde**. Sıcaklığın yönü tek başına belirlemez: hedef politikanın ağırlıklı olarak hangi kolu seçtiğine bağlıdır. Bu dünyada hedeflerin %71–100'ü kol 0 seçiyor ve kol 0 kütlenin çoğunu taşıyor. **Bunu ilk yazışımda yanlış tahmin ettim** — `reports/m6.md` §6.1 |
| **Yanlış ayarın belirtisi** | `kalibrasyon_hatasi > 0,05` iken `propensity` kalemi hâlâ sıfıra yakınsa ölçüm bozuktur (bozma uygulanmıyor). Tersine `log_orani` sıfırken kalem sıfırdan farklıysa merdivenin üçüncü basamağı sızdırıyordur |
| **Etkileşim** | `kirpma_alt` (kestirilen propensity'nin tabanı — `0.5` sıcaklığında bağlayan taban budur), `kirpma_esigi` (bozulmuş propensity büyük ağırlık üretir, tavan onu keser ve iki kalem karışır), `ortusme.esik` |
| **Teşhis** | `uv run python -m experiments.sweep --knob ope.propensity.sicaklik --values 0.5,0.8,1.0,1.5,2.0 --seeds 5 --profil fast --asama m4,m6 --sabit ope.rollout.ufuk_hafta=1 --sabit 'ope.rollout.raporlanan_ufuklar=[1]' --sabit ope.rollout.teklif_penceresi_hafta=1 --sabit 'ope.rollout.politikalar=["teklif_yok"]'` |

---

## M6-A3. `uplift.kayit.kesif_orani` — M4'ün knob'ı, M6'nın faturası (**beklentim tutmadı**)

M4'te doğdu (`TUNING.md` M4-A5), ama asıl bedeli burada ödeniyor: keşif oranı kayıt politikasının her izinli kola verdiği taban olasılıktır ve OPE'nin bütün ağırlık aritmetiği onun üstünde durur. CLAUDE.md'nin M6 talimatı da doğrudan bunu hedefliyor: *"Offline ile online şüpheli derecede uyuşuyorsa exploration fazla geniş demektir — daralt ve tekrar koş."*

**Daralttım ve tekrar koştum.** `profil=fast`, 5 seed, hedef `uplift_x`:

| `kesif_orani` | IPS sapma % | DR sapma % | `max 1/π` | ESS/n | kırpılan kütle | örtüşme ihlali | kör değer payı | sd(IPS) |
|---|---|---|---|---|---|---|---|---|
| `0.05` | +7,4 | +8,6 | 60,6 | 0,193 | %3,0 | %0,76 | %1,29 | 5,74 |
| `0.10` | +6,3 | +8,5 | 59,0 | 0,193 | %3,0 | %0,66 | %0,98 | 5,83 |
| **`0.25`** | +6,6 | +10,4 | 51,3 | 0,195 | %2,8 | %0,46 | %1,01 | 6,60 |
| `0.50` | +8,7 | +6,5 | 47,0 | 0,193 | %2,2 | %0,24 | %0,25 | 6,83 |

**Beklediğim şey olmadı ve sebebi öğretici.** Keşifi 10 kat daraltmak (0,50 → 0,05) tahminci sapmasını kıpırdatmadı (+%8,7 → +%7,4, seed gürültüsünün içinde), ESS'i hiç değiştirmedi (0,193 ↔ 0,195), azami ağırlığı yalnızca %29 büyüttü (47 → 61).

**Sebep frekans tavanında.** `politika.kisit.eczane_haftalik_teklif_tavani = 5` yüzünden hedef politikaların seçimlerinin **%71'i kol 0** ("teklif yok"). Kol 0'ın propensity'si `1 − q(x)` ≈ 0,4–0,6'dır ve `kesif_orani`ndan **bağımsızdır** — keşif oranı yalnızca teklif kolları *arasında* kütle dağıtır. Yani bu kurulumda OPE'yi ayakta tutan şey keşif genişliği değil, **politikanın çoğu satırda teklif vermemesi**.

**CLAUDE.md'nin teşhisi bu koşuda geçerli değil** ve bunu ölçtüm: offline ile online şüpheli derecede uyuşmuyor — ufuk kalemi taban net marjın **%373'ü** (`fast`), **%169'u** (`full`) (`scripts/verify_m6.py::kontrol_ufuk_ayrismasi`). Yani "fazla geniş exploration" hipotezi burada reddediliyor; ayrışmanın kaynağı örtüşme değil, **ufuk** (§M6-A4).

| Alan | İçerik |
|---|---|
| **Varsayılan / aralık** | `0.25`, makul aralık `0.05–0.50` |
| **Artırınca** | Örtüşme rahatlar (ihlal %0,76 → %0,24, kör değer payı %1,29 → %0,25), canlıda maliyet artar (rassal aksiyon marj yakar) |
| **Azaltınca** | Örtüşme ihlali 3 katına çıkar ama **mutlak seviyesi hâlâ %1'in altında**; bu kurulumda bağlamıyor |
| **Yanlış ayarın belirtisi** | `ortusme_ihlali_orani > %5` VE `kirpilan_kutle_orani > %20` birlikte görülüyorsa keşif gerçekten dar. Tek başına ikisinden biri yeterli değil |
| **Ne zaman gerçekten bağlar** | Frekans tavanı gevşetilirse (`eczane_haftalik_teklif_tavani` ↑) hedef politika daha çok satırda teklif verir, kol 0'ın koruyuculuğu azalır ve keşif oranı birinci derece knob hâline gelir. **İkisi birlikte süpürülmeli** |
| **Etkileşim** | `politika.kisit.eczane_haftalik_teklif_tavani` (asıl eş-knob), `ope.tahminci.kirpma_esigi`, `ope.ortusme.esik` |
| **Teşhis** | `uv run python -m experiments.sweep --knob uplift.kayit.kesif_orani --values 0.05,0.10,0.25,0.50 --seeds 5 --profil fast --asama m4,m6 --sabit 'ope.rollout.politikalar=["teklif_yok","uplift_x","agresif"]'` |

---

## M6-A4. `ope.rollout.ufuk_hafta` — SPEC §5'in "en öğretici anı"; **iddia bu dünyada doğrulanmadı**

| Alan | İçerik |
|---|---|
| **Ne yapar** | Politikanın dünyayla etkileşip sonucunun ölçüldüğü hafta sayısı. Offline tahminciler **tek adımlık** bir ödül ölçer; bu knob o kararın kaç hafta sonrasına kadar takip edildiğini belirler |
| **Varsayılan / aralık** | `52`, makul aralık `4–52` (üst sınır `baslangic_hafta + ufuk <= hafta_sayisi`, `_m6_rollout_kilidi`) |
| **SPEC'in iddiası** | *"Kısa ufukta agresif iskonto kazanır, uzun ufukta kaybeder. Bu kontrastı görmek POC'un en öğretici anı."* |

**Kurulum iddiayı sınayacak biçimde yapıldı:** teklif yalnızca **ilk 4 hafta** veriliyor (`teklif_penceresi_hafta = 4`), ölçüm penceresi değişiyor. Aynı müdahale, farklı ufuk. `profil=fast`, **8 seed**, taban `teklif_yok`:

| `ufuk_hafta` | `uplift_x` artımsal TL (±SE) | `agresif` (MF 0,10 + vade 120) | `agresif_vade` (MF 0 + vade 120) |
|---|---|---|---|
| `4` | −8.676 (±15.842) | **−81.923 (±21.160)** ✔ | −8.843 (±9.972) |
| `13` | −845 (±10.512) | **−88.750 (±19.858)** ✔ | +875 (±13.115) |
| `26` | −4.408 (±25.362) | **−95.488 (±31.412)** ✔ | +22.996 (±20.044) |
| `52` | −11.955 (±51.793) | −80.128 (±53.975) | +64.284 (±62.977) |

✔ = ortalama, standart hatanın iki katından büyük (seed gürültüsünden ayrışıyor).

**İddia doğrulanmadı ve sebebi kâğıt üzerinde görünüyor.** `agresif` **her ufukta** negatif; kısa ufukta kazanıp uzun ufukta kaybetmiyor, hiç kazanmıyor. Sebep tek bir tabloda (`reports/m6.md` §6.2):

| kol | MF | vade | kabul başına marj | p(kabul) | **artımsal TL/satır** |
|---|---|---|---|---|---|
| 0 (teklif yok) | — | 60 | 85,16 | 0,145 | 0 |
| 1 (bedelsiz teklif) | 0,00 | 60 | 85,16 | 0,165 | **+1,77** |
| 3 (`agresif_vade`) | 0,00 | 120 | **21,39** | 0,226 | −6,93 |
| 12 (`agresif`) | 0,10 | 120 | **−128,02** | 0,346 | **−61,16** |

En derin kolun **kabul başına marjı negatif**. Kabul olasılığı 0,145'ten 0,346'ya çıksa bile satılan her adet zarar ediyor. **Kısa ufukta kazanacak bir şey yok ki uzun ufukta kaybetsin.** Aritmetik: tedarikçi desteği sonrası MF maliyeti ≈ DSF'in %4,7'si, vade 60→120 fonlama maliyeti ciro'nun %4,9'u; depo marjı %3–9. Aksiyon uzayının derin ucu **maliyetin altında fiyatlanmış**.

`agresif_vade` ise SPEC'in şeklini **ters yönde** gösteriyor: kısa ufukta negatif (−8.843), uzun ufukta pozitife dönüyor (+64.284). **Ama hiçbiri istatistiksel olarak çözülmüş değil** (SE ≈ ortalama) ve yön iddiasını taşıyacak kadar güçlü değil.

| Alan | İçerik |
|---|---|
| **Artırınca** | Gecikmeli kanallar (kanibalizm / iade / SOW) birikmeye zaman bulur; **aynı anda seed gürültüsü de birikir** — SE 4 haftada ~10–21 bin TL, 52 haftada ~52–63 bin TL. Sinyal ve gürültü birlikte büyüyor |
| **Azaltınca** | Ölçüm keskinleşir ama **ölçünün kendi yanlılığı** büyür: kesim anında rafta duran fazla malın marjı yazılmış, akıbeti belli değil (`terminal_riskli_marj`, `uplift_x` için 10.769 TL / 1.278 adet). Bu yanlılık **teklif veren politikanın lehine** çalışır |
| **Yanlış ayarın belirtisi** | `m6.kopru.<pol>.<tahminci>@<ufuk>.ufuk_kalemi` ufukla **doğrusal** büyüyorsa sürekli müdahaleyi ölçüyorsunuzdur, gecikmeli bedeli değil — `teklif_penceresi_hafta`yı kısaltın (A5) |
| **Bu dünyada karar veremeyeceğiniz şey** | TL cinsinden ufuk karşılaştırması. 8 seed'de SE ortalamanın mertebesinde; `agresif` dışında hiçbir hücre çözülmüyor. **Adet metrikleri (kanibalizm, iade, terminal stok) sağlam, TL değil** |
| **Etkileşim** | `teklif_penceresi_hafta` (A5 — ikisi ayrı soru), `politika.aksiyon.mf_oranlari` / `vade_gunleri` (aksiyon uzayının derin ucu bu dünyada zararına; D1), `politika.skor.yillik_fonlama_orani` ve `tedarikci_mf_destek_orani` (agresif kolun bedelini belirleyen iki knob) |
| **Teşhis** | `uv run python -m experiments.sweep --knob ope.rollout.ufuk_hafta --values 4,13,26,52 --seeds 8 --profil fast --asama m4,m6 --sabit ope.rollout.teklif_penceresi_hafta=4 --sabit 'ope.rollout.raporlanan_ufuklar=[4]' --sabit 'ope.rollout.politikalar=["teklif_yok","uplift_x","agresif","agresif_vade"]'` |

---

## M6-A5. `ope.rollout.teklif_penceresi_hafta` — müdahale süresi ile ölçüm süresini ayıran knob

| Alan | İçerik |
|---|---|
| **Ne yapar** | Rollout `ufuk_hafta` kadar koşar ama teklif yalnızca ilk `teklif_penceresi_hafta` haftada verilir. Sonrasında dünya kendi başına döner |
| **Varsayılan / aralık** | `52` (ufka eşit = her hafta teklif), makul aralık `1–ufuk_hafta` |
| **Neden ayrı bir knob** | "Kaç hafta teklif veriyorsun" ile "kaç hafta bakıyorsun" **farklı sorulardır** ve SPEC §5'in öğretici karşıtlığı ikincisidir. Tek knob olsaydı ufku uzatmak müdahaleyi de uzatırdı ve gecikmeli bedel hiç izole edilemezdi: her hafta yeni teklif geldiği için kanibalizm sürekli tazelenir ve "bugünün kararının 40 hafta sonraki faturası" görülemezdi |
| **Artırınca** | Teklif hacmi ve anlık marj artar; kanibalizm sürekli tazelendiği için gecikmeli bedel **birikimli olarak görünmez hâle gelir** |
| **Azaltınca** | Müdahale sabitlenir, ufuk uzadıkça yalnızca **sonuç** birikir. §M6-A4'ün tablosu bu ayarla (`4`) üretildi — SPEC'in iddiasının sınanabileceği tek kurulum budur |
| **Yanlış ayarın belirtisi** | Pencere = ufuk iken `m6.gecikmeli.<pol>.kanibalizm_organik_siparis_farki` ufukla **doğrusal** büyüyorsa gecikmeli bedeli değil, sürekli müdahaleyi ölçüyorsunuzdur |
| **Etkileşim** | `ufuk_hafta` (A4), `karar_araligi_hafta`, `politika.kisit.eczane_haftalik_teklif_tavani` |
| **Teşhis** | `uv run python -m experiments.run --profil fast --asama m4,m6 --knob ope.rollout.teklif_penceresi_hafta=4 --ad m6_pencere4` |

---

## M6-A6. `ope.ortusme.esik` ve `dusuk_destek_orneklemi` — teşhisin çözünürlüğü

| Alan | İçerik |
|---|---|
| **Ne yapar** | `esik`: hedef politikanın seçtiği kolun loglanan olasılığı bunun altındaysa satır "örtüşme ihlali" sayılır. `dusuk_destek_orneklemi`: bir kolun "destekli" sayılması için gereken asgari loglanmış örnek; altında kalan kollarda DR'nin sonuç modeli **ekstrapole** eder |
| **Varsayılan / aralık** | `0.02` (aralık `0.005–0.10`), `200` (aralık `50–1000`) |
| **İkisi de KALEM DEĞİL, SEBEP** | Sapma ayrıştırmasının üç kalemi (varyans / kırpma / propensity) toplamı sapmaya **tam eşittir**; örtüşme ve ekstrapolasyon o toplamın *içinde* bir kalem değil, kırpma kaleminin **nereden geldiğini** söyleyen teşhislerdir. `kirpma_ortusmeden_gelen_pay` bu bağı sayıya döker |
| **Mekanik kilit** | Eşik, kayıt politikasının taban propensity'sinin (`q_alt × kesif_orani / |kol|`) **altına inemez** — inerse hiçbir satır işaretlenemez ve teşhis ölür. `core/config.py::_m6_ortusme_kilidi`, `tests/test_ope.py::test_ortusme_kilidi_olu_teshisi_reddediyor`. Bu, M5'in "temizlik penceresi boş, rejim ölü" kilidiyle aynı disiplin |
| **Artırınca** | Daha çok satır ihlal sayılır. Bu **yanlış bir ölçüm değil, katı bir ölçümdür** ve `ortusme_ihlali_orani`nde açıkça görünür. `dusuk_destek_orneklemi` artırınca daha çok kol "zayıf destekli" sayılır ve DR'nin ekstrapolasyon teşhisi büyür |
| **Azaltınca** | Teşhis körelir; sınırda tamamen ölür (kilit yakalar) |
| **Yanlış ayarın belirtisi** | `ortusme_ihlali_orani` her koşuda tam **0,000** ise eşik çok düşüktür (ya da kilit devre dışı kalmıştır). `dusuk_destekli_kol_sayisi` **bütün** kolları kapsıyorsa eşik anlamsızdır |
| **Etkileşim** | `uplift.kayit.kesif_orani` (kilit üzerinden), `ope.tahminci.kirpma_esigi` (`1/esik` bir satırın taşıyabileceği azami ağırlık), `uplift.model.min_kol_orneklemi` (M4'ün aynı fikirdeki eşiği) |
| **Teşhis** | `uv run python -m experiments.sweep --knob ope.ortusme.esik --values 0.005,0.02,0.05,0.10 --seeds 3 --profil fast --asama m4,m6 --sabit ope.rollout.ufuk_hafta=1 --sabit 'ope.rollout.raporlanan_ufuklar=[1]' --sabit ope.rollout.teklif_penceresi_hafta=1 --sabit 'ope.rollout.politikalar=["teklif_yok"]'` |

---

## M6-A7. `ope.kayit.tekrar_sayisi` — varyansı BOOTSTRAP'tan değil GERÇEK TEKRARDAN ölçmek

| Alan | İçerik |
|---|---|
| **Ne yapar** | Aynı ölçüm origin'lerinde kayıt politikasını kaç kez bağımsız koşacağı. Her tekrar aynı satırları, farklı aksiyon ve farklı kabul zarlarını görür |
| **Varsayılan / aralık** | `8`, makul aralık `1–20` |
| **Neden var** | `1`de tahmincinin varyansı **ölçülemez**, yalnızca bir çekilişi görülür. `m6.sapma_sd.*` bu tekrarların sd'sidir ve **blok bootstrap'tan farklıdır**: bootstrap aynı logları yeniden örnekler, tekrar dünyayı aynı politikayla **yeniden loglar**. İkincisi tahmincinin gerçek tekrar varyansı, birincisi onun örneklem-içi vekili |
| **Artırınca** | sd tahmini kararlılaşır; maliyet **doğrusal** (her tekrar bir tepki örnekleme koşusu, `n` satırı `R` katına çıkarır). `8` → `20` OPE süresini ~2,5 katına çıkarır |
| **Azaltınca** | `1`de `m6.sapma_sd.*` tanımsız (NaN) döner ve `verify_m6`'nın "DR varyans kazancı" kontrolü ölçülemez hâle gelir |
| **Dikkat: seviye değil, sd değişir** | Tekrar sayısı IPS/SNIPS/DR'nin **beklenen değerini** kaydırmaz (satır başına normalize ediliyor); yalnızca o beklenen değerin ne kadar güvenilir ölçüldüğünü belirler. Ölçüm aletinin ayarı, dünyanın değil |
| **Yanlış ayarın belirtisi** | `m6.sapma_sd.<pol>.ips_sapma` NaN ise tekrar 1'dir. sd'ler seed'e göre 2 kat oynuyorsa tekrar sayısı yetersizdir |
| **Etkileşim** | `ope.degerlendirme.bootstrap_orneklem` (ikisi aynı belirsizliğin iki farklı tahmini; raporda yan yana), `ope.kayit.seed` |
| **Teşhis** | `uv run python -m experiments.sweep --knob ope.kayit.tekrar_sayisi --values 1,4,8,16 --seeds 3 --profil fast --asama m4,m6 --sabit ope.rollout.ufuk_hafta=1 --sabit 'ope.rollout.raporlanan_ufuklar=[1]' --sabit ope.rollout.teklif_penceresi_hafta=1 --sabit 'ope.rollout.politikalar=["teklif_yok"]'` |

---

## M6-B. Aile satırları

### M6-B1. `ope.rollout.*` — kapalı döngünün kurulum kadranları

| # | Üye | Ne yapar (mekanizma) | Varsayılan | Yanlış ayarın belirtisi |
|---|---|---|---|---|
| B1a | `baslangic_hafta` | Dünyanın hangi haftasında dallanılacağı. Taban dünya bu haftaya kadar **bire bir** yeniden koşulur (aynı tohum, aynı çekiliş), sonra teklifler devreye girer. `52` seçildi: 104 haftalık dünyada 52 haftalık ufka yer kalsın diye. **Bedeli var ve ölçülüyor** — M4'ün eğitim penceresi (38–87) ile **36 hafta örtüşüyor** (`m6.rollout.egitim_ortusmesi_hafta`) | `52` | Config yüklemesi `baslangic + ufuk > hafta_sayisi` durumunu **reddeder** (`_m6_rollout_kilidi`): aksi hâlde tablo "52 hafta" yazarken 36 hafta gösterirdi. Örtüşme hata sayılmıyor, **sayıyla raporlanıyor** |
| B1b | `karar_araligi_hafta` | Kaç haftada bir yeniden plan yapılacağı. `1` = her hafta. Maliyetin baskın kalemi budur: her karar haftası tam politika hattını (aday havuzu → veto → CATE → seçim) yeniden koşar | `1` | Büyütmek koşuyu hızlandırır ama politikayı bayatlatır; belirti `m6.online.<pol>.kabul_sayisi`nin düşmesi ve `kabul_olasiligi_gercek`in `kabul_olasiligi_tahmin`den sapması |
| B1c | `raporlanan_ufuklar` | Birikimli değerin raporlandığı kesim noktaları. **Tek bir rollout bütün önekleri verir**; ayrı koşu gerekmez | `[4,13,26,52]` | `ufuk_hafta`yı aşan bir değer yüklemede reddedilir: koşulmamış bir hafta etiketlenemez |
| B1d | `politikalar` | Koşulacak politika listesi. Her biri **tam bir rollout** (ısınma dahil) demek; süre doğrusal artar | `[teklif_yok, propensity, uplift_x, agresif, agresif_vade, lp]` | `teklif_yok` zorunlu (yükleme reddeder): kapalı döngüde artımsal değer ancak teklifsiz dünyaya göre tanımlıdır. Tanımsız ad da reddedilir — config'e yazılıp kodun tanımadığı politika sessizce atlanmasın |
| B1e | `seed` | Rollout haftalarının **ortak rassal sayı (CRN)** tohumu. Her hafta kendi üretecini bundan türetir; iki politika aynı tüketim çekilişini, aynı tedarikçi seçim gürültüsünü ve aynı kabul zarlarını görür | `60230813` | Değiştirmek politikalar arası **farkı** değil, hepsinin ortak dünyasını kaydırır. Fark seed'e göre işaret değiştiriyorsa ölçüm yeterince tekrarlanmamıştır (`--seeds`) |

**CRN'in sınırı açıkça yazılı:** hafta İÇİNDE ikmal döngüsündeki çekiliş sayısı veriye bağlı olduğu için akışlar ayrışır. Büyük varyans kaynakları — tüketim, tedarikçi seçimi, kabul — haftanın başında ve aynı akıştan çekiliyor (`sim/rollout.py::rollout_kos`).

### M6-B2. `ope.propensity.*` — kestirim tarafının ayarları

| # | Üye | Ne yapar | Varsayılan | Not |
|---|---|---|---|---|
| B2a | `kirpma_alt` | Kestirilen ya da bozulmuş propensity'nin alt sınırı. Sıfıra yaklaşan payda tek başına tahmini patlatır | `0.005` | **Sayısal taban, tuning kadranı değil.** `sicaklik = 0.5` taramasında bağlayan sınır budur: `max 1/π = 200 = 1/0.005`. Yükseltmek keskinleştirme bozmasının etkisini yapay olarak küçültür |
| B2b | `model.*` (6 alan) | `kaynak = tahmin` iken propensity'yi kestiren `HistGradientBoostingClassifier`. Kayıt politikası bir softmax karışımı; ağaç modeli onu **tam öğrenemez** ve kalan hata M6'nın "propensity kalibrasyonu" kalemini **mekanik olarak** doğurur | `0.08 / 150 / 31 / 40 / 1.0 / seed` | Modeli güçlendirmek (`azami_agac` ↑) kalibrasyon hatasını düşürür ve IPS'i `loglanan` sonucuna yaklaştırır. **Bu bir iyileştirme değil, ölçülen olgunun küçültülmesidir**: M6-A2'nin gösterdiği şey tam olarak "kestirim ne kadar iyi olursa olsun loglamanın yerini tutmuyor" |
| B2c | `kalibrasyon_kova_sayisi` | ECE ölçümünün eşit frekanslı kova sayısı | `10` | Kova sayısı arttıkça ECE **büyür** (daha ince ayrım). Mutlak değeri değil, knob değerleri arasındaki **sırayı** okuyun |

### M6-B3. `ope.tahminci.dr_kirpma_esigi` ve `ope.degerlendirme.*` — ölçüm aletinin geri kalanı

| # | Üye | Ne yapar | Varsayılan | Not |
|---|---|---|---|---|
| B3a | `dr_kirpma_esigi` | DR'nin artık teriminde kullanılan ayrı tavan | `20.0` | **Ayrı olmak zorunda**: IPS'in ağırlığı ödülün tamamını, DR'ninki yalnızca **artığı** ölçekler. Aynı sayı ikisinde farklı anlam taşır. M6-A1 tablosunda DR sütununun sabit kalması bu ayrımın görünür hâlidir |
| B3b | `degerlendirme.bootstrap_orneklem` | Eczane **blok** bootstrap'ının tekrar sayısı. Bağımsızlık birimi satır değil eczane (M4/M5 ile aynı disiplin): aynı eczanenin satırları ortak frekans tavanını paylaşıyor | `300` | Düşürmek `%95` aralığını oynak yapar ve `verify_m6`'nın "aralık oracle'i kapsıyor" kontrolü seed'e göre değişir |
| B3c | `degerlendirme.bootstrap_seed`, `kayit.seed` | Sırasıyla bootstrap ve kayıt koşusunun tohumları. `kayit.seed` aşama adına gömülüdür (`core/rng.py`): değiştirmek loglanan aksiyonları değiştirir, **dünyanın çekiliş akışını değiştirmez** | `60230814`, `60230812` | Değiştirildiğinde `dunya_hash` sabit kalmalı; kalmıyorsa seed disiplini kırılmıştır |

---

# M7 knob'ları — senaryo yorumu, LLM katmanı ve eval harness

**19 yeni knob:** `senaryo` (3 + rejim başına 5 alan) + `ajan` (9) + `harness` (5). Üç ayrı dosyada, üç ayrı sorumluluk: `config/scenarios.yaml` **karara** girer, `config/agent.yaml` **anlatıma**, `config/harness.yaml` **denetime**. Bu ayrım D8'in kendisi — ajan ve harness knob'larından hiçbiri bir teklifi, bir kolu ya da bir vetoyu değiştiremez.

**Dünya değişmedi.** Üçü de `DUNYA_BOLUMLERI` dışında; `dunya_hash = 9d6191c761d43e52` (M1'den beri aynı). Tam config hash'i değişti (`503ff732…` → `11077f16a27c61a9`, `profil=full`) çünkü `Config`'e üç blok eklendi — M1–M6 sayıları geçerli, hiçbiri bu blokları okumuyor. `scripts/verify_m7.py::kontrol_dunya_degismedi` bunu manifeste bakarak değil, **knob'ları fiilen oynatarak** doğruluyor.

**M7 tablolarında dikkat — `beklenen_artimsal_marj` her rejimde okunamaz.** Erteleme kalemi büyüdükçe "teklif yok" kolunun senaryo marjı negatife dönüyor (`sok`: satırların %96,2'si) ve amaç fonksiyonu `p × marj` olduğu için politika **kabul olasılığı en düşük** kolu seçmeye başlıyor. Aksiyon uzayında (D1) talebi baskılayan bir kol yok; bu bir modelleme artefaktı. `m7.senaryo.<rejim>.talep_baskilayan_orani` her koşuda raporlanıyor ve **sıfırdan belirgin biçimde büyükse o rejimin marj sayısı tek başına okunamaz**. Ayrıntı `reports/m7.md` §6.1.

## M7 metrik sözlüğü

| Metrik | Ne ölçer |
|---|---|
| `m7.senaryo.<rejim>.teklif_sayisi` | O rejimde politikanın çıkardığı teklif satırı |
| `m7.senaryo.<rejim>.artimsal_marj` | Σ `p(a)·marj(a) − p(0)·marj(0)`, **senaryo marjıyla**. Yukarıdaki uyarı geçerli |
| `m7.senaryo.<rejim>.erteleme_tl_adet` | Erteleme kazancının satır ortalaması (TL/adet). Taban rejimde tanımı gereği `0` |
| `m7.senaryo.<rejim>.bekleyemeyen_pay` / `…_teklif_pay` | Lotu güncellemeyi bekleyemeyen satırların oranı (aday / teklif verilen). **Kapının hedefleme yapıp yapmadığı** |
| `m7.senaryo.<rejim>.negatif_taban_marj_orani` | Kol 0'ın senaryo marjı negatif olan satırlar. İşaret dönmesinin ön koşulu |
| `m7.senaryo.<rejim>.talep_baskilayan_orani` | Seçilen kolun kabul olasılığı kontrolünkinden **düşük** olan teklifler. Artefaktın kendisi |
| `m7.fark.<rejim>.kol_degisen` | Tabana göre kolu değişen satır sayısı. **Rejim ayrışmasının birincil ölçüsü** |
| `m7.fark.<rejim>.teklife_giren` / `teklifden_cikan` | Kümenin değişimi; toplamları `teklif_farki`na eşit değildir (satır kümesi de kayıyor) |
| `m7.harness.temiz_bulgu` | Temiz vakalarda çıkan toplam bulgu. **Sıfır olmak zorunda** (yanlış alarm) |
| `m7.harness.mutant_yakalanan` / `mutant_sayisi` | İkisi eşit olmak zorunda. Ayrışırsa bir denetçi o ayarda kör |
| `m7.harness.bulgu.<tip>` | Tip başına toplam bulgu (6 tip) |

---

## M7-A1. `senaryo.ikame_ufku_hafta` — **D4'ün sinyalini karara çeviren kadran**

| Alan | İçerik |
|---|---|
| **Ne yapar** | Depodaki bir adedin, satılmayıp beklendiğinde yeniden fiyatlanmış olarak satılabileceği ufuk. Erteleme kazancı bu ufka göre ölçeklenir: `pay = kırp(1 − güncelleme_beklentisi_hafta / bu, 0, 1)`. Güncelleme ufkun **dışındaysa kalem tam sıfırdır** — taban rejimin nötrlüğü buradan geliyor |
| **Varsayılan / aralık** | `12.0`, makul aralık `4–26` |
| **Neden birinci derece** | Rejimlerin birbirinden ayrışıp ayrışmadığını tek başına belirliyor. `4`te `yuksek` rejimi (beklenti 8 hafta) ufkun dışında kalıyor ve **tabandan ayırt edilemez** hâle geliyor; `26`da her iki sert rejim de tam güçle bağlıyor |

`profil=fast`, 4 seed, `--asama m7` (`m7.senaryo.*`):

| `ikame_ufku_hafta` | `yuksek` erteleme (TL/adet) | `yuksek` artımsal marj | `yuksek` kol değişen | `sok` teklif | `sok` erteleme | `sok` artımsal marj | `sok` kol değişen |
|---|---|---|---|---|---|---|---|
| `4` | **0,00** (ufkun dışında) | 6.730 | 133,3 **± 21,1** | 272,0 **± 30,3** | 14,88 | 512 | 477,5 **± 10,3** |
| **`12`** | 3,18 **± 0,50** | 3.003 | 192,0 **± 18,1** | 206,0 **± 60,0** | 24,80 | 1.946 | 425,3 **± 31,7** |
| `26` | 6,61 **± 1,03** | 1.268 | 291,5 **± 12,7** | 209,3 **± 57,7** | 27,47 | 2.389 | 427,5 **± 29,3** |

**± değerleri 4 seed'in standart sapmasıdır** (`ozet.csv`'deki `*_sd` kolonları, n=4 → standart hata ≈ sd/2). Ortalamayı sapmasız okumak bu tabloda iki farklı sonuç veriyor ve ayrımı yapmadan knob'ı ayarlamak mümkün değil:

| Karşılaştırma | Fark | Okuma |
|---|---|---|
| `yuksek kol_degisen`, üç değer arası | 133,3 → 192,0 → 291,5, SH ≈ 10,5 / 9,1 / 6,3 | **Monoton ve net ayrık.** Bu knob'ın etkisini taşıyan sütun budur |
| `yuksek erteleme` | 0,00 → 3,18 → 6,61, SH ≤ 0,52 | **Net.** `4`teki sıfır tam sıfır, seed varyansı yok (ufkun dışında) |
| `sok kol_degisen`, `4` vs `12` | 52,3 fark, SH<sub>fark</sub> ≈ 16,6 → t ≈ 3,1 | Gerçek |
| `sok kol_degisen`, `12` vs `26` | 2,3 fark, SH<sub>fark</sub> ≈ 21,6 → t ≈ 0,1 | **Ayırt edilemez — gürültü** |
| `sok teklif`, `12` vs `26` | 3,3 fark, SH<sub>fark</sub> ≈ 41,6 → t ≈ 0,08 | **Ayırt edilemez — gürültü** |

| Alan | İçerik |
|---|---|
| **Artırınca** | Daha uzak bir güncelleme bile bugünün kararına girer: erteleme kalemi büyür, MF'li teklif sayısı düşer (bedava adet ertelemenin tam maliyetini öder), `yuksek` rejimin artımsal marjı **6.730 → 1.268**'e iner |
| **Azaltınca** | Rejimler tabana yakınsar. `4`te `yuksek` fiilen **ölür** (`erteleme = 0,00`); geriye yalnızca antisipasyon ve fonlama kanalları kalır ve bu yüzden marjı tabanın **üstüne** çıkar |
| **Yanlış ayarın belirtisi** | *Çok küçük:* `m7.senaryo.yuksek.erteleme_tl_adet = 0` ve `m7.fark.yuksek.kol_degisen` çöker — üç rejim yerine fiilen iki rejim koşuyorsunuz. *Çok büyük:* `m7.senaryo.<rejim>.talep_baskilayan_orani` `sok`ta %89'a çıkar ve artımsal marj metriği okunamaz hâle gelir (§6.1) |
| **`sok` sütunundaki ters yön bir bulgudur, gürültü değil** | Erteleme büyüdükçe `sok`un artımsal marjı **artıyor** (512 → 2.389). Sebep yukarıdaki artefakt: kol 0'ın marjı negatife dönüyor, çıkarılan taban da negatif oluyor ve fark büyüyor. Yani `sok`ta **artımsal marj okunmaz** |
| **⚠ Ama `sok`un yerine okunacak sütun da `sok`ta değil** | İlk yazımda buraya "doğru okunacak sütun `teklif` ve `kol_degisen`" yazmıştım. **SD ile bakınca yanlış:** `sok`un her iki sütunu da `12` ile `26`yı ayırt edemiyor (t ≈ 0,1 ve t ≈ 0,08 — yukarıdaki tablo). O sütunlar yalnızca `4`ü diğer ikisinden ayırıyor. Knob'ın etkisini üç değer boyunca taşıyan sütun **`yuksek kol_degisen`** (133,3 → 192,0 → 291,5, SH ≤ 10,5) ve **`yuksek erteleme`**. `sok` rejimi bu knob için gürültülü bir okuma penceresidir: `sok`ta erteleme kalemi zaten neredeyse her satıra uygulanıyor (bekleyemeyen aday payı yalnızca %3,8) ve ufku değiştirmek seçilen kol kümesini rastgele kaydırıyor |
| **Etkileşim** | `tahsis.temizlik.normal_realizasyon_orani` (aynı formülün "beklersem satılır mı" çarpanı — M5-A1), `politika.kisit.asgari_kalan_raf_omru_gun` (bekleme kapısının eşiği), `politika.skor.yillik_fonlama_orani` (ikinci kanal) |
| **Teşhis** | `uv run python -m experiments.sweep --knob senaryo.ikame_ufku_hafta --values 4,12,26 --seeds 4 --profil fast --asama m7` |

---

## M7-A2. `senaryo.rejimler[].guncelleme_beklentisi_hafta` — D4'ün asıl sinyali

| Alan | İçerik |
|---|---|
| **Ne yapar** | Referans kur güncellemesine kalan süre. **Rejimin birinci parametresi kurun seviyesi değil budur** (D4: "asıl makro sinyal referans kur güncelleme beklentisidir"). İki yere birden girer: erteleme payına (`1 − bu/ikame_ufku`) ve bekleme kapısına (`kalan_gun − bu×7 ≥ asgari_raf_omru`) |
| **Varsayılan / aralık** | `baz 26` / `yuksek 8` / `sok 2` hafta; makul aralık `1–52` |
| **İki yere birden girmesi tesadüf değil** | Aynı fiziksel olgu iki farklı kısıt üretiyor: güncelleme **yakınsa** beklemenin değeri yüksektir (pay ↑) ama **beklenebilecek lot sayısı da fazladır** (kapı gevşer, `sok`ta bekleyemeyen pay yalnızca %3,8). Uzaksa tersi. Bu yüzden hedefleme etkisi uçlarda değil **ortada** en güçlü: `full`da bekleyemeyen aday payı `baz %27,6 → yuksek %10,2 → sok %3,8` |
| **Artırınca** | Erteleme payı düşer, kalem küçülür; ama bekleme kapısı **sıkılaşır** ve daha çok satır "bekleyemez" olur. İki etki zıt yönde |
| **Azaltınca** | Kalem büyür, kapı gevşer. `sok`ta olan bu: kalem 33,95 TL/adet, bekleyemeyen aday payı %3,8 — yani kalem neredeyse **herkese** uygulanıyor ve rejim hedefleme değil **seviye** kaydırıyor |
| **Yanlış ayarın belirtisi** | `bekleyemeyen_pay` 0 ya da 1'e yapışırsa kanal hedefleme yapmıyordur; `verify_m7::kontrol_bekleme_kapisi` bunu ayrı bir kalem olarak kontrol ediyor |
| **Etkileşim** | `ikame_ufku_hafta` (A1), `politika.kisit.asgari_kalan_raf_omru_gun` (kapının kendisi), `lot.raf_omru.*` (dünyanın lot yaşı dağılımı — kapının fiilen bağlayıp bağlamadığını **dünya** belirliyor) |
| **Teşhis** | `uv run python -m scripts.verify_m7 --kosu full` → "erteleme kapisi ayrim yapiyor" satırı |

---

## M7-A3. `senaryo.rejimler[].antisipasyon_talep_carpani` — teklifi YUKARI çeken tek kanal

| Alan | İçerik |
|---|---|
| **Ne yapar** | Hız tahminini ve dolayısıyla teklif adedini ölçekler (SPEC §2.4: güncelleme beklentisi stoklama tetikler, talep 3–5x). Emilim tavanı da aynı hızla hesaplandığı için **veto sınırı birlikte kayar**: `full`da vetolanan satır `baz 1.276 → sok 1.015` |
| **Varsayılan / aralık** | `1.0 / 1.35 / 2.20`; makul aralık `1.0–5.0` |
| **Neden ayrı bir kanal** | Diğer iki kanal (erteleme, fonlama) teklifi **aşağı** çekiyor. Net sonuç kâğıt üzerinde belirsiz ve katmanın varlık sebebi bu belirsizlik. `full`da net sonuç: `sok`ta teklif sayısı **993 → 713** (aşağı kanallar baskın), ama adet başına teklif büyüyor |
| **Artırınca** | Satır başına adet artar, emilim tavanı gevşer, aday havuzu büyür (`full`: 2.724 → 2.985). Erteleme kaleminin **toplam** bedeli de büyür çünkü kalem adetle çarpılıyor |
| **Azaltınca** | Rejim yalnızca fiyat/fonlama tarafından okunur; stoklama davranışı görünmez olur ve SPEC §2.4'ün anticipation dinamiği modelin dışında kalır |
| **Yanlış ayarın belirtisi** | `m7.senaryo.<rejim>.teklif_adedi` fırlarken `kisit.emilim_tavani` vetoları **düşüyorsa** tavan artık bağlamıyordur: eczaneye emebileceğinden fazla mal öneriliyor ve bunun bedeli M6'nın kapalı döngüsünde iade olarak çıkar |
| **Basitleştirme** | Çarpan aday **sıralamasını** değiştirmiyor, yalnızca adedi ve kısıt katmanını. Gerçekte stoklama beklentisi hangi ürünlerin isteneceğini de değiştirir (akut/kronik ayrımı). `reports/m7.md` §7, borç #2 |
| **Etkileşim** | `politika.kisit.azami_kapsama_hafta` (emilim tavanı), `aday.teklif_kapsama_hafta` ve `hiz_telafi_katsayisi` (adet formülünün diğer iki çarpanı) |
| **Teşhis** | `uv run python -m experiments.run --profil full --asama m7 --ad m7_full` → rejim tablosunda `adet` ve `vetolu` sütunları |

---

## M7-A4. `senaryo.rejimler[].fonlama_orani_carpani` — var olan knob'ın rejim altındaki hâli

| Alan | İçerik |
|---|---|
| **Ne yapar** | `politika.skor.yillik_fonlama_orani`nı rejim altında ölçekler. **Yeni aritmetik yok**: `agent/scenario.py::rejim_config` yalnızca o alanı kopyalayıp değiştiriyor, marj formülü M4'ünkinin aynısı |
| **Varsayılan / aralık** | `1.0 / 1.25 / 1.60`; makul aralık `0.5–3.0` |
| **Neden çarpan, mutlak oran değil** | Mutlak yazılsaydı taban fonlama oranı değiştiğinde rejimler onunla **birlikte hareket etmezdi** ve "sert rejim daha pahalı fonlama demektir" ilişkisi config'te iki yerde tutulurdu |
| **Artırınca** | Uzun vade kolları cezalanır, ortalama vade düşer. `full`: `baz 63,1 → yuksek 60,5 gün`. `sok`ta **yükseliyor** (64,5) çünkü orada baskın kanal fonlama değil erteleme ve seçilen kol kümesi tamamen değişiyor |
| **Azaltınca** | Vade boyutu rejime duyarsız hâle gelir; D1'in iki ekseninden biri senaryo katmanında ölü kalır |
| **Yanlış ayarın belirtisi** | `m7.fark.<rejim>.vade_farki ≈ 0` iken `fonlama_orani_carpani > 1` ise kanal bağlamıyordur — muhtemelen aksiyon uzayındaki vade seçenekleri (`aksiyon.vade_gunleri`) birbirine çok yakındır |
| **Etkileşim** | `politika.skor.yillik_fonlama_orani` ve `tedarikci_vade_gun` (M4-A6 — fonlama kaleminin işaretini belirleyen çift), `aksiyon.vade_gunleri` |
| **Teşhis** | `uv run python -m experiments.sweep --knob senaryo.ikame_ufku_hafta --values 4,26 --seeds 3 --profil fast --asama m7` → `m7.senaryo.*.ortalama_vade` sütunları |

---

## M7-A5. `harness.sayi_toleransi_bagil` ve `mutasyon_sapmasi` — **denetçinin kadranı, mekanik olarak kilitli**

| Alan | İçerik |
|---|---|
| **Ne yapar** | `sayi_toleransi_bagil`: metindeki bir sayının defterdeki bir sayıyla eşleşmiş sayılması için izin verilen bağıl fark. `mutasyon_sapmasi`: mutasyon üretecinin bir sayıyı bozarken uyguladığı bağıl kayma |
| **Varsayılan / aralık** | `0.005` ve `0.25`; toleransta makul aralık `0.001–0.05` |
| **İkisi ayrı okunamaz** | Tolerans mutasyon sapmasını yakalayamayacak kadar genişse **bozulmuş sayı da "eşleşti" sayılır** ve sayı denetçisi ölür — üstelik sessizce, çünkü harness "bütün vakalar temiz" der. `core/config.py::_m7_harness_kilidi` `tolerans >= sapma` durumunu yükleme anında **reddediyor**. M5'in "temizlik penceresi boş, rejim ölü" ve M6'nın "örtüşme eşiği tabanın altında, teşhis ölü" kilitleriyle aynı disiplin |
| **Toleransın mutlak tabanı YOK ve bu bilinçli** | `max(|s|, 1)` gibi bir taban kabul olasılıkları ve MF oranları için (`0,03` mertebesinde) **±0,005'lik bir kör bölge** yaratırdı; %25 bozulmuş bir olasılık o bölgede yakalanmazdı. Görüntüleme yuvarlamasını `yuvarlama_basamaklari` zaten karşılıyor |
| **Artırınca** | Yanlış alarm düşer, **kaçan uydurma artar**. Belirti: `m7.harness.mutant_yakalanan < mutant_sayisi` |
| **Azaltınca** | Temiz brifing bile bulgu vermeye başlar (`m7.harness.temiz_bulgu > 0`) çünkü brifing sayıları yuvarlanmış yazılıyor |
| **Etkileşim** | `harness.yuvarlama_basamaklari` (eşleşmenin ikinci yolu), `harness.yoksayilan_tamsayi_ust_siniri` |
| **Teşhis** | `uv run python -m harness.run --kosu full` → `mutant_sayi_bozma` satırı |

---

## M7-A6. `harness.yoksayilan_tamsayi_ust_siniri` — **körleştirme kadranı, varsayılanı sıfır**

| Alan | İçerik |
|---|---|
| **Ne yapar** | Bu değerin altındaki tamsayılar sayı denetiminden muaf tutulur. `0` = muafiyet yok, her sayı hesap veriyor |
| **Varsayılan / aralık** | `0`, makul aralık `0–10` (ve `0`ın dışına çıkmak için **gerekçe gerekir**) |
| **Neden var** | Madde numarası / başlık numarası gibi "sayı olmayan sayılar" için bir kaçış yolu gerekebilir. Bu koşuda gerekmedi: `harness/denetim.py::satiri_temizle` madde imlerini ve kimlikleri metinden zaten ayıklıyor, `0` çalışıyor |
| **Artırınca** | Denetçi körleşir: "3 eczanede", "5 satırda" gibi **uydurulmuş küçük tamsayılar görünmez olur**. Bu knob bir çözüm değil, bir tavizdir; büyütülüyorsa sebebi rapora yazılmalı |
| **Azaltınca** | Zaten taban `0` |
| **Yanlış ayarın belirtisi** | Yanlış alarmı bu knob'ı büyüterek susturmak: `m7.harness.temiz_bulgu` düşer ama `mutant_yakalanan` da düşer. **İkisi birlikte okunmalı** |
| **Etkileşim** | `sayi_toleransi_bagil` (aynı takas, farklı eksen) |
| **Teşhis** | `uv run python -m harness.run --kosu fast --vaka mutant_sayi_bozma` |

---

## M7-B. Aile satırları

### M7-B1. `ajan.*` — brifingin boyutu ve taşıyıcısı

| # | Üye | Ne yapar (mekanizma) | Varsayılan | Yanlış ayarın belirtisi |
|---|---|---|---|---|
| B1a | `istemci` | `sablon` (LLM yok, deterministik referans) / `kayitli` (diskteki konuşma oynatılır) / `anthropic` (canlı API). **Regresyonun determinizmi bu knob'ta duruyor** | `kayitli` | `anthropic`e alınıp CI'a sokulursa koşu ağa ve anahtara bağlanır; `uv run pytest` anthropic paketi kurulu olmayan ortamda düşer |
| B1b | `brifing_teklif_sayisi` | Rejim başına olgu paketine giren teklif satırı. Büyütmek modele daha çok olgu verir ama **uydurma yüzeyini de büyütür**: her sayı bir uydurma fırsatı | `5` | Çok küçük: brifing eksik, saha listenin tamamını göremiyor. Çok büyük: istem şişer, `azami_token` sınırına dayanır ve model metni keser (`kesildi = true` → `bicim_ihlali`) |
| B1c | `brifing_veto_sayisi` | "Neden bu ürün listede yok" gerekçesi yazılan veto satırı sayısı | `3` | `0` yapılırsa kısıt bölümü boşalır; D6'nın görünürlüğü kaybolur ama harness bunu **yakalamaz** (bölüm yine var, içi boş) |
| B1d | `kol_ekonomisi_kol_sayisi` | `kol_ekonomisi` aracının döndürdüğü kol sayısı (kol 0 + en iyi N−1). `reports/m6.md` §6.2'nin tablosunun satır bazlı hâli | `6` | `2`nin altına inemez (yükleme reddeder): tek kollu tablo bir karşılaştırma değildir |
| B1e | `model`, `azami_token`, `sicaklik`, `azami_tur` | Canlı API ayarları. `azami_tur` araç döngüsünün üst sınırı: model bu kadar turda bitirmezse konuşma kesilir ve `bicim_ihlali` çıkar — sonsuz döngü yok | `claude-opus-4-5 / 4096 / 0.0 / 8` | `sicaklik > 0` canlı koşuda tekrar üretilebilirliği tamamen bitirir. **Sıfırda bile garanti yok** — bu yüzden regresyon canlı API'ya değil kayda bakıyor |
| B1f | `kayit_dizini` | Kayıtlı konuşmaların dizini. Kayıt istemle birlikte doğrulanır: istem değiştiyse oynatma **hata verir** | `harness/fixtures` | Bayat kaydı sessizce oynatmak, eski cevabı yeni soruya karşı test etmek olurdu; `agent/client.py` bunu reddediyor |

### M7-B2. `senaryo.politika` ve `senaryo.taban_ad` — ölçümün çerçevesi

| # | Üye | Ne yapar | Varsayılan | Not |
|---|---|---|---|---|
| B2a | `politika` | Senaryo altında koşulan teslim politikası. `experiments/run.py::_gozlemlenebilir_politikalar` anahtarlarıyla birebir aynı olmalı; yükleme tanımsız adı reddeder | `uplift_x` | `teklif_yok` **bilerek dışarıda**: hiç teklif vermeyen politikanın rejim duyarlılığı tanımı gereği sıfırdır, senaryo katmanı ölü olurdu. `propensity` seçilirse rejim farkı **büyür** (seviye optimize eden politika erteleme kalemine daha duyarlı) |
| B2b | `taban_ad` | Farkların ölçüldüğü referans rejim. **Nötr olmak zorunda** (`_m7_senaryo_kilidi`) | `baz` | Nötr olmayan taban, "tabana göre fark" sütunlarını iki müdahalenin farkı hâline getirirdi |
| B2c | `rejimler[].aciklama` | Brifingde aynen kullanılan tek satırlık rejim tanımı | — | Metin knob'ı: modelin yazdığı gerekçe buradan geliyor. Değiştirmek kararı değiştirmez, **anlatımı** değiştirir (D8'in tam olarak izin verdiği şey) |

### M7-B3. `harness.yuvarlama_basamaklari` ve `temiz_vaka_bulgu_tavani`

| # | Üye | Ne yapar | Varsayılan | Not |
|---|---|---|---|---|
| B3a | `yuvarlama_basamaklari` | Defterdeki değerin hangi ondalık basamaklara yuvarlanmasının **meşru** sayılacağı. Brifing "0,35" yazarken defterde "0,3462" durabilir | `[0,1,2,3,4]` | Listeden `2`yi çıkarmak temiz vakayı düşürür (şablon iki basamak yazıyor). Uzatmak toleransı gizlice genişletir: `4`ün ötesi pratikte "her şey eşleşir" demektir |
| B3b | `temiz_vaka_bulgu_tavani` | Temiz vakada kabul edilen azami bulgu | `0` | `0` dışına çıkmak "biraz yanlış alarm normaldir" demektir; o noktadan sonra gerçek bulgu gürültünün içinde kaybolur |

### M7-B4. `senaryo.rejimler[].referans_kur_artisi` ve `fiyat_gecis_katsayisi` — **politikanın inancı, dünyanın gerçeği değil**

| # | Üye | Ne yapar (mekanizma) | Varsayılan | Yanlış ayarın belirtisi |
|---|---|---|---|---|
| B4a | `referans_kur_artisi` | Beklenen güncellemenin büyüklüğü. Erteleme kazancının **ölçek çarpanı**: `birim = dsf × bu × fiyat_gecisi × realizasyon × pay`. Sıfırsa rejim fiyat tarafından nötrdür (taban rejimin nötrlüğü kısmen buradan) | `0.00 / 0.15 / 0.30` | Büyütmek erteleme kalemini doğrusal büyütür ve `negatif_taban_marj_orani`nı 1'e iter; o noktadan sonra `artimsal_marj` metriği **okunamaz** (§M7 giriş uyarısı). Belirti: `talep_baskilayan_orani`nın fırlaması |
| B4b | `fiyat_gecis_katsayisi` | Referans kur artışının DSF'e yansıyan payı. **Bu dünyanın gerçeği değil, POLİTİKANIN İNANCIDIR**: dünya tarafındaki geçiş katsayısı `config/events.yaml::olay.referans_kur.fiyat_gecis_katsayisi`ta ve ikisi bilerek ayrı tutuldu — M5'in `normal_realizasyon_orani` knob'ıyla ("stok ileride satılır mı" inancı) aynı disiplin | `0.00 / 0.60 / 0.85` | İkisini eşitlemek "politika dünyayı doğru biliyor" varsayımını sessizce içeri sokar. Ayrı tutulduğu için **inancın yanlış olmasının bedeli ölçülebilir**: dünya geçişi ile senaryo geçişini ayrı ayrı süpürüp farkı okuyun |

**Neden ikisi ayrı knob:** `referans_kur_artisi × fiyat_gecis_katsayisi` çarpımı tek bir sayıya indirilebilirdi. İndirilmedi çünkü ikisinin **kaynağı farklı**: artış oranı bir regülatör kararına dair beklenti (D4), geçiş katsayısı ise o kararın fiyata ne kadar yansıyacağına dair bir modelleme varsayımı. Tek sayıya indirmek, hangisinin yanlış olduğunu ayırt edilemez kılardı.

