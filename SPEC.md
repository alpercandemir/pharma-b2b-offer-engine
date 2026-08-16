# B2B İlaç Pazaryeri — Teklif/Kampanya Karar Motoru (POC Spesifikasyonu)

> Bu dosya doğrudan Claude Code'a verilecek proje brief'idir. Repo kökünde `SPEC.md` veya `CLAUDE.md` olarak tutulabilir.
> Amaç: üretim sistemi değil, **akışı ve tuning'i uçtan uca görülebilir kılan** bir öğrenme substratı.

---

## 0. Sistemin tek cümlelik tanımı

Sonlu stok altında, her eczane–ürün ikilisi için "şimdi teklif ver / verme" ve "hangi mal fazlası + vade ile" kararını veren; kararını offline olarak ölçebilen; kararın gerekçesini doğal dilde üretebilen bir karar motoru.

## 1. Kritik tasarım kararları (bunlar tartışmaya kapalı, POC'un öğretmek istediği şey bunlar)

| # | Karar | Gerekçe |
|---|---|---|
| D1 | Aksiyon uzayı **yüzde iskonto değil**, `(mal_fazlası_oranı, vade_günü)` çiftidir | TR ilaç piyasasında fiyat regüle; iskonto fiilen MF (10+1, 10+2) ve vade olarak veriliyor |
| D2 | "Aynı ürünü yeni alana önerme" bir **exclusion rule değil**, bir tahmin problemidir | Doğru soru "aldı mı" değil, "eldeki stoğu ne zaman bitecek" |
| D3 | Kur **tahmin hedefi değil, senaryo girdisidir** | 12 ay öteye USD/TRY tahmini çözülmüş problem değil; sistem rejim altında koşullu öneri üretir |
| D4 | Asıl makro sinyali piyasa kuru değil, **referans kur güncelleme beklentisidir** | İlaç fiyatı Bakanlık referans avro kuruna bağlı; stokçuluk davranışını bu tetikliyor |
| D5 | Skorlama ile **tahsis ayrı katmanlardır** | Stok paylaşılan kaynak; ranking tek başına 400 eczaneye aynı kıt SKU'yu önerir |
| D6 | Kısıt katmanının ML skoru üzerinde **veto yetkisi** vardır | Kırmızı/yeşil reçete, SGK listesi, soğuk zincir — bunlar soft penalty değil hard constraint |
| D7 | Her gösterimde seçim **olasılığı (propensity) loglanır** | Loglanmazsa off-policy evaluation imkânsız; her değişiklik canlı A/B'ye mahkûm |
| D8 | LLM karar noktasında **yoktur**; orkestrasyon, açıklama ve senaryo yorumundadır | Latency + maliyet + nondeterminizm getirir, karar kalitesine katkısı yok |
| D9 | Miad temizliği ayrı bir motor değil, **tahsis katmanının bir rejimidir** | Salvage value negatife dönünce aynı LP işaret değiştirir; ayrıca teklif adedi tüketim hızıyla sınırlı (§2.5) |

---

## 2. Sektör parametreleri (mock veri üreteci bunları modellemeli)

### 2.1 Ürün tarafı

**Sınıflandırma**
- `atc_kodu` — 5 seviyeli hiyerarşi (örn. `J01CA04`). Sub-grup ikamesi ve cross-sell'in temeli. Ana gruplar: `J01` antibiyotik, `R05` öksürük/soğuk algınlığı, `R06` antihistaminik, `N02` analjezik, `A02` antiasit, `C09` antihipertansif, `A10` antidiyabetik.
- `etken_madde` (INN) — eşdeğer ilaç grubu. Aynı etken maddede marka değiştirme önerisi ayrı bir aksiyon tipi.
- `urun_tipi` — `RX` (reçeteli) / `OTC` / `TEG` (takviye edici gıda) / `DERMOKOZMETIK` / `MEDIKAL`

**Regülasyon bayrakları** (kısıt katmanını besler)
- `recete_rengi` — `NORMAL` / `KIRMIZI` (narkotik) / `YESIL` (psikotrop) / `MOR` / `TURUNCU`.
  → Kırmızı ve yeşil: promosyon/kampanya **yasak**, kontrollü dağıtım. Hard veto.
- `sgk_geri_odeme` — bool. Geri ödeme listesindeki ürünlerde iskonto serbestisi kısıtlı.
- `titck_tedarik_guclugu` — bool. Piyasada bulunamayan ilaç → kampanya değil, **tahsis** problemi.
- `promosyon_serbest` — türetilmiş: `TEG`, `DERMOKOZMETIK`, `OTC` genelde serbest; kampanya mantığının asıl yaşadığı yer burası.

**Fiyat ve marj**
- `depocu_satis_fiyati` (DSF), `perakende_satis_fiyati` (PSF), `kdv_orani`
- `depo_kar_marji` — kademeli, fiyat bandına bağlı (ucuz ilaçta yüksek %, pahalı ilaçta düşük %)
- `referans_avro_kuru` — regülatör tarafından belirlenir, piyasa kurunun gerisinde, periyodik güncellenir
- **Not:** Güncel marj kademelerini ve referans kur oranını doğrula; POC'ta parametrik tut, sabit gömme.

**Lojistik / stok**
- `soguk_zincir` — bool (aşı, insülin, biyolojikler). Min sipariş, kargo penceresi, maliyet farkı.
- `birim_hacim`, `koli_ici_adet` — MF oranları koli katlarına yuvarlanır
- `its_serilestirilmis` — bool (İlaç Takip Sistemi)
- **Miad → §2.5'e bakınız.** Miad bir ürün niteliği değil, **lot niteliğidir** ve ayrı bir karar katmanı gerektirir.

### 2.2 Alıcı (eczane) tarafı

- `konum` (il/ilçe/semt) + `hastane_yakinligi_km` → **reçete miksini belirleyen en güçlü feature**. Devlet hastanesi yanındaki eczanenin ATC dağılımı ile AVM içindeki eczanenin dağılımı tamamen farklı.
- `semt_sosyoekonomik_index` → dermokozmetik ve TEG payı
- `turizm_bolgesi` bool → yaz aylarında 2–3x ciro sıçraması (Bodrum, Antalya, Çeşme)
- `nobet_rotasyon_gunleri` → nöbet günlerinde akut ilaç talebinde spike
- `aylik_ciro_bandi`, `aylik_recete_adedi`
- `share_of_wallet` — bizden alım / toplam tahmini alım. Multi-homing gerçeği: eczane 3–4 depoyla çalışır. **Gözlemlenmeyen değişken** — sistemin en büyük belirsizliği, simülatörde latent olarak tut.
- `vade_riski_skoru`, `dbs_limiti` (Doğrudan Borçlandırma Sistemi)
- `sgk_recete_orani` vs serbest satış oranı
- `stokculuk_egilimi` — latent kişilik parametresi; kur/fiyat artışı beklentisine tepki katsayısı

### 2.3 Mevsimsellik (simülatöre gömülecek)

| Dönem | Yükselen kategoriler |
|---|---|
| Ekim–Mart | `J01` antibiyotik, `R05` soğuk algınlığı, grip aşısı (Eylül–Kasım penceresi), C ve D vitamini, inhaler |
| Mart–Mayıs | `R06` antihistaminik, göz damlası (polen) |
| Haziran–Ağustos | Güneş koruyucu, ORS, antidiyareik, böcek ısırığı; turizm bölgelerinde genel ciro sıçraması |
| Eylül | Okul açılışı: çocuk vitaminleri, bit tedavisi, aşı takvimi |
| Ramazan | TEG, sindirim, multivitamin |
| Aralık | Yıl sonu SGK/katılım payı değişim beklentisiyle stoklama |

Kronik ilaçlarda (`C09`, `A10`) mevsimsellik **yok**, talep düz ve öngörülebilir — bu kontrast önemli, model ikisini ayırt edebilmeli.

### 2.4 Rejim / olay katmanı (POC'un en özgün parçası)

Bunlar sürekli değişken değil, **kesikli olaylardır** ve anticipation window'ları vardır:

| Olay | Etki | Anticipation |
|---|---|---|
| `REFERANS_KUR_GUNCELLEME` | Fiyat artışı → duyuru öncesi kitlesel stoklama | 2–6 hafta önce beklenti oluşur, talep 3–5x |
| `SGK_LISTE_GUNCELLEME` | Ürün listeye girer/çıkar → talep seviyesi kalıcı kayar | 1–2 hafta |
| `TITCK_GERI_CEKME` | Lot/ürün iptali → anlık sıfırlanma + ikame ürüne kayma | 0 (şok) |
| `TEDARIK_KRIZI` | Global shortage → tahsis moduna geçiş | değişken |
| `EPIDEMI_DALGASI` | Akut kategorilerde 2–4x, 3–8 hafta süreli | 1–2 hafta |

Simülatör bu olayları üretmeli, model **feature olarak görmeli**, politika bunlara koşullu davranmalı.

### 2.5 Miad ekonomisi ve stok temizliği (D9)

> **D9 — Miad, marj maksimizasyonundan farklı bir amaç fonksiyonudur.**
> Normal teklifte birim marjı maksimize edersin. Miadı yaklaşan lotta ürünün değeri sabit bir tarihten sonra **negatife** düşer (imha + iade işlem maliyeti). Amaç maksimizasyon değil, **zarar minimizasyonu**. Normalde irrasyonel bir MF derinliği burada rasyoneldir.

#### Veri modeli değişikliği: lot boyutu

Stok SKU seviyesinde tutulamaz. Miad partinin niteliğidir; aynı SKU'nun farklı miatlı birden fazla lotu bulunur.

```
lot(lot_id, sku_id, miad_tarihi, adet, maliyet, giris_tarihi)
```

- Tahsis **FEFO** (First Expired First Out) çalışır
- Teklif satır bazında lot referansı taşır — hangi lottan verildiği izlenir
- `raf_omru_kalan_gun` = türetilmiş, koşu tarihine göre point-in-time hesaplanır

Bu M1'i doğrudan etkiler; sonradan eklemek acılıdır.

#### M2 kuplajı: temizlik bir iskonto değil, hedefleme problemidir

Kısa miatlı stoğu eczaneye yıkmak zararı transfer eder — satamaz, iade eder, ilişki zarar görür. Teklif adedi tüketim hızıyla sınırlanmak zorunda:

```
max_teklif_adedi = tuketim_hizi × (miada_kalan_gun − eczaci_guvenlik_marji) × guvenlik_katsayisi
```

Sonuç sıfır veya negatifse **o eczane o lot için aday değildir**, iskonto ne kadar derin olursa olsun. M2'nin çıktısı doğrudan bu kısıtı besler.

#### Amaç fonksiyonu: dinamik salvage value

```
lot_birim_degeri(t) = normal_marj                       (t > temizlik_esigi)
                    = azalan_fonksiyon(kalan_gun)       (temizlik penceresi)
                    = −imha_maliyeti                    (miad sonrası)
```

**D5 ile bağlantı:** Tahsis LP'sinde stok kıt kaynaktır, shadow price pozitiftir. Miadı yaklaşan lotta stok yükümlülüktür, **shadow price negatife döner**. Ayrı bir sistem gerekmez — aynı LP, işaret değişimi. Bu yüzden temizlik ayrı bir motor değil, tahsis katmanının bir rejimidir.

#### Alıcı tarafı direnci

Miad çift yönlü etki yaratır: senin motivasyonun, eczacının direnci. Eczacılar kısa miatlı ürün almak istemez; kabul eşiği kategoriye göre değişir (kronik ilaçta yüksek tolerans, mevsimsel üründe düşük). Uplift modelinde `raf_omru_kalan_gun` hem teklif motivasyonu hem de dönüşüm olasılığını **düşüren** bir feature olarak yer alır. Simülatörde eczane persona'sının `miad_toleransi` parametresi olmalı.

#### Kısıt katmanı (D6 ile)

- `promosyon_serbest = false` olan ürünlerde (kırmızı/yeşil reçete) temizlik kampanyası **yapılamaz** — miad baskısı bu vetoyu aşmaz
- SGK geri ödeme kapsamındaki üründe iskonto serbestisi kısıtlıdır; temizlik burada MF yerine **vade** ile yapılır
- Soğuk zincir ürünlerde miad daha kritik, tolerans penceresi dar
- Temizliğin en rahat işlediği yer: `TEG`, `DERMOKOZMETIK`, `OTC` — serbest fiyatlı, promosyon serbest

#### Doğrulanması gereken

- Miad iade mekanizmasının fiili işleyişi (depo → üretici iade koşulları, kredi oranı, zaman penceresi)
- İTS üzerinden miad takibi ve imha prosedürü
- Kısa miatlı ürün satışına dair mevzuat kısıtı olup olmadığı

POC'ta bunlar parametrik tutulur; **doğru olmaları değil, değiştirilebilir olmaları** gerekir.

---

## 3. Mimari

```
sim/          Sentetik dünya (ground truth burada, model asla göremez)
  world.py         Latent tüketim hızları, eczane persona'ları, stok seviyeleri
  events.py        Rejim olayları + anticipation dinamiği
  response.py      Teklife tepki fonksiyonu (uplift ground truth)
  rollout.py       Closed-loop: politika aksiyon alır, dünya tepki verir

data/         Simülatörün ürettiği "gözlemlenebilir" katman
              (sipariş geçmişi, teklif logları, propensity'ler)

features/     Feature builder — leakage guard'lı, point-in-time doğru

models/
  depletion.py     Tükenme zamanı: hazard / survival modeli
  demand.py        Talep tahmini: global LightGBM + Croston baseline
  affinity.py      Candidate generation: CF + market basket
  uplift.py        CATE: T-learner / X-learner

policy/
  candidates.py    Aday üretimi
  scorer.py        Skorlama + beklenen marj
  constraints.py   Hard veto katmanı (regülasyon, soğuk zincir, kredi limiti)
  allocate.py      Kıt stok altında tahsis (LP / shadow price'lı greedy)
  bandit.py        Thompson sampling — exploration ve propensity logging

eval/
  ope.py           IPS / SNIPS / Doubly-Robust
  oracle.py        Sentetik olduğu için gerçek counterfactual — OPE'yi denetler
  report.py        Politika karşılaştırma raporu

agent/
  narrative.py     KAM/saha için teklif brifingi (LLM)
  scenario.py      Makro rejim yorumu (LLM, senaryo altında koşullu)
  tools.py         LLM'in çağırabileceği fonksiyonlar

harness/
  cases.yaml       Eval senaryoları
  run.py           Regresyon koşusu
```

---

## 4. Milestone'lar

Her milestone kendi başına çalışır ve gösterilebilir çıktı üretir. Sırayı bozma.

### M1 — Simülatör ve ground truth
Latent tüketim hızları, eczane persona'ları, mevsimsellik, intermittent talep, rejim olayları, **lot boyutu ve miad dağılımı**.
**Çıkış kriteri:** 200 eczane × 300 SKU × 104 hafta sentetik geçmiş; talep dağılımı gerçekçi (çoğu hücre sıfır, uzun kuyruk); mevsimsellik ve olay etkisi grafiklerde görünür; stok lot seviyesinde, FEFO tüketiliyor, eczane persona'sında `miad_toleransi` var.

### M2 — Tükenme modeli (D2)
Sipariş miktarından tüketim hızı çıkarımı, eldeki stok tahmini, tükenme zamanı hazard modeli.
**Çıkış kriteri:** Tahmin edilen tükenme günü ile simülatörün gerçek stok sıfırlanma günü karşılaştırması; MAE ve kalibrasyon eğrisi.
**Öğrenilen:** "Son 30 günde aldı mı" kuralı ile hazard modeli arasındaki fark ölçülebilir hale gelir.

### M3 — Aday üretimi + kısıt katmanı
CF/market basket ile aday havuzu, üzerine hard veto.
**Çıkış kriteri:** Kırmızı reçeteli ürünün hiçbir koşulda öneri listesinde çıkmadığı test; soğuk zincir min sipariş kuralı.

### M4 — Uplift ve aksiyon seçimi (D1)
`(MF oranı, vade)` aksiyon uzayında CATE modeli. Propensity vs uplift karşılaştırması.
**Çıkış kriteri:** Propensity-based politika ile uplift-based politika arasındaki **marj farkı** ölçülür. Propensity'nin nasıl marj yaktığı sayıyla gösterilir.

### M5 — Tahsis ve miad rejimi (D5 + D9)
Sonlu stok altında LP. Shadow price yorumu. Miadı yaklaşan lotta negatif shadow price rejimi.
**Çıkış kriteri:** (a) Kıt SKU senaryosunda ranking-only politika ile LP politikası karşılaştırması; stockout ve karşılanmayan talep sayıları. (b) Kısa miatlı lot senaryosunda üç politika: temizlik yok / kör iskonto / M2 kuplajlı hedefli temizlik. Metrik: imha edilen adet, iade edilen adet, net marj, eczane memnuniyeti proxy'si.
**Öğrenilen:** Kör iskontonun imhayı azaltırken iadeyi ve marj kaybını nasıl artırdığı sayıyla görülür.

### M6 — Off-policy evaluation ve tuning döngüsü
IPS/SNIPS/DR ile offline skor, ardından closed-loop rollout ile gerçek sonuç.
**Çıkış kriteri:** "Offline tahmin +%12 dedi, gerçek -%3 çıktı, neden?" sorusunun cevaplanabildiği bir rapor. Variance, overlap ihlali, extrapolation.
**Bu milestone POC'un asıl amacıdır.** Diğerleri buraya kurulum.

### M7 — LLM katmanı (D8)
Senaryo yorumu (baz/yüksek/şok kur rejimi altında politika ne öneriyor) + KAM brifingi + açıklanabilirlik.
**Çıkış kriteri:** Eval harness'ta LLM çıktısı deterministik olarak test ediliyor — hallüsinasyon, kısıt ihlali iddiası, sayı uydurma yakalanıyor.

---

## 5. Tuning için özellikle görülebilir kılınacak knob'lar

Öğrenmenin olduğu yer burası. Hepsi config'ten değiştirilebilir, etkisi rapora yansımalı.

**Simülatör tarafı**
- `share_of_wallet` gözlemlenebilirliği — açık/kapalı. Kapalıyken modelin ne kadar kör olduğunu gör.
- Uplift heterojenliği — segmentler ne kadar farklı tepki veriyor
- Olay sıklığı ve anticipation penceresi
- Talep seyrekliği (zero-inflation oranı)

**Politika tarafı**
- Exploration oranı (ε veya Thompson posterior genişliği)
- Tükenme eşiği — "kaç gün kala teklif ver"
- Marj tabanı, vade maliyeti (TRY fonlama oranı)
- Frekans tavanı — eczane başına haftalık maksimum teklif
- Tahsis hedefi: kısa vadeli marj vs share-of-wallet büyütme

**Miad tarafı (§2.5)**
- `clearance.trigger_days` — kaç gün kala temizlik rejimine geç (60 / 90 / 120)
- `clearance.salvage_curve` — değer düşüş eğrisi: lineer / eksponansiyel / basamaklı
- `clearance.safety_factor` — tüketim hızı kuplajındaki güvenlik payı (0.5–1.0)
- `clearance.pharmacist_margin_days` — eczacının kabul edeceği minimum kalan raf ömrü
- `disposal_cost_per_unit` — imha + iade işlem maliyeti
- **En öğretici sweep:** `trigger_days` çok erken → gereksiz marj bırakılıyor; çok geç → imha patlıyor. Arada bir optimum var ve `disposal_cost_per_unit`'e duyarlı. Bu ikisini birlikte süpür.

**Eval tarafı**
- OPE estimator seçimi ve clipping eşiği
- Değerlendirme ufku — 4 hafta vs 52 hafta. **Kısa ufukta agresif iskonto kazanır, uzun ufukta kaybeder.** Bu kontrastı görmek POC'un en öğretici anı.

---

## 5b. Öğretme sözleşmesi (zorunlu — atlanamaz)

Bu POC'un çıktısı sadece kod değil, **kodu tune edebilir hale gelmiş bir insan**. Bu yüzden her milestone iki artifact üretir: kod ve öğretim materyali.

### 5b.1 `TUNING.md` — her milestone sonunda güncellenir

O milestone'da ortaya çıkan her knob için tek bir tablo satırı:

| Alan | İçerik |
|---|---|
| Knob | Config yolu (`policy.depletion_threshold_days`) |
| Ne yapar | Bir cümle, mekanizma düzeyinde |
| Varsayılan ve makul aralık | `7`, aralık `3–21` |
| Artırınca ne olur | Beklenen yön + hangi metrikte |
| Azaltınca ne olur | Beklenen yön + hangi metrikte |
| Yanlış ayarın belirtisi | Gözlemlenebilir semptom ("teklif sayısı patlıyor ama dönüşüm düz") |
| Etkileşim | Hangi diğer knob'la birlikte hareket eder |
| Teşhis komutu | Çalıştırılabilir tek satır |

**Kural:** Bir knob config'e girdiyse `TUNING.md`'de satırı olmak zorunda. Satırı yoksa knob değil, sihirli sayıdır ve koddan çıkarılır.

### 5b.2 `experiments/` — config sweep koşucusu

M2'den itibaren var olmalı, sonradan eklenmez.

```
experiments/
  run.py           tek config → tek koşu → metrik seti
  sweep.py         knob listesi + değer aralığı → paralel koşu → karşılaştırma tablosu
  compare.py       iki koşuyu yan yana koy, farkı ve istatistiksel anlamlılığı ver
  runs/            seed'li, versiyonlu çıktılar
```

Kullanım şu kadar basit olmalı:

```bash
python -m experiments.sweep --knob policy.depletion_threshold_days --values 3,7,14,21 --seeds 5
```

Çıktı: her değer için metrik tablosu + hangi yönde hareket ettiğini gösteren tek grafik. Bu komut çalışmıyorsa milestone bitmemiştir.

### 5b.3 Her milestone raporunda zorunlu bölüm

Claude Code milestone bitirdiğinde şunları da yazacak:

1. **Bu milestone'da hangi knob'lar ortaya çıktı** ve neden knob olarak bırakıldı (sabit gömülmedi)
2. **Hangi üçü en çok fark yaratıyor** — sweep sonucuyla, iddia olarak değil
3. **Beklentiyle gerçeğin ayrıştığı yer** — "şunu artırınca X bekliyordum, Y oldu, sebebi Z"
4. **Bir sonraki milestone'a taşınan borç** — hangi basitleştirme yapıldı, ne zaman patlar

### 5b.4 Kalibrasyon egzersizleri

Her milestone en az bir tane "önce tahmin et, sonra çalıştır" egzersizi bırakacak:

> `uplift.min_effect_threshold` değerini 0.02'den 0.10'a çıkarıyoruz.
> Teklif sayısı ne olur? Toplam marj ne olur? Marj/teklif oranı ne olur?
> Tahminini yaz, sonra `experiments/sweep.py` ile koştur.

Bu format kritik: knob'ın etkisini okumakla, yanlış tahmin edip nedenini bulmak arasında öğrenme farkı büyük. Sistemi tune edebilir hale gelmek buradan geçiyor.

---

## 6. Teknik kısıtlar

- Python 3.11+, `uv` veya `poetry`
- LightGBM, scikit-learn, `lifelines` (survival), `PuLP` veya `scipy.optimize` (LP), `polars` veya `pandas`
- LLM katmanı: Anthropic API, tool use ile
- Notebook yok — her şey çalıştırılabilir script + config. Notebook öğrenmeyi tembelleştirir, reproducibility'yi öldürür.
- Her model artifact'i versiyonlu, her rollout seed'li ve tekrar üretilebilir
- Config: tek bir `config/` altında YAML; kod içinde sihirli sayı yok

---

## 7. Çalıştırma

Bu dosya referans spesifikasyondur, çalıştırılabilir bir şey değildir.

- **Kalıcı kurallar:** `CLAUDE.md` — Claude Code her oturumda otomatik okur
- **Milestone komutları:** `PROMPTS.md` — kopyala-yapıştır
- **İnsan tarafı çalışma döngüsü ve denetim listesi:** `README.md`

---

## 8. Doğrulanması gereken sektör bilgileri

POC'u parametrik tut ama gerçek değerleri şuralardan teyit et:
- TİTCK — fiyat kararnamesi, referans avro kuru, tedarik güçlüğü listesi
- SGK — Ek-4/A geri ödeme listesi
- Depo/eczacı kâr marjı kademeleri (fiyat bandına göre değişen yapı)
- Reçete renk sınıflandırması ve promosyon kısıtları
- İTS serileştirme kuralları

Bu POC'ta bu değerlerin **doğru olması gerekmiyor** — parametrik ve değiştirilebilir olması gerekiyor.
