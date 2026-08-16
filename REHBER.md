# Rehber — Bu proje ne, nasıl çalıştırılır, bir teklif nasıl görülür

Bu dosya `README.md`'nin yerine geçmez. `README.md` projenin İngilizce tanıtımı — ne çözdüğü, hangi komutun ne ürettiği, nelerin eksik kaldığı; `WORKING_GUIDE.md` ise **projeyi inşa eden insan** için yazılmış bir çalışma disiplini belgesi. Bu dosya ise **projeye ilk kez bakan insan** için: sistem ne yapıyor, hangi parça neden var, hangi komut ne üretiyor. Verinin kendisi — hangi tablo hangi kolonu taşıyor, gözlemlenebilir/ground truth sınırı nerede — `DATA.md`'de.

Sıfırdan okunacak şekilde yazıldı. Terimler ilk geçtikleri yerde açıklanıyor.

---

## 1. Tek cümlede ne var burada

> Bir ilaç deposunun elindeki **sonlu stoku**, hangi eczaneye, hangi hafta, hangi **mal fazlası + vade** şartıyla teklif edeceğine karar veren; kararının değerini canlıya çıkmadan ölçebilen; kararın gerekçesini doğal dilde anlatabilen bir karar motoru — ve bunun sentetik bir dünya üzerinde uçtan uca çalışan hâli.

İş bağlamı: Türkiye'de ilaç fiyatı regüle. Depo eczaneye "%15 indirim" veremez. Verebildiği iki şey var:

- **Mal fazlası (MF):** "10 al, 1 bedava" (`mf = 0.10`), "20+1" (`0.05`), "50+1" (`0.02`)
- **Vade:** ödemeyi kaç gün sonra istediği (60 gün, 90 gün...)

Bu yüzden sistemin **aksiyon uzayı** `(mal_fazlası_oranı, vade_günü)` çiftidir. Yüzde iskonto knob'u yoktur ve eklenmeyecektir (`SPEC.md` §1, karar D1).

Bir aksiyon adı koda şöyle geçiyor: `mf0.05_v60` = "20+1, 60 gün vade".

---

## 2. Simülasyon mu?

**Evet, iki ayrı anlamda — ve ikisini karıştırmamak bu projeyi anlamanın anahtarı.**

### 2.1 Veri üreteci olarak simülasyon

Gerçek eczane/sipariş verisi yok. `sim/` altındaki kod 104 haftalık sentetik bir geçmiş üretiyor: eczaneler, ürünler, hasta tüketimi, siparişler, sevkiyatlar, stok lotları, miatlar, imhalar, iadeler, regülasyon olayları.

Kritik nokta: bu dünya **iki katmana** ayrılmış olarak diske yazılıyor.

```
data/fast/
  observable/      <- modelin GÖREBİLDİĞİ her şey (siparişler, sevkiyat, stok, fiyat, olaylar)
  ground_truth/    <- dünyanın GERÇEĞİ (latent tüketim hızı, share_of_wallet, gerçek tükenme anları)
```

`models/`, `features/`, `policy/` altındaki hiçbir kod `ground_truth/` okumaz. Okusaydı proje anlamını yitirirdi. Doğrulama scriptleri bunu kaynak taramasıyla mekanik olarak sınıyor (örn. `scripts/verify_m4.py` tepki fonksiyonunun politika tarafından görülmediğini kontrol ediyor).

**Ground truth'un olması bir lüks:** gerçek hayatta "bu teklifi vermeseydim ne olurdu" sorusunun cevabı yoktur. Burada var. Bu yüzden offline değerlendirmenin (bkz. §8, M6) **ne zaman yalan söylediğini** ölçebiliyorsun. Projenin asıl amacı bu.

### 2.2 Kapalı döngü (closed-loop) olarak simülasyon

İkinci anlam `sim/rollout.py`'de: politika bir hafta karar veriyor → dünya o kararı yiyor (mal eczaneye gidiyor) → ertesi hafta eczane **daha az sipariş veriyor** (kanibalizm), fazlası **iade oluyor**, iade **share_of_wallet'ı düşürüyor** → politika bir sonraki haftayı bu bozulmuş dünyada karşılıyor.

Tek adımlık hiçbir tahminci bu gecikmeli bedeli göremez. "Kısa ufukta kazanan agresif iskonto, uzun ufukta kaybeder" cümlesi tam olarak burada ölçülüyor.

### 2.3 Simülatörün zor olması bir tasarım şartı

`CLAUDE.md` §7: sentetik veri öğrenmesi kolay olmamalı. Dünyaya bilerek konmuş zorluklar (`sim/world.py` başlığında listeli):

- `share_of_wallet` (eczanenin bizden alım payı) **latent** ve zamanla kayıyor — model asla göremiyor
- Talep intermittent: (eczane, SKU) hücrelerinin çoğu, çoğu hafta sıfır
- Koli yuvarlaması tüketim → sipariş eşlemesini kırıyor
- Eczane kendi (sansürlü) EWMA'sına göre sipariş veriyor, gerçek hızına göre değil
- Bizim stoksuzluğumuz siparişi **görünmez** kılıyor (rakibe gidiyor, kayda geçmiyor)
- Kur beklentisinde tüketim değişmiyor, sadece sipariş öne çekiliyor — sipariş serisinden tüketim çıkaran model yanılıyor

Bir modelin metriği şüpheli derecede iyiyse önce leakage, sonra "simülatör fazla kolay mı" kontrol edilir.

---

## 3. Zihinsel model: dört katman

```
        ┌──────────────────────────────────────────────────────────────┐
        │  1. DÜNYA          sim/        gerçek ama görünmez           │
        │     tüketim, latent SOW, gerçek tükenme anları, tepki fonk.  │
        └───────────────────────────┬──────────────────────────────────┘
                                    │ sadece "bize gelen sipariş" ve
                                    │ "bizim sevkiyatımız" sızar
        ┌───────────────────────────▼──────────────────────────────────┐
        │  2. GÖZLEMLENEBİLİR VERİ   data/<koşu>/observable/           │
        │     + features/  (point-in-time, leakage guard'lı)           │
        └───────────────────────────┬──────────────────────────────────┘
                                    │
        ┌───────────────────────────▼──────────────────────────────────┐
        │  3. KARAR ZİNCİRİ                                            │
        │     models/depletion  → ne zaman tükenecek (hazard modeli)   │
        │     policy/candidates → aday havuzu (CF + sepet + popülerlik)│
        │     policy/constraints→ HARD VETO (regülasyon, kredi, miad)  │
        │     models/uplift     → hangi kol ne kadar EK fayda üretir   │
        │     policy/scorer     → kol seçimi, frekans tavanı           │
        │     policy/allocate   → kıt stok altında LP tahsisi          │
        │     policy/bandit     → keşif + propensity loglama           │
        └───────────────────────────┬──────────────────────────────────┘
                                    │
        ┌───────────────────────────▼──────────────────────────────────┐
        │  4. ÖLÇÜM ve ANLATIM                                         │
        │     eval/ope     IPS / SNIPS / DR — offline tahmin           │
        │     eval/oracle  gerçek karşı-olgusal — OPE'yi DENETLER      │
        │     sim/rollout  kapalı döngü — gerçek sonuç                 │
        │     agent/       LLM: senaryo yorumu + KAM brifingi          │
        │     harness/     LLM çıktısının deterministik regresyonu     │
        └──────────────────────────────────────────────────────────────┘
```

**LLM nerede yok:** karar noktasında. `agent/tools.py` `policy/`, `models/`, `sim/`, `eval/` modüllerinden hiçbirini import **edemez**; ederse `scripts/verify_m7.py` koşuyu düşürür. LLM sadece hazır tabloyu okur ve anlatır (karar D8).

---

## 4. Repo haritası

| Dizin | Ne yapar |
|---|---|
| `config/` | **Bütün sayılar burada.** 16 YAML dosyası, her knob'un başında neden o değerde olduğunu anlatan yorum |
| `config/profiles/` | Ölçek profilleri: `fast` (60 eczane × 100 SKU × 104 hafta), `full` (200 × 300 × 104) |
| `core/` | Config yükleyici (pydantic ile doğrulanıyor), seed bankası, parquet I/O |
| `sim/` | Sentetik dünya: `world` (haftalık döngü), `events` (rejim olayları), `response` (teklife tepki = uplift ground truth), `lots` (miad), `rollout` (kapalı döngü) |
| `features/` | Point-in-time feature builder: `hiz` (tüketim hızı), `stok` (eldeki stok tahmini), `teklif`, `panel` |
| `models/` | `depletion` (tükenme/hazard), `uplift` (T-learner / X-learner CATE) |
| `policy/` | `candidates`, `constraints`, `scorer`, `allocate` (LP), `bandit` |
| `eval/` | `ope` (IPS/SNIPS/DR), `oracle` (gerçek karşı-olgusal), `uplift`, `allocation`, `metrics`, `report` |
| `agent/` | LLM katmanı: `scenario` (kur rejimi yorumu), `narrative` (KAM brifingi), `tools` (salt-okur araçlar), `client` |
| `harness/` | LLM regresyon koşusu: `cases.yaml` (12 vaka), `mutasyon.py` (brifingi bozar), `denetim.py` (bozulmayı yakalar) |
| `experiments/` | `run.py` (tek koşu), `sweep.py` (knob taraması), `compare.py` (iki koşu farkı) |
| `scripts/` | `generate_world.py` + `verify_m1..m7.py` (her milestone'un çıkış kriteri) |
| `tests/` | 213 test |
| `data/` | Üretilmiş dünyalar: `fast`, `full` |
| `reports/` | Milestone raporları `m1.md … m7.md` + `figures/` |
| `TUNING.md` | **216 KB.** Her knob için mekanizma, aralık, artırınca/azalınca ne olur, yanlış ayarın belirtisi, teşhis komutu |

---

## 5. Tech stack — ne, neden, nasıl

### 5.1 `uv` — paket ve ortam yöneticisi

**Ne yapar:** Python'ın `pip` + `venv` + `pip-tools` + `pyenv` işlerini tek bir hızlı araçta toplar (Rust ile yazılmış).

**Neden gerekli:** Bir Python projesi çalıştırmak için üç şeye ihtiyacın var: (1) doğru Python sürümü, (2) o projeye özel izole bir kütüphane klasörü — yoksa iki proje aynı kütüphanenin farklı sürümünü isteyince çakışırlar, (3) kimin hangi sürümü kurduğunun kaydı. `uv` üçünü de yapar.

**Bu repoda nasıl çalışıyor:**

| Dosya | Rolü |
|---|---|
| `pyproject.toml` | *Ne istiyoruz:* `numpy>=1.26`, `polars>=1.0`, ... (gevşek sınırlar) |
| `uv.lock` | *Fiilen ne kuruldu:* her paketin tam sürümü + hash'i (305 KB). Tekrar üretilebilirliğin garantisi |
| `.venv/` | İzole kütüphane klasörü. `uv` otomatik oluşturur, elle dokunmazsın |

Üç komut yeterli:

```bash
uv sync                          # .venv'i uv.lock'a birebir eşitle (kur/sil/güncelle)
uv run python -m scripts.verify_m1 --kosu fast    # .venv içinde çalıştır
uv run pytest                    # aynı şey, testler için
```

`uv run` her seferinde ortamın lock ile senkron olduğunu kontrol edip öyle çalıştırır. Bu yüzden `source .venv/bin/activate` yapmana **gerek yok** — bu repoda hiçbir komut ondan başlamıyor.

`uv sync --extra llm` ise opsiyonel `anthropic` paketini de kurar; sadece canlı API koşusu için gerekli (§7.4).

### 5.2 Diğerleri ve seçilme gerekçeleri

| Kütüphane | Nerede | Neden bu |
|---|---|---|
| **polars** | her yerde tablo işi | pandas'tan hızlı, lazy/kolon tabanlı; parquet ile doğal çalışıyor. 100 bin+ satırlık panellerde sweep'i taşıyabilecek tek makul seçenek |
| **numpy** | matris işleri | Aksiyon uzayı `[n_satır × n_kol]` matrisleri (adet, marj, izinli, bedava) tamamen numpy. Aday üretimi de (SKU×SKU 300×300) numpy — seyrek matris kütüphanesi bu ölçekte gereksiz |
| **pyarrow** | disk formatı | Dünya `.parquet` olarak yazılıyor: kolonlu, sıkıştırılmış, tipli |
| **pydantic** | `core/config.py` | YAML'dan gelen config **doğrulanarak** yükleniyor. Yanlış tipte knob, eksik alan, birbiriyle çelişen iki knob → koşu daha başlamadan düşer |
| **scikit-learn** | `models/` | `HistGradientBoosting` kullanılıyor. **LightGBM denenmiş ve geri alınmış:** macOS'ta `brew install libomp` gerektiriyor, yoksa import anında patlıyor. sklearn'ünki aynı algoritma ailesi (histogram binning, kategorik destek), sistem bağımlılığı yok. Gerekçe `pyproject.toml`'da yazılı |
| **scipy** | `policy/allocate.py` | `scipy.optimize.linprog(method="highs")` tahsis LP'sini çözüyor. PuLP yerine seçilme sebebi: HiGHS **dual çözümü** doğrudan veriyor ve gölge fiyat M5'in merkezinde |
| **matplotlib** | `verify_*.py` | Grafikler `reports/figures/mN/` altına PNG olarak yazılıyor (`Agg` backend, ekran gerekmiyor) |
| **pytest** | `tests/` | 213 test; leakage guard'lar, değişmezler, determinizm |
| **anthropic** | `agent/client.py` | **Opsiyonel.** Varsayılan koşu kayıtlı konuşmaları oynatır; ağ ve API anahtarı istemez (§7.4) |

**Notebook yok, olmayacak.** Her şey `python -m ...` ile çalışan script + YAML config. Gerekçe `SPEC.md` §6: notebook reproducibility'yi öldürür.

---

## 6. Kurulum ve ilk koşu

```bash
cd /Users/alpercandemir/Development/sandbox/pharma-b2b-offer-engine
uv sync
```

Dünyalar (`data/fast`, `data/full`) zaten üretilmiş durumda. Sıfırdan üretmek istersen:

```bash
uv run python -m scripts.generate_world --profil fast --kosu fast
uv run python -m scripts.generate_world --profil full --kosu full
```

Sistemin ayakta olduğunu 20 saniyede doğrulayan komut:

```bash
uv run python -m harness.run --kosu fast
```

Beklenen çıktı: `SONUC: 12/12 vaka gecti`.

Tam test paketi (bu makinede **213 passed, ~7 dakika**):

```bash
uv run pytest -q
```

---

## 7. **Bir eczaneye şu hafta ne teklif edilecek — nasıl görürüm?**

Önce iki kavram:

- **origin (karar haftası):** Dünyanın başlangıcından itibaren hafta numarası. Sistem "şimdi" derken bir origin'i kastediyor. 104 haftalık dünyada ölçüm origin'leri **91, 95, 99** (config: `politika.aday.degerlendirme.ufuk_hafta: 4`, `origin_sayisi: 3` — sondan geriye, pencereler örtüşmesin diye 4'er hafta arayla).
- **politika:** Aynı aday listesine farklı amaç fonksiyonları uygulayan karar kuralları. Başlıcaları:
  - `uplift_x` — **teslim politikası.** Kolu, teklif vermemeye kıyasla **ek** marja göre seçer
  - `propensity` — sadece "kabul olasılığı × marj" maksimize eder (zaten alacak eczaneye MF yakar)
  - `m3_sabit_kampanya` — "herkese aynı kampanya", saha refleksi
  - `agresif` / `agresif_vade` — en derin MF + en uzun vade
  - `teklif_yok` — taban çizgisi

### 7.1 En hızlı yol: ham teklif tablosu

```bash
uv run python -m scripts.teklif_bak --kosu fast --eczane ECZ0003
```

```
kosu=fast eczane=ECZ0003 hafta=99 politika=uplift_x | olcum origin'leri=[91, 95, 99]
aday satiri (veto sonrasi, bu eczane)=17 -> teklife donen=5
┌─────────┬───────────┬────────────┬──────┬────────┬─────────────────────┬──────────┐
│ sku     ┆ lot_id    ┆ kol        ┆ adet ┆ bedava ┆ kabul_sartiyla_marj ┆ skor     │
╞═════════╪═══════════╪════════════╪══════╪════════╪═════════════════════╪══════════╡
│ SKU0019 ┆ LOT000553 ┆ mf0.00_v60 ┆ 18.0 ┆ 0.0    ┆ 103.94              ┆ 0.9508   │
│ SKU0076 ┆ LOT000840 ┆ mf0.00_v60 ┆ 27.0 ┆ 0.0    ┆ 279.20              ┆ 0.9328   │
│ SKU0000 ┆ LOT000813 ┆ mf0.05_v60 ┆ 28.0 ┆ 1.0    ┆ 140.00              ┆ 0.7974   │
│ ...     ┆           ┆            ┆      ┆        ┆                     ┆          │
└─────────┴───────────┴────────────┴──────┴────────┴─────────────────────┴──────────┘
```

Okunuşu: *"99. haftada ECZ0003'e beş satır teklif çıkıyor. Üçüncü satır: SKU0000, LOT000813 partisinden 28 adet, 20+1 (bir adet bedava), 60 gün vade. Kabul edilirse beklenen brüt marj 140 TL."*

Teklif satırı **lot referansı taşır** — hangi partiden verildiği izlenir, çünkü miad partinin niteliğidir (karar D9).

Diğer kullanımlar:

```bash
# aynı eczaneye propensity politikası ne verirdi (karşılaştırma)
uv run python -m scripts.teklif_bak --kosu fast --eczane ECZ0003 --politika propensity

# daha eski bir karar haftası: origin listesini geriye aç
uv run python -m scripts.teklif_bak --kosu fast --origin-sayisi 6 --hafta 83

# büyük dünya
uv run python -m scripts.teklif_bak --kosu full --eczane ECZ0042
```

> `scripts/teklif_bak.py` bu rehberle birlikte eklendi; bir milestone artifact'i değil, okuma aracı. Karar üretmez, diske bir şey yazmaz — `experiments/run.py::m4_boru_hatti`'nin zaten ürettiği seçimi bir eczane için tabloya döker. Milestone disiplinini kirli tutmak istemezsen silebilirsin.

### 7.2 İnsan diline çevrilmiş hâli: KAM brifingi

```bash
uv run python -m harness.run --kosu fast --vaka temiz_sablon --metin
```

Bu, aynı kararı üç kur rejimi altında yan yana koyan bir brifing basar:

```
# Teklif brifingi | ECZ0000 | hafta 99 | politika uplift_x

## Eczane
Konya / Konya-ILCE1, XL ciro bandi, aylik 2629 recete. Hastaneye 2,03 km.
Vade riski 0,45, DBS limiti 2.113.390 TL, acik bakiye 97.366 TL. Haftalik teklif tavani 5 satir.

## Rejim: baz
- SKU0000 (TEG), lot LOT000813: MF yok, 60 gun vade, 135 adet + 0 bedava.
  Kabul olasiligi 0,41 (teklifsiz 0,30), artimsal beklenen marj 88,88 TL.
  Lotun kalan raf omru 472 gun; guncellemeyi bekleyebilir.
  ...
En iyi satirin (SKU0000) kol ekonomisi:
- teklif_yok  : kabul 0,30, kabul sartiyla marj 837,46 TL, artimsal 0 TL
- mf0.00_v60  : kabul 0,41, kabul sartiyla marj 837,46 TL, artimsal 88,88 TL (secilen)
- mf0.02_v60  : kabul 0,30, kabul sartiyla marj 772,46 TL, artimsal -19,80 TL
- mf0.10_v60  : kabul 0,45, kabul sartiyla marj 414,97 TL, artimsal -67,01 TL

## Rejim: yuksek   (referans kur güncellemesi 8 hafta içinde)
## Rejim: sok      (2 hafta içinde, stoklama dalgası başlamış)

## Kisit notlari
- [baz] SKU0099: raf_omru -- Eldeki lotlarin kalan raf omru asgari esigin altinda.
- [sok] SKU0099: depo_stogu -- Depoda yeterli adet yok.
```

Burada dört şeyi birden görüyorsun ve karar mantığının tamamı bu dört satırda:

1. **Kol ekonomisi:** MF derinleştikçe kabul olasılığı artıyor ama *artımsal* marj negatife düşüyor. `mf0.10` kabulü 0,30'dan 0,45'e çıkarıyor ama 67 TL yakıyor — çünkü o eczane zaten %30 ihtimalle alacaktı ve MF'i ona bedavaya vermiş oluyorsun. **Propensity ile uplift'in farkı tam olarak bu.**
2. **Rejim koşullu okuma:** Kur tahmin edilmiyor, senaryolaştırılıyor (karar D3). Şok rejiminde erteleme kazancı adet başına 21,77 TL'ye çıkıyor — bugün satmanın fırsat maliyeti. Bu yüzden şok rejiminde ayakta kalan teklifler **kısa miatlı lotlar** oluyor (beklemek onlar için bedava değil).
3. **Kısıt notları:** Neden bazı ürünler listede yok. Bunlar soft penalty değil hard veto (karar D6).
4. **Rejim başlığının zorunluluğu:** Brifing tek bir rejimi öne çıkaramaz; çıkarırsa harness `bicim_ihlali` bulgusu üretir.

Başka bir eczane için: `harness/cases.yaml` içinde `eczane: otomatik` yerine `eczane: ECZ0007` yaz.

### 7.3 Portföyün tamamı ve politika karşılaştırması

```bash
uv run python -m experiments.run --profil fast --asama m4 --ad deneme
```

Bu, 8 politikayı aynı aday kümesi üzerinde koşturup toplam/artımsal marjı, teklif sayısını, yakılan marjı, oracle açığını basar. Sonuçlar `experiments/runs/deneme/metrikler.json`'a düz bir sözlük olarak yazılır.

### 7.4 LLM gerçekten çağrılıyor mu?

Varsayılan olarak **hayır**. `config/agent.yaml` → `ajan.istemci: kayitli`: harness `harness/fixtures/` altındaki kayıtlı konuşmayı oynatır. Sebep determinizm — `CLAUDE.md` "iki kez çalıştırınca aynı sonuç" diyor, bir dil modeli bunu garanti edemez. Gerçek API ile koşmak istersen:

```bash
uv sync --extra llm
export ANTHROPIC_API_KEY=...
uv run python -m harness.run --kosu fast --canli --kaydet yeni_konusma.json
```

Regresyon yine o kaydın üzerinde döner.

---

## 8. Milestone haritası — hangi soru, hangi komut

Sistem yedi aşamada inşa edilmiş; her aşamanın kendi çıkış kriteri, doğrulama scripti ve raporu var. Hepsi tamamlanmış durumda.

| # | Ne kuruldu | Öğrettiği şey | Doğrulama |
|---|---|---|---|
| M1 | Simülatör, lot/miad, rejim olayları | Sentetik verinin **zorluğunu ayarlamak** | `uv run python -m scripts.verify_m1 --kosu full` |
| M2 | Tükenme (hazard) modeli | "Son 30 günde aldı mı" kuralı ile hazard modelinin farkı — ölçülebilir hâlde (karar D2) | `verify_m2` |
| M3 | Aday üretimi + kısıt katmanı | Hard constraint'in ML skorunu nasıl **veto ettiği**; recall'un veto sonrası ne kadar düştüğü | `verify_m3` |
| M4 | Uplift / CATE, aksiyon seçimi | Propensity'nin marjı nasıl yediği — TL cinsinden | `verify_m4` |
| M5 | Tahsis LP + miad rejimi | Gölge fiyat; miadı yaklaşan lotta işaretin **negatife dönmesi** (karar D5+D9) | `verify_m5` |
| **M6** | **Off-policy evaluation + kapalı döngü** | **Offline tahminin ne zaman yalan söylediği** | `verify_m6` |
| M7 | LLM senaryo + brifing + eval harness | LLM çıktısını **deterministik** test etmek (karar D8) | `verify_m7` |

`README.md`'nin dediği gibi: **M6 projenin amacıdır, diğerleri oraya kurulum.**

M6 raporunun (`reports/m6.md`) özet bulgusu, sistemin ne öğrettiğini tek satırda anlatıyor: SPEC'in beklediği "kısa ufukta kazanır, uzun ufukta kaybeder" karşıtlığı agresif iskontoda **çıkmadı**, `lp` tahsis politikasında çıktı — @4 hafta **+%36,4**, @26 hafta **−%3,7**, @52 hafta **−%20,4**. İşaret ufka göre dönüyor.

Raporlar bir milestone'un ne yaptığını anlamanın en hızlı yolu; her biri "beklentiyle gerçeğin ayrıştığı yer" ve "bir sonraki milestone'a taşınan borç" bölümleri taşıyor.

---

## 9. Config, knob ve tuning döngüsü

### 9.1 Kural

`CLAUDE.md` §2: kodda çıplak sayı olmaz. Bir sayı ya `config/` altında knob'dur ve `TUNING.md`'de satırı vardır, ya da domain sabitidir ve yanında neden sabit olduğunu açıklayan yorum bulunur. Üçüncü seçenek yok.

Sonuç: `config/` altındaki 16 YAML dosyası sistemin **kontrol paneli**. Kod okumadan davranışı değiştirebilirsin.

| Dosya | Neyi kontrol eder |
|---|---|
| `sim.yaml`, `products.yaml`, `pharmacies.yaml`, `lots.yaml`, `events.yaml` | Dünyanın kendisi |
| `features.yaml`, `depletion.yaml` | Feature builder + tükenme modeli |
| `policy.yaml` | Aday üretimi, **kısıtlar**, aksiyon uzayı, skorlama |
| `uplift.yaml`, `response.yaml` | CATE modeli / dünyanın tepki fonksiyonu (ikisi ayrı — biri politikanın inancı, diğeri gerçek) |
| `allocation.yaml` | LP tahsisi, miad/temizlik rejimi |
| `ope.yaml` | Off-policy tahminciler, rollout |
| `scenarios.yaml`, `agent.yaml`, `harness.yaml` | Kur rejimleri, LLM, regresyon |

### 9.2 Bir knob'u geçici olarak değiştirmek

Dosyayı düzenlemeden, komut satırından:

```bash
uv run python -m experiments.run --profil fast --asama m3 \
    --knob politika.kisit.asgari_kalan_raf_omru_gun=60
```

### 9.3 Bir knob'u süpürmek (asıl öğrenme aracı)

```bash
uv run python -m experiments.sweep \
    --knob politika.kisit.asgari_kalan_raf_omru_gun \
    --values 60,90,120,180 --seeds 5 --profil fast --asama m3,m4
```

Çıktı: her değer için metrik tablosu + yönü gösteren grafik. `SPEC.md` §5b.2'nin kuralı: **bu komut çalışmıyorsa milestone bitmemiştir.**

İki koşuyu yan yana koymak:

```bash
uv run python -m experiments.compare --a kosu_A --b kosu_B
```

### 9.4 `TUNING.md` nasıl okunur

216 KB, milestone milestone bölünmüş. Her knob için sekiz alan var: ne yapar (mekanizma düzeyinde), varsayılan ve makul aralık, artırınca ne olur, azaltınca ne olur, **yanlış ayarın gözlemlenebilir belirtisi**, hangi knob'la etkileşir, ve **çalıştırılabilir teşhis komutu**.

Kullanım şekli: bir metrik tuhaf çıktığında `TUNING.md`'de o metriğin adını ara — hangi knob'ların onu ittiğini ve hangi belirtiye baktığını orada bulursun.

### 9.5 Kalibrasyon disiplini

`notes/predictions.md`: **önce tahmin et, sonra koştur.** Öğrenme tahminin tuttuğu yerde değil, tutmadığı yerde oluyor. Yeni bir knob'la oynamadan önce ne bekliyorsan yaz, sonra sweep'i koş, farkı ve sebebini kaydet.

---

## 10. Komut kartı

```bash
# --- kurulum ---
uv sync                                                  # ortamı lock'a eşitle
uv sync --extra llm                                      # + anthropic (sadece canlı API için)

# --- sağlık kontrolü ---
uv run pytest -q                                         # 213 test, ~7 dk
uv run python -m harness.run --kosu fast                 # 12/12 vaka, ~20 sn

# --- dünya üretimi ---
uv run python -m scripts.generate_world --profil fast --kosu fast
uv run python -m scripts.generate_world --profil full --kosu full

# --- TEKLİF GÖRME ---
uv run python -m scripts.teklif_bak --kosu fast --eczane ECZ0003
uv run python -m scripts.teklif_bak --kosu fast --eczane ECZ0003 --politika propensity
uv run python -m scripts.teklif_bak --kosu fast --origin-sayisi 6 --hafta 83
uv run python -m harness.run --kosu fast --vaka temiz_sablon --metin      # brifing

# --- koşu ve ölçüm ---
uv run python -m experiments.run --profil fast --asama m2,m3 --ad deneme
uv run python -m experiments.run --profil full --asama m4,m5,m6 --ad tam --veri-tut
uv run python -m experiments.sweep --knob <yol> --values a,b,c --seeds 5 --profil fast
uv run python -m experiments.compare --a kosu_A --b kosu_B

# --- çıkış kriterleri (grafikleri reports/figures/mN/ altına yazar) ---
uv run python -m scripts.verify_m1 --kosu full
uv run python -m scripts.verify_m6 --kosu full
uv run python -m scripts.verify_m7 --kosu fast --hizli
```

Yaklaşık süreler (bu makine, `fast` profil): `experiments.run --asama m2,m3` ≈ 15 sn · `harness.run` ≈ 20 sn · `teklif_bak` ≈ 20 sn · `pytest` ≈ 7 dk. `full` profil M4+M5+M6 rollout ile birkaç dakika (rapora göre rollout tek başına ~164 sn).

---

## 11. Dikkat edilecekler

- **`fast` ile `full` karıştırılmaz.** `fast` (60×100) sweep ve iterasyon içindir, M1 çıkış kriterini doğrulamaz. Raporlanan her sayının yanında hangi profilde koştuğu yazılı olmalı.
- **Her koşu seed'li.** `--seed` ile `profil.temel_seed` ezilir. Aynı config + aynı seed = aynı sonuç; testler bunu doğruluyor.
- **`experiments.run` kendi dünyasını üretir ve iş bitince siler.** Saklamak istiyorsan `--veri-tut` ver. (Not: `experiments/run.py` dosya başlığındaki yorum "varsayılan olarak silmez" diyor ama kod tersini yapıyor — `run.py:1648`. Yorum yanlış, davranış doğru.)
- **`ground_truth/` okumak yasak.** Politika ve model tarafında bir yerde onu okuduğunu görürsen bu bir hatadır, doğrulama scriptleri de bunu arıyor.
- **Metrik şüpheli derecede iyiyse** sırayla: leakage mı, simülatör mü kolay, ölçüm maskesi mi yanlış.
- **Türkçe kod.** Fonksiyon ve değişken adları Türkçe (`tepki_hesapla`, `kisit_uygula`, `dunya_kur`). Yorumlar ASCII (Türkçe karakter yok), dokümanlar tam Türkçe.

---

## 12. Okuma sırası (yeni gelen için)

1. Bu dosya
2. `SPEC.md` §0–§1 — D1–D9 tasarım kararları. Bunlar tartışmaya kapalı ve sistemin şeklini bunlar belirliyor
3. `uv run python -m harness.run --kosu fast --vaka temiz_sablon --metin` — çıktıyı gördükten sonra geri kalanı okumak çok daha kolay
4. `reports/m6.md` §0–§1 — projenin varlık sebebi
5. `config/policy.yaml` — kısıt ve aksiyon blokları; sistemin karar mantığı yorumlarda anlatılmış
6. `sim/world.py` başlık yorumu — dünyanın haftalık döngüsü ve zorluk kaynakları
7. `TUNING.md` — baştan sona değil, ihtiyaç oldukça
