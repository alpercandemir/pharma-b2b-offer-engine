# Claude Code Komutları

Kopyala-yapıştır. Her milestone için temiz oturum (`/clear`).

---

## 0. Bootstrap (bir kez)

```
CLAUDE.md ve SPEC.md'yi oku.

Henüz kod yazma. Bana şunu ver:

1. SPEC'i anladığını gösteren 10 satırlık bir özet — sistemin ne yaptığı,
   hangi katmanlardan oluştuğu
2. D1-D9 tasarım kararlarından hangilerini uygulamada zorlayıcı bulduğun
3. SPEC'te belirsiz, çelişkili veya eksik bulduğun noktalar
4. Önerdiğin proje iskeleti (dizin yapısı) ve bağımlılık listesi

Sadece bunlar. Dosya oluşturma.
```

> Bu adım önemli. Claude Code'un SPEC'i yanlış anladığı yeri kod yazmadan yakalarsın. Belirsizlik listesini de gerçekten oku — SPEC'te düzeltilmesi gereken şeyler çıkacak.

---

## 1. Milestone komutları

Her biri ayrı oturumda. Öncekini kabul etmeden sonrakine geçme.

### M1 — Simülatör ve ground truth

```
CLAUDE.md ve SPEC.md'yi oku. Sadece M1'i implemente et.

Kapsam: simülatör ve ground truth katmanı. Latent tüketim hızları,
eczane persona'ları, mevsimsellik, intermittent talep, rejim olayları,
lot boyutu ve miad dağılımı (SPEC §2.5).

Diğer milestone'lara ait hiçbir dosya, stub veya iskelet oluşturma.

Teslim:
1. M1 çıkış kriterini karşılayan çalışan kod
2. Doğrulama scripti: talep dağılımı, mevsimsellik, olay etkisi,
   seyreklik, FEFO tüketimi — hepsi görselleştirilmiş
3. TUNING.md (§5b.1 formatında)
4. reports/m1.md (§5b.3 formatında)
5. Bir kalibrasyon egzersizi (§5b.4)

CLAUDE.md §7'yi özellikle dikkate al: simülatör öğrenmesi kolay olmayacak.
Raporda dünyayı ne kadar zorlaştırdığını ve neden o seviyeyi seçtiğini yaz.
```

### M2 — Tükenme modeli

```
CLAUDE.md ve SPEC.md'yi oku. M1 tamamlandı. Sadece M2'yi implemente et.

Kapsam: sipariş miktarından tüketim hızı çıkarımı, eldeki stok tahmini,
tükenme zamanı hazard modeli (D2).

Ayrıca bu milestone'da experiments/ altyapısı kurulacak:
run.py, sweep.py, compare.py, runs/ (SPEC §5b.2).
Şu komut çalışmadan M2 bitmiş sayılmaz:
  python -m experiments.sweep --knob <knob> --values a,b,c --seeds 5

Teslim listesi CLAUDE.md §3'teki gibi.

Ek olarak raporda: "son N günde aldı mı" basit kuralı ile hazard modelinin
karşılaştırması. Fark ölçülmüş olacak, iddia edilmiş değil.

Metriğin şüpheli derecede iyiyse önce leakage kontrol et, sonra bana söyle.
```

### M3 — Aday üretimi + kısıt katmanı

```
CLAUDE.md ve SPEC.md'yi oku. M1-M2 tamamlandı. Sadece M3'ü implemente et.

Kapsam: CF / market basket ile aday havuzu üretimi, üzerine hard veto
katmanı (D6). Cold start için eczane attribute'ları.

Kısıt katmanı test edilebilir olacak. En az şu testler:
- Kırmızı ve yeşil reçeteli ürün hiçbir koşulda öneri listesinde çıkmıyor
- Soğuk zincir min sipariş kuralı ihlal edilmiyor
- Kredi limiti aşan teklif üretilmiyor
- Miad baskısı promosyon vetosunu AŞMIYOR (§2.5)

Teslim listesi CLAUDE.md §3'teki gibi.
```

### M4 — Uplift ve aksiyon seçimi

```
CLAUDE.md ve SPEC.md'yi oku. M1-M3 tamamlandı. Sadece M4'ü implemente et.

Kapsam: (mal_fazlası_oranı, vade_günü) aksiyon uzayında CATE modeli.
T-learner ve X-learner, karşılaştırmalı.

Raporun merkezinde şu olacak: propensity-based politika ile uplift-based
politika arasındaki MARJ FARKI. Sayıyla. Propensity'nin nerede ve ne kadar
marj yaktığını göster.

İki politika aynı sonucu veriyorsa simülatörde uplift heterojenliği yok
demektir — kod yazmayı bırak, bana söyle.

Teslim listesi CLAUDE.md §3'teki gibi.
```

### M5 — Tahsis ve miad rejimi

```
CLAUDE.md ve SPEC.md'yi oku. M1-M4 tamamlandı. Sadece M5'i implemente et.

Kapsam: sonlu stok altında LP tahsis (D5) + miadı yaklaşan lotta negatif
shadow price rejimi (D9, §2.5).

İki karşılaştırma raporlanacak:

(a) Kıt SKU senaryosu: ranking-only vs LP.
    Metrik: stockout sayısı, karşılanmayan talep, toplam marj.

(b) Kısa miatlı lot senaryosu: üç politika —
    temizlik yok / kör iskonto / M2 kuplajlı hedefli temizlik.
    Metrik: imha adedi, iade adedi, net marj, eczane memnuniyeti proxy'si.

(b) şıkkında max_teklif_adedi kısıtı M2'nin tüketim hızı çıktısından
gelecek (§2.5). Sabit bir tavan koyma.

Shadow price'ları raporda göster ve yorumla — miad rejiminde işaret
değişimini görebilmeliyim.

Teslim listesi CLAUDE.md §3'teki gibi.
```

### M6 — Off-policy evaluation

```
CLAUDE.md ve SPEC.md'yi oku. M1-M5 tamamlandı. Sadece M6'yı implemente et.

Bu milestone projenin amacıdır. Diğerleri buraya kurulumdu.

Kapsam:
1. IPS, SNIPS, Doubly-Robust estimator'lar
2. eval/oracle.py — sentetik olduğu için gerçek counterfactual
3. sim/rollout.py — closed-loop: politika aksiyon alır, dünya tepki verir
4. Karşılaştırma raporu

Rapor şu soruyu cevaplayabilmeli:
"Offline estimator +%X dedi, closed-loop rollout %Y çıktı. Neden?"

Sapmanın kaynağını ayrıştır: variance, overlap ihlali, extrapolation,
propensity kalibrasyonu. Her birini ayrı ayrı göster.

Ayrıca değerlendirme ufkunu knob yap: 4 hafta vs 52 hafta.
Kısa ufukta agresif iskontonun kazanıp uzun ufukta kaybettiğini
gösteren sweep'i koştur.

Offline ile online şüpheli derecede uyuşuyorsa exploration fazla geniş
demektir — daralt ve tekrar koş.

Teslim listesi CLAUDE.md §3'teki gibi.
```

### M7 — LLM katmanı

```
CLAUDE.md ve SPEC.md'yi oku. M1-M6 tamamlandı. Sadece M7'yi implemente et.

Kapsam (D8 — LLM karar noktasında YOK):
1. agent/scenario.py — baz/yüksek/şok kur rejimi altında politikanın ne
   önerdiğini yorumlayan katman. Kur TAHMİN ETMİYOR, senaryo altında
   koşullu okuma yapıyor (D3).
2. agent/narrative.py — KAM/saha için teklif brifingi
3. agent/tools.py — LLM'in çağırabileceği fonksiyonlar (tool use)
4. harness/ — eval harness

Eval harness'ta LLM çıktısı deterministik test edilecek. En az:
- Sayı uydurma yakalanıyor mu (rapordaki rakam gerçek çıktıyla uyuşuyor mu)
- Kısıt ihlali iddiası yakalanıyor mu (vetolanmış ürünü önerdi mi)
- Senaryo karıştırma yakalanıyor mu (baz senaryonun sayısını şok senaryoya
  yazdı mı)
- Hallüsinasyon: var olmayan eczane/SKU/lot referansı

Teslim listesi CLAUDE.md §3'teki gibi.
```

---

## 2. Kabul komutu

Milestone bittiğinde, kabul etmeden önce:

```
Çıkış kriterini kendin doğrula.

1. Doğrulama scriptini çalıştır ve çıktısını göster
2. experiments/sweep.py'yi TUNING.md'deki knob'lardan biriyle koştur,
   sonucu göster
3. Kodda config'e çıkmamış çıplak sayı kaldı mı — tara ve listele
4. İleriye dönük stub/iskelet dosya oluşturdun mu — kontrol et
5. Raporda yazdığın her ölçüm gerçekten koşturuldu mu

Bulduğun ihlalleri düzelt, sonra bana özet ver.
```

---

## 3. Tuning oturumu komutları

Milestone kabul edildikten sonra, öğrenme kısmı. Kod yazdırma, sweep koştur.

```
TUNING.md'deki knob'lardan <KNOB> için sweep koştur:
değerler <a,b,c,d>, 5 seed.

Sonucu tablo + tek grafik olarak ver. Yorum yazma, sadece sayı.
```

```
<KNOB_A> ve <KNOB_B> için 2D sweep koştur. Etkileşim var mı,
yoksa bağımsız mı — heatmap ile göster.
```

```
Şu iki config'i karşılaştır: <A> ve <B>.
compare.py ile çalıştır, farkı ve istatistiksel anlamlılığını ver.
Hangi metrikte fark var, hangisinde yok.
```

> Sweep sonucunu görmeden önce tahminini `notes/predictions.md`'ye yaz.
> README'deki kalibrasyon disiplini bu.

---

## 4. Müdahale komutları

Claude Code kaydığında. Kısa tut, açıklama yapma.

**İleriye dönük dosya oluşturduysa:**
```
CLAUDE.md §1 ihlali. Şu dosyalar bu milestone'un kapsamında değil: <liste>
Sil. Sadece bu milestone'un çıkış kriterini karşıla.
```

**Sihirli sayı bıraktıysa:**
```
CLAUDE.md §2 ihlali. Şu satırlarda çıplak sayı var: <liste>
Her biri ya config'e çıkacak ve TUNING.md'de satırı olacak, ya domain
sabiti olarak gerekçeli yorumla kalacak. Karar ver ve uygula.
```

**TUNING.md içi boşsa:**
```
TUNING.md satırları bilgi taşımıyor. Her knob için:
mekanizma (parametre adının tekrarı değil), makul aralık, artırınca ve
azaltınca hangi metrikte hangi yönde değişim, yanlış ayarın gözlemlenebilir
belirtisi, çalıştırılabilir teşhis komutu.
Yeniden yaz.
```

**Model şüpheli derecede iyiyse:**
```
Bu metrik gerçekçi değil. Sırayla kontrol et:
1. Feature builder geleceği görüyor mu — point-in-time doğruluğu test et
2. Target leakage var mı
3. Simülatör fazla kolay mı — gürültü seviyesi, seyreklik, heterojenlik
Bulgunu raporla, tahmin yürütme.
```

**Bir tasarım kararını sessizce ihlal ettiyse:**
```
<D_NO> ihlal edildi: <ne yaptı>
CLAUDE.md §4: bu kararlar tartışmaya kapalı, ihlal önce sorulur.
Geri al. İhlal gerekli olduğunu düşünüyorsan gerekçeni yaz, uygulamadan bekle.
```

**Doğrulamadan "bitti" dediyse:**
```
Doğrulama scriptini çalıştırmadan çıkış kriteri karşılandı diyemezsin.
Çalıştır, ham çıktıyı göster.
```

**Kapsamı genişlettiyse:**
```
Bu milestone'un kapsamı <X>. Yaptığın <Y> sonraki milestone'a ait.
Geri al, raporda öneri olarak yaz.
```
