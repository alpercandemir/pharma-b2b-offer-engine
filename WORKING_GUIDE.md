# B2B İlaç Pazaryeri Karar Motoru — Çalışma Rehberi

Bu repo bir üretim sistemi değil. Amacı: **teklif/kampanya karar motorunun uçtan uca akışını ve tuning'ini elle görmek.**

Sentetik veriyle çalışıyor. Ground truth'u sen ürettiğin için offline evaluation'ın doğru olup olmadığını denetleyebiliyorsun — gerçek hayatta olmayan bir lüks. Projenin asıl öğretici anı burada.

---

## Dosyalar ve kimin için oldukları

| Dosya | Okuyucu | İşlevi |
|---|---|---|
| `WORKING_GUIDE.md` (bu dosya) | Sen | Kurulum, çalışma döngüsü, Claude Code çıktısını nasıl denetleyeceğin |
| `PROMPTS.md` | Sen | Claude Code'a kopyala-yapıştır vereceğin komutlar |
| `CLAUDE.md` | Claude Code | Her oturumda otomatik okunan kalıcı kurallar |
| `SPEC.md` | Claude Code | Teknik spesifikasyon — tasarım kararları, sektör parametreleri, milestone'lar |
| `README.md` | Dışarıdan bakan | Projenin İngilizce tanıtımı — proje bittikten sonra yazıldı |
| `REHBER.md` · `DATA.md` | Dışarıdan bakan | İlk kez bakan için giriş · veri sözlüğü |

`SPEC.md` çalıştırılabilir bir şey değil, referans dokümandır. Sen `PROMPTS.md`'den komut verirsin, Claude Code `CLAUDE.md` + `SPEC.md` okuyarak uygular.

---

## Kurulum

```bash
mkdir pharma-offer-engine && cd pharma-offer-engine
git init

# Dört dosyayı repo köküne kopyala
# WORKING_GUIDE.md  PROMPTS.md  CLAUDE.md  SPEC.md

git add . && git commit -m "spec + guidance"

claude
```

**Gereksinimler:** Claude Code kurulu, Python 3.11+, `uv` (veya `poetry`).
Ortam kurulumunu Claude Code M1'de kendisi yapacak — sen önden `pyproject.toml` hazırlama.

---

## Çalışma döngüsü

Milestone başına **bir oturum**. Kural bu, esnetme.

```
1. claude başlat, /clear ile temiz context
2. PROMPTS.md'den ilgili milestone komutunu yapıştır
3. Claude Code çalışır
4. SEN doğrularsın (aşağıdaki denetim listesi)
5. Kabul edersen commit, etmezsen düzeltme komutu ver
6. Kalibrasyon egzersizini YAP
7. /clear, sonraki milestone
```

Context'i milestone'lar arası taşıma. Uzun context'te Claude Code önceki milestone'un detayına takılıp yenisini yüzeysel yapıyor, ayrıca ileriye dönük iskelet dosya üretme eğilimi artıyor.

Her milestone sonunda commit at — `git tag m1`, `git tag m2` gibi. Bir milestone'u yeniden yapmak isteyeceksin, geri dönebilmen lazım.

---

## Claude Code çıktısını denetleme

Bu bölüm rehberin asıl değerli kısmı. Kod çalışıyor diye kabul etme.

### Her milestone'da kontrol et

- [ ] **Sadece o milestone yapılmış mı?** İleriye dönük boş dosya, stub, "sonra doldururuz" yorumu varsa reddet.
- [ ] **Config'te knob olmayan sayı var mı?** Kodda `0.85`, `if days < 30` gibi çıplak sayı görüyorsan ya config'e çıkacak ya `TUNING.md`'de satırı olacak. Arada bir şey yok.
- [ ] **`TUNING.md` satırları gerçek bilgi taşıyor mu?** "Bu parametre eşiği belirler" yazmışsa değersiz. Mekanizma, yön, yanlış ayarın belirtisi ve teşhis komutu olacak.
- [ ] **Doğrulama scripti gerçekten koşuyor mu?** Claude Code "çıkış kriteri karşılandı" der ama çalıştırmamış olabilir. Sen çalıştır.
- [ ] **Notebook var mı?** Varsa sil, script iste.
- [ ] **Seed'li ve tekrar üretilebilir mi?** İki kez koştur, aynı sonucu ver mi.
- [ ] **Raporda "basitleştirme" bölümü dolu mu?** Boşsa ya yalan söylüyor ya farkında değil; ikisi de sorun.

### Milestone'a özel tuzaklar

**M1 — simülatörün fazla kolay olması.** En sinsi tuzak. Claude Code temiz, gürültüsüz, öğrenmesi kolay bir dünya üretirse M2'de model %98 doğrulukla çalışır ve sen hiçbir şey öğrenmezsin. Kontrol:
- Talep matrisinin çoğu hücresi sıfır mı? (İlaç dağıtımında öyle olmalı)
- Uzun kuyruk var mı, yoksa SKU'lar birbirine mi benziyor?
- `share_of_wallet` latent mi, yoksa gözlemlenebilir mi bırakılmış?
- Gürültü var mı, yoksa tüketim deterministik mi?

**M2 — model çok iyi çalışıyorsa şüphelen.** Tükenme tahmininde MAE çok düşükse ya leakage var ya simülatör kolay. İkisini de kontrol et. Leakage için: feature'lar point-in-time mi, geleceği görüyor mu?

**M4 — propensity ile uplift'in aynı sonucu vermesi.** Aynı çıkıyorsa simülatörde uplift heterojenliği yok demektir. Farklı segmentlerin tekliflere farklı tepki verdiğinden emin ol, yoksa M4'ün öğreteceği şey kaybolur.

**M6 — offline ile online'ın uyuşması.** Uyuşuyorsa ya şanslısın ya bir şey yanlış. IPS'in sapması normal ve öğretici; sapma yoksa overlap fazla iyi, exploration fazla geniş demektir. Exploration'ı daralt, tekrar koş.

### Claude Code kaydığında

Kalıcı kural ihlali görürsen `PROMPTS.md` §4'teki müdahale komutlarını kullan. Uzun uzun açıklama yazma, kısa ve net düzeltme ver.

---

## Milestone haritası

| # | Ne yapıyorsun | Ne öğreniyorsun |
|---|---|---|
| M1 | Simülatör, ground truth, lot/miad | Sentetik veri tasarımı — zorluğu ayarlamak |
| M2 | Tükenme/hazard modeli | "Aldı mı" kuralı ile hazard modeli farkı, ölçülebilir |
| M3 | Aday üretimi + kısıt katmanı | Hard constraint'in ML skorunu nasıl veto ettiği |
| M4 | Uplift / CATE | Propensity'nin marjı nasıl yediği, sayıyla |
| M5 | Tahsis LP + miad rejimi | Shadow price, kıt kaynak, negatif salvage value |
| **M6** | **Off-policy evaluation** | **Offline tahminin ne zaman yalan söylediği** |
| M7 | LLM orkestrasyon + eval harness | LLM'i deterministik test etmek |

**M6 projenin amacıdır.** Diğerleri oraya kurulum. M6'ya varmadan bırakırsan elinde yarım bir öneri sistemi kalır; asıl öğrenilecek şey orada.

Zaman bütçen daralırsa M3 ve M7'yi kısabilirsin. M1–M2–M4–M5–M6 hattını bozma.

---

## Kalibrasyon disiplini

Her milestone bir "önce tahmin et, sonra çalıştır" egzersizi bırakıyor. Bunu atlama.

Tahminini **yazılı** olarak kaydet — `notes/predictions.md` gibi bir dosya tut. Sonra sweep'i koştur, sonucu yanına yaz.

```markdown
## M4 / uplift.min_effect_threshold: 0.02 → 0.10
Tahmin: teklif sayısı ~%40 düşer, toplam marj hafif düşer, marj/teklif belirgin artar
Gerçek: teklif sayısı %63 düştü, toplam marj %8 ARTTI, marj/teklif 2.4x
Neden yanıldım: düşük etkili tekliflerin marj katkısı negatifmiş — MF maliyeti
                zaten alacak eczaneye gidiyormuş. Eşik yükselmesi onları kesti.
```

Bu dosya projenin en değerli çıktısı olacak. Tuning öğrenmesi tahminin tuttuğu yerde değil, **tutmadığı yerde** oluyor.

---

## Bittiğinde elinde ne olur

- Çalışan bir karar motoru ve onu tune edebilme yetisi
- Hangi problemin ML, hangisinin optimizasyon, hangisinin LLM olduğunu ayırt eden bir örnek
- Offline evaluation'ın ne zaman yalan söylediğine dair birinci elden deneyim
- Anlatılabilir bir vaka: "regüle fiyat altında mal fazlası aksiyon uzayı", "miadı yaklaşan lotta shadow price'ın işaret değiştirmesi", "referans kur güncellemesinin tahmin değil rejim problemi olması"

Son madde mülakat masasında en çok işe yarayan kısım. Model eğitmek değil, problemi doğru sınıflandırmak ayırt edici olan.
