# Kalıcı Kurallar

Bu repo bir öğrenme substratıdır. Çıktısı sadece çalışan kod değil, **kodu tune edebilir hale gelmiş bir insandır.** Kurallar buna göre.

Her oturumda önce `SPEC.md` oku.

---

## 1. Kapsam disiplini

- **Tek seferde tek milestone.** Sana hangi milestone söylendiyse sadece onu yap.
- **İleriye dönük dosya oluşturma.** Sonraki milestone'lar için stub, iskelet, boş modül, `pass` içeren fonksiyon, "TODO: M4'te doldurulacak" yorumu — hiçbiri olmayacak.
- Sonraki milestone'un işine yarayacağını düşündüğün bir şey varsa yapma, **raporda öner.**
- Milestone çıkış kriterini karşılamadan "bitti" deme.

## 2. Sihirli sayı yasağı

Kodda çıplak sayı olmaz. Bir sayı ya:
- `config/` altında YAML'da knob olur ve `TUNING.md`'de satırı bulunur, **ya da**
- Domain sabitidir, kodda kalır ve yanında neden sabit olduğunu açıklayan tek satır yorum bulunur.

Arada üçüncü seçenek yok. `if kalan_gun < 30:` gibi bir satır görüyorsan o 30 config'e çıkacak.

## 3. Her milestone dört artifact üretir

1. **Çalışan kod** — çıkış kriterini karşılayan
2. **Doğrulama scripti** — kriterin karşılandığını gösteren, çalıştırılabilir
3. **`TUNING.md` güncellemesi** — SPEC §5b.1 formatında, o milestone'da doğan her knob için
4. **Milestone raporu** — SPEC §5b.3 formatında, `reports/mN.md` olarak

Artı: SPEC §5b.4 formatında en az bir kalibrasyon egzersizi.

`TUNING.md` satırı "bu parametre eşiği belirler" gibi içi boş olamaz. Mekanizma, makul aralık, artırınca/azaltınca ne olur, yanlış ayarın gözlemlenebilir belirtisi, çalıştırılabilir teşhis komutu.

## 4. Tasarım kararları (D1–D9)

`SPEC.md` §1'deki D1–D9 tartışmaya kapalıdır. Bir tanesini ihlal etmen gerektiğini düşünüyorsan **uygulamadan önce sor**, gerekçeni yaz, onay bekle.

Özellikle:
- Aksiyon uzayı `(mal_fazlası, vade)` — yüzde iskonto ekleme
- Kur tahmin edilmez, senaryolaştırılır
- Kısıt katmanının ML skoru üzerinde veto yetkisi var
- LLM karar noktasında yok
- Miad temizliği ayrı motor değil, tahsis rejimi

## 5. Teknik

- Notebook yok. Çalıştırılabilir script + config.
- Her koşu seed'li ve tekrar üretilebilir. İki kez çalıştırınca aynı sonuç.
- Point-in-time doğruluk: feature builder geleceği göremez. Leakage guard yaz ve test et.
- `experiments/sweep.py` M2'den itibaren çalışır durumda olacak. Bir milestone'un bitmiş sayılmasının şartı sweep komutunun koşmasıdır.
- Python 3.11+, `uv`. LightGBM, scikit-learn, lifelines, PuLP/scipy, polars veya pandas.

## 6. Dürüstlük

- Basitleştirme yaptıysan raporda yaz. Hangi basitleştirme, ne zaman patlar.
- Çıkış kriterini kısmen karşıladıysan "kısmen" de, "tamam" deme.
- Doğrulama scriptini çalıştırmadan sonucunu raporlama.
- Bir şeyin işe yaradığını iddia ediyorsan ölçümünü göster.
- Emin olmadığın sektör bilgisini uydurma — parametrik bırak ve `SPEC.md` §8'deki doğrulama listesine ekle.

## 7. Simülatör zorluğu

Sentetik veri **öğrenmesi kolay olmamalı.** Modeller neredeyse mükemmel çalışıyorsa simülatör yanlış tasarlanmış demektir ve proje amacını kaybeder.

- Gürültü gerçek olacak, tüketim deterministik olmayacak
- Talep intermittent: çoğu (eczane, SKU) hücresi çoğu hafta sıfır
- `share_of_wallet` latent kalacak — model göremeyecek
- Uplift heterojen olacak: segmentler tekliflere farklı tepki verecek

Bir modelin metriği şüpheli derecede iyiyse önce leakage, sonra simülatör kolaylığı kontrol edilir.
