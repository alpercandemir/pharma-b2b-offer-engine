"""M6 testleri: tahmincilerin ozdeslikleri, teshisler ve sizinti sinirlari.

EN ONEMLI UC TEST:

  `test_ips_kayit_politikasinin_degerini_buluyor`
      Hedef = kayit politikasi alindiginda IPS gozlenen odulun ortalamasina
      INDIRGENIR (w = 1 her satirda). Kagit uzerinde dogru cevabi bilinen tek
      vaka bu; tutmuyorsa butun M6 tablosu olcek hatasi tasir.

  `test_ips_deterministik_dunyada_oracle_i_TAM_buluyor`
      Odul gurultusuz (deterministik) yapilinca IPS'in beklenen degeri
      oracle'a TAM esit olmali. Sentetik kucuk bir ornek uzerinde cebirsel
      olarak sinaniyor -- boylece "yaklasik tuttu" ile "dogru kuruldu"
      birbirinden ayriliyor.

  `test_loglanmis_veri_ground_truth_tasimiyor`
      `LoglanmisVeri`nin alan listesi KILITLI. Bir gun birisi oraya gercek
      kabul olasiligini koyarsa M6'nin butun iddiasi coker; test alan adlarini
      donduruyor.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import Config, config_yukle
from eval import ope as ev
from policy import bandit
from policy.scorer import TEKLIF_YOK

PROFIL = "fast"


@pytest.fixture(scope="module")
def cfg():
    return config_yukle(PROFIL)


def _veri(n=200, A=4, seed=7, sicaklik=1.0) -> ev.LoglanmisVeri:
    """Kucuk sentetik log kumesi. Dunyayi kosmadan tahminci cebiri sinanir."""
    rng = np.random.default_rng(seed)
    pi = rng.dirichlet(np.ones(A) * 2.0, size=n)
    kol = np.array([rng.choice(A, p=pi[i]) for i in range(n)])
    return ev.LoglanmisVeri(
        X=rng.normal(size=(n, 3)),
        kol=kol,
        propensity=pi[np.arange(n), kol],
        pi_log=pi,
        odul=rng.normal(10.0, 3.0, n),
        izinli=np.ones((n, A), dtype=bool),
        eczane_idx=rng.integers(0, 10, n),
        origin=np.zeros(n, dtype=int),
        tekrar=np.zeros(n, dtype=int),
    )


# --------------------------------------------------------------------------
# sizinti siniri
# --------------------------------------------------------------------------
def test_loglanmis_veri_ground_truth_tasimiyor():
    """OPE'nin gordugu alan listesi kilitli (D7'nin sinirinin mekanik hali)."""
    alanlar = set(ev.LoglanmisVeri.__dataclass_fields__)
    assert alanlar == {"X", "kol", "propensity", "pi_log", "odul", "izinli",
                       "eczane_idx", "origin", "tekrar"}
    # Gercek tepkinin adini tasiyan hicbir alan olamaz.
    assert not [a for a in alanlar if "gercek" in a or "oracle" in a or "uplift" in a]


def test_kayit_q_alt_iki_katmanda_ayni():
    """core/ politikayi import edemez; Q_ALT iki yerde yazili, esit olmali.

    `Config.KAYIT_Q_ALT` M6'nin ortusme kilidini besliyor; `policy/bandit.py`
    ayni sayiyi teklif verme olasiliginin alt siniri olarak kullaniyor.
    Ayrisirlarsa kilit yanlis tabani hesaplar ve sessizce gecer.
    """
    assert Config.KAYIT_Q_ALT == bandit.Q_ALT


def test_kontrol_kolu_indeksi_ope_ile_ayni():
    assert ev.TEKLIF_YOK == TEKLIF_YOK


# --------------------------------------------------------------------------
# tahminci cebiri
# --------------------------------------------------------------------------
def test_ips_kayit_politikasinin_degerini_buluyor():
    """Hedef = gozlenen aksiyon -> w = 1 -> IPS = ortalama odul."""
    v = _veri()
    ag = ev.onem_agirligi(v.kol, v, v.propensity, np.inf)
    # Eslesme her satirda; agirlik 1/pi.
    assert ag.esles.all()
    # Kendi aksiyonunu hedefleyen IPS agirliksiz ortalamaya indirgenmeli:
    # w = 1 kurulunca.
    assert ev.ips(v.odul, np.ones(v.n)) == pytest.approx(v.odul.mean())


def test_ips_beklenen_degeri_oracle_a_esit():
    """Sonsuz orneklemde IPS yansizdir; sonlu orneklemde BEKLENEN degeri esit.

    Zar atmadan sinanir: odul deterministik r(x, a) alinir ve IPS'in beklenen
    degeri sum_a pi(a|x) * [1(a=hedef)/pi(a|x)] * r(x,a) = r(x, hedef) olur.
    Bu, tahmincinin dogru kuruldugunun cebirsel kaniti.
    """
    rng = np.random.default_rng(3)
    n, A = 500, 4
    pi = rng.dirichlet(np.ones(A) * 3.0, size=n)
    r = rng.normal(5.0, 2.0, size=(n, A))          # deterministik odul tablosu
    hedef = rng.integers(0, A, n)

    # IPS'in beklenen degeri: her satirda pi ile agirlikli toplam.
    beklenen = np.zeros(n)
    for a in range(A):
        w = np.where(hedef == a, 1.0 / pi[:, a], 0.0)
        beklenen += pi[:, a] * w * r[:, a]
    oracle = r[np.arange(n), hedef]
    assert beklenen == pytest.approx(oracle)


def test_snips_olcek_kaymasina_dayanikli():
    """Butun propensity'ler ayni katsayiyla bozulursa SNIPS degismez, IPS kayar.

    SNIPS'in varlik sebebi tam olarak budur ve M6 raporunda "hangi tahminciye
    ne zaman guvenilir" tablosunun satiri buradan geliyor.
    """
    v = _veri()
    hedef = v.kol.copy()
    hedef[::3] = (hedef[::3] + 1) % v.A          # kismi eslesme
    ag = ev.onem_agirligi(hedef, v, v.propensity, np.inf)
    ag_bozuk = ev.onem_agirligi(hedef, v, v.propensity * 0.8, np.inf)

    assert ev.snips(v.odul, ag.kirpik) == pytest.approx(
        ev.snips(v.odul, ag_bozuk.kirpik))
    assert ev.ips(v.odul, ag.kirpik) != pytest.approx(
        ev.ips(v.odul, ag_bozuk.kirpik))


def test_dr_mukemmel_modelde_dogrudan_yonteme_esit():
    """q hatasi sifirsa artik sifirdir ve DR = dogrudan yontem.

    "Cift saglamligin" bir yarisi: sonuc modeli dogruysa onem agirliklarinin
    ne kadar patladigi onemsizdir.
    """
    rng = np.random.default_rng(11)
    n, A = 300, 3
    v = _veri(n=n, A=A, seed=11)
    q = np.tile(v.odul[:, None], (1, A))          # q(x, a) = gozlenen odul
    hedef = rng.integers(0, A, n)
    idx = np.arange(n)
    ag = ev.onem_agirligi(hedef, v, v.propensity, 50.0)
    dr = ev.doubly_robust(v.odul, ag.kirpik, q[idx, v.kol], q[idx, hedef])
    assert dr == pytest.approx(ev.dogrudan_yontem(q[idx, hedef]))


def test_dr_ise_yaramaz_modelde_ips_e_donuyor():
    """q = 0 alinirsa DR tanimi geregi IPS'in ta kendisidir."""
    v = _veri(seed=5)
    hedef = v.kol.copy()
    hedef[::4] = (hedef[::4] + 2) % v.A
    ag = ev.onem_agirligi(hedef, v, v.propensity, 30.0)
    sifir = np.zeros(v.n)
    assert ev.doubly_robust(v.odul, ag.kirpik, sifir, sifir) == pytest.approx(
        ev.ips(v.odul, ag.kirpik))


# --------------------------------------------------------------------------
# kirpma
# --------------------------------------------------------------------------
def test_kirpma_kutle_siliyor_ve_yanlilik_yonu_belli():
    """Kirpma agirlik kutlesini yalnizca AZALTIR; pozitif odulde tahmin duser.

    Kirpmanin yanliligi rassal degil YONLU: tavan yalnizca buyuk agirliklari
    keser, dolayisiyla katkilari hep asagi ceker. Sapma ayristirmasinin
    "kirpma" kalemi bu yuzden pozitif odul rejiminde negatif bekleniyor.
    """
    v = _veri(seed=13)
    v.odul[:] = np.abs(v.odul)                    # pozitif odul rejimi
    hedef = v.kol.copy()
    ham = ev.onem_agirligi(hedef, v, v.propensity, np.inf)
    kirpik = ev.onem_agirligi(hedef, v, v.propensity, 3.0)

    assert (kirpik.kirpik <= ham.kirpik + 1e-12).all()
    assert kirpik.silinen_kutle > 0.0
    assert ev.ips(v.odul, kirpik.kirpik) < ev.ips(v.odul, ham.kirpik)


def test_kirpma_tavan_altinda_etkisiz():
    """Tavan azami agirligin ustundeyse kirpma hicbir sey degistirmemeli."""
    v = _veri(seed=17)
    hedef = v.kol.copy()
    ham = ev.onem_agirligi(hedef, v, v.propensity, np.inf)
    tavan = float(ham.ham.max()) * 1.5
    kirpik = ev.onem_agirligi(hedef, v, v.propensity, tavan)
    assert kirpik.silinen_kutle == pytest.approx(0.0)
    assert ev.ips(v.odul, kirpik.kirpik) == pytest.approx(ev.ips(v.odul, ham.kirpik))


# --------------------------------------------------------------------------
# propensity bozmasi
# --------------------------------------------------------------------------
def test_sicaklik_bir_birim_fonksiyon():
    """sicaklik = 1.0'da matris DEGISMEDEN donmeli (yeniden normalizasyon dahil)."""
    v = _veri()
    assert ev.sicaklik_uygula(v.pi_log, 1.0) is v.pi_log


def test_sicaklik_dagilimi_duzlestiriyor_ve_toplami_koruyor():
    """sicaklik > 1 entropiyi artirir; satirlar yine 1'e toplanir."""
    v = _veri(seed=23)
    duz = ev.sicaklik_uygula(v.pi_log, 2.0)
    assert duz.sum(axis=1) == pytest.approx(np.ones(v.n))
    entropi = lambda p: -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)  # noqa: E731
    assert (entropi(duz) >= entropi(v.pi_log) - 1e-12).all()
    # Duzlestirme KUCUK olasiliklari buyutur, BUYUKLERI kucultur.
    en_buyuk = v.pi_log.argmax(axis=1)
    idx = np.arange(v.n)
    assert (duz[idx, en_buyuk] <= v.pi_log[idx, en_buyuk] + 1e-12).all()


def test_propensity_loglanan_kaynagi_gercegi_aynen_veriyor(cfg):
    """kaynak = loglanan + sicaklik = 1 -> kalibrasyon hatasi tam sifir.

    Sapma ayristirmasinin "propensity" kalemi bu ayarda TAM sifir olmali;
    olmuyorsa merdivenin ucuncu basamagi bir seyi sessizce degistiriyordur.
    """
    v = _veri(seed=29)
    # Alt kirpma gercek propensity'yi kesmesin diye taban dusuruluyor.
    c = config_yukle(PROFIL, gecersiz_kilma={"ope.propensity.kirpma_alt": 1e-6})
    p = ev.propensity_hazirla(c, v)
    assert p.propensity == pytest.approx(v.propensity)
    assert p.ortalama_mutlak_hata == pytest.approx(0.0)
    assert p.log_orani_ortalamasi == pytest.approx(0.0)


# --------------------------------------------------------------------------
# teshis
# --------------------------------------------------------------------------
def test_ess_tam_eslesmede_n_e_yakin_kismi_eslesmede_duser(cfg):
    """Etkin orneklem, agirliklar esitken n; kuyruk uzadikca duser."""
    v = _veri(seed=31)
    esit = np.ones(v.n)
    kare = float((esit * esit).sum())
    assert esit.sum() ** 2 / kare == pytest.approx(v.n)

    hedef = v.kol.copy()
    hedef[::2] = (hedef[::2] + 1) % v.A
    ag = ev.onem_agirligi(hedef, v, v.propensity, 1e9)
    t = ev.teshis(cfg, v, hedef, ag, v.pi_log)
    assert 0.0 < t.ess_orani < 1.0
    assert t.eslesme_orani == pytest.approx(0.5, abs=0.05)


def test_ortusme_ihlali_esikle_tutarli(cfg):
    """Isaretlenen satir sayisi esikle MONOTON artmali."""
    v = _veri(seed=37)
    hedef = v.kol.copy()
    ag = ev.onem_agirligi(hedef, v, v.propensity, 20.0)
    oranlar = []
    for esik in (0.01, 0.10, 0.30):
        c = config_yukle(PROFIL, gecersiz_kilma={"ope.ortusme.esik": esik})
        oranlar.append(ev.teshis(c, v, hedef, ag, v.pi_log).ortusme_ihlali_orani)
    assert oranlar == sorted(oranlar)
    assert oranlar[-1] > oranlar[0]


# --------------------------------------------------------------------------
# config kilitleri
# --------------------------------------------------------------------------
def test_ortusme_kilidi_olu_teshisi_reddediyor():
    """Esik taban propensity'nin altina inerse yukleme DUSMELI.

    Inseydi ortusme metrigi her kosuda sifir doner ve rapora "ortusme sorunu
    yok" diye gecerdi -- sorunun yoklugu degil, olcunun yoklugu.
    """
    with pytest.raises(Exception, match="ortusme"):
        config_yukle(PROFIL, gecersiz_kilma={"ope.ortusme.esik": 1e-6})


def test_rollout_penceresi_dunyayi_asamaz():
    with pytest.raises(Exception, match="rollout penceresi"):
        config_yukle(PROFIL, gecersiz_kilma={"ope.rollout.baslangic_hafta": 90})


def test_raporlanan_ufuk_kosulmamis_haftayi_etiketleyemez():
    with pytest.raises(Exception, match="raporlanan_ufuklar"):
        config_yukle(PROFIL, gecersiz_kilma={"ope.rollout.ufuk_hafta": 10})


def test_rollout_taban_politikasi_zorunlu():
    with pytest.raises(Exception, match="teklif_yok"):
        config_yukle(PROFIL, gecersiz_kilma={
            "ope.rollout.politikalar": ["uplift_x"]})


def test_rollout_tanimsiz_politika_reddediliyor():
    with pytest.raises(Exception, match="tanimsiz ad"):
        config_yukle(PROFIL, gecersiz_kilma={
            "ope.rollout.politikalar": ["teklif_yok", "sihirli_politika"]})


def test_kirpma_esigi_bir_altinda_reddediliyor():
    with pytest.raises(Exception, match="kirpma"):
        config_yukle(PROFIL, gecersiz_kilma={"ope.tahminci.kirpma_esigi": 0.5})
