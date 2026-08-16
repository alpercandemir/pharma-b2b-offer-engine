"""Seed disiplini.

CLAUDE.md 5: "Her kosu seed'li ve tekrar uretilebilir. Iki kez calistirinca
ayni sonuc."

Tek bir global RNG yeterli DEGIL: bir asamanin cekilis sayisi degisince
sonraki asamalarin akisi kayar ve "sadece SKU sayisini degistirdim, eczane
persona'lari da degisti" tipi sahte farklar dogar. Bu sweep karsilastirmalarini
kirletir.

Cozum: her asama kendi bagimsiz Generator'unu isimden turetir.
    seed(asama) = SHA256(temel_seed | asama_adi)
Asamalar birbirinin akisini bozmaz.
"""

from __future__ import annotations

import hashlib

import numpy as np


class SeedBankasi:
    def __init__(self, temel_seed: int) -> None:
        self.temel_seed = int(temel_seed)
        self._verilen: dict[str, int] = {}

    def tohum(self, asama: str) -> int:
        ham = f"{self.temel_seed}|{asama}".encode("utf-8")
        deger = int.from_bytes(hashlib.sha256(ham).digest()[:8], "big")
        # numpy SeedSequence 64-bit isaretsiz kabul eder.
        self._verilen[asama] = deger
        return deger

    def uretec(self, asama: str) -> np.random.Generator:
        return np.random.default_rng(self.tohum(asama))

    @property
    def verilen(self) -> dict[str, int]:
        return dict(self._verilen)


def agirlikli_secim(
    rng: np.random.Generator, secenekler: list, agirliklar: list[float], boyut: int
) -> np.ndarray:
    p = np.asarray(agirliklar, dtype=float)
    p = p / p.sum()
    idx = rng.choice(len(secenekler), size=boyut, p=p)
    return np.asarray(secenekler)[idx]
