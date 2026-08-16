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


class SeedBank:
    def __init__(self, temel_seed: int) -> None:
        self.temel_seed = int(temel_seed)
        self._issued: dict[str, int] = {}

    def seed_for(self, stage: str) -> int:
        raw = f"{self.temel_seed}|{stage}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
        # numpy SeedSequence 64-bit isaretsiz kabul eder.
        self._issued[stage] = value
        return value

    def generator(self, stage: str) -> np.random.Generator:
        return np.random.default_rng(self.seed_for(stage))

    @property
    def issued(self) -> dict[str, int]:
        return dict(self._issued)


def weighted_choice(
    rng: np.random.Generator, options: list, weights: list[float], size: int
) -> np.ndarray:
    p = np.asarray(weights, dtype=float)
    p = p / p.sum()
    idx = rng.choice(len(options), size=size, p=p)
    return np.asarray(options)[idx]
