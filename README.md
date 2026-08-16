# Pharma B2B Offer Engine

A decision engine for a B2B pharmaceutical marketplace: for every (pharmacy, SKU) pair it decides
**whether to make an offer now**, and **with which free-goods ratio and payment term** — under finite
stock, batch-level expiry, and hard regulatory constraints. Every decision is measurable offline and
explainable in natural language.

This is a **learning substrate, not a production system.** It runs entirely on synthetic data, because
the whole point is that the ground truth is known: you can check whether your off-policy evaluation was
telling the truth, which is a luxury you never have in the real world.

---

## Table of contents

- [What it solves](#what-it-solves)
- [Design decisions](#design-decisions)
- [What is implemented](#what-is-implemented)
- [Tech stack](#tech-stack)
- [Getting started](#getting-started)
- [Running the engine](#running-the-engine)
- [Repository layout](#repository-layout)
- [Configuration and tuning](#configuration-and-tuning)
- [Reproducibility](#reproducibility)
- [Roadmap and known gaps](#roadmap-and-known-gaps)
- [Documentation map](#documentation-map)
- [Caveats](#caveats)

---

## What it solves

A pharmaceutical wholesaler serves several hundred pharmacies with thousands of SKUs. Every week it has
to decide where to push stock. Four things make this harder than a generic recommender:

| Problem | Why the obvious answer fails |
|---|---|
| **Prices are regulated.** | You cannot discount by percentage. Real leverage is free goods (10+1, 10+2) and payment terms — a two-dimensional, discrete action space. |
| **Stock is a shared, finite resource.** | Ranking alone recommends the same scarce SKU to 400 pharmacies. Scoring and allocation have to be separate layers. |
| **Expiry lives on lots, not SKUs.** | After a certain date a unit's value turns *negative* (disposal + return handling). The objective flips from margin maximization to loss minimization. |
| **Demand is intermittent and macro-driven.** | Most (pharmacy, SKU, week) cells are zero. Reference-FX updates trigger anticipatory stockpiling worth 3–5× baseline demand. |

The engine answers all four in one pipeline: depletion forecasting → candidate generation → hard
constraint veto → uplift (CATE) scoring → LP allocation with shadow prices → off-policy evaluation →
LLM narrative and scenario commentary.

Its most interesting output is not the recommendations. It is the ability to answer:

> *"Offline evaluation said +12%, the closed loop delivered −3%. Why?"*

and to decompose the answer into variance, weight clipping, propensity error, overlap violation, and
horizon effects — each as a measured number rather than a guess.

## Design decisions

Nine decisions are fixed and drive the entire architecture (full rationale in [`SPEC.md`](SPEC.md) §1):

| # | Decision |
|---|---|
| D1 | The action space is `(free_goods_ratio, payment_term_days)` — **never a percentage discount** |
| D2 | "Don't offer what they just bought" is a **prediction problem** (when will their stock run out?), not an exclusion rule |
| D3 | FX is a **scenario input**, not a forecast target |
| D4 | The real macro signal is the **expectation of a reference-rate update**, not the market rate |
| D5 | Scoring and allocation are **separate layers** |
| D6 | The constraint layer holds **veto power over the ML score** — red/green prescriptions are hard constraints, not soft penalties |
| D7 | Selection **propensity is logged on every impression** — without it, off-policy evaluation is impossible |
| D8 | The LLM is **not in the decision path**; it handles orchestration, explanation and scenario commentary |
| D9 | Expiry clearance is not a separate engine, but a **regime of the allocation layer** (the same LP with a sign flip) |

## What is implemented

All seven milestones are complete. Each has a verification script, a sweep, and a report under
[`reports/`](reports/).

| # | Scope | Headline result | Report |
|---|---|---|---|
| **M1** | Simulator and ground truth — latent consumption, personas, seasonality, regime events, lot-level stock with FEFO | 200 pharmacies × 300 SKUs × 104 weeks; intermittent demand, latent share-of-wallet, expiry tolerance per pharmacy | [m1.md](reports/m1.md) |
| **M2** | Depletion model (D2) — consumption inference, on-hand estimation, discrete-time hazard | Hazard model vs. "bought in the last 30 days" rule, measured side by side with calibration curves | [m2.md](reports/m2.md) |
| **M3** | Candidate generation + constraint layer (D6) | 5 generators; 8 veto reasons; the veto provably cuts the top of the ML score | [m3.md](reports/m3.md) |
| **M4** | Uplift / CATE and action selection (D1) | The money is not in "level vs. uplift" (+4.1k TRY) but in whether margin is in the objective at all (**214k TRY**) | [m4.md](reports/m4.md) |
| **M5** | Allocation LP + expiry regime (D5 + D9) | Shadow prices go negative on **9.8%** of lots in clearance regime; targeted clearance beats blind discounting on both disposal and returns | [m5.md](reports/m5.md) |
| **M6** | Off-policy evaluation and the tuning loop | DR catches the oracle in 8/8 policies; bias decomposition has **exactly zero residual**; the `lp` policy flips sign from **+36.4% @4 weeks to −20.4% @52** | [m6.md](reports/m6.md) |
| **M7** | LLM layer (D8) — scenario commentary, KAM briefing, eval harness | 10/10 injected mutations caught, 0 false alarms on clean cases; **live model leg not yet run** | [m7.md](reports/m7.md) |

**M6 is the point of the project.** Everything before it is setup.

Current state: **213 tests passing**, 71 Python modules (~22.5k lines), 354 configurable knobs, all
catalogued in [`TUNING.md`](TUNING.md).

## Tech stack

| Layer | Choice | Note |
|---|---|---|
| Language | Python 3.11+ | |
| Package manager | [`uv`](https://docs.astral.sh/uv/) | `uv.lock` is committed |
| Data | `polars`, `pyarrow` | All intermediate data is Parquet |
| Config | `pydantic` v2 + YAML | `extra="forbid"`, no Python-side defaults — a missing knob raises, it does not fall back |
| ML | `scikit-learn` (`HistGradientBoosting*`) | LightGBM was tried and reverted: it needs `brew install libomp` on macOS. Same algorithm family, one fewer system dependency — rationale in [`pyproject.toml`](pyproject.toml) |
| Optimization | `scipy.optimize.linprog` (HiGHS) | Chosen over PuLP because HiGHS returns the dual directly, and shadow prices are central to M5 |
| Plots | `matplotlib` | Written to `reports/figures/` |
| LLM | `anthropic` (**optional extra**) | The harness tests recorded conversations, so the full test suite runs without the package or an API key |
| Tests | `pytest` | |

No notebooks, by design. Everything is an executable script plus config.

## Getting started

**Requirements:** Python 3.11+ and `uv`.

```bash
uv sync --python 3.12
```

Optional, only needed for live LLM calls:

```bash
uv sync --extra llm
```

Then generate the world and verify the baseline:

```bash
uv run python -m scripts.generate_world --profil full --kosu full
```

```bash
uv run python -m scripts.verify_m1 --kosu full
```

Two scale profiles are available in [`config/profiles/`](config/profiles/):

- `full` — 200 pharmacies × 300 SKUs × 104 weeks. This is the profile every reported number uses.
- `fast` — 60 × 100 × 104. For sweeps, iteration and tests. **It does not satisfy the M1 exit criterion**; never quote a headline number from it.

## Running the engine

Note: CLI flags and module internals are in Turkish (`--kosu` = run, `--asama` = stage, `--profil` =
profile, `--sabit` = fixed override). Documentation and reports are Turkish as well.

### Verify a milestone

Each script re-checks that milestone's exit criterion and regenerates its figures.

```bash
uv run python -m scripts.verify_m6 --kosu full
```

Swap `m6` for `m1`…`m7`. `verify_m6` runs 13 checks and produces 4 plots; the others are similar.

### Reference run

`experiments/run.py` executes one config end to end and writes a metric set to
`experiments/runs/<name>/`.

```bash
uv run python -m experiments.run --profil full --asama m4,m5,m6 --ad m6_full --veri-tut
```

Stages are `m2` … `m7`; `--veri-tut` keeps the generated world on disk.

### Sweep a knob

This is where the learning happens. Any config path can be swept across values and seeds, in parallel.

```bash
uv run python -m experiments.sweep --knob tahsis.temizlik.tetik_gun --values 60,90,120,180 --seeds 3 --asama m5 --profil full --sabit tahsis.senaryo.miad_hizlandirma_gun=60
```

Output: a metric table per value plus a plot showing the direction of movement. Use `--sabit
path=value` to pin other knobs, and `--isci N` to set the worker count.

### Compare two runs

```bash
uv run python -m experiments.compare --a m5_full --b m6_full
```

### LLM regression harness

Runs the eval cases against recorded conversations — no API key required.

```bash
uv run python -m harness.run --kosu full
```

### Tests

```bash
uv run pytest -q
```

The full suite takes about 8 minutes.

## Repository layout

```
sim/          Synthetic world — the ground truth the models never see
              world.py, events.py, response.py, rollout.py, lots.py, pharmacies.py, products.py
data/         What the world exposes: observable/ (orders, shipments, lots, prices)
              vs ground_truth/ (latent SOW, true consumption, real events)
features/     Point-in-time feature builders with a leakage guard
models/       depletion.py (hazard), uplift.py (T- and X-learner)
policy/       candidates.py, constraints.py (hard veto), scorer.py, allocate.py (LP), bandit.py
eval/         ope.py (IPS/SNIPS/DR), oracle.py (true counterfactual), report.py, metrics.py
agent/        narrative.py, scenario.py, tools.py, client.py — the LLM layer (D8)
harness/      cases.yaml, run.py, denetim.py (auditors), mutasyon.py (mutation injection)
core/         config.py (pydantic schema + mechanical locks), rng.py, io.py
config/       15 YAML files + profiles/ — every knob in the system
experiments/  run.py, sweep.py, compare.py, runs/
scripts/      generate_world.py, verify_m1.py … verify_m7.py
tests/        14 test modules, 213 tests
reports/      m1.md … m7.md + figures/
```

## Configuration and tuning

Two rules govern every number in this codebase:

1. **No magic numbers.** A number is either a knob in `config/` with a row in `TUNING.md`, or a domain
   constant that stays in the code with a one-line comment explaining why it is constant. There is no
   third option.
2. **Every knob is accountable.** `tests/test_config.py::test_tuning_md_her_knobu_kapsiyor` mechanically
   asserts that every knob has a `TUNING.md` entry. A knob without a row is a magic number and gets
   removed.

Each `TUNING.md` entry states the mechanism, the default and sensible range, what happens when you turn
it up or down (with the metric), the observable symptom of a wrong setting, interacting knobs, and a
runnable diagnostic command.

The schema also enforces **mechanical locks** in `core/config.py` — combinations that would silently
produce meaningless output are rejected at load time. For example, the M6 overlap threshold cannot go
below the logging policy's base propensity (no row could ever be flagged, so the diagnostic would die
quietly), and the rollout window cannot exceed the world length (the table would say "52 weeks" while
showing 36). These guard against the worst failure mode: code runs, produces a number, number is wrong.

[`notes/predictions.md`](notes/predictions.md) is the calibration log — every knob exercise is recorded
as *predict first, then run*. The entries where the prediction was wrong are the most valuable part of
the repository.

## Reproducibility

- Every run is seeded and reproducible; running twice gives identical results.
- The config is hashed. `dunya_hash` covers only the world-defining sections, so it has stayed
  `9d6191c761d43e52` from M1 through M7 — proving that M1–M6 numbers remain valid as later milestones
  added config blocks.
- Feature builders are point-in-time correct, with leakage guards under test.
- Base seed `20260812`; the full world generates in about 3 seconds.

## Roadmap and known gaps

Every milestone report closes with a *debt* section. The items below are the ones still open, ordered by
how much they affect the trustworthiness of the reported numbers. Each is traceable to the report that
raised it.

> Per [`CLAUDE.md`](CLAUDE.md) §4, anything that touches decisions D1–D9 or changes the config contract
> requires explicit approval before implementation. Those items are marked **needs approval**.

### Priority 1 — measurement integrity

These do not add features; they establish how much the current headline numbers can be trusted.

| Gap | Why it matters | Suggested fix | Source |
|---|---|---|---|
| **No seed repetition on the `full` rollout** | §4 of the M6 report — the largest numbers in the project — comes from a single world. The uncertainty of the @52 deltas is simply unknown. On `fast`, the standard error of TRY metrics is of the same order as the mean. | Run the `full` rollout with `--seeds 3` (~8 min per seed). Until then, treat @52 figures as direction, not magnitude. | m6 §8 #2 |
| **Rollout branches inside the CATE training window** | The rollout branches at week 52; the T/X learners were trained on weeks 38–87 — a 36-week overlap. Closed-loop policy performance may look better than it is. Reported, never measured. | Leakage-free control run: `baslangic_hafta=88, ufuk_hafta=16`, then compare the @16 columns. The config already supports it; it has not been run. | m6 §8 #1 |
| **Terminal stock bias is bounded, not corrected** | 73% of `uplift_x`'s +3.17M TRY at 52 weeks is margin booked on goods still sitting on pharmacy shelves. It is reported as an upper bound, deliberately not netted out. | A 156-week profile: 52 weeks of offers + 52 weeks of decay, so the bias is measured rather than bounded. One profile file; it changes the world, so it must live in a separate profile. | m6 §8 #3 |

### Priority 2 — policy and model gaps with a measured cost

| Gap | Measured cost | Suggested fix | Source |
|---|---|---|---|
| **Allocation LP is single-period** | This is no longer a hypothesis: the `lp` policy goes **+36.4% @4 weeks → −3.7% @26 → −20.4% @52**. It exhausts the cheapest-opportunity-cost lots early, and disposal ends up 54% above `uplift_x`. | Rolling horizon, or carry the lot constraint's dual across weeks as an inter-period "stock price". | m5 §8 #3, cost measured in m6 §4.1 |
| **`hiz_telafi_katsayisi` is still 1.0** | Open since M3 and not closed in M4, M5, M6 or M7. The measured correct value is ~2.6. M7 made it worse: the anticipation multiplier scales the same quantity formula, so two multipliers now stack. | Cross-sweep `hiz_telafi_katsayisi` × `guvenlik_katsayisi`. | m3 → m7 §8 #6 |
| **Negative-margin arms stay in the action space** — **needs approval** | Under a hard FX regime the objective loses meaning: the policy optimizes for *not selling*. The deepest arm has a per-acceptance margin of **−128 TRY**. | Two options, both touching the M4 objective: (a) exclude arms where `p_a < p_0`, (b) clip incremental value at `max(0, ·)`. (a) is more honest — it closes a lever that is not really in the action space. | m7 §8 #3 |
| **Anticipation does not change candidate ranking** | The multiplier only moves quantities and the constraint layer. In reality, stockpiling expectations change *which* products are wanted — acute vs. chronic behave differently (SPEC §2.3). | A category-based anticipation multiplier instead of regime-linked mixture weights. Does not touch D1–D9, but does grow the config contract. | m7 §8 #2 |

### Priority 3 — LLM layer

| Gap | Why it matters | Suggested fix | Source |
|---|---|---|---|
| **The live model leg has never been run** | The harness proves the auditors catch broken output; it does **not** prove they stay quiet on real model prose. The false-alarm rate against genuine output is unknown. This is M7's one open leg. | `uv sync --extra llm` + `ANTHROPIC_API_KEY`, then `uv run python -m harness.run --kosu full --canli --kaydet sablon_konusma_full.json`. Freeze the recording as a fixture and add it to the regression suite. ~1 minute. | m7 §8 #1 |
| **Scenario commentary runs at a single origin** | The variability of the regime difference is unknown. | `senaryolari_kos(cfg, m4, t)` already takes an origin parameter — run it across M4's three origins. About half an hour of work. | m7 §8 #4 |

### Priority 4 — engineering and instrumentation

| Gap | Why it matters | Suggested fix | Source |
|---|---|---|---|
| **The `tepki` block is outside `DUNYA_BOLUMLERI`** — **needs approval** | Open since M4. Response knobs now genuinely change the world in the rollout, but `dunya_hash` is blind to them. | Adding `tepki` to `DUNYA_BOLUMLERI` would invalidate the M1–M5 hashes; a separate `rollout_hash` field is less destructive. Changes the config contract. | m4 → m7 §8 #7 |
| **Sweep's console table and `ozet.csv` are maintained by hand** | Two hard-coded metric lists (`izlenen`, `_sd`). Forgetting to update them is silent — the sweep just prints an unremarkable average. This actually happened in M7. | Derive both lists from `ONE_CIKAN` by filtering on the stage prefix. New milestones would then work with no manual update. | m7 §8 #8 |
| **`kesif_orani` × frequency-cap cross-sweep never run** | Narrowing exploration 10× moved nothing, because the frequency cap forces 71% of decisions onto arm 0, whose propensity is exploration-independent. D7's insurance policy was never actually stress-tested. | Sweep both knobs together. Loosening the cap should make exploration a first-order knob again. | m6 §8 #4 |
| **Policy belief vs. world truth never measured** | If `fiyat_gecis_katsayisi` is wrong, the scenario layer recommends the wrong regime — and the belief and the truth are already two separate knobs. | Cross-sweep `senaryo.rejimler[].fiyat_gecis_katsayisi` × `olay.referans_kur.fiyat_gecis_katsayisi`. | m7 §8 #5 |

### Domain parameters requiring real-world validation

The POC intentionally keeps these parametric rather than accurate (SPEC §8). Before any of this informs
a real decision, confirm against source: TİTCK pricing decree and reference euro rate, SGK Annex-4/A
reimbursement list, wholesaler/pharmacist margin tiers, prescription colour classification and its
promotion restrictions, İTS serialization rules, and the actual mechanics of expiry returns
(wholesaler → manufacturer credit ratio and time window).

## Documentation map

| File | Audience | Purpose |
|---|---|---|
| `README.md` | Everyone | This file — what, why, how to run, what is left |
| [`SPEC.md`](SPEC.md) | Claude Code | Technical specification: design decisions, domain parameters, milestone definitions |
| [`CLAUDE.md`](CLAUDE.md) | Claude Code | Standing rules, read at the start of every session |
| [`TUNING.md`](TUNING.md) | Operators | The knob catalogue — 354 knobs, mechanism, ranges, symptoms, diagnostics |
| [`WORKING_GUIDE.md`](WORKING_GUIDE.md) | Maintainer | Session workflow and the audit checklist for reviewing generated code |
| [`PROMPTS.md`](PROMPTS.md) | Maintainer | Copy-paste milestone commands |
| [`notes/predictions.md`](notes/predictions.md) | Maintainer | Calibration log: predict first, then run |
| [`reports/`](reports/) | Everyone | Per-milestone results, deviations from expectation, simplifications and debts |

Reports and inline documentation are written in Turkish, as is the code's naming convention.

## Caveats

- **Synthetic data only.** No real pharmacy, product, or transaction data is used anywhere.
- **Not a production system.** There is no service layer, no persistence beyond Parquet files, no
  authentication, and no deployment path. It is built to be read, run, and tuned.
- **Domain parameters are illustrative.** They are designed to be *changeable*, not correct — see the
  validation list above.
- **The simulator is deliberately hard.** Demand is intermittent, consumption is stochastic,
  share-of-wallet stays latent, and uplift is heterogeneous across segments. If a model here scores
  suspiciously well, suspect leakage first and simulator design second.
