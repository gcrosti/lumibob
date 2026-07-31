# Earnings Events and Catastrophic Pair Losses

**Date:** 2026-07-31
**Scripts:** `tuning/studies/study2_event_retro/` (build_outcomes → fetch_earnings → event_analysis)
**Data:** the three Pass A v3 gate runs (`4f419e` sideways_2022, `4b26c6` bull_2023,
`bcb308` mixed_2023_q4); 175 analyzable round trips, z0.5 take-profit exit,
40-trading-day cap — the same per-pair methodology as
`notebooks/pass_a_v3_score_signal_retroactive.ipynb`, replicated to the decimal
(fold means +21.6 / −22.4 / +8.7 bps; worst −1554; catastrophic share 9.3 / 8.3 / 6.6%).
**Acted on by:** `docs/plans/2026-07-31_composite-score-overhaul.md` (WS2, Study E1/E2).

## Question

The 2026-07-18 exit-redesign update ended on: are the catastrophic pairs
(round trips losing more than 100 bps gross — the ~8% of pairs owning 45–80% of
all losses) identifiable *at entry*? This analysis tests the event hypothesis:
that disasters coincide with earnings announcements on a leg during the hold.

## Findings

### 1. Two-thirds of catastrophic loss magnitude is earnings-adjacent

Of 14 catastrophic round trips, 6 (43%) had an earnings announcement on a leg
inside the holding window; those 6 account for **−2,847 of −4,318 bps = 66% of
all catastrophic losses**. The worst single trade (QRVO/NVDA, −1,554 bps,
entered 2023-02-07) sits across NVDA's 2023-02-22 guidance blow-out.

### 2. The stock-legged disaster class is almost entirely event-driven

| | ≥1 stock leg | earnings in window (ev_any) | median hold |
|---|---|---|---|
| Catastrophic (n=14) | 50% | 43% | — |
| Rest (n=161) | 19% | 4% | — |
| Catastrophic, stock-legged (n=7) | — | **86%** | 25d |
| Rest, stock-legged (n=30) | — | 23% | 8d |

Fisher exact, stock-legged only: OR ≈ 20, p = 0.004. Stock-legged pairs are
over-represented 2.4× in the disaster class overall.

An EDGAR spot-check upgraded the one apparent exception: RGLD/ASA (−170 bps,
"no earnings in window" per Yahoo) had three RGLD 8-Ks inside its window,
including an Item 2.02 interim results release (2023-10-11) and an Item 5.02
officer change (2023-09-18). With item-coded 8-Ks, **7 of 7 stock-legged
disasters had a material disclosure in-window** — and Yahoo's calendar is
demonstrably incomplete, which is why the plan's WS2a makes EDGAR the
production source.

### 3. Earnings exposure is two-sided — the veto ledger

Event-exposed survivors were the biggest winners (+150 bps average). Removing
every event-exposed pair:

| | bps | pairs |
|---|---|---|
| Losses avoided | −3,034 | 9 |
| Wins missed | +1,239 | 4 |
| **Net** | **+1,795** | +10.3 bps/trade pooled |

Per fold: bull_2023 +2,345 (all of the net benefit; contains the −1,554
blow-up), sideways_2022 −270, mixed_2023_q4 −280. In a normal fold the veto
*costs* ~270 bps; it pays for itself when a mega-blow-out lands. Strip
QRVO/NVDA alone and the veto is roughly breakeven.

**Design consequence (decided in the overhaul plan):** event-exposed candidates
are excluded entirely. The event-window wins are surprise-direction luck the
strategy has no edge in, so exclusion is variance reduction at roughly zero
expected cost — the left tail that flips fold means negative is removed, and
the ~270 bps/fold is the accepted insurance premium.

### 4. The fund-only remainder is not earnings-shaped

7 fund-only disasters (−1,301 bps, 30% of catastrophic losses): MXF/MXE (both
Mexico closed-end funds) and VKQ/VMO (muni CEFs, Oct-2023 rate spike) look like
CEF discount dislocations — addressable with daily NAV data (plan WS3).
WCBR/SKYU spans the Feb-2022 Ukraine invasion; that class is macro shock, not
addressable with any filing or NAV data, and falls to position sizing.

## Caveats

- **n = 14 disasters** (7 stock-legged). The composition and loss-share numbers
  are the robust part; the 86% rate has a wide interval.
- **Duration confound:** disasters hold longer by construction (median 25d vs
  8d), and longer windows mechanically catch more earnings — non-catastrophic
  stock pairs held 16–30d show a 40% hit rate (n = 5). Part of the 86% is
  exposure time, not prediction.
- **yfinance coverage:** 182 of 242 symbols are funds with no earnings by
  construction; LBTYK and BATRA have gapped history and were treated as
  unknown, not event-free; dates are day-granular (before/after close not
  modeled).
- **Proxy:** actual announcement dates stand in for what was *expected* at
  entry. Companies pre-announce reporting dates weeks out and rarely move them;
  for a ±window feature the error is negligible, but a live rule uses the
  scheduled calendar, which has its own blind spots (dates published only 2–4
  weeks ahead) — hence the pre-event exit in plan Study E2.
- **Range restriction:** entered pairs only. Rates among all scored candidates
  may differ; Study E1 re-runs this on the full replay pool.
- Single-stock leveraged ETFs (AAPB/AAPU→AAPL, FBL→META) were mapped to their
  underlying's earnings; other derivative structures (preferreds, baby bonds)
  were not classified.
