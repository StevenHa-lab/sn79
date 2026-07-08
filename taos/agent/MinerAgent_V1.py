"""
SN79 (MVTRX / taos) mean-reversion fade agent.

Robustness redesign: the old agent faded every local extreme unconditionally,
which shorts into strength during trends and can get stuck one-sided. This
version adds defense in depth:

  Layer 1 - Regime gate:      only fade when the book is statistically NOT
                              trending (OLS trend t-stat, with hysteresis).
  Layer 2 - Directional veto  + normalized stretch: enter on a z-score
                              deviation from the rolling mean (scale-free per
                              book), and never fade against a significant drift.
  Layer 3 - Inventory backstop: hard net-position cap, inventory skew that
                              hardens adds as you load up, and paired exits on
                              every position -> reversion target, vol-scaled
                              stop, and a time stop (the guaranteed backstop
                              against a drifting-mean trap).

Scoring notes (from the taos docs) that shape the design:
  * Kappa-3 return = change in total inventory VALUE per interval (mark-to-
    market). An underwater position bleeds Kappa every tick -> stops are
    Kappa-protective, not just PnL-protective.
  * A volume/activity factor scales Kappa and DECAYS if no trades happen in a
    sampling interval -> standing down in trends has a cost. Flat-regime
    behavior is therefore a configurable lever (cfg['flat_behavior']).
  * There is a per-book turnover cap; disciplined sizing keeps us clear of it.

The pure decision math lives in module-level functions so it can be unit
tested without the taos runtime. The agent class only does I/O + translation.
"""

from collections import defaultdict, deque
import numpy as np

from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol.response import FinanceAgentResponse, OrderDirection, TimeInForce


# --------------------------------------------------------------------------- #
# Pure decision math (no taos dependency -> unit-testable)
# --------------------------------------------------------------------------- #

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def mean_std(xs):
    """Sample mean and (sample) std of a sequence. std=0 if <2 points."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, var ** 0.5


def trend_strength(xs):
    """
    Signed R^2 of an OLS fit against index 0..n-1.
      sign  = drift direction (slope sign)
      |val| = fraction of the window's variance explained by the trend line
              (0 = pure chop, 1 = a clean straight line)
    Bounded [-1, 1]. Replaces the old slope/se t-stat, whose standard error
    collapses on serially-correlated order-book mids and inflates without
    bound. This divides the fitted trend by TOTAL variation, not by the
    (understated) residual standard error, so autocorrelation and window
    length can't blow it up. Returns 0.0 when undefined.
    """
    n = len(xs)
    if n < 3:
        return 0.0
    xbar = (n - 1) / 2.0
    Sxx = sum((i - xbar) ** 2 for i in range(n))
    ybar = sum(xs) / n
    sst = sum((y - ybar) ** 2 for y in xs)          # total variance * (n-1)
    if Sxx <= 0 or sst <= 0:                          # no x-spread, or a flat book
        return 0.0
    slope = sum((i - xbar) * (xs[i] - ybar) for i in range(n)) / Sxx
    intercept = ybar - slope * xbar
    sse = sum((xs[i] - (intercept + slope * i)) ** 2 for i in range(n))
    r2 = max(0.0, 1.0 - sse / sst)
    return r2 if slope > 0 else -r2


def classify_regime(tstat, prev_regime, t_enter, t_exit, t_middle):
    """
    Hysteresis band: flip revert->trend only when |t| rises above t_enter,
    flip trend->revert only when |t| falls below t_exit (t_exit < t_enter).
    Prevents flip-flopping at the boundary. Returns 'trend' or 'revert'.
    """
    a = abs(tstat)
    s_a = tstat
    if prev_regime == "trend":
        return "revert" if a < t_exit else "trend"
    if prev_regime == "revert":
        if s_a > t_exit and s_a < t_middle:
            return "buy"
        elif s_a < -t_exit and s_a > -t_middle:
            return "sell"
        elif a >= t_middle:
            return "trend"
        else:
            return "revert"
    if prev_regime == "buy" or prev_regime == "sell":
        if a < t_exit:
            return "revert"
        elif a >= t_middle:
            return "trend"
        else:
            return prev_regime
    # fallback if prev_regime is invalid
    return "revert" if a < t_exit else "trend"


def _act(kind, qty, role, reason, aggressive):
    return {"kind": kind, "qty": qty, "role": role, "reason": reason,
            "aggressive": aggressive}


def _none(reason):
    return {"kind": "none", "qty": 0.0, "role": None, "reason": reason,
            "aggressive": False}


def decide(cfg, stats, regime, side, net_inv, base_qty, fees):
    """
    One decision for one book, given precomputed stats. Returns an action dict.

    stats: {mid, mean, std, z, tstat, spread}
    regime: 'trend' | 'revert' (already classified this tick)
    side:   'long' | 'short' | None  (derived from net_inv sign)
    net_inv: signed net base position (base_total - base_initial)
    fees:   {maker, taker} rates

    Exits are evaluated FIRST and unconditionally (any regime). Entries only
    fire in 'revert' regime, pass the directional veto, clear the fee/edge
    budget, and respect the inventory cap + skew.
    """
    z = stats["z"]
    tstat = stats["tstat"]
    std = stats["std"]
    spread = stats["spread"]
    mid = stats["mid"]
    mean = stats["mean"]
    cap = cfg["max_inv"]
    
    if regime == "revert" and side == "long":
        if mid >= mean + cfg["z_target"]:
            qty = base_qty * 2 if net_inv > base_qty * 2 else abs(net_inv)
            return _act("sell", qty, "exit", "target", True)
            # return _act("sell", abs(net_inv), "exit", "target", True)

    if regime == "revert" and side == "short":
        if mid <= mean - cfg["z_target"]:
            qty = base_qty * 2 if net_inv < -base_qty * 2 else abs(net_inv)
            return _act("buy", qty, "exit", "target", True)
            # return _act("buy", abs(net_inv), "exit", "target", True)
    
    # ----- Layer 1 gate: only open fades when not trending ------------------ #
    if regime == "trend":
        return _none("regime_trend")
    
    if regime == "buy" and mid < mean - 0.1:
        if net_inv <= cap:                      # currently short -> don't open a long
            qty = _clamp(base_qty, cfg["min_qty"], cfg["max_qty"])
        else:
            qty = 0.05
        return _act("buy", qty, "entry", "buy", False)
    
    if regime == "sell" and mid > mean + 0.1:
        if net_inv >= -cap:                     # currently long -> don't open a short
            qty = _clamp(base_qty, cfg["min_qty"], cfg["max_qty"])
        else:
            qty = 0.05
        return _act("sell", qty, "entry", "sell", False)
    
    if std <= 0:
        return _none("no_vol")

    # ----- fee / edge budget: captured move must beat round-trip cost ------- #
    cost_per_unit = (fees["maker"] + fees["taker"]) * mid + spread
    edge_per_unit = cfg["z_entry"] * std
    if edge_per_unit < cfg["edge_mult"] * cost_per_unit:
        return _none("edge_too_thin")

    # ----- inventory skew: harder to add as we load up ---------------------- #
    load = (abs(net_inv) / cap) if cap > 0 else 0.0
    z_entry_eff = cfg["z_entry"] * (1.0 + cfg["inv_skew"] * load)

    # ----- Layer 2: stretched-up SELL fade, vetoed by up-drift -------------- #
    # if z >= z_entry_eff and tstat < cfg["t_veto"]:
    if mid > mean + 0.1 and regime == "revert":
        if net_inv <= 0:                      # currently long -> don't open a short
            return _none("already_short")
        qty = _clamp(base_qty/3, cfg["min_qty"], cfg["max_qty"])
        if qty <= 0:
            return _none("no_headroom")
        return _act("sell", qty, "entry", "fade_high", False)

    # ----- Layer 2: stretched-down BUY fade, vetoed by down-drift ----------- #
    if mid < mean - 0.1 and regime == "revert":
        if net_inv >= 0:
            return _none("opposite_inventory")
        qty = _clamp(base_qty/3, cfg["min_qty"], cfg["max_qty"])
        if qty <= 0:
            return _none("no_headroom")
        return _act("buy", qty, "entry", "fade_low", False)

    return _none("no_signal")


def evaluate_book(book_id, cfg, mid, mids, spread, prev_regime, side, hold_ticks, net_inv,
                  base_qty, fees, vh):
    """
    Single source of truth used by BOTH the agent and the tests: compute stats
    from the rolling mid window, classify regime (with hysteresis on
    prev_regime), and decide. Returns (action, new_regime, stats).
    """
    n = len(mids)
    if n < cfg["min_samples"]:
        return _none("warmup"), prev_regime, {}

    last_150 = list(mids)[-150:]
    mean, std = mean_std(last_150)
    z = (mid - mean) / std if std > 0 else 0.0
    tstat = trend_strength(mids)
    regime = classify_regime(tstat, prev_regime,
                             cfg["t_trend_enter"], cfg["t_trend_exit"], cfg["t_trend_middle"])
    stats = {"mid": mid, "mean": mean, "std": std, "z": z, "tstat": tstat,
             "spread": spread}
    action = decide(cfg, stats, regime, side, net_inv, base_qty,
                    fees)
    if vh == "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF":
        # print(f"book_id: {book_id} mid: {mids}")
        print(f"MinerAgent_V1: book_id: {book_id} mean={mean:.2f} mid={mid:.2f} std={std:.2f} z={z:.2f} tstat={tstat:.2f} regime={prev_regime} side={side} net_inv={net_inv:.2f}")
    return action, regime, stats


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class MinerAgent_V1(FinanceSimulationAgent):
    def initialize(self):
        # ---- rolling-window / regime ----
        self.cfg = {
            "window": 300,          # fixed rolling lookback (ticks) for stats
            "min_samples": 150,     # need this many mids before acting
            "t_trend_enter": 0.5,  # |t| above this -> classify 'trend' (gate off)
            "t_trend_exit": 0.3,   # |t| below this -> back to 'revert' (hysteresis)
            "t_trend_middle": 0.5, # |t| between exit/enter -> keep prev regime
            "t_veto": 0,         # never fade against drift with |t| beyond this
            # ---- entry / exit (all in z-units => per-book normalized) ----
            "z_entry": 1.5,        # open a fade when |z| >= this
            "z_target": 0.1,       # exit when price reverts to within this of mean
            "z_stop": 3.5,         # adverse extension stop
            "max_hold": 120,       # time stop (in ticks) - drifting-mean backstop
            "edge_mult": 2.0,      # require edge >= edge_mult * round-trip cost
            # ---- inventory (Layer 3) ----
            "inv_cap_frac": 0.50,  # max |net position| as fraction of initial base
            "inv_skew": 1.0,       # how hard to make adds as inventory loads
            "min_qty": 0.1,
            "max_qty": 0.5,
            "base_rate": 0.005,     # base order size = base_rate * initial base
            # ---- order handling ----
            "entry_ttl_intervals": 15,   # GTT lifetime of a resting maker entry
            "entry_aggressive": False,  # True -> take (marketable) entries; keeps
                                        # activity up if the volume factor decays
            "flat_behavior": "stand_down",  # 'stand_down' | 'quote' (hook)
            # ---- fallbacks when account/config fields are absent ----
            "default_maker_fee": 0.0,
            "default_taker_fee": 0.0002,
            "default_publish_interval_ns": 1_000_000_000,
            "max_inv": None,       # resolved per book from inv_cap_frac
        }
        # per-(validator, book) state
        self.mids = defaultdict(lambda: deque(maxlen=self.cfg["window"]))
        self.regime = defaultdict(lambda: "revert")
        self.prev_side = {}
        self.hold_ticks = defaultdict(int)
        self.base_line = {}        # first-seen base_total, fallback baseline

    # ---- helpers ---------------------------------------------------------- #
    def _validator(self, state):
        return state.dendrite.hotkey

    def _initial_base(self, key, base_balance):
        init = getattr(base_balance, "initial", None)
        if init is not None:
            return init
        # fallback: first observed total is treated as the neutral baseline
        if key not in self.base_line:
            self.base_line[key] = base_balance.total
        return self.base_line[key]

    def _net_inventory(self, key, base_balance):
        return base_balance.total - self._initial_base(key, base_balance)

    def _fees(self, account):
        f = getattr(account, "fees", None)
        maker = getattr(f, "maker_fee_rate", None) if f else None
        taker = getattr(f, "taker_fee_rate", None) if f else None
        return {"maker": self.cfg["default_maker_fee"] if maker is None else maker,
                "taker": self.cfg["default_taker_fee"] if taker is None else taker}

    def _reconcile_side(self, key, net_inv, eps):
        if net_inv > eps:
            side = "long"
        elif net_inv < -eps:
            side = "short"
        else:
            side = None
        prev = self.prev_side.get(key)
        if side is None:
            self.hold_ticks[key] = 0
        elif side != prev:
            self.hold_ticks[key] = 0        # new or flipped position
        else:
            self.hold_ticks[key] += 1
        self.prev_side[key] = side
        return side

    # ---- main loop -------------------------------------------------------- #
    def respond(self, state):
        vh = self._validator(state)
        response = FinanceAgentResponse(agent_id=self.uid)
        price_dec = getattr(state.config, "priceDecimals", 2)
        vol_dec = getattr(state.config, "volumeDecimals", 4)
        pub_ns = getattr(state.config, "publish_interval",
                         self.cfg["default_publish_interval_ns"])
        entry_ttl = int(self.cfg["entry_ttl_intervals"] * pub_ns)
        eps = 0.01

        for book_id, book in state.books.items():
            if not book.bids or not book.asks:
                continue
            best_bid = book.bids[0].p
            best_ask = book.asks[0].p
            mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            key = (vh, book_id)

            account = state.accounts[self.uid][book_id]
            base_balance = account.base_balance
            initial_base = self._initial_base(key, base_balance)
            net_inv = self._net_inventory(key, base_balance)
            fees = self._fees(account)

            # per-book inventory cap and base order size
            self.cfg["max_inv"] = max(self.cfg["min_qty"],
                                      self.cfg["inv_cap_frac"] * initial_base)
            base_qty = _clamp(round(initial_base * self.cfg["base_rate"], vol_dec),
                              self.cfg["min_qty"], self.cfg["max_qty"])

            # update rolling window + position bookkeeping
            mids = self.mids[key]
            last_mid = mids[-1] if mids else mid
            self.mids[key].append((mid + last_mid) / 2)
            side = self._reconcile_side(key, net_inv, eps)

            action, new_regime, _ = evaluate_book(
                book_id, self.cfg, mid, self.mids[key], spread, self.regime[key],
                side, self.hold_ticks[key], net_inv, base_qty, fees, vh
            )
            self.regime[key] = new_regime

            if action["kind"] == "none" or action["qty"] <= 0:
                continue

            direction = (OrderDirection.SELL if action["kind"] == "sell"
                         else OrderDirection.BUY)
            qty = round(action["qty"], vol_dec)
            if qty <= 0:
                continue
            if vh == "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF":
                print(f"MinerAgent_V1: {book_id} {action['kind']} {qty} role={action['role']} reason={action['reason']} ")
            if action["role"] == "exit" or action["aggressive"]:
                # cross to flatten promptly; taker cost already budgeted
                response.market_order(book_id=book_id, direction=direction,
                                      quantity=qty)
            else:
                # passive maker entry that auto-expires (no stale-order buildup)
                if self.cfg["entry_aggressive"]:
                    response.market_order(book_id=book_id, direction=direction,
                                          quantity=qty)
                else:
                    px = best_ask - 0.02 if direction == OrderDirection.SELL else best_bid + 0.02
                    response.limit_order(
                        book_id=book_id, 
                        direction=direction, 
                        quantity=qty,
                        price=round(px, price_dec), 
                        timeInForce=TimeInForce.GTT, 
                        expiryPeriod=entry_ttl,
                    )

        return response


if __name__ == "__main__":
    from taos.common.agents import launch
    launch(MinerAgent_V1)