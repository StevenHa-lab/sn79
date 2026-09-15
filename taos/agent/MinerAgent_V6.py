"""
MinerAgent_V6 -- per-book passive round-trip market maker for SN79 (taos 0.6.0).

Design (derived from the validator's scoring code, see notes in the repo discussion):

* The trading score is 0.79 * kappa + 0.21 * pnl.  Kappa-3 is computed per book on a
  ONE-ENTRY-PER-SIMULATION-SECOND series of realized (FIFO, fee-inclusive) P&L, MAD
  normalised, so absolute size is irrelevant and what counts is how many DISTINCT seconds
  carry a small positive realization, with losses entering as a sum of cubes.
* Every one of the books therefore gets the same treatment: rest one small bid and one
  small ask around the mid, and whenever a lot is filled the opposite quote becomes that
  lot's exit, priced so that the FIFO head closes at a profit after both legs' fees.
* Size is the exchange minimum (0.25 by default).  Inventory is capped per book so no
  lump loss can ever build up; open lots that never reach their exit simply die at the
  daily simulation restart, where the validator discards them unrealized.
* The rolling 24 h per-book notional cap (capital_turnover_cap * miner_wealth, i.e.
  500,000 quote at defaults) is tracked locally, both legs counted, and entries stop at
  a configurable fraction of it so a book is never locked out.
* At most 3 instructions per book per tick (one cancel batch + two placements), well
  inside the validator's limit of 5.  No file I/O, no pandas, no external feeds in the
  hot path; the base class's per-tick debug rendering is bypassed.

All parameters can be overridden with ``--agent.params key=value ...``:

  size            order size in base units                     (default 0.25, floored at min_order_size)
  edge_bps        target round-trip edge, basis points of mid    (default 6.0)
  min_edge_ticks  floor on the round-trip edge, in ticks         (default 4)
  exit_bps        minimum profit on a FIFO-head exit, bps        (default 2.0)
  exit_ticks      floor on that profit, in ticks                 (default 2)
  skew_bps        quote skew per 1.0 unit of inventory, bps      (default 3.0)
  max_inv         inventory cap per book, base units             (default 2.0)
  ttl_s           GTT lifetime of every order, sim seconds       (default 20)
  requote_ticks   re-quote when the desired price moves by more  (default 2)
  cap_frac        fraction of the 24 h notional cap to use       (default 0.85)
  pace_headroom   share of the daily budget allowed ahead of the
                  linear daily pace (0.05 = 5%)                  (default 0.05)
  max_resting     hard cap on my resting orders per book         (default 6)
  log_every       ticks between summary log lines                (default 30)

Pacing: the validator's cap is a rolling 24 h window per book that survives the daily
simulation restart, so the agent keeps its per-book volume history across restarts and only
quotes new entries while the rolling notional is below cap_frac * cap * (day_progress +
pace_headroom).  Exits (quotes that reduce inventory) are always allowed.
"""

import math
from collections import deque

import bittensor as bt

from taos.im.agents import FinanceAgent
from taos.im.protocol.response import OrderDirection, TimeInForce

DAY_NS = 86_400_000_000_000


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _attr(obj, *names, default=None):
    """First present attribute among `names` (wire field or property), else default."""
    for n in names:
        try:
            v = getattr(obj, n)
        except Exception:
            continue
        if callable(v):
            try:
                v = v()
            except Exception:
                continue
        if v is not None:
            return v
    return default


def _f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


class _Lot:
    __slots__ = ("price", "qty", "fee", "ts")

    def __init__(self, price, qty, fee, ts):
        self.price = price
        self.qty = qty
        self.fee = fee      # total fee paid on the open leg for this lot
        self.ts = ts


class _BookState:
    """Per (validator, book) bookkeeping."""
    __slots__ = ("longs", "shorts", "vol", "vol_sum", "realized", "wins", "losses",
                 "n_fills", "last_bid", "last_ask", "rejects")

    def __init__(self):
        self.longs = deque()      # FIFO lots
        self.shorts = deque()
        self.vol = deque()        # (ts, notional) for the rolling 24 h cap
        self.vol_sum = 0.0
        self.realized = 0.0
        self.wins = 0
        self.losses = 0
        self.n_fills = 0
        self.last_bid = None
        self.last_ask = None
        self.rejects = 0

    def inventory(self):
        return sum(l.qty for l in self.longs) - sum(l.qty for l in self.shorts)

    # -- validator-equivalent FIFO matching ---------------------------------- #
    def apply_fill(self, is_buy, qty, price, fee, ts):
        """Mirror the validator's FIFO realized-P&L attribution. Returns realized P&L."""
        pnl = 0.0
        remaining = qty
        fee_per_unit = fee / qty if qty > 0 else 0.0
        book = self.shorts if is_buy else self.longs
        while remaining > 1e-12 and book:
            lot = book[0]
            closed = min(remaining, lot.qty)
            open_fee_share = lot.fee * (closed / lot.qty) if lot.qty > 0 else 0.0
            close_fee_share = fee_per_unit * closed
            if is_buy:      # closing a short: sold high at lot.price, buying back at price
                pnl += (lot.price - price) * closed - open_fee_share - close_fee_share
            else:           # closing a long
                pnl += (price - lot.price) * closed - open_fee_share - close_fee_share
            lot.qty -= closed
            lot.fee -= open_fee_share
            remaining -= closed
            if lot.qty <= 1e-12:
                book.popleft()
        if remaining > 1e-12:
            new = _Lot(price, remaining, fee_per_unit * remaining, ts)
            (self.longs if is_buy else self.shorts).append(new)
        if abs(pnl) > 0:
            self.realized += pnl
            if pnl > 0:
                self.wins += 1
            else:
                self.losses += 1
        self.n_fills += 1
        return pnl

    def add_volume(self, ts, notional):
        self.vol.append((ts, notional))
        self.vol_sum += notional
        self._trim(ts)

    def volume_24h(self, ts):
        self._trim(ts)
        return self.vol_sum

    def _trim(self, ts):
        cutoff = ts - DAY_NS
        while self.vol and self.vol[0][0] < cutoff:
            _, n = self.vol.popleft()
            self.vol_sum -= n
        if self.vol_sum < 0:
            self.vol_sum = 0.0


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class MinerAgent_V6(FinanceAgent):

    # ---- parameters -------------------------------------------------------- #
    def _p(self, name, default):
        v = getattr(self.config, name, None)
        if v is None:
            return default
        try:
            return type(default)(v)
        except Exception:
            return default

    def initialize(self):
        self.size = self._p("size", 0.25)
        self.edge_bps = self._p("edge_bps", 6.0)
        self.min_edge_ticks = self._p("min_edge_ticks", 4)
        self.exit_bps = self._p("exit_bps", 2.0)
        self.exit_ticks = self._p("exit_ticks", 2)
        self.skew_bps = self._p("skew_bps", 3.0)
        self.max_inv = self._p("max_inv", 2.0)
        self.ttl_s = self._p("ttl_s", 20)
        self.requote_ticks = self._p("requote_ticks", 2)
        self.cap_frac = self._p("cap_frac", 0.85)
        self.pace_headroom = self._p("pace_headroom", 0.05)
        self.max_resting = self._p("max_resting", 6)
        self.log_every = self._p("log_every", 30)

        # per validator hotkey -> {book_id: _BookState}
        self.state_by_validator = {}
        self.last_ts = {}
        self.ticks = 0

    # ---- lean per-tick plumbing (bypass the base class debug rendering) ---- #
    def update(self, state):
        self.simulation_config = state.config
        self.accounts = (state.accounts or {}).get(self.uid, {}) or {}
        self.events = (state.notices or {}).get(self.uid, []) or []
        self._exchange_mode = False

    def report(self, state, response):
        # The base implementation logs every instruction at INFO each tick; keep it quiet.
        return

    # ---- helpers ----------------------------------------------------------- #
    def _books(self, vh):
        d = self.state_by_validator.get(vh)
        if d is None:
            d = self.state_by_validator[vh] = {}
        return d

    def _reset_validator(self, vh, reason):
        """Simulation restart: open lots are discarded by the validator, but its 24 h volume
        cap is a rolling window that survives, so keep the volume history."""
        bt.logging.info(f"V6: resetting lots for validator {vh[:8]}... ({reason})")
        for bs in self.state_by_validator.get(vh, {}).values():
            bs.longs.clear()
            bs.shorts.clear()

    @staticmethod
    def _round_down(x, tick):
        return math.floor(x / tick + 1e-9) * tick

    @staticmethod
    def _round_up(x, tick):
        return math.ceil(x / tick - 1e-9) * tick

    def _fees(self, account):
        f = _attr(account, "fees", default=None)
        maker = _f(_attr(f, "maker_fee_rate", "m", default=0.0)) if f is not None else 0.0
        taker = _f(_attr(f, "taker_fee_rate", "t", default=0.00023)) if f is not None else 0.00023
        return maker, taker

    # ---- fills ------------------------------------------------------------- #
    def _process_notices(self, vh, books, ts):
        """Feed this tick's notices into the FIFO ledgers. Detects simulation restarts."""
        restarted = False
        for ev in self.events:
            et = _attr(ev, "type", "y", default="")
            if et in ("EVENT_SIMULATION_START", "ESS"):
                restarted = True
                continue
            if et not in ("EVENT_TRADE", "ET"):
                if et in ("ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT", "ERDPOL",
                          "ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET", "ERDPOM"):
                    b = _attr(ev, "bookId", "b", default=None)
                    if b is not None:
                        bs = books.get(b)
                        if bs is not None:
                            bs.rejects += 1
                    msg = _attr(ev, "message", "m", default="")
                    if self.ticks % 50 == 0:
                        bt.logging.debug(f"V6: order rejected on book {b}: {msg}")
                continue
            b = _attr(ev, "bookId", "b", default=None)
            if b is None:
                continue
            taker = _attr(ev, "takerAgentId", "Ta", default=-1)
            maker = _attr(ev, "makerAgentId", "Ma", default=-1)
            is_taker = taker == self.uid
            is_maker = maker == self.uid
            if not (is_taker or is_maker):
                continue
            side = int(_attr(ev, "side", "s", default=0))
            # validator convention: side is the taker's direction, 0 = buy
            is_buy = (is_taker and side == 0) or (is_maker and side == 1)
            price = _f(_attr(ev, "price", "p", default=0.0))
            qty = _f(_attr(ev, "quantity", "q", default=0.0))
            fee = _f(_attr(ev, "takerFee", "Tf", default=0.0) if is_taker else _attr(ev, "makerFee", "Mf", default=0.0))
            ets = int(_attr(ev, "timestamp", "t", default=ts) or ts)
            if qty <= 0 or price <= 0:
                continue
            bs = books.get(b)
            if bs is None:
                bs = books[b] = _BookState()
            bs.apply_fill(is_buy, qty, price, fee, ets)
            bs.add_volume(ets, qty * price)
        return restarted

    # ---- main -------------------------------------------------------------- #
    def respond(self, state):
        vh = state.dendrite.hotkey
        ts = int(state.timestamp)
        cfg = state.config
        self.ticks += 1

        books = self._books(vh)
        # simulation restart: timestamp went backwards, or a start notice arrived
        prev = self.last_ts.get(vh)
        if prev is not None and ts < prev:
            self._reset_validator(vh, "timestamp went backwards")
            books = self._books(vh)
        restarted = self._process_notices(vh, books, ts)
        if restarted:
            self._reset_validator(vh, "EVENT_SIMULATION_START")
            books = self._books(vh)
        self.last_ts[vh] = ts

        price_dec = int(_attr(cfg, "priceDecimals", default=2))
        vol_dec = int(_attr(cfg, "volumeDecimals", default=4))
        tick = 10.0 ** (-price_dec)
        min_size = _f(_attr(cfg, "min_order_size", default=0.0))
        size = round(max(self.size, min_size, 10.0 ** (-vol_dec)), vol_dec)
        pub_ns = int(_attr(cfg, "publish_interval", default=1_000_000_000) or 1_000_000_000)
        ttl_ns = int(max(1, self.ttl_s) * 1_000_000_000)
        wealth = _f(_attr(cfg, "miner_wealth", default=50_000.0), 50_000.0)
        vol_cap = 10.0 * wealth * self.cap_frac      # scoring.activity.capital_turnover_cap = 10

        response = self.make_response(exchange_mode=False)
        accounts = self.accounts

        for book_id, book in (state.books or {}).items():
            try:
                self._quote_book(response, book_id, book, accounts.get(book_id), books, ts,
                                 tick, price_dec, vol_dec, size, ttl_ns, vol_cap)
            except Exception as e:      # one bad book must never cost the whole response
                bt.logging.warning(f"V6: book {book_id} skipped: {e!r}")

        if self.log_every and self.ticks % self.log_every == 0:
            self._log_summary(vh, books, ts)
        return response

    def _quote_book(self, response, book_id, book, account, books, ts,
                    tick, price_dec, vol_dec, size, ttl_ns, vol_cap):
        bids = _attr(book, "bids", "b", default=None)
        asks = _attr(book, "asks", "a", default=None)
        if not bids or not asks or account is None:
            return
        best_bid = _f(_attr(bids[0], "price", "p", default=0.0))
        best_ask = _f(_attr(asks[0], "price", "p", default=0.0))
        if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
            return
        mid = 0.5 * (best_bid + best_ask)
        spread = best_ask - best_bid

        bs = books.get(book_id)
        if bs is None:
            bs = books[book_id] = _BookState()

        maker_rate, taker_rate = self._fees(account)
        maker_fee_px = max(maker_rate, 0.0) * mid          # per unit, one maker leg
        rt_fee_px = 2.0 * maker_fee_px

        # round-trip edge target and half-width
        rt_edge = max(self.edge_bps * 1e-4 * mid, self.min_edge_ticks * tick) + rt_fee_px
        h = 0.5 * rt_edge
        exit_edge = max(self.exit_bps * 1e-4 * mid, self.exit_ticks * tick)

        inv = bs.inventory()
        # inventory skew: lean quotes away from the side we are long
        skew = self.skew_bps * 1e-4 * mid * inv
        mid_r = mid - skew

        bid_px = self._round_down(mid_r - h, tick)
        ask_px = self._round_up(mid_r + h, tick)
        # post-only safety: never cross the touch
        bid_px = min(bid_px, self._round_down(best_ask - tick, tick))
        ask_px = max(ask_px, self._round_up(best_bid + tick, tick))

        # FIFO-aware exits: the head lot must close at a profit after both legs' fees
        if bs.longs:
            head = bs.longs[0]
            floor_ask = head.price + (head.fee / head.qty if head.qty > 0 else 0.0) + maker_fee_px + exit_edge
            ask_px = max(ask_px, self._round_up(floor_ask, tick))
        if bs.shorts:
            head = bs.shorts[0]
            cap_bid = head.price - (head.fee / head.qty if head.qty > 0 else 0.0) - maker_fee_px - exit_edge
            bid_px = min(bid_px, self._round_down(cap_bid, tick))

        # gating: inventory cap and the rolling 24 h notional budget, paced linearly
        # through the simulation day so the cap is never exhausted early
        want_bid = inv < self.max_inv - 1e-9
        want_ask = inv > -self.max_inv + 1e-9
        day_pos = (ts % DAY_NS) / DAY_NS
        allowed = vol_cap * min(1.0, day_pos + self.pace_headroom)
        over_budget = bs.volume_24h(ts) >= allowed
        if over_budget:
            want_bid = want_bid and inv < -1e-9      # only if it reduces a short
            want_ask = want_ask and inv > 1e-9       # only if it reduces a long
        # never quote an entry that would need more free balance than we have
        base_free = _f(_attr(_attr(account, "base_balance", "bb", default=None), "free", "f", default=0.0))
        quote_free = _f(_attr(_attr(account, "quote_balance", "qb", default=None), "free", "f", default=0.0))
        if want_ask and base_free < size:
            want_ask = False
        if want_bid and quote_free < size * bid_px * 1.01:
            want_bid = False
        if bid_px <= 0:
            want_bid = False

        # reconcile with what is actually resting
        my_orders = _attr(account, "orders", "o", default=[]) or []
        rest_bids, rest_asks, all_ids = [], [], []
        for o in my_orders:
            oid = _attr(o, "id", "i", default=None)
            if oid is None:
                continue
            side = int(_attr(o, "side", "s", default=0))
            px = _f(_attr(o, "price", "p", default=0.0))
            q = _f(_attr(o, "quantity", "q", default=0.0))
            all_ids.append(oid)
            (rest_bids if side == 0 else rest_asks).append((oid, px, q))

        to_cancel = []
        tol = self.requote_ticks * tick + 1e-9

        def _reconcile(resting, want, target_px):
            keep = None
            if want:
                # keep the single closest order if it is near the target and still has size
                best = None
                for oid, px, q in resting:
                    if abs(px - target_px) <= tol and q >= 0.5 * size:
                        if best is None or abs(px - target_px) < abs(best[1] - target_px):
                            best = (oid, px, q)
                keep = best[0] if best else None
            for oid, px, q in resting:
                if oid != keep:
                    to_cancel.append(oid)
            return keep is not None

        have_bid = _reconcile(rest_bids, want_bid, bid_px)
        have_ask = _reconcile(rest_asks, want_ask, ask_px)

        # hard cap on resting orders (stale accumulation guard)
        if len(all_ids) > self.max_resting:
            for oid in all_ids:
                if oid not in to_cancel:
                    to_cancel.append(oid)
            have_bid = have_ask = False

        if to_cancel:
            response.cancel_orders(book_id, to_cancel)
        if want_bid and not have_bid:
            response.limit_order(book_id=book_id, direction=OrderDirection.BUY,
                                 quantity=size, price=round(bid_px, price_dec),
                                 postOnly=True, timeInForce=TimeInForce.GTT,
                                 expiryPeriod=ttl_ns)
        if want_ask and not have_ask:
            response.limit_order(book_id=book_id, direction=OrderDirection.SELL,
                                 quantity=size, price=round(ask_px, price_dec),
                                 postOnly=True, timeInForce=TimeInForce.GTT,
                                 expiryPeriod=ttl_ns)
        bs.last_bid, bs.last_ask = bid_px, ask_px

    # ---- logging ----------------------------------------------------------- #
    def _log_summary(self, vh, books, ts):
        n = len(books)
        if n == 0:
            return
        fills = sum(b.n_fills for b in books.values())
        wins = sum(b.wins for b in books.values())
        losses = sum(b.losses for b in books.values())
        realized = sum(b.realized for b in books.values())
        inv_abs = sum(abs(b.inventory()) for b in books.values())
        active = sum(1 for b in books.values() if b.wins + b.losses >= 3)
        rejects = sum(b.rejects for b in books.values())
        vol = sum(b.volume_24h(ts) for b in books.values())
        bt.logging.info(
            f"V6 [{vh[:8]}] t={ts // 1_000_000_000}s books={n} scored>=3:{active} "
            f"fills={fills} wins={wins} losses={losses} realized={realized:.4f} "
            f"|inv|={inv_abs:.2f} vol24h={vol:,.0f} rejects={rejects}"
        )


if __name__ == "__main__":
    from taos.common.agents import launch
    launch(MinerAgent_V6)
