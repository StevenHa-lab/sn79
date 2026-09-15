"""
MinerAgent_V7 -- V6 maker engine + rebate-aware exits, trend lean and frozen-book control.

What the live leaderboard showed (validator Prometheus, sim 20260913_0722): the top miners
buy in size as TAKERS, then harvest with streams of 0.25 MAKER sells so every second
realises a small FIFO gain; books that go under water are simply frozen (never realised,
cleared at the daily restart) inside the 48-book allowance.  Makers currently receive a
rebate of roughly 10 bps per fill because the miner population is ~90% taker.

V7 keeps V6's two-sided 0.25 maker engine on every book and adds:
  * rebate-aware exit pricing: a negative maker rate lets the exit rest INSIDE the entry
    price and still realise a net gain, so fills come faster in the rebate regime;
  * a per-book trend lean (mid drift over trend_window_s): quotes shift with the drift, the
    inventory cap on the trend side rises to max_inv_trend, and entries AGAINST a strong
    drift stop once inventory is half full (that is what freezes books in V6);
  * frozen-book control: a book whose FIFO head is under water by freeze_bps at the
    inventory cap is "frozen".  Up to max_frozen books may stay frozen (the scorer ignores
    48 None books); beyond that the oldest frozen books are cut one 0.25 lot per
    cut_every_s with an IOC order at the touch, so losses stay tiny and spread over seconds.

Original V6 design notes follow.

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
  save_every      ticks between state snapshots to disk          (default 60, 0 = off)
  rebate_aware    use the signed maker rate in exit pricing      (default 1)
  trend_window_s  mid-drift window for the trend lean, sim s     (default 300)
  trend_bps       drift (bps over the window) that counts as a
                  full-strength trend                            (default 15.0)
  lean_frac       quote shift per unit trend, as a fraction of
                  the half-edge h                                (default 0.6)
  max_inv_trend   inventory cap on the trend side, base units    (default 4.0)
  freeze_bps      FIFO head under water by this much at the cap
                  marks the book frozen                          (default 25.0)
  max_frozen      frozen books tolerated per validator           (default 36)
  unfreeze_after_s frozen this long before it may be cut         (default 1800)
  cut_every_s     minimum spacing between cuts on one book       (default 60)

State (open lots, rolling volume in 5-minute buckets, last timestamp per validator) is
persisted to agent_state/v6_state_<uid>.json from a background thread and reloaded on start,
so a process restart neither loses the cost basis of open lots nor under-counts the cap.

Pacing: the validator's cap is a rolling 24 h window per book that survives the daily
simulation restart, so the agent keeps its per-book volume history across restarts and only
quotes new entries while the rolling notional is below cap_frac * cap * (day_progress +
pace_headroom).  Exits (quotes that reduce inventory) are always allowed.
"""

import json
import math
import os
import threading
from collections import deque

import bittensor as bt

from taos.im.agents import FinanceAgent
from taos.im.protocol.response import OrderDirection, TimeInForce

DAY_NS = 86_400_000_000_000
VOL_BUCKET_NS = 300_000_000_000      # rolling-volume resolution (5 sim-minutes)


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
                 "n_fills", "last_bid", "last_ask", "rejects", "mids", "frozen_since", "last_cut_ts")

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
        self.mids = deque()           # (ts, mid) for the trend window
        self.frozen_since = None
        self.last_cut_ts = 0

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
        b = (ts // VOL_BUCKET_NS) * VOL_BUCKET_NS
        if self.vol and self.vol[-1][0] == b:
            self.vol[-1] = (b, self.vol[-1][1] + notional)
        else:
            self.vol.append((b, notional))
        self.vol_sum += notional
        self._trim(ts)

    # -- persistence ---------------------------------------------------------- #
    def to_json(self):
        return {"L": [[l.price, l.qty, l.fee, l.ts] for l in self.longs],
                "S": [[l.price, l.qty, l.fee, l.ts] for l in self.shorts],
                "V": list(self.vol),
                "st": [self.realized, self.wins, self.losses, self.n_fills, self.rejects],
                "fz": self.frozen_since}

    @classmethod
    def from_json(cls, d):
        bs = cls()
        bs.longs = deque(_Lot(*x) for x in d.get("L", []))
        bs.shorts = deque(_Lot(*x) for x in d.get("S", []))
        bs.vol = deque((int(t), float(n)) for t, n in d.get("V", []))
        bs.vol_sum = sum(n for _, n in bs.vol)
        st = d.get("st") or [0.0, 0, 0, 0, 0]
        bs.realized, bs.wins, bs.losses, bs.n_fills, bs.rejects = st
        bs.frozen_since = d.get("fz")
        return bs

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

class MinerAgent_V7(FinanceAgent):

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
        self.save_every = self._p("save_every", 60)
        self.rebate_aware = self._p("rebate_aware", 1)
        self.trend_window_s = self._p("trend_window_s", 300)
        self.trend_bps = self._p("trend_bps", 15.0)
        self.lean_frac = self._p("lean_frac", 0.6)
        self.max_inv_trend = self._p("max_inv_trend", 4.0)
        self.freeze_bps = self._p("freeze_bps", 25.0)
        self.max_frozen = self._p("max_frozen", 36)
        self.unfreeze_after_s = self._p("unfreeze_after_s", 1800)
        self.cut_every_s = self._p("cut_every_s", 60)

        # per validator hotkey -> {book_id: _BookState}
        self.state_by_validator = {}
        self.last_ts = {}
        self.ticks = 0
        self._save_lock = threading.Lock()
        self.diag_fills = 0
        self.diag_side_mismatch = 0
        self.diag_rebase = 0
        self.diag_dup_fills = 0
        self.diag_event_fills = 0
        self._seen_tids = set()
        self._seen_order = deque()
        self.reconcile = self._p("reconcile", 1)
        self._load_state()

    # ---- persistence ------------------------------------------------------- #
    def _state_path(self):
        os.makedirs("agent_state", exist_ok=True)
        return f"agent_state/v7_state_{self.uid}.json"

    def _load_state(self):
        path = self._state_path()
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for vh, books in data.get("books", {}).items():
                self.state_by_validator[vh] = {int(b): _BookState.from_json(d) for b, d in books.items()}
            self.last_ts = {vh: int(t) for vh, t in data.get("last_ts", {}).items()}
            n = sum(len(b) for b in self.state_by_validator.values())
            bt.logging.info(f"V7: restored state for {len(self.state_by_validator)} validator(s), {n} books from {path}")
        except Exception as e:
            bt.logging.warning(f"V6: could not restore state from {path}: {e!r}; starting fresh")
            self.state_by_validator, self.last_ts = {}, {}

    def _save_state_async(self):
        # snapshot on the request thread (cheap), serialise + write on a helper thread
        data = {"books": {vh: {str(b): bs.to_json() for b, bs in books.items()}
                          for vh, books in self.state_by_validator.items()},
                "last_ts": dict(self.last_ts)}
        path = self._state_path()

        def _write():
            if not self._save_lock.acquire(blocking=False):
                return
            try:
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(data, f)
                os.replace(tmp, path)
            except Exception as e:
                bt.logging.warning(f"V6: state save failed: {e!r}")
            finally:
                self._save_lock.release()

        t = threading.Thread(target=_write, daemon=True)
        t.start()
        return t

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
            tid = _attr(ev, "tradeId", "i", default=None)
            taker = _attr(ev, "takerAgentId", "Ta", default=-1)
            maker = _attr(ev, "makerAgentId", "Ma", default=-1)
            if taker != self.uid and maker != self.uid:
                continue
            bs = books.get(b)
            if bs is None:
                bs = books[b] = _BookState()
            ets = int(_attr(ev, "timestamp", "t", default=ts) or ts)
            self._apply_trade(bs, b, tid, taker, maker, _attr(ev, "side", "s", default=0),
                              _f(_attr(ev, "price", "p", default=0.0)), _f(_attr(ev, "quantity", "q", default=0.0)),
                              _attr(ev, "takerFee", "Tf", default=0.0), _attr(ev, "makerFee", "Mf", default=0.0), ets)
        return restarted

    def _apply_trade(self, bs, b, tid, taker, maker, side, price, qty, fee_taker, fee_maker, ets):
        """Apply one trade (from a notice or a book event) to the ledger, deduped by trade id."""
        is_taker = taker == self.uid
        is_maker = maker == self.uid
        if not (is_taker or is_maker):
            return False
        if tid is not None:
            key = (b, int(tid))
            if key in self._seen_tids:
                self.diag_dup_fills += 1
                return False
            self._seen_tids.add(key)
            self._seen_order.append(key)
            if len(self._seen_order) > 200_000:
                old = self._seen_order.popleft()
                self._seen_tids.discard(old)
        side = int(side)
        is_buy = (is_taker and side == 0) or (is_maker and side == 1)
        fee = _f(fee_taker if is_taker else fee_maker)
        if qty <= 0 or price <= 0:
            return False
        if bs.last_bid is not None and bs.last_ask is not None:
            nearest_is_bid = abs(price - bs.last_bid) < abs(price - bs.last_ask)
            self.diag_fills += 1
            if nearest_is_bid != is_buy:
                self.diag_side_mismatch += 1
        bs.apply_fill(is_buy, qty, price, fee, ets)
        bs.add_volume(ets, qty * price)
        return True

    def _fills_from_book_events(self, bs, book_id, book, ts):
        """The book's L3 event list carries every trade with both agent ids: a complete
        record of my fills even when a notice is missing."""
        events = _attr(book, "events", "e", default=None)
        if not events:
            return
        for ev in events:
            y = _attr(ev, "y", "type", default=None)
            if y not in ("t", "trade"):
                # TradeInfo has taker/maker ids; orders and cancellations do not
                if _attr(ev, "Ta", "taker_agent_id", default=None) is None:
                    continue
            taker = _attr(ev, "Ta", "taker_agent_id", default=-1)
            maker = _attr(ev, "Ma", "maker_agent_id", default=-1)
            if taker != self.uid and maker != self.uid:
                continue
            tid = _attr(ev, "i", "id", default=None)
            ets = int(_attr(ev, "t", "timestamp", default=ts) or ts)
            if self._apply_trade(bs, book_id, tid, taker, maker, _attr(ev, "s", "side", default=0),
                                 _f(_attr(ev, "p", "price", default=0.0)), _f(_attr(ev, "q", "quantity", default=0.0)),
                                 _attr(ev, "Tf", "taker_fee", default=0.0), _attr(ev, "Mf", "maker_fee", default=0.0), ets):
                self.diag_event_fills += 1

    def _reconcile_inventory(self, bs, account, book_id, mid, ts):
        """The account on every state update is the authority. If the FIFO ledger's net
        inventory drifts from (base total - base initial) by more than one lot, rebase the
        ledger with a synthetic lot at the current mid so the caps bind on REAL inventory."""
        bb = _attr(account, "base_balance", "bb", default=None)
        if bb is None:
            return
        total = _f(_attr(bb, "total", "t", default=float("nan")), float("nan"))
        init = _f(_attr(bb, "initial", "i", default=float("nan")), float("nan"))
        if not (total == total and init == init):
            return
        acct_inv = total - init
        ledger_inv = bs.inventory()
        diff = acct_inv - ledger_inv
        if abs(diff) <= 1.5 * self.size:
            return
        self.diag_rebase += 1
        # Cost basis for the synthetic lot: the average price implied by the account's own
        # net quote/base movement (what the validator's FIFO is holding, on average), not the
        # current mid -- rebasing at mid made exits realise the OLD lots' losses on 2026-09-15.
        qb = _attr(account, "quote_balance", "qb", default=None)
        q_total = _f(_attr(qb, "total", "t", default=float("nan")), float("nan")) if qb is not None else float("nan")
        q_init = _f(_attr(qb, "initial", "i", default=float("nan")), float("nan")) if qb is not None else float("nan")
        basis = mid
        if q_total == q_total and q_init == q_init and abs(acct_inv) > 1e-9:
            implied = (q_init - q_total) / acct_inv          # quote spent per unit of net base held
            if 0.3 * mid < implied < 3.0 * mid:
                basis = implied
        if self.diag_rebase <= 5 or self.diag_rebase % 200 == 0:
            bt.logging.warning(f"V7 rebase: book {book_id} ledger inv {ledger_inv:+.2f} vs account {acct_inv:+.2f} "
                               f"(total {total:.2f} initial {init:.2f}); rebasing by {diff:+.2f} at basis {basis:.2f} (mid {mid:.2f})")
        if diff > 0:
            while diff > 1e-9 and bs.shorts:
                lot = bs.shorts[0]; take = min(diff, lot.qty); lot.qty -= take; diff -= take
                if lot.qty <= 1e-9:
                    bs.shorts.popleft()
            if diff > 1e-9:
                bs.longs.append(_Lot(basis, diff, 0.0, ts))
        else:
            diff = -diff
            while diff > 1e-9 and bs.longs:
                lot = bs.longs[0]; take = min(diff, lot.qty); lot.qty -= take; diff -= take
                if lot.qty <= 1e-9:
                    bs.longs.popleft()
            if diff > 1e-9:
                bs.shorts.append(_Lot(basis, diff, 0.0, ts))

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
        if self.save_every and self.ticks % self.save_every == 0:
            self._save_state_async()
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

        self._fills_from_book_events(bs, book_id, book, ts)
        if self.reconcile:
            self._reconcile_inventory(bs, account, book_id, mid, ts)

        maker_rate, taker_rate = self._fees(account)
        # signed when rebate_aware: a negative maker rate (rebate) lowers the exit floor
        maker_fee_px = (maker_rate if self.rebate_aware else max(maker_rate, 0.0)) * mid
        rt_fee_px = 2.0 * maker_fee_px

        # ---- trend lean: mid drift over the window ----
        win_ns = int(self.trend_window_s) * 1_000_000_000
        bs.mids.append((ts, mid))
        while bs.mids and bs.mids[0][0] < ts - win_ns:
            bs.mids.popleft()
        trend = 0.0
        if len(bs.mids) >= 2 and ts - bs.mids[0][0] >= win_ns // 2:
            m0 = bs.mids[0][1]
            drift_bps = 1e4 * (mid - m0) / m0 if m0 > 0 else 0.0
            trend = max(-1.0, min(1.0, drift_bps / max(self.trend_bps, 1e-9)))

        # round-trip edge target and half-width
        rt_edge = max(self.edge_bps * 1e-4 * mid, self.min_edge_ticks * tick) + max(rt_fee_px, 0.0)
        h = 0.5 * rt_edge
        exit_edge = max(self.exit_bps * 1e-4 * mid, self.exit_ticks * tick)

        inv = bs.inventory()
        # inventory skew: lean quotes away from the side we are long
        skew = self.skew_bps * 1e-4 * mid * inv
        mid_r = mid - skew + self.lean_frac * trend * h      # shift quotes with the drift

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
        max_long = self.max_inv_trend if trend > 0.5 else self.max_inv
        max_short = self.max_inv_trend if trend < -0.5 else self.max_inv
        want_bid = inv < max_long - 1e-9
        want_ask = inv > -max_short + 1e-9
        # do not keep adding against a strong drift once half full: that is what freezes books
        if trend < -0.5 and inv >= 0.5 * self.max_inv:
            want_bid = False
        if trend > 0.5 and inv <= -0.5 * self.max_inv:
            want_ask = False

        # ---- frozen-book bookkeeping ----
        under_water = False
        if bs.longs and inv > 0:
            head = bs.longs[0]
            under_water = head.price + (head.fee / head.qty if head.qty > 0 else 0.0) > mid * (1 + self.freeze_bps * 1e-4)
        elif bs.shorts and inv < 0:
            head = bs.shorts[0]
            under_water = head.price - (head.fee / head.qty if head.qty > 0 else 0.0) < mid * (1 - self.freeze_bps * 1e-4)
        at_cap = (inv > 0 and not want_bid) or (inv < 0 and not want_ask)
        if under_water and at_cap:
            if bs.frozen_since is None:
                bs.frozen_since = ts
        else:
            bs.frozen_since = None
        cut = None
        if bs.frozen_since is not None and ts - bs.frozen_since >= int(self.unfreeze_after_s) * 1_000_000_000 \
                and ts - bs.last_cut_ts >= int(self.cut_every_s) * 1_000_000_000:
            frozen_n = sum(1 for b in books.values() if b.frozen_since is not None)
            if frozen_n > self.max_frozen:
                cut = "sell" if inv > 0 else "buy"
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
        if cut is not None:
            # one small IOC lot at the touch: realises a small, bounded loss on the FIFO head
            cut_qty = round(max(min(size, 0.25), 10.0 ** (-vol_dec)), vol_dec)   # always the smallest lot
            if cut == "sell":
                response.limit_order(book_id=book_id, direction=OrderDirection.SELL, quantity=cut_qty,
                                     price=round(best_bid, price_dec), timeInForce=TimeInForce.IOC)
            else:
                response.limit_order(book_id=book_id, direction=OrderDirection.BUY, quantity=cut_qty,
                                     price=round(best_ask, price_dec), timeInForce=TimeInForce.IOC)
            bs.last_cut_ts = ts
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
        frozen = sum(1 for b in books.values() if b.frozen_since is not None)
        vol = sum(b.volume_24h(ts) for b in books.values())
        bt.logging.info(
            f"V7 [{vh[:8]}] t={ts // 1_000_000_000}s books={n} scored>=3:{active} frozen={frozen} "
            f"fills={fills} wins={wins} losses={losses} realized={realized:.4f} "
            f"|inv|={inv_abs:.2f} vol24h={vol:,.0f} rejects={rejects} "
            f"side_mismatch={self.diag_side_mismatch}/{self.diag_fills} rebases={self.diag_rebase} dup_fills={self.diag_dup_fills} event_only_fills={self.diag_event_fills}"
        )


if __name__ == "__main__":
    from taos.common.agents import launch
    launch(MinerAgent_V7)
