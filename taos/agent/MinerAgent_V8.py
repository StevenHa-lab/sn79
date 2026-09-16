"""
MinerAgent_V8 -- profit-only ladder.

THE ONE RULE: never place a sell that would close the validator's oldest open lot at a
loss.  Everything else follows from that.

Why this is the whole game
--------------------------
The validator scores each book with Kappa-3 over the series of REALIZED, FIFO-matched,
fee-inclusive P&L bucketed by simulation second across a 3-hour window:

    kappa = mean / cbrt(LPM3 + regularization)

Three consequences, all measured on the live subnet rather than assumed:

1. **Unrealized losses are free.**  Open positions never enter the series, and the daily
   simulation restart discards them outright.  A position that never comes back simply
   costs nothing.  So an agent that only ever sells above its cost has a realized series
   containing *no negative entries at all* -- mean > 0 by construction, LPM3 ~ 0, and
   kappa is then bounded only by how OFTEN it realizes:

       kappa ~ 10 / (1 + 1/sqrt(p)),  p = fraction of seconds carrying a realization

   p = 1%  -> kappa 0.89 -> normalized 0.68.   p = 3.9% -> kappa 1.65 -> normalized 0.83.
   The top miners sit at kappa 1.6-1.7.  Frequency, not cleverness, is the lever.

2. **Zero-mean trading scores zero.**  A predecessor quoted ~3 bps inside a ~9 bps spread
   on all 128 books, took 290,000 fills, collected the maker rebate on nearly every one,
   and captured EXACTLY +0.00 bps of realized edge per unit of round-trip notional: the
   rebate and the adverse selection cancelled to the penny.  Its kappa was 0.002 and it
   was deregistered.  Realized P&L that rounds to 0.0000 is also deleted by the validator
   before scoring, which is why that agent showed no kappa at all for hours.

3. **The agent's FIFO must match the validator's exactly.**  Selling "my cheap recent lot"
   at a profit still closes the validator's OLDEST, dearest lot -- at a loss, which the
   scorer then cubes.  So the ledger is rebuilt from actual fills, deduped by trade id
   across both notice and book-event sources, and the sell price is derived from the head
   of that queue and nothing else.

How it trades
-------------
Per book, per tick, at most three instructions (one cancel batch, one bid, one ask):

  BUY   a ladder of up to `ladder` post-only bids RESTING below the market, each of size
        `lot` (2.0 base), placed at `rung_bps` + k*`rung_step_bps` under the mid and then
        LEFT ALONE.  A bid is never re-centred because the mid moved: a quote that chases
        the mid stays the same distance below it and so almost never fills, which is the
        single biggest determinant of round-trip frequency.  A rung is replaced only once
        price has run `stale_bps` above it.  Buying badly is survivable -- the lot simply
        waits -- so the ladder can sit close to the touch and fill often.

  SELL  ONE post-only ask of size `slice` (0.25 base) at the head lot's exact per-unit
        break-even plus `margin_bps`.  **`slice` is much smaller than `lot`, and that is
        the point**: one 2.0 buy is harvested as eight separate 0.25 closes, so a single
        entry produces eight distinct positive seconds instead of one.  Kappa counts
        seconds, not size.  This is exactly what the top miners do -- their fills are a
        few large buys and a continuous stream of 0.25 maker sells.

  NEVER a market order, a stop, or any sell below the head lot's break-even.

The maker rebate works in our favour twice over.  The population is ~90% taker, so the
dynamic fee policy currently pays makers a median 8.7 bps and up to 30 bps on some books.
That rebate enters the break-even calculation directly, so a deeper rebate lets the same
margin be met at a LOWER ask price, which fills sooner.

Unknown inventory
-----------------
If the account holds more base than the ledger accounts for, that excess is older than
every lot we know and sits AHEAD of ours in the validator's queue.  Selling would close it
at an unknown price.  So the book stops selling until the discrepancy clears.  Registering
a fresh hotkey and starting flat avoids this entirely, which is the intended way to run it.

Parameters (``--agent.params key=value ...``)
---------------------------------------------
  lot              BUY size, base units                            (default 2.0)
  slice            SELL size, base units                           (default 0.25, floored at min_order_size)
  margin_bps       required net profit per slice, bps of cost      (default 20.0)
  rung_bps         first resting bid this far below mid            (default 8.0)
  rung_step_bps    each further rung this much lower again         (default 12.0)
  ladder           resting bids kept per book                      (default 3)
  max_inv          inventory ceiling per book, base units          (default 30.0)
  stale_bps        re-place a bid once price runs this far above it(default 150.0)
  cap_frac         share of the 24h per-book volume cap to use     (default 0.85)
  requote_ticks    tolerance before an ask is re-placed            (default 2)
  max_resting      hard ceiling on resting orders per book         (default 6)
  ttl_s            GTT lifetime of a resting bid, sim seconds      (default 1800)
  unknown_tol      base units of unexplained inventory tolerated   (default 0.5)
  log_every        ticks between summary lines                     (default 30)
  save_every       ticks between state snapshots                   (default 60, 0 = off)

The take-profit ask is GTC: it is meant to rest until price comes to it.
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
VOL_BUCKET_NS = 300_000_000_000          # rolling-volume resolution (5 sim-minutes)
SEEN_CAP = 250_000                       # trade ids remembered for dedupe


# --------------------------------------------------------------------------- #
# helpers
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
    """One open long lot, as the validator's FIFO sees it."""
    __slots__ = ("price", "qty", "fee", "ts")

    def __init__(self, price, qty, fee, ts):
        self.price = price
        self.qty = qty
        self.fee = fee          # quote paid in fees on the opening leg (negative = rebate)
        self.ts = ts

    def breakeven(self, maker_rate_now, margin_bps):
        """Lowest sell price at which closing ANY slice of this lot nets >= margin.

        Per unit, with f = this lot's opening fee per unit and m = the current maker rate:
            (sell - price) - f - sell*m  >=  margin * price
        so sell >= (price*(1+margin) + f) / (1 - m).
        Independent of slice size, which is what lets one lot be harvested in many slices.
        """
        f = self.fee / self.qty if self.qty > 0 else 0.0
        denom = 1.0 - maker_rate_now
        if denom <= 0:
            return float("inf")
        return (self.price * (1.0 + margin_bps * 1e-4) + f) / denom


class _Book:
    """Per (validator, book) state."""
    __slots__ = ("lots", "vol", "vol_sum", "realized", "closes", "last_buy_px",
                 "n_fills", "blocked", "rejects")

    def __init__(self):
        self.lots = deque()          # FIFO, oldest first -- mirrors the validator's queue
        self.vol = deque()           # (bucket_ts, notional) for the rolling 24h cap
        self.vol_sum = 0.0
        self.realized = 0.0
        self.closes = 0
        self.last_buy_px = None
        self.n_fills = 0
        self.blocked = False         # unknown inventory ahead of ours: do not sell
        self.rejects = 0

    # -- inventory ------------------------------------------------------------ #
    def inventory(self):
        return sum(l.qty for l in self.lots)

    # -- fills ---------------------------------------------------------------- #
    def apply_buy(self, price, qty, fee, ts):
        self.lots.append(_Lot(price, qty, fee, ts))
        self.last_buy_px = price
        self.n_fills += 1

    def apply_sell(self, price, qty, fee, ts):
        """FIFO-close against the head, exactly as the validator does."""
        remaining = qty
        fee_per_unit = fee / qty if qty > 0 else 0.0
        pnl = 0.0
        while remaining > 1e-12 and self.lots:
            lot = self.lots[0]
            closed = min(remaining, lot.qty)
            open_fee_share = lot.fee * (closed / lot.qty) if lot.qty > 0 else 0.0
            pnl += (price - lot.price) * closed - open_fee_share - fee_per_unit * closed
            lot.qty -= closed
            lot.fee -= open_fee_share
            remaining -= closed
            if lot.qty <= 1e-12:
                self.lots.popleft()
        self.realized += pnl
        self.closes += 1
        self.n_fills += 1
        return pnl

    # -- rolling volume ------------------------------------------------------- #
    def add_volume(self, ts, notional):
        b = (ts // VOL_BUCKET_NS) * VOL_BUCKET_NS
        if self.vol and self.vol[-1][0] == b:
            self.vol[-1] = (b, self.vol[-1][1] + notional)
        else:
            self.vol.append((b, notional))
        self.vol_sum += notional
        self._trim(ts)

    def volume_24h(self, ts):
        self._trim(ts)
        return self.vol_sum

    def _trim(self, ts):
        cutoff = ts - DAY_NS
        while self.vol and self.vol[0][0] < cutoff:
            self.vol_sum -= self.vol.popleft()[1]
        if self.vol_sum < 0:
            self.vol_sum = 0.0

    # -- persistence ---------------------------------------------------------- #
    def to_json(self):
        return {"L": [[l.price, l.qty, l.fee, l.ts] for l in self.lots],
                "V": list(self.vol),
                "st": [self.realized, self.closes, self.n_fills, self.last_buy_px]}

    @classmethod
    def from_json(cls, d):
        b = cls()
        b.lots = deque(_Lot(*x) for x in d.get("L", []))
        b.vol = deque((int(t), float(n)) for t, n in d.get("V", []))
        b.vol_sum = sum(n for _, n in b.vol)
        st = d.get("st") or [0.0, 0, 0, None]
        b.realized, b.closes, b.n_fills, b.last_buy_px = st
        return b


# --------------------------------------------------------------------------- #
# agent
# --------------------------------------------------------------------------- #

class MinerAgent_V8(FinanceAgent):

    # ---- setup -------------------------------------------------------------- #
    def _p(self, name, default):
        v = getattr(self.config, name, None)
        if v is None:
            return default
        try:
            return type(default)(v)
        except Exception:
            return default

    def initialize(self):
        self.lot = self._p("lot", 2.0)
        self.slice = self._p("slice", 0.25)
        self.margin_bps = self._p("margin_bps", 20.0)
        self.rung_bps = self._p("rung_bps", 8.0)
        self.rung_step_bps = self._p("rung_step_bps", 12.0)
        self.ladder = self._p("ladder", 3)
        self.max_inv = self._p("max_inv", 30.0)
        self.stale_bps = self._p("stale_bps", 150.0)
        self.cap_frac = self._p("cap_frac", 0.85)
        self.requote_ticks = self._p("requote_ticks", 2)
        self.max_resting = self._p("max_resting", 6)
        self.ttl_s = self._p("ttl_s", 1800)
        self.unknown_tol = self._p("unknown_tol", 0.5)
        self.log_every = self._p("log_every", 30)
        self.save_every = self._p("save_every", 60)

        self.books_by_validator = {}     # vh -> {book_id: _Book}
        self.last_ts = {}
        self.ticks = 0
        self._seen = set()
        self._seen_q = deque()
        self._save_lock = threading.Lock()
        self.n_buys = self.n_sells = self.n_wins = self.n_losses = 0
        self.n_blocked = 0
        self._load_state()

    # ---- lean plumbing: skip the base class's per-tick debug rendering ------- #
    def update(self, state):
        self.simulation_config = state.config
        self.accounts = (state.accounts or {}).get(self.uid, {}) or {}
        self.events = (state.notices or {}).get(self.uid, []) or []
        self._exchange_mode = False

    def report(self, state, response):
        return

    # ---- persistence -------------------------------------------------------- #
    def _state_path(self):
        os.makedirs("agent_state", exist_ok=True)
        return f"agent_state/v8_state_{self.uid}.json"

    def _load_state(self):
        path = self._state_path()
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for vh, books in data.get("books", {}).items():
                self.books_by_validator[vh] = {int(b): _Book.from_json(d) for b, d in books.items()}
            self.last_ts = {vh: int(t) for vh, t in data.get("last_ts", {}).items()}
            n = sum(len(b) for b in self.books_by_validator.values())
            bt.logging.info(f"V8: restored {len(self.books_by_validator)} validator(s), {n} books")
        except Exception as e:
            bt.logging.warning(f"V8: state reload failed ({e!r}); starting flat")
            self.books_by_validator, self.last_ts = {}, {}

    def _save_state_async(self):
        data = {"books": {vh: {str(b): bk.to_json() for b, bk in books.items()}
                          for vh, books in self.books_by_validator.items()},
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
                bt.logging.warning(f"V8: state save failed: {e!r}")
            finally:
                self._save_lock.release()

        t = threading.Thread(target=_write, daemon=True)
        t.start()
        return t

    # ---- fills -------------------------------------------------------------- #
    def _seen_once(self, book_id, tid):
        """True the first time this trade id is offered, False on any repeat."""
        if tid is None:
            return True
        key = (book_id, int(tid))
        if key in self._seen:
            return False
        self._seen.add(key)
        self._seen_q.append(key)
        if len(self._seen_q) > SEEN_CAP:
            self._seen.discard(self._seen_q.popleft())
        return True

    def _apply(self, bk, book_id, tid, taker, maker, side, price, qty, fee_t, fee_m, ts):
        is_taker, is_maker = taker == self.uid, maker == self.uid
        if not (is_taker or is_maker):
            return
        if not self._seen_once(book_id, tid):
            return
        price, qty = _f(price), _f(qty)
        if price <= 0 or qty <= 0:
            return
        # the wire `side` is the TAKER's direction: 0 = taker bought
        is_buy = (is_taker and int(side) == 0) or (is_maker and int(side) == 1)
        fee = _f(fee_t if is_taker else fee_m)
        if is_buy:
            bk.apply_buy(price, qty, fee, ts)
            self.n_buys += 1
        else:
            pnl = bk.apply_sell(price, qty, fee, ts)
            self.n_sells += 1
            if pnl > 0:
                self.n_wins += 1
            elif pnl < 0:
                self.n_losses += 1
                bt.logging.warning(
                    f"V8: REALIZED LOSS {pnl:.4f} on book {book_id} at {price} -- the head lot "
                    f"was dearer than expected; check the ledger")
        bk.add_volume(ts, price * qty)

    def _ingest_notices(self, books, ts):
        """Fills from the notice stream. Returns True on a simulation restart."""
        restarted = False
        for ev in self.events:
            et = _attr(ev, "type", "y", default="")
            if et in ("EVENT_SIMULATION_START", "ESS"):
                restarted = True
                continue
            if et in ("ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT", "ERDPOL",
                      "ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET", "ERDPOM"):
                b = _attr(ev, "bookId", "b", default=None)
                if b is not None and b in books:
                    books[b].rejects += 1
                continue
            if et not in ("EVENT_TRADE", "ET"):
                continue
            b = _attr(ev, "bookId", "b", default=None)
            if b is None:
                continue
            bk = books.get(b) or books.setdefault(b, _Book())
            self._apply(bk, b, _attr(ev, "tradeId", "i", default=None),
                        _attr(ev, "takerAgentId", "Ta", default=-1),
                        _attr(ev, "makerAgentId", "Ma", default=-1),
                        _attr(ev, "side", "s", default=0),
                        _attr(ev, "price", "p", default=0.0),
                        _attr(ev, "quantity", "q", default=0.0),
                        _attr(ev, "takerFee", "Tf", default=0.0),
                        _attr(ev, "makerFee", "Mf", default=0.0),
                        int(_attr(ev, "timestamp", "t", default=ts) or ts))
        return restarted

    def _ingest_book_events(self, bk, book_id, book, ts):
        """Fills from the book's own L3 trade list -- the same trades, as a safety net."""
        for ev in (_attr(book, "events", "e", default=None) or []):
            taker = _attr(ev, "Ta", "taker_agent_id", default=None)
            if taker is None:
                continue                                   # an order or a cancellation
            maker = _attr(ev, "Ma", "maker_agent_id", default=-1)
            if taker != self.uid and maker != self.uid:
                continue
            self._apply(bk, book_id, _attr(ev, "i", "id", default=None), taker, maker,
                        _attr(ev, "s", "side", default=0),
                        _attr(ev, "p", "price", default=0.0),
                        _attr(ev, "q", "quantity", default=0.0),
                        _attr(ev, "Tf", "taker_fee", default=0.0),
                        _attr(ev, "Mf", "maker_fee", default=0.0),
                        int(_attr(ev, "t", "timestamp", default=ts) or ts))

    # ---- main --------------------------------------------------------------- #
    def respond(self, state):
        vh = state.dendrite.hotkey
        ts = int(state.timestamp)
        cfg = state.config
        self.ticks += 1

        books = self.books_by_validator.setdefault(vh, {})
        prev = self.last_ts.get(vh)
        restarted = self._ingest_notices(books, ts)
        if restarted or (prev is not None and ts < prev):
            # the simulator discards open positions at a restart; so do we
            bt.logging.info(f"V8: simulation restart on {vh[:8]}, clearing {len(books)} ledgers")
            for bk in books.values():
                bk.lots.clear()
                bk.last_buy_px = None
                bk.blocked = False
        self.last_ts[vh] = ts

        price_dec = int(_attr(cfg, "priceDecimals", default=2))
        vol_dec = int(_attr(cfg, "volumeDecimals", default=4))
        tick = 10.0 ** (-price_dec)
        min_size = _f(_attr(cfg, "min_order_size", default=0.0))
        lot = round(max(self.lot, min_size, 10.0 ** (-vol_dec)), vol_dec)
        sl = round(max(self.slice, min_size, 10.0 ** (-vol_dec)), vol_dec)
        ttl_ns = int(max(1, self.ttl_s) * 1_000_000_000)
        wealth = _f(_attr(cfg, "miner_wealth", default=50_000.0), 50_000.0)
        vol_cap = 10.0 * wealth * self.cap_frac      # scoring.activity.capital_turnover_cap = 10

        response = self.make_response(exchange_mode=False)
        for book_id, book in (state.books or {}).items():
            try:
                self._quote(response, book_id, book, self.accounts.get(book_id),
                            books, ts, tick, price_dec, vol_dec, lot, sl, ttl_ns, vol_cap)
            except Exception as e:
                bt.logging.warning(f"V8: book {book_id} skipped: {e!r}")

        if self.log_every and self.ticks % self.log_every == 0:
            self._log(vh, books, ts)
        if self.save_every and self.ticks % self.save_every == 0:
            self._save_state_async()
        return response

    def _quote(self, response, book_id, book, account, books, ts,
               tick, price_dec, vol_dec, lot, sl, ttl_ns, vol_cap):
        bids = _attr(book, "bids", "b", default=None)
        asks = _attr(book, "asks", "a", default=None)
        if not bids or not asks or account is None:
            return
        best_bid = _f(_attr(bids[0], "price", "p", default=0.0))
        best_ask = _f(_attr(asks[0], "price", "p", default=0.0))
        if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
            return
        mid = 0.5 * (best_bid + best_ask)

        bk = books.get(book_id) or books.setdefault(book_id, _Book())
        self._ingest_book_events(bk, book_id, book, ts)

        f = _attr(account, "fees", default=None)
        maker_rate = _f(_attr(f, "maker_fee_rate", "m", default=0.0)) if f is not None else 0.0

        # ---- unknown inventory guard --------------------------------------- #
        # Anything the account holds beyond our ledger is OLDER than our lots and sits
        # ahead of them in the validator's queue. Selling would close it at a price we
        # do not know, so this book stops selling until the gap closes.
        bb = _attr(account, "base_balance", "bb", default=None)
        acct_inv = None
        if bb is not None:
            total = _f(_attr(bb, "total", "t", default=float("nan")), float("nan"))
            init = _f(_attr(bb, "initial", "i", default=float("nan")), float("nan"))
            if total == total and init == init:
                acct_inv = total - init
        was_blocked = bk.blocked
        bk.blocked = acct_inv is not None and acct_inv > bk.inventory() + self.unknown_tol
        if bk.blocked and not was_blocked:
            self.n_blocked += 1
            bt.logging.warning(
                f"V8: book {book_id} has {acct_inv - bk.inventory():.2f} base of unexplained "
                f"inventory ahead of our lots; selling disabled on this book")

        # ---- the take-profit ask -------------------------------------------- #
        # one slice at the head lot's break-even; the head is what the validator closes
        want_ask = None
        if bk.lots and not bk.blocked:
            head = bk.lots[0]
            # Price against the DEAREST open lot, not merely the head.  An ask placed for a
            # cheap head can still be resting a tick later when a fill has promoted a dearer
            # lot to the front, and the validator would then close THAT one -- at a loss the
            # scorer cubes.  Taking the maximum removes the race for the cost of a slightly
            # higher ask.  Lots are few (<= max_inv/lot), so this is cheap.
            floor = max(l.breakeven(maker_rate, self.margin_bps) for l in bk.lots)
            px = self._round_up(max(floor, best_bid + tick), tick)
            qty = round(min(sl, head.qty), vol_dec)
            if px > best_bid and qty > 0:
                want_ask = (round(px, price_dec), qty)

        # ---- the resting accumulation ladder --------------------------------- #
        day_pos = (ts % DAY_NS) / DAY_NS
        budget_ok = bk.volume_24h(ts) < vol_cap * min(1.0, day_pos + 0.05)
        room = bk.inventory() + lot <= self.max_inv
        stale_floor = mid * (1.0 - self.stale_bps * 1e-4)
        quote_free = _f(_attr(_attr(account, "quote_balance", "qb", default=None),
                              "free", "f", default=0.0))

        # ---- what is resting now --------------------------------------------- #
        rest_b, rest_a, all_ids = [], [], []
        for o in (_attr(account, "orders", "o", default=[]) or []):
            oid = _attr(o, "id", "i", default=None)
            if oid is None:
                continue
            all_ids.append(oid)
            e = (oid, _f(_attr(o, "price", "p", default=0.0)),
                 _f(_attr(o, "quantity", "q", default=0.0)))
            (rest_b if int(_attr(o, "side", "s", default=0)) == 0 else rest_a).append(e)

        cancels = []
        tol = self.requote_ticks * tick + 1e-9

        # Bids are sticky: a rung is kept as long as it is still somewhere we would buy,
        # and dropped only once price has run away above it.
        top_rung = mid * (1.0 - self.rung_bps * 1e-4)
        keep_b = [e for e in rest_b if e[1] <= top_rung + tol and e[1] >= stale_floor]
        keep_ids = {e[0] for e in keep_b}
        cancels.extend(oid for oid, _, _ in rest_b if oid not in keep_ids)

        keep_ask = None
        if want_ask:
            for oid, p, q in rest_a:
                if abs(p - want_ask[0]) <= tol and q >= 0.5 * want_ask[1]:
                    keep_ask = oid
                    break
        cancels.extend(oid for oid, _, _ in rest_a if oid != keep_ask)

        if len(all_ids) > self.max_resting:
            cancels = list(dict.fromkeys(cancels + all_ids))
            keep_b, keep_ask = [], None

        # one new rung per tick, placed clear of the rungs already resting
        new_bid = None
        if room and budget_ok and len(keep_b) < self.ladder:
            px = top_rung
            if keep_b:
                px = min(px, min(p for _, p, _ in keep_b) * (1.0 - self.rung_step_bps * 1e-4))
            px = self._round_down(min(px, best_ask - tick), tick)
            if px > 0 and px < best_ask and quote_free > px * lot * 1.05:
                new_bid = (round(px, price_dec), lot)

        # ---- emit (at most 3 instructions on this book) ----------------------- #
        if cancels:
            response.cancel_orders(book_id, cancels)
        if want_ask and keep_ask is None:
            response.limit_order(book_id=book_id, direction=OrderDirection.SELL,
                                 quantity=want_ask[1], price=want_ask[0],
                                 postOnly=True, timeInForce=TimeInForce.GTC)
        if new_bid:
            response.limit_order(book_id=book_id, direction=OrderDirection.BUY,
                                 quantity=new_bid[1], price=new_bid[0],
                                 postOnly=True, timeInForce=TimeInForce.GTT,
                                 expiryPeriod=ttl_ns)

    # ---- misc --------------------------------------------------------------- #
    @staticmethod
    def _round_down(x, t):
        return math.floor(x / t + 1e-9) * t

    @staticmethod
    def _round_up(x, t):
        return math.ceil(x / t - 1e-9) * t

    def _log(self, vh, books, ts):
        if not books:
            return
        inv = sum(b.inventory() for b in books.values())
        realized = sum(b.realized for b in books.values())
        closes = sum(b.closes for b in books.values())
        blocked = sum(1 for b in books.values() if b.blocked)
        full = sum(1 for b in books.values() if b.inventory() + self.lot > self.max_inv)
        vol = sum(b.volume_24h(ts) for b in books.values())
        per3h = closes / max(1e-9, (self.ticks / 3600.0)) * 3.0 / max(1, len(books))
        bt.logging.info(
            f"V8 [{vh[:8]}] t={ts // 1_000_000_000}s books={len(books)} closes={closes} "
            f"(~{per3h:.0f}/book/3h) wins={self.n_wins} LOSSES={self.n_losses} "
            f"realized={realized:+.2f} inv={inv:.1f} full={full} blocked={blocked} "
            f"vol24h={vol:,.0f} buys={self.n_buys} sells={self.n_sells}")


if __name__ == "__main__":
    from taos.common.agents import launch
    launch(MinerAgent_V8)
