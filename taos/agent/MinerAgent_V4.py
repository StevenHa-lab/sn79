import os
import json
from collections import defaultdict, deque
import numpy as np

from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol.response import FinanceAgentResponse, OrderDirection, TimeInForce

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

    n = len(xs)
    if n < 3:
        return 0.0
    xbar = (n - 1) / 2.0
    Sxx = sum((i - xbar) ** 2 for i in range(n))
    ybar = sum(xs) / n
    sst = sum((y - ybar) ** 2 for y in xs)
    if Sxx <= 0 or sst <= 0:
        return 0.0
    slope = sum((i - xbar) * (xs[i] - ybar) for i in range(n)) / Sxx
    intercept = ybar - slope * xbar
    sse = sum((xs[i] - (intercept + slope * i)) ** 2 for i in range(n))
    r2 = max(0.0, 1.0 - sse / sst)
    return r2 if slope > 0 else -r2

def trend_move(xs):
    """
    Signed net move over the window from the OLS fit, as a fraction of price
    (scale-free across books). Unchanged from V2 (keeps the same 20x scaling so
    the slope thresholds carry over identically).
    """
    n = len(xs)
    if n < 3:
        return 0.0
    xbar = (n - 1) / 2.0
    Sxx = sum((i - xbar) ** 2 for i in range(n))
    if Sxx <= 0:
        return 0.0
    ybar = sum(xs) / n
    slope = sum((i - xbar) * (xs[i] - ybar) for i in range(n)) / Sxx
    fitted_move = slope * (n - 1)
    return 20 * fitted_move / ybar

def round_trip_fee_offset(cfg, fees, mid):
    """
    Round-trip fee expressed as a PRICE offset, scaled by edge_mult. A close
    must beat its entry by at least this to guarantee a net-positive realized
    P&L after both legs' fees. Process (c).
    """
    maker_rate = fees.get("maker", 0.0)
    taker_rate = fees.get("taker", 0.0)
    maker_fee = maker_rate * mid
    taker_fee = taker_rate * mid
    return maker_fee, taker_fee


def _act(kind, qty, role, reason, aggressive):
    return {"kind": kind, "qty": qty, "role": role, "reason": reason,
            "aggressive": aggressive}


def _none(reason):
    return {"kind": "none", "qty": 0.0, "role": None, "reason": reason,
            "aggressive": False}

def decide_exit(cfg, stats, best_bids, sold_price, fees):

    best_bid = stats["best_bid"]
    best_ask = stats["best_ask"]
    idle_period = stats["idle_period"]
    n = len(best_bids)
    if n == 0:
        return _none("no_signal")
    mean = sum(best_bids) / n
    maker_fee, taker_fee = round_trip_fee_offset(cfg, fees, best_bid)

    if sold_price and (sold_price - best_ask - taker_fee) > 0.3:
        return _act("buy", 0.25, "should_exit", "exit_sell", True)
    if sold_price and (sold_price - best_bid - maker_fee) > 0.3:
        return _act("buy", 0.25, "exit", "exit_sell", False)
    if sold_price and (sold_price - best_bid - maker_fee) > 0.1:
        return _act("buy", 0.25, "low_exit", "exit_sell", False)
    return _none("no_signal")

def decide(cfg, stats, sold_price, base_qty, fees, best_asks):

    best_ask = stats["best_ask"]
    previous = stats["previous"]
    median_ask = np.median(list(best_asks))
    mean_ask = stats["mean"]
    max_ask = max(list(best_asks))
    min_ask = min(list(best_asks))
    delta_ask = max_ask - min_ask
    if abs(best_ask - min_ask) < 0.01:
        return _act("sell", base_qty, "entry", "entry_sell", False)    
    return _none("no_signal")


def evaluate_exit_book(cfg, best_bid, best_ask, best_bids, idle_period, time_started, sold_price, fees, vh):
    """
    Single source of truth for agent + tests: compute stats from the rolling
    """

    if time_started < 5100:
        return _none("no_enough_time")

    stats = {"best_bid": best_bid, "best_ask": best_ask, "idle_period": idle_period}
    action = decide_exit(cfg, stats, best_bids, sold_price, fees)

    return action

def evaluate_book(cfg, best_ask, previous, best_asks, time_started, sold_price, base_qty, fees, vh):
    """
    Single source of truth for agent + tests: compute stats from the rolling
    """
    if time_started < 2500:
        return _none("no_enough_time")
    n = len(best_asks)
    if n < cfg["min_samples"]:
        return _none("warmup")
    mean, std = mean_std(best_asks)
    stats = {"best_ask": best_ask, "previous": previous, "mean": mean}
    action = decide(cfg, stats, sold_price, base_qty, fees, best_asks)

    return action

# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class MinerAgent_V4(FinanceSimulationAgent):
    def initialize(self):
        self.cfg = {
            
            "window": 20,
            "min_samples": 20,
            "t_trend_enter": 0.025,
            "t_trend_exit": 0.02,
            # ---- entry / exit (z-units => per-book normalized) ----
            "z_entry": 1.5,          # (a) fade when |z| >= this
            "z_stop": 3.5,           # (d) adverse-stretch stop (only if stop_enabled)
            "max_hold": 4,         # (d) time stop in ticks (only if stop_enabled)
            "edge_mult": 1.0,        # (c) require edge >= edge_mult * round-trip fee
            # ---- inventory ----
            "inv_cap_frac": 0.05,
            "min_qty": 0.25,
            "max_qty": 3,
            "base_qty_fixed": 30,  # keep V2's fixed base size for uniformity
            # ---- order handling ----
            "entry_ttl_intervals": 20,
            # ---- fallbacks ----
            "default_maker_fee": 0.0,
            "default_taker_fee": 0.0002,
            "default_publish_interval_ns": 1_000_000_000,
            "max_inv": 0.75,
        }
        # per-(validator, book) state
        self.mids = defaultdict(lambda: deque(maxlen=self.cfg["window"]))
        self.best_bids = defaultdict(lambda: deque(maxlen=self.cfg["window"]))
        self.best_asks = defaultdict(lambda: deque(maxlen=self.cfg["window"]))
        self.bought_prices = defaultdict(lambda: deque(maxlen=35))
        self.sold_prices = defaultdict(lambda: deque(maxlen=35))
        self.last_traded_price = defaultdict(float)
        self.c_trade = defaultdict(bool)
        self.prev_mid = {}
        self.prev_bid = {}
        self.prev_ask = {}
        self.base_line = {}
        self.time_started = 0
        self.idle_ticks = defaultdict(int)     # ticks since last order on a book
        self.trade_ticks = defaultdict(int)
        self._load_state()

    # ---- helpers ---------------------------------------------------------- #
    def _validator(self, state):
        return state.dendrite.hotkey

    def _state_path(self) -> str:
        """Path to the persisted state file, unique per miner UID."""
        os.makedirs("agent_state", exist_ok=True)
        return f"agent_state/state_{self.uid}.json"

    def _save_state(self) -> None:
        """
        Persist bought_prices, sold_prices, mids, last_traded_price,
        and prev_mid to disk as JSON.
        Called at the end of every respond() tick.
        """
        data = {
            "bought_prices": {
                str(k): list(v)
                for k, v in self.bought_prices.items()
            },
            "sold_prices": {
                str(k): list(v)
                for k, v in self.sold_prices.items()
            },
            "best_bids": {
                str(k): list(v)
                for k, v in self.best_bids.items()
            },
            "best_asks": {
                str(k): list(v)
                for k, v in self.best_asks.items()
            },
            "last_traded_price": {
                str(k): v
                for k, v in self.last_traded_price.items()
            },
            "prev_bid": {
                str(k): v
                for k, v in self.prev_bid.items()
            },
            "prev_ask": {
                str(k): v
                for k, v in self.prev_ask.items()
            },
            "idle_ticks": {
                str(k): v
                for k, v in self.idle_ticks.items()
            },
            "c_trade": {
                str(k): v
                for k, v in self.c_trade.items()
            },
            "time_started": self.time_started,
        }
        tmp_path = self._state_path() + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, self._state_path())   # atomic write — no corruption on crash

    def _load_state(self) -> None:
        """
        Reload persisted state on startup.
        Called once at the end of initialize().
        Silently skips if no state file exists yet (first run).
        """
        path = self._state_path()
        if not os.path.exists(path):
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupted file — start fresh
            return

        # Keys were serialized as strings — convert back to tuples
        def to_key(s: str):
            # Keys are stored as "(vh, book_id)" string representation
            # e.g. "('5EWwd...', 42)"
            try:
                return eval(s)   # safe here — we wrote these ourselves
            except Exception:
                return s

        maxlen_bp = self.bought_prices.default_factory().maxlen    # 35
        maxlen_sp = self.sold_prices.default_factory().maxlen      # 35
        maxlen_m  = self.mids.default_factory().maxlen             # 50

        for k_str, v in data.get("bought_prices", {}).items():
            self.bought_prices[to_key(k_str)] = deque(v, maxlen=maxlen_bp)

        for k_str, v in data.get("sold_prices", {}).items():
            self.sold_prices[to_key(k_str)] = deque(v, maxlen=maxlen_sp)

        for k_str, v in data.get("best_bids", {}).items():
            self.best_bids[to_key(k_str)] = deque(v, maxlen=maxlen_m)
        for k_str, v in data.get("best_asks", {}).items():
            self.best_asks[to_key(k_str)] = deque(v, maxlen=maxlen_m)

        for k_str, v in data.get("last_traded_price", {}).items():
            self.last_traded_price[to_key(k_str)] = v

        for k_str, v in data.get("prev_bid", {}).items():
            self.prev_bid[to_key(k_str)] = v
        for k_str, v in data.get("prev_ask", {}).items():
            self.prev_ask[to_key(k_str)] = v

        for k_str, v in data.get("idle_ticks", {}).items():
            self.idle_ticks[to_key(k_str)] = v

        for k_str, v in data.get("c_trade", {}).items():
            self.c_trade[to_key(k_str)] = v

        self.time_started = data.get("time_started", 0)

    def _initial_base(self, key, base_balance):
        init = getattr(base_balance, "initial", None)
        if init is not None:
            return init
        if key not in self.base_line:
            self.base_line[key] = base_balance.total
        return self.base_line[key]

    def _net_inventory(self, key, base_balance):
        inv = base_balance.total - self._initial_base(key, base_balance)
        return inv

    def _fees(self, account):
        f = getattr(account, "fees", None)
        maker = getattr(f, "maker_fee_rate", None) if f else None
        taker = getattr(f, "taker_fee_rate", None) if f else None
        return {"maker": self.cfg["default_maker_fee"] if maker is None else maker,
                "taker": self.cfg["default_taker_fee"] if taker is None else taker}

    # ---- main loop -------------------------------------------------------- #
    def respond(self, state):
        vh = self._validator(state)
        response = FinanceAgentResponse(agent_id=self.uid)
        vol_dec = getattr(state.config, "volumeDecimals", 4)
        price_dec = getattr(state.config, "priceDecimals", 2)
        pub_ns = getattr(state.config, "publish_interval",
                         self.cfg["default_publish_interval_ns"])
        entry_ttl = int(self.cfg["entry_ttl_intervals"] * pub_ns)
        current_time = state.timestamp
        self.time_started = self.time_started + 1 if self.time_started < 7000 else 7000

        for book_id, book in state.books.items():
            if not book.bids or not book.asks:
                continue
            best_bid = round(book.bids[0].p, price_dec)
            best_ask = round(book.asks[0].p, price_dec)
            mid = (best_bid + best_ask) / 2
            mid = round(mid, price_dec)
            spread = best_ask - best_bid
            key = (vh, book_id)

            account = state.accounts[self.uid][book_id]
            base_balance = account.base_balance
            initial_base = self._initial_base(key, base_balance)
            net_inv = self._net_inventory(key, base_balance)
            fees = self._fees(account)
            maker_fee, taker_fee = round_trip_fee_offset(self.cfg, fees, mid)
            base_qty = self.cfg["base_qty_fixed"]
            sold_prices = list(self.sold_prices[key])
            bought_prices = list(self.bought_prices[key])
            
            if net_inv >= 0.2:
                self.sold_prices[key].clear()
                a = int((net_inv + 29.8) / base_qty)
                last_traded_price = self.last_traded_price.get(key, best_ask)
                if base_balance.total < 3:
                    self.c_trade[key] = False
                    self.idle_ticks[key] = current_time
                if len(bought_prices) < a:
                    self.bought_prices[key].append(last_traded_price)
                if len(bought_prices) > a:
                    self.bought_prices[key].popleft()
            if net_inv <= -0.2:
                self.bought_prices[key].clear()
                a = int((29.8 - net_inv) / base_qty)
                last_traded_price = self.last_traded_price.get(key, best_bid)
                if base_balance.total < 3:
                    self.c_trade[key] = False
                    self.idle_ticks[key] = current_time
                if len(sold_prices) < a:
                    self.sold_prices[key].append(last_traded_price)
                if len(sold_prices) > a:
                    self.sold_prices[key].popleft()
            if abs(net_inv) < 0.2:
                self.bought_prices[key].clear()
                self.sold_prices[key].clear()

            mids = self.mids[key]
            best_bids = self.best_bids[key]
            best_asks = self.best_asks[key]
            if not best_asks:
                self.mids[key].append(mid)
                self.best_bids[key].append(best_bid)
                self.best_asks[key].append(best_ask)
                self.prev_bid[key] = best_bid
                self.prev_ask[key] = best_ask
                continue
            
            max_bid = max(best_bids)
            max_ask = max(best_asks)
            min_bid = min(best_bids)
            min_ask = min(best_asks)
            delta_bid = min(max((max_bid - min_bid) / 20, 0.25), 0.3)
            delta_ask = min(max((max_ask - min_ask) / 20, 0.25), 0.3)
            previous_bid = self.prev_bid.get(key, best_bid)
            previous_ask = self.prev_ask.get(key, best_ask)
            if abs(best_bid - previous_bid) > delta_bid:
                self.best_bids[key].append(best_bid)
                self.prev_bid[key] = best_bid
            if abs(best_ask - previous_ask) > delta_ask:
                self.best_asks[key].append(best_ask)
                self.prev_ask[key] = best_ask
            
            if best_bid > max_bid:
                self.best_bids[key][-1] = best_bid
            if best_ask > max_ask:
                self.best_asks[key][-1] = best_ask
            if best_bid < min_bid:
                self.best_bids[key][-1] = best_bid
            if best_ask < min_ask:
                self.best_asks[key][-1] = best_ask

            if self.time_started == 2000:
                self.idle_ticks[key] = current_time

            sold_prices = list(self.sold_prices[key])
            bought_prices = list(self.bought_prices[key])
            first_bought_price = bought_prices[0] if len(bought_prices) else 0
            last_bought_price = bought_prices[-1] if len(bought_prices) else 0
            first_sold_price = sold_prices[0] if len(sold_prices) else 0
            last_sold_price = sold_prices[-1] if len(sold_prices) else 0
            idle_tick = self.idle_ticks.get(key, current_time)
            
            if vh == "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF":
                print(f"book_id {book_id}: time_dif: {current_time - idle_tick}, previous_bid: {previous_bid}, previous_ask: {previous_ask} best_bid: {best_bid}, best_ask: {best_ask}, bought: {bought_prices}, sold: {sold_prices}, net_inv: {net_inv} len: {len(best_bids)} time_started: {self.time_started}")
            
            idle_period = current_time - idle_tick
            action = evaluate_exit_book(
                self.cfg, best_bid, best_ask, self.best_bids[key], idle_period, self.time_started, first_sold_price, fees, vh
                )

            if action["role"] == "should_exit" and self.c_trade.get(key, False) == False:    
                response.market_order(
                    book_id=book_id,
                    direction=OrderDirection.BUY,
                    quantity=0.25
                )
                continue
            if action["role"] == "exit" and self.c_trade.get(key, False) == False:
                price = round(best_bid + 0.01, price_dec)
                response.limit_order(
                    book_id=book_id,
                    direction=OrderDirection.BUY,
                    quantity=0.25,
                    price=price,
                    timeInForce=TimeInForce.GTT,
                    expiryPeriod=entry_ttl,
                )
                continue
            if action["role"] == "low_exit" and self.c_trade.get(key, False) == False:
                price = round(first_sold_price - 0.3 + maker_fee, price_dec)
                response.limit_order(
                    book_id=book_id,
                    direction=OrderDirection.BUY,
                    quantity=0.25,
                    price=price,
                    timeInForce=TimeInForce.GTT,
                    expiryPeriod=entry_ttl,
                )
                continue

            action = evaluate_book(
                self.cfg, best_ask, previous_ask, self.best_asks[key], self.time_started, last_sold_price, base_qty, fees, vh
            )

            # qty = round(action["qty"], vol_dec)
            # if qty <= 0:
            #     continue

            if (action["role"] == "entry" and self.time_started < 5400) or (self.time_started > 4000 and self.time_started < 4500):
                if base_balance.total > 1:
                    price = round(best_ask - 0.01, price_dec)
                    response.limit_order(
                        book_id=book_id,
                        direction=OrderDirection.SELL,
                        quantity=max(0.25, min(base_qty, base_balance.total - 1)),
                        price=price,
                        timeInForce=TimeInForce.GTT,
                        expiryPeriod=entry_ttl,
                    )
                    self.last_traded_price[key] = price + round(maker_fee, price_dec)
                continue
            
            if idle_period > 2000000000000 and self.time_started > 5400 and base_balance.total > 1 and action["role"] == "entry":
                self.c_trade[key] = True
            if self.c_trade.get(key, False):
                price = round(best_ask - 0.01, price_dec)
                response.limit_order(
                    book_id=book_id,
                    direction=OrderDirection.SELL,
                    quantity=max(0.25, min(base_qty, base_balance.total - 1)),
                    price=price,
                    timeInForce=TimeInForce.GTT,
                    expiryPeriod=entry_ttl,
                )
                self.last_traded_price[key] = price + round(maker_fee, price_dec)
                continue
        self._save_state()
        return response

if __name__ == "__main__":
    from taos.common.agents import launch
    launch(MinerAgent_V4)