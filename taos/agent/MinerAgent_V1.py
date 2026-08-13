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
    # rt_rate = fees.get("maker", 0.0)
    rt_rate = fees.get("taker", 0.0)
    fee = rt_rate * mid
    return fee


def _act(kind, qty, role, reason, aggressive):
    return {"kind": kind, "qty": qty, "role": role, "reason": reason,
            "aggressive": aggressive}


def _none(reason):
    return {"kind": "none", "qty": 0.0, "role": None, "reason": reason,
            "aggressive": False}

def decide_exit(cfg, stats, bought_price, sold_price, base_qty, net_inv, mids, fees):

    mid = stats["mid"]
    mids = list(mids)
    best_ask = stats["best_ask"]
    best_bid = stats["best_bid"]
    fee_off = round_trip_fee_offset(cfg, fees, mid)
    max_mid = max(mids)
    min_mid = min(mids)
    offset = (max_mid - min_mid) / 2

    gcfs_m = best_bid > bought_price + fee_off + 1 if bought_price else False    
    gcfb_m = best_ask < sold_price - fee_off - 1 if sold_price else False

    if bought_price and gcfs_m and mids[-1] >= mid:
        return _act("sell", min(base_qty, 0.25), "should_exit", "should_exit_sell", True)

    return _none("no_signal")

def decide(cfg, stats, bought_price, sold_price, net_inv, base_qty, fees, mids):

    mid = stats["mid"]
    best_ask = stats["best_ask"]
    previous = stats["previous"]
    spread = stats["spread"]
    fee_off = round_trip_fee_offset(cfg, fees, mid)
    median_mid = np.median(list(mids))
    mean_mid = stats["mean"]
    max_mid = max(mids)
    min_mid = min(mids)
    if mid > previous  and mid < (mean_mid + min_mid) / 2 and previous != min_mid and (max_mid - best_ask) > (max_mid- min_mid) * 0.7 and (max_mid - best_ask) > 2:
        if net_inv > -0.2 and net_inv < 30:
            return _act("buy", 3, "entry", "z_buy", False)
        else:
            return _none("sold_exist_buy")
    
    return _none("no_signal")


def evaluate_exit_book(cfg, mid, best_ask, best_bid, mids,
                  bought_price, sold_price, base_qty, net_inv, fees, trade_ticks, vh):
    """
    Single source of truth for agent + tests: compute stats from the rolling
    """

    stats = {"mid": mid, "best_ask": best_ask, "best_bid": best_bid}
    action = decide_exit(cfg, stats, bought_price, sold_price, base_qty, net_inv, mids, fees)

    return action

def evaluate_book(cfg, mid, best_ask, previous, spread, mids, time_started,
                  bought_price, sold_price, net_inv, base_qty, fees, trade_ticks, vh):
    """
    Single source of truth for agent + tests: compute stats from the rolling
    """
    # if time_started < 600:
    #     return _none("no_enough_time")
    # n = len(mids)
    # if n < cfg["min_samples"]:
    #     return _none("warmup")
    mean, std = mean_std(mids)
    if trade_ticks <= cfg["max_hold"]:
        return _none("tight_trade")
    stats = {"mid": mid, "best_ask": best_ask, "previous": previous, "mean": mean, "spread": spread}
    action = decide(cfg, stats, bought_price, sold_price, net_inv, base_qty, fees, mids)

    return action


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class MinerAgent_V1(FinanceSimulationAgent):
    def initialize(self):
        self.cfg = {
            
            "window": 50,
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
            "base_qty_fixed": 3,  # keep V2's fixed base size for uniformity
            # ---- order handling ----
            "entry_ttl_intervals": 4,
            # ---- fallbacks ----
            "default_maker_fee": 0.0,
            "default_taker_fee": 0.0002,
            "default_publish_interval_ns": 1_000_000_000,
            "max_inv": 0.75,
        }
        # per-(validator, book) state
        self.mids = defaultdict(lambda: deque(maxlen=self.cfg["window"]))
        self.bought_prices = defaultdict(lambda: deque(maxlen=35))
        self.sold_prices = defaultdict(lambda: deque(maxlen=35))
        self.last_traded_price = defaultdict(float)
        self.net_inv_offset = defaultdict(float)
        self.prev_mid = {}
        self.base_line = {}
        self.sell_count = 0
        self.initial_phase = defaultdict(bool)
        self.second_phase = defaultdict(bool)
        self.third_phase = False
        self.main_phase = defaultdict(bool)
        # V3 additions
        self.time_started = 0
        self.trade_ticks = defaultdict(int)        # ticks current net position held
        self.idle_ticks = defaultdict(int)     # ticks since last order on a book

    # ---- helpers ---------------------------------------------------------- #
    def _validator(self, state):
        return state.dendrite.hotkey

    def _initial_base(self, key, base_balance):
        init = getattr(base_balance, "initial", None)
        if init is not None:
            return init
        if key not in self.base_line:
            self.base_line[key] = base_balance.total
        return self.base_line[key]

    def _net_inventory(self, key, base_balance, offset):
        inv = base_balance.total - self._initial_base(key, base_balance) - offset
        if abs(inv) < 0.35:
            threshold = 0.8
        if abs(inv) < 0.6 and abs(inv) >= 0.35:
            threshold = 0.5
        if abs(inv) >= 0.6:
            threshold = 0.2
        return inv, threshold

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
        self.sell_count = 0

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
            offset = self.net_inv_offset.get(key, 0)
            net_inv, threshold = self._net_inventory(key, base_balance, offset)
            fees = self._fees(account)
            fee_off = round_trip_fee_offset(self.cfg, fees, mid)
            base_qty = self.cfg["base_qty_fixed"]
            sold_prices = list(self.sold_prices[key])
            bought_prices = list(self.bought_prices[key])

            # if self.time_started == 1:
            #     response.market_order(
            #         book_id=book_id,
            #         direction=OrderDirection.SELL,
            #         quantity=0.25
            #     )
            #     continue
            
            if net_inv > 0:
                self.sold_prices[key].clear()
                a = int((net_inv + 2.7) / base_qty)
                last_traded_price = self.last_traded_price.get(key, max(315, best_ask))
                if len(bought_prices) < a:
                    self.bought_prices[key].append(last_traded_price)
                    self.trade_ticks[key] = 5
                    self.idle_ticks[key] = current_time
                if len(bought_prices) > a:
                    self.bought_prices[key].popleft()
                    self.trade_ticks[key] = 5
                    self.idle_ticks[key] = current_time
            if net_inv < 0:
                self.bought_prices[key].clear()
                a = int((2.7 - net_inv) / base_qty)
                last_traded_price = self.last_traded_price.get(key, max(315, best_bid))
                if len(sold_prices) < a:
                    self.sold_prices[key].append(last_traded_price)
                    self.trade_ticks[key] = 5
                    self.idle_ticks[key] = current_time
                if len(sold_prices) > a:
                    self.sold_prices[key].popleft()
                    self.trade_ticks[key] = 5
                    self.idle_ticks[key] = current_time
            if net_inv == 0:
                self.bought_prices[key].clear()
                self.sold_prices[key].clear()

            mids = self.mids[key]

            if not mids:
                self.mids[key].append(mid)
                self.prev_mid[key] = mid
                continue

            max_mid = max(mids)
            min_mid = min(mids)
            mean_mid = sum(mids) / len(mids)
            delta_mid = min(max((max_mid - min_mid) / 20, 0.3), 0.4)
            previous_mid = self.prev_mid.get(key, mid)

            if abs(mid - previous_mid) > delta_mid:
                self.mids[key].append(mid)
                self.prev_mid[key] = mid

            if self.time_started < 5:
                if book_id == 127:
                    print(f"book_id: {book_id} time_started: {self.time_started} warmup")
                continue

            if self.time_started == 5:
                self.initial_phase[key] = True
                # self.main_phase[key] = True

            if self.initial_phase.get(key, False) and self.time_started < 4800:
                if book_id == 127:
                    print(f"book_id: {book_id} intial_phase, net_inv: {net_inv}")
                if net_inv < 59 and net_inv > -0.3:
                    price = round(best_ask, price_dec)
                    response.market_order(
                        book_id=book_id,
                        direction=OrderDirection.BUY,
                        quantity=base_qty,
                    )
                    self.last_traded_price[key] = price
                continue

            if self.initial_phase.get(key, False) and self.time_started >= 5000 and not self.second_phase.get(key, False):
                self.initial_phase[key] = False
                self.second_phase[key] = True
                if book_id == 127:
                    print(f"book_id: {book_id} second_phase, net_inv: {net_inv}")
                continue

            sold_prices = list(self.sold_prices[key])
            bought_prices = list(self.bought_prices[key])
            first_bought_price = bought_prices[0] if len(bought_prices) else 0
            last_bought_price = bought_prices[-1] if len(bought_prices) else 0
            first_sold_price = sold_prices[0] if len(sold_prices) else 0
            last_sold_price = sold_prices[-1] if len(sold_prices) else 0
            idle_tick = self.idle_ticks.get(key, current_time)

            if self.third_phase and self.second_phase.get(key, False):
                self.second_phase[key] = False

            if self.second_phase.get(key, False) and  net_inv >= 0.2 and not self.third_phase:
                if best_bid > last_bought_price + fee_off + 0.3:
                    self.sell_count += 1
                if self.sell_count >= 65 or (self.sell_count >= 50 and self.time_started >= 6000) or self.time_started >= 7000:
                    self.third_phase = True
                    continue
                else:
                    continue
            if self.third_phase:
                if best_bid > last_bought_price + fee_off + 0.2:
                    response.market_order(
                        book_id=book_id,
                        direction=OrderDirection.SELL,
                        quantity=0.25
                    )
                
                if book_id == 127:
                    print(f"book_id: {book_id} third_phase, net_inv: {net_inv} time_started: {self.time_started}")
                continue
            
            if self.third_phase and net_inv < 0.2 and net_inv > -0.2 and not self.main_phase.get(key, False):
                self.main_phase[key] = True
                if book_id == 127:
                    print(f"book_id: {book_id} main_phase, net_inv: {net_inv} time_started: {self.time_started}")

            if self.third_phase and self.time_started >= 7500 and not self.main_phase.get(key, False):
                self.third_phase = False
                self.main_phase[key] = True
                if book_id == 127:
                    print(f"book_id: {book_id} main_phase, net_inv: {net_inv} time_started: {self.time_started}")

            if self.third_phase and self.time_started < 7500:
                if book_id == 127:
                    print(f"book_id: {book_id} third_phase, net_inv: {net_inv} time_started: {self.time_started}")
                continue
            
            if current_time - idle_tick > 10800000000000:
                self.net_inv_offset[key] = net_inv
                self.bought_prices[key].clear()
                self.sold_prices[key].clear()
            
            self.trade_ticks[key] += 1
            
            action = evaluate_exit_book(
                self.cfg, mid, best_ask, best_bid, self.mids[key], first_bought_price, first_sold_price,
                base_qty, net_inv, fees, self.trade_ticks[key], vh
            )

            qty = round(action["qty"], vol_dec)

            if action["role"] == "should_exit":    
                if action["kind"] == "sell":
                    price = round(mid, price_dec)
                    response.market_order(
                        book_id=book_id,
                        direction=OrderDirection.SELL,
                        quantity=qty
                    )
                else:
                    price = round(mid, price_dec)
                    response.market_order(
                        book_id=book_id,
                        direction=OrderDirection.BUY,
                        quantity=qty
                    )
                continue

            if fee_off > 0:
                continue
            
            action = evaluate_book(
                self.cfg, mid, best_ask, previous_mid, spread, self.mids[key], self.time_started, last_bought_price, last_sold_price, net_inv,
                base_qty, fees, self.trade_ticks[key], vh
            )

            qty = round(action["qty"], vol_dec)
            if qty <= 0:
                continue

            if action["role"] == "entry":
                if action["kind"] == "sell":
                    price = round(best_bid, price_dec)
                    response.market_order(
                        book_id=book_id,
                        direction=OrderDirection.SELL,
                        quantity=qty,
                        # price=price,
                        # timeInForce=TimeInForce.GTT,
                        # expiryPeriod=entry_ttl,
                    )
                    self.trade_ticks[key] = 2
                    self.last_traded_price[key] = price - round(fee_off, price_dec)
                else:
                    price = round(best_ask, price_dec)
                    response.market_order(
                        book_id=book_id,
                        direction=OrderDirection.BUY,
                        quantity=qty,
                        # price=price,
                        # timeInForce=TimeInForce.GTT,
                        # expiryPeriod=entry_ttl,
                    )
                    self.trade_ticks[key] = 2
                    self.last_traded_price[key] = price + round(fee_off, price_dec)
                continue
        return response

if __name__ == "__main__":
    from taos.common.agents import launch
    launch(MinerAgent_V1)