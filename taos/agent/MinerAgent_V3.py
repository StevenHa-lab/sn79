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
    rt_rate = fees.get("maker", 0.0)
    #  + fees.get("taker", 0.0))
    fee = rt_rate * mid
    return fee if fee > 0 else -fee


def _act(kind, qty, role, reason, aggressive):
    return {"kind": kind, "qty": qty, "role": role, "reason": reason,
            "aggressive": aggressive}


def _none(reason):
    return {"kind": "none", "qty": 0.0, "role": None, "reason": reason,
            "aggressive": False}


def decide_exit(cfg, stats, bought_price, sold_price, base_qty, net_inv, fees):

    mid = stats["mid"]
    best_ask = stats["best_ask"]
    best_bid = stats["best_bid"]
    fee_off = round_trip_fee_offset(cfg, fees, mid)
    
    gcfs = mid > bought_price + fee_off + 0.1 if bought_price else False
    gcfs_m = best_bid > bought_price + fee_off + 0.1 if bought_price else False
    
    gcfb = mid < sold_price - fee_off - 0.1 if sold_price else False
    gcfb_m = best_ask < sold_price - fee_off - 0.1 if sold_price else False

    if bought_price and gcfs_m:
        return _act("sell", min(base_qty, net_inv), "should_exit", "should_exit_sell", True)
    
    if bought_price and gcfs:
        return _act("sell", min(base_qty, net_inv), "exit", "exit_sell", False)

    if sold_price and gcfb_m:
        return _act("buy", min(base_qty, -net_inv), "should_exit", "should_exit_buy", True)

    if sold_price and gcfb:
        return _act("buy", min(base_qty, -net_inv), "exit", "exit_buy", False)
    
    return _none("no_signal")

def decide(cfg, stats, bought_price, sold_price, net_inv, base_qty, fees, mids):

    mid = stats["mid"]
    previous = stats["previous"]
    fee_off = round_trip_fee_offset(cfg, fees, mid)
    median_mid = np.median(list(mids))
    mean_mid = stats["mean"]
    max_mid = max(mids)
    min_mid = min(mids)
    if mid < median_mid - fee_off - 0.2 and mid > previous and mid < (mean_mid + min_mid) / 2:
        if net_inv < 9.5 and net_inv > -0.2:
            return _act("buy", base_qty, "entry", "z_buy", False)
        else:
            return _none("sold_exist_buy")

    if mid > median_mid + fee_off + 0.2 and mid < previous and mid > (mean_mid + max_mid) / 2:
        if net_inv < 0.2 and net_inv > -9.5:
            return _act("sell", base_qty, "entry", "z_sell", False)
        else:
            return _none("bought_exist_sell")
    
    return _none("no_signal")


def evaluate_exit_book(cfg, mid, best_ask, best_bid, mids,
                  bought_price, sold_price, base_qty, net_inv, fees, trade_ticks, vh):
    """
    Single source of truth for agent + tests: compute stats from the rolling
    """
    n = len(mids)
    if n < cfg["min_samples"]:
        return _none("warmup")

    if trade_ticks <= cfg["max_hold"]:
        return _none("tight_trade")
    stats = {"mid": mid, "best_ask": best_ask, "best_bid": best_bid}
    action = decide_exit(cfg, stats, bought_price, sold_price, base_qty, net_inv, fees)

    return action

def evaluate_book(cfg, mid, previous, mids,
                  bought_price, sold_price, net_inv, base_qty, fees, trade_ticks, vh):
    """
    Single source of truth for agent + tests: compute stats from the rolling
    """
    n = len(mids)
    if n < cfg["min_samples"]:
        return _none("warmup")
    mean, std = mean_std(mids)
    if trade_ticks <= cfg["max_hold"]:
        return _none("tight_trade")
    stats = {"mid": mid, "previous": previous, "mean": mean}
    action = decide(cfg, stats, bought_price, sold_price, net_inv, base_qty, fees, mids)

    return action


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class MinerAgent_V3(FinanceSimulationAgent):
    def initialize(self):
        self.cfg = {
            
            "window": 50,
            "min_samples": 30,
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
            "max_qty": 1,
            "base_qty_fixed": 0.5,  # keep V2's fixed base size for uniformity
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
        self.bought_prices = defaultdict(lambda: deque(maxlen=20))
        self.sold_prices = defaultdict(lambda: deque(maxlen=20))
        self.last_traded_price = defaultdict(float)
        self.net_inv_offset = defaultdict(float)
        self.prev_mid = {}
        self.base_line = {}
        # V3 additions
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
            
            if net_inv > 0:
                self.sold_prices[key].clear()
                a = int((net_inv + 0.3) / base_qty)
                last_traded_price = self.last_traded_price.get(key, 200)
                if len(bought_prices) < a:
                    self.bought_prices[key].append(last_traded_price)
                    self.idle_ticks[key] = current_time
                    self.trade_ticks[key] = 5
                if len(bought_prices) > a:
                    self.bought_prices[key].popleft()
                    self.idle_ticks[key] = current_time
                    self.trade_ticks[key] = 5
            if net_inv < 0:
                self.bought_prices[key].clear()
                a = int((0.3 - net_inv) / base_qty)
                last_traded_price = self.last_traded_price.get(key, 400)
                if len(sold_prices) < a:
                    self.sold_prices[key].append(last_traded_price)
                    self.idle_ticks[key] = current_time
                    self.trade_ticks[key] = 5
                if len(sold_prices) > a:
                    self.sold_prices[key].popleft()
                    self.idle_ticks[key] = current_time
                    self.trade_ticks[key] = 5
            if net_inv == 0:
                self.bought_prices[key].clear()
                self.sold_prices[key].clear()

            mids = self.mids[key]
            previous_mid = self.prev_mid.get(key, mid)
            sold_prices = list(self.sold_prices[key])
            bought_prices = list(self.bought_prices[key])
            first_bought_price = bought_prices[0] if len(bought_prices) else 0
            last_bought_price = bought_prices[-1] if len(bought_prices) else 0
            first_sold_price = sold_prices[0] if len(sold_prices) else 0
            last_sold_price = sold_prices[-1] if len(sold_prices) else 0
            idle_tick = self.idle_ticks.get(key, current_time)
            
            if not mids:
                self.mids[key].append(mid)
                self.prev_mid[key] = mid
                continue
            
            if current_time - idle_tick > 10800000000000:
                self.net_inv_offset[key] = net_inv
                self.bought_prices[key].clear()
                self.sold_prices[key].clear()
   
            if vh == "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF":
                print(f"book_id {book_id}: time_dif: {current_time - idle_tick}, previous_mid: {previous_mid}, mid: {mid}, bought: {bought_prices}, sold: {sold_prices}, net_inv: {net_inv} fees: {fee_off} mid_len: {len(mids)}")
            
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
                    self.trade_ticks[key] = 3
                else:
                    price = round(mid, price_dec)
                    response.market_order(
                        book_id=book_id,
                        direction=OrderDirection.BUY,
                        quantity=qty
                    )
                    self.trade_ticks[key] = 3
                continue

            if action["role"] == "exit":
                price = round(mid, price_dec)
                if action["kind"] == "sell":
                    response.limit_order(
                        book_id=book_id,
                        direction=OrderDirection.SELL,
                        quantity=qty,
                        price=price,
                        timeInForce=TimeInForce.GTT,
                        expiryPeriod=entry_ttl,
                    )
                    self.trade_ticks[key] = 0
                else:
                    price = round(mid, price_dec)
                    response.limit_order(
                        book_id=book_id,
                        direction=OrderDirection.BUY,
                        quantity=qty,
                        price=price,
                        timeInForce=TimeInForce.GTT,
                        expiryPeriod=entry_ttl,
                    )
                    self.trade_ticks[key] = 0
                continue

            if abs(mid - previous_mid) < 0.3:
                continue
            max_mid = max(mids)
            min_mid = min(mids)
            delta_mid = max((max_mid - min_mid) / 20, 0.3)

            if abs(mid - previous_mid) > delta_mid:
                self.mids[key].append(mid)
                self.prev_mid[key] = mid

            if fee_off > 0.2:
                continue
            
            action = evaluate_book(
                self.cfg, mid, previous_mid, self.mids[key], last_bought_price, last_sold_price, net_inv,
                base_qty, fees, self.trade_ticks[key], vh
            )

            qty = round(action["qty"], vol_dec)
            if qty <= 0:
                continue

            if action["role"] == "entry":
                if action["kind"] == "sell":
                    price = round(mid, price_dec)
                    response.limit_order(
                        book_id=book_id,
                        direction=OrderDirection.SELL,
                        quantity=qty,
                        price=price,
                        timeInForce=TimeInForce.GTT,
                        expiryPeriod=entry_ttl,
                    )
                    self.trade_ticks[key] = 0
                    self.last_traded_price[key] = price - round(fee_off, price_dec)
                else:
                    price = round(mid, price_dec)
                    response.limit_order(
                        book_id=book_id,
                        direction=OrderDirection.BUY,
                        quantity=qty,
                        price=price,
                        timeInForce=TimeInForce.GTT,
                        expiryPeriod=entry_ttl,
                    )
                    self.trade_ticks[key] = 0
                    self.last_traded_price[key] = price + round(fee_off, price_dec)
                continue
        return response


if __name__ == "__main__":
    from taos.common.agents import launch
    launch(MinerAgent_V3)