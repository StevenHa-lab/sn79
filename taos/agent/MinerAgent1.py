import os
import numpy as np
from pathlib import Path
import bittensor as bt
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol.response import FinanceAgentResponse, OrderDirection, TimeInForce, LoanSettlementOption, STP
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse, FinanceEventNotification
from taos.im.protocol.events import LimitOrderPlacementEvent, MarketOrderPlacementEvent, OrderCancellationEvent, TradeEvent

MAX_LEN = 1000

class PriceArray:
    """
    Manages the price history array for a single book.

    Invariant:
        - prices[0] is always the most recent OPPOSITE extreme
          relative to the latest incoming price direction.
        - When a new MAX arrives  → array starts from the most recent MIN
        - When a new MIN arrives  → array starts from the most recent MAX
        - max_price / min_price are updated continuously.
    """

    def __init__(self, max_len: int = 1000):
        self.prices: list[float] = []
        self.max_price: float | None = None
        self.min_price: float | None = None
        self.max_len = max_len

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def avg(self) -> float | None:
        if self.max_price is None or self.min_price is None:
            return None
        return (self.max_price + self.min_price) / 2.0

    @property
    def range(self) -> float:
        if self.max_price is None or self.min_price is None:
            return 0.0
        return self.max_price - self.min_price

    def __len__(self):
        return len(self.prices)

    def _enforce_max_length(self) -> None:
        """
        When prices exceeds MAX_LEN, reanchor the array to the most recent
        opposite extreme within the kept window.

        Strategy:
        - Keep only the last MAX_LEN prices
        - Among those, find the earlier of (new_max_idx, new_min_idx)
            relative to the current tail direction
        - Reanchor from that earlier extreme so prices[0] is always
            the most recent opposite extreme
        """
        if len(self.prices) <= MAX_LEN:
            return

        # Trim to last MAX_LEN prices
        self.prices = self.prices[-MAX_LEN:]

        # Recompute true max/min within the kept window
        self.max_price = max(self.prices)
        self.min_price = min(self.prices)

        # Find the index of max and min within the trimmed window
        max_idx = next(i for i, p in enumerate(self.prices) if p == self.max_price)
        min_idx = next(i for i, p in enumerate(self.prices) if p == self.min_price)

        # The earlier one becomes the new anchor (first element)
        # because the later one is the more recent extreme,
        # meaning the earlier one is the "opposite" pivot
        anchor_idx = min(max_idx, min_idx)
        self.prices = self.prices[anchor_idx:]

    # ------------------------------------------------------------------
    # Core update logic
    # ------------------------------------------------------------------

    def push(self, price: float) -> str:
        """
        Append *price* to the array, update extremes, prune if needed.
        """
        # ── First price ever ───────────────────────────────────────────
        if not self.prices:
            self.prices.append(price)
            self.max_price = price
            self.min_price = price
            return "init"

        self._enforce_max_length()
        
        prev = self.prices[-1]
        self.prices.append(price)

        # ── New maximum ────────────────────────────────────────────────
        if price > self.max_price:
            old_min = self.min_price
            self.max_price = price
            # Prune: keep from the most recent occurrence of old_min onward
            if self.prices[0] == self.max_price:
                self._prune_to_last_occurrence(old_min)
            return "new_max"

        # ── New minimum ────────────────────────────────────────────────
        if price < self.min_price:
            old_max = self.max_price
            self.min_price = price
            # Prune: keep from the most recent occurrence of old_max onward
            if self.prices[0] == self.min_price:
                self._prune_to_last_occurrence(old_max)
            return "new_min"

        # ── Ordinary movement ──────────────────────────────────────────
        if price > prev:
            return "rising"
        if price < prev:
            return "falling"
        return "unchanged"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_to_last_occurrence(self, pivot: float) -> None:
        """
        Delete every element BEFORE the last occurrence of *pivot* in
        self.prices, so the array starts at that pivot.

        Example (new_max case):
            Before: [old_max, …, recent_min, …, new_max]
            After:  [recent_min, …, new_max]
        """
        # Walk backwards to find the last (most recent) occurrence of pivot
        idx = None
        for i in range(len(self.prices) - 2, -1, -1):   # skip the just-appended tail
            if self.prices[i] == pivot:
                idx = i
                break

        if idx is not None and idx > 0:
            self.prices = self.prices[idx:]


class MinerAgent1(FinanceSimulationAgent):

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self):
        # price_arrays[validator_hotkey][book_id] → PriceArray
        self.price_arrays: dict[str, dict[str, PriceArray]] = {}

        # Order-sizing config
        self.min_qty      = 0.1
        self.max_qty      = 2.0
        self.base_rate    = 0.01     # fraction of initial_volume used as base_qty
        self.price_offset = 0.01     # ticks inside spread for aggressive fill

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def respond(self, state):
        validator_hotkey = state.dendrite.hotkey
        response         = FinanceAgentResponse(agent_id=self.uid)

        price_decimals = getattr(state.config, "priceDecimals", 2)
        vol_decimals   = getattr(state.config, "volumeDecimals", 4)

        for book_id, book in state.books.items():

            if not book.bids or not book.asks:
                continue

            best_bid = book.bids[0].p
            best_ask = book.asks[0].p
            mid      = round((best_bid + best_ask) / 2.0, price_decimals)

            account        = state.accounts[self.uid][book_id]
            initial_volume = account.base_balance.initial
            base_qty       = initial_volume * self.base_rate

            # ── Retrieve / create PriceArray ───────────────────────────
            pa = self._get_price_array(validator_hotkey, book_id)

            # ── Update array with current mid, get movement label ──────
            movement = pa.push(mid)

            # Need at least 2 prices to trade
            if movement in ("init",):
                continue

            avg = pa.avg  # guaranteed non-None after first push

            # ── Regime decision ────────────────────────────────────────
            action, signal_distance = self._decide(movement, mid, avg, pa.range)

            if action == "SKIP":
                continue

            # ── Quantity sizing (proportional to signal strength) ──────
            qty = self._size_qty(signal_distance, pa.range, base_qty, vol_decimals)
            
            # ── Emit order ─────────────────────────────────────────────
            if action == "BUY":
                response.limit_order(
                    book_id     = book_id,
                    direction   = OrderDirection.BUY,
                    quantity    = qty,
                    price       = round(best_bid + self.price_offset, price_decimals),
                    timeInForce = TimeInForce.GTC,
                )

            elif action == "SELL":
                response.limit_order(
                    book_id     = book_id,
                    direction   = OrderDirection.SELL,
                    quantity    = qty,
                    price       = round(best_ask - self.price_offset, price_decimals),
                    timeInForce = TimeInForce.GTC,
                )
            
            stale_order_ids = []
            for order in account.o:
                # Cancel if open too long or too far from current touch/mid
                if order.s == 0 and abs(order.p - (best_bid if best_bid is not None else mid)) > 3:
                    stale_order_ids.append(order.i)
                elif order.s == 1 and abs(order.p - (best_ask if best_ask is not None else mid)) > 3:
                    stale_order_ids.append(order.i)
            if len(stale_order_ids) > 0:
                response.cancel_orders(book_id=book_id, order_ids=stale_order_ids)

        return response

    # ------------------------------------------------------------------
    # Regime decision  (pure function – no side effects)
    # ------------------------------------------------------------------

    @staticmethod
    def _decide(
        movement: str,
        price: float,
        avg: float,
        price_range: float,
    ) -> tuple[str, float]:
        """
        Return (action, signal_distance).

        action          : "BUY" | "SELL" | "SKIP"
        signal_distance : abs distance from avg used for qty sizing
        """

        # Priority 1 – new session maximum
        if movement == "new_max":
            if price_range > 3:
                return "SKIP", 0.0
            # Narrow range → genuine breakout → BUY
            return "BUY", 3 - price_range

        # Priority 2 – new session minimum
        if movement == "new_min":
            if price_range > 3:
                return "SKIP", 0.0
            # Narrow range → genuine breakdown → SELL
            return "SELL", 3 - price_range

        # Priority 3 – price rising (not a new extreme)
        if movement == "rising":
            if price < avg:
                # Rising but still below midpoint → upside room → BUY
                return "BUY", avg - price
            else:
                return "SELL", price - avg

        # Priority 4 – price falling (not a new extreme)
        if movement == "falling":
            if price > avg:
                # Falling but still above midpoint → downside room → SELL
                return "SELL", price - avg
            else:
                return "BUY", avg - price

        # Price unchanged
        return "SKIP", 0.0

    # ------------------------------------------------------------------
    # Quantity sizing
    # ------------------------------------------------------------------

    def _size_qty(
        self,
        signal_distance: float,
        range: float,
        base_qty: float,
        vol_decimals: int,
    ) -> float:
        """
        qty = base_qty * (signal_distance / range)
        Clamped to [min_qty, max_qty] and rounded to vol_decimals.
        """
        if range <= 0:
            return self.min_qty

        raw_qty = base_qty * (signal_distance / range)
        qty     = max(self.min_qty, min(self.max_qty, raw_qty))
        return round(qty, vol_decimals)

    # ------------------------------------------------------------------
    # PriceArray registry
    # ------------------------------------------------------------------

    def _get_price_array(self, validator_hotkey: str, book_id: str) -> PriceArray:
        if validator_hotkey not in self.price_arrays:
            self.price_arrays[validator_hotkey] = {}
        if book_id not in self.price_arrays[validator_hotkey]:
            self.price_arrays[validator_hotkey][book_id] = PriceArray()
        return self.price_arrays[validator_hotkey][book_id]


if __name__ == "__main__":
    from taos.common.agents import launch
    launch(MinerAgent1)