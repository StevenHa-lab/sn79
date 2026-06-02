import os
import numpy as np
from pathlib import Path
import pandas as pd
import csv
import bittensor as bt
from collections import defaultdict, deque
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol.response import FinanceAgentResponse, OrderDirection, TimeInForce, LoanSettlementOption, STP
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse, FinanceEventNotification
from taos.im.protocol.events import LimitOrderPlacementEvent, MarketOrderPlacementEvent, OrderCancellationEvent, TradeEvent
from taos.im.utils import duration_from_timestamp, timestamp_from_duration

class MinerAgent(FinanceSimulationAgent):
    def initialize(self):
        self.min_spread = 0.01
        self.history_dirs = {}
        self.price_buffers = {}        # in-memory, no disk I/O per tick
        self.order_tracker = {}        # track open orders to cancel stale ones
        self.window = 50
        self.vol_window = 20
        self.target_inventory = 0.5    # target base ratio (0=all quote, 1=all base)
        self.max_position_pct = 0.8    # max % of capital in base

    # --- Core improvements ---

    def get_mid_and_spread(self, book):
        best_bid = book.bids[0].p
        best_ask = book.asks[0].p
        return (best_bid + best_ask) / 2, best_ask - best_bid

    def get_price_buffer(self, key):
        if key not in self.price_buffers:
            self.price_buffers[key] = deque(maxlen=self.window)
        return self.price_buffers[key]

    def compute_volatility(self, prices):
        if len(prices) < 3:
            return 0.01
        arr = np.array(prices)
        returns = np.diff(arr) / arr[:-1]
        return np.std(returns[-self.vol_window:]) + 1e-8

    def compute_trend(self, prices):
        """Simple linear regression slope as trend signal"""
        if len(prices) < 5:
            return 0.0
        y = np.array(prices[-10:])
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        return slope / (np.mean(y) + 1e-8)   # normalized slope

    def respond(self, state):
        response = FinanceAgentResponse(agent_id=self.uid)
        price_decimals = getattr(state.config, "priceDecimals", 2)
        vol_decimals = getattr(state.config, "volumeDecimals", 4)

        for book_id, book in state.books.items():
            if not book.bids or not book.asks:
                continue

            best_bid = book.bids[0].p
            best_ask = book.asks[0].p
            mid, raw_spread = self.get_mid_and_spread(book)
            account = state.accounts[self.uid][book_id]

            own_base = account.own_base
            own_quote = account.own_quote
            portfolio_value = own_quote + mid * own_base
            base_ratio = (mid * own_base) / (portfolio_value + 1e-8)

            # --- Update in-memory price buffer ---
            buf = self.get_price_buffer((state.dendrite.hotkey, book_id))
            buf.append(mid)
            prices = list(buf)

            vol = self.compute_volatility(prices)
            trend = self.compute_trend(prices)

            # --- Cancel stale orders first ---
            response.cancel_all_orders(book_id=book_id)

            # --- Dynamic spread: widen in high vol, tighten in low vol ---
            spread_multiplier = max(1.0, vol * 500)
            half_spread = round(max(self.min_spread, raw_spread * spread_multiplier) / 2,
                                price_decimals)

            # --- Inventory skew: skew quotes to mean-revert position ---
            skew = (base_ratio - self.target_inventory) * mid * 0.5
            bid_price = round(mid - half_spread - skew, price_decimals)
            ask_price = round(mid + half_spread - skew, price_decimals)

            # --- Trend filter: lean directionally when trend is strong ---
            trend_threshold = 0.0003
            base_qty = round(
                min(0.6, max(0.1, portfolio_value * 0.01 / mid)),
                vol_decimals
            )

            if trend > trend_threshold:
                # Bullish: place more aggressive buy, wider ask
                response.limit_order(book_id=book_id,
                                     direction=OrderDirection.BUY,
                                     quantity=round(base_qty * 1.5, vol_decimals),
                                     price=bid_price,
                                     timeInForce=TimeInForce.GTC)
                response.limit_order(book_id=book_id,
                                     direction=OrderDirection.SELL,
                                     quantity=base_qty,
                                     price=ask_price,
                                     timeInForce=TimeInForce.GTC)

            elif trend < -trend_threshold:
                # Bearish: place more aggressive sell
                response.limit_order(book_id=book_id,
                                     direction=OrderDirection.BUY,
                                     quantity=base_qty,
                                     price=bid_price,
                                     timeInForce=TimeInForce.GTC)
                response.limit_order(book_id=book_id,
                                     direction=OrderDirection.SELL,
                                     quantity=round(base_qty * 1.5, vol_decimals),
                                     price=ask_price,
                                     timeInForce=TimeInForce.GTC)
            else:
                # Neutral: symmetric market making
                response.limit_order(book_id=book_id,
                                     direction=OrderDirection.BUY,
                                     quantity=base_qty,
                                     price=bid_price,
                                     timeInForce=TimeInForce.GTC)
                response.limit_order(book_id=book_id,
                                     direction=OrderDirection.SELL,
                                     quantity=base_qty,
                                     price=ask_price,
                                     timeInForce=TimeInForce.GTC)

            # --- Hard inventory guard (replaces the crude market dump) ---
            if base_ratio > self.max_position_pct:
                excess = own_base - (self.max_position_pct * portfolio_value / mid)
                response.market_order(book_id=book_id,
                                      direction=OrderDirection.SELL,
                                      quantity=round(excess * 0.5, vol_decimals))

        return response
    
if __name__ == "__main__":
    from taos.common.agents import launch
    launch(MinerAgent)