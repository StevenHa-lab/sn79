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

import random

class OCO(FinanceSimulationAgent):
    def initialize(self):
        self.min_spread = 0.01
        self.history_dirs = {}
        self.overall_window_size = 35
        self.local_window_size = 30
        self.regime_window = 250
        self.t_threshold = 0.5
        self.status = [0]*128

    def get_validator_hotkey(self, state):
        return state.dendrite.hotkey

    def get_history_dir(self, validator_hotkey):
        if validator_hotkey not in self.history_dirs:
            path = os.path.abspath(f"history/history_{validator_hotkey}_{self.uid}")
            os.makedirs(path, exist_ok=True)
            self.history_dirs[validator_hotkey] = path
        return self.history_dirs[validator_hotkey]

    def get_history_file(self, validator_hotkey, book_id):
        return os.path.join(self.get_history_dir(validator_hotkey), f"price_history_{book_id}.csv")

    def append_price(self, validator_hotkey, book_id, timestamp, best_bid, best_ask, current_balance):
        file_path = self.get_history_file(validator_hotkey, book_id)
        mid_value = round((best_bid + best_ask) / 2, 3)
        with open(file_path, "a") as f:
            f.write(f"{timestamp},{best_bid},{best_ask},{mid_value},{current_balance}\n")

    def trime_price(self, validator_hotkey, book_id):
        file_path = self.get_history_file(validator_hotkey, book_id)
        lines = []
        
        with open(file_path, "r") as f:
            lines = f.readlines()
        with open(file_path, "w") as f:
            f.writelines(lines[-2:])  # Keep only the last two lines

    def load_windowed_history(self, validator_hotkey, book_id):
        file_path = self.get_history_file(validator_hotkey, book_id)
        if not os.path.exists(file_path):
            # Pad with default values if file doesn't exist
            return [(1.0, 1.0, 1.0)]
        
        with open(file_path, "r") as f:
            lines = f.readlines()
            price_pairs = [
                tuple(map(float, line.strip().split(",")[1:4]))
                for line in lines
            ]
        return price_pairs

    def regime_detection(self, validator_hotkey, book_id):
        price_history = self.load_windowed_history(validator_hotkey, book_id)
        bids = [p[0] for p in price_history]
        asks = [p[1] for p in price_history]
        prices = [p[2] for p in price_history]
        prices = np.asarray(prices, dtype=float)
        # Need at least window+1 prices to compute window returns
        if len(prices) <4:
            return "neutral"
        if self.status[book_id] == 0:
            if bids[-2] == max(bids) and bids[-1] < bids[-2] and bids[-1] > bids[0] + 0.7:
                self.trime_price(validator_hotkey, book_id)
                self.status[book_id] = 2
                return "sell"
            elif asks[-2] == min(asks) and asks[-1] > asks[-2] and asks[-1] < asks[0] - 0.7:
                self.trime_price(validator_hotkey, book_id)
                self.status[book_id] = 1
                return "buy"
            else:
                return "neutral"
        elif self.status[book_id] == 1:
            if (bids[-1] == max(bids) and bids[-1] > bids[0] + 0.6) or bids[-1] < asks[0] - 0.3:
                self.status[book_id] = 0
                return "sell"
            else:
                return "neutral"
        elif self.status[book_id] ==2:
            if (asks[-1] == min(asks) and asks[-1] < asks[0] - 0.6) or asks[-1] > bids[0] + 0.3:
                self.status[book_id] = 0
                return "buy"
            else:
                return "neutral"
        else:    
            return "neutral"
        
    def update(self, state : MarketSimulationStateUpdate) -> None:

        self.simulation_config = state.config
        self.accounts = state.accounts[self.uid]
        self.events = state.notices[self.uid]
        validator_hotkey = self.get_validator_hotkey(state)
        
        simulation_ended = False
        for book_id in range(self.simulation_config.book_count):
            traded_count = 0
            for event in self.events:
                if hasattr(event, 'bookId') and event.bookId == book_id:
                    match event.type:
                        case "EVENT_TRADE" | "ET":
                            traded_count += 1
                            if traded_count > 1:
                                continue
                            role = "taker" if self.uid == event.takerAgentId else "maker"
                            trade_text = f"{'BUY ' if event.side == 0 else 'SELL'} TRADE #{event.tradeId} : YOUR {'AGGRESSIVE' if role=='taker' else 'PASSIVE'} " + \
                                f"ORDER #{event.takerOrderId if role=='taker' else event.makerOrderId} (AGENT {event.takerAgentId if role=='taker' else event.makerAgentId}) " + \
                                f"MATCHED AGAINST #{event.makerOrderId if role=='taker' else event.takerOrderId} (AGENT {event.makerAgentId if role=='taker' else event.takerAgentId}) " + \
                                f"FOR {event.quantity}@{event.price} AT {duration_from_timestamp(event.timestamp)} (T={event.timestamp})"
                            bt.logging.info(f"TRADE EVENT PROCESSED: {trade_text}")
                            if self.status[book_id] == 0:
                                self.status[book_id] = 2 if event.side == 1 else 1
                            else:
                                self.status[book_id] = 0
                        case _:
                            bt.logging.warning(f"Unknown event : {event}")
                            
    def respond(self, state):
        validator_hotkey = self.get_validator_hotkey(state)
        response = FinanceAgentResponse(agent_id=self.uid)
        price_decimals = getattr(state.config, "priceDecimals", 2)
        vol_decimals = getattr(state.config, "volumeDecimals", 4)
        min_qty = 1.0
        max_qty = 1.1
        # self.update(state)

        for book_id, book in state.books.items():

            if not book.bids or not book.asks:
                continue
            best_bid = book.bids[0].p
            best_ask = book.asks[0].p
            mid = (best_bid + best_ask) / 2
            account = state.accounts[self.uid][book_id]

            own_base = account.own_base
            own_quote = account.own_quote
            current_balance = own_quote + mid * own_base

            self.append_price(validator_hotkey, book_id, state.timestamp, best_bid, best_ask, current_balance)

            trend = self.regime_detection(validator_hotkey, book_id)

            rate = 0.012
            initial_volume = account.base_balance.initial
            base_volume = account.base_balance.total
            
            qty = min(max(min_qty, round(initial_volume*rate, vol_decimals)), max_qty)
            qty = 1.0
            if base_volume > 20:
                response.market_order(
                    book_id=book_id, 
                    direction=OrderDirection.SELL, 
                    quantity=base_volume - 5,
                )
                continue
            if trend == 'sell':
                response.market_order(
                    book_id=book_id, 
                    direction=OrderDirection.SELL, 
                    quantity=qty,
                )
            elif trend == 'buy':
                response.market_order(
                    book_id=book_id, 
                    direction=OrderDirection.BUY, 
                    quantity=qty,
                )
            else:
                pass
          
        if(validator_hotkey == "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF"):
            print(f"status: {self.status}")
        return response

if __name__ == "__main__":
    from taos.common.agents import launch
    launch(OCO)
