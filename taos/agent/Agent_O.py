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

import random

class Agent_O(FinanceSimulationAgent):
    def initialize(self):
        self.min_spread = 0.01
        self.history_dirs = {}
        self.overall_window_size = 35
        self.local_window_size = 30
        self.regime_window = 250
        self.t_threshold = 0.5

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

        lines = []
        with open(file_path, "r") as f:
            lines = f.readlines()
        if len(lines) > 2000:
            # Keep only the last 500
            lines = lines[-500:]
            with open(file_path, "w") as f:
                f.writelines(lines)



    def load_windowed_history(self, validator_hotkey, book_id, window_size):
        file_path = self.get_history_file(validator_hotkey, book_id)
        if not os.path.exists(file_path):
            # Pad with default values if file doesn't exist
            return [(1.0, 1.0, 1.0)] * window_size
        with open(file_path, "r") as f:
            lines = f.readlines()
            # Parse last window_size lines and extract best_bid and best_ask
            price_pairs = [
                tuple(map(float, line.strip().split(",")[1:4]))
                for line in lines[-window_size:]
            ]
        if len(price_pairs) > window_size:
            price_pairs = price_pairs[-window_size:]
        return price_pairs



    def regime_detection(self, prices):
        prices = np.asarray(prices, dtype=float)

        # Need at least window+1 prices to compute window returns
        if len(prices) < 30:
            return "neutral"

        min_i, min_value = min(enumerate(prices), key=lambda x: x[1])
        max_i, max_value = max(enumerate(prices), key=lambda x: x[1])
        if min_i < len(prices) - 2 and max_i < len(prices) - 2:
            prices = prices[max(min_i, max_i):]
        else:
            prices = prices[min(min_i, max_i):]
        # Log returns
        returns = np.diff(np.log(prices))

        # Use only the most recent window
        window = len(returns)
        # window = min(len(returns), self.regime_window)        
        mu = np.mean(returns)
        mean = np.mean(prices)
        sigma = np.std(returns, ddof=1)

        if sigma == 0:
            return "neutral", mean, sigma

        t_stat = mu / (sigma / np.sqrt(len(returns)))
        if prices[0] == min(prices) and prices[-2] == max(prices) and t_stat > self.t_threshold and returns[-1] < 0:
            if prices[-1] > prices[0] + 0.33:
                return "sell"
            else:
                return "neutral"
        elif prices[0] == max(prices) and prices[-2] == min(prices) and t_stat < -self.t_threshold and returns[-1] > 0:
            if prices[0] > prices[-1] + 0.33:
                return "buy"
            else:
                return "neutral"
        else:
            return "neutral"

    def respond(self, state):
        validator_hotkey = self.get_validator_hotkey(state)
        response = FinanceAgentResponse(agent_id=self.uid)
        price_decimals = getattr(state.config, "priceDecimals", 2)
        vol_decimals = getattr(state.config, "volumeDecimals", 4)
        min_qty = 1.0
        max_qty = 1.1
        is_need = False

        base_dir = self.simulation_output_dir(state)

        for book_id, book in state.books.items():

            if not book.bids or not book.asks:
                continue
            best_bid = book.bids[0].p
            best_ask = book.asks[0].p
            mid = (best_bid + best_ask) / 2
            leverage = 0.7
            spread = best_ask - best_bid
            spread = round(spread, 2)
            account = state.accounts[self.uid][book_id]

            own_base = account.own_base
            own_quote = account.own_quote
            current_balance = own_quote + mid * own_base

            self.append_price(validator_hotkey, book_id, state.timestamp, best_bid, best_ask, current_balance)
            # overall_prices = self.load_windowed_history(validator_hotkey, book_id, self.overall_window_size)
            local_prices = self.load_windowed_history(validator_hotkey, book_id, self.overall_window_size)
            trend = self.regime_detection([p[2] for p in local_prices])
            prices = [p[2] for p in local_prices]

            rate = 0.012
            initial_volume = account.base_balance.initial
            qty = min(max(min_qty, round(initial_volume*rate, vol_decimals)), max_qty)
            
            if trend == 'sell':
                response.limit_order(
                    book_id=book_id, 
                    direction=OrderDirection.SELL, 
                    quantity=qty,
                    price = best_ask - 0.02,
                    timeInForce=TimeInForce.GTC,    
                )
            elif trend == 'buy':
                response.limit_order(
                    book_id=book_id, 
                    direction=OrderDirection.BUY, 
                    quantity=qty,
                    price = best_bid + 0.02,
                    timeInForce=TimeInForce.GTC,    
                )
            else:
                pass
          
            if(validator_hotkey == "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF"):
                print(f"book_id: {book_id}, qty: {qty}, trend: {trend}, {np.mean([p[2] for p in local_prices])} ")

        return response

if __name__ == "__main__":
    from taos.common.agents import launch
    launch(Agent_O)