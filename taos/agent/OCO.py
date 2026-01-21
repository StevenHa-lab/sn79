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

class OCO(FinanceSimulationAgent):
    def initialize(self):
        self.observate_time = 150
        self.window_size = 50
        self.t_threshold = 0.5
        self.history_dirs = {}
        self.data_dir = './data'
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        self.output_dir = os.path.join(self.data_dir, str(self.uid))
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def get_validator_hotkey(self, state):
        return state.dendrite.hotkey
    
    def simulation_output_dir(self, state : MarketSimulationStateUpdate):
        simulation_output_dir = os.path.join(self.output_dir, state.dendrite.hotkey)
        os.makedirs(simulation_output_dir, exist_ok=True)
        return simulation_output_dir
    
    def get_history_dir(self, validator_hotkey):
        if validator_hotkey not in self.history_dirs:
            path = os.path.abspath(f"history/history_{validator_hotkey}_{self.uid}")
            os.makedirs(path, exist_ok=True)
            self.history_dirs[validator_hotkey] = path
        return self.history_dirs[validator_hotkey]
    
    def get_history_file(self, validator_hotkey, book_id):
        return os.path.join(self.get_history_dir(validator_hotkey), f"price_history_{book_id}.csv")
    
    def append_price(self, validator_hotkey, book_id, timestamp, best_bid, best_ask):
        file_path = self.get_history_file(validator_hotkey, book_id)
        with open(file_path, "a") as f:
            f.write(f"{timestamp},{best_bid},{best_ask},{(best_bid + best_ask)/2}\n")

        lines = []
        with open(file_path, "r") as f:
            lines = f.readlines()
        if len(lines) > 1000:
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
        if len(prices) < 30:
            return "INITIAL", 0.0, 0.0
        returns = np.diff(np.log(prices))
        local_window = min(len(returns), self.window_size)
        mu = np.mean(returns[-local_window:])
        mean = np.mean(prices[-local_window:])
        sigma = np.std(returns[-local_window:], ddof=1)
        if sigma == 0:
            return "neutral", mean, sigma
        t_stat = mu / (sigma / np.sqrt(local_window))
        if t_stat > self.t_threshold:
            return "up"
        elif t_stat < -self.t_threshold:
            return "down"
        else:
            return "neutral"

    def log_order_event(self, event : LimitOrderPlacementEvent | MarketOrderPlacementEvent, state : MarketSimulationStateUpdate, book_id):
        """Log LimitOrderPlacementEvent or MarketOrderPlacementEvent to CSV."""
        orders_log_file = os.path.join(self.simulation_output_dir(state), f"orders_{book_id}.csv")
        file_exists = os.path.exists(orders_log_file)
        with open(orders_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'bookId', 'orderId', 'clientOrderId',
                    'side', 'price', 'currency', 'quantity', 'leverage', 'settleFlag',
                    'success', 'message'
                ])
            writer.writerow([
                int(event.timestamp//1e9),
                getattr(event, 'bookId', None),
                getattr(event, 'orderId', None),
                getattr(event, 'clientOrderId', None),
                getattr(event, 'side', None),
                getattr(event, 'price', None),
                getattr(event, 'currency', None),
                getattr(event, 'quantity', None),
                getattr(event, 'leverage', None),
                getattr(event, 'settleFlag', None),
                event.success,
                event.message
            ])

    def log_cancellation_event(self, event : OrderCancellationEvent, state : MarketSimulationStateUpdate, book_id):
        """Log OrderCancellationEvent to CSV."""
        cancellations_log_file = os.path.join(self.simulation_output_dir(state), f"cancellations_{book_id}.csv")
        file_exists = os.path.exists(cancellations_log_file)
        with open(cancellations_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'bookId', 'orderId', 'quantity', 'success', 'message'
                ])
            writer.writerow([
                int(event.timestamp//1e9),
                event.bookId,
                event.orderId,
                event.quantity,
                event.success,
                event.message
            ])

    def log_trade_event(self, event : TradeEvent, state : MarketSimulationStateUpdate, book_id: int):
        """Log TradeEvent to CSV."""
        trades_log_file = os.path.join(self.simulation_output_dir(state), f'trades_{book_id}.csv')
        file_exists = os.path.exists(trades_log_file)
        with open(trades_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'bookId', 'tradeId', 'clientOrderId',
                    'takerAgentId', 'takerOrderId', 'takerFee',
                    'makerAgentId', 'makerOrderId', 'makerFee',
                    'side', 'price', 'quantity'
                ])
            writer.writerow([
                int(event.timestamp//1e9),
                event.bookId,
                event.tradeId,
                event.clientOrderId,
                event.takerAgentId,
                event.takerOrderId,
                event.takerFee,
                event.makerAgentId,
                event.makerOrderId,
                event.makerFee,
                event.side,
                event.price,
                event.quantity
            ])

    def update(self, state : MarketSimulationStateUpdate) -> None:

        self.simulation_config = state.config
        self.accounts = state.accounts[self.uid]
        self.events = state.notices[self.uid]

        simulation_ended = False

        for book_id in range(self.simulation_config.book_count):

            for event in self.events:
                if hasattr(event, 'bookId') and event.bookId == book_id:
                    match event.type:
                        case "RESPONSE_DISTRIBUTED_PLACE_ORDER_LIMIT" | "RESPONSE_DISTRIBUTED_PLACE_ORDER_MARKET" | "RDPOL" | "RDPOM":
                            self.log_order_event(event, state, book_id)

                        case "EVENT_TRADE" | "ET":
                            self.log_trade_event(event, state, book_id)
                        case "RESPONSE_DISTRIBUTED_CANCEL_ORDERS" | "RDCO":
                            for cancellation in event.cancellations:
                                self.log_cancellation_event(cancellation, state, book_id)
                        case _:
                            bt.logging.warning(f"Unknown event : {event}")
    
    def trim_trades_csv(self, csv_path: str):
        MAX_ROWS = 400
        TRIM_ROWS = 250
        if not os.path.isfile(csv_path):
            return 
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))

        if len(rows) <= MAX_ROWS + 1:  # +1 for header
            return

        header = rows[0]
        data_rows = rows[1:]

        # Keep only last 500 rows
        trimmed_rows = data_rows[-TRIM_ROWS:]

        # Rewrite file
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(trimmed_rows)
    
    def last_trade_role_for_uid(self, df: pd.DataFrame, df_order: pd.DataFrame, book_id: int, timestamp: int) -> dict:
        
        orders = df_order[df_order["bookId"] == book_id]
        trades = df[df["bookId"] == book_id]
        if len(orders) == 0:
            return None
        last_order = orders.iloc[-1]
        
        if len(trades) == 0:
            return {
                "bookId": book_id,
                "clientId": last_order["clientOrderId"],
                "timestamp": last_order["timestamp"],
                "price":last_order["price"],                
                "side": side,
                "role": "INITIAL",
            }
        last_trade = trades.iloc[-1]
        price = last_trade["price"]
        trade_clientOrderId = int(last_trade["clientOrderId"])
        order_clientOrderId = int(last_order["clientOrderId"])
        
        if trade_clientOrderId != order_clientOrderId:
            role = 'ORDERED_WAIT'
            side = 'BUY' if last_trade["side"] == 0 else 'SELL'
            clientOrderId = trade_clientOrderId
        else:
            if order_clientOrderId == int(orders.iloc[-2]["clientOrderId"]) + 1 or order_clientOrderId == int(orders.iloc[-2]["clientOrderId"]) + 2:
                role = 'DONE'
            else:
                role = 'WAIT'
            side = 'BUY' if last_trade["side"] == 0 else 'SELL'
            clientOrderId = trade_clientOrderId
        
        return {
            "bookId": book_id,
            "clientId": clientOrderId,
            "timestamp": last_trade["timestamp"],
            "price": price,
            "side": side,
            "role": role,
        }
    
    def respond(self, state):
        
        validator_hotkey = self.get_validator_hotkey(state)
        response = FinanceAgentResponse(agent_id=self.uid)
        min_qty = 1.0
        max_qty = 1.1
        price_decimals = getattr(state.config, "priceDecimals", 2)
        vol_decimals = getattr(state.config, "volumeDecimals", 4)
        
        base_dir = self.simulation_output_dir(state)
        self.update(state)
        
        for book_id, book in state.books.items():
            
            if not os.path.isfile(os.path.join(base_dir, f"trades_{book_id}.csv")):
                trades_log_file = os.path.join(base_dir, f'trades_{book_id}.csv')
                file_exists = os.path.exists(trades_log_file)
                with open(trades_log_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow([
                            'timestamp', 'bookId', 'tradeId', 'clientOrderId',
                            'takerAgentId', 'takerOrderId', 'takerFee',
                            'makerAgentId', 'makerOrderId', 'makerFee',
                            'side', 'price', 'quantity'
                        ])
                        writer.writerow([
                            state.timestamp//1e9,40,1128366,0,201,2725396,0.006332678,26,2725324,0.0098073419,0,269.9,0.26
                        ])
            if not os.path.isfile(os.path.join(base_dir, f"orders_{book_id}.csv")):
                orders_log_file = os.path.join(base_dir, f'orders_{book_id}.csv')
                file_exists = os.path.exists(orders_log_file)
                with open(orders_log_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow([
                            'timestamp', 'bookId', 'orderId', 'clientOrderId',
                            'side', 'price', 'currency', 'quantity', 'leverage', 'settleFlag',
                            'success', 'message'
                        ])
                        writer.writerow([
                            state.timestamp//1e9,40,180068,0,1,289.61,0,2.2734,0.0,-2,True
                        ])

            self.trim_trades_csv(os.path.join(base_dir, f"trades_{book_id}.csv"))
            self.trim_trades_csv(os.path.join(base_dir, f"orders_{book_id}.csv"))
            df = pd.read_csv(os.path.join(base_dir, f"trades_{book_id}.csv"))
            df_order = pd.read_csv(os.path.join(base_dir, f"orders_{book_id}.csv"))


            if not book.bids or not book.asks:
                continue
            best_bid = book.bids[0].p
            best_ask = book.asks[0].p
            mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            spread = round(spread, 2)
            account = state.accounts[self.uid][book_id]
            initial_volume = account.base_balance.initial
            
            rate = 0.02
            qty = min(max(min_qty, round(initial_volume*rate, vol_decimals)), max_qty)
            current_time = int(state.timestamp//1e9)
            
            self.append_price(validator_hotkey, book_id, state.timestamp, best_bid, best_ask)
            
            local_prices = self.load_windowed_history(validator_hotkey, book_id, self.window_size)
            trend = self.regime_detection([p[2] for p in local_prices])
            last_trade = self.last_trade_role_for_uid(df, df_order, book_id, state.timestamp)
                
            if last_trade is None:
                if len(local_prices) > 30:
                    if trend == 'up' and spread < 0.025:
                        response.market_order(
                            book_id=book_id, 
                            direction=OrderDirection.BUY, 
                            clientOrderId = current_time,
                            quantity=qty,
                        )
                        continue                    
                    elif trend == "down" and spread < 0.025:
                        qty = min(max(min_qty, round(initial_volume*rate, vol_decimals)), max_qty)
                        response.market_order(
                            book_id=book_id, 
                            direction=OrderDirection.SELL, 
                            clientOrderId = current_time,
                            quantity=qty,
                        )
                        continue
                    else:
                        continue
                else:
                    continue

            gap_seconds = current_time - last_trade['timestamp']
            if gap_seconds < 0:
                gap_seconds = gap_seconds + 86400
            
            bt.logging.info(f"Book {book_id} | Trend: {trend} | Spread: {spread} | Last Trade Role: {last_trade['role']} | Gap Seconds: {gap_seconds}")
            
            if last_trade['role'] == "DONE" and gap_seconds > 3:
                if trend == 'up' and spread < 0.025:
                    response.market_order(
                        book_id=book_id, 
                        direction=OrderDirection.BUY, 
                        clientOrderId = current_time,
                        quantity=qty,
                    )
                
                elif trend == "down" and spread < 0.025:
                    qty = min(max(min_qty, round(initial_volume*rate, vol_decimals)), max_qty)
                    response.market_order(
                        book_id=book_id, 
                        direction=OrderDirection.SELL, 
                        clientOrderId = current_time,
                        quantity=qty,
                    )
                else:
                    pass
            elif last_trade['role'] == "ORDERED_WAIT" and gap_seconds <= self.observate_time:
                pass
            elif last_trade['role'] == "WAIT" and gap_seconds <= self.observate_time and gap_seconds > 3:
                if last_trade['side'] == 'BUY':
                    if best_bid > last_trade['price'] + 0.1:
                        response.market_order(
                            book_id=book_id, 
                            direction=OrderDirection.SELL, 
                            clientOrderId = last_trade['clientId'] + 1,
                            quantity=qty,
                        )
                    if best_bid < last_trade['price'] - 0.05:
                        response.market_order(
                            book_id=book_id, 
                            direction=OrderDirection.SELL, 
                            clientOrderId = last_trade['clientId'] + 2,
                            quantity=qty,
                        )
                if last_trade['side'] == 'SELL':
                    if best_ask < last_trade['price'] - 0.1:
                        response.market_order(
                            book_id=book_id, 
                            direction=OrderDirection.BUY, 
                            clientOrderId = last_trade['clientId'] + 1,
                            quantity=qty,
                        )
                    if best_ask > last_trade['price'] + 0.05:
                        response.market_order(
                            book_id=book_id, 
                            direction=OrderDirection.BUY, 
                            clientOrderId = last_trade['clientId'] + 2,
                            quantity=qty,
                        )
                else:
                    pass
            elif gap_seconds > self.observate_time:
                if last_trade['side'] == 'BUY':
                    response.market_order(
                        book_id=book_id, 
                        direction=OrderDirection.SELL, 
                        clientOrderId = last_trade['clientId'] + 1,
                        quantity=qty,
                    )
                elif last_trade['side'] == 'SELL':
                    response.market_order(
                        book_id=book_id, 
                        direction=OrderDirection.BUY, 
                        clientOrderId = last_trade['clientId'] + 1,
                        quantity=qty,
                    )
                else:
                    pass
            else:
                pass
                    
            
        return response
if __name__ == "__main__":
    from taos.common.agents import launch
    launch(OCO)