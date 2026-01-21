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
        self.observate_time = 150
        self.history_dirs = {}
        self.overall_window_size = 80
        self.local_window_size = 30
        self.regime_window = 250
        self.t_threshold = 1
        self.fill_history = defaultdict(lambda: deque(maxlen=20))
        self.order_fills = defaultdict(set)
        self.order_history = defaultdict(dict)
        self.pnl_history = defaultdict(list)
        self.last_inventory_value = {}
        self.max_base_loan = 50.0
        self.max_leverage = 10.0
        self.min_base_volume_size = 20
        self.expiry_period = 10_000_000_000
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



    def append_price(self, validator_hotkey, book_id, timestamp, best_bid, best_ask, current_balance):
        file_path = self.get_history_file(validator_hotkey, book_id)
        with open(file_path, "a") as f:
            f.write(f"{timestamp},{best_bid},{best_ask},{(best_bid + best_ask)/2},{current_balance}\n")

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
            return "INITIAL", 0.0, 0.0

        # Log returns
        returns = np.diff(np.log(prices))

        # Use only the most recent window
        window = min(len(returns), self.regime_window)
        local_window = min(len(returns), self.local_window_size)
        r = returns[-window:]
        
        mu = np.mean(returns[-local_window:])
        mean = np.mean(prices[-local_window:])
        sigma = np.std(returns[-local_window:], ddof=1)

        if sigma == 0:
            return "neutral", mean, sigma

        t_stat = mu / (sigma / np.sqrt(local_window))

        if t_stat > self.t_threshold:
            return "up", mean, t_stat
        elif t_stat < -self.t_threshold:
            return "down", mean, t_stat
        else:
            return "neutral", mean, t_stat
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
                            print(f"TradeEvent: {event}")
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

        # Filter trades by book
        orders = df_order[df_order["bookId"] == book_id]
        trades = df[df["bookId"] == book_id]

        if len(orders) == 0:
            return None

        last_order = orders.iloc[-1]

        if len(trades) == 0:
            return {
                "bookId": book_id,
                "orderId": last_order["orderId"],
                "timestamp": last_order["timestamp"],
                "side": side,
                "role": "INITIAL",
            }
        
        last_trade = trades.iloc[-1]
        side = "BUY" if last_trade["side"] == 0 else "SELL"
        
        role = 'WAIT'
        last_clientOrderId = int(last_order["clientOrderId"])
        clientOrderId = int(last_trade["clientOrderId"])
        print(f"last_clientOrderId: {last_clientOrderId}, clientOrderId: {clientOrderId}")
        orderId = last_order["orderId"]
        if clientOrderId == last_clientOrderId:
            if last_clientOrderId == int(last_order["timestamp"]) + 2:
                role = 'SL_DONE'
            elif last_clientOrderId == int(last_order["timestamp"]) + 1:
                role = 'TP_DONE'
                sl_order = df_order[df_order["clientOrderId"] == clientOrderId + 1]
                orderId = sl_order.iloc[-1]["orderId"] if len(sl_order) > 0 else None
            else:
                print(f"unknown case for clientOrderId == last_clientOrderId")
        elif clientOrderId == last_clientOrderId - 2:
            print(f"sdfsfsdfsd1")
            role = 'WAIT'
            sl_order = df_order[df_order["clientOrderId"] == clientOrderId + 2]
            orderId = sl_order.iloc[-1]["orderId"] if len(sl_order) > 0 else None
        else:
            print(f"sdfsfsdfsd4")
            role = 'UNKNOWN'
        print(f"orderId: {orderId}, role: {role}, side: {side}, last_trade: {last_trade['timestamp']}")
        return {
            "bookId": book_id,
            "orderId": orderId,
            "clientId": clientOrderId,
            "timestamp": last_trade["timestamp"],
            "side": side,
            "role": role,
        }

    def respond(self, state):
        validator_hotkey = self.get_validator_hotkey(state)
        response = FinanceAgentResponse(agent_id=self.uid)
        price_decimals = getattr(state.config, "priceDecimals", 2)
        vol_decimals = getattr(state.config, "volumeDecimals", 4)
        min_qty = 1.0
        max_qty = 1.1
        is_need = False

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
            leverage = 0.7
            spread = best_ask - best_bid
            spread = round(spread, 2)
            account = state.accounts[self.uid][book_id]

            own_base = account.own_base
            own_quote = account.own_quote
            current_balance = own_quote + mid * own_base

            self.append_price(validator_hotkey, book_id, state.timestamp, best_bid, best_ask, current_balance)
            # overall_prices = self.load_windowed_history(validator_hotkey, book_id, self.overall_window_size)
            local_prices = self.load_windowed_history(validator_hotkey, book_id, self.local_window_size)
            trend, mean, t_stat = self.regime_detection([p[2] for p in local_prices])
            prices = [p[2] for p in local_prices]

            base_volume = account.base_balance.total
            quote_volume = account.quote_balance.total
            initial_volume = account.base_balance.initial

            buy_position = best_bid
            sell_position = best_ask
            buy_qty = min_qty
            sell_qty = min_qty

            buy_rate = 0.012
            # buy_rate = 0.004 if base_volume > (initial_volume * 1.4) else 0.01 if base_volume < (initial_volume * 0.4) else 0.007
            sell_rate = 0.012
            # sell_rate = 0.01 if base_volume > (initial_volume * 1.4) else 0.004 if base_volume < (initial_volume * 0.4) else 0.007

            last_trade = self.last_trade_role_for_uid(df, df_order, book_id, state.timestamp)
            current_time = int(state.timestamp//1e9)
            if last_trade is None:
                last_trade_timestamp_ns = state.timestamp - min(len(local_prices), 21) * 1e9
                gap_seconds = (state.timestamp - last_trade_timestamp_ns) / 1e9
                if gap_seconds > 20:
                    if trend == 'up' and spread < 0.025:
                        buy_qty = min(max(min_qty, round(initial_volume*buy_rate, vol_decimals)), max_qty)
                        response.market_order(
                            book_id=book_id, 
                            direction=OrderDirection.BUY, 
                            clientOrderId = current_time,
                            quantity=buy_qty,
                        )
                        response.limit_order(
                            book_id=book_id, 
                            direction=OrderDirection.SELL, 
                            clientOrderId = current_time + 2,
                            quantity=buy_qty,
                            price = best_bid - 0.05, 
                            timeInForce=TimeInForce.GTC,    
                        )
                        continue
                    elif trend == 'down' and spread < 0.025:
                        sell_qty = min(max(min_qty, round(initial_volume*sell_rate, vol_decimals)), max_qty)
                        response.market_order(
                            book_id=book_id, 
                            direction=OrderDirection.SELL, 
                            clientOrderId = current_time,
                            quantity=sell_qty,
                        )
                        response.limit_order(
                            book_id=book_id, 
                            direction=OrderDirection.BUY, 
                            clientOrderId = current_time + 2,
                            quantity=sell_qty,
                            price = best_ask + 0.05,
                            timeInForce=TimeInForce.GTC,    
                        )
                        continue
                    elif trend == 'neutral' and spread < 0.025:
                        if prices[-1] > prices[-2]:
                            sell_qty = min(max(min_qty, round(initial_volume*sell_rate, vol_decimals)), max_qty)
                            response.market_order(
                                book_id=book_id, 
                                direction=OrderDirection.SELL, 
                                clientOrderId = current_time,
                                quantity=sell_qty,
                            )
                            response.limit_order(
                                book_id=book_id, 
                                direction=OrderDirection.BUY, 
                                clientOrderId = current_time + 2,
                                quantity=sell_qty,
                                price = best_ask + 0.03,
                                timeInForce=TimeInForce.GTC,    
                            )
                            continue
                        else:
                            buy_qty = min(max(min_qty, round(initial_volume*buy_rate, vol_decimals)), max_qty)
                            response.market_order(
                                book_id=book_id, 
                                direction=OrderDirection.BUY, 
                                clientOrderId = current_time,
                                quantity=buy_qty,
                            )
                            response.limit_order(
                                book_id=book_id, 
                                direction=OrderDirection.SELL, 
                                clientOrderId = current_time + 2,
                                quantity=sell_qty,
                                price = best_bid - 0.03,
                                timeInForce=TimeInForce.GTC,    
                            )
                            continue
                    continue
            else:
                last_trade_timestamp_ns = last_trade['timestamp']
                gap_seconds = (state.timestamp - last_trade_timestamp_ns * 1e9) / 1e9
                if gap_seconds < 0:
                    gap_seconds = 86400 + gap_seconds  # handle day wrap-around
                if last_trade['role'] == "SL_DONE":
                    if trend == 'up' and spread < 0.025:
                        buy_qty = min(max(min_qty, round(initial_volume*buy_rate, vol_decimals)), max_qty)
                        response.market_order(
                            book_id=book_id, 
                            direction=OrderDirection.BUY, 
                            clientOrderId = current_time,
                            quantity=buy_qty,
                        )
                        response.limit_order(
                            book_id=book_id, 
                            direction=OrderDirection.SELL, 
                            clientOrderId = current_time + 2,
                            quantity=buy_qty,
                            price = best_bid - 0.05,
                            timeInForce=TimeInForce.GTC,    
                        )
                        continue
                    elif trend == 'down' and spread < 0.025:
                        sell_qty = min(max(min_qty, round(initial_volume*sell_rate, vol_decimals)), max_qty)
                        response.market_order(
                            book_id=book_id, 
                            direction=OrderDirection.SELL, 
                            clientOrderId = current_time,
                            quantity=sell_qty,
                        )
                        response.limit_order(
                            book_id=book_id, 
                            direction=OrderDirection.BUY, 
                            clientOrderId = current_time + 2,
                            quantity=sell_qty,
                            price = best_ask + 0.05,
                            timeInForce=TimeInForce.GTC,    
                        )
                        continue
                if last_trade['role'] == "TP_DONE":
                    if trend == 'up' and spread < 0.025:
                        buy_qty = min(max(min_qty, round(initial_volume*buy_rate, vol_decimals)), max_qty)
                        response.market_order(
                            book_id=book_id, 
                            direction=OrderDirection.BUY, 
                            clientOrderId = current_time,
                            quantity=buy_qty,
                        )
                        response.limit_order(
                            book_id=book_id, 
                            direction=OrderDirection.SELL, 
                            clientOrderId = current_time + 2,
                            quantity=buy_qty,
                            price = best_bid - 0.05,
                            timeInForce=TimeInForce.GTC,    
                        )
                        response.cancel_order(
                            book_id=book_id,
                            order_id=last_trade['orderId']
                        )
                        continue
                    elif trend == 'down' and spread < 0.025:
                        sell_qty = min(max(min_qty, round(initial_volume*sell_rate, vol_decimals)), max_qty)
                        response.market_order(
                            book_id=book_id, 
                            direction=OrderDirection.SELL, 
                            clientOrderId = current_time,
                            quantity=sell_qty,
                        )
                        response.limit_order(
                            book_id=book_id, 
                            direction=OrderDirection.BUY, 
                            clientOrderId = current_time + 2,
                            quantity=sell_qty,
                            price = best_ask + 0.05,
                            timeInForce=TimeInForce.GTC,    
                        )
                        response.cancel_order(
                            book_id=book_id,
                            order_id=last_trade['orderId']
                        )
                        continue
                if last_trade['role'] == "WAIT" and gap_seconds <= self.observate_time:
                    sell_qty = min(max(min_qty, round(initial_volume*sell_rate, vol_decimals)), max_qty)
                    last_clientOrderId = int(last_trade['timestamp'])
                    if last_trade['side'] == "BUY" and trend == 'up' and prices[-1] < prices[-2] and best_bid > last_trade.get('price', 0):
                        response.market_order(
                            book_id=book_id, 
                            direction=OrderDirection.SELL, 
                            clientOrderId = last_clientOrderId + 1,
                            quantity=sell_qty,
                        )
                        response.cancel_order(
                            book_id=book_id,
                            order_id=last_trade['orderId']
                        )
                        continue
                    elif last_trade['side'] == "SELL" and trend == 'down' and prices[-1] > prices[-2]:
                        response.market_order(
                            book_id=book_id, 
                            direction=OrderDirection.BUY, 
                            clientOrderId = last_clientOrderId + 1,
                            quantity=buy_qty,
                        )
                        response.cancel_order(
                            book_id=book_id,
                            order_id=last_trade['orderId']
                        )
                        continue
                if gap_seconds > self.observate_time:
                    last_clientOrderId = int(last_trade['timestamp'])
                    sell_qty = min(max(min_qty, round(initial_volume*sell_rate, vol_decimals)), max_qty)
                    if last_trade['side'] == "BUY":
                        response.market_order(
                            book_id=book_id, 
                            direction=OrderDirection.SELL, 
                            clientOrderId = last_clientOrderId + 1,
                            quantity=sell_qty,
                        )
                        response.cancel_order(
                            book_id=book_id,
                            order_id=last_trade['orderId']
                        )
                        continue
                    elif last_trade['side'] == "SELL":
                        response.market_order(
                            book_id=book_id, 
                            direction=OrderDirection.BUY, 
                            clientOrderId = last_clientOrderId + 1,
                            quantity=buy_qty,
                        )
                        response.cancel_order(
                            book_id=book_id,
                            order_id=last_trade['orderId']
                        )
                        continue
                else:
                    continue
            if(validator_hotkey == "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF"):
                print(f"book_id: {book_id}, own_base: {own_base}")
                print(f"buy_qty: {buy_qty}, sell_qty: {sell_qty}, buy_position: {best_bid}, sell_position: {best_ask}, trend: {trend}, {np.mean([p[2] for p in local_prices])} ")

        return response

if __name__ == "__main__":
    from taos.common.agents import launch
    launch(Agent_O)