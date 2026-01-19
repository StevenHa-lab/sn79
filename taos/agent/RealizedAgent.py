import os
import numpy as np
import random
import csv
import bittensor as bt
from pathlib import Path
import pandas as pd
from collections import defaultdict, deque
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol.response import FinanceAgentResponse, OrderDirection, TimeInForce, LoanSettlementOption, STP
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse, FinanceEventNotification
from taos.im.protocol.events import LimitOrderPlacementEvent, MarketOrderPlacementEvent, OrderCancellationEvent, TradeEvent
from taos.im.utils import duration_from_timestamp, timestamp_from_duration

class RealizedAgent(FinanceSimulationAgent):
    def initialize(self):
        self.min_spread = 0.01
        self.stale_order_time = 300
        self.history_dirs = {}
        self.overall_window_size = 80
        self.local_window_size = 80
        self.regime_window = 250
        self.t_threshold = 1.3
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
                duration_from_timestamp(event.timestamp),
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
                duration_from_timestamp(event.timestamp),
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
                duration_from_timestamp(event.timestamp),
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

    def trim_trades_csv(self, csv_path: str):
        MAX_ROWS = 3000
        TRIM_ROWS = 2000
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

    def last_trade_role_for_uid(self, df: pd.DataFrame, df_order: pd.DataFrame, book_id: int, uid: int) -> dict:

        # Filter trades by book
        trades = df[df["bookId"] == book_id]
        orders = df_order[df_order["bookId"] == book_id]

        if len(orders) == 0:
            return None

        last_order = orders.iloc[-1]
        side = "BUY" if last_order["side"] == 0 else "SELL"

        if len(trades) == 0:
            return {
                "bookId": book_id,
                "orderId": last_order["orderId"],
                "timestamp": last_order["timestamp"],
                "side": side,
                "role": "WAIT",
            }
        role = 'WAIT'
        last_order_id = last_order["orderId"]
        for i in range(len(trades)-1, -1, -1):
            trade = trades.iloc[i]
            if trade["makerOrderId"] == last_order_id or trade["takerOrderId"] == last_order_id:
                role = "DONE"
                break

        return {
            "bookId": book_id,
            "orderId": last_order["orderId"],
            "timestamp": last_order["timestamp"],
            "side": side,
            "role": role,
        }

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

                        case "RESPONSE_DISTRIBUTED_CANCEL_ORDERS" | "RDCO":
                            for cancellation in event.cancellations:
                                self.log_cancellation_event(cancellation, state, book_id)

                        case "EVENT_TRADE" | "ET":
                            print(f"TradeEvent: {event}")
                            self.log_trade_event(event, state, book_id)

                        case _:
                            bt.logging.warning(f"Unknown event : {event}")

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
        if len(lines) > 2000:
            # Keep only the last 500
            lines = lines[-500:]
            with open(file_path, "w") as f:
                f.writelines(lines)

    def load_windowed_history(self, validator_hotkey, book_id, window_size):
        file_path = self.get_history_file(validator_hotkey, book_id)
        if not os.path.exists(file_path):
            # Pad with default values if file doesn't exist
            return [(1.0, 1.0, 1.0)]
        with open(file_path, "r") as f:
            lines = f.readlines()
            window_size = min(window_size, len(lines))
            # Parse last window_size lines and extract best_bid and best_ask
            price_pairs = [
                tuple(map(float, line.strip().split(",")[1:4]))
                for line in lines[-window_size:]
            ]
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

        mu = np.mean(r)
        mean = np.mean(prices[-window:])
        sigma = np.std(r, ddof=1)
        local_mu = np.mean(returns[-local_window:])
        local_mean = np.mean(prices[-local_window:])
        local_sigma = np.std(returns[-local_window:], ddof=1)
        if local_sigma > 0.001:
            return "BUBBLES", mean, local_sigma
        if sigma == 0:
            return "NEUTRAL", mean, local_sigma

        t_stat = mu / (sigma / np.sqrt(window))

        if t_stat > self.t_threshold:
            return "UP", local_mean, t_stat
        elif t_stat < -self.t_threshold:
            return "DOWN", local_mean, t_stat
        else:
            return "NEUTRAL", local_mean, t_stat

    def respond(self, state):
        validator_hotkey = self.get_validator_hotkey(state)
        response = FinanceAgentResponse(agent_id=self.uid)
        price_decimals = getattr(state.config, "priceDecimals", 2)
        vol_decimals = getattr(state.config, "volumeDecimals", 4)
        min_qty = 1.0
        max_qty = 1.1

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
                            duration_from_timestamp(state.timestamp),40,1128366,0,201,2725396,0.006332678,26,2725324,0.0098073419,0,269.9,0.26
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
                            duration_from_timestamp(state.timestamp),40,180068,0,1,289.61,0,2.2734,0.0,-2,True
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

            self.append_price(validator_hotkey, book_id, state.timestamp, best_bid, best_ask)
            local_prices = self.load_windowed_history(validator_hotkey, book_id, self.overall_window_size)

            base_volume = account.base_balance.total
            quote_volume = account.quote_balance.total
            initial_volume = account.base_balance.initial

            trend, mean, t_stat = self.regime_detection([p[2] for p in local_prices])
            if(validator_hotkey == "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF"):
                print(f"book_id: {book_id}, trend: {trend}, mean: {mean}, mid: {mid}, t_stat: {t_stat}")
            rate = 0.01
            last_trade = self.last_trade_role_for_uid(df, df_order, book_id, self.uid)
            if last_trade is None:
                last_trade_timestamp_ns = state.timestamp - max(len(local_prices), 100) * 1e9
                gap_seconds = (state.timestamp - last_trade_timestamp_ns) / 1e9
                if gap_seconds > 70:
                    if trend == 'UP':
                        qty = min(max(min_qty, round(initial_volume*rate, vol_decimals)), max_qty)
                        response.limit_order(
                            book_id=book_id, 
                            direction=OrderDirection.BUY, 
                            quantity=qty,
                            price=base_bid + 0.01,
                            timeInForce=TimeInForce.GTC
                        )
                        response.limit_order(
                            book_id=book_id,
                            direction=OrderDirection.SELL,
                            quantity=qty,
                            price=best_bid + 0.15,
                            timeInForce=TimeInForce.GTC
                        )
                    elif trend == 'DOWN':
                        qty = min(max(min_qty, round(initial_volume*rate, vol_decimals)), max_qty)
                        response.limit_order(
                            book_id=book_id, 
                            direction=OrderDirection.SELL, 
                            quantity=qty,
                            price=best_ask - 0.01
                        )
                        response.limit_order(
                            book_id=book_id,
                            direction=OrderDirection.BUY,
                            quantity=qty,
                            price=best_ask - 0.15,
                            timeInForce=TimeInForce.GTC
                        )
                    elif trend == 'NEUTRAL':
                        if t_stat < 0:
                            qty = min(max(min_qty, round(initial_volume*rate, vol_decimals)), max_qty)

                            response.limit_order(
                                book_id=book_id, 
                                direction=OrderDirection.SELL, 
                                quantity=qty,
                                price=best_ask - 0.01,
                                timeInForce=TimeInForce.GTC
                            )
                            response.limit_order(
                                book_id=book_id,
                                direction=OrderDirection.BUY,
                                quantity=qty,
                                price=best_ask - 0.05,
                                timeInForce=TimeInForce.GTC
                            )
                        elif t_stat > 0:
                            qty = min(max(min_qty, round(initial_volume*rate, vol_decimals)), max_qty)
                            response.limit_order(
                                book_id=book_id, 
                                direction=OrderDirection.BUY, 
                                quantity=qty,
                                price=best_bid + 0.01
                            )
                            response.limit_order(
                                book_id=book_id,
                                direction=OrderDirection.SELL,
                                quantity=qty,
                                price=best_bid + 0.05,
                                timeInForce=TimeInForce.GTC
                            )
                    else:
                        continue
            else:
                last_trade_timestamp_ns = timestamp_from_duration(last_trade['timestamp'])
                gap_seconds = (state.timestamp - last_trade_timestamp_ns) / 1e9

                if last_trade['role'] == "DONE" or gap_seconds > self.stale_order_time or gap_seconds < 0:
                    if last_trade['role'] == "DONE":
                        if trend == 'UP':
                            qty = min(max(min_qty, round(initial_volume*rate, vol_decimals)), max_qty)
                            response.market_order(
                                book_id=book_id, 
                                direction=OrderDirection.BUY, 
                                quantity=qty
                            )
                            response.limit_order(
                                book_id=book_id,
                                direction=OrderDirection.SELL,
                                quantity=qty,
                                price=best_ask + 0.15,
                                timeInForce=TimeInForce.GTC
                            )
                        elif trend == 'DOWN':
                            qty = min(max(min_qty, round(initial_volume*rate, vol_decimals)), max_qty)
                            response.market_order(
                                book_id=book_id, 
                                direction=OrderDirection.SELL, 
                                quantity=qty
                            )
                            response.limit_order(
                                book_id=book_id,
                                direction=OrderDirection.BUY,
                                quantity=qty,
                                price=best_bid - 0.15,
                                timeInForce=TimeInForce.GTC
                            )
                        elif trend == 'NEUTRAL':
                            if t_stat < 0:
                                qty = min(max(min_qty, round(initial_volume*rate, vol_decimals)), max_qty)
                                response.market_order(
                                    book_id=book_id, 
                                    direction=OrderDirection.SELL, 
                                    quantity=qty,
                                )
                                response.limit_order(
                                    book_id=book_id,
                                    direction=OrderDirection.BUY,
                                    quantity=qty,
                                    price=best_bid - 0.05,
                                    timeInForce=TimeInForce.GTC
                                )
                            elif t_stat > 0:
                                qty = min(max(min_qty, round(initial_volume*rate, vol_decimals)), max_qty)

                                response.market_order(
                                    book_id=book_id, 
                                    direction=OrderDirection.BUY, 
                                    quantity=qty
                                )
                                response.limit_order(
                                    book_id=book_id,
                                    direction=OrderDirection.SELL,
                                    quantity=qty,
                                    price=best_ask + 0.05,
                                    timeInForce=TimeInForce.GTC
                                )
                        else:
                            continue
                    else:
                        if trend == 'UP':
                            qty = min(max(min_qty, round(initial_volume*rate, vol_decimals)), max_qty)
                            response.market_order(
                                book_id=book_id, 
                                direction=OrderDirection.BUY, 
                                quantity=qty
                            )
                            response.limit_order(
                                book_id=book_id,
                                direction=OrderDirection.SELL,
                                quantity=qty,
                                price=best_ask + 0.15,
                                timeInForce=TimeInForce.GTC
                            )
                        elif trend == 'DOWN':
                            qty = min(max(min_qty, round(initial_volume*rate, vol_decimals)), max_qty)
                            response.market_order(
                                book_id=book_id, 
                                direction=OrderDirection.SELL, 
                                quantity=qty
                            )
                            response.limit_order(
                                book_id=book_id,
                                direction=OrderDirection.BUY,
                                quantity=qty,
                                price=best_bid - 0.15,
                                timeInForce=TimeInForce.GTC
                            )
                        elif trend == 'NEUTRAL':
                            if t_stat < 0:
                                qty = min(max(min_qty, round(initial_volume*rate, vol_decimals)), max_qty)
                                response.market_order(
                                    book_id=book_id, 
                                    direction=OrderDirection.SELL, 
                                    quantity=qty
                                )
                                response.limit_order(
                                    book_id=book_id,
                                    direction=OrderDirection.BUY,
                                    quantity=qty,
                                    price=best_bid - 0.05,
                                    timeInForce=TimeInForce.GTC
                                )
                            elif t_stat > 0:
                                qty = min(max(min_qty, round(initial_volume*rate, vol_decimals)), max_qty)

                                response.market_order(
                                    book_id=book_id, 
                                    direction=OrderDirection.BUY, 
                                    quantity=qty
                                )
                                response.limit_order(
                                    book_id=book_id,
                                    direction=OrderDirection.SELL,
                                    quantity=qty,
                                    price=best_ask + 0.05,
                                    timeInForce=TimeInForce.GTC
                                )
                            elif t_stat == 0:
                                qty = min(max(min_qty, round(initial_volume*rate, vol_decimals)), max_qty)

                                response.market_order(
                                    book_id=book_id, 
                                    direction=OrderDirection.BUY, 
                                    quantity=qty
                                )
                                response.market_order(
                                    book_id=book_id,
                                    direction=OrderDirection.SELL,
                                    quantity=qty
                                )
                        else:
                            continue
                elif last_trade['role'] == "WAIT":
                    continue
                else:
                    continue           

        return response

if __name__ == "__main__":
    from taos.common.agents import launch
    launch(RealizedAgent)