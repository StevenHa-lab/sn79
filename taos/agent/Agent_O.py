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

class Agent_O(FinanceSimulationAgent):
    def initialize(self):
        self.min_spread = 0.01
        self.stale_order_time = 40
        self.history_dirs = {}
        self.overall_window_size = 300
        self.local_window_size = 25
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
        if len(price_pairs) < window_size:
            pad_pair = price_pairs[0] if price_pairs else (1.0, 1.0, 1.0)
            price_pairs = [pad_pair] * (window_size - len(price_pairs)) + price_pairs
        return price_pairs[-window_size:]



    def regime_detection(self, prices):
        if len(prices) < 3:
            return 'neutral'  # not enough data
        
        returns = np.diff(prices)
        # mean = np.mean(prices[-20:])
        # o_mean = np.mean(prices[-200:])
        returns = [r for r in returns if np.abs(r) > 1e-3]
        
        if len(returns) < 2:
            return 'neutral'  # not enough valid changes
        bubbles = False
        bursts = False        
        for r in returns[:-1]:
            if r > 2:
                bubbles = True
            elif r < -2:
                bursts = True
        
        if returns[-1] > 2:
            if bubbles == True:
                return "bubble"
            else:
                return "bubble"
        
        if returns[-1] < -2:
            if bursts == True:
                return "burst"
            else:
                return "burst"
        
        if returns[-1] > 0 and returns[-2] > 0:
            return 'up'
        elif returns[-1] < 0 and returns[-2] < 0:
            return "down"
        else:
            return 'neutral'
        
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
            trend = self.regime_detection([p[2] for p in local_prices])
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
            
            if trend == "bubble":
                buy_qty = quote_volume/mid*0.2
                # response.market_order(
                #     book_id=book_id, 
                #     direction=OrderDirection.BUY, 
                #     quantity=buy_qty,
                # )
                continue
            elif trend == "burst":
                sell_qty = base_volume * 0.2
                # response.market_order(
                #     book_id=book_id, 
                #     direction=OrderDirection.SELL, 
                #     quantity=sell_qty,
                # )
                continue
            
            last_trade = self.last_trade_role_for_uid(df, df_order, book_id, self.uid)
            
            if last_trade is None:
                last_trade_timestamp_ns = state.timestamp - min(len(local_prices), 21) * 1e9
                gap_seconds = (state.timestamp - last_trade_timestamp_ns) / 1e9
                if gap_seconds > 20:
                    if trend == 'up':
                        buy_qty = min(max(min_qty, round(initial_volume*buy_rate, vol_decimals)), max_qty)
                        leverage = 0.0
                        # leverage = 0.4 if account.quote_loan == 0.0 else 0.0
                        settlement = LoanSettlementOption.NONE if account.base_loan == 0.0 else LoanSettlementOption.FIFO
                        if spread > 0.015:
                            response.limit_order(
                                book_id=book_id,
                                direction=OrderDirection.BUY,
                                quantity=buy_qty,
                                price=best_bid + 0.01,
                                timeInForce=TimeInForce.GTT,
                                expiryPeriod=5_000_000_000,
                                stp=STP.CANCEL_OLDEST, 
                                leverage=leverage,
                                settlement_option=settlement
                            )
                        else:
                            response.limit_order(
                                book_id=book_id,
                                direction=OrderDirection.BUY,
                                quantity=buy_qty,
                                price=best_ask,
                                timeInForce=TimeInForce.GTT,
                                expiryPeriod=5_000_000_000,
                                stp=STP.CANCEL_OLDEST, 
                                leverage=leverage,
                                settlement_option=settlement
                            )
                    elif trend == 'down':
                        
                        sell_qty = min(max(min_qty, round(initial_volume*sell_rate, vol_decimals)), max_qty)
                        settlement = LoanSettlementOption.NONE if account.quote_loan == 0 else LoanSettlementOption.FIFO
                        leverage = 0.0
                        # leverage = 0.4 if account.base_loan < min(15, initial_volume/5) and base_volume < 25 else 0.0
                        if spread < 0.015:
                            response.limit_order(
                                book_id=book_id,
                                direction=OrderDirection.SELL,
                                quantity=sell_qty,
                                price=best_bid,
                                timeInForce=TimeInForce.GTT,
                                expiryPeriod=5_000_000_000,
                                stp=STP.CANCEL_OLDEST, 
                                leverage=leverage,
                                settlement_option=settlement
                            )
                        else:
                            response.limit_order(
                                book_id=book_id,
                                direction=OrderDirection.SELL,
                                quantity=sell_qty,
                                price=best_ask - 0.01,
                                timeInForce=TimeInForce.GTT,
                                expiryPeriod=5_000_000_000,
                                stp=STP.CANCEL_OLDEST, 
                                leverage=leverage,
                                settlement_option=settlement
                            )
                    elif trend == 'neutral':
                        if spread > 0 and spread < 0.025:
                            buy_position = best_bid - 0.01
                            sell_position = best_ask + 0.01
                        else:
                            buy_position = best_bid
                            sell_position = best_ask
                        response.limit_order(
                            book_id=book_id,
                            direction=OrderDirection.BUY,
                            quantity=buy_qty,
                            price=best_bid + 0.01,
                            timeInForce=TimeInForce.GTC
                        )    
                        response.limit_order(
                            book_id=book_id,
                            direction=OrderDirection.SELL,
                            quantity=sell_qty,
                            price=best_ask - 0.01,
                            timeInForce=TimeInForce.GTC
                        )
                        stale_order_ids = []
                        for order in account.o:
                            # Cancel if open too long or too far from current touch/mid
                            if order.s == 0 and abs(order.p - (best_bid if best_bid is not None else mid)) > 10:
                                stale_order_ids.append(order.i)
                            elif order.s == 1 and abs(order.p - (best_ask if best_ask is not None else mid)) > 10:
                                stale_order_ids.append(order.i)
                        if len(stale_order_ids) > 0:
                            response.cancel_orders(book_id=book_id, order_ids=stale_order_ids)
                        print(f"trend: {trend}")
            else:
                last_trade_timestamp_ns = timestamp_from_duration(last_trade['timestamp'])
                gap_seconds = (state.timestamp - last_trade_timestamp_ns) / 1e9
                if gap_seconds > self.stale_order_time or gap_seconds < 0:
                    if trend == 'up':
                        buy_qty = min(max(min_qty, round(initial_volume*buy_rate, vol_decimals)), max_qty)
                        leverage = 0.0
                        # leverage = 0.4 if account.quote_loan == 0.0 else 0.0
                        settlement = LoanSettlementOption.NONE if account.base_loan == 0.0 else LoanSettlementOption.FIFO
                        if spread > 0.015:
                            response.limit_order(
                                book_id=book_id,
                                direction=OrderDirection.BUY,
                                quantity=buy_qty,
                                price=best_bid + 0.01,
                                timeInForce=TimeInForce.GTT,
                                expiryPeriod=5_000_000_000,
                                stp=STP.CANCEL_OLDEST, 
                                leverage=leverage,
                                settlement_option=settlement
                            )
                        else:
                            response.limit_order(
                                book_id=book_id,
                                direction=OrderDirection.BUY,
                                quantity=buy_qty,
                                price=best_ask,
                                timeInForce=TimeInForce.GTT,
                                expiryPeriod=5_000_000_000,
                                stp=STP.CANCEL_OLDEST, 
                                leverage=leverage,
                                settlement_option=settlement
                            )
                    elif trend == 'down':
                        
                        sell_qty = min(max(min_qty, round(initial_volume*sell_rate, vol_decimals)), max_qty)
                        settlement = LoanSettlementOption.NONE if account.quote_loan == 0 else LoanSettlementOption.FIFO
                        leverage = 0.0
                        # leverage = 0.4 if account.base_loan < min(15, initial_volume/5) and base_volume < 25 else 0.0
                        if spread < 0.015:
                            response.limit_order(
                                book_id=book_id,
                                direction=OrderDirection.SELL,
                                quantity=sell_qty,
                                price=best_bid,
                                timeInForce=TimeInForce.GTT,
                                expiryPeriod=5_000_000_000,
                                stp=STP.CANCEL_OLDEST, 
                                leverage=leverage,
                                settlement_option=settlement
                            )
                        else:
                            response.limit_order(
                                book_id=book_id,
                                direction=OrderDirection.SELL,
                                quantity=sell_qty,
                                price=best_ask - 0.01,
                                timeInForce=TimeInForce.GTT,
                                expiryPeriod=5_000_000_000,
                                stp=STP.CANCEL_OLDEST, 
                                leverage=leverage,
                                settlement_option=settlement
                            )
                    
                        stale_order_ids = []
                        for order in account.o:
                            # Cancel if open too long or too far from current touch/mid
                            if order.s == 0 and abs(order.p - (best_bid if best_bid is not None else mid)) > 10:
                                stale_order_ids.append(order.i)
                            elif order.s == 1 and abs(order.p - (best_ask if best_ask is not None else mid)) > 10:
                                stale_order_ids.append(order.i)
                        if len(stale_order_ids) > 0:
                            response.cancel_orders(book_id=book_id, order_ids=stale_order_ids)
                        print(f"trend: {trend}")
                else:
                    continue
            if(validator_hotkey == "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF"):
                print(f"book_id: {book_id}, own_base: {own_base}")
                print(f"buy_qty: {buy_qty}, sell_qty: {sell_qty}, buy_position: {best_bid}, sell_position: {best_ask}, trend: {trend}, {np.mean([p[2] for p in local_prices])} ")

        return response

if __name__ == "__main__":
    from taos.common.agents import launch
    launch(Agent_O)