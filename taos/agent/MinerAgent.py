import os
import numpy as np
from collections import defaultdict, deque
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol.response import FinanceAgentResponse, OrderDirection, TimeInForce, LoanSettlementOption, STP
import random

class MinerAgent(FinanceSimulationAgent):
    def initialize(self):
        self.min_spread = 0.01
        self.stale_order_time = 50000000000
        self.history_dirs = {}
        self.overall_window_size = 200
        self.local_window_size = 20
        self.fill_history = defaultdict(lambda: deque(maxlen=20))
        self.order_fills = defaultdict(set)
        self.order_history = defaultdict(dict)
        self.pnl_history = defaultdict(list)
        self.last_inventory_value = {}
        self.max_base_loan = 50.0
        self.max_leverage = 10.0
        self.min_base_volume_size = 20
        self.expiry_period = 10_000_000_000
        
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
            if r > 0.5:
                bubbles = True
            elif r < -0.5:
                bursts = True
        
        if returns[-1] > 0.5:
            if bubbles == True:
                return "neutral"
            else:
                return "neutral"
        
        if returns[-1] < -0.5:
            if bursts == True:
                return "neutral"
            else:
                return "neutral"
        
        if returns[-1] > 0 and returns[-2] > 0:
            if abs(prices[-1] - prices[-2]) > 0.003:
                return 'up'
            else:
                return 'slow_up'
        elif returns[-1] < 0 and returns[-2] < 0:
            if abs(prices[-1] - prices[-2]) > 0.003:
                return "down"
            else:
                return "slow_down"
        else:
            return 'neutral'



    def respond(self, state):
        validator_hotkey = self.get_validator_hotkey(state)
        response = FinanceAgentResponse(agent_id=self.uid)
        price_decimals = getattr(state.config, "priceDecimals", 2)
        vol_decimals = getattr(state.config, "volumeDecimals", 4)
        min_qty = 1.0
        max_qty = 2.0
        is_need = False
            
        for book_id, book in state.books.items():
            if not book.bids or not book.asks:
                continue
            best_bid = book.bids[0].p
            best_ask = book.asks[0].p
            
            print(f"Book {book_id}: Best Bid = {best_bid}, Best Ask = {best_ask}")
            
            mid = (best_bid + best_ask) / 2
            leverage = 0.7
            spread = best_ask - best_bid
            spread = round(spread, 2)
            
            own_base = account.own_base
            own_quote = account.own_quote
            current_balance = own_quote + mid * own_base
            
            self.append_price(validator_hotkey, book_id, state.timestamp, best_bid, best_ask, current_balance)
            # overall_prices = self.load_windowed_history(validator_hotkey, book_id, self.overall_window_size)
            local_prices = self.load_windowed_history(validator_hotkey, book_id, self.local_window_size)
            trend = self.regime_detection([p[2] for p in local_prices])
            prices = [p[2] for p in local_prices]
            
            account = state.accounts[self.uid][book_id]
            base_volume = account.base_balance.total
            initial_volume = account.base_balance.initial
            
            buy_position = best_bid
            sell_position = best_ask
            buy_qty = min_qty
            sell_qty = min_qty



            buy_rate = 0.012
            # buy_rate = 0.004 if base_volume > (initial_volume * 1.4) else 0.01 if base_volume < (initial_volume * 0.4) else 0.007
            sell_rate = 0.012
            # sell_rate = 0.01 if base_volume > (initial_volume * 1.4) else 0.004 if base_volume < (initial_volume * 0.4) else 0.007
          
            stale_order_ids = []
            for order in account.o:
                # Cancel if open too long or too far from current touch/mid
                if order.s == 0 and abs(order.p - (best_bid if best_bid is not None else mid)) > self.min_spread * 2.5:
                    stale_order_ids.append(order.i)
                elif order.s == 1 and abs(order.p - (best_ask if best_ask is not None else mid)) > self.min_spread * 2.5:
                    stale_order_ids.append(order.i)
            if len(stale_order_ids) > 0:
                response.cancel_orders(book_id=book_id, order_ids=stale_order_ids)
            print(f"trend: {trend}")
            
            if trend == "bubble":
                buy_qty = base_quote/mid*0.2
                response.market_order(
                    book_id=book_id, 
                    direction=OrderDirection.BUY, 
                    quantity=buy_qty,
                )
                continue
            elif trend == "burst":
                sell_qty = base_volume * 0.2
                response.market_order(
                    book_id=book_id, 
                    direction=OrderDirection.SELL, 
                    quantity=sell_qty,
                )
                continue
            elif trend == 'up':
                buy_qty = min(max(min_qty, round(initial_volume*buy_rate, vol_decimals)), max_qty)
                leverage = 0.5 if account.quote_loan < 9000 else 0.0
                settlement = LoanSettlementOption.NONE if account.base_loan == 0 else LoanSettlementOption.FIFO
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
            elif trend == 'slow_up':
                buy_qty = min(max(min_qty, round(initial_volume*buy_rate/5, vol_decimals)), max_qty)
                leverage = 0.5 if account.quote_loan < 9000 else 0.0
                settlement = LoanSettlementOption.NONE if account.base_loan == 0.0 else LoanSettlementOption.FIFO
                response.limit_order(
                    book_id=book_id,
                    direction=OrderDirection.BUY,
                    quantity=buy_qty,
                    price=best_bid - 0.03,
                    timeInForce=TimeInForce.GTT,
                    expiryPeriod=5_000_000_000,
                    stp=STP.CANCEL_OLDEST, 
                    leverage=leverage,
                    settlement_option=settlement
                )
            elif trend == 'down' and base_volume > 0.1:
                
                sell_qty = min(max(min_qty, round(initial_volume*sell_rate, vol_decimals)), max_qty)
                settlement = LoanSettlementOption.NONE if account.quote_balance == 0 else LoanSettlementOption.FIFO
                leverage = 0.5 if account.base_loan < min(18, initial_volume/5) else 0.0
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
            elif trend == 'slow_down' and base_volume > 0.1:
                
                sell_qty = min(max(min_qty, round(initial_volume*sell_rate/5, vol_decimals)), max_qty)
                leverage = 0.5 if account.base_loan < min(18, initial_volume/5) else 0.0
                settlement = LoanSettlementOption.NONE if account.quote_loan == 0 else LoanSettlementOption.FIFO
                response.limit_order(
                    book_id=book_id,
                    direction=OrderDirection.SELL,
                    quantity=sell_qty,
                    price=best_ask + 0.03,
                    timeInForce=TimeInForce.GTT,
                    expiryPeriod=5_000_000_000,
                    stp=STP.CANCEL_OLDEST, 
                    leverage=leverage,
                    settlement_option=settlement
                )
            else:
                if spread > 0 and spread < 0.025:
                    buy_position = best_bid - 0.01
                    sell_position = best_ask + 0.01
                else:
                    buy_position = best_bid
                    sell_position = best_ask
                if prices[-1] - prices[-2] > 0.003 and base_volume < 10:
                    leverage = 0.5 if account.quote_loan < 9000 else 0.0
                    settlement = LoanSettlementOption.NONE if account.base_loan == 0.0 else LoanSettlementOption.FIFO
                    response.limit_order(
                        book_id=book_id,
                        direction=OrderDirection.BUY,
                        quantity=min_qty,
                        price=buy_position,
                        timeInForce=TimeInForce.GTT,
                        expiryPeriod=5_000_000_000,
                        stp=STP.CANCEL_OLDEST, 
                        leverage = leverage,
                        settlement_option= settlement
                    )
                elif prices[-1] - prices[-2] < -0.003 and base_volume > 0.1:
                    leverage = 0.5 if account.base_loan < min(18, initial_volume/5) else 0.0
                    settlement = LoanSettlementOption.NONE if account.quote_loan == 0 else LoanSettlementOption.FIFO
                    response.limit_order(
                        book_id=book_id,
                        direction=OrderDirection.SELL,
                        quantity=min_qty,
                        price=sell_position,
                        timeInForce=TimeInForce.GTT,
                        expiryPeriod=5_000_000_000,
                        stp=STP.CANCEL_OLDEST, 
                        leverage = leverage,
                        settlement_option= settlement
                    )    
            if(validator_hotkey == "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF"):
                print(f"book_id: {book_id}, own_base: {own_base}")
                print(f"buy_qty: {buy_qty}, sell_qty: {sell_qty}, buy_position: {best_bid}, sell_position: {best_ask}, trend: {trend}, {np.mean([p[2] for p in local_prices])} ")

        return response

if __name__ == "__main__":
    from taos.common.agents import launch
    launch(MinerAgent)