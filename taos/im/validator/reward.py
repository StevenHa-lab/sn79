# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Rayleigh Research

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import math
import traceback
import random
import bittensor as bt
import numpy as np
from typing import Dict, Tuple
from taos.im.neurons.validator import Validator
from taos.im.protocol import MarketSimulationStateUpdate, FinanceAgentResponse
from taos.im.protocol.models import Account, Book, TradeInfo
from taos.im.protocol.events import TradeEvent
from taos.im.utils import normalize
from taos.im.utils.sharpe import sharpe, batch_sharpe

def get_inventory_value(account: Account, book: Book, method='midquote') -> float:
    """
    Calculates the instantaneous total value of an account's inventory using the specified method

    Args:
        account (taos.im.protocol.models.Account) : Object representing the state of the account to be evaluated
        book : Object representing the orderbook with which the account is associated
        method : String identifier of the method by which the value should be calculated; options are
            a) `best_bid` : Calculates base currency balance value using only the top level bid price
            b) `midquote` : Calculates base currency balance value using the midquote price `(bid + ask) / 2`
            c) `liquidation` : Calculates base currency balance value by evaluating the total amount received if base balance is sold immediately and in isolation into the current book

    Returns:
        float: Total inventory value of the account.
    """
    quote_balance = account['qb']['t'] - account['ql'] + account['qc']
    base_balance = account['bb']['t'] - account['bl'] + account['bc']
    
    book_a = book['a']
    book_b = book['b']
    has_orders = len(book_a) > 0 and len(book_b) > 0
    
    if method == "best_bid":
        price = book_b[0]['p'] if has_orders else 0.0
        return quote_balance + price * base_balance
    elif method == "midquote":
        price = (book_a[0]['p'] + book_b[0]['p']) / 2 if has_orders else 0.0
        return quote_balance + price * base_balance
    else:  # liquidation
        liq_value = 0.0
        to_liquidate = account['bb']['t']
        for bid in book_b:
            if to_liquidate == 0:
                break
            level_liq = min(to_liquidate, bid['q'])
            liq_value += level_liq * bid['p']
            to_liquidate -= level_liq
        return quote_balance + liq_value

def score_inventory_value(self: Validator, uid: int, inventory_values: Dict[int, Dict[int, float]]) -> float:
    """
    Calculates the new score value for a specific UID

    Args:
        self (taos.im.neurons.validator.Validator) : Validator instance
        uid (int) : UID of miner being scored
        inventory_values (Dict[Dict[int, float]]) : Array of last `config.scoring.sharpe.lookback` inventory values for the miner

    Returns: 
        float: The new score value for the given UID.
    """
    if not self.sharpe_values[uid]: return 0.0
    sharpes = self.sharpe_values[uid]['books']
    norm_min = self.config.scoring.sharpe.normalization_min
    norm_max = self.config.scoring.sharpe.normalization_max
    normalized_sharpes = {book_id: normalize(norm_min, norm_max, sharpe) 
                         for book_id, sharpe in sharpes.items()}
    
    volume_cap = round(self.config.scoring.activity.capital_turnover_cap * self.simulation.miner_wealth, 
                      self.simulation.volumeDecimals)
    lookback_threshold = self.simulation_timestamp - self.config.scoring.sharpe.lookback * self.simulation.publish_interval
    inactivity_decay_factor = 2 ** (-1 / self.config.scoring.sharpe.lookback)
    
    miner_volumes = {}
    latest_volumes = {}
    trade_volumes_uid = self.trade_volumes[uid]
    
    for book_id, book_volume in trade_volumes_uid.items():
        total_trades = book_volume['total']
        volume = 0.0
        for t, vol in sorted(total_trades.items(), reverse=True):
            if t >= lookback_threshold:
                volume += vol
            else:
                break
        miner_volumes[book_id] = volume
        latest_volumes[book_id] = list(total_trades.values())[-1] if total_trades else 0.0
    
    # Calculate the factor to be multiplied on the Sharpes when there has been no trading activity in the previous Sharpe assessment window
    # This factor is designed to reduce the activity multiplier by half after each `sharpe.lookback` steps of inactivity
    activity_factors_uid = self.activity_factors[uid]
    for book_id, miner_volume in miner_volumes.items():
        if latest_volumes[book_id] > 0:
            activity_factors_uid[book_id] = min(1 + (miner_volume / volume_cap), 2.0)
        else:
            activity_factors_uid[book_id] *= inactivity_decay_factor
    
    # Calculate the activity factors to be multiplied onto the Sharpes to obtain the final values for assessment
    # If the miner has traded in the previous Sharpe assessment window, the factor is equal to the ratio of the miner trading volume to the cap
    # If the miner has not traded, their existing activity factor is decayed by the factor defined above so as to halve the miner score over each Sharpe assessment window where they remain inactive
    activity_weighted_normalized_sharpes = []
    for book_id, activity_factor in activity_factors_uid.items():
        norm_sharpe = normalized_sharpes[book_id]
        if activity_factor < 1 or norm_sharpe > 0.5:
            weighted = activity_factor * norm_sharpe
        else:
            weighted = (2 - activity_factor) * norm_sharpe
        activity_weighted_normalized_sharpes.append(weighted)
    
    self.sharpe_values[uid]['books_weighted'] = {book_id: weighted_sharpe 
                                                  for book_id, weighted_sharpe in enumerate(activity_weighted_normalized_sharpes)}
    
    # Use the 1.5 rule to detect left-hand outliers in the activity-weighted Sharpes
    data = np.array(activity_weighted_normalized_sharpes)
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_threshold = q1 - 1.5 * iqr
    outliers = data[data < lower_threshold]
    
    # Outliers detected here are activity-weighted Sharpes which are significantly lower than those achieved on other books
    # A penalty equal to 67% of the difference between the mean outlier value and the value at the centre of the possible activity weighted Sharpe values is calculated
    outlier_penalty = (0.5 - np.mean(outliers)) / 1.5 if len(outliers) > 0 and np.mean(outliers) < 0.5 else 0
    
    # The median of the activity weighted Sharpes provides the base score for the miner
    activity_weighted_normalized_median = np.median(data)
    # The penalty factor is subtracted from the base score to punish particularly poor performance on any particular book
    sharpe_score = max(activity_weighted_normalized_median - abs(outlier_penalty), 0.0)
    
    self.sharpe_values[uid]['activity_weighted_normalized_median'] = activity_weighted_normalized_median
    self.sharpe_values[uid]['penalty'] = abs(outlier_penalty)
    self.sharpe_values[uid]['score'] = sharpe_score
    return self.reward_weights['sharpe'] * sharpe_score

def score_inventory_values(self, inventory_values):
    if self.config.scoring.sharpe.parallel_workers == 0:
        self.sharpe_values = {uid.item(): sharpe(uid, inventory_values[uid], self.config.scoring.sharpe.lookback, 
                                                 self.config.scoring.sharpe.normalization_min, 
                                                 self.config.scoring.sharpe.normalization_max, 
                                                 self.config.scoring.sharpe.min_lookback, 
                                                 self.simulation.grace_period, self.deregistered_uids) 
                             for uid in self.metagraph.uids}
    else:
        num_processes = self.config.scoring.sharpe.parallel_workers
        batch_size = int(256 / num_processes)
        batches = [self.metagraph.uids[i:i+batch_size] for i in range(0, 256, batch_size)]
        self.sharpe_values = batch_sharpe(inventory_values, batches, self.config.scoring.sharpe.lookback, 
                                          self.config.scoring.sharpe.normalization_min, 
                                          self.config.scoring.sharpe.normalization_max, 
                                          self.config.scoring.sharpe.min_lookback, 
                                          self.simulation.grace_period, self.deregistered_uids)

    inventory_scores = {uid: score_inventory_value(self, uid, self.inventory_history[uid]) 
                       for uid in self.metagraph.uids}
    return inventory_scores

def reward(self: Validator, synapse: MarketSimulationStateUpdate) -> Tuple[list[float], bool]:
    """
    Calculate and store the scores for a particular miner.

    Args:
        self (taos.im.neurons.validator.Validator) : Validator instance
        synapse (taos.im.protocol.MarketSimulationStateUpdate) : The latest state update synapse

    Returns:
        list[float]: The new score values for all uids in the subnet.
    """
    for bookId, book in synapse.books.items():
        trades = [TradeInfo.model_construct(**event) for event in book['e'] if event['y'] == 't']
        if trades:
            recent_trades_book = self.recent_trades[bookId]
            recent_trades_book.extend(trades)
            del recent_trades_book[:-25]  # Keep only last 25
    
    sampled_timestamp = math.ceil(synapse.timestamp / self.config.scoring.activity.trade_volume_sampling_interval) * self.config.scoring.activity.trade_volume_sampling_interval
    prune_threshold = synapse.timestamp - self.config.scoring.activity.trade_volume_assessment_period
    volume_decimals = self.simulation.volumeDecimals
    
    for uid in self.metagraph.uids:
        try:
            trade_volumes_uid = self.trade_volumes[uid]
            for book_id, role_trades in trade_volumes_uid.items():
                for role, trades in role_trades.items():
                    keys_to_delete = [time for time in trades if time < prune_threshold]
                    for time in keys_to_delete:
                        del trades[time]
                book_trade_volumes = trade_volumes_uid[book_id]
                if sampled_timestamp not in book_trade_volumes['total']:
                    book_trade_volumes['total'][sampled_timestamp] = 0.0
                    book_trade_volumes['maker'][sampled_timestamp] = 0.0
                    book_trade_volumes['taker'][sampled_timestamp] = 0.0
                    book_trade_volumes['self'][sampled_timestamp] = 0.0
            
            # Update trade volume history with new trades since the previous step
            trades = [TradeEvent.model_construct(**notice) for notice in synapse.notices[uid] 
                     if notice['y'] in ['EVENT_TRADE', "ET"]]
            
            recent_miner_trades_uid = self.recent_miner_trades[uid]
            for trade in trades:
                
                is_maker = trade.makerAgentId == uid
                is_taker = trade.takerAgentId == uid
                
                if is_maker:
                    recent_miner_trades_uid[trade.bookId].append([trade, "maker"])
                if is_taker:
                    recent_miner_trades_uid[trade.bookId].append([trade, "taker"])
                
                recent_miner_trades_uid[trade.bookId] = recent_miner_trades_uid[trade.bookId][-5:]
                
                book_volumes = trade_volumes_uid[trade.bookId]
                trade_value = round(trade.quantity * trade.price, volume_decimals)
                book_volumes['total'][sampled_timestamp] = round(book_volumes['total'][sampled_timestamp] + trade_value, volume_decimals)
                
                if trade.makerAgentId == trade.takerAgentId:
                    book_volumes['self'][sampled_timestamp] = round(book_volumes['self'][sampled_timestamp] + trade_value, volume_decimals)
                elif is_maker:
                    book_volumes['maker'][sampled_timestamp] = round(book_volumes['maker'][sampled_timestamp] + trade_value, volume_decimals)
                elif is_taker:
                    book_volumes['taker'][sampled_timestamp] = round(book_volumes['taker'][sampled_timestamp] + trade_value, volume_decimals)
            self.recent_miner_trades[uid] = recent_miner_trades_uid            
            if uid in synapse.accounts:
                initial_balances_uid = self.initial_balances[uid]
                accounts_uid = synapse.accounts[uid]
                
                for bookId, account in accounts_uid.items():
                    initial_balance_book = initial_balances_uid[bookId]
                    if initial_balance_book['BASE'] is None:
                        initial_balance_book['BASE'] = account['bb']['t']
                    if initial_balance_book['QUOTE'] is None:
                        initial_balance_book['QUOTE'] = account['qb']['t']
                    if initial_balance_book['WEALTH'] is None:
                        initial_balance_book['WEALTH'] = get_inventory_value(account, synapse.books[bookId])
                
                self.inventory_history[uid][synapse.timestamp] = {
                    book_id: get_inventory_value(accounts_uid[book_id], book) - initial_balances_uid[book_id]['WEALTH']
                    for book_id, book in synapse.books.items()
                }
            else:
                self.inventory_history[uid][synapse.timestamp] = {book_id: 0.0 for book_id in synapse.books}
            inventory_hist = self.inventory_history[uid]
            if len(inventory_hist) > self.config.scoring.sharpe.lookback:
                keys = sorted(inventory_hist.keys())
                for key in keys[:-self.config.scoring.sharpe.lookback]:
                    del inventory_hist[key]
                    
        except Exception as ex:
            bt.logging.error(f"Failed to update reward data for UID {uid} at step {self.step} : {traceback.format_exc()}")

    if synapse.timestamp % self.config.scoring.interval != 0:
        return None, False

    inventory_scores = score_inventory_values(self, self.inventory_history)
    rewards = list(inventory_scores.values())
    return rewards, True

def distribute_rewards(self: Validator, rewards: torch.FloatTensor):
    rng = np.random.default_rng(self.config.rewarding.seed)
    num_uids = len(self.metagraph.uids)
    distribution = torch.FloatTensor(sorted(self.config.rewarding.pareto.scale * rng.pareto(self.config.rewarding.pareto.shape, num_uids)))
    sorted_rewards, sorted_indices = rewards.sort()
    distributed_rewards = distribution * sorted_rewards
    return torch.gather(distributed_rewards, 0, sorted_indices.argsort())

def get_rewards(self: Validator, synapse: MarketSimulationStateUpdate) -> Tuple[torch.FloatTensor, bool]:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
        self (taos.im.neurons.validator.Validator) : Validator instance
        synapse (taos.im.protocol.MarketSimulationStateUpdate) : The latest state update object.

    Returns:
        torch.FloatTensor: A tensor of rewards for the given query and responses.
    """
    rewards, updated = reward(self, synapse)
    if updated:
        return distribute_rewards(self, torch.FloatTensor(rewards).to(self.device)), True
    else:
        return None, False

def set_delays(self: Validator, synapse_responses: dict[int, MarketSimulationStateUpdate]) -> list[FinanceAgentResponse]:
    """
    Applies base delay based on process time using an exponential mapping,
    and adds a per-book Gaussian-distributed random latency instruction_delay to instructions,
    with zero instruction_delay applied to the first instruction per book.

    Args:
        self (taos.im.neurons.validator.Validator): Validator instance.
        synapse_responses (dict[int, MarketSimulationStateUpdate]): Latest state updates.

    Returns:
        list[FinanceAgentResponse]: Delayed finance responses.
    """
    responses = []
    timeout = self.config.neuron.timeout
    min_delay = self.config.scoring.min_delay
    max_delay = self.config.scoring.max_delay
    min_instruction_delay = self.config.scoring.min_instruction_delay
    max_instruction_delay = self.config.scoring.max_instruction_delay

    def compute_delay(p_time: float) -> int:
        """Exponential scaling of process time into delay."""
        t = p_time / timeout
        exp_scale = 5
        delay_frac = (np.exp(exp_scale * t) - 1) / (np.exp(exp_scale) - 1)
        delay = min_delay + delay_frac * (max_delay - min_delay)
        return int(delay)

    for uid, synapse_response in synapse_responses.items():
        response = synapse_response.response
        if response:
            base_delay = compute_delay(synapse_response.dendrite.process_time)

            seen_books = set()
            for instruction in response.instructions:
                book_id = instruction.bookId

                # Zero instruction_delay for first instruction per book
                if book_id not in seen_books:
                    instruction_delay = 0
                    seen_books.add(book_id)
                else:
                    instruction_delay = random.randint(min_instruction_delay, max_instruction_delay)

                instruction.delay += base_delay + instruction_delay

            responses.append(response)
            bt.logging.info(
                f"UID {response.agent_id} responded with {len(response.instructions)} instructions "
                f"after {synapse_response.dendrite.process_time:.4f}s – base delay {base_delay}{self.simulation.time_unit}"
            )

    return responses