# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
import numpy as np
import traceback
from loky.backend.context import set_start_method
set_start_method('forkserver', force=True)
from loky import get_reusable_executor

from taos.im.utils import normalize


def sharpe(uid, inventory_values, lookback, norm_min, norm_max, min_lookback, grace_period, deregistered_uids) -> dict:
    """
    Calculates intraday Sharpe ratios for a particular UID using the change in inventory values over previous `config.scoring.sharpe.lookback` observations to represent returns.
    Values are also stored to a property of the Validator class to be accessed later for scoring and reporting purposes.

    Args:
        self (taos.im.neurons.validator.Validator) : Validator instance
        uid (int) : UID of miner being scored
        inventory_values (Dict[Dict[int, float]]) : Array of last `config.scoring.sharpe.lookback` inventory values for the miner

    Returns:
    dict: A dictionary containing all relevant calculated Sharpe values for the UID.  This includes Sharpe for their total inventory value and Sharpe calculated on each book along with
          several aggregate values obtained from the values for each book and their normalized counterparts.
    """
    try:
        num_values = len(inventory_values)
        if uid in deregistered_uids or num_values < min(min_lookback, lookback):
            return None
        timestamps = list(inventory_values.keys())
        book_ids = list(next(iter(inventory_values.values())).keys())

        np_inventory_values = np.array([
            [inventory_values[ts][book_id] for book_id in book_ids]
            for ts in timestamps
        ], dtype=np.float64).T
        changeover_mask = None
        if grace_period > 0:
            ts_array = np.array(timestamps, dtype=np.int64)
            time_diffs = np.diff(ts_array)
            changeover_indices = np.where(time_diffs >= grace_period)[0]
            
            if len(changeover_indices) > 0:
                changeover_mask = np.ones(len(timestamps) - 1, dtype=bool)
                changeover_mask[changeover_indices] = False
        returns = np.diff(np_inventory_values, axis=1)  # Shape: (num_books, num_timestamps-1)
        
        if changeover_mask is not None:
            returns = returns[:, changeover_mask]
        means = returns.mean(axis=1)
        stds = returns.std(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            sharpe_ratios = np.where(
                stds != 0.0,
                np.sqrt(returns.shape[1]) * (means / stds),
                0.0
            )
        sharpe_values = {
            'books': {book_id: float(sharpe_ratios[i]) for i, book_id in enumerate(book_ids)}
        }
        sharpe_values['average'] = float(sharpe_ratios.mean())
        sharpe_values['median'] = float(np.median(sharpe_ratios))
        total_inventory = np_inventory_values.sum(axis=0)
        total_returns = np.diff(total_inventory)
        
        if changeover_mask is not None:
            total_returns = total_returns[changeover_mask]
        
        total_std = total_returns.std()
        sharpe_values['total'] = float(
            np.sqrt(len(total_returns)) * (total_returns.mean() / total_std)
            if total_std != 0.0 else 0.0
        )

        sharpe_values['normalized_average'] = normalize(norm_min, norm_max, sharpe_values['average'])
        sharpe_values['normalized_median'] = normalize(norm_min, norm_max, sharpe_values['median'])
        sharpe_values['normalized_total'] = normalize(norm_min, norm_max, sharpe_values['total'])        
        return sharpe_values        
    except Exception as ex:
        print(f"Failed to calculate Sharpe for UID {uid}: {traceback.format_exc()}")
        return None


def sharpe_batch(inventory_values, lookback, norm_min, norm_max, min_lookback, grace_period, deregistered_uids):
    """Process a batch of UIDs for Sharpe calculation"""
    return {uid: sharpe(uid, inventory_value, lookback, norm_min, norm_max, min_lookback, grace_period, deregistered_uids) 
            for uid, inventory_value in inventory_values.items()}


def batch_sharpe(inventory_values, batches, lookback, norm_min, norm_max, min_lookback, grace_period, deregistered_uids):
    """Parallel processing of Sharpe calculations across multiple batches"""
    pool = get_reusable_executor(max_workers=len(batches))
    
    # Submit all tasks
    tasks = [pool.submit(sharpe_batch, 
                        {uid: inventory_values[uid] for uid in batch}, 
                        lookback, norm_min, norm_max, min_lookback, grace_period, deregistered_uids) 
             for batch in batches]
    
    # Collect results and merge into single dictionary
    result = {}
    for task in tasks:
        batch_result = task.result()
        for k, v in batch_result.items():
            result[int(k)] = v
    
    return result