# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT

from typing import Dict

def get_inventory_value(account: Dict, book: Dict, method='midquote') -> float:
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