"""Offline conformance test for MinerAgent_V8 against the real taos 0.6.0 protocol models.

The invariant under test: the agent must never place a sell that would close the head of
its FIFO queue at a loss, and must never realize a negative P&L, on any price path.
"""
import random
import sys
import time
from types import SimpleNamespace

import bittensor as bt
from taos.im.protocol import MarketSimulationStateUpdate
from taos.im.protocol.models import Book, Account
from taos.im.protocol.events import TradeEvent, SimulationStartEvent
from taos.im.protocol.response import OrderDirection, TimeInForce
from taos.agent.MinerAgent_V8 import MinerAgent_V8

UID = 7
VH = "5ValidatorHotkeyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
S = 1_000_000_000
SIMCFG = SimpleNamespace(priceDecimals=2, volumeDecimals=4, min_order_size=0.25,
                         publish_interval=S, miner_wealth=50000.0, book_count=2,
                         grace_period=600 * S)


def mk_book(i, bid, ask, events=None):
    return Book.model_validate({"id": i, "MTR": 0.4,
                                "bids": [{"price": bid, "quantity": 50.0}],
                                "asks": [{"price": ask, "quantity": 50.0}],
                                "events": events or []})


def mk_acct(i, orders=None, base_total=80.0, maker=-0.0008):
    return Account.model_validate({
        "agent_id": UID, "book_id": i,
        "base_balance": {"currency": "BASE", "total": base_total, "free": base_total,
                         "reserved": 0.0, "initial": 80.0},
        "quote_balance": {"currency": "QUOTE", "total": 30000.0, "free": 30000.0,
                          "reserved": 0.0, "initial": 30000.0},
        "orders": orders or [], "loans": {},
        "fees": {"maker_fee_rate": maker, "taker_fee_rate": 0.0011}})


def mk_state(ts, books, accounts, notices):
    st = MarketSimulationStateUpdate.model_construct(
        timestamp=ts, config=SIMCFG, books=books,
        accounts={UID: accounts}, notices={UID: notices})
    st.dendrite = bt.TerminalInfo(hotkey=VH)
    return st


_tid = [0]


def trade(ts, book, i_bought, price, qty, maker=True, tid=None):
    """A fill for us. maker=True means we were the resting side."""
    _tid[0] += 1
    t = _tid[0] if tid is None else tid
    # wire `side` is the TAKER's direction: 0 = taker bought
    side = 1 if (maker and i_bought) else (0 if maker else (0 if i_bought else 1))
    fee = -0.0008 * price * qty if maker else 0.0011 * price * qty
    return TradeEvent.model_validate({
        "type": "ET", "timestamp": ts, "agentId": UID, "bookId": book, "tradeId": t,
        "takerAgentId": 99 if maker else UID, "takerOrderId": 5,
        "takerFee": 0.0 if maker else fee,
        "makerAgentId": UID if maker else 99, "makerOrderId": 11,
        "makerFee": fee if maker else 0.0,
        "side": side, "price": price, "quantity": qty})


def orders_of(resp):
    out = []
    for ins in resp.instructions:
        out.append((getattr(ins, "bookId", None), type(ins).__name__,
                    int(getattr(ins, "direction", -1)) if getattr(ins, "direction", None) is not None else None,
                    getattr(ins, "price", None), getattr(ins, "quantity", None)))
    return out


def sells(resp, book=None):
    return [o for o in orders_of(resp) if o[2] == 1 and (book is None or o[0] == book)]


def buys(resp, book=None):
    return [o for o in orders_of(resp) if o[2] == 0 and (book is None or o[0] == book)]


def new_agent(**params):
    cfg = SimpleNamespace(lazy_load=False, history_len=0, log_every=10**9, save_every=0, **params)
    return MinerAgent_V8(uid=UID, config=cfg, log_dir="/tmp/v8smoke")


def main():
    bt.logging.set_debug(False)
    agent = new_agent()
    ts = 1000 * S

    # 1. flat book -> a bid, no ask
    st = mk_state(ts, {0: mk_book(0, 299.90, 300.10), 1: mk_book(1, 199.90, 200.10)},
                  {0: mk_acct(0), 1: mk_acct(1)}, [])
    r = agent.handle(st)
    assert len(buys(r, 0)) == 1 and not sells(r, 0), orders_of(r)
    bid_px = buys(r, 0)[0][3]
    assert bid_px < 299.90, f"entry must be passive, got {bid_px} vs best bid 299.90"
    print(f"1 flat: passive bid {bid_px} (mid 300.00), no ask   OK")

    # 2. the bid fills -> an ask appears, strictly above cost, and profitable
    ts += S
    st = mk_state(ts, {0: mk_book(0, 299.90, 300.10), 1: mk_book(1, 199.90, 200.10)},
                  {0: mk_acct(0, base_total=80.25), 1: mk_acct(1)},
                  [trade(ts, 0, True, bid_px, 0.25)])
    r = agent.handle(st)
    bk = agent.books_by_validator[VH][0]
    assert len(bk.lots) == 1 and abs(bk.inventory() - 0.25) < 1e-9
    ask = sells(r, 0)
    assert len(ask) == 1, orders_of(r)
    ask_px = ask[0][3]
    head = bk.lots[0]
    realized = (ask_px - head.price) * 0.25 - head.fee - (-0.0008 * ask_px * 0.25)
    assert ask_px > head.price and realized > 0, (ask_px, head.price, realized)
    print(f"2 fill: lot @{head.price} fee {head.fee:+.4f} -> ask {ask_px}, realized would be {realized:+.4f}   OK")

    # 3. the ask fills -> realized P&L is positive and the book is flat again
    ts += S
    st = mk_state(ts, {0: mk_book(0, ask_px - 0.02, ask_px), 1: mk_book(1, 199.90, 200.10)},
                  {0: mk_acct(0), 1: mk_acct(1)}, [trade(ts, 0, False, ask_px, 0.25)])
    agent.handle(st)
    assert not bk.lots and bk.realized > 0 and agent.n_losses == 0, (bk.realized, agent.n_losses)
    print(f"3 harvest: realized {bk.realized:+.4f}, flat, losses {agent.n_losses}   OK")

    # 4. THE INVARIANT: on a long adverse path with REAL resting-order fills, the agent must
    #    never place a sell at or below the head lot, and must never realize a loss.
    for label, drift, vol, nticks in (("down", -0.00006, 0.0012, 2000),
                                      ("choppy", 0.0, 0.0018, 2000),
                                      ("up", +0.00006, 0.0012, 2000)):
        agent2 = new_agent()
        random.seed(11)
        px = 300.0
        ts2 = 5000 * S
        resting = {}          # oid -> (side, price, qty); side 0 = our bid
        next_oid = [1]
        violations = placed_sells = fills = 0
        maxinv = 0.0
        for step in range(nticks):
            ts2 += S
            px = max(30.0, px * (1 + random.gauss(drift, vol)))
            b, a = round(px - 0.05, 2), round(px + 0.05, 2)
            bk2 = agent2.books_by_validator.get(VH, {}).get(0)
            inv = bk2.inventory() if bk2 else 0.0
            maxinv = max(maxinv, inv)
            # a resting bid fills when the market's ask falls to it; a resting ask when the bid rises to it
            notices = []
            for oid, (side, p, q) in list(resting.items()):
                if side == 0 and a <= p:
                    notices.append(trade(ts2, 0, True, p, q)); del resting[oid]; fills += 1
                elif side == 1 and b >= p:
                    notices.append(trade(ts2, 0, False, p, q)); del resting[oid]; fills += 1
            orders = [{"id": oid, "timestamp": ts2, "quantity": q, "side": sd, "price": p}
                      for oid, (sd, p, q) in resting.items()]
            acct = mk_acct(0, orders=orders, base_total=80.0 + inv)
            r = agent2.handle(mk_state(ts2, {0: mk_book(0, b, a)}, {0: acct}, notices))
            bk2 = agent2.books_by_validator[VH][0]
            for ins in r.instructions:
                nm = type(ins).__name__
                if nm.startswith("CancelOrders"):
                    for c in (getattr(ins, "cancellations", None) or []):
                        resting.pop(getattr(c, "orderId", None), None)
                elif nm.startswith("PlaceLimitOrder"):
                    d = int(ins.direction)
                    if d == 1:
                        placed_sells += 1
                        if bk2.lots and float(ins.price) <= bk2.lots[0].price:
                            violations += 1
                    resting[next_oid[0]] = (d, float(ins.price), float(ins.quantity))
                    next_oid[0] += 1
        bk2 = agent2.books_by_validator[VH][0]
        assert violations == 0, f"{label}: {violations} sells at or below the head lot"
        assert agent2.n_losses == 0, f"{label}: {agent2.n_losses} realized losses"
        assert bk2.realized >= 0, f"{label}: negative realized {bk2.realized}"
        per3h = bk2.closes / (nticks / 10800.0)
        print(f"4-{label:6s} {nticks} ticks: fills {fills:4d} closes {bk2.closes:4d} "
              f"(~{per3h:5.0f}/3h) realized {bk2.realized:+8.3f} below-cost {violations} "
              f"losses {agent2.n_losses} open {len(bk2.lots)} maxinv {maxinv:.2f} realized/close {bk2.realized/max(1,bk2.closes):+.4f}   OK")

    # 5. unknown inventory blocks selling entirely
    agent3 = new_agent()
    ts3 = 9000 * S
    st = mk_state(ts3, {0: mk_book(0, 299.90, 300.10)}, {0: mk_acct(0, base_total=85.0)}, [])
    r = agent3.handle(st)
    assert agent3.books_by_validator[VH][0].blocked and not sells(r, 0), orders_of(r)
    print("5 unknown inventory (+5.0 base we cannot explain): selling disabled   OK")

    # 6. duplicate trade ids are applied once
    agent4 = new_agent()
    ts4 = 11000 * S
    ev = trade(ts4, 0, True, 299.0, 0.25, tid=4242)
    agent4.handle(mk_state(ts4, {0: mk_book(0, 299.90, 300.10)}, {0: mk_acct(0, base_total=80.25)}, [ev, ev]))
    b4 = agent4.books_by_validator[VH][0]
    assert len(b4.lots) == 1, f"duplicate applied twice: {len(b4.lots)} lots"
    # the same trade arriving again in the book event list must also be ignored
    tinfo = {"y": "t", "id": 4242, "side": 1, "timestamp": ts4, "quantity": 0.25,
             "price": 299.0, "taker_agent_id": 99, "maker_agent_id": UID,
             "taker_fee": 0.0, "maker_fee": -0.06}
    agent4.handle(mk_state(ts4 + S, {0: mk_book(0, 299.90, 300.10, events=[tinfo])},
                           {0: mk_acct(0, base_total=80.25)}, []))
    assert len(b4.lots) == 1, f"book-event duplicate applied: {len(b4.lots)} lots"
    print("6 dedupe: the same trade id from notice and book event applies once   OK")

    # 7. simulation restart clears the ledger
    ts5 = 700 * S
    ess = SimulationStartEvent.model_validate({"type": "ESS", "timestamp": ts5, "agentId": UID, "logDir": "/x"})
    agent4.handle(mk_state(ts5, {0: mk_book(0, 300.0, 300.2)}, {0: mk_acct(0)}, [ess]))
    assert not agent4.books_by_validator[VH][0].lots
    print("7 restart: lots cleared   OK")

    # 8. response time on 128 books
    agent5 = new_agent()
    books = {i: mk_book(i, 299.90 + i * 0.01, 300.10 + i * 0.01) for i in range(128)}
    accts = {i: mk_acct(i) for i in range(128)}
    st = mk_state(20000 * S, books, accts, [])
    t0 = time.perf_counter()
    r = agent5.handle(st)
    dt = time.perf_counter() - t0
    per_book = {}
    for o in orders_of(r):
        per_book[o[0]] = per_book.get(o[0], 0) + 1
    assert dt < 0.5, dt
    assert max(per_book.values()) <= 5, per_book
    print(f"8 timing: 128 books in {dt*1000:.1f} ms, {len(r.instructions)} instructions, "
          f"max {max(per_book.values())}/book (limit 5)   OK")

    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
