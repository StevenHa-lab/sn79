"""Chunked Arrow event buffer parity.

Pin the rule: splitting collect_row() dicts into columnar chunks and
recombining with pa.Table.from_batches yields a table byte-identical to the
prior single-pass list[dict] -> pa.table path, with the same schema and the
split volume_int/volume_dec fields preserved.

Run: pytest GenTRX/tests/test_event_chunking.py -v
"""

import numpy as np
import pyarrow as pa

from GenTRX.src.util.schema import LOB_DEPTH, order_stream_schema
from taos.im.agents.gentrx import _events_to_batch


def _make_events(n: int) -> list[dict]:
    events = []
    for i in range(n):
        row = {
            "timestamp": 1_000_000_000 * (i + 1),
            "order_type": i % 3,
            "rel_price": (-1) ** i * (i + 1),
            "volume_int": i,
            "volume_dec": (i % 7) / 7.0,
            "interval_ns": 1_000 * i,
            "mid_price": 100_000 + i,
            "time_of_day_s": (i * 13) % 86400,
            "mid_price_delta": i - 5,
        }
        for j in range(LOB_DEPTH):
            row[f"lob_ask_vol_{j + 1}"] = float(i + j) / 10.0
            row[f"lob_bid_vol_{j + 1}"] = float(i - j) / 10.0
        events.append(row)
    return events


def _reference_table(events: list[dict]) -> pa.Table:
    """The pre-refactor single-pass list[dict] -> pa.table construction."""
    columns: dict = {
        "timestamp": pa.array(
            [e["timestamp"] for e in events], type=pa.timestamp("ns")
        ),
        "order_type": np.array([e["order_type"] for e in events], dtype=np.int8),
        "rel_price": np.array([e["rel_price"] for e in events], dtype=np.int32),
        "volume_int": np.array([e["volume_int"] for e in events], dtype=np.int32),
        "volume_dec": np.array([e["volume_dec"] for e in events], dtype=np.float32),
        "interval_ns": np.array([e["interval_ns"] for e in events], dtype=np.int64),
        "mid_price": np.array([e["mid_price"] for e in events], dtype=np.int64),
        "time_of_day_s": np.array(
            [e["time_of_day_s"] for e in events], dtype=np.int32
        ),
        "mid_price_delta": np.array(
            [e["mid_price_delta"] for e in events], dtype=np.int32
        ),
    }
    for i in range(LOB_DEPTH):
        k_ask = f"lob_ask_vol_{i + 1}"
        k_bid = f"lob_bid_vol_{i + 1}"
        columns[k_ask] = np.array([e[k_ask] for e in events], dtype=np.float64)
        columns[k_bid] = np.array([e[k_bid] for e in events], dtype=np.float64)
    return pa.table(columns, schema=order_stream_schema())


def _chunked(events: list[dict], chunk_size: int) -> list[list[dict]]:
    return [events[i : i + chunk_size] for i in range(0, len(events), chunk_size)]


def test_chunk_path_equals_single_pass():
    events = _make_events(25)
    reference = _reference_table(events)

    for chunk_size in (1, 3, 10, 25, 100):
        batches = [_events_to_batch(c) for c in _chunked(events, chunk_size)]
        table = pa.Table.from_batches(batches, schema=order_stream_schema())
        assert table.schema.equals(order_stream_schema())
        assert table.equals(reference), f"mismatch at chunk_size={chunk_size}"


def test_volume_split_preserved():
    events = _make_events(8)
    table = pa.Table.from_batches(
        [_events_to_batch(events)], schema=order_stream_schema()
    )
    assert table.column("volume_int").to_pylist() == [e["volume_int"] for e in events]
    np.testing.assert_allclose(
        table.column("volume_dec").to_pylist(),
        np.array([e["volume_dec"] for e in events], dtype=np.float32),
    )
