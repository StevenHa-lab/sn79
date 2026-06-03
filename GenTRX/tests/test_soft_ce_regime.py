"""Tests for live soft-CE wiring and the train-regime checkpoint gate.

Pins two rules:
  - soft CE (label_smooth_sigma>0) actually changes the loss, and the live
    config defaults carry it (so a launch trains/scores under soft CE, not the
    strict-CE library default of compute_loss/WindowConfig).
  - the gradient server refuses to resume a checkpoint stamped below the
    current train regime, falling back to warmup instead.

Run: pytest GenTRX/tests/test_soft_ce_regime.py -v
"""

import torch

from GenTRX.src.distributed import WindowConfig
from GenTRX.src.miner_training_service import MinerTrainingConfig
from GenTRX.src.model import compute_loss
from GenTRX.src.version import TRAIN_REGIME_VERSION


# ---------------------------------------------------------------------------
# Soft CE changes the loss
# ---------------------------------------------------------------------------


def test_soft_ce_differs_from_strict_on_ordinal_field():
    torch.manual_seed(0)
    logits = {"price": torch.randn(2, 4, 100)}
    labels = {"price": torch.randint(0, 100, (2, 4))}
    strict, _ = compute_loss(logits, labels, label_smooth_sigma=0.0)
    soft, _ = compute_loss(logits, labels, label_smooth_sigma=1.0)
    assert abs(strict.item() - soft.item()) > 1e-4


def test_compute_loss_library_default_is_strict():
    """The primitive defaults to no-op (0.0); the regime value lives in config."""
    torch.manual_seed(0)
    logits = {"price": torch.randn(2, 4, 100)}
    labels = {"price": torch.randint(0, 100, (2, 4))}
    default, _ = compute_loss(logits, labels)
    strict, _ = compute_loss(logits, labels, label_smooth_sigma=0.0)
    assert default.item() == strict.item()


# ---------------------------------------------------------------------------
# Live config defaults carry sigma=1.0
# ---------------------------------------------------------------------------


def test_miner_config_default_sigma_is_one():
    assert MinerTrainingConfig(uid=0).label_smooth_sigma == 1.0


def test_window_config_library_default_is_strict():
    assert WindowConfig().label_smooth_sigma == 0.0


# ---------------------------------------------------------------------------
# Regime gate
# ---------------------------------------------------------------------------


class _MetaStore:
    def __init__(self, meta):
        self._meta = meta

    def get_latest_meta(self, uid):
        return self._meta


def _make_aggregator(tmp_path, validator_store=None):
    from GenTRX.src.gradient_server import GradientAggregator

    return GradientAggregator(
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        val_data_path=str(tmp_path / "val"),
        output_path=str(tmp_path / "out.pt"),
        books_per_miner=1,
        interval=60,
        window_ns=50,
        warmup_rounds=0,
        rollback=False,
        validator_store=validator_store,
    )


def test_regime_incompatible_when_stamp_below_current(tmp_path):
    store = _MetaStore({"train_regime_version": TRAIN_REGIME_VERSION - 1})
    agg = _make_aggregator(tmp_path, validator_store=store)
    assert agg._regime_incompatible() is True


def test_regime_compatible_when_stamp_current(tmp_path):
    store = _MetaStore({"train_regime_version": TRAIN_REGIME_VERSION})
    agg = _make_aggregator(tmp_path, validator_store=store)
    assert agg._regime_incompatible() is False


def test_unstamped_checkpoint_is_incompatible(tmp_path):
    """A pre-versioning checkpoint (no stamp) reads as regime 0 → warmup."""
    store = _MetaStore({"version": 7})
    agg = _make_aggregator(tmp_path, validator_store=store)
    assert agg._regime_incompatible() is True


def test_no_store_defaults_compatible(tmp_path):
    agg = _make_aggregator(tmp_path, validator_store=None)
    assert agg._regime_incompatible() is False
