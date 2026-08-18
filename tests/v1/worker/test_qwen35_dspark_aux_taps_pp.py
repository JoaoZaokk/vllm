# SPDX-License-Identifier: Apache-2.0
"""Aux-tap accounting for THIS deployment: Qwen3.5-27B + DSpark under PP=2.

The upstream accounting test covers Kimi-K3 (93 layers, taps 3/24/48/72/90).
This one pins the shape we actually run, because the failure mode here is
silent: a drafter fed taps that are missing, duplicated or out of order still
emits syntactically valid proposals. It just gets more of them rejected. So a
boot proves nothing and only the acceptance rate would ever complain -- much
too late and much too vague.

Numbers come from the checkpoint, not from a guess:
    models/dspark-qwen38/config.json
      dflash_config.target_layer_ids = [4, 16, 28, 40, 52]
      dspark_target_layer_ids        = [4, 16, 28, 40, 52]
    get_eagle3_aux_layers_from_config adds 1 to DFlash/DSpark ids, so the aux
    layers are 5, 17, 29, 41, 53 of 64.

The 56/8 split is not arbitrary either: it is the only partition measured to
boot on this pair of cards (3090 + 3080 Ti), since the 3080 Ti runs out of KV
memory beyond 8 layers as the last stage.

CPU only, no distributed init.
"""

import pytest

from vllm.distributed.utils import get_pp_indices
from vllm.model_executor.models.interfaces import EagleModelMixin

NUM_LAYERS = 64
TARGET_LAYER_IDS = (4, 16, 28, 40, 52)          # do checkpoint
AUX_IDS = tuple(i + 1 for i in TARGET_LAYER_IDS)  # (5, 17, 29, 41, 53)


def taps(start, end, is_first_rank):
    return list(EagleModelMixin.local_aux_tap_ids(start, end, AUX_IDS, is_first_rank))


def gather(pp_size):
    out = []
    for rank in range(pp_size):
        start, end = get_pp_indices(NUM_LAYERS, rank, pp_size)
        out.extend(taps(start, end, rank == 0))
    return out


@pytest.mark.parametrize("pp_size", [1, 2])
def test_all_five_taps_arrive_in_order(pp_size):
    # Order is load-bearing: the drafter concatenates the taps positionally.
    assert gather(pp_size) == list(AUX_IDS)


@pytest.mark.parametrize("pp_size", [1, 2])
def test_no_tap_is_duplicated(pp_size):
    got = gather(pp_size)
    assert len(got) == len(set(got)) == len(AUX_IDS)


def test_measured_split_puts_every_tap_upstream(monkeypatch):
    """56/8, the split that actually boots here.

    Every tap sits at layer 53 or below, so the 3090 emits all five and the
    3080 Ti contributes none. That is the maximum forwarding case -- worth
    stating explicitly, because it is also the arrangement whose cost we pay:
    5 x 5120 x 2 bytes = 50 KiB per position crossing PCIe every step.
    """
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "56,8")
    start0, end0 = get_pp_indices(NUM_LAYERS, 0, 2)
    start1, end1 = get_pp_indices(NUM_LAYERS, 1, 2)
    assert (start0, end0, start1, end1) == (0, 56, 56, 64)

    assert taps(start0, end0, True) == list(AUX_IDS)
    assert taps(start1, end1, False) == []
    assert gather(2) == list(AUX_IDS)


def test_split_below_the_last_tap_still_loses_nothing(monkeypatch):
    """A cut at 48 leaves tap 53 downstream: the last stage must emit it.

    Not a configuration we run, but it is the one that would break a
    forwarding rule written as "the first stage owns every tap".
    """
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "48,16")
    start0, end0 = get_pp_indices(NUM_LAYERS, 0, 2)
    start1, end1 = get_pp_indices(NUM_LAYERS, 1, 2)
    assert taps(start0, end0, True) == [5, 17, 29, 41]
    assert taps(start1, end1, False) == [53]
    assert gather(2) == list(AUX_IDS)


def test_drafterless_run_forwards_nothing(monkeypatch):
    # Without a drafter the payload must stay empty: PP alone pays no tax.
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "56,8")
    for rank in range(2):
        start, end = get_pp_indices(NUM_LAYERS, rank, 2)
        assert list(EagleModelMixin.local_aux_tap_ids(start, end, (), rank == 0)) == []
