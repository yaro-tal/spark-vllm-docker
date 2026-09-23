#!/usr/bin/env python3
"""Checks for patches 04 (wave readiness evaluated at load) and 05 (staged
wave rows held against eviction until their load retires).

1. Replays a remembered-then-evicted wave key on vLLM's real
   CPUOffloadingManager: a wave key becomes resident (ref_cnt 0, evictable)
   while the rest of its wave is still promoting, another request's store
   evicts it, then the rest lands. A driver that remembered the first key as
   ready would call prepare_load and die on `assert block is not None`.
   wave_lookup must answer HIT_PENDING while a promotion is writing, MISS
   after the eviction, and HIT only when prepare_load is safe.
2. A completed promotion is held non-evictable until its load retires, and the
   hold is released by the load, by a re-stage, and at finalize. A key two
   requests' waves share is held for each of them, whether it lands from a
   read or from another request's store.
   Staging also holds a wave's already-resident keys and expects the whole
   wave; a finished wave neither holds nor expects anything. A promotion no
   wave staged is not held (so the patch is inert with wave streaming off),
   and a zero-key promotion is never submitted.
3. The wave-retry counter reaches Prometheus with its `reason` label.

Run inside an image with the patch applied (no GPU, no model):
  docker run --rm --entrypoint python3 -v "$PWD/<this file>:/t.py:ro" <image> /t.py
"""
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from prometheus_client import REGISTRY, Counter, Gauge, Histogram, generate_latest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    OffloadPromMetrics,
    _ConnectorMetricName,
)
from vllm.v1.kv_offload.base import LookupResult, ReqContext
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager

# 1. readiness
A, B, C, D = (bytes([i]) * 36 for i in range(4))
ctx = ReqContext(req_id="wave")
other = ReqContext(req_id="other")
primary = CPUOffloadingManager(num_blocks=2)
tiering = SimpleNamespace(primary_tier=primary)


def wave_lookup(keys):
    return TieringOffloadingManager.wave_lookup(tiering, keys, ctx)


def store(keys, req_context, complete=True):
    assert primary.prepare_store(keys, req_context) is not None
    if complete:
        primary.complete_store(keys, req_context)


store([A], ctx)                       # A resident, evictable
store([B], ctx, complete=False)       # B still promoting
assert wave_lookup([A, B]) is LookupResult.HIT_PENDING

store([C], other)                     # another request's store evicts A (LRU)
primary.complete_store([B], ctx)      # the rest of the wave lands
assert wave_lookup([A, B]) is LookupResult.MISS
try:
    primary.prepare_load([A, B], ctx)
except AssertionError as e:           # what a remembered readiness would hit
    assert "not found in cache" in str(e)
else:
    raise SystemExit("expected prepare_load to reject an evicted key")

store([A], ctx)                       # re-staged: evicts C, A and B resident
assert wave_lookup([A, B]) is LookupResult.HIT
primary.prepare_load([A, B], ctx)     # pins both; safe
assert primary.prepare_store([D], other) is None  # pinned rows are not evictable
print("OK wave_lookup: pending -> evicted MISS -> re-staged HIT, prepare_load safe")

# 2. a completed promotion is held against another request's store, and the
#    hold is released by the load, by a re-stage, and at finalize
class FakeFsTier:  # a class, not SimpleNamespace: it is used as a dict key
    tier_type = "fs"

    @staticmethod
    def get_finished_jobs():
        return [SimpleNamespace(job_id=0, success=True)]

    @staticmethod
    def on_request_finished(req_context):
        return None


def bind(tier):
    """Give a stripped-down manager the helpers the code under test calls."""
    for name in ("_release_staged_pins", "_hold_for_waiting_waves",
                 "_maybe_finalize_request"):
        setattr(tier, name, partial(getattr(TieringOffloadingManager, name), tier))
    return tier


def staging_tier(num_blocks=1, expect=(A,)):
    """A tiering manager reduced to what the staging hold touches, with a
    secondary tier that reports one finished promotion of [A]."""
    primary = CPUOffloadingManager(num_blocks=num_blocks)
    primary.complete_write = primary.complete_store
    primary.prepare_write = primary.prepare_store
    state = SimpleNamespace(req_context=ctx, staged_pins=set(),
                            wave_expect=set(expect))
    job = SimpleNamespace(
        job_id=0, keys=[A], block_ids=np.array([0]), is_promotion=True,
        req_context=ctx,
    )
    fs = FakeFsTier()
    tier = SimpleNamespace(
        primary_tier=primary, _req_state={"wave": state},
        _transfer_jobs={0: job}, secondary_tiers=[fs], _promoted_rows=[],
    )
    bind(tier)
    assert primary.prepare_store([A], ctx) is not None   # promotion reserved it
    TieringOffloadingManager._process_finished_jobs(tier)  # promotion completes
    assert state.wave_expect == set(), "a landed key is no longer expected"
    if expect:
        assert state.staged_pins == {A}, "a completed promotion is held"
        if num_blocks == 1:  # with a spare row a store need not evict anything
            assert primary.prepare_store([B], other) is None, "the hold blocks it"
    return tier, primary, state


tier, primary, state = staging_tier()
primary.prepare_load([A], ctx)        # the wave ships: build_connector_meta pins
TieringOffloadingManager.complete_load(tier, [A], ctx)   # ... and the load retires
assert state.staged_pins == set()
assert primary.prepare_store([B], other) is not None, "evictable again"

tier, primary, state = staging_tier()
TieringOffloadingManager.release_wave_pins(tier, ctx)    # the wave re-stages
assert state.staged_pins == set(), "a re-staging wave holds nothing"
assert state.wave_expect == set(), "and expects nothing, so a late landing is free"
assert primary.prepare_store([B], other) is not None

# finalize through the real path: on_request_finished -> _maybe_finalize_request
tier, primary, state = staging_tier(num_blocks=2)
state.is_finished = False
state.pending_primary_stores = 0
state.request_level_tiers = None
state.sync_lookup_delay = 0.0
state.secondary_lookup_start_time = None
tier._maybe_observe_lookup_sync_delay = lambda st: None
tier._maybe_observe_lookup_async_delay = lambda st: None
TieringOffloadingManager.on_request_finished(tier, ctx)
assert "wave" not in tier._req_state, "the request state is gone"
assert primary.prepare_store([B, C], other) is not None, "its rows are evictable"

# a wave of one resident key and one it must read: BOTH are held, and the
# resident one survives a competing store until the load retires
resident = CPUOffloadingManager(num_blocks=2)
resident.prepare_write = resident.prepare_store
assert resident.prepare_store([A], ctx) is not None
resident.complete_store([A], ctx)                    # A is resident, evictable
rstate = SimpleNamespace(req_context=ctx, staged_pins=set(), wave_expect=set())
rtier = SimpleNamespace(
    primary_tier=resident, _req_state={"wave": rstate},
    secondary_tiers=[FakeFsTier()], _pending_load_submissions={},
)
assert TieringOffloadingManager.promote_for_staging(rtier, [A, B], ctx) is True
assert rstate.staged_pins == {A}, "the wave's resident key is held at staging"
assert rstate.wave_expect == {A, B}, "the whole wave is expected, not just B"
assert resident.prepare_store([C], other) is None, "so a store cannot take it"
TieringOffloadingManager._release_staged_pins(rtier, "wave", [A, B])  # load retires
assert rstate.staged_pins == set() and rstate.wave_expect == set(), (
    "a finished wave neither holds nor expects anything"
)
print("OK staging holds a wave's resident keys; finalize frees everything")


# a promotion no wave staged is NOT held: with wave streaming off every queried
# key is promoted, and holding those would pin rows no load will ask for
tier, primary, state = staging_tier(expect=())
assert state.staged_pins == set(), "only a wave's own promotion is held"
assert primary.prepare_store([B], other) is not None, "so the row stays evictable"

# two requests share a key: B's wave needs X, which A's wave is reading. When
# A's read lands, B holds X too, so A finishing with X cannot expose it to a
# store while B still waits for Y.
X, Y, Z, W = (bytes([10 + i]) * 36 for i in range(4))
wa, wb, wc = (ReqContext(req_id=r) for r in ("wa", "wb", "wc"))
shared = CPUOffloadingManager(num_blocks=3)
shared.prepare_write = shared.prepare_store
shared.complete_write = shared.complete_store
sa, sb = (SimpleNamespace(req_context=c, staged_pins=set(), wave_expect=set())
          for c in (wa, wb))
landing = SimpleNamespace(job_id=0, keys=[X], block_ids=np.array([0]),
                          is_promotion=True, req_context=wa)
stier = bind(SimpleNamespace(
    primary_tier=shared, _req_state={"wa": sa, "wb": sb},
    secondary_tiers=[FakeFsTier()], _pending_load_submissions={},
    _transfer_jobs={0: landing}, _promoted_rows=[],
))
assert TieringOffloadingManager.promote_for_staging(stier, [X], wa)      # A reads X
assert TieringOffloadingManager.promote_for_staging(stier, [X, Y], wb)   # B reads Y
assert sb.wave_expect == {X, Y} and sb.staged_pins == set(), "X is A's to read"
TieringOffloadingManager._process_finished_jobs(stier)                  # X lands
assert sa.staged_pins == {X} and sb.staged_pins == {X}, "both waves hold X"
shared.prepare_load([X], wa)                                           # A ships
TieringOffloadingManager.complete_load(stier, [X], wa)                 # ... retires
assert sa.staged_pins == set() and sb.staged_pins == {X}, "B still holds X"
assert shared.prepare_store([Z, W], wc) is None, "so a store cannot take X"
shared.complete_write([Y], wb)                                         # Y lands
TieringOffloadingManager._hold_for_waiting_waves(stier, [Y])
shared.prepare_load([X, Y], wb)                                        # B ships
TieringOffloadingManager.complete_load(stier, [X, Y], wb)              # ... retires
assert sb.staged_pins == set() and sb.wave_expect == set()
assert shared.prepare_store([Z, W], wc) is not None, "all evictable again"

# the store path: request C's GPU store writes a key B's wave is waiting for
Q = bytes([20]) * 36
store_path = CPUOffloadingManager(num_blocks=2)
qb = SimpleNamespace(req_context=wb, staged_pins=set(), wave_expect={Q})
qc = SimpleNamespace(req_context=wc, staged_pins=set(), wave_expect=set(),
                     pending_primary_stores=1, is_finished=False)
qtier = bind(SimpleNamespace(primary_tier=store_path, secondary_tiers=[],
                             _req_state={"wb": qb, "wc": qc}))
assert store_path.prepare_store([Q], wc) is not None                   # C stores Q
TieringOffloadingManager.complete_store(qtier, [Q], wc, True)
assert qb.staged_pins == {Q}, "B's wave holds the key C just wrote"
print("OK a shared key is held for every waiting wave, from a read or a store")

# a wave whose keys are all resident reserves nothing, and a zero-key job never
# completes on a tier that counts tasks, so it must not be submitted at all
submitted = []


class FakeTier:  # SimpleNamespace defines __eq__, so it cannot be a dict key
    submit_load = submitted.append


empty = SimpleNamespace(
    _pending_load_submissions={
        FakeTier(): {"wave": SimpleNamespace(keys=[], block_ids=[], req_context=ctx)}
    },
    _transfer_jobs={},
    _next_job_id=lambda: 1,
)
TieringOffloadingManager._flush_pending_promotions(empty)
assert submitted == [] and empty._transfer_jobs == {}, "no zero-key promotion"
print("OK held against a store; released by load, re-stage, finalize;"
      " inert without a wave; no zero-key job")

# 3. counter
stats = OffloadingConnectorStats()
for reason in ("miss", "miss", "promote_refused"):
    stats.increase_counter(_ConnectorMetricName.WAVE_RETRY, labelvalues=(reason,))


class NoSpecMetrics:  # OffloadPromMetrics requires a spec class
    @staticmethod
    def build_metric_definitions(extra_config):
        return {}


with patch.object(OffloadingSpecFactory, "get_spec_cls", return_value=NoSpecMetrics):
    prom = OffloadPromMetrics(
        vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config={})
        ),
        metric_types={Gauge: Gauge, Counter: Counter, Histogram: Histogram},
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["m", "0"]},
    )
prom.observe(stats.data)
text = generate_latest(REGISTRY).decode()
for reason, value in (("miss", "2.0"), ("promote_refused", "1.0")):
    sample = (f'vllm:kv_offload_wave_retry_total{{engine="0",model_name="m",'
              f'reason="{reason}"}} {value}')
    assert sample in text, sample
print("OK vllm:kv_offload_wave_retry_total{reason} exported")
