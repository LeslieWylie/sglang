"""Placement on parts where host and device share one memory pool (GB10, Jetson)."""

import torch

from sglang.multimodal_gen.runtime.managers.memory_managers import (
    auto_residency,
    host_memory_budget,
    layerwise_offload,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.auto_residency import (
    PAGEABLE_H2D_COST_MULTIPLIER,
    RankResidencyReport,
    ResidencyTarget,
    plan_auto_residency,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    LAYERWISE_OFFLOAD,
)
from sglang.multimodal_gen.runtime.platforms import current_platform

GIB_BYTES = 1024**3


def _share_pool(monkeypatch, shared: bool) -> None:
    monkeypatch.setattr(
        type(current_platform),
        "device_shares_host_memory",
        classmethod(lambda cls: shared),
    )


def test_shared_pool_hosting_keeps_every_mapped_layer_mapped():
    hosting = layerwise_offload._shared_pool_hosting(
        {0: 10, 1: 10, 2: 5}, {0: 10, 1: 4, 2: 0}
    )
    assert hosting == {0: "mapped", 1: "mapped", 2: "pageable"}


def test_no_pin_capacity_when_the_device_reads_host_pages(monkeypatch):
    _share_pool(monkeypatch, True)
    assert host_memory_budget.HostPinBudget.for_local_worker(2).spendable_bytes == 0
    _share_pool(monkeypatch, False)
    monkeypatch.setattr(
        host_memory_budget, "host_memory_available_bytes", lambda: 64 * GIB_BYTES
    )
    assert host_memory_budget.HostPinBudget.for_local_worker(1).spendable_bytes > 0


def test_pageable_penalty_disappears_on_a_shared_pool(monkeypatch):
    _share_pool(monkeypatch, True)
    assert auto_residency._pageable_h2d_cost_multiplier() == 1
    _share_pool(monkeypatch, False)
    assert (
        auto_residency._pageable_h2d_cost_multiplier() == PAGEABLE_H2D_COST_MULTIPLIER
    )


def test_cold_advice_is_harmless_on_an_ordinary_tensor():
    layerwise_offload._advise_mapped_source_cold(torch.zeros(4096, dtype=torch.uint8))
    layerwise_offload._advise_mapped_source_cold(torch.zeros(0))


def _pin_frontier() -> tuple[ResidencyTarget, ResidencyTarget]:
    pageable = ResidencyTarget(
        component_name="transformer",
        residency_mode=LAYERWISE_OFFLOAD,
        target_residency_mode=LAYERWISE_OFFLOAD,
        target_resident_weight_bytes=5 * GIB_BYTES,
        h2d_bytes_per_request=10 * GIB_BYTES,
        target_layerwise_resident_layers=(5,),
        target_layerwise_pinned_layers=((),),
        target_device_weight_bytes=5 * GIB_BYTES,
        current_placement=True,
    )
    pinned = ResidencyTarget(
        component_name="transformer",
        residency_mode=LAYERWISE_OFFLOAD,
        target_residency_mode=LAYERWISE_OFFLOAD,
        target_resident_weight_bytes=5 * GIB_BYTES,
        h2d_bytes_per_request=11 * GIB_BYTES,
        target_layerwise_resident_layers=(5,),
        target_layerwise_pinned_layers=((0,),),
        pinned_host_delta_bytes=2 * GIB_BYTES,
        target_device_weight_bytes=5 * GIB_BYTES,
        target_pinned_host_bytes=2 * GIB_BYTES,
    )
    return pageable, pinned


def _report(*, shared: bool, candidates) -> RankResidencyReport:
    pageable, pinned = candidates
    # 100 GiB budget, 89 GiB measured peak, 10 GiB reserve: one GiB of device
    # headroom, while the host side would allow ten GiB of pins.
    return RankResidencyReport(
        rank=0,
        budget_bytes=100 * GIB_BYTES,
        estimated_peak_bytes=89 * GIB_BYTES,
        host_pin_capacity_bytes=10 * GIB_BYTES,
        host_shares_device_pool=shared,
        candidates=[pageable, pinned],
        estimated_request_duration_ns=1_000_000_000,
        candidate_latency_savings_ns={
            pageable.option_key(): 100_000_000,
            pinned.option_key(): 110_000_000,
        },
    )


def test_pins_are_charged_to_device_phases_on_a_shared_pool():
    pageable, pinned = _pin_frontier()
    discrete = plan_auto_residency(
        reports=[_report(shared=False, candidates=(pageable, pinned))]
    )
    assert discrete.changes == [pinned]
    shared = plan_auto_residency(
        reports=[_report(shared=True, candidates=(pageable, pinned))]
    )
    assert shared.changes == []


class _FakeManager:
    def __init__(self, mapped: dict[int, int], *, resident_layers: int = 0):
        self.num_layers = 4
        self.residency_policy = "leading"
        self.resident_layers = resident_layers
        self.enabled = True
        self._mapped = mapped

    def mapped_layer_bytes(self) -> dict[int, int]:
        return self._mapped


def test_streamed_mapped_bytes_count_only_streamed_layers_with_a_mapping():
    manager = _FakeManager({0: 10, 1: 10, 2: 0, 3: 10})
    streamed = auto_residency._layerwise_streamed_mapped_bytes(
        managers=[manager], resident_layers=(0,), residency_policies=("leading",)
    )
    assert streamed == 30
    # A layer read once per request is a bounded re-read, not a pool claim.
    assert (
        auto_residency._layerwise_streamed_mapped_bytes(
            managers=[manager],
            resident_layers=(0,),
            residency_policies=("leading",),
            repeated=False,
        )
        == 0
    )
    assert (
        auto_residency._layerwise_streamed_mapped_bytes(
            managers=[manager],
            resident_layers=(0,),
            residency_policies=("leading",),
            layer_uses=((20, 20, 1, 1),),
        )
        == 20
    )
    # Leading residency keeps the first layers on the device; their mapped
    # bytes stop being page cache the stream depends on.
    assert (
        auto_residency._layerwise_streamed_mapped_bytes(
            managers=[manager], resident_layers=(2,), residency_policies=("leading",)
        )
        == 10
    )
    assert (
        auto_residency._layerwise_streamed_mapped_bytes(
            managers=[manager], resident_layers=(4,), residency_policies=("leading",)
        )
        == 0
    )


def test_estimated_manager_has_no_mapping():
    manager = auto_residency._EstimatedLayerwiseManager(
        layers_attr_str="blocks",
        layer_bytes={0: GIB_BYTES, 1: GIB_BYTES},
        prefetch_size=1,
        residency_policy="leading",
        pin_cpu_memory=False,
    )
    assert manager.mapped_layer_bytes() == {}


def test_no_resident_seed_on_a_shared_pool(monkeypatch):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.managers.memory_managers import initial_residency

    _share_pool(monkeypatch, True)
    monkeypatch.setattr(
        initial_residency, "auto_residency_static_skip_reason", lambda args: None
    )
    monkeypatch.setattr(type(current_platform), "is_cuda", lambda self: True)
    seeded: list[str] = []
    server_args = SimpleNamespace(
        use_fsdp_inference=False,
        set_auto_residency_mode=lambda name, mode: seeded.append(name),
    )
    monkeypatch.setattr(
        initial_residency.current_platform,
        "get_available_gpu_memory",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("seed must not size the pool")
        ),
    )
    initial_residency.maybe_seed_initial_residency(server_args, inventory=[])
    assert seeded == []


def test_mapped_layers_carry_no_transfer_work_on_a_shared_pool(monkeypatch):
    class _Manager(_FakeManager):
        def layer_weight_bytes(self):
            return {0: 10, 1: 10, 2: 10, 3: 10}

    manager = _Manager({0: 10, 1: 10, 2: 4, 3: 0})
    _share_pool(monkeypatch, False)
    discrete = auto_residency._layerwise_transfer_work_bytes(
        managers=[manager],
        resident_layers=(0,),
        pinned_layers=((),),
        uses_per_streamed_layer=20,
    )
    _share_pool(monkeypatch, True)
    shared = auto_residency._layerwise_transfer_work_bytes(
        managers=[manager],
        resident_layers=(0,),
        pinned_layers=((),),
        uses_per_streamed_layer=20,
    )
    # Every layer streams 20 times; only the bytes off the mapping still cost.
    assert discrete == 20 * PAGEABLE_H2D_COST_MULTIPLIER * 40
    assert shared == 20 * (6 + 10)


def test_shared_pool_keeps_only_the_measured_and_streamed_layouts():
    kept = auto_residency._shared_pool_resident_targets(
        [(0, 0), (0, 10), (0, 25), (2, 50), (0, 50)], current_resident_layers=(0, 25)
    )
    assert kept == [(0, 0), (0, 25)]
    assert auto_residency._shared_pool_resident_targets([(0, 50), (0, 0)], None) == [
        (0, 0)
    ]


def test_populate_mapped_source_is_harmless_on_anonymous_memory():
    # Anonymous pages are already present; the advice must not raise or copy.
    layerwise_offload.populate_mapped_source([torch.zeros(1 << 16, dtype=torch.uint8)])
    layerwise_offload.populate_mapped_source([torch.zeros(0)])


def test_mapped_stream_cost_applies_when_the_cache_cannot_hold_the_cycle(monkeypatch):
    class _Manager(_FakeManager):
        def layer_weight_bytes(self):
            return {0: 10, 1: 10, 2: 10, 3: 10}

    manager = _Manager({0: 10, 1: 10, 2: 10, 3: 10})
    _share_pool(monkeypatch, True)
    free = auto_residency._layerwise_transfer_work_bytes(
        managers=[manager],
        resident_layers=(0,),
        pinned_layers=((),),
        uses_per_streamed_layer=19,
    )
    priced = auto_residency._layerwise_transfer_work_bytes(
        managers=[manager],
        resident_layers=(0,),
        pinned_layers=((),),
        uses_per_streamed_layer=19,
        mapped_stream_cost_multiplier=auto_residency.DISK_MISS_COST_MULTIPLIER,
    )
    assert free == 0
    assert priced == 19 * 24 * 40
    # Resident layers stop paying the per-step disk price.
    half = auto_residency._layerwise_transfer_work_bytes(
        managers=[manager],
        resident_layers=(2,),
        pinned_layers=((),),
        uses_per_streamed_layer=19,
        mapped_stream_cost_multiplier=auto_residency.DISK_MISS_COST_MULTIPLIER,
    )
    assert half == 19 * 24 * 20 + 1 * 24 * 20
