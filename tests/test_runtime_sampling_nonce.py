"""CPU-only regressions for runtime sampling, STUN inputs and replay state.

The statistical tests exercise the public decoder with causal categorical models.
They do not claim WAN, GPU performance, or a floating-point implementation proof.
"""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hivemind.swarm import inference_ring as inference
from hivemind.swarm import quic_transport as quic
from hivemind.swarm.nat_traversal import DEFAULT_STUN_SERVERS, NatMapping


@pytest.fixture(scope="module", autouse=True)
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, list(DEFAULT_STUN_SERVERS)),
        ({"stun_servers": DEFAULT_STUN_SERVERS}, list(DEFAULT_STUN_SERVERS)),
        ({"stun_servers": list(DEFAULT_STUN_SERVERS)}, list(DEFAULT_STUN_SERVERS)),
        ({"stun_servers": "stun.example.test:3478"}, ["stun.example.test:3478"]),
        ({"stun_servers": ("127.0.0.1", 3478)}, [("127.0.0.1", 3478)]),
        ({"stun_servers": (("127.0.0.1", 3478), ("127.0.0.2", 3478))},
         [("127.0.0.1", 3478), ("127.0.0.2", 3478)]),
        ({"stun_servers": [("127.0.0.1", 3478), "stun.example.test:3478"]},
         [("127.0.0.1", 3478), "stun.example.test:3478"]),
    ],
)
def test_stun_single_address_and_server_sequences(monkeypatch, kwargs, expected):
    class Socket:
        closed = False

        def close(self):
            self.closed = True

    sock = Socket()
    monkeypatch.setattr(quic, "udp_socket", lambda address: sock)
    calls = []

    def classify(socket, servers, timeout):
        assert socket is sock
        calls.append(servers)
        return NatMapping("no_nat", ("127.0.0.1", 4000), [("127.0.0.1", 4000)],
                          [("127.0.0.1", 3478)], [0.1])

    monkeypatch.setattr(quic, "classify_nat_mapping", classify)
    endpoint = quic.AsyncQUICGeoRingTransport().discover_stun_turn_endpoint("peer", **kwargs)
    assert calls == [expected]
    assert endpoint["reflexive_port"] == 4000
    assert sock.closed


@pytest.mark.parametrize("servers", [[], ()])
def test_empty_stun_sequence_fails_before_opening_socket(monkeypatch, servers):
    monkeypatch.setattr(quic, "udp_socket", lambda _: pytest.fail("socket should not be opened"))
    with pytest.raises(ValueError, match="at least one STUN"):
        quic.AsyncQUICGeoRingTransport().discover_stun_turn_endpoint("peer", stun_servers=servers)


@pytest.mark.parametrize("first_wrapper", [False, True])
def test_nonce_apis_share_one_monotonic_strike_register(first_wrapper):
    transport = quic.AsyncQUICGeoRingTransport()
    ticket = transport.issue_0rtt_ticket("peer")
    detailed = lambda nonce: transport.verify_0rtt_ticket_anti_replay("peer", nonce)[0]
    simple = lambda nonce: transport.verify_and_advance_0rtt_nonce("peer", nonce)
    first, second = (simple, detailed) if first_wrapper else (detailed, simple)
    # The old split state accepted [100, 1, 100] when alternating APIs.
    assert [first(100), second(1), first(100)] == [True, False, False]
    assert second(101)
    assert ticket.last_accepted_nonce == transport.anti_replay_max_nonce["peer"] == 101
    assert ticket.resumed_count == 2


def test_ticket_rotation_does_not_rewind_nonce(monkeypatch):
    monkeypatch.setattr(quic.time, "time", lambda: 1000.0)
    transport = quic.AsyncQUICGeoRingTransport()
    old = transport.issue_0rtt_ticket("peer")
    assert transport.verify_and_advance_0rtt_nonce("peer", 100)
    monkeypatch.setattr(quic.time, "time", lambda: 2000.0)
    fresh = transport.issue_0rtt_ticket("peer")
    assert fresh is not old
    assert fresh.issued_at == 2000.0
    assert fresh.last_accepted_nonce == 100
    assert not transport.verify_and_advance_0rtt_nonce("peer", 1)
    assert not transport.verify_0rtt_ticket_anti_replay("peer", 100)[0]
    assert transport.verify_0rtt_ticket_anti_replay("peer", 101)[0]


@pytest.mark.parametrize("now, accepted, reason", [
    (999.999, False, "future_0rtt_session_ticket"),
    (1000.0, True, "0rtt_nonce_accepted"),
    (1600.0, True, "0rtt_nonce_accepted"),
    (1600.001, False, "expired_0rtt_session_ticket"),
    (float("nan"), False, "invalid_0rtt_ticket_time"),
    (float("inf"), False, "invalid_0rtt_ticket_time"),
])
def test_nonce_helpers_enforce_same_ticket_time_interval(monkeypatch, now, accepted, reason):
    transport = quic.AsyncQUICGeoRingTransport()
    ticket = transport.issue_0rtt_ticket("peer")
    ticket.issued_at = 1000.0
    monkeypatch.setattr(quic.time, "time", lambda: now)
    assert transport.verify_0rtt_ticket_anti_replay("peer", 100) == (accepted, reason)
    # A distinct nonce avoids changing this from a time check into a replay check.
    assert transport.verify_and_advance_0rtt_nonce("peer", 101) is accepted
    assert ticket.last_accepted_nonce == (101 if accepted else 0)


@pytest.mark.parametrize("nonce", [-1, 0, 1.5, float("nan"), float("inf"), True, "1", None])
def test_invalid_nonce_cannot_poison_or_rewind_register(nonce):
    transport = quic.AsyncQUICGeoRingTransport()
    ticket = transport.issue_0rtt_ticket("peer")
    assert not transport.verify_and_advance_0rtt_nonce("peer", nonce)
    assert ticket.last_accepted_nonce == transport.anti_replay_max_nonce["peer"] == 0
    assert transport.verify_and_advance_0rtt_nonce("peer", 1)


@pytest.mark.parametrize("age", [-1.0, float("nan"), float("inf")])
def test_invalid_ticket_lifetime_fails_closed(age):
    transport = quic.AsyncQUICGeoRingTransport()
    transport.issue_0rtt_ticket("peer")
    assert transport.verify_0rtt_ticket_anti_replay("peer", 1, max_age_sec=age) == (
        False, "invalid_0rtt_ticket_time"
    )


def test_missing_ticket_is_not_implicitly_minted():
    transport = quic.AsyncQUICGeoRingTransport()
    assert not transport.verify_and_advance_0rtt_nonce("unknown", 1)
    assert transport.verify_0rtt_ticket_anti_replay("unknown", 1) == (
        False, "missing_0rtt_session_ticket"
    )
    assert transport.zero_rtt_tickets == {}


def test_nonce_update_is_atomic_across_helpers():
    transport = quic.AsyncQUICGeoRingTransport()
    ticket = transport.issue_0rtt_ticket("peer")

    def check(i):
        if i % 2:
            return transport.verify_and_advance_0rtt_nonce("peer", 100)
        return transport.verify_0rtt_ticket_anti_replay("peer", 100)[0]

    with ThreadPoolExecutor(max_workers=16) as pool:
        accepted = list(pool.map(check, range(128)))
    assert sum(accepted) == 1
    assert ticket.last_accepted_nonce == transport.anti_replay_max_nonce["peer"] == 100
    assert ticket.resumed_count == 1


class CategoricalModel(nn.Module):
    """Causal first-order model, with one next-token distribution per input ID."""

    def __init__(self, probabilities, record=False):
        super().__init__()
        table = torch.tensor(probabilities, dtype=torch.float64)
        if table.ndim == 1:
            table = table.repeat(table.numel(), 1)
        self.register_buffer("logits_table", table.log())
        self.inputs = []
        self.record = record

    def forward(self, input_ids):
        if self.record:
            self.inputs.append(input_ids.detach().clone())
        return SimpleNamespace(logits=self.logits_table[input_ids])

    def reset_routing_telemetry(self):
        pass

    def get_expert_utilization(self):
        return {}


def decode(target, draft, count=1, gamma=1, seed=42, generator=None, prompt=None):
    return inference.RegionalDHTSwarmRing().run_speculative_inference(
        target, draft, torch.tensor([0]) if prompt is None else prompt,
        max_new_tokens=count, draft_gamma=gamma, simulated_network_rtt_ms=0,
        generator=generator if generator is not None else torch.Generator().manual_seed(seed),
    )


@pytest.mark.parametrize("p, q, proposal, correction, expected_residual", [
    # The former 0.65 support test accepted proposal 1, despite p/q = 0.5.
    ([0.55, 0.45], [0.1, 0.9], 1, 0, [1.0, 0.0]),
    # Matching the target argmax still does not guarantee p/q acceptance.
    ([0.6, 0.4], [0.9, 0.1], 0, 1, [0.0, 1.0]),
    # The correction must sample the complete residual, not the target argmax.
    ([0.2, 0.3, 0.5], [0.8, 0.1, 0.1], 0, 1, [0.0, 1 / 3, 2 / 3]),
])
def test_exact_rejection_counterexamples(monkeypatch, p, q, proposal, correction, expected_residual):
    samples = []
    choices = iter([proposal, correction])

    def multinomial(probabilities, num_samples, generator):
        samples.append(probabilities.clone())
        return torch.tensor([next(choices)])

    monkeypatch.setattr(torch, "multinomial", multinomial)
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.9, dtype=torch.float64))
    result = decode(CategoricalModel(p), CategoricalModel(q))
    assert result.generated_tokens == [correction]
    assert result.mean_acceptance_rate == 0.0
    assert torch.allclose(samples[0], torch.tensor(q, dtype=torch.float64))
    assert torch.allclose(samples[1], torch.tensor(expected_residual, dtype=torch.float64))


def test_seeded_decoder_reproduces_tokens_without_consuming_global_rng():
    target, draft = CategoricalModel([0.2, 0.3, 0.5]), CategoricalModel([0.7, 0.2, 0.1])
    state = torch.random.get_rng_state()
    first = decode(target, draft, count=80, gamma=5, seed=912)
    second = decode(target, draft, count=80, gamma=5, seed=912)
    assert first.generated_tokens == second.generated_tokens
    assert first.swarm_verification_passes == second.swarm_verification_passes
    assert first.mean_acceptance_rate == second.mean_acceptance_rate
    assert torch.equal(torch.random.get_rng_state(), state)


def test_identical_models_sample_distribution_and_accept_all_proposals():
    result = decode(CategoricalModel([0.25, 0.75]), CategoricalModel([0.25, 0.75]), count=800, gamma=5)
    assert result.mean_acceptance_rate == 1.0
    assert result.swarm_verification_passes == 134
    assert result.generated_tokens.count(0) / 800 == pytest.approx(0.25, abs=0.05)
    assert result.generated_tokens.count(1) / 800 == pytest.approx(0.75, abs=0.05)


@pytest.mark.parametrize("gamma", [1, 4])
def test_causal_target_alignment_rejection_resync_and_bonus(gamma):
    target = CategoricalModel([[0, 1, 0], [0, 0, 1], [1, 0, 0]], record=True)
    draft = CategoricalModel([1, 0, 0])
    result = decode(target, draft, count=23, gamma=gamma)
    assert result.generated_tokens == ([1, 2, 0] * 8)[:23]
    # Initial failures must resynchronize on the correction, discarding the
    # unused zero-valued proposals before making the next target call.
    assert target.inputs[0][0].tolist() == [0] + [0] * gamma
    assert target.inputs[1][0].tolist() == [0, 1] + [0] * gamma
    assert target.inputs[2][0].tolist() == [0, 1, 2] + [0] * gamma
    assert result.mean_acceptance_rate < 1.0


def test_speculative_one_token_marginal_matches_target():
    target, draft = CategoricalModel([0.05, 0.3, 0.65]), CategoricalModel([0.85, 0.1, 0.05])
    generator = torch.Generator().manual_seed(3742)
    counts = [0, 0, 0]
    for _ in range(4000):
        token = decode(target, draft, generator=generator).generated_tokens[0]
        counts[token] += 1
    assert [n / 4000 for n in counts] == pytest.approx([0.05, 0.3, 0.65], abs=0.025)


@pytest.mark.parametrize("gamma", [1, 2])
def test_speculative_joint_distribution_matches_causal_target(gamma):
    target = CategoricalModel([[0.3, 0.7], [0.8, 0.2]])
    draft = CategoricalModel([[0.9, 0.1], [0.1, 0.9]])
    generator = torch.Generator().manual_seed(1909)
    counts = [0, 0, 0, 0]
    for _ in range(4000):
        a, b = decode(target, draft, count=2, gamma=gamma, generator=generator).generated_tokens
        counts[2 * a + b] += 1
    assert [n / 4000 for n in counts] == pytest.approx([0.09, 0.21, 0.56, 0.14], abs=0.025)


@pytest.mark.parametrize("kwargs", [
    {"max_new_tokens": -1}, {"max_new_tokens": 1.5}, {"draft_gamma": 0},
    {"draft_gamma": -1}, {"draft_gamma": 1.5}, {"simulated_network_rtt_ms": -1},
    {"simulated_network_rtt_ms": float("nan")},
])
def test_decoder_rejects_invalid_limits_without_hanging(kwargs):
    with pytest.raises(ValueError):
        inference.RegionalDHTSwarmRing().run_speculative_inference(
            CategoricalModel([0.5, 0.5]), CategoricalModel([0.5, 0.5]), torch.tensor([0]), **kwargs
        )


@pytest.mark.parametrize("prompt", [torch.empty(0), torch.empty(1, 0), torch.ones(2, 3), torch.ones(1, 2, 3)])
def test_decoder_rejects_empty_or_batched_prompts(prompt):
    with pytest.raises(ValueError, match="one nonempty prompt"):
        decode(CategoricalModel([0.5, 0.5]), CategoricalModel([0.5, 0.5]), prompt=prompt)


def test_zero_token_budget_avoids_model_calls_and_preserves_training_flags():
    model = CategoricalModel([0.5, 0.5], record=True)
    model.train()
    result = decode(model, model, count=0)
    assert result.generated_tokens == [] and result.swarm_verification_passes == 0
    assert model.inputs == [] and model.training


def test_model_state_is_restored_on_forward_error():
    class BrokenModel(CategoricalModel):
        def forward(self, input_ids):
            raise RuntimeError("intentional test failure")

    target, draft = CategoricalModel([0.5, 0.5]), BrokenModel([0.5, 0.5])
    target.eval()
    draft.train()
    with pytest.raises(RuntimeError, match="intentional test failure"):
        decode(target, draft)
    assert not target.training and draft.training


def test_nonfinite_model_probabilities_are_rejected():
    with pytest.raises(ValueError, match="finite categorical probabilities"):
        decode(CategoricalModel([0.5, 0.5]), CategoricalModel([0, 0]))


@pytest.mark.parametrize("fail", [False, True])
def test_shared_moe_routing_modes_are_scoped_and_restored(fail):
    class SharedMoE(inference.UpcycledSparseMoEModel):
        def __init__(self):
            nn.Module.__init__(self)
            self.register_buffer("logits", torch.tensor([0.0, 0.0], dtype=torch.float64))
            self.block = SimpleNamespace(top_k=3, router=SimpleNamespace(top_k=4))
            self.top_k = 2
            self.modes = []

        def get_moe_blocks(self):
            return [self.block]

        def reset_routing_telemetry(self):
            pass

        def get_expert_utilization(self):
            return {}

        def forward(self, input_ids):
            self.modes.append((self.block.top_k, self.block.router.top_k))
            if fail:
                raise RuntimeError("intentional MoE failure")
            return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, 2))

    model = SharedMoE()
    if fail:
        with pytest.raises(RuntimeError, match="intentional MoE failure"):
            decode(model, model, count=2)
        assert model.modes == [(1, 1)]
    else:
        result = decode(model, model, count=2)
        assert len(result.generated_tokens) == 2
        assert model.modes == [(1, 1), (2, 2)]
    assert (model.block.top_k, model.block.router.top_k) == (3, 4)
    assert model.training


@pytest.mark.parametrize("websocket", [False, True])
def test_hidden_state_frame_roundtrip_and_strict_lengths(websocket):
    from hivemind.swarm.swarm_fabric import compress_hidden_states_fp8

    transport = quic.AsyncQUICGeoRingTransport()
    packet = compress_hidden_states_fp8(torch.randn(2, 3, 8, dtype=torch.float16))
    frame = transport.serialize_packet_0rtt("peer", packet, use_websocket_fallback=websocket)
    restored, protocol = transport.deserialize_packet_0rtt(frame)
    assert restored.original_shape == packet.original_shape
    assert torch.equal(restored.fp8_data.view(torch.uint8), packet.fp8_data.view(torch.uint8))
    assert torch.equal(restored.absmax_scale, packet.absmax_scale)
    assert protocol == ("libp2p_websocket" if websocket else "quic")
    for malformed in [frame[:10], b"BAD!" + frame[4:], frame[:20], frame[:-1], frame + b"tail"]:
        with pytest.raises(ValueError):
            transport.deserialize_packet_0rtt(malformed)


@pytest.mark.parametrize("shape, fp8_size, scale_size", [
    ((), 1, 2), ((0, 8), 0, 0), ((4, 8), 31, 8), ((4, 8), 32, 7),
    ((2 ** 32 - 1, 2 ** 32 - 1), 1, 2),
])
def test_hidden_state_frame_rejects_inconsistent_shape_before_allocation(shape, fp8_size, scale_size):
    import struct

    header = struct.pack(f"!4s12sB{len(shape)}III", quic.MAGIC_QUIC_0RTT, b"x" * 12,
                         len(shape), *shape, fp8_size, scale_size)
    with pytest.raises(ValueError):
        quic.AsyncQUICGeoRingTransport().deserialize_packet_0rtt(header + b"\0" * 128)


class FakeSocket:
    closed = False

    def close(self):
        self.closed = True


def test_open_peer_socket_closes_socket_when_stun_fails(monkeypatch):
    sock = FakeSocket()
    monkeypatch.setattr(quic, "udp_socket", lambda _: sock)

    def fail(*args, **kwargs):
        raise quic.StunError("test STUN failure")

    monkeypatch.setattr(quic, "classify_nat_mapping", fail)
    with pytest.raises(quic.StunError, match="test STUN failure"):
        quic.AsyncQUICGeoRingTransport()._open_peer_socket("peer", "example.test:3478")
    assert sock.closed


@pytest.mark.parametrize("method", ["perform_udp_hole_punch", "execute_udp_hole_punch"])
def test_peer_pair_initialization_failure_closes_first_socket(monkeypatch, method):
    sock = FakeSocket()
    transport = quic.AsyncQUICGeoRingTransport()

    def open_peer(peer_id, stun_server):
        if peer_id == "second":
            raise quic.StunError("second peer failure")
        return sock, ("127.0.0.1", 4000), {"nat_type": "unknown"}

    monkeypatch.setattr(transport, "_open_peer_socket", open_peer)
    with pytest.raises(quic.StunError, match="second peer failure"):
        getattr(transport, method)("first", "second")
    assert sock.closed


def test_relay_pair_initialization_failure_closes_first_client(monkeypatch):
    first = FakeSocket()
    count = 0

    def client(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("second relay failure")
        return first

    monkeypatch.setattr(quic, "TurnClient", client)
    with pytest.raises(OSError, match="second relay failure"):
        quic.AsyncQUICGeoRingTransport._relay_pair("example.test", "user", "test-only")
    assert first.closed


def test_failed_turn_allocation_closes_unregistered_client(monkeypatch):
    class FailedClient(FakeSocket):
        def allocate(self):
            raise quic.StunError("allocation failure")

    client, sock = FailedClient(), FakeSocket()
    monkeypatch.setattr(quic, "TurnClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(quic, "udp_socket", lambda _: sock)
    monkeypatch.setattr(quic, "classify_nat_mapping", lambda *args: NatMapping(
        "no_nat", ("127.0.0.1", 4000), [("127.0.0.1", 4000)], [("127.0.0.1", 3478)], [0.1]
    ))
    transport = quic.AsyncQUICGeoRingTransport()
    with pytest.raises(quic.StunError, match="allocation failure"):
        transport.discover_stun_turn_endpoint("peer", stun_servers="stun.example.test",
                                             turn_server="turn.example.test", turn_username="user")
    assert client.closed and sock.closed
    assert transport.turn_clients == {}


def streamer(budget=64):
    from hivemind.swarm.swarm_fabric import PinnedRAMExpertDMAStreamer

    return PinnedRAMExpertDMAStreamer(device=torch.device("cpu"), max_pinned_ram_bytes=budget)


def weights(count=4):
    return {"weight": torch.ones(count, dtype=torch.float16)}


def test_dma_restage_accounts_replaced_bytes_without_unnecessary_eviction():
    dma = streamer(24)
    dma.stage_expert_in_pinned_ram("a", weights())
    dma.stage_expert_in_pinned_ram("b", weights())
    dma.stage_expert_in_pinned_ram("a", weights(6))
    assert dma.current_pinned_bytes == 20
    assert set(dma.pinned_host_cache) == {"a", "b"}
    assert dma.lru_order == ["b", "a"]
    dma.stage_expert_in_pinned_ram("a", weights(2))
    assert dma.current_pinned_bytes == 12
    assert set(dma.pinned_host_cache) == {"a", "b"}


def test_dma_oversized_expert_fails_without_evicting_existing_cache():
    dma = streamer(8)
    dma.stage_expert_in_pinned_ram("a", weights())
    for name in ("a", "oversized"):
        with pytest.raises(ValueError, match="exceeds"):
            dma.stage_expert_in_pinned_ram(name, weights(5))
        assert set(dma.pinned_host_cache) == {"a"}
        assert dma.current_pinned_bytes == 8


def test_dma_host_lru_and_snapshot_ownership():
    dma = streamer(16)
    source = weights()
    dma.stage_expert_in_pinned_ram("a", source)
    source["weight"].zero_()
    assert torch.all(dma.pinned_host_cache["a"]["weight"] == 1)
    dma.stage_expert_in_pinned_ram("b", weights())
    dma.async_dma_prefetch_to_vram("a")  # a is now the most recently used.
    dma.stage_expert_in_pinned_ram("c", weights())
    assert dma.lru_order == ["a", "c"]
    assert dma.current_pinned_bytes == 16
    assert set(dma.pinned_host_cache) == {"a", "c"}


def test_dma_standalone_transfers_keep_only_two_live_slots():
    dma = streamer(80)
    for index in range(10):
        name = f"expert-{index}"
        dma.stage_expert_in_pinned_ram(name, weights())
        info = dma.async_dma_prefetch_to_vram(name)
        assert info["overlap_is_simulated"] and info["transfer_synchronized"]
        assert len(dma.active_vram_cache) <= 2
        assert all(len(slot) <= 1 for slot in dma.double_buffer_slots.values())
        expected = {f"expert-{i}" for i in range(max(0, index - 1), index + 1)}
        assert set(dma.active_vram_cache) == expected


def test_dma_restage_invalidates_old_device_copy():
    dma = streamer()
    dma.stage_expert_in_pinned_ram("a", weights())
    dma.async_dma_prefetch_to_vram("a")
    assert "a" in dma.active_vram_cache
    dma.stage_expert_in_pinned_ram("a", {"weight": torch.zeros(4)})
    assert "a" not in dma.active_vram_cache
    assert all("a" not in slot for slot in dma.double_buffer_slots.values())
    dma.async_dma_prefetch_to_vram("a")
    assert torch.all(dma.active_vram_cache["a"]["weight"] == 0)


def test_dma_lookahead_groups_experts_and_releases_reused_slots():
    dma = streamer()
    for name in "abcdef":
        dma.stage_expert_in_pinned_ram(name, weights())

    def router(names):
        return lambda _: {"selected_macro_cluster": "test", "selected_micro_experts": dict.fromkeys(names, 1.0)}

    dma.prefetch_lookahead_layer(router("ab"), torch.ones(1), layer_idx=0)
    dma.prefetch_lookahead_layer(router("cd"), torch.ones(1), layer_idx=1)
    report = dma.prefetch_lookahead_layer(router(["e", "f", "missing"]), torch.ones(1), layer_idx=2)
    assert set(dma.double_buffer_slots["slot_A"]) == {"c", "d"}
    assert set(dma.double_buffer_slots["slot_B"]) == {"e", "f"}
    assert set(dma.active_vram_cache) == {"c", "d", "e", "f"}
    assert report["prefetched_experts"] == ["e", "f"]
    assert report["missing_experts"] == ["missing"]


@pytest.mark.parametrize("budget", [-1, 1.5, True])
def test_dma_rejects_invalid_budget(budget):
    with pytest.raises(ValueError, match="nonnegative integer"):
        streamer(budget)
