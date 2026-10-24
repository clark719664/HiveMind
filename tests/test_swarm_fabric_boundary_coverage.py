"""Device-adapter unit contracts; mocks do not claim to verify GPU kernel execution."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, sentinel

import pytest
import torch

from hivemind.swarm import swarm_fabric as fabric


@pytest.mark.parametrize(
    "device_name, hip, mode",
    [("cpu", None, "cpu_copy"), ("mps", None, "mps_unified"), ("cuda:0", None, "cuda_stream"), ("cuda:0", "6.2", "hip_stream")],
)
def test_streamer_selects_staging_backend_without_allocating_on_accelerator(monkeypatch, device_name, hip, mode):
    stream = Mock()
    monkeypatch.setattr(torch.cuda, "Stream", stream)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    device = torch.device(device_name)
    streamer = fabric.PinnedRAMExpertDMAStreamer(device=device)
    assert streamer.staging_mode == mode
    assert streamer.current_pinned_bytes == 0
    if device.type == "cuda":
        stream.assert_called_once_with(device=device)
        assert streamer._dma_stream is stream.return_value
    else:
        stream.assert_not_called()


def test_dma_adapter_requests_nonblocking_transfer_and_waits_on_stream(monkeypatch):
    stream = Mock(name="dma_stream")
    current_stream = Mock()
    start, end = Mock(), Mock()
    start.elapsed_time.return_value = 3.25
    event = Mock(side_effect=[start, end])
    context = Mock(return_value=nullcontext())
    monkeypatch.setattr(torch.cuda, "Stream", Mock(return_value=stream))
    monkeypatch.setattr(torch.cuda, "Event", event)
    monkeypatch.setattr(torch.cuda, "stream", context)
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=current_stream))
    device = torch.device("cuda:0")
    streamer = fabric.PinnedRAMExpertDMAStreamer(device=device)
    weight = Mock(name="staged_weight")
    weight.to.return_value = sentinel.resident_weight
    streamer.pinned_host_cache["expert"] = {"weight": weight}
    streamer.lru_order.append("expert")

    result = streamer.async_dma_prefetch_to_vram("expert", layer_idx=2)

    weight.to.assert_called_once_with(device=device, non_blocking=True)
    context.assert_called_once_with(stream)
    start.record.assert_called_once_with(stream)
    end.record.assert_called_once_with(stream)
    end.synchronize.assert_called_once_with()
    current_stream.wait_stream.assert_called_once_with(stream)
    assert result["raw_dma_transfer_ms"] == 3.25
    assert result["exposed_swap_latency_ms"] == 0.75
    assert result["tensors_streamed"] == 1
    assert result["slot"] == "slot_A"
    assert streamer.active_vram_cache["expert"]["weight"] is sentinel.resident_weight
    assert streamer.double_buffer_slots["slot_A"]["expert"]["weight"] is sentinel.resident_weight


def test_fp8_adapter_falls_back_to_cpu_math_after_kernel_dispatch_failure(monkeypatch):
    values = torch.tensor([[1.0, -1.0]])

    class DeviceInput:
        """Only the device boundary is simulated; fallback uses real CPU tensors."""

        shape = values.shape
        is_cuda = True

        def numel(self):
            return values.numel()

        def element_size(self):
            return values.element_size()

        def detach(self):
            return values

    kernel = MagicMock()
    kernel.__getitem__.return_value.side_effect = RuntimeError("fixture accelerator dispatch failure")
    monkeypatch.setattr(fabric, "_HAS_TRITON", True)
    monkeypatch.setattr(fabric, "triton", SimpleNamespace(next_power_of_2=lambda n: 2), raising=False)
    monkeypatch.setattr(fabric, "_triton_fp8_quant_kernel", kernel, raising=False)

    packet = fabric.compress_hidden_states_fp8(DeviceInput())

    kernel.__getitem__.assert_called_once_with((1,))
    assert kernel.__getitem__.return_value.call_count == 1
    restored = packet.fp8_data.to(torch.float32) * packet.absmax_scale.to(torch.float32)
    torch.testing.assert_close(restored, values, atol=0.01, rtol=0)
    assert packet.original_shape == (1, 2)
    assert packet.original_bytes == 8
    assert packet.compressed_bytes == 4


def test_skip_attention_prefetch_requires_a_name_for_every_expert():
    streamer = fabric.PinnedRAMExpertDMAStreamer(device=torch.device("cpu"))
    with pytest.raises(ValueError, match="expert_names length"):
        streamer.prefetch_and_route_skip_attention_car(
            torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(2), torch.zeros(2, 2), ["only-one"]
        )
