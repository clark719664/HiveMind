"""Offline ABI/credit safety tests; these do not certify a live Steam session."""
import ctypes
from types import SimpleNamespace
import pytest
from hivemind.client.steamworks import SteamworksSDKBridge

class NativeFunction:
    def __init__(self, result):
        self.result = result
        self.calls = []
    def __call__(self, *args):
        self.calls.append(args)
        return self.result

@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setattr(SteamworksSDKBridge, '_try_load_native_steam_dll', lambda self, directory: {})
    return SteamworksSDKBridge()

@pytest.mark.parametrize('success', [False, True])
def test_presence_passes_native_receiver_and_honors_result(bridge, success):
    getter = NativeFunction(0x123456789)
    setter = NativeFunction(success)
    bridge.native_dll_handle = SimpleNamespace(SteamAPI_SteamFriends_v017=getter,
        SteamAPI_ISteamFriends_SetRichPresence=setter)
    bridge.native_steam_api_initialized = True
    result = bridge.set_rich_presence('1', mission_name='Testing')
    assert setter.calls[0][0] == 0x123456789
    assert len(setter.calls[0]) == 3
    assert getter.restype is ctypes.c_void_p
    assert setter.argtypes == [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    assert result.native_dll_broadcast is success

def test_presence_does_not_call_uninitialized_native(bridge):
    setter = NativeFunction(True)
    bridge.native_dll_handle = SimpleNamespace(SteamAPI_ISteamFriends_SetRichPresence=setter)
    assert not bridge.set_rich_presence('1').native_dll_broadcast
    assert not setter.calls

def test_workunit_credit_is_idempotent_and_binds_recipient(bridge):
    first = bridge.record_validated_workunit('1', 'Tester', 'work1', 10.0, 2.0)
    second = bridge.record_validated_workunit('1', 'Tester', 'work1', 10.0, 2.0)
    assert first is second
    assert first.workunits_completed == 1
    assert first.api_credits_balance == 2.0
    with pytest.raises(ValueError):
        bridge.record_validated_workunit('2', 'Other', 'work1', 10.0, 2.0)
    with pytest.raises(ValueError):
        bridge.record_validated_workunit('1', 'Tester', 'work1', 10.0, 3.0)

@pytest.mark.parametrize('bad', [-1, float('nan'), float('inf')])
def test_invalid_credits_cannot_mutate_ledger(bridge, bad):
    with pytest.raises(ValueError):
        bridge.record_validated_workunit('1', 'Tester', 'work1', 10.0, bad)
    assert not bridge.leaderboard
