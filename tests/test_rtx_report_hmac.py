"""Local report tags must not embed or reuse a deployment credential."""

import ast
import hashlib
import hmac
import inspect

import hivemind.game.rtx_steam_tool_deployer as deployer


def test_report_hmac_requests_a_fresh_256_bit_key_for_every_report(monkeypatch):
    requested_sizes = []
    keys = iter((bytes(range(32)), bytes(reversed(range(32)))))

    def next_key(size):
        requested_sizes.append(size)
        return next(keys)

    monkeypatch.setattr(deployer.secrets, "token_bytes", next_key)
    payload = "local simulation report"
    first = deployer._ephemeral_report_hmac(payload)
    second = deployer._ephemeral_report_hmac(payload)

    assert requested_sizes == [32, 32]
    assert first == hmac.new(bytes(range(32)), payload.encode(), hashlib.sha256).hexdigest()
    assert second == hmac.new(bytes(reversed(range(32))), payload.encode(), hashlib.sha256).hexdigest()
    assert first != second


def test_identical_reports_do_not_reuse_the_real_ephemeral_key():
    first = deployer._ephemeral_report_hmac("same report")
    second = deployer._ephemeral_report_hmac("same report")
    assert len(first) == len(second) == 64
    assert first != second


def test_stress_report_uses_ephemeral_helper_without_embedded_hmac_keys():
    tree = ast.parse(inspect.getsource(deployer))
    fixed_key_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "hmac"
        and node.func.attr == "new"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    assert not fixed_key_calls
    suite = ast.parse(inspect.getsource(deployer.run_intense_rtx_stress_suite))
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_ephemeral_report_hmac"
        for node in ast.walk(suite)
    )
