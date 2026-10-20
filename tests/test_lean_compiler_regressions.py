"""Proof certification must require real, complete compiler evidence."""


from __future__ import annotations


import copy


import subprocess


from pathlib import Path


import pytest


from paper import autoevolve_peer_review_pipeline as proof


@pytest.fixture
def lean_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    names = [f"Example.checked_{index}" for index in range(14)]
    source = "namespace Example\n" + "\n".join(
        f"theorem checked_{index} : True := by trivial" for index in range(14)
    ) + "\nend Example\n"
    path = tmp_path / "HiveMindSpec.lean"
    path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(proof, "LEAN4_SPEC_PATH", path)
    return names


def compiler_output(names: list[str], axioms: str | None = None) -> str:
    suffix = f"depends on axioms: [{axioms}]" if axioms else "does not depend on any axioms"
    return "\n".join(f"'{name}' {suffix}" for name in names)


def mock_compiler(monkeypatch: pytest.MonkeyPatch, output: str, returncode: int = 0) -> None:
    monkeypatch.setattr(proof, "_find_lean_compiler", lambda: "/test/bin/lean")

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command == ["/test/bin/lean", "--stdin"]
        assert kwargs["cwd"] == proof.LEAN4_SPEC_PATH.parent
        assert kwargs["timeout"] == 120
        assert "#print axioms Example.checked_13" in kwargs["input"]
        return subprocess.CompletedProcess(command, returncode, stdout=output, stderr="")

    monkeypatch.setattr(proof.subprocess, "run", run)


def test_missing_compiler_is_blocked_not_machine_checked(lean_spec, monkeypatch) -> None:
    monkeypatch.setattr(proof, "_find_lean_compiler", lambda: None)
    result = proof.verify_lean4_specification()
    assert result["theorem_count"] == 14
    assert result["compiler_binary"] is None
    assert result["compiler_return_code"] is None
    assert result["compiler_executed"] is False
    assert result["verification_status"] == "blocked"
    assert result["tier_a_machine_checked"] is False
    with pytest.raises(proof.FormalVerificationError):
        proof.require_machine_checked(result)


def test_lean_bin_override_takes_precedence(monkeypatch) -> None:
    monkeypatch.setenv("LEAN_BIN", "/configured/bin/lean")
    monkeypatch.setattr(proof.shutil, "which", lambda name: name)
    assert proof._find_lean_compiler() == "/configured/bin/lean"


def test_missing_lean_bin_falls_back_to_path(monkeypatch) -> None:
    monkeypatch.setenv("LEAN_BIN", "/missing/lean")
    monkeypatch.setattr(proof.shutil, "which", lambda name: "/path/bin/lean" if name == "lean" else None)
    assert proof._find_lean_compiler() == "/path/bin/lean"


def test_audit_paths_remain_portable(lean_spec, monkeypatch) -> None:
    mock_compiler(monkeypatch, compiler_output(lean_spec))
    result = proof.verify_lean4_specification()
    assert result["spec_file"] == "HiveMindSpec.lean"
    assert result["compiler_binary"] == "lean"
    assert proof._rel(proof.PROJECT_ROOT / "paper" / "PEER_REVIEW_REPORT.json") == "paper/PEER_REVIEW_REPORT.json"


@pytest.mark.parametrize("failure", ["nonzero", "empty", "incomplete", "sorryAx", "customAxiom", "Lean.ofReduceBool"])
def test_bad_compiler_evidence_fails_closed(lean_spec, monkeypatch, failure: str) -> None:
    names = lean_spec[:-1] if failure == "incomplete" else lean_spec
    output = "" if failure == "empty" else compiler_output(names)
    if failure in {"sorryAx", "customAxiom", "Lean.ofReduceBool"}:
        output = compiler_output(names, f"propext,\n {failure}")
    mock_compiler(monkeypatch, output, returncode=1 if failure == "nonzero" else 0)
    result = proof.verify_lean4_specification()
    assert result["tier_a_machine_checked"] is False
    assert result["verification_status"] == "failed"
    with pytest.raises(proof.FormalVerificationError):
        proof.require_machine_checked(result)


@pytest.mark.parametrize("error", [OSError("not executable"), subprocess.TimeoutExpired("lean", 120)])
def test_compiler_launch_failure_is_blocked(lean_spec, monkeypatch, error) -> None:
    monkeypatch.setattr(proof, "_find_lean_compiler", lambda: "/test/bin/lean")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(proof.subprocess, "run", fail)
    result = proof.verify_lean4_specification()
    assert result["tier_a_machine_checked"] is False
    assert result["verification_status"] == "blocked"
    assert result["compiler_return_code"] is None


@pytest.mark.parametrize("declaration", ["example : True := by sorry", "example : True := by admit", "axiom unsound : False"])
def test_source_placeholders_or_axioms_fail_even_with_clean_output(lean_spec, monkeypatch, declaration) -> None:
    with proof.LEAN4_SPEC_PATH.open("a", encoding="utf-8") as source:
        source.write(declaration + "\n")
    mock_compiler(monkeypatch, compiler_output(lean_spec))
    assert proof.verify_lean4_specification()["tier_a_machine_checked"] is False


def test_complete_compiler_evidence_passes(lean_spec, monkeypatch) -> None:
    mock_compiler(monkeypatch, compiler_output(lean_spec, "propext,\n Classical.choice,\n Quot.sound"))
    result = proof.verify_lean4_specification()
    assert result["tier_a_machine_checked"] is True
    assert len(result["axioms_by_theorem"]) == len(lean_spec)
    assert result["missing_axiom_checks"] == []
    proof.require_machine_checked(result)


def test_inventory_ignores_nested_comments_and_strings(lean_spec, monkeypatch) -> None:
    with proof.LEAN4_SPEC_PATH.open("a", encoding="utf-8") as source:
        source.write('/- outer /- inner -/ theorem ignored : sorry axiom fake -/\n')
        source.write('#eval "theorem fake := sorry -- axiom anything"\n')
    mock_compiler(monkeypatch, compiler_output(lean_spec))
    result = proof.verify_lean4_specification()
    assert result["theorem_count"] == 14
    assert result["sorry_count"] == result["user_axiom_count"] == 0
    assert result["tier_a_machine_checked"] is True


def test_unsupported_theorem_syntax_cannot_silently_escape_inventory(lean_spec, monkeypatch) -> None:
    with proof.LEAN4_SPEC_PATH.open("a", encoding="utf-8") as source:
        source.write("private theorem hidden : True := by trivial\n")
    mock_compiler(monkeypatch, compiler_output(lean_spec))
    assert proof.verify_lean4_specification()["tier_a_machine_checked"] is False


@pytest.mark.parametrize("value", [1, "true", "false", {}, None])
def test_truthy_values_are_not_machine_checked_booleans(lean_spec, monkeypatch, value) -> None:
    mock_compiler(monkeypatch, compiler_output(lean_spec))
    result = proof.verify_lean4_specification()
    result["tier_a_machine_checked"] = value
    with pytest.raises(proof.FormalVerificationError):
        proof.require_machine_checked(result)


def test_peer_review_does_not_publish_acceptance_without_proofs(lean_spec, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(proof, "_find_lean_compiler", lambda: None)
    review_path, manifest_path = tmp_path / "review.json", tmp_path / "manifest.json"
    monkeypatch.setattr(proof, "PEER_REVIEW_JSON", review_path)
    monkeypatch.setattr(proof, "AUTOEVOLVE_MANIFEST_JSON", manifest_path)
    with pytest.raises(proof.FormalVerificationError):
        proof.run_autoevolve_peer_review()
    assert not review_path.exists()
    assert not manifest_path.exists()


def test_real_repository_proofs_compile_when_lean_is_available() -> None:
    if proof._find_lean_compiler() is None:
        pytest.skip("Lean unavailable locally; CI separately requires the pinned compiler")
    result = proof.verify_lean4_specification()
    proof.require_machine_checked(result)
    assert result["theorem_count"] >= 32
    assert len(result["axioms_by_theorem"]) == result["theorem_count"]

