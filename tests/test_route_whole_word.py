"""Prompt-level keyword routing: whole-word matching and abstention on no hit (regression cases from the 2026-10-24 probe)."""

import pytest

from hivemind.core.btm_moe import Season3DomainExpertRegistry, _keyword_hit
from hivemind.core.serving_experts import PYTHON_CODE_GWAYA, route_with_serving_experts

OFF_DOMAIN = [
    "Explain how the heart pumps blood",
    "What is the capital of France?",
    "Prove that the sum of two even numbers is even",
    "Is the search nation public maybe a bypass?",  # short keywords (arc, nat, pub, by) inside longer words
]


@pytest.mark.parametrize("prompt", OFF_DOMAIN)
def test_off_domain_prompt_abstains(prompt):
    assert Season3DomainExpertRegistry().route_domain_query(prompt) == ([], {})
    assert route_with_serving_experts(prompt) == ([], {})


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("Audit this Solidity ERC20 smart contract for reentrancy in msg.sender", "Solidity_Smart_Contracts"),
        ("Prove this Lean 4 theorem by induction and omega tactic", "Lean4_Theorem_Prover"),
        ("fn unsafe mut rust borrow", "Rust_Systems_Security"),
        ("Explain the patient clinical trial design", "Biomedical_Clinical_QA"),
    ],
)
def test_in_domain_prompt_still_routes_first(prompt, expected):
    top, weights = Season3DomainExpertRegistry().route_domain_query(prompt, top_k=2)
    assert top[0] == expected and len(weights) == 2 and abs(sum(weights.values()) - 1.0) < 1e-3


def test_python_prompt_reaches_the_serving_expert_only_with_whole_word_hits():
    top, _ = route_with_serving_experts("fix this def with a traceback in pandas")
    assert top[0] == PYTHON_CODE_GWAYA
    assert route_with_serving_experts("define a classification of subclasses and an important matter") == ([], {})


def test_keyword_hit_boundaries():
    assert _keyword_hit("def ", "write a def that")
    assert not _keyword_hit("def ", "define it")
    assert _keyword_hit(".py", "open main.py now")  # keyword starting with punctuation needs no leading boundary
    assert _keyword_hit("msg.sender", "check msg.sender here")
    assert not _keyword_hit("pub", "a public method")


def test_min_weight_drops_weak_secondary_but_keeps_the_best():
    reg = Season3DomainExpertRegistry()
    top, w = reg.route_domain_query("Explain the patient clinical trial design", top_k=2, min_weight=0.2)
    assert top == ["Biomedical_Clinical_QA"] and w == {"Biomedical_Clinical_QA": 1.0}
    top, _ = reg.route_domain_query("Explain the patient clinical trial design", top_k=2)
    assert len(top) == 2  # default keeps the existing two-expert contract
