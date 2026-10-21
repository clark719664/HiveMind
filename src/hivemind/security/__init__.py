"""Project HiveMind — Repository, Pull Request, Runtime Swarm, Federated Learning, GPU Hardware & Red-Team Guardrail Security Package.

Provides:
1. Automated 5-Layer Pull Request & Repository Security Scanner (`pr_security_scanner.py`)
2. Phase S-2 Swarm GPU Training & Content-Addressed Data Store Runtime Hardening (`swarm_runtime_hardening.py`)
3. Phase S-3 Federated Learning & Byzantine-Resilient DiLoCo Hardening (`federated_learning_hardening.py`)
4. Phase S-4 Federated GPU Execution & Hardware Safety Circuit-Breaker (`gpu_execution_circuit_breaker.py`)
5. Phase S-5 25-Vector End-to-End Multi-Layer Attack Simulation & Guardrail Verification (`guardrail_attack_simulator.py`)
"""

from importlib import import_module
from typing import Any

# Resolved on first access so importing one submodule does not import the others:
# eager imports here created a core.diloco_outer <-> boinc.coordinator_api cycle and
# made the torch-free PR scanner require torch.
_EXPORTS = {
    "CircuitBreakerStatus": "gpu_execution_circuit_breaker",
    "DPGradientSanitizerConfig": "federated_learning_hardening",
    "FederatedDeltaAuditReport": "federated_learning_hardening",
    "FederatedLearningSecurityGuard": "federated_learning_hardening",
    "GPUExecutionCircuitBreaker": "gpu_execution_circuit_breaker",
    "GPUHardwareTelemetry": "gpu_execution_circuit_breaker",
    "PRSecurityAuditResult": "pr_security_scanner",
    "PRSecurityScanner": "pr_security_scanner",
    "SecurityFinding": "pr_security_scanner",
    "SecuritySeverity": "pr_security_scanner",
    "SimulatedAttackScenarioResult": "guardrail_attack_simulator",
    "SwarmGuardrailAttackSimulator": "guardrail_attack_simulator",
    "SwarmHardeningVerdict": "swarm_runtime_hardening",
    "SwarmSecurityViolationError": "swarm_runtime_hardening",
    "SwarmTrainingAndStoreGuard": "swarm_runtime_hardening",
    "compute_shannon_entropy": "swarm_runtime_hardening",
    "query_live_gpu_hardware_telemetry": "gpu_execution_circuit_breaker",
    "run_cli_security_scan": "pr_security_scanner",
    "run_federated_hardening_self_test": "federated_learning_hardening",
    "run_gpu_circuit_breaker_self_test": "gpu_execution_circuit_breaker",
    "run_guardrail_attack_simulation": "guardrail_attack_simulator",
    "run_swarm_hardening_self_test": "swarm_runtime_hardening",
}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value

__all__ = [
    "CircuitBreakerStatus",
    "DPGradientSanitizerConfig",
    "FederatedDeltaAuditReport",
    "FederatedLearningSecurityGuard",
    "GPUExecutionCircuitBreaker",
    "GPUHardwareTelemetry",
    "PRSecurityAuditResult",
    "PRSecurityScanner",
    "SecurityFinding",
    "SecuritySeverity",
    "SimulatedAttackScenarioResult",
    "SwarmGuardrailAttackSimulator",
    "SwarmHardeningVerdict",
    "SwarmSecurityViolationError",
    "SwarmTrainingAndStoreGuard",
    "compute_shannon_entropy",
    "query_live_gpu_hardware_telemetry",
    "run_cli_security_scan",
    "run_federated_hardening_self_test",
    "run_gpu_circuit_breaker_self_test",
    "run_guardrail_attack_simulation",
    "run_swarm_hardening_self_test",
]
