"""Project HiveMind — Repository, Pull Request & Runtime Swarm Security Package.

Provides:
1. Automated 5-Layer Pull Request & Repository Security Scanner (`pr_security_scanner.py`)
2. Phase S-2 Swarm GPU Training & Content-Addressed Data Store Runtime Hardening (`swarm_runtime_hardening.py`)
"""

from hivemind.security.pr_security_scanner import (
    PRSecurityAuditResult,
    PRSecurityScanner,
    SecurityFinding,
    SecuritySeverity,
    run_cli_security_scan,
)
from hivemind.security.swarm_runtime_hardening import (
    SwarmHardeningVerdict,
    SwarmSecurityViolationError,
    SwarmTrainingAndStoreGuard,
    compute_shannon_entropy,
    run_swarm_hardening_self_test,
)

__all__ = [
    "PRSecurityAuditResult",
    "PRSecurityScanner",
    "SecurityFinding",
    "SecuritySeverity",
    "SwarmHardeningVerdict",
    "SwarmSecurityViolationError",
    "SwarmTrainingAndStoreGuard",
    "compute_shannon_entropy",
    "run_cli_security_scan",
    "run_swarm_hardening_self_test",
]
