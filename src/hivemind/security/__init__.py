"""Project HiveMind — Repository & Pull Request Security Scanner Package.

Provides automated 5-layer defense against:
1. Payment Solicitation, Financial Extortion & Crypto Wallet Injection
2. "Code Boubou" Attacks (Hollow Bounty-Farming PRs, Fake Issue Closures, Binary/Cache Pollution)
3. Dangerous Python AST Sinks (eval, exec, pickle, shell=True, os.system, obfuscated droppers)
4. Credential Exfiltration & Hardcoded Secrets (HF_TOKEN, GITHUB_TOKEN, private keys, webhooks)
5. Trojan Source Bidirectional Unicode Attacks (CVE-2021-42574) & Protected Governance Tampering
"""

from hivemind.security.pr_security_scanner import (
    PRSecurityAuditResult,
    PRSecurityScanner,
    SecurityFinding,
    SecuritySeverity,
    run_cli_security_scan,
)

__all__ = [
    "PRSecurityAuditResult",
    "PRSecurityScanner",
    "SecurityFinding",
    "SecuritySeverity",
    "run_cli_security_scan",
]
