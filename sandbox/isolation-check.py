"""Trusted preflight, before any project source is executed."""
import os
import pathlib
import re

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

require(os.getuid() == 10001 and os.getgid() == 10001, "Unexpected container identity")
status = pathlib.Path("/proc/self/status").read_text()
require(re.search(r"^CapEff:\s+0+$", status, re.M), "Capabilities must be empty")
require(re.search(r"^NoNewPrivs:\s+1$", status, re.M), "Privilege escalation must be disabled")
require(sorted(p.name for p in pathlib.Path("/sys/class/net").iterdir()) == ["lo"], "Container must have loopback only")
mounts = {fields[4]: fields[5].split(",") for line in pathlib.Path("/proc/self/mountinfo").read_text().splitlines() if (fields := line.split())}
require("ro" in mounts.get("/", []), "Root filesystem must be read-only")
require("ro" in mounts.get("/input", []), "Project input must be read-only")
require(not pathlib.Path("/var/run/docker.sock").exists(), "Docker socket must not be exposed")
require(not any(re.search(r"TOKEN|SECRET|PASSWORD|PRIVATE_KEY|^GITHUB_|^ACTIONS_", key, re.I) for key in os.environ), "Credential or Actions environment exposed")
print("ISOLATION VERIFIED: non-root; no network; no capabilities; read-only root/input; no credential environment", flush=True)
