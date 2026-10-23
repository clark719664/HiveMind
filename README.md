# Isolated HiveMind tests

This branch contains a reviewed test harness, not the HiveMind source. It fetches one exact public source revision on a fresh GitHub-hosted Linux VM. Project code runs only inside a network-disabled, unprivileged container with a read-only root and no personal folders or credentials. The container and VM are discarded after the job.

`revision.txt` selects the full source commit. `source-repository.txt` accepts only `xaviercallens/HiveMind` or `clark719664/HiveMind`. Pushing this branch starts a run. The harness installs reviewed dependencies before fetching source; it never runs project installation hooks or project workflows.

The preflight verifies non-root identity, disabled networking/capabilities, read-only mounts, and absent credential environment before running tests. The coverage gate is 95%. Coverage and test results measure behavior, not security, and logs remain untrusted data.

On Clark's PC, use the trusted launcher outside all project checkouts:

```powershell
& C:\Users\lil_c\HiveMind-Quarantine\Run-Remote.ps1
```

It verifies the external reviewed-file manifest, selects upstream main by default, and pushes only this harness branch on Clark's fork. It can also accept `-Revision FULL_COMMIT_SHA -Repository clark719664/HiveMind` for a pushed contribution. No HiveMind execution on Windows or ordinary WSL is permitted. Isolation reduces exposure; it is not a promise of absolute immunity. The local VirtualBox guest remains disabled and unprovisioned.
