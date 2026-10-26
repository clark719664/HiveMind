#!/bin/bash
# Install the T4 node's batch layer + boot tooling under /opt/hivemind/t4 and register the systemd units. Run as root:
#   sudo bash deploy/t4_node/install_node.sh
# Idempotent. It copies a snapshot of the code (the git worktree this runs from may be deleted later), installs and enables the
# node units, and DOES NOT start anything: starting is a separate step (bin/startup.sh) so the owner controls when services move.
#
# "Only via repo" (owner, 2026-10-26): the CONTROL PLANE (governor, monitor, work-generator scripts, helper shell scripts and every unit file) is deployed from
# COMMITTED git content (HEAD), never from the working tree, so an uncommitted edit cannot reach the live node through this installer; a skipped uncommitted
# change is reported. The pipeline sources under src/ are still copied from the working tree (documented exception: the owner asked to leave a contributor's
# uncommitted edits there for later); the installer lists them. A hash manifest of what was installed lets the monitor flag later hand edits (CODE_DRIFT).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OPT=/opt/hivemind/t4
USER_NAME=callensxavier_gmail_com
GIT=(git -C "$REPO" -c "safe.directory=$REPO")

committed() {  # print the committed content of a repo-relative path; warn if the working copy differs from it
  local rel="$1"
  if ! "${GIT[@]}" diff --quiet HEAD -- "$rel" 2>/dev/null; then
    echo "WARNING: $rel has uncommitted changes; deploying the COMMITTED version (git HEAD). Commit and review the change to deploy it." >&2
  fi
  "${GIT[@]}" show "HEAD:$rel"
}

install -d -o "$USER_NAME" -g "$USER_NAME" "$OPT" "$OPT/bin" "$OPT/scripts" "$OPT/src"
rsync -a --delete --exclude '__pycache__' "$REPO/src/hivemind/" "$OPT/src/hivemind/"
if [ -n "$("${GIT[@]}" status --porcelain -- src 2>/dev/null)" ]; then
  echo "NOTE: src/ is copied from the working tree and has uncommitted changes (deployed as they are):" >&2
  "${GIT[@]}" status --porcelain -- src >&2
fi
for s in t4_governor.py t4_monitor.py t4_workgen.py; do
  committed "scripts/$s" > "$OPT/scripts/$s"
  chmod 0755 "$OPT/scripts/$s"
done
for b in "$REPO"/deploy/t4_node/node/bin/*.sh; do
  committed "deploy/t4_node/node/bin/$(basename "$b")" > "$OPT/bin/$(basename "$b")"
  chmod 0755 "$OPT/bin/$(basename "$b")"
done
chown -R "$USER_NAME:$USER_NAME" "$OPT"

for u in "$REPO"/deploy/t4_node/node/*.service; do
  committed "deploy/t4_node/node/$(basename "$u")" > "/etc/systemd/system/$(basename "$u")"
  chmod 0644 "/etc/systemd/system/$(basename "$u")"
done
# the older bee/coder producers' units (mirroring the running processes); installed and enabled so a reboot brings them back
for u in hivemind-live-bee-llama hivemind-live-shield-bee hivemind-live-agent-bee hivemind-live-shield-coder hivemind-live-agent-coder; do
  committed "deploy/t4_node/live/$u.service" > "/etc/systemd/system/$u.service"
  chmod 0644 "/etc/systemd/system/$u.service"
done
# Governance: record (1) the arguments the governor template defines, (2) the hash of every control-plane file installed, so the monitor can flag any
# difference (a changed argument, a drop-in, or an edited script), and warn about drop-in overrides, which silently beat the unit files written above.
"${GIT[@]}" show "HEAD:deploy/t4_node/node/hivemind-t4-governor.service" | grep -m1 '^ExecStart=' | sed -E 's/.*t4_governor\.py //' > "$OPT/expected_governor_args.txt"
( cd "$OPT" && sha256sum scripts/*.py bin/*.sh ) > "$OPT/expected_code_hashes.txt"
chown "$USER_NAME:$USER_NAME" "$OPT/expected_governor_args.txt" "$OPT/expected_code_hashes.txt"
overrides="$(ls -d /etc/systemd/system/hivemind-*.service.d 2>/dev/null || true)"
if [ -n "$overrides" ]; then
  echo "WARNING: systemd drop-in overrides exist and take precedence over the unit files this installer writes:"
  echo "$overrides"
fi
systemctl daemon-reload
for u in hivemind-boinc-db hivemind-boinc-apache hivemind-boinc-daemons hivemind-boinc-client hivemind-t4-governor hivemind-t4-workgen hivemind-t4-monitor \
         hivemind-live-bee-llama hivemind-live-shield-bee hivemind-live-agent-bee hivemind-live-shield-coder hivemind-live-agent-coder; do
  systemctl enable "$u.service" >/dev/null 2>&1 && echo "enabled $u" || echo "FAILED to enable $u"
done
echo "installed to $OPT from committed content; nothing was started"
