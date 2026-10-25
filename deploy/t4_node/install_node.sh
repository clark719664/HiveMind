#!/bin/bash
# Install the T4 node's batch layer + boot tooling under /opt/hivemind/t4 and register the systemd units. Run as root:
#   sudo bash deploy/t4_node/install_node.sh
# Idempotent. It copies a snapshot of the code (the git worktree this runs from may be deleted later), installs and enables the
# node units, and DOES NOT start anything: starting is a separate step (bin/startup.sh) so the owner controls when services move.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OPT=/opt/hivemind/t4
USER_NAME=callensxavier_gmail_com

install -d -o "$USER_NAME" -g "$USER_NAME" "$OPT" "$OPT/bin" "$OPT/scripts" "$OPT/src"
rsync -a --delete --exclude '__pycache__' "$REPO/src/hivemind/" "$OPT/src/hivemind/"
for s in t4_governor.py t4_monitor.py t4_workgen.py; do install -m 0755 -o "$USER_NAME" -g "$USER_NAME" "$REPO/scripts/$s" "$OPT/scripts/$s"; done
for b in "$REPO"/deploy/t4_node/node/bin/*.sh; do install -m 0755 "$b" "$OPT/bin/$(basename "$b")"; done
chown -R "$USER_NAME:$USER_NAME" "$OPT"

for u in "$REPO"/deploy/t4_node/node/*.service; do install -m 0644 "$u" "/etc/systemd/system/$(basename "$u")"; done
# the older bee/coder producers' units (mirroring the running processes); installed and enabled so a reboot brings them back
for u in hivemind-live-bee-llama hivemind-live-shield-bee hivemind-live-agent-bee hivemind-live-shield-coder hivemind-live-agent-coder; do
  install -m 0644 "$REPO/deploy/t4_node/live/$u.service" "/etc/systemd/system/$u.service"
done
systemctl daemon-reload
for u in hivemind-boinc-db hivemind-boinc-apache hivemind-boinc-daemons hivemind-boinc-client hivemind-t4-governor hivemind-t4-workgen hivemind-t4-monitor \
         hivemind-live-bee-llama hivemind-live-shield-bee hivemind-live-agent-bee hivemind-live-shield-coder hivemind-live-agent-coder; do
  systemctl enable "$u.service" >/dev/null 2>&1 && echo "enabled $u" || echo "FAILED to enable $u"
done
echo "installed to $OPT; nothing was started"
