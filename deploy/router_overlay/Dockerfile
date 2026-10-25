# Overlay on the live HiveMind OpenRouter image (built from source that is not on main yet): adds the activity window on
# consumer bids (PR #182) and the points-only display wording. Build context: this directory plus points_wording.py,
# staged by deploy/router_overlay/build.sh.
FROM us-central1-docker.pkg.dev/gen-lang-client-0625573011/cloud-run-source-deploy/hivemind-openrouter@sha256:d3f9251328a71458d5370943a7142e4af9a70ada9c49326b6c0a1f32f3e9c074
USER root
RUN mv /app/server.py /app/server_orig.py
COPY points_wording.py /app/src/hivemind/router/points_wording.py
COPY server.py /app/server.py
USER app
