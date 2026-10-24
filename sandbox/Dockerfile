FROM python:3.11-slim-bookworm@sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl zstd git libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
RUN curl --fail --location --proto '=https' --tlsv1.2 \
      https://github.com/leanprover/lean4/releases/download/v4.33.0-rc2/lean-4.33.0-rc2-linux.tar.zst -o /tmp/lean.tar.zst \
    && echo '48010c7d6264dc992574e9b0e09ece1f9c648e8f5106066b2cdb6834a6a60a6c  /tmp/lean.tar.zst' | sha256sum -c - \
    && mkdir /opt/lean && tar --zstd -xf /tmp/lean.tar.zst -C /opt/lean --strip-components=1 \
    && rm /tmp/lean.tar.zst
# Pinned official Node binary for upstream's embedded JavaScript syntax checks.
RUN curl --fail --location --proto '=https' --tlsv1.2 \
      https://nodejs.org/dist/v22.23.4/node-v22.23.4-linux-x64.tar.gz -o /tmp/node.tar.gz \
    && echo 'b8492284c76a54baf58d61668dd0e41ee0fc98adabf9503d30f05d08f55b4406  /tmp/node.tar.gz' | sha256sum -c - \
    && tar -xzf /tmp/node.tar.gz -C /usr/local/bin --strip-components=2 node-v22.23.4-linux-x64/bin/node \
    && rm /tmp/node.tar.gz \
    && node --version
RUN python -m pip install --no-cache-dir --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu'
COPY requirements.txt /opt/guard/requirements.txt
RUN python -m pip install --no-cache-dir --only-binary=:all: -r /opt/guard/requirements.txt \
    && python -m pip check
COPY isolation-check.py /opt/guard/isolation-check.py
COPY run-tests.sh /opt/guard/run-tests.sh
RUN chmod 0555 /opt/guard/run-tests.sh && mkdir /input /work
ENV PATH="/opt/lean/bin:${PATH}" PYTHONDONTWRITEBYTECODE=1 PIP_NO_INDEX=1
USER 10001:10001
WORKDIR /work
ENTRYPOINT ["/bin/sh", "/opt/guard/run-tests.sh"]
