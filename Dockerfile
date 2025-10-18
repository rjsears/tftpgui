# Dockerfile
# Headless AsyncIO TFTP server container
# Image: docker.io/rjsears/tftpgui:<tag>

FROM python:3.12-slim

# Metadata
LABEL org.opencontainers.image.title="tftpgui" \
      org.opencontainers.image.description="Headless TFTP server with progress & logging (GUI available outside container)" \
      org.opencontainers.image.source="https://github.com/rjsears/tftpgui" \
      org.opencontainers.image.licenses="MIT"

# System deps (headless) + grant python low-port bind capability
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates tzdata iproute2 netcat-traditional libcap2-bin; \
    for py in /usr/local/bin/python /usr/local/bin/python3; do \
        if [ -x "$py" ]; then setcap 'cap_net_bind_service=+ep' "$py" || true; fi; \
    done; \
    rm -rf /var/lib/apt/lists/*

# App dirs
WORKDIR /app
RUN mkdir -p /app /data /logs

# Non-root user (use CAP_NET_BIND_SERVICE to bind low UDP ports)
RUN useradd -u 10001 -r -s /usr/sbin/nologin tftpuser \
    && chown -R tftpuser:tftpuser /app /data /logs

# Copy code
COPY tftpgui_enterprise.py /app/tftpgui_enterprise.py
# Provide a default container config path
COPY container.tftpgui_config.json /app/.tftpgui_config.json

# Expose TFTP control port and a small UDP ephemeral range
# NOTE: If you use host networking, this is not strictly needed.
EXPOSE 69/udp
EXPOSE 50000-50100/udp

USER tftpuser

# Healthcheck: confirm process is up; optional UDP port check is fragile in containers
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD pgrep -f "tftpgui_enterprise.py" >/dev/null || exit 1

# Default: run headless with the container config
CMD ["python", "/app/tftpgui_enterprise.py", "-c", "/app/.tftpgui_config.json"]
