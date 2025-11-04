# Dockerfile
# Headless AsyncIO TFTP server container
# Image: docker.io/rjsears/tftpgui:<tag>

# Use a slim Python image (no Tk in container; this image is for headless runs)
FROM python:3.12-slim

# Ensure predictable runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

# System deps useful for networking & logging
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates tzdata iproute2 netcat-traditional libcap2-bin; \
    rm -rf /var/lib/apt/lists/*

# App directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy your app (adjust filename if different)
# Ensure tftpgui_enterprise.py is in the repo root alongside this Dockerfile
COPY tftpgui_enterprise.py /app/tftpgui_enterprise.py

# Non-root user (matches our docs & permissions guidance)
RUN set -eux; \
    adduser --disabled-password --gecos "" --uid 10001 tftpuser; \
    chown -R 10001:10001 /app

USER tftpuser

# Expose the container's internal ports (docs/hint only)
# - 1069 is the internal TFTP listener (we map host 69 -> 1069 in compose)
# - 50000-50100 is the recommended fixed data port range for TFTP transfers
# - 8080 is the web UI port
EXPOSE 1069/udp
EXPOSE 50000-50100/udp
EXPOSE 8080/tcp

# Default command: headless server using config mounted at /app/.tftpgui_config.json
# (GUI is disabled in container; run GUI on desktop outside Docker.)
CMD ["python", "/app/tftpgui_enterprise.py", "--headless", "-c", "/app/.tftpgui_config.json"]
