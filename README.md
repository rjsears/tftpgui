<div align="center">

  ## TFTPGui – A Modern GUI & Headless TFTP Server in Python3 (v1.0.4)

  
![Last Commit](https://img.shields.io/github/last-commit/rjsears/tftpgui)
![Issues](https://img.shields.io/github/issues/rjsears/tftpgui)
![License](https://img.shields.io/badge/license-MIT-green)
![Contributors](https://img.shields.io/github/contributors/rjsears/tftpgui)
![Release](https://img.shields.io/github/v/release/rjsears/tftpgui)
[![Docker Hub](https://img.shields.io/docker/pulls/rjsears/tftpgui.svg)](https://hub.docker.com/r/rjsears/tftpgui)

</div>

## What is TFTPGui?

TFTPGui is a fully modernized TFTP server written in Python 3 with both a **graphical user interface** and a **headless (CLI) mode** for true flexibility. It was built as a replacement for outdated Python2/xinetd TFTP setups and provides real-time monitoring, logging, security controls, and an easy configuration system.

If you work with firmware, embedded devices, PXE booting, routers/switches, or lab environments, this tool gives you all the control and visibility that the old stuff is missing.

---
![TFTP GUI Screenshot](https://raw.githubusercontent.com/rjsears/tftpgui/main/images/tftpgui3.png)

---

## Features

- Runs in GUI or headless mode  
- Real-time transfer progress bars  
- JSON-based configuration  
- IP allowlist / denylist support  
- Enforced root directory scoping  
- Audit logging + log rotation  
- Writable subdirectory controls  
- CLI override for config file location  
- Built-in status banner in the GUI  
- MIT licensed  
- Docker image

---

## System Requirements

You only need a few basics:

- Python 3.10 or newer  
- `tkinter` installed (for GUI mode)  
- Linux, macOS, or WSL on Windows  
- Permissions to bind to TFTP ports (default 69)  
- Docker (optional, headless)

Example for Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install python3 python3-tk
````

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/rjsears/tftpgui.git
cd tftpgui
```

### 2. Optional: Use a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

No external Python modules are required — it uses only the standard library.

---

## Configuration

TFTPGui uses a JSON configuration file:

```
.tftpgui_config.json
```

The script checks for it in this order:

1. The directory where the script is run
2. Your home directory
3. A custom path specified with `--config` or `-c`

### Example Config

```json
{
  "host": "0.0.0.0",
  "port": 69,
  "root_dir": "/home/tftpd/tftp",

  "allow_write": true,
  "writable_subdirs": ["uploads", "staging"],
  "enforce_chroot": false,

  "filename_allowlist": [".bin", ".cfg", ".hex"],
  "allowlist_ips": ["127.0.0.1/32", "::1/128", "192.168.0.0/24", "10.200.50.0/24"],
  "denylist_ips": [],

  "timeout_sec": 3.0,
  "max_retries": 5,
  "log_level": "INFO",

  "log_file": "/home/tftpd/tftpgui/logs/tftpgui.log",
  "audit_log_file": "/home/tftpd/tftpgui/logs/audit.jsonl",
  "transfer_log_file": "/home/tftpd/tftpgui/logs/transfers.csv",

  "metrics_window_sec": 5,
  "ephemeral_ports": true,

  "log_rotation": "size",
  "log_max_bytes": 5000000,
  "log_backup_count": 5,
  "log_when": "midnight",
  "log_interval": 1
}

```

You can override the config file on launch:

```bash
python3 tftpgui_enterprise.py -c /path/to/custom.json
```

This is handy if you want to run both a docker headless version as well as the gui version on the same system (at different times).

---

## Running the Program

### GUI Mode

```bash
python3 tftpgui_enterprise.py
```

Or with a specific config:

```bash
python3 tftpgui_enterprise.py -c /path/config.json
```

If you need privileged ports (like port 69):

```bash
sudo python3 tftpgui_enterprise.py
```

### Headless Mode

```bash
python3 tftpgui_enterprise.py --headless
```

Logs and audit events will print to stdout and your configured log files.

---

## Docker Support

You don’t have to run TFTPGui directly on your host — it can also run inside a Docker container.  
The container is built to run in **headless mode** (no GUI), perfect for lab servers and appliances.

### Pull the image

```bash
docker pull rjsears/tftpgui:1.0.3
````

Or always grab the newest build:

```bash
docker pull rjsears/tftpgui:latest
```

---

### Quick run with host networking (Linux only, run as root in the container (quickest))

```bash
docker run --rm -it --network host --user 0:0 \
  -v /home/crypto/tftp:/data \
  -v /home/crypto/.tftpgui_config.json:/app/.tftpgui_config.json:ro \
  -v /home/crypto/tftpgui/logs:/logs \
  rjsears/tftpgui:1.0.3
```
### Quick run with host networking (Linux only, allow low prots for unprivileged (per-run sysctl))

```bash
docker run --rm -it --network host \
  --sysctl net.ipv4.ip_unprivileged_port_start=0 \
  -v /home/crypto/tftp:/data \
  -v /home/crypto/.tftpgui_config.json:/app/.tftpgui_config.json:ro \
  -v /home/crypto/tftpgui/logs:/logs \
  rjsears/tftpgui:1.0.3
```

These use your host’s network stack directly, so you don’t have to map UDP ports.

---

### Run with bridged networking (portable, recommended)

```bash
docker run --rm -it \
  -p 69:1069/udp \
  -p 50000-50100:50000-50100/udp \
  -v /home/crypto/tftp:/data \
  -v /home/crypto/container.tftpgui_config.json:/app/.tftpgui_config.json:ro \
  -v /home/crypto/tftpgui/logs:/logs \
  rjsears/tftpgui:1.0.3
```

This maps **UDP port 69** and an ephemeral range (`50000–50100`) that the server uses for data transfers.
Adjust the range in both your config and command if you need more concurrent sessions.

---

### docker-compose example (includes profiles for Bridged and Host networking).

```yaml
version: "3.9"

services:
  # -------------------------------------------------------------
  # Recommended: Bridged networking with mapped ports (portable)
  # -------------------------------------------------------------
  tftpgui:
    # Use the published image, or uncomment "build" to build locally
    image: rjsears/tftpgui:1.0.3
    # build:
    #   context: .
    #   dockerfile: Dockerfile

    container_name: tftpgui
    restart: unless-stopped

    # Map host 69 -> container 1069, and a fixed data port range 50000-50100
    ports:
      - "69:1069/udp"
      - "50000-50100:50000-50100/udp"

    # Volumes:
    # - Map your TFTP data root to /data
    # - Mount your Docker-specific config to /app/.tftpgui_config.json (read-only)
    # - Mount a logs directory to /logs so files persist on the host
    volumes:
      - ./data:/data
      - ./container.tftpgui_config.json:/app/.tftpgui_config.json:ro
      - ./logs:/logs

    # Command delegates to the default CMD in Dockerfile, but you can override here if desired
    command: ["--headless", "-c", "/app/.tftpgui_config.json"]

    # Helpful labels and resource hints (optional)
    labels:
      com.rjsears.tftpgui: "bridged"

  # -------------------------------------------------------------
  # Advanced: Host networking (binds 69/udp directly on the host)
  # Enable with:  docker compose --profile hostnet up -d tftpgui-host
  # -------------------------------------------------------------
  tftpgui-host:
    # Use the same image
    image: rjsears/tftpgui:1.0.3
    # build:
    #   context: .
    #   dockerfile: Dockerfile

    container_name: tftpgui-host
    restart: unless-stopped

    profiles: ["hostnet"]

    # Host networking means no port mappings; the app binds directly to host interfaces
    network_mode: host

    # Choose ONE of the following permission strategies (uncomment as needed):
    #
    # 1) Run as root inside container (simplest):
    # user: "0:0"
    #
    # 2) Allow non-root to bind low ports on this container:
    # sysctls:
    #   - net.ipv4.ip_unprivileged_port_start=0
    #
    # 3) If your image grants python cap_net_bind_service and your host allows it:
    # cap_add:
    #   - NET_BIND_SERVICE

    volumes:
      - ./data:/data
      - ./host.tftpgui_config.json:/app/.tftpgui_config.json:ro
      - ./logs:/logs

    # For host mode, your config should typically set "port": 69 and "root_dir": "/data"
    command: ["--headless", "-c", "/app/.tftpgui_config.json"]

    labels:
      com.rjsears.tftpgui: "hostnet"
```

### Example Directory Structure

```
tftpgui/
├─ Dockerfile
├─ docker-compose.yml
├─ tftpgui_enterprise.py
├─ container.tftpgui_config.json
├─ host.tftpgui_config.json
├─ data/                # your TFTP root (host)
│  ├─ uploads/
│  └─ staging/
└─ logs/                # audit, transfer, app logs (host)
```
---

### Container config file (Bridged)

Inside the container, the root directory is `/data` and logs are in `/logs`.
Here’s an example `container.tftpgui_config.json` you can mount:

```json
{
  "host": "0.0.0.0",
  "port": 1069,
  "transfer_port_min": 50000,
  "transfer_port_max": 50100,
  "root_dir": "/data",

  "allow_write": true,
  "writable_subdirs": ["uploads", "staging"],
  "enforce_chroot": false,

  "filename_allowlist": [".bin", ".cfg", ".hex"],
  "allowlist_ips": ["192.168.0.0/24", "10.200.50.0/24"],
  "denylist_ips": [],

  "timeout_sec": 3.0,
  "max_retries": 5,
  "log_level": "INFO",

  "log_file": "/logs/tftpgui.log",
  "audit_log_file": "/logs/audit.jsonl",
  "transfer_log_file": "/logs/transfers.csv",

  "metrics_window_sec": 5,
  "ephemeral_ports": true,

  "log_rotation": "size",
  "log_max_bytes": 5000000,
  "log_backup_count": 5,
  "log_when": "midnight",
  "log_interval": 1
}
```

### Container config file (Host)

Inside the container, the root directory is `/data` and logs are in `/logs`.
Here’s an example `host.tftpgui_config.json` you can mount:

```json
{
  "host": "0.0.0.0",
  "port": 69,
  "root_dir": "/data",

  "allow_write": true,
  "writable_subdirs": ["uploads", "staging"],
  "enforce_chroot": false,

  "filename_allowlist": [".bin", ".cfg", ".hex"],
  "allowlist_ips": ["127.0.0.1/32", "::1/128", "192.168.0.0/24", "10.200.50.0/24"],
  "denylist_ips": [],

  "timeout_sec": 3.0,
  "max_retries": 5,
  "log_level": "INFO",

  "log_file": "/logs/tftpgui.log",
  "audit_log_file": "/logs/audit.jsonl",
  "transfer_log_file": "/logs/transfers.csv",

  "metrics_window_sec": 5,
  "ephemeral_ports": true,

  "log_rotation": "size",
  "log_max_bytes": 5000000,
  "log_backup_count": 5,
  "log_when": "midnight",
  "log_interval": 1
}
```

## Create directories and set permissions

```bash
mkdir -p data/uploads data/staging logs
# If needed:
sudo chown -R 10001:10001 data logs
```


### How to run

## Bridged (recommended):
```bash
docker compose up -d tftpgui
```

## Host networking
```bash
docker compose --profile hostnet up -d tftpgui-host
```



---

### Notes

* GUI mode isn’t supported inside Docker — use headless mode.
* Host networking is simplest for Linux deployments; bridged networking works anywhere, but you must map UDP ports correctly.
* Make sure mounted volumes (`/data`, `/logs`) are writable by the container user.

## Logging

TFTPGui supports two types of logs:

### Standard Log (`log_file`)

Includes:

* Timestamp
* Client IP
* Filename
* Total bytes
* Success/failure state

### Audit Log (`audit_log_file`)

In JSONL format for automation and compliance.

Log rotation is controlled with:

* `log_max_bytes`
* `log_backup_count`
* `log_when`
* `log_interval`

---

## Planned Enhancements

Some future improvements already under consideration:

* GUI-based root directory selector
* Dark mode / theming
* System tray integration
* Built-in TFTP client tester
* Optional web UI
* Service install script

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025
Richard J. Sears
```

---

## Contributing

Suggestions and pull requests are welcome!
Feel free to fork the project, open issues, or submit ideas.

---

## Credits

Created by **Richard J. Sears**
Built in Python3 with a focus on real-world use, stability, and control.

## Acknowledgments
* **My Amazing and loving family!** My wonderful wife and kids put up with all my coding and automation projects and encouraged me in everything. Without them, my projects would not be possible.
* **My brother James**, who is a continual source of inspiration to me and others. Everyone should have a brother as awesome as mine!

