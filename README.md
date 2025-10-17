## TFTPGui – A Modern GUI & Headless TFTP Server in Python3 (v1.0.0)

![Last Commit](https://img.shields.io/github/last-commit/rjsears/tftpgui)
![Issues](https://img.shields.io/github/issues/rjsears/tftpgui)
![License](https://img.shields.io/badge/license-MIT-green)
![Contributors](https://img.shields.io/github/contributors/rjsears/tftpgui)
![Release](https://img.shields.io/github/v/release/rjsears/tftpgui)
![Docker Hub](https://img.shields.io/badge/docker-image%20coming%20soon-blue)

## What is TFTPGui?

TFTPGui is a fully modernized TFTP server written in Python 3 with both a **graphical user interface** and a **headless (CLI) mode** for true flexibility. It was built as a replacement for outdated Python2/xinetd TFTP setups and provides real-time monitoring, logging, security controls, and an easy configuration system.

If you work with firmware, embedded devices, PXE booting, routers/switches, or lab environments, this tool gives you all the control and visibility that the old stuff is missing.

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
- Docker image planned (coming soon)

---

## System Requirements

You only need a few basics:

- Python 3.10 or newer  
- `tkinter` installed (for GUI mode)  
- Linux, macOS, or WSL on Windows  
- Permissions to bind to TFTP ports (default 69)  
- Docker (optional, image coming soon)

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
3. A custom path specified with `--config`

### Example Config

```json
{
  "host": "0.0.0.0",
  "port": 69,
  "root_dir": "/tftp",

  "allow_write": true,
  "writable_subdirs": ["uploads", "staging"],
  "enforce_chroot": false,

  "filename_allowlist": [".bin", ".cfg", ".hex"],
  "allowlist_ips": ["192.168.0.0/24", "10.200.50.0/24"],
  "denylist_ips": [],

  "timeout_sec": 3.0,
  "max_retries": 5,
  "log_level": "INFO",

  "log_file": "/logs/tftpgui/tftpgui.log",
  "audit_log_file": "/logs/tftpgui/audit.jsonl",
  "transfer_log_file": "/logs/tftpgui/transfers.csv",

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
python3 tftpgui_enterprise.py --config /path/to/custom.json
```

---

## Running the Program

### GUI Mode

```bash
python3 tftpgui_enterprise.py
```

Or with a specific config:

```bash
python3 tftpgui_enterprise.py --config /path/config.json
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
docker pull rjsears/tftpgui:1.0.0
````

Or always grab the newest build:

```bash
docker pull rjsears/tftpgui:latest
```

---

### Quick run with host networking (Linux only)

```bash
docker run --rm -it --network host \
  -v /home/tftpd/tftp:/data \
  -v /home/tftpd/.tftpgui_config.json:/app/.tftpgui_config.json:ro \
  rjsears/tftpgui:1.0.0 -c /app/.tftpgui_config.json
```

This uses your host’s network stack directly, so you don’t have to map UDP ports.

---

### Run with bridged networking (portable)

```bash
docker run --rm -it \
  -p 69:69/udp \
  -p 50000-50100:50000-50100/udp \
  -v /home/tftpd/tftp:/data \
  -v /home/tftpd/.tftpgui_config.json:/app/.tftpgui_config.json:ro \
  rjsears/tftpgui:1.0.0 -c /app/.tftpgui_config.json
```

This maps **UDP port 69** and an ephemeral range (`50000–50100`) that the server uses for data transfers.
Adjust the range in both your config and command if you need more concurrent sessions.

---

### docker-compose example

```yaml
services:
  tftpgui:
    image: rjsears/tftpgui:1.0.0
    container_name: tftpgui
    restart: unless-stopped
    cap_add:
      - NET_BIND_SERVICE
    network_mode: host   # easiest option (Linux only)
    volumes:
      - /home/tftpd/tftp:/data:rw
      - /home/tftpd/.tftpgui_config.json:/app/.tftpgui_config.json:ro
      - /home/tftpd/tftpgui/logs:/logs
    command: ["-c", "/app/.tftpgui_config.json"]
```

If you prefer port mappings instead of host networking, remove `network_mode: host` and add `ports:` for `69/udp` and your ephemeral range.

---

### Container config file

Inside the container, the root directory is `/data` and logs are in `/logs`.
Here’s an example `container.tftpgui_config.json` you can mount:

```json
{
  "host": "0.0.0.0",
  "port": 69,
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

---

### Notes

* GUI mode isn’t supported inside Docker — use headless mode.
* Host networking is simplest for Linux deployments; bridged networking works anywhere, but you must map UDP ports correctly.
* Make sure mounted volumes (`/data`, `/logs`) are writable by the container user.


## GUI Overview (Screenshots Coming Soon)

### Main Window

*(screenshot soon)*

### Transfer Progress View

*(screenshot soon)*

Shows:

* Filename
* Remote IP
* Total bytes
* Transfer progress bar
* Errors/success state

### Configuration Panel

*(screenshot soon)*

Editable fields for:

* IP rules
* Logging settings
* Rotation
* Write permissions
* Timeouts

---

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
* **My Amazing and loving family!** My wonderful wife and kids put up with all my coding and automation projects and encouraged me in everything. Without them, this project would not be possible.
* **My brother James**, who is a continual source of inspiration to me and others. Everyone should have a brother as awesome as mine!

