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
  "root_dir": "/home/username/tftp",
  "allow_write": true,
  "writable_subdirs": ["uploads", "staging"],
  "enforce_chroot": false,
  "filename_allowlist": [".bin", ".cfg", ".hex"],
  "allowlist_ips": ["192.168.0.0/24", "10.200.50.0/24"],
  "denylist_ips": ["0.0.0.0/8"],
  "log_file": "/home/username/tftpd.log",
  "audit_log_file": "/var/log/tftpgui/audit.jsonl",
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

## Docker Support (Coming Soon)

A Docker image will be published to Docker Hub under:

```
rjsears/tftpgui
```

Once live, you’ll be able to run:

```bash
docker pull rjsears/tftpgui:1.0.0
```

Then:

```bash
docker run -it --rm \
  -p 69:69/udp \
  -v /home/crypto/tftp:/tftp-root \
  -v /home/crypto/.tftpgui_config.json:/app/.tftpgui_config.json \
  rjsears/tftpgui:1.0.0
```

Until then, the Dockerfile and image setup will remain in progress.

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

