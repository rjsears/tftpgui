#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations


"""
TFTP Server (GUI + Headless) with optional Web UI (FastAPI).

- Supports RRQ/WRQ (octet), blocksize negotiation, tsize
- Progress, rate, ETA, hashes; audit JSONL and CSV transfer logs
- GUI mode (Tk) with colored banner + live table
- Headless mode with console progress
- Optional Web UI: dashboard, /api/status, /api/events (SSE)
- Config file: looks in CWD .tftpgui_config.json, then HOME; override -c

CLI:
  python3 tftpgui_enterprise.py -c /path/conf.json       # GUI
  python3 tftpgui_enterprise.py --headless -c conf.json   # Headless
  python3 tftpgui_enterprise.py -c conf.json -w -p 8080   # Start Web UI too
"""

__author__ = "Richard J. Sears"
VERSION = "1.1.0 (2025-10-19)"

import argparse
import asyncio
import csv
import hashlib
import ipaddress
import json
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import os
import queue
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

# ---------- Optional web deps (gracefully absent if not installed) ----------
try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
    import uvicorn
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore
    uvicorn = None  # type: ignore

# ---------- Optional GUI (Tk) ----------
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
except Exception:  # pragma: no cover
    tk = None  # type: ignore
    ttk = None  # type: ignore
    ScrolledText = None  # type: ignore

# ---------- TFTP constants ----------
TFTP_PORT_DEFAULT = 69
DEFAULT_BLOCK_SIZE = 512
MIN_BLOCK_SIZE = 8
MAX_BLOCK_SIZE = 65464
OP_RRQ = 1
OP_WRQ = 2
OP_DATA = 3
OP_ACK = 4
OP_ERROR = 5
OP_OACK = 6

ERR_NOT_DEFINED = 0
ERR_FILE_NOT_FOUND = 1
ERR_ACCESS_VIOLATION = 2
ERR_DISK_FULL = 3
ERR_ILLEGAL_OPERATION = 4
ERR_UNKNOWN_TID = 5
ERR_FILE_EXISTS = 6
ERR_NO_SUCH_USER = 7

CWD_CONFIG_NAME = ".tftpgui_config.json"
HOME_CONFIG_NAME = ".tftpgui_config.json"

# ---------- Config ----------

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = TFTP_PORT_DEFAULT
    root_dir: Path = Path("")
    allow_write: bool = False
    writable_subdirs: List[str] = field(default_factory=list)
    enforce_chroot: bool = False
    filename_allowlist: List[str] = field(default_factory=list)
    allowlist_ips: List[str] = field(default_factory=list)
    denylist_ips: List[str] = field(default_factory=list)

    timeout_sec: float = 3.0
    max_retries: int = 5
    log_level: str = "INFO"

    log_file: Optional[Path] = None
    log_rotation: str = "size"  # "size" or "time"
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 5
    log_when: str = "midnight"
    log_interval: int = 1

    audit_log_file: Optional[Path] = None
    transfer_log_file: Optional[Path] = None

    metrics_window_sec: int = 5
    ephemeral_ports: bool = True
    transfer_port_min: Optional[int] = None
    transfer_port_max: Optional[int] = None

    web: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "host": "0.0.0.0", "port": 8080})

    config_file: Path = field(default=Path.cwd() / CWD_CONFIG_NAME)

    def to_json(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "root_dir": str(self.root_dir),
            "allow_write": self.allow_write,
            "writable_subdirs": self.writable_subdirs,
            "enforce_chroot": self.enforce_chroot,
            "filename_allowlist": self.filename_allowlist,
            "allowlist_ips": self.allowlist_ips,
            "denylist_ips": self.denylist_ips,
            "timeout_sec": self.timeout_sec,
            "max_retries": self.max_retries,
            "log_level": self.log_level,
            "log_file": str(self.log_file) if self.log_file else None,
            "log_rotation": self.log_rotation,
            "log_max_bytes": self.log_max_bytes,
            "log_backup_count": self.log_backup_count,
            "log_when": self.log_when,
            "log_interval": self.log_interval,
            "audit_log_file": str(self.audit_log_file) if self.audit_log_file else None,
            "transfer_log_file": str(self.transfer_log_file) if self.transfer_log_file else None,
            "metrics_window_sec": self.metrics_window_sec,
            "ephemeral_ports": self.ephemeral_ports,
            "transfer_port_min": self.transfer_port_min,
            "transfer_port_max": self.transfer_port_max,
            "web": self.web,
        }

    @classmethod
    def from_json(cls, data: dict) -> "ServerConfig":
        cfg = cls()
        cfg.host = str(data.get("host", cfg.host))
        cfg.port = int(data.get("port", cfg.port))
        cfg.root_dir = Path(data.get("root_dir", ""))
        cfg.allow_write = bool(data.get("allow_write", cfg.allow_write))
        cfg.writable_subdirs = list(data.get("writable_subdirs", cfg.writable_subdirs))
        cfg.enforce_chroot = bool(data.get("enforce_chroot", cfg.enforce_chroot))
        cfg.filename_allowlist = [s.lower() for s in data.get("filename_allowlist", cfg.filename_allowlist)]
        cfg.allowlist_ips = list(data.get("allowlist_ips", cfg.allowlist_ips))
        cfg.denylist_ips = list(data.get("denylist_ips", cfg.denylist_ips))
        cfg.timeout_sec = float(data.get("timeout_sec", cfg.timeout_sec))
        cfg.max_retries = int(data.get("max_retries", cfg.max_retries))
        cfg.log_level = str(data.get("log_level", cfg.log_level))

        lf = data.get("log_file")
        cfg.log_file = Path(lf) if lf else None
        cfg.log_rotation = str(data.get("log_rotation", cfg.log_rotation)).lower()
        cfg.log_max_bytes = int(data.get("log_max_bytes", cfg.log_max_bytes))
        cfg.log_backup_count = int(data.get("log_backup_count", cfg.log_backup_count))
        cfg.log_when = str(data.get("log_when", cfg.log_when))
        cfg.log_interval = int(data.get("log_interval", cfg.log_interval))

        alf = data.get("audit_log_file")
        cfg.audit_log_file = Path(alf) if (alf not in (None, "")) else None
        tlf = data.get("transfer_log_file")
        cfg.transfer_log_file = Path(tlf) if (tlf not in (None, "")) else None

        cfg.metrics_window_sec = int(data.get("metrics_window_sec", cfg.metrics_window_sec))
        cfg.ephemeral_ports = bool(data.get("ephemeral_ports", cfg.ephemeral_ports))
        cfg.transfer_port_min = data.get("transfer_port_min", None)
        cfg.transfer_port_max = data.get("transfer_port_max", None)
        if cfg.transfer_port_min is not None:
            cfg.transfer_port_min = int(cfg.transfer_port_min)
        if cfg.transfer_port_max is not None:
            cfg.transfer_port_max = int(cfg.transfer_port_max)

        cfg.web = dict(data.get("web", {"enabled": False, "host": "0.0.0.0", "port": 8080}))
        return cfg


def default_config_template() -> dict:
    return {
        "host": "0.0.0.0",
        "port": 69,
        "root_dir": "",
        "allow_write": False,
        "writable_subdirs": [],
        "enforce_chroot": False,
        "filename_allowlist": [],
        "allowlist_ips": [],
        "denylist_ips": [],
        "timeout_sec": 3.0,
        "max_retries": 5,
        "log_level": "INFO",
        "log_file": None,
        "log_rotation": "size",
        "log_max_bytes": 5000000,
        "log_backup_count": 5,
        "log_when": "midnight",
        "log_interval": 1,
        "audit_log_file": None,
        "transfer_log_file": None,
        "metrics_window_sec": 5,
        "ephemeral_ports": True,
        "transfer_port_min": 50000,
        "transfer_port_max": 50100,
        "web": {"enabled": False, "host": "0.0.0.0", "port": 8080},
    }


def resolve_config_path(cli_override: Optional[str] = None) -> Path:
    if cli_override:
        p = Path(cli_override).expanduser()
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(default_config_template(), indent=2), encoding="utf-8")
        return p
    for p in [Path.cwd() / CWD_CONFIG_NAME, Path.home() / HOME_CONFIG_NAME]:
        if p.exists():
            return p
    # Create in CWD if none found
    create_path = Path.cwd() / CWD_CONFIG_NAME
    create_path.write_text(json.dumps(default_config_template(), indent=2), encoding="utf-8")
    return create_path


def load_config(path: Path) -> ServerConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = ServerConfig.from_json(data)
    cfg.config_file = path
    return cfg


def save_config(cfg: ServerConfig) -> None:
    if not cfg.config_file:
        raise ValueError("Config file path is not set on ServerConfig.")
    cfg.config_file.write_text(json.dumps(cfg.to_json(), indent=2), encoding="utf-8")


def validate_root_dir(cfg: ServerConfig) -> None:
    if not str(cfg.root_dir).strip():
        raise ValueError("root_dir is not set. Edit the config file and set it to an existing directory.")
    root = Path(cfg.root_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"root_dir '{root}' does not exist or is not a directory.")


def validate_transfer_port_range(cfg: ServerConfig) -> None:
    if cfg.transfer_port_min is None or cfg.transfer_port_max is None:
        return
    lo, hi = cfg.transfer_port_min, cfg.transfer_port_max
    if not (1024 <= lo < hi <= 65535):
        raise ValueError(f"Invalid transfer port range: {lo}-{hi}")

# ---------- Utility ----------

def safe_join(root: Path, user_path: str) -> Optional[Path]:
    candidate = (root / user_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def ext_allowed(filename: str, allowlist: List[str]) -> bool:
    if not allowlist:
        return True
    ext = Path(filename).suffix.lower()
    return ext in set(s.lower() for s in allowlist)


def ip_in_list(ip: str, patterns: List[str]) -> bool:
    if not patterns:
        return False
    try:
        ip_obj = ipaddress.ip_address(ip)
        for item in patterns:
            try:
                net = ipaddress.ip_network(item, strict=False)
                if ip_obj in net:
                    return True
            except ValueError:
                if ip == item:
                    return True
    except ValueError:
        return False
    return False


def parse_null_fields(data: bytes, start: int = 2) -> List[str]:
    parts = data[start:].split(b"\x00")
    out: List[str] = []
    for bval in parts:
        if not bval:
            continue
        try:
            out.append(bval.decode(errors="ignore"))
        except Exception:
            out.append("")
    return out

def parse_rrq_wrq_with_options(data: bytes) -> Tuple[int, str, str, Dict[str, str]]:
    opcode = struct.unpack("!H", data[:2])[0]
    fields = parse_null_fields(data, 2)
    if len(fields) < 2:
        raise ValueError("Malformed RRQ/WRQ")
    filename = fields[0]
    mode = fields[1].lower()
    options: Dict[str, str] = {}
    for i in range(2, len(fields), 2):
        if i + 1 < len(fields):
            options[fields[i].lower()] = fields[i + 1]
    return opcode, filename, mode, options


def build_data(block_no: int, payload: bytes) -> bytes:
    return struct.pack("!HH", OP_DATA, block_no) + payload


def build_ack(block_no: int) -> bytes:
    return struct.pack("!HH", OP_ACK, block_no)


def build_error(code: int, msg: str) -> bytes:
    return struct.pack("!HH", OP_ERROR, code) + msg.encode() + b"\x00"


def build_oack(options: Dict[str, str]) -> bytes:
    payload = b""
    for k, v in options.items():
        payload += k.encode() + b"\x00" + v.encode() + b"\x00"
    return struct.pack("!H", OP_OACK) + payload


def clamp_blksize(req: Optional[int]) -> int:
    if not req:
        return DEFAULT_BLOCK_SIZE
    return max(MIN_BLOCK_SIZE, min(MAX_BLOCK_SIZE, int(req)))


def fmt_eta(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02}"


def file_hashes(path: Path) -> Dict[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as fobj:
        for chunk in iter(lambda: fobj.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def write_audit(audit_path: Optional[Path], record: dict) -> None:
    if not audit_path:
        return
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as fobj:
            fobj.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def write_transfer_log_csv(path: Optional[Path], row: Dict[str, object]) -> None:
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as fobj:
            writer = csv.DictWriter(
                fobj,
                fieldnames=[
                    "timestamp",
                    "client_ip",
                    "client_port",
                    "direction",
                    "filename",
                    "total_size",
                    "bytes_done",
                    "percent",
                    "status",
                    "message",
                    "md5",
                    "sha256",
                    "duration_sec",
                    "avg_rate_bps",
                    "block_size",
                ],
            )
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        pass
@dataclass
class Session:
    client: Tuple[str, int]
    mode: str
    file_path: Path
    is_write: bool
    total_size: int = 0
    bytes_done: int = 0
    start_time: float = field(default_factory=time.time)
    last_block_sent: int = 0
    last_packet: bytes = b""
    last_send_time: float = 0.0
    retries: int = 0
    complete: bool = False
    error: Optional[str] = None
    fh: Optional[object] = None

    block_size: int = DEFAULT_BLOCK_SIZE
    expect_size: Optional[int] = None
    sent_oack: bool = False
    requested_opts: Dict[str, str] = field(default_factory=dict)
    history: Deque[Tuple[float, int]] = field(default_factory=deque)

    @property
    def percent(self) -> float:
        total = self.total_size or self.expect_size or 0
        if total <= 0:
            return 0.0
        return min(100.0, (self.bytes_done / total) * 100.0)

    def rate_bps_window(self, window_sec: int) -> float:
        now = time.time()
        self.history.append((now, self.bytes_done))
        while self.history and (now - self.history[0][0]) > window_sec:
            self.history.popleft()
        if len(self.history) < 2:
            elapsed = max(1e-6, now - self.start_time)
            return self.bytes_done / elapsed
        t0, b0 = self.history[0]
        t1, b1 = self.history[-1]
        dt = max(1e-6, t1 - t0)
        db = max(0, b1 - b0)
        return db / dt

    def eta_seconds(self, window_sec: int) -> Optional[float]:
        total = self.total_size or self.expect_size or 0
        if total <= 0:
            return None
        remaining = max(0, total - self.bytes_done)
        rate = self.rate_bps_window(window_sec)
        if rate <= 1e-9:
            return None
        return remaining / rate


@dataclass
class Event:
    kind: str
    client: Tuple[str, int]
    filename: str
    is_write: bool
    bytes_done: int = 0
    total_size: int = 0
    percent: float = 0.0
    rate_avg: float = 0.0
    eta: str = "—"
    blk: int = DEFAULT_BLOCK_SIZE
    opts: str = ""
    message: str = ""
    hashes: Optional[Dict[str, str]] = None


class EventFanout:
    """Broadcasts events to multiple consumers (GUI, Web, console)."""
    def __init__(self, history_max: int = 1000) -> None:
        self._subs: List[queue.Queue] = []
        self._lock = threading.Lock()
        self.recent: Deque[Event] = deque(maxlen=history_max)

    def register(self) -> "queue.Queue[Event]":
        q: "queue.Queue[Event]" = queue.Queue()
        with self._lock:
            self._subs.append(q)
        return q

    def put_nowait(self, evt: Event) -> None:
        self.recent.append(evt)
        with self._lock:
            for q in list(self._subs):
                try:
                    q.put_nowait(evt)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Server Thread (Asyncio listener and transfer logic)
# ---------------------------------------------------------------------------

class ServerThread:
    """Runs the asynchronous TFTP server in a background thread with proper
    event fanout and web/UI synchronization.
    """

    def __init__(self, cfg: ServerConfig, logger: logging.Logger, event_q: "EventFanout"):
        self.cfg = cfg
        self.logger = logger
        self.event_q = event_q
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._transport: Optional[asyncio.transports.DatagramTransport] = None
        self.is_running: bool = False  # actual server state flag

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _broadcast_server(self, running: bool) -> None:
        """Send a 'server' event to all subscribers and update .is_running."""
        self.is_running = running
        try:
            evt = Event(
                kind="server",
                client=(self.cfg.host, self.cfg.port),
                filename="",
                is_write=False,
                message="running" if running else "stopped",
            )
            self.event_q.put_nowait(evt)
        except Exception:
            pass

    def _maybe_chroot(self) -> None:
        """Perform chroot if enabled and running as root."""
        if not self.cfg.enforce_chroot:
            return
        try:
            validate_root_dir(self.cfg)
            if hasattr(os, "geteuid") and os.geteuid() != 0:
                self.logger.warning(
                    "enforce_chroot requested but not running as root; skipping chroot."
                )
                return
            os.chroot(Path(self.cfg.root_dir))
            os.chdir("/")
            self.logger.info("Chrooted to %s", self.cfg.root_dir)
            self.cfg.root_dir = Path("/")
        except Exception as exc:
            self.logger.error("Chroot failed: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle methods
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the asyncio server loop in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run_loop, name="TFTP-Server", daemon=True
        )
        self._thread.start()

    def _run_loop(self) -> None:
        """Main asyncio loop for the TFTP listener and transfers."""
        try:
            validate_root_dir(self.cfg)
            validate_transfer_port_range(self.cfg)
            self._maybe_chroot()

            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            listen = self._loop.create_datagram_endpoint(
                lambda: ListenerProtocol(self.cfg, self.logger, self.event_q),
                local_addr=(self.cfg.host, self.cfg.port),
                allow_broadcast=True,
                reuse_port=False,
            )

            self._transport, _ = self._loop.run_until_complete(listen)

            # Announce server running
            self._broadcast_server(True)
            self.logger.info("Listener bound on %s:%s", self.cfg.host, self.cfg.port)

            self._loop.run_forever()

        except (OSError, ValueError) as exc:
            self.logger.error("Failed to start server: %s", exc)

        finally:
            # Always announce stopped
            self._broadcast_server(False)
            self.is_running = False

            if self._transport:
                self._transport.close()

            if self._loop:
                # Cancel all pending tasks cleanly
                pending = asyncio.all_tasks(loop=self._loop)
                for task in pending:
                    task.cancel()
                try:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                except Exception:
                    pass
                self._loop.stop()
                self._loop.close()
            self.logger.info("TFTP server loop stopped.")

    def stop(self) -> None:
        """Stop the TFTP server cleanly."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self.is_running = False
        self._broadcast_server(False)
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        self.logger.info("TFTP server stopped.")

# ---------------------------------------------------------------------------
# Asyncio listener + per-transfer protocols
# ---------------------------------------------------------------------------

class ListenerProtocol(asyncio.DatagramProtocol):
    def __init__(self, cfg: ServerConfig, logger: logging.Logger, event_q: EventFanout):
        self.cfg = cfg
        self.logger = logger
        self.event_q = event_q
        self.transport: Optional[asyncio.transports.DatagramTransport] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore
        self.loop = asyncio.get_event_loop()
        self.logger.info("Listener bound on %s:%s", self.cfg.host, self.cfg.port)

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        client_ip = addr[0]

        # Access policy checks
        if self.cfg.allowlist_ips and not ip_in_list(client_ip, self.cfg.allowlist_ips):
            self._send_error(addr, ERR_ACCESS_VIOLATION, "Access denied (allowlist)")
            return
        if ip_in_list(client_ip, self.cfg.denylist_ips):
            self._send_error(addr, ERR_ACCESS_VIOLATION, "Access denied (denylist)")
            return

        try:
            opcode = struct.unpack("!H", data[:2])[0]
        except Exception:
            return

        if opcode not in (OP_RRQ, OP_WRQ):
            return

        try:
            opcode, filename, mode, options = parse_rrq_wrq_with_options(data)
        except Exception:
            self._send_error(addr, ERR_ILLEGAL_OPERATION, "Malformed RRQ/WRQ")
            return

        if mode != "octet":
            self._send_error(addr, ERR_ILLEGAL_OPERATION, "Only octet mode supported")
            return

        if not ext_allowed(filename, self.cfg.filename_allowlist):
            self._send_error(addr, ERR_ACCESS_VIOLATION, "Disallowed file type")
            return

        try:
            validate_root_dir(self.cfg)
        except ValueError as e:
            self._send_error(addr, ERR_ACCESS_VIOLATION, "Server root not configured")
            self.logger.error(str(e))
            return

        root_dir = Path(self.cfg.root_dir).expanduser().resolve()
        path = safe_join(root_dir, filename)
        if path is None:
            self._send_error(addr, ERR_ACCESS_VIOLATION, "Access violation")
            return

        want_blksize = clamp_blksize(int(options.get("blksize", DEFAULT_BLOCK_SIZE)) if "blksize" in options else None)
        want_tsize = options.get("tsize")

        if opcode == OP_RRQ:  # DOWNLOAD
            if not path.exists() or not path.is_file():
                self._send_error(addr, ERR_FILE_NOT_FOUND, "File not found")
                return
            try:
                fh = path.open("rb")
                total = path.stat().st_size
            except Exception:
                self._send_error(addr, ERR_ACCESS_VIOLATION, "Cannot open file")
                return

            sess = Session(
                client=addr,
                mode=mode,
                file_path=path,
                is_write=False,
                fh=fh,
                total_size=total,
                block_size=want_blksize,
                requested_opts=options,
            )
            self.loop.create_task(self._spawn_transfer(sess, options, rrq=True))

        elif opcode == OP_WRQ:  # UPLOAD
            if not self.cfg.allow_write:
                self._send_error(addr, ERR_ACCESS_VIOLATION, "Writes disabled")
                return

            # Restrict to writable subdirs if configured
            if self.cfg.writable_subdirs:
                ok = False
                for sub in self.cfg.writable_subdirs:
                    sub_root = (root_dir / sub).resolve()
                    try:
                        path.relative_to(sub_root)
                        ok = True
                        break
                    except Exception:
                        continue
                if not ok:
                    self._send_error(addr, ERR_ACCESS_VIOLATION, "Write not allowed in this path")
                    return

            if path.exists():
                self._send_error(addr, ERR_FILE_EXISTS, "File exists")
                return

            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                fh = path.open("wb")
            except Exception:
                self._send_error(addr, ERR_ACCESS_VIOLATION, "Cannot create file")
                return

            sess = Session(
                client=addr,
                mode=mode,
                file_path=path,
                is_write=True,
                fh=fh,
                block_size=want_blksize,
                requested_opts=options,
            )
            if want_tsize is not None:
                try:
                    sess.expect_size = int(want_tsize)
                except Exception:
                    pass

            self.loop.create_task(self._spawn_transfer(sess, options, rrq=False))

    async def _spawn_transfer(self, sess: Session, options: Dict[str, str], rrq: bool) -> None:
        host = self.cfg.host
        lo, hi = self.cfg.transfer_port_min, self.cfg.transfer_port_max

        if lo is not None and hi is not None and lo < hi:
            transport = None
            last_exc = None
            for p in range(lo, hi + 1):
                try:
                    transport, protocol = await self.loop.create_datagram_endpoint(  # type: ignore
                        lambda: TransferProtocol(self.cfg, self.logger, self.event_q, sess),
                        local_addr=(host, p),
                    )
                    break
                except Exception as e:
                    last_exc = e
                    continue
            if transport is None:
                self.logger.error("No free data port in %s-%s: %s", lo, hi, last_exc)
                self._send_error(sess.client, ERR_NOT_DEFINED, "No data port")
                return
        else:
            local_addr = (host, 0) if self.cfg.ephemeral_ports else (host, self.cfg.port)
            transport, protocol = await self.loop.create_datagram_endpoint(  # type: ignore
                lambda: TransferProtocol(self.cfg, self.logger, self.event_q, sess),
                local_addr=local_addr,
            )

        # OACK options
        oack_opts: Dict[str, str] = {}
        if "blksize" in options:
            oack_opts["blksize"] = str(sess.block_size)
        if rrq and "tsize" in options:
            oack_opts["tsize"] = str(sess.total_size)
        if (not rrq) and ("tsize" in options) and sess.expect_size is not None:
            oack_opts["tsize"] = str(sess.expect_size)

        if rrq:
            protocol.start_download(oack_opts if oack_opts else None)
        else:
            protocol.start_upload(oack_opts if oack_opts else None)

        # Emit start event
        opts_str = ",".join(f"{k}={v}" for k, v in options.items()) if options else ""
        self.event_q.put_nowait(
            Event(
                kind="start",
                client=sess.client,
                filename=sess.file_path.name,
                is_write=sess.is_write,
                bytes_done=sess.bytes_done,
                total_size=sess.total_size or (sess.expect_size or 0),
                percent=sess.percent,
                rate_avg=0.0,
                eta="—",
                blk=sess.block_size,
                opts=opts_str,
                message="Download started" if not sess.is_write else "Upload started",
            )
        )

    def _send_error(self, addr: Tuple[str, int], code: int, msg: str) -> None:
        if self.transport is not None:
            self.transport.sendto(build_error(code, msg), addr)
        # Audit error
        write_audit(
            self.cfg.audit_log_file,
            {"ts": time.time(), "event": "error", "client_ip": addr[0], "client_port": addr[1], "code": code, "message": msg},
        )


class TransferProtocol(asyncio.DatagramProtocol):
    def __init__(self, cfg: ServerConfig, logger: logging.Logger, event_q: EventFanout, sess: Session):
        self.cfg = cfg
        self.logger = logger
        self.event_q = event_q
        self.sess = sess
        self.transport: Optional[asyncio.transports.DatagramTransport] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._retransmit_task: Optional[asyncio.Task] = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore
        self.loop = asyncio.get_event_loop()
        self._retransmit_task = self.loop.create_task(self._retransmit_loop())

    def connection_lost(self, exc: Optional[Exception]) -> None:
        if self._retransmit_task:
            self._retransmit_task.cancel()
        try:
            if self.sess.fh:
                self.sess.fh.close()
        except Exception:
            pass
        self.sess.complete = True

    async def _retransmit_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(0.1)
                now = time.time()
                s = self.sess
                if s.complete:
                    continue
                if s.last_packet and (now - s.last_send_time) > self.cfg.timeout_sec:
                    if s.retries >= self.cfg.max_retries:
                        self._emit_event("error", "Max retries reached")
                        self._end_session()
                        continue
                    self._sendto(s.last_packet, s.client)
                    s.last_send_time = now
                    s.retries += 1
        except asyncio.CancelledError:
            return

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        if addr != self.sess.client:
            self._sendto(build_error(ERR_UNKNOWN_TID, "Unknown transfer ID"), addr)
            return
        try:
            opcode = struct.unpack("!H", data[:2])[0]
        except Exception:
            return

        if opcode == OP_ACK:
            self._on_ack(data)
        elif opcode == OP_DATA:
            self._on_data(data)
        else:
            self._sendto(build_error(ERR_ILLEGAL_OPERATION, "Illegal TFTP operation"), addr)

    # --- Initiators ---
    def start_download(self, oack: Optional[Dict[str, str]]) -> None:
        if oack:
            self._send_oack(oack)
        else:
            self._send_block(1)

    def start_upload(self, oack: Optional[Dict[str, str]]) -> None:
        if oack:
            self._send_oack(oack)
        else:
            self._send_ack(0)

    # --- Handlers ---
    def _on_ack(self, data: bytes) -> None:
        s = self.sess
        try:
            _, block_no = struct.unpack("!HH", data[:4])
        except Exception:
            return

        # First ACK after OACK
        if s.sent_oack and block_no == 0:
            s.sent_oack = False
            if s.is_write:
                return
            self._send_block(1)
            return

        if s.is_write:
            return  # uploads expect DATA, not ACK

        if block_no == s.last_block_sent:
            inc = s.block_size
            if s.total_size:
                s.bytes_done = min(s.total_size, s.bytes_done + inc)
            else:
                s.bytes_done += inc
            self._emit_event("progress", f"Sent block {block_no}")
            self._send_block(block_no + 1)

            if s.total_size > 0 and s.bytes_done >= s.total_size:
                self._emit_event("complete", "Download complete")
                self._end_session()

    def _on_data(self, data: bytes) -> None:
        s = self.sess
        if not s.is_write:
            return
        try:
            _, block_no = struct.unpack("!HH", data[:4])
            payload = data[4:]
        except Exception:
            return

        if not s.fh:
            return
        try:
            s.fh.write(payload)
            s.fh.flush()
        except Exception:
            self._sendto(build_error(ERR_DISK_FULL, "Disk full or allocation exceeded"), s.client)
            self._emit_event("error", "Disk full or write error")
            self._end_session()
            return

        s.bytes_done += len(payload)
        self._emit_event("progress", f"Received block {block_no}")
        self._send_ack(block_no)

        if len(payload) < s.block_size:
            hashes = file_hashes(s.file_path)
            self._emit_event("complete", "Upload complete", hashes=hashes)
            self._end_session()

    # --- Senders ---
    def _send_oack(self, options: Dict[str, str]) -> None:
        s = self.sess
        packet = build_oack(options)
        self._sendto(packet, s.client)
        s.last_packet = packet
        s.last_send_time = time.time()
        s.retries = 0
        s.sent_oack = True
        self._emit_event("oack", f"OACK {options}")

    def _send_block(self, block_no: int) -> None:
        s = self.sess
        if not s.fh:
            return
        s.fh.seek((block_no - 1) * s.block_size)
        chunk = s.fh.read(s.block_size)
        packet = build_data(block_no, chunk)
        self._sendto(packet, s.client)
        s.last_packet = packet
        s.last_block_sent = block_no
        s.last_send_time = time.time()
        s.retries = 0

    def _send_ack(self, block_no: int) -> None:
        s = self.sess
        packet = build_ack(block_no)
        self._sendto(packet, s.client)
        s.last_packet = packet
        s.last_send_time = time.time()
        s.retries = 0

    def _sendto(self, payload: bytes, addr: Tuple[str, int]) -> None:
        if self.transport is not None:
            self.transport.sendto(payload, addr)

    def _end_session(self) -> None:
        s = self.sess
        try:
            if s.fh:
                s.fh.close()
        except Exception:
            pass
        s.complete = True
        if self.transport is not None:
            self.transport.close()

    def _emit_event(self, kind: str, message: str, hashes: Optional[Dict[str, str]] = None) -> None:
        s = self.sess
        avg_rate = s.rate_bps_window(window_sec=self.cfg.metrics_window_sec)
        eta_txt = fmt_eta(s.eta_seconds(self.cfg.metrics_window_sec))
        opts_str = ",".join(f"{k}={v}" for k, v in s.requested_opts.items()) if s.requested_opts else ""
        evt = Event(
            kind=kind,
            client=s.client,
            filename=s.file_path.name,
            is_write=s.is_write,
            bytes_done=s.bytes_done,
            total_size=s.total_size or (s.expect_size or 0),
            percent=s.percent,
            rate_avg=avg_rate,
            eta=eta_txt,
            blk=s.block_size,
            opts=opts_str,
            message=message,
            hashes=hashes,
        )
        self.event_q.put_nowait(evt)

        # Structured log + audit + transfer CSV
        direction = "UPLOAD" if s.is_write else "DOWNLOAD"
        self.logger.info(
            "%s %s %s:%s %s %s/%s avg=%.0fB/s ETA=%s blk=%d opts=%s - %s",
            kind.upper(), direction, s.client[0], s.client[1], s.file_path.name,
            s.bytes_done, s.total_size or (s.expect_size or 0), avg_rate, eta_txt,
            s.block_size, opts_str, message,
        )

        write_audit(
            self.cfg.audit_log_file,
            {
                "ts": time.time(),
                "event": kind,
                "client_ip": s.client[0],
                "client_port": s.client[1],
                "file": str(s.file_path),
                "direction": "upload" if s.is_write else "download",
                "bytes_done": s.bytes_done,
                "total": s.total_size or (s.expect_size or 0),
                "percent": s.percent,
                "avg_rate_bps": avg_rate,
                "eta": eta_txt,
                "blk": s.block_size,
                "opts": s.requested_opts,
                "hashes": hashes,
            },
        )

        if kind in ("start", "complete", "error"):
            nowiso = datetime.now(timezone.utc).isoformat()
            md5 = hashes.get("md5") if hashes else ""
            sha256 = hashes.get("sha256") if hashes else ""
            status = "ok" if kind == "complete" else ("start" if kind == "start" else "error")
            duration = time.time() - s.start_time if s.start_time else 0.0
            write_transfer_log_csv(
                self.cfg.transfer_log_file,
                {
                    "timestamp": nowiso,
                    "client_ip": s.client[0],
                    "client_port": s.client[1],
                    "direction": direction.lower(),
                    "filename": s.file_path.name,
                    "total_size": s.total_size or (s.expect_size or 0),
                    "bytes_done": s.bytes_done,
                    "percent": round(s.percent, 3),
                    "status": status,
                    "message": message,
                    "md5": md5,
                    "sha256": sha256,
                    "duration_sec": round(duration, 3),
                    "avg_rate_bps": round(avg_rate, 1),
                    "block_size": s.block_size,
                },
            )
# ---------------------------------------------------------------------------
# GUI log handler (writes logs into ScrolledText)
# ---------------------------------------------------------------------------

class TextHandler(logging.Handler):
    def __init__(self, widget: ScrolledText):
        super().__init__()
        self.widget = widget

    def emit(self, self_record: logging.LogRecord) -> None:
        try:
            if not self.widget or not self.widget.winfo_exists():
                return
            msg = self.format(self_record)
            self.widget.configure(state="normal")
            self.widget.insert("end", msg + "\n")
            self.widget.configure(state="disabled")
            self.widget.yview_moveto(1.0)
        except Exception:
            return


# --- Headless-safe Tk base selection ----------------------------------------
if tk is None:
    class _TkBase:
        def __init__(self, *args, **kwargs) -> None: ...
        def mainloop(self) -> None: ...
        def protocol(self, *args, **kwargs) -> None: ...
        def after(self, *args, **kwargs): return None
        def after_cancel(self, *args, **kwargs) -> None: ...
        def destroy(self) -> None: ...
        def title(self, *args, **kwargs) -> None: ...
        def geometry(self, *args, **kwargs) -> None: ...
        def config(self, *args, **kwargs) -> None: ...
else:
    _TkBase = tk.Tk


# ---------------------------------------------------------------------------
# Tk GUI application
# ---------------------------------------------------------------------------

class TFTPApp(_TkBase):  # type: ignore
    """Tkinter GUI application for configuring and running the TFTP server."""

    def __init__(self, cfg: ServerConfig, start_web: bool = False):
        super().__init__()
        self._start_web = bool(start_web)
        self.cfg = cfg

        self.title(f"TFTP Server Python3 -- Version {VERSION} {__author__}")
        self.geometry("1280x800")
        self._closing = False
        self._poll_after_id: Optional[str] = None

        self._build_menubar()

        self.logger = logging.getLogger("tftpgui")
        self.logger.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))
        self._setup_logging()

        # Event fanout: GUI subscribes; server publishes to the fanout
        self._fanout = EventFanout()
        self.event_q: "queue.Queue[Event]" = self._fanout.register()
        self.server_thread = ServerThread(self.cfg, self.logger, self._fanout)
        self.running = False

        self._build_widgets()
        self._load_config_file()
        self._update_buttons()
        self._update_banner()
        self._schedule_poll()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Optionally launch the Web UI within GUI mode
        if self._start_web and FastAPI is not None:
            try:
                app_web = create_web_app(self.cfg, self._fanout, self.server_thread)
                threading.Thread(
                    target=run_web,
                    args=(
                        app_web,
                        str(self.cfg.web.get("host", "0.0.0.0")),
                        int(self.cfg.web.get("port", 8080)),
                    ),
                    name="TFTP-Web",
                    daemon=True,
                ).start()
                print(f"Web UI starting on {self.cfg.web.get('host', '0.0.0.0')}:{self.cfg.web.get('port', 8080)}")
            except Exception as e:
                print(f"Web UI failed to start: {e}")

    # ---- Menus --------------------------------------------------------------

    def _build_menubar(self) -> None:
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Reload Config", command=self._load_config_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About…", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _show_about(self) -> None:
        try:
            messagebox.showinfo(
                "About TFTP Server",
                (
                    "TFTP Server Python3\n\n"
                    f"Version: {VERSION}\n"
                    f"Author: {__author__}\n\n"
                    "https://github.com/rjsears/tftpgui"
                ),
            )
        except Exception:
            print(f"TFTP Server Python3 - Version {VERSION} - Author: {__author__}")

    # ---- Logging wiring -----------------------------------------------------

    def _setup_logging(self) -> None:
        self.log_text = ScrolledText(self, state="disabled", height=12, wrap="word")
        handler = TextHandler(self.log_text)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        for h in list(root_logger.handlers):
            root_logger.removeHandler(h)
        root_logger.addHandler(handler)
        root_logger.setLevel(getattr(logging, self.cfg.log_level.upper(), logging.INFO))

        if self.cfg.log_file:
            try:
                self.cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
                if self.cfg.log_rotation == "time":
                    fhandler = TimedRotatingFileHandler(
                        self.cfg.log_file,
                        when=self.cfg.log_when,
                        interval=self.cfg.log_interval,
                        backupCount=self.cfg.log_backup_count,
                        encoding="utf-8",
                    )
                else:
                    fhandler = RotatingFileHandler(
                        self.cfg.log_file,
                        maxBytes=self.cfg.log_max_bytes,
                        backupCount=self.cfg.log_backup_count,
                        encoding="utf-8",
                    )
                fhandler.setFormatter(formatter)
                root_logger.addHandler(fhandler)
            except Exception as exc:
                print(f"Logging: failed to attach file logger: {exc}")

    # ---- Widgets ------------------------------------------------------------

    def _build_widgets(self) -> None:
        pad = {"padx": 6, "pady": 6}

        # Colored status banner
        self.banner_label = tk.Label(self, text="Server stopped", fg="white", bg="darkred", anchor="center")
        self.banner_label.pack(fill="x")

        frm = ttk.Frame(self)
        frm.pack(fill="x", **pad)

        ttk.Label(frm, text="Host:").grid(row=0, column=0, sticky="w")
        self.host_var = tk.StringVar(value=self.cfg.host)
        ttk.Entry(frm, textvariable=self.host_var, width=16).grid(row=0, column=1, sticky="w")

        ttk.Label(frm, text="Port:").grid(row=0, column=2, sticky="w")
        self.port_var = tk.StringVar(value=str(self.cfg.port))
        ttk.Entry(frm, textvariable=self.port_var, width=8).grid(row=0, column=3, sticky="w")

        ttk.Label(frm, text="Root Directory (set in config):").grid(row=1, column=0, sticky="w")
        self.root_var = tk.StringVar(value=str(self.cfg.root_dir))
        ttk.Entry(frm, textvariable=self.root_var, width=60, state="readonly").grid(row=1, column=1, columnspan=3, sticky="we")
        ttk.Button(frm, text="Open Config…", command=self._open_config).grid(row=1, column=4, sticky="w", padx=4)

        self.write_var = tk.BooleanVar(value=self.cfg.allow_write)
        ttk.Checkbutton(frm, text="Allow Uploads (WRQ)", variable=self.write_var).grid(row=2, column=0, sticky="w")

        ttk.Label(frm, text="Writable Subdirs (comma)").grid(row=2, column=1, sticky="e")
        self.wsubs_var = tk.StringVar(value=",".join(self.cfg.writable_subdirs))
        ttk.Entry(frm, textvariable=self.wsubs_var, width=28).grid(row=2, column=2, columnspan=2, sticky="w")

        self.chroot_var = tk.BooleanVar(value=self.cfg.enforce_chroot)
        ttk.Checkbutton(frm, text="Enforce chroot (root only)", variable=self.chroot_var).grid(row=2, column=4, sticky="w")

        ttk.Label(frm, text="Allowed Extensions (comma)").grid(row=3, column=0, sticky="w")
        self.exts_var = tk.StringVar(value=",".join(self.cfg.filename_allowlist))
        ttk.Entry(frm, textvariable=self.exts_var, width=40).grid(row=3, column=1, columnspan=2, sticky="w")

        ttk.Label(frm, text="Allowlist IPs (comma)").grid(row=3, column=3, sticky="e")
        self.allow_ips_var = tk.StringVar(value=",".join(self.cfg.allowlist_ips))
        ttk.Entry(frm, textvariable=self.allow_ips_var, width=30).grid(row=3, column=4, sticky="w")

        ttk.Label(frm, text="Denylist IPs (comma)").grid(row=4, column=0, sticky="w")
        self.deny_ips_var = tk.StringVar(value=",".join(self.cfg.denylist_ips))
        ttk.Entry(frm, textvariable=self.deny_ips_var, width=40).grid(row=4, column=1, columnspan=2, sticky="w")

        ttk.Label(frm, text="Timeout (s):").grid(row=4, column=3, sticky="e")
        self.timeout_var = tk.StringVar(value=str(self.cfg.timeout_sec))
        ttk.Entry(frm, textvariable=self.timeout_var, width=6).grid(row=4, column=4, sticky="w")

        ttk.Label(frm, text="Max Retries:").grid(row=4, column=5, sticky="e")
        self.retries_var = tk.StringVar(value=str(self.cfg.max_retries))
        ttk.Entry(frm, textvariable=self.retries_var, width=6).grid(row=4, column=6, sticky="w")

        ttk.Label(frm, text="Metrics Window (s):").grid(row=5, column=0, sticky="w")
        self.window_var = tk.StringVar(value=str(self.cfg.metrics_window_sec))
        ttk.Entry(frm, textvariable=self.window_var, width=6).grid(row=5, column=1, sticky="w")

        ttk.Label(frm, text="Log File:").grid(row=5, column=2, sticky="e")
        self.logfile_var = tk.StringVar(value=str(self.cfg.log_file) if self.cfg.log_file else "")
        ttk.Entry(frm, textvariable=self.logfile_var, width=38).grid(row=5, column=3, columnspan=2, sticky="we")
        ttk.Button(frm, text="Choose…", command=self._choose_logfile).grid(row=5, column=5, sticky="w", padx=4)

        ttk.Label(frm, text="Transfer CSV Log:").grid(row=6, column=0, sticky="w")
        self.tlog_var = tk.StringVar(value=str(self.cfg.transfer_log_file) if self.cfg.transfer_log_file else "")
        ttk.Entry(frm, textvariable=self.tlog_var, width=40).grid(row=6, column=1, columnspan=2, sticky="w")

        ttk.Label(frm, text="Rotation:").grid(row=6, column=3, sticky="e")
        self.rotation_var = tk.StringVar(value=self.cfg.log_rotation)
        ttk.Combobox(frm, textvariable=self.rotation_var, values=["size", "time"], width=8, state="readonly").grid(row=6, column=4, sticky="w")

        ttk.Label(frm, text="Max Bytes:").grid(row=6, column=5, sticky="e")
        self.maxbytes_var = tk.StringVar(value=str(self.cfg.log_max_bytes))
        ttk.Entry(frm, textvariable=self.maxbytes_var, width=12).grid(row=6, column=6, sticky="w")

        ttk.Label(frm, text="Backup Count:").grid(row=7, column=0, sticky="w")
        self.backups_var = tk.StringVar(value=str(self.cfg.log_backup_count))
        ttk.Entry(frm, textvariable=self.backups_var, width=6).grid(row=7, column=1, sticky="w")

        ttk.Label(frm, text="When:").grid(row=7, column=2, sticky="e")
        self.when_var = tk.StringVar(value=self.cfg.log_when)
        ttk.Entry(frm, textvariable=self.when_var, width=12).grid(row=7, column=3, sticky="w")

        ttk.Label(frm, text="Interval:").grid(row=7, column=4, sticky="e")
        self.interval_var = tk.StringVar(value=str(self.cfg.log_interval))
        ttk.Entry(frm, textvariable=self.interval_var, width=6).grid(row=7, column=5, sticky="w")

        ttk.Label(frm, text="Audit JSONL:").grid(row=7, column=6, sticky="e")
        self.audit_var = tk.StringVar(value=str(self.cfg.audit_log_file) if self.cfg.audit_log_file else "")
        ttk.Entry(frm, textvariable=self.audit_var, width=28).grid(row=7, column=7, sticky="w")

        ttk.Label(frm, text="Ephemeral Ports (per transfer):").grid(row=8, column=0, sticky="w")
        self.ephemeral_var = tk.BooleanVar(value=self.cfg.ephemeral_ports)
        ttk.Checkbutton(frm, variable=self.ephemeral_var).grid(row=8, column=1, sticky="w")

        btns = ttk.Frame(self)
        btns.pack(fill="x", **pad)
        self.start_btn = ttk.Button(btns, text="Start Server", command=self.start_server)
        self.stop_btn = ttk.Button(btns, text="Stop Server", command=self.stop_server, state="disabled")
        self.reload_btn = ttk.Button(btns, text="Reload Config", command=self._load_config_file)
        self.save_btn = ttk.Button(btns, text="Save Config", command=self._save_config_file)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn.pack(side="left", padx=4)
        self.reload_btn.pack(side="left", padx=4)
        self.save_btn.pack(side="left", padx=4)

        table_frame = ttk.LabelFrame(self, text="Transfers")
        table_frame.pack(fill="both", expand=False, padx=6, pady=4)
        self.tree = ttk.Treeview(
            table_frame,
            columns=("client", "direction", "file", "blk", "opts", "progress", "bytes", "rate_avg", "eta", "status", "hashes"),
            show="headings",
            height=16,
        )
        for col, text, width, anchor in [
            ("client", "Client", 170, "w"),
            ("direction", "Dir", 70, "center"),
            ("file", "File", 300, "w"),
            ("blk", "Blk", 60, "e"),
            ("opts", "Opts", 200, "w"),
            ("progress", "Progress", 100, "e"),
            ("bytes", "Bytes", 170, "e"),
            ("rate_avg", "Avg (B/s)", 110, "e"),
            ("eta", "ETA", 90, "center"),
            ("status", "Status", 140, "w"),
            ("hashes", "MD5/SHA256", 360, "w"),
        ]:
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor=anchor)
        self.tree.pack(fill="x", padx=6, pady=6)

        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)

        self.status = tk.StringVar(value="Stopped")
        status_bar = ttk.Label(self, textvariable=self.status, anchor="w", relief="sunken")
        status_bar.pack(fill="x")

        self.columnconfigure(0, weight=1)
        self._rows: Dict[Tuple[str, int, str, bool], str] = {}

    # ------- Banner helpers -------
    def _set_banner(self, text: str, bg: str) -> None:
        try:
            self.banner_label.config(text=text, bg=bg)
        except Exception:
            pass

    def _update_banner(self) -> None:
        try:
            validate_root_dir(self.cfg)
            if self.running:
                self._set_banner(f"Server running on {self.cfg.host}:{self.cfg.port}", "green")
            else:
                self._set_banner("Server stopped", "darkred")
        except Exception as exc:
            self._set_banner(f"Config error: {exc}", "orange")

    # ------- Polling / event processing -------
    def _schedule_poll(self) -> None:
        if self._closing:
            return
        self._poll_after_id = self.after(200, self._poll_queue)

    def _poll_queue(self) -> None:
        try:
            while True:
                evt = self.event_q.get_nowait()
                self._process_event(evt)
        except queue.Empty:
            pass
        finally:
            if not self._closing:
                self._schedule_poll()

    def _process_event(self, evt: Event) -> None:
        if evt.kind == "server":
            running = (evt.message == "running")
            self.running = running
            self._update_banner()
            self._update_buttons()
            return

        key = (evt.client[0], evt.client[1], evt.filename, evt.is_write)
        iid = self._rows.get(key)
        direction = "UPLOAD" if evt.is_write else "DOWNLOAD"
        hashes = ""
        if evt.hashes:
            md5 = evt.hashes.get("md5", "")
            sha = evt.hashes.get("sha256", "")
            hashes = f"md5={md5} sha256={sha}"
        progress = f"{evt.percent:.1f}%" if evt.total_size > 0 else "—"
        bytes_txt = f"{evt.bytes_done}/{evt.total_size}"
        status = f"{evt.kind}: {evt.message}"

        values = (
            f"{evt.client[0]}:{evt.client[1]}",
            direction,
            evt.filename,
            evt.blk,
            evt.opts,
            progress,
            bytes_txt,
            f"{evt.rate_avg:.0f}",
            evt.eta,
            status,
            hashes,
        )
        if iid is None:
            iid = self.tree.insert("", "end", values=values)
            self._rows[key] = iid
        else:
            for idx, val in enumerate(values):
                self.tree.set(iid, self.tree["columns"][idx], val)

    # ------- Window actions -------
    def on_close(self) -> None:
        self._closing = True
        try:
            if self._poll_after_id:
                self.after_cancel(self._poll_after_id)
        except Exception:
            pass
        try:
            self.stop_server()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    # ------- UI helpers -------
    def _open_config(self) -> None:
        try:
            messagebox.showinfo("Config File", f"Edit this file and set a valid 'root_dir':\n\n{self.cfg.config_file}")
        except Exception:
            print(f"Edit this file and set a valid 'root_dir': {self.cfg.config_file}")

    def _choose_logfile(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".log", initialfile="tftpgui.log")
        if path:
            self.logfile_var.set(path)

    # ------- Server controls -------
    def start_server(self) -> None:
        try:
            self._load_config_file()
            self._set_banner("Starting server…", "blue")
            validate_root_dir(self.cfg)

            host = self.host_var.get().strip() or "0.0.0.0"
            port = int(self.port_var.get())
            self.cfg.host = host
            self.cfg.port = port
            self.cfg.allow_write = bool(self.write_var.get())
            self.cfg.writable_subdirs = [s.strip() for s in self.wsubs_var.get().split(",") if s.strip()]
            self.cfg.enforce_chroot = bool(self.chroot_var.get())
            self.cfg.filename_allowlist = [s.strip().lower() for s in self.exts_var.get().split(",") if s.strip()]
            self.cfg.allowlist_ips = [s.strip() for s in self.allow_ips_var.get().split(",") if s.strip()]
            self.cfg.denylist_ips = [s.strip() for s in self.deny_ips_var.get().split(",") if s.strip()]
            self.cfg.timeout_sec = float(self.timeout_var.get())
            self.cfg.max_retries = int(self.retries_var.get())
            self.cfg.log_file = Path(self.logfile_var.get().strip()) if self.logfile_var.get().strip() else None
            self.cfg.audit_log_file = Path(self.audit_var.get().strip()) if self.audit_var.get().strip() else None
            self.cfg.transfer_log_file = Path(self.tlog_var.get().strip()) if self.tlog_var.get().strip() else None
            self.cfg.log_rotation = self.rotation_var.get().strip().lower() or "size"
            self.cfg.log_max_bytes = int(self.maxbytes_var.get())
            self.cfg.log_backup_count = int(self.backups_var.get())
            self.cfg.log_when = self.when_var.get().strip() or "midnight"
            self.cfg.log_interval = int(self.interval_var.get())
            self.cfg.metrics_window_sec = int(self.window_var.get())
            self.cfg.ephemeral_ports = bool(self.ephemeral_var.get())

            self._setup_logging()

            self.server_thread = ServerThread(self.cfg, self.logger, self._fanout)
            self.server_thread.start()
            self.running = True
            self.status.set(f"Running on {self.cfg.host}:{self.cfg.port} — Root: {self.cfg.root_dir}")
        except Exception as exc:
            try:
                messagebox.showerror("Start Failed", str(exc))
            except Exception:
                print(f"Start Failed: {exc}")
        finally:
            self._update_banner()
            self._update_buttons()

    def stop_server(self) -> None:
        try:
            self.server_thread.stop()
            self.running = False
            self.status.set("Stopped")
        except Exception as exc:
            print(f"Stop Failed: {exc}")
        finally:
            self._update_banner()
            self._update_buttons()

    def _update_buttons(self) -> None:
        try:
            valid = False
            try:
                validate_root_dir(self.cfg)
                valid = True
            except Exception:
                valid = False

            if self._closing:
                return
            if self.running:
                self.start_btn.config(state="disabled")
                self.stop_btn.config(state="normal")
            else:
                self.start_btn.config(state="normal" if valid else "disabled")
                self.stop_btn.config(state="disabled")
        except Exception:
            pass

    def _load_config_file(self) -> None:
        try:
            path = self.cfg.config_file if self.cfg.config_file else resolve_config_path()
            data = json.loads(path.read_text(encoding="utf-8"))
            loaded = ServerConfig.from_json(data)
            loaded.config_file = path
            self.cfg = loaded

            self.host_var.set(self.cfg.host)
            self.port_var.set(str(self.cfg.port))
            self.root_var.set(str(self.cfg.root_dir))
            self.write_var.set(self.cfg.allow_write)
            self.wsubs_var.set(",".join(self.cfg.writable_subdirs))
            self.chroot_var.set(self.cfg.enforce_chroot)
            self.exts_var.set(",".join(self.cfg.filename_allowlist))
            self.allow_ips_var.set(",".join(self.cfg.allowlist_ips))
            self.deny_ips_var.set(",".join(self.cfg.denylist_ips))
            self.timeout_var.set(str(self.cfg.timeout_sec))
            self.retries_var.set(str(self.cfg.max_retries))
            self.logfile_var.set(str(self.cfg.log_file) if self.cfg.log_file else "")
            self.tlog_var.set(str(self.cfg.transfer_log_file) if self.cfg.transfer_log_file else "")
            self.rotation_var.set(self.cfg.log_rotation)
            self.maxbytes_var.set(str(self.cfg.log_max_bytes))
            self.backups_var.set(str(self.cfg.log_backup_count))
            self.when_var.set(self.cfg.log_when)
            self.interval_var.set(str(self.cfg.log_interval))
            self.audit_var.set(str(self.cfg.audit_log_file) if self.cfg.audit_log_file else "")
            self.window_var.set(str(self.cfg.metrics_window_sec))
            self.ephemeral_var.set(self.cfg.ephemeral_ports)
        except Exception as exc:
            self.logger.warning("Failed to load config: %s", exc)
        finally:
            self._update_banner()
            self._update_buttons()

    def _save_config_file(self) -> None:
        try:
            self.cfg.host = self.host_var.get().strip() or "0.0.0.0"
            self.cfg.port = int(self.port_var.get())
            self.cfg.allow_write = bool(self.write_var.get())
            self.cfg.writable_subdirs = [s.strip() for s in self.wsubs_var.get().split(",") if s.strip()]
            self.cfg.enforce_chroot = bool(self.chroot_var.get())
            self.cfg.filename_allowlist = [s.strip().lower() for s in self.exts_var.get().split(",") if s.strip()]
            self.cfg.allowlist_ips = [s.strip() for s in self.allow_ips_var.get().split(",") if s.strip()]
            self.cfg.denylist_ips = [s.strip() for s in self.deny_ips_var.get().split(",") if s.strip()]
            self.cfg.timeout_sec = float(self.timeout_var.get())
            self.cfg.max_retries = int(self.retries_var.get())

            lf = self.logfile_var.get().strip() or None
            self.cfg.log_file = Path(lf) if lf else None

            tlf = self.tlog_var.get().strip() or None
            self.cfg.transfer_log_file = Path(tlf) if tlf else None

            self.cfg.log_rotation = self.rotation_var.get().strip().lower() or "size"
            self.cfg.log_max_bytes = int(self.maxbytes_var.get())
            self.cfg.log_backup_count = int(self.backups_var.get())
            self.cfg.log_when = self.when_var.get().strip() or "midnight"
            self.cfg.log_interval = int(self.interval_var.get())

            alf = self.audit_var.get().strip() or None
            self.cfg.audit_log_file = Path(alf) if alf else None

            self.cfg.metrics_window_sec = int(self.window_var.get())
            self.cfg.ephemeral_ports = bool(self.ephemeral_var.get())

            save_config(self.cfg)
            try:
                messagebox.showinfo("Saved", f"Config saved to {self.cfg.config_file}")
            except Exception:
                print(f"Config saved to {self.cfg.config_file}")
        except Exception as exc:
            try:
                messagebox.showerror("Save Failed", str(exc))
            except Exception:
                print(f"Save Failed: {exc}")
        finally:
            self._update_banner()
            self._update_buttons()


# ---------------------------------------------------------------------------
# Root logger setup (headless mode)
# ---------------------------------------------------------------------------

def configure_root_logging(cfg: ServerConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if cfg.log_file:
        try:
            cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
            if cfg.log_rotation == "time":
                fh = TimedRotatingFileHandler(
                    cfg.log_file,
                    when=cfg.log_when,
                    interval=cfg.log_interval,
                    backupCount=cfg.log_backup_count,
                    encoding="utf-8",
                )
            else:
                fh = RotatingFileHandler(
                    cfg.log_file, maxBytes=cfg.log_max_bytes, backupCount=cfg.log_backup_count, encoding="utf-8"
                )
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
            logging.getLogger().addHandler(fh)
        except Exception as exc:
            print(f"Logging attach failed: {exc}")
# ---------------------------------------------------------------------------
# Web UI (FastAPI) - optional
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TFTP Server Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,'Helvetica Neue',Arial,sans-serif;margin:0;background:#0b1020;color:#e9eef8}
  header{padding:12px 16px;background:#0f1530;border-bottom:1px solid #1a2347}
  .banner{padding:10px 14px;text-align:center;font-weight:600}
  .banner.ok{background:#0f3d0f;color:#e6ffe6}
  .banner.bad{background:#6b1111;color:#fff}
  .wrap{max-width:1100px;margin:18px auto;padding:0 16px}
  table{border-collapse:collapse;width:100%;background:#111833;border:1px solid #1f2a55}
  th,td{padding:8px 10px;border-bottom:1px solid #1f2a55}
  th{background:#0f1733;text-align:left}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace}
  .muted{opacity:.8}
  .pill{display:inline-block;padding:2px 8px;border-radius:999px;background:#203066}
  .right{float:right}
</style>
</head>
<body>
<header>
  <div id="topline">
    <strong>TFTP Server</strong>
    <span id="subtitle" class="muted"></span>
    <span class="right muted"><a href="/api/status" style="color:#9fb7ff;text-decoration:none">/api/status</a></span>
  </div>
</header>
<div id="banner" class="banner bad">Server stopped</div>

<div class="wrap">
  <h3>Active & Recent Transfers</h3>
  <table id="tbl">
    <thead>
      <tr>
        <th>Client</th>
        <th>Dir</th>
        <th>File</th>
        <th>Blk</th>
        <th>Opts</th>
        <th>Progress</th>
        <th>Bytes</th>
        <th>Avg (B/s)</th>
        <th>ETA</th>
        <th>Status</th>
        <th>Hashes</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>
</div>

<script>
const banner = document.getElementById('banner');
const subtitle = document.getElementById('subtitle');
const tbody = document.querySelector('#tbl tbody');

function setBanner(running, host, port){
  banner.textContent = running ? `Server running on ${host}:${port}` : 'Server stopped';
  banner.className = 'banner ' + (running ? 'ok' : 'bad');
}

function refreshStatus(){
  fetch('/api/status').then(r=>r.json()).then(s=>{
    setBanner(!!s.running, s.host, s.port);
    subtitle.textContent = ` root: ${s.root_dir}`;
  }).catch(()=>{});
}

function up(evt){
  const key = `${evt.client[0]}:${evt.client[1]}|${evt.filename}|${evt.is_write}`;
  let tr = document.querySelector(`tr[data-key="${key}"]`);
  const direction = evt.is_write ? 'UPLOAD' : 'DOWNLOAD';
  const hashes = evt.hashes ? `md5=${evt.hashes.md5||''} sha256=${evt.hashes.sha256||''}` : '';
  const progress = evt.total_size > 0 ? `${evt.percent.toFixed(1)}%` : '—';
  const bytes = `${evt.bytes_done}/${evt.total_size||0}`;
  const values = [
    `${evt.client[0]}:${evt.client[1]}`,
    direction,
    evt.filename,
    evt.blk,
    evt.opts || '',
    progress,
    bytes,
    Math.round(evt.rate_avg).toString(),
    evt.eta || '—',
    `${evt.kind}: ${evt.message||''}`,
    hashes
  ];
  if(!tr){
    tr = document.createElement('tr');
    tr.setAttribute('data-key', key);
    values.forEach(v=>{
      const td = document.createElement('td');
      td.textContent = v;
      tr.appendChild(td);
    });
    tbody.prepend(tr);
  }else{
    Array.from(tr.children).forEach((td, i)=> td.textContent = values[i]);
  }
}

function connectSSE(){
  const es = new EventSource('/api/events');
  es.onmessage = (ev)=>{
    try{
      const evt = JSON.parse(ev.data);
      if(evt.kind === 'server'){
        const running = evt.message === 'running';
        setBanner(running, evt.client[0], evt.client[1]);
        return;
      }
      up(evt);
    }catch(e){}
  };
  es.onerror = ()=>{ setTimeout(connectSSE, 1500); };
}

refreshStatus();
setInterval(refreshStatus, 2000);
connectSSE();
</script>
</body>
</html>
"""

def create_web_app(cfg: ServerConfig, fanout: EventFanout, server: "ServerThread"):
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed")

    app = FastAPI(title="TFTP Server UI", version=VERSION)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return DASHBOARD_HTML

    @app.get("/api/status", response_class=JSONResponse)
    def api_status():
        return {
            "running": bool(getattr(server, "is_running", False)),
            "host": str(cfg.host),
            "port": int(cfg.port),
            "root_dir": str(cfg.root_dir),
            "metrics_window_sec": int(cfg.metrics_window_sec),
            "allow_write": bool(cfg.allow_write),
            "ephemeral_ports": bool(cfg.ephemeral_ports),
        }

    @app.get("/api/events")
    def api_events():
        # Server-Sent Events stream of recent + future events
        async def gen():
            # 1) replay recent buffered
            for e in list(fanout.recent):
                yield f"data: {json.dumps(_event_to_json(e))}\n\n"
            # 2) live tail
            q = fanout.register()
            try:
                while True:
                    e = await asyncio.get_event_loop().run_in_executor(None, q.get)
                    yield f"data: {json.dumps(_event_to_json(e))}\n\n"
            except asyncio.CancelledError:
                return

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def _event_to_json(e: Event) -> dict:
    return {
        "kind": e.kind,
        "client": [e.client[0], e.client[1]],
        "filename": e.filename,
        "is_write": e.is_write,
        "bytes_done": e.bytes_done,
        "total_size": e.total_size,
        "percent": e.percent,
        "rate_avg": e.rate_avg,
        "eta": e.eta,
        "blk": e.blk,
        "opts": e.opts,
        "message": e.message,
        "hashes": e.hashes or {},
    }


def run_web(app, host: str, port: int) -> None:
    if uvicorn is None:
        raise RuntimeError("uvicorn not installed")
    uvicorn.run(app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# CLI + entrypoint
# ---------------------------------------------------------------------------

def configure_root_logging(cfg: ServerConfig) -> None:
    """(Re-declared here in case module order shifts)"""
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if cfg.log_file:
        try:
            cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
            if cfg.log_rotation == "time":
                fh = TimedRotatingFileHandler(
                    cfg.log_file,
                    when=cfg.log_when,
                    interval=cfg.log_interval,
                    backupCount=cfg.log_backup_count,
                    encoding="utf-8",
                )
            else:
                fh = RotatingFileHandler(
                    cfg.log_file, maxBytes=cfg.log_max_bytes, backupCount=cfg.log_backup_count, encoding="utf-8"
                )
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
            logging.getLogger().addHandler(fh)
        except Exception as exc:
            print(f"Logging attach failed: {exc}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tkinter TFTP server with progress, audit, CSV transfer logging, and optional web UI."
    )
    p.add_argument("--config", "-c", help="Path to a config JSON file to use/initialize.")
    p.add_argument("--headless", action="store_true", help="Run in server-only mode without launching the GUI.")
    p.add_argument("--web", "-w", action="store_true", help="Start the built-in web UI.")
    p.add_argument("--web-port", "-p", type=int, help="Override the web UI port (default from config).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = resolve_config_path(args.config)
    cfg = load_config(cfg_path)

    # derive desired web state
    start_web = bool(args.web or cfg.web.get("enabled", False))
    if args.web_port is not None:
        cfg.web["port"] = int(args.web_port)

    # HEADLESS if Tk is missing or --headless requested
    if (tk is None) or bool(args.headless):
        try:
            validate_root_dir(cfg)
            validate_transfer_port_range(cfg)
        except Exception as exc:
            print(f"ERROR: {exc}")
            print(f"Edit this file and set 'root_dir': {cfg.config_file}")
            return

        configure_root_logging(cfg)
        logger = logging.getLogger("tftpgui")
        fanout = EventFanout()
        print_q = fanout.register()

        server = ServerThread(cfg, logger, fanout)
        server.start()

        # Optionally run web UI
        if start_web and FastAPI is not None:
            try:
                app_web = create_web_app(cfg, fanout, server)
                threading.Thread(
                    target=run_web,
                    args=(app_web, str(cfg.web.get("host", "0.0.0.0")), int(cfg.web.get("port", 8080))),
                    name="TFTP-Web",
                    daemon=True,
                ).start()
                print(f"Web UI starting on {cfg.web.get('host', '0.0.0.0')}:{cfg.web.get('port', 8080)}")
            except Exception as e:
                print(f"Web UI failed to start: {e}")
        elif start_web and FastAPI is None:
            print("Web UI requested but FastAPI/uvicorn not installed.")

        print(f"TFTP server running with config: {cfg.config_file}. Press Ctrl+C to stop.")
        try:
            while True:
                try:
                    evt = print_q.get(timeout=0.5)
                    if evt.kind == "server":
                        # status line
                        running = (evt.message == "running")
                        state = "RUNNING" if running else "STOPPED"
                        print(f"[server] {state} on {evt.client[0]}:{evt.client[1]}")
                        continue
                    direction = "UPLOAD" if evt.is_write else "DOWNLOAD"
                    prog_txt = f"{evt.percent:.1f}%" if evt.total_size > 0 else "—"
                    hashes = ""
                    if evt.hashes:
                        hashes = f" md5={evt.hashes.get('md5','')} sha256={evt.hashes.get('sha256','')}"
                    print(
                        f"[{evt.kind}] {direction} {evt.client} {evt.filename} {prog_txt} "
                        f"{evt.bytes_done}/{evt.total_size} avg={evt.rate_avg:.0f}B/s ETA={evt.eta} "
                        f"blk={evt.blk} opts={evt.opts} - {evt.message}{hashes}"
                    )
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            print("Stopping...")
            server.stop()
        return

    # GUI mode (Tk available and --headless not set)
    app = TFTPApp(cfg, start_web=start_web)
    app.mainloop()
    try:
        app.stop_server()
    except Exception:
        pass


if __name__ == "__main__":
    main()
