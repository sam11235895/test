"""Automation tool for collecting diagnostic information from Juniper switches.

This module connects to one or more Juniper devices, runs a predefined set of
diagnostic commands, stores the raw outputs for auditing purposes, and performs
simple heuristic checks to highlight potential anomalies.

Example usage::

    python juniper_audit.py --inventory inventory.yaml --output-dir outputs

The inventory file is expected to be a YAML document with the following
structure::

    devices:
      - host: 172.31.84.200
        username: root
        password: hauman@12223097
        device_name: access-switch-01
        port: 22  # optional, defaults to 22

All command output is written to ``<output-dir>/<device_name>.log`` and a
summary report is produced in JSON format describing any anomalies detected.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

try:  # pragma: no cover - exercised indirectly
    import yaml
except ModuleNotFoundError:  # pragma: no cover - only when PyYAML missing
    yaml = None  # type: ignore[assignment]

try:  # pragma: no cover - exercised indirectly
    from netmiko import ConnectHandler
    from netmiko.ssh_exception import NetmikoAuthenticationException, NetmikoTimeoutException
except ModuleNotFoundError:  # pragma: no cover - only when netmiko missing
    ConnectHandler = None  # type: ignore[assignment]

    class NetmikoAuthenticationException(Exception):
        """Fallback exception when netmiko is not installed."""

    class NetmikoTimeoutException(Exception):
        """Fallback exception when netmiko is not installed."""


# Commands provided by the user.
COMMANDS: List[str] = [
    "cli",
    "set cli screen-length 0",
    "show version",
    "show chassis hardware",
    "show chassis environment",
    "show chassis alarms",
    "show system alarms",
    "show system processes extensive | no-more",
    "show configuration | display set | no-more",
    "show interface terse | no-more",
    "show ethernet-switching interface | no-more",
    "show vlan | no-more",
    "show lldp nei",
    "show route",
    "show log messages",
]

# Commands whose outputs should not trigger an alarm if they contain keywords
# such as "down". These commands are mostly informational and the heuristic
# checks would generate excessive false positives if applied directly.
_SKIP_KEYWORD_CHECK_COMMANDS = {
    "show interface terse | no-more",
    "show ethernet-switching interface | no-more",
    "show vlan | no-more",
    "show lldp nei",
    "show route",
}

# Heuristic patterns used to flag suspicious lines in command output.
DEFAULT_ANOMALY_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "error": re.compile(r"\berror\b", re.IGNORECASE),
    "fail": re.compile(r"\bfail(?:ed|ure)?\b", re.IGNORECASE),
    "alarm": re.compile(r"\balarm\b", re.IGNORECASE),
    "down": re.compile(r"\bdown\b", re.IGNORECASE),
    "critical": re.compile(r"\bcritical\b", re.IGNORECASE),
}

# Commands for which the absence of "No alarms currently active" should raise a flag.
_EXPECT_NO_ALARMS = {
    "show chassis alarms",
    "show system alarms",
}


@dataclass
class DeviceConfig:
    """Simple container describing how to connect to a device."""

    host: str
    username: str
    password: str
    device_name: str
    port: int = 22

    def to_netmiko_dict(self) -> Dict[str, object]:
        return {
            "device_type": "juniper",
            "host": self.host,
            "username": self.username,
            "password": self.password,
            "port": self.port,
            "fast_cli": False,
        }


class AuditError(Exception):
    """Custom exception used for audit-related errors."""


def _load_inventory_document(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - only when invalid
        raise AuditError(
            "PyYAML is not installed and the inventory file is not valid JSON; "
            "install PyYAML or provide JSON-formatted inventory"
        ) from exc


def load_inventory(path: Path) -> List[DeviceConfig]:
    """Load device definitions from a YAML inventory file."""

    try:
        data = _load_inventory_document(path)
    except FileNotFoundError as exc:  # pragma: no cover - pass through
        raise AuditError(f"Inventory file not found: {path}") from exc

    if not isinstance(data, MutableMapping) or "devices" not in data:
        raise AuditError("Inventory file must contain a top-level 'devices' list")

    devices_raw = data["devices"]
    if not isinstance(devices_raw, Iterable):
        raise AuditError("'devices' must be a list of device definitions")

    devices: List[DeviceConfig] = []
    for entry in devices_raw:
        if not isinstance(entry, Mapping):
            raise AuditError("Each device entry must be a mapping")

        try:
            device = DeviceConfig(
                host=str(entry["host"]),
                username=str(entry["username"]),
                password=str(entry["password"]),
                device_name=str(entry.get("device_name", entry["host"])),
                port=int(entry.get("port", 22)),
            )
        except KeyError as exc:
            raise AuditError(f"Missing required field in device entry: {exc}") from exc
        except ValueError as exc:
            raise AuditError(f"Invalid value in device entry: {exc}") from exc

        devices.append(device)

    if not devices:
        raise AuditError("Inventory does not contain any devices")

    return devices


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _current_timestamp() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _analyze_command_output(command: str, output: str) -> List[str]:
    """Analyze a single command's output and return a list of anomaly messages."""

    anomalies: List[str] = []

    if command in _EXPECT_NO_ALARMS:
        if "No alarms currently active" not in output:
            anomalies.append("Alarms reported")

    if command not in _SKIP_KEYWORD_CHECK_COMMANDS:
        for name, pattern in DEFAULT_ANOMALY_PATTERNS.items():
            if pattern.search(output):
                anomalies.append(f"Keyword '{name}' detected")

    return anomalies


def analyze_outputs(outputs: Mapping[str, str]) -> Dict[str, List[str]]:
    """Analyze the outputs of multiple commands.

    Args:
        outputs: Mapping of command string to its raw output.

    Returns:
        A dictionary mapping each command to a list of anomaly messages.
    """

    return {
        command: _analyze_command_output(command, output)
        for command, output in outputs.items()
    }


def _collect_command_outputs(connection: ConnectHandler, commands: Iterable[str]) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    for command in commands:
        outputs[command] = connection.send_command(command, expect_string=r"[#>]", read_timeout=90)
    return outputs


def _format_device_output(device: DeviceConfig, outputs: Mapping[str, str]) -> str:
    sections = [f"# Device: {device.device_name} ({device.host})"]
    for command, output in outputs.items():
        sections.append(f"\n$ {command}\n{output.strip()}\n")
    return "\n".join(sections)


def _write_device_log(output_dir: Path, device: DeviceConfig, content: str) -> Path:
    filename = f"{device.device_name}_{_current_timestamp()}.log"
    path = output_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


def _write_summary(output_dir: Path, summary: Mapping[str, object]) -> Path:
    filename = f"summary_{_current_timestamp()}.json"
    path = output_dir / filename
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path


def audit_device(device: DeviceConfig, output_dir: Path) -> Dict[str, object]:
    """Connect to a device, run commands, and store outputs.

    Returns a dictionary containing metadata about the audit, including the
    path to the raw log file and any anomalies that were detected.
    """

    if ConnectHandler is None:
        raise AuditError(
            "netmiko is required to connect to devices. Install it with 'pip install netmiko'."
        )

    logging.info("Connecting to %s (%s)", device.device_name, device.host)
    try:
        with ConnectHandler(**device.to_netmiko_dict()) as connection:
            outputs = _collect_command_outputs(connection, COMMANDS)
    except (NetmikoAuthenticationException, NetmikoTimeoutException) as exc:
        logging.error("Failed to connect to %s: %s", device.device_name, exc)
        return {
            "device": device.device_name,
            "host": device.host,
            "log_file": None,
            "anomalies": {"connection": [str(exc)]},
            "success": False,
        }

    formatted_output = _format_device_output(device, outputs)
    log_path = _write_device_log(output_dir, device, formatted_output)
    anomalies = analyze_outputs(outputs)
    success = all(not issues for issues in anomalies.values())

    return {
        "device": device.device_name,
        "host": device.host,
        "log_file": str(log_path),
        "anomalies": anomalies,
        "success": success,
    }


def run_audit(
    inventory: Path,
    output_dir: Path,
    max_workers: int = 5,
) -> Dict[str, object]:
    """Run the audit across every device in the inventory."""

    devices = load_inventory(inventory)
    _ensure_output_dir(output_dir)

    results: List[Dict[str, object]] = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_device = {
            executor.submit(audit_device, device, output_dir): device
            for device in devices
        }

        for future in as_completed(future_to_device):
            device = future_to_device[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - unexpected failure
                logging.exception("Unhandled error auditing %s: %s", device.device_name, exc)
                result = {
                    "device": device.device_name,
                    "host": device.host,
                    "log_file": None,
                    "anomalies": {"unexpected": [str(exc)]},
                    "success": False,
                }

            with lock:
                results.append(result)

    summary = {
        "inventory": str(inventory),
        "output_dir": str(output_dir),
        "timestamp": _current_timestamp(),
        "results": results,
    }

    _write_summary(output_dir, summary)
    return summary


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Juniper switches and store diagnostics")
    parser.add_argument(
        "--inventory",
        required=True,
        type=Path,
        help="Path to the inventory YAML file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where logs and summaries will be written",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of concurrent device connections",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run_audit(args.inventory, args.output_dir, max_workers=args.max_workers)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
