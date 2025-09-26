import json
import sys
from pathlib import Path
from typing import Dict

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import juniper_audit as audit


def make_inventory(tmp_path: Path, devices: Dict[str, Dict[str, object]]) -> Path:
    inventory = {"devices": list(devices.values())}
    path = tmp_path / "inventory.yaml"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    return path


def test_load_inventory(tmp_path: Path) -> None:
    inventory_path = make_inventory(
        tmp_path,
        {
            "device1": {
                "host": "192.0.2.1",
                "username": "user",
                "password": "pass",
                "device_name": "switch-1",
                "port": 2201,
            }
        },
    )

    devices = audit.load_inventory(inventory_path)

    assert len(devices) == 1
    device = devices[0]
    assert device.host == "192.0.2.1"
    assert device.username == "user"
    assert device.password == "pass"
    assert device.device_name == "switch-1"
    assert device.port == 2201


def test_load_inventory_missing_devices(tmp_path: Path) -> None:
    path = tmp_path / "inventory.yaml"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(audit.AuditError):
        audit.load_inventory(path)


def test_analyze_outputs_detects_keywords() -> None:
    outputs = {
        "show chassis alarms": "Major alarm detected",
        "show version": "error found in module",
        "show interface terse | no-more": "ge-0/0/0 up",
    }

    anomalies = audit.analyze_outputs(outputs)

    assert "Alarms reported" in anomalies["show chassis alarms"]
    assert any("error" in msg for msg in anomalies["show version"])
    # Keyword checks are skipped for show interface terse
    assert anomalies["show interface terse | no-more"] == []


def test_audit_device_handles_connection_errors(monkeypatch, tmp_path: Path) -> None:
    class FakeConnect:
        def __init__(self, **_: object) -> None:
            raise audit.NetmikoTimeoutException("timeout")

    monkeypatch.setattr(audit, "ConnectHandler", FakeConnect)

    device = audit.DeviceConfig(
        host="192.0.2.1",
        username="user",
        password="pass",
        device_name="switch-1",
    )

    result = audit.audit_device(device, tmp_path)

    assert result["success"] is False
    assert "connection" in result["anomalies"]
    assert result["log_file"] is None


def test_run_audit_writes_summary(monkeypatch, tmp_path: Path) -> None:
    inventory_path = make_inventory(
        tmp_path,
        {
            "device1": {
                "host": "192.0.2.1",
                "username": "user",
                "password": "pass",
                "device_name": "switch-1",
            }
        },
    )

    outputs = {command: "No alarms currently active" for command in audit.COMMANDS}

    class FakeConnect:
        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self) -> "FakeConnect":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def send_command(self, command: str, expect_string: str, read_timeout: int) -> str:
            return outputs[command]

    monkeypatch.setattr(audit, "ConnectHandler", FakeConnect)

    summary = audit.run_audit(inventory_path, tmp_path / "outputs", max_workers=1)

    assert summary["results"][0]["success"] is True
    files = list((tmp_path / "outputs").glob("summary_*.json"))
    assert files, "summary file not written"
