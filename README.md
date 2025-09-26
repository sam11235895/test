# Juniper Switch Audit Tool

This repository provides a Python utility for collecting diagnostic
information from multiple Juniper switches. The script connects to each device
listed in an inventory file, executes a predefined set of operational
commands, stores the raw outputs for later reference, and applies simple
heuristic checks to highlight potential anomalies.

## Features

- Executes a comprehensive set of Junos CLI commands, including hardware,
  environment, alarm, interface, VLAN, LLDP, routing, and log inspections.
- Stores all command output per device in timestamped log files.
- Generates a JSON summary that records success/failure status and any
  anomalies detected for each command.
- Supports concurrent execution against multiple switches for faster data
  collection.

## Requirements

Install the required Python dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** The tool uses [`netmiko`](https://github.com/ktbyers/netmiko) to
> communicate with the switches. Ensure that outbound SSH connectivity to your
> devices is permitted from the host running the script.

## Preparing the Inventory

The script expects a YAML file describing the switches you want to audit. A
minimal example is shown below:

```yaml
devices:
  - host: 192.0.2.10
    username: netadmin
    password: SuperSecret
    device_name: access-switch-01
    port: 22  # optional, defaults to 22
  - host: 192.0.2.11
    username: netadmin
    password: SuperSecret
    device_name: access-switch-02
```

## Running the Audit

```bash
python juniper_audit.py --inventory inventory.yaml --output-dir outputs
```

Command-line arguments:

- `--inventory`: Path to the inventory YAML file.
- `--output-dir`: Directory where per-device logs and the JSON summary will be
  stored. The directory is created automatically if it does not already exist.
- `--max-workers`: (Optional) Maximum number of concurrent device sessions.
  Defaults to `5`.
- `--log-level`: (Optional) Logging verbosity. Defaults to `INFO`.

## Output Artifacts

For each switch the tool creates a timestamped log file containing every
command executed and its output. A JSON summary file is also generated with the
overall status of each device and any anomalies detected (e.g., alarms
reported, error keywords found, or connection failures).

## Testing

Unit tests mock the network connections and validate the parsing and anomaly
logic. Run the test suite with:

```bash
pytest
```

## Disclaimer

The anomaly detection implemented here is intentionally conservative and based
on simple keyword checks. Review the generated logs to confirm the status of
your devices and adjust the heuristics to fit your operational requirements.
