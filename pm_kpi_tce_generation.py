#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["requests>=2.31"]
# ///

"""
Network KPI + Alert Generator for VictoriaMetrics
=================================================
✅ Compatible with Python 3.13.7

- Generates synthetic PM & KPI data
- Uploads to VictoriaMetrics
- Evaluates alerts (Critical, Major, Minor, Warning)
- Saves alert (TCE) events to file and optionally pushes them
"""

from __future__ import annotations
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
OUTPUT_FILE = Path("network_pm_and_kpi.prom")
ALERT_FILE = Path("tce_alerts.prom")
VICTORIA_URL = "http://localhost:8428/api/v1/import/prometheus"

NUM_CELLS = 5
NUM_SAMPLES = 100  # per metric per cell
START_TIME = datetime.now(timezone.utc) - timedelta(hours=1)

# Base PM Counters
METRICS = {
    "throughput_mbps": (50, 200),
    "packet_loss_rate": (0.0, 2.0),
    "latency_ms": (10, 120),
    "cell_availability": (95.0, 100.0),
    "handover_success_rate": (80.0, 100.0),
}

# KPI Alert Thresholds (tune as needed)
ALERT_THRESHOLDS = {
    "cell_efficiency": {
        "critical": 40.0,
        "major": 55.0,
        "minor": 70.0,
        "warning": 85.0,
    },
    "network_quality_index": {
        "critical": 40.0,
        "major": 60.0,
        "minor": 75.0,
        "warning": 90.0,
    },
    "availability_score": {
        "critical": 60.0,
        "major": 75.0,
        "minor": 85.0,
        "warning": 95.0,
    },
}

# -----------------------------------------------------------------------------
# PM + KPI Generation
# -----------------------------------------------------------------------------
def generate_pm_data() -> list[dict[str, float | str | datetime]]:
    samples: list[dict[str, float | str | datetime]] = []
    for cell_id in range(1, NUM_CELLS + 1):
        timestamp = START_TIME
        for _ in range(NUM_SAMPLES):
            record: dict[str, float | str | datetime] = {
                "cell": f"cell_{cell_id}",
                "timestamp": timestamp,
            }
            for metric, (min_val, max_val) in METRICS.items():
                record[metric] = round(random.uniform(min_val, max_val), 2)
            samples.append(record)
            timestamp += timedelta(seconds=30)
    return samples


def derive_kpis(pm_data: list[dict[str, float | str | datetime]]) -> list[dict[str, float | str | datetime]]:
    for record in pm_data:
        throughput = float(record["throughput_mbps"])
        loss = float(record["packet_loss_rate"])
        latency = float(record["latency_ms"])
        avail = float(record["cell_availability"])
        handover = float(record["handover_success_rate"])

        record["cell_efficiency"] = round((throughput / 200) * 100, 2)
        record["network_quality_index"] = round((100 - loss) * (100 / latency), 2)
        record["availability_score"] = round((avail * handover) / 100, 2)
    return pm_data


# -----------------------------------------------------------------------------
# Alerts (TCE) Generation
# -----------------------------------------------------------------------------
def evaluate_alerts(kpi_data: list[dict[str, float | str | datetime]]) -> list[str]:
    """Generate alert events (TCEs) when KPI crosses thresholds."""
    alerts: list[str] = []
    for record in kpi_data:
        timestamp = int(record["timestamp"].timestamp() * 1000)
        cell = record["cell"]

        for kpi, thresholds in ALERT_THRESHOLDS.items():
            value = float(record[kpi])
            severity = None

            # Assign severity based on thresholds
            if value < thresholds["critical"]:
                severity = "critical"
            elif value < thresholds["major"]:
                severity = "major"
            elif value < thresholds["minor"]:
                severity = "minor"
            elif value < thresholds["warning"]:
                severity = "warning"

            if severity:
                alert_line = (
                    f'tce_event{{cell="{cell}", kpi="{kpi}", severity="{severity}"}} {value} {timestamp}'
                )
                alerts.append(alert_line)

    return alerts


# -----------------------------------------------------------------------------
# Formatting & Export
# -----------------------------------------------------------------------------
def format_prometheus_lines(data: list[dict[str, float | str | datetime]]) -> list[str]:
    lines: list[str] = []
    for record in data:
        ts = int(record["timestamp"].timestamp() * 1000)
        cell = record["cell"]
        for metric, value in record.items():
            if metric in {"cell", "timestamp"}:
                continue
            lines.append(f'{metric}{{cell="{cell}"}} {value} {ts}')
    return lines


def save_to_file(lines: list[str], path: Path) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {len(lines)} lines to {path.resolve()}")


def push_to_victoriametrics(lines: list[str], url: str = VICTORIA_URL) -> None:
    """Push Prometheus lines to VictoriaMetrics via HTTP POST."""
    data = "\n".join(lines)
    try:
        resp = requests.post(url, data=data.encode("utf-8"), timeout=10)
        resp.raise_for_status()
        print(f"Pushed {len(lines)} samples to VictoriaMetrics at {url}")
    except requests.RequestException as e:
        print(f"Push failed: {e}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("Generating PM data...")
    pm_data = generate_pm_data()

    print("Deriving KPIs...")
    kpi_data = derive_kpis(pm_data)

    print("Evaluating alerts (TCEs)...")
    alerts = evaluate_alerts(kpi_data)

    print("Formatting metrics and alerts...")
    kpi_lines = format_prometheus_lines(kpi_data)

    save_to_file(kpi_lines, OUTPUT_FILE)
    save_to_file(alerts, ALERT_FILE)

    print("Uploading all metrics to VictoriaMetrics...")
    push_to_victoriametrics(kpi_lines + alerts)

    print("All done! KPIs + Alerts generated successfully.")


if __name__ == "__main__":
    main()