import json
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import urlopen

DAYS = 30
MONITORS_URL = "https://api.uptimerobot.com/v2/getMonitors"
STATUS_NAMES = {0: "paused", 1: "pending", 2: "up", 8: "warning", 9: "down"}


def get_uptime():
    now = datetime.now(timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    first_day = today - timedelta(days=DAYS - 1)
    days = [first_day + timedelta(days=i) for i in range(DAYS)]

    windows = [(day, min(day + timedelta(days=1), now)) for day in days]
    windows.append((first_day, now))
    ranges = "-".join(f"{int(start.timestamp())}_{int(end.timestamp())}" for start, end in windows)

    body = urlencode(
        {
            "api_key": os.environ["UPTIMEROBOT_MONITOR_KEY"],
            "format": "json",
            "custom_uptime_ranges": ranges,
        }
    ).encode("ascii")

    with urlopen(MONITORS_URL, data=body, timeout=8) as response:
        payload = json.load(response)

    monitors = payload.get("monitors", [])
    if payload.get("stat") != "ok" or len(monitors) != 1:
        raise ValueError("Expected one UptimeRobot monitor.")

    monitor = monitors[0]
    status = STATUS_NAMES.get(monitor["status"], "unknown")
    created_at = int(monitor.get("create_datetime") or 0)
    raw_ratios = str(monitor.get("custom_uptime_ranges", "")).split("-")
    ratios = []

    if len(raw_ratios) == len(windows):
        for raw, (_, end) in zip(raw_ratios, windows):
            try:
                ratio = float(raw)
            except ValueError:
                ratio = None

            if (
                ratio is None
                or not 0 <= ratio <= 100
                or not created_at
                or end.timestamp() <= created_at
            ):
                ratio = None

            ratios.append(ratio)
    else:
        ratios = [None] * len(windows)

    return {
        "status": status,
        "uptime": ratios[-1] if status != "pending" else None,
        "days": [
            {"date": day.date().isoformat(), "uptime": ratio if status != "pending" else None}
            for day, ratio in zip(days, ratios)
        ],
    }
