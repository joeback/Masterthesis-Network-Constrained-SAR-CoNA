from pathlib import Path
import requests
from google.transit import gtfs_realtime_pb2
import pandas as pd
import time
from datetime import datetime, timezone
import os
from zoneinfo import ZoneInfo   # Python 3.9+

# --- Config ---
URL = "https://production.gtfsrt.vbb.de/data"
PROJECT_ROOT = Path("/Users/joeback/Documents/Joe Uni/Masterarbeit/Data")

DATA_DIR = PROJECT_ROOT / "gtfs_data"
LOG_DIR  = PROJECT_ROOT / "gtfs_logs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE_BASE = DATA_DIR / "vbb_realtime_delays_buses"
LOG_FILE = LOG_DIR / "collector_status.log"

INTERVAL_SEC = 1800  # 30 minutes

print(f"Data will be written to: {OUT_FILE_BASE}", flush=True)
print(f"Log will be written to: {LOG_FILE}", flush=True)

BERLIN = ZoneInfo("Europe/Berlin")

def time_of_day_bin(dt):
    hour = dt.hour
    if 0 <= hour < 6:
        return "0-6"
    elif 6 <= hour < 12:
        return "6-12"
    elif 12 <= hour < 18:
        return "12-18"
    else:
        return "18-24"

print("Starte GTFS-Realtime-Collector (nur Buslinien 3/700)... (STRG+C zum Stoppen)")

while True:
    try:
        response = requests.get(URL, timeout=10)
        response.raise_for_status()

        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(response.content)

        trip_updates = []

        # Timestamp of the request
        ts_utc = datetime.now(timezone.utc)
        ts_berlin = ts_utc.astimezone(BERLIN)

        OUT_FILE = OUT_FILE_BASE.with_name(
            f"{OUT_FILE_BASE.name}_{ts_berlin.date()}.csv"
        )
        ts_iso = ts_utc.isoformat()

        tod_label = time_of_day_bin(ts_berlin)

        for entity in feed.entity:
            if not entity.HasField("trip_update"):
                continue

            trip = entity.trip_update.trip
            route_id = trip.route_id.lower() if trip.route_id else ""

            # Filter: only bus lines containing "3/" or "700"
            if not ("3/" in route_id or "700" in route_id):
                continue

            for i, stu in enumerate(entity.trip_update.stop_time_update):
                delay = None
                if stu.HasField("arrival") and stu.arrival.HasField("delay"):
                    delay = stu.arrival.delay
                elif stu.HasField("departure") and stu.departure.HasField("delay"):
                    delay = stu.departure.delay

                trip_updates.append({
                    "timestamp_utc": ts_utc.isoformat(),
                    "timestamp": ts_berlin.isoformat(),
                    "time_bin": tod_label,
                    "route_id": route_id,
                    "trip_id": trip.trip_id,
                    "stop_id": stu.stop_id,
                    "stop_order": i,
                    "delay_seconds": delay
                })

        if trip_updates:
            df = pd.DataFrame(trip_updates)

            write_header = not os.path.exists(OUT_FILE)

            df.to_csv(
                OUT_FILE,
                mode="a",
                index=False,
                header=write_header,
            )

            print(
                f"[{ts_berlin.strftime('%Y-%m-%d %H:%M:%S %Z')}] "
                f"Gespeichert: {len(df)} Einträge (Bin: {tod_label})"
            )

            with open(LOG_FILE, "a") as f:
                f.write(
                    f"{ts_berlin.isoformat()} | "
                    f"rows={len(df)} | "
                    f"bin={tod_label}\n"
                )

        time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("Beende Collector.")
        break

    except Exception as e:
        print(f"Fehler: {e}")
        time.sleep(INTERVAL_SEC)