import requests
from google.transit import gtfs_realtime_pb2
import pandas as pd
import time
from datetime import datetime
import os

# --- Konfiguration ---
URL = "https://production.gtfsrt.vbb.de/data"
OUT_FILE = "vbb_realtime_delays_buses.csv"
INTERVAL_SEC = 1800   # Abrufintervall (z. B. alle 30 Minuten)

print("Starte GTFS-Realtime-Collector (nur Buslinien 3/700)... (STRG+C zum Stoppen)")

# --- Endlosschleife zum periodischen Abruf ---
while True:
    try:
        # 1. Feed laden
        response = requests.get(URL, timeout=10)
        response.raise_for_status()

        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(response.content)

        # 2. Daten extrahieren
        trip_updates = []
        timestamp = datetime.utcnow().isoformat()

        for entity in feed.entity:
            if not entity.HasField("trip_update"):
                continue

            trip = entity.trip_update.trip
            route_id = trip.route_id.lower() if trip.route_id else ""

            # ✅ Filter: Nur Buslinien mit Tag "3/" oder "700"
            if not ("3/" in route_id or "700" in route_id):
                continue

            for stu in entity.trip_update.stop_time_update:
                delay = None
                if stu.HasField("arrival") and stu.arrival.HasField("delay"):
                    delay = stu.arrival.delay
                elif stu.HasField("departure") and stu.departure.HasField("delay"):
                    delay = stu.departure.delay

                trip_updates.append({
                    "timestamp": timestamp,     # Zeitpunkt des Abrufs
                    "route_id": route_id,
                    "trip_id": trip.trip_id,
                    "stop_id": stu.stop_id,
                    "delay_seconds": delay
                })

        # 3. In DataFrame und CSV speichern
        if trip_updates:
            df = pd.DataFrame(trip_updates)

            write_header = not os.path.exists(OUT_FILE)
            df.to_csv(OUT_FILE, mode='a', index=False, header=write_header)
            print(f"[{timestamp}] Gespeichert: {len(df)} Bus-Einträge")

        # 4. Warten bis nächster Abruf
        time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("Beende Collector.")
        break

    except Exception as e:
        print(f"Fehler: {e}")
        time.sleep(INTERVAL_SEC)
