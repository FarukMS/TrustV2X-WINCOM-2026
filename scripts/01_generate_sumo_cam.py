import os
import sys
import csv
import math
import subprocess
from pathlib import Path

import traci


# ============================================================
# TrustV2X - Step 01
# Generate a larger SUMO-based normal CAM-like dataset
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUMO_DIR = PROJECT_ROOT / "sumo"
NETWORK_DIR = SUMO_DIR / "network"
ROUTES_DIR = SUMO_DIR / "routes"
CONFIG_DIR = SUMO_DIR / "config"
OUTPUT_DIR = SUMO_DIR / "outputs"

DATA_OUTPUT_DIR = PROJECT_ROOT / "data" / "sumo_generated"

NETWORK_FILE = NETWORK_DIR / "trustv2x_grid.net.xml"
ROUTE_FILE = ROUTES_DIR / "trustv2x_routes.rou.xml"
CONFIG_FILE = CONFIG_DIR / "trustv2x.sumocfg"
CAM_OUTPUT_FILE = DATA_OUTPUT_DIR / "normal_cam_messages.csv"


# Larger simulation for final experiment
SIMULATION_END_TIME = 670
TRIP_PERIOD = 2
SEED = 42
GRID_NUMBER = 4
GRID_LENGTH = 220
DEFAULT_SPEED = 13.89
NEIGHBOUR_DISTANCE_METRES = 50.0


def ensure_directories():
    for folder in [NETWORK_DIR, ROUTES_DIR, CONFIG_DIR, OUTPUT_DIR, DATA_OUTPUT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def get_sumo_home():
    sumo_home = os.environ.get("SUMO_HOME")

    if sumo_home and Path(sumo_home).exists():
        return Path(sumo_home)

    common_paths = [
        Path(r"C:\Program Files (x86)\Eclipse\Sumo"),
        Path(r"C:\Program Files\Eclipse\Sumo"),
        Path(r"C:\sumo"),
    ]

    for path in common_paths:
        if path.exists():
            os.environ["SUMO_HOME"] = str(path)
            os.environ["PATH"] = str(path / "bin") + os.pathsep + os.environ.get("PATH", "")
            return path

    raise FileNotFoundError(
        "SUMO_HOME was not found. Please set SUMO_HOME to your SUMO installation folder."
    )


def run_command(command, description):
    print(f"\n[Running] {description}")
    print(" ".join(str(x) for x in command))

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {description}")

    print(f"[OK] {description}")


def create_sumo_network():
    command = [
        "netgenerate",
        "--grid",
        "--grid.number", str(GRID_NUMBER),
        "--grid.length", str(GRID_LENGTH),
        "--default.speed", str(DEFAULT_SPEED),
        "--tls.guess", "true",
        "--output-file", str(NETWORK_FILE),
    ]

    run_command(command, "Creating larger SUMO grid network")


def create_vehicle_routes(sumo_home):
    random_trips_script = sumo_home / "tools" / "randomTrips.py"

    if not random_trips_script.exists():
        raise FileNotFoundError(f"randomTrips.py not found at: {random_trips_script}")

    command = [
        sys.executable,
        str(random_trips_script),
        "-n", str(NETWORK_FILE),
        "-r", str(ROUTE_FILE),
        "-e", str(SIMULATION_END_TIME),
        "--period", str(TRIP_PERIOD),
        "--seed", str(SEED),
        "--validate",
    ]

    run_command(command, "Generating larger vehicle route set")


def create_sumo_config():
    config_content = f"""<?xml version="1.0" encoding="UTF-8"?>

<configuration>
    <input>
        <net-file value="../network/{NETWORK_FILE.name}"/>
        <route-files value="../routes/{ROUTE_FILE.name}"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="{SIMULATION_END_TIME}"/>
        <step-length value="1"/>
    </time>

    <processing>
        <time-to-teleport value="-1"/>
    </processing>

    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
    </report>
</configuration>
"""

    CONFIG_FILE.write_text(config_content, encoding="utf-8")
    print(f"[OK] SUMO config created: {CONFIG_FILE}")


def calculate_distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def count_neighbours(vehicle_id, all_positions):
    if vehicle_id not in all_positions:
        return 0

    current_position = all_positions[vehicle_id]
    neighbour_count = 0

    for other_id, other_position in all_positions.items():
        if other_id == vehicle_id:
            continue

        distance = calculate_distance(current_position, other_position)

        if distance <= NEIGHBOUR_DISTANCE_METRES:
            neighbour_count += 1

    return neighbour_count


def run_simulation_and_generate_cam():
    sumo_command = [
        "sumo",
        "-c", str(CONFIG_FILE),
    ]

    print("\n[Running] SUMO simulation and generating larger CAM-like dataset")

    traci.start(sumo_command)

    previous_speed = {}
    previous_time = {}
    message_sequence = {}
    cam_rows = []

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        current_time = traci.simulation.getTime()
        vehicle_ids = traci.vehicle.getIDList()

        all_positions = {
            veh_id: traci.vehicle.getPosition(veh_id)
            for veh_id in vehicle_ids
        }

        for veh_id in vehicle_ids:
            x_position, y_position = traci.vehicle.getPosition(veh_id)
            speed = traci.vehicle.getSpeed(veh_id)
            heading = traci.vehicle.getAngle(veh_id)
            lane_id = traci.vehicle.getLaneID(veh_id)
            lane_position = traci.vehicle.getLanePosition(veh_id)
            road_id = traci.vehicle.getRoadID(veh_id)

            prev_speed = previous_speed.get(veh_id, speed)
            acceleration = speed - prev_speed
            previous_speed[veh_id] = speed

            prev_time = previous_time.get(veh_id, current_time)
            inter_arrival_time = current_time - prev_time
            previous_time[veh_id] = current_time

            message_sequence[veh_id] = message_sequence.get(veh_id, 0) + 1

            neighbour_count = count_neighbours(veh_id, all_positions)

            cam_rows.append({
                "vehicle_id": veh_id,
                "message_sequence": message_sequence[veh_id],
                "timestamp": current_time,
                "x_position": round(x_position, 3),
                "y_position": round(y_position, 3),
                "speed": round(speed, 3),
                "acceleration": round(acceleration, 3),
                "heading": round(heading, 3),
                "lane_id": lane_id,
                "lane_position": round(lane_position, 3),
                "road_id": road_id,
                "neighbour_count": neighbour_count,
                "inter_arrival_time": round(inter_arrival_time, 3),
                "message_type": "CAM",
                "label": 0,
                "attack_type": "normal",
                "attack_severity": "none",
                "source": "SUMO"
            })

    traci.close()

    if not cam_rows:
        raise RuntimeError("No CAM messages were generated.")

    with open(CAM_OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = list(cam_rows[0].keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cam_rows)

    print(f"[OK] Normal CAM-like messages generated: {CAM_OUTPUT_FILE}")
    print(f"[INFO] Total normal CAM messages: {len(cam_rows)}")

    if len(cam_rows) < 28000:
        print("[WARNING] Fewer than 28,000 normal messages generated.")
        print("[ACTION] Increase SIMULATION_END_TIME or reduce TRIP_PERIOD if needed.")


def main():
    print("=" * 70)
    print("TrustV2X - Larger SUMO CAM-like Message Generator")
    print("=" * 70)

    ensure_directories()
    sumo_home = get_sumo_home()

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] SUMO_HOME: {sumo_home}")
    print(f"[INFO] Simulation end time: {SIMULATION_END_TIME} seconds")
    print(f"[INFO] Grid: {GRID_NUMBER} x {GRID_NUMBER}")
    print(f"[INFO] Trip period: {TRIP_PERIOD}")

    create_sumo_network()
    create_vehicle_routes(sumo_home)
    create_sumo_config()
    run_simulation_and_generate_cam()

    print("\nDone. Next step: run 02_inject_attacks.py")


if __name__ == "__main__":
    main()