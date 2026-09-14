import os
import math
import time
from pathlib import Path


# ============================================================
# TrustV2X - Step 13
# Run 4x4 100m C-ITS Behavioural TraCI Demo
# ============================================================
#
# Demonstrates:
# - 4x4 100m signalised C-ITS road network
# - RSU/MEC/TrustV2X infrastructure
# - Road incident on central edge J11_J21
# - DENM-style hazard warning
# - TrustV2X ACCEPT decision
# - RSU-assisted rerouting
# - Ambulance, police and fire priority routes
# - Vehicles yielding/clearing for emergency vehicles
#
# This is a visual/behavioural demo, not the quantitative 4x4 experiment.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = (
    PROJECT_ROOT
    / "sumo"
    / "visual_4x4_100m_cits_demo"
    / "config"
    / "trustv2x_4x4_100m.sumocfg"
)

SCREENSHOT_DIR = (
    PROJECT_ROOT
    / "sumo"
    / "visual_4x4_100m_cits_demo"
    / "screenshots"
)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

VIEW_ID = "View #0"

SIMULATION_STEPS = 420
INCIDENT_START_STEP = 110
INCIDENT_END_STEP = 340
DEMO_RESPONSE_INSERT_STEP = 130

SCREENSHOT_STEPS = [30, 90, 120, 160, 220, 300, 380]

BLOCKED_EDGE = "J11_J21"
HAZARD_X = 50
HAZARD_Y = 33.3

REROUTED_NORMAL = set()
REROUTED_PRIORITY = set()
YIELDED_FOR_PRIORITY = set()
RUNTIME_ROUTES_READY = False
RESPONSE_INSERTED = False


def get_sumo_gui_binary():
    sumo_home = os.environ.get("SUMO_HOME")

    if sumo_home:
        candidate = Path(sumo_home) / "bin" / "sumo-gui.exe"
        if candidate.exists():
            return str(candidate)

    return "sumo-gui"


def import_traci():
    sumo_home = os.environ.get("SUMO_HOME")

    if sumo_home:
        tools = Path(sumo_home) / "tools"
        if tools.exists():
            import sys
            sys.path.append(str(tools))

    import traci
    return traci


def read_net_boundary(traci):
    boundary = traci.simulation.getNetBoundary()

    if len(boundary) == 2:
        (xmin, ymin), (xmax, ymax) = boundary
    else:
        xmin, ymin, xmax, ymax = boundary

    return float(xmin), float(ymin), float(xmax), float(ymax)


def set_gui_view(traci):
    try:
        xmin, ymin, xmax, ymax = read_net_boundary(traci)

        pad_x = (xmax - xmin) * 0.06
        pad_y = (ymax - ymin) * 0.06

        traci.gui.setBoundary(
            VIEW_ID,
            xmin - pad_x,
            ymin - pad_y,
            xmax + pad_x,
            ymax + pad_y,
        )

        try:
            traci.gui.setScheme(VIEW_ID, "real world")
        except Exception:
            pass

    except Exception as e:
        print(f"[WARNING] Could not set GUI view: {e}")


def is_priority_vehicle(veh_id):
    v = veh_id.lower()
    return (
        "ambulance" in v
        or "police" in v
        or "fire" in v
        or "emergency" in v
    )


def is_ambulance(veh_id):
    return "ambulance" in veh_id.lower()


def is_police(veh_id):
    return "police" in veh_id.lower()


def is_fire(veh_id):
    return "fire" in veh_id.lower()


def set_vehicle_colour(traci, veh_id, colour):
    try:
        traci.vehicle.setColor(veh_id, colour)
    except Exception:
        pass


def apply_priority_colour(traci, veh_id):
    if is_ambulance(veh_id):
        set_vehicle_colour(traci, veh_id, (0, 220, 255, 255))
    elif is_police(veh_id):
        set_vehicle_colour(traci, veh_id, (0, 80, 255, 255))
    elif is_fire(veh_id):
        set_vehicle_colour(traci, veh_id, (255, 80, 0, 255))
    else:
        set_vehicle_colour(traci, veh_id, (0, 220, 255, 255))


def vehicle_distance(traci, veh_a, veh_b):
    ax, ay = traci.vehicle.getPosition(veh_a)
    bx, by = traci.vehicle.getPosition(veh_b)

    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def vehicle_distance_to_hazard(traci, veh_id):
    x, y = traci.vehicle.getPosition(veh_id)
    return math.sqrt((x - HAZARD_X) ** 2 + (y - HAZARD_Y) ** 2)


def same_lane_and_ahead(traci, vehicle_id, priority_id):
    try:
        veh_lane = traci.vehicle.getLaneID(vehicle_id)
        pri_lane = traci.vehicle.getLaneID(priority_id)

        if veh_lane != pri_lane:
            return False

        veh_pos = traci.vehicle.getLanePosition(vehicle_id)
        pri_pos = traci.vehicle.getLanePosition(priority_id)

        return veh_pos > pri_pos

    except Exception:
        return False


def add_route_if_missing(traci, route_id, edges):
    try:
        if route_id not in traci.route.getIDList():
            traci.route.add(route_id, edges)
    except Exception as e:
        print(f"[WARNING] Could not add route {route_id}: {e}")


def setup_runtime_routes(traci):
    global RUNTIME_ROUTES_READY

    if RUNTIME_ROUTES_READY:
        return

    add_route_if_missing(
        traci,
        "runtime_we_r1_north_alt",
        ["W1_J01", "J01_J02", "J02_J12", "J12_J22", "J22_J32", "J32_E2"],
    )

    add_route_if_missing(
        traci,
        "runtime_we_r1_south_alt",
        ["W1_J01", "J01_J00", "J00_J10", "J10_J20", "J20_J30", "J30_E0"],
    )

    add_route_if_missing(
        traci,
        "runtime_ambulance_priority_alt",
        ["S0_J00", "J00_J01", "J01_J02", "J02_J03", "J03_N0"],
    )

    add_route_if_missing(
        traci,
        "runtime_police_priority_alt",
        ["E3_J33", "J33_J23", "J23_J13", "J13_J03", "J03_W3"],
    )

    add_route_if_missing(
        traci,
        "runtime_fire_priority_alt",
        ["W0_J00", "J00_J10", "J10_J20", "J20_J30", "J30_E0"],
    )

    RUNTIME_ROUTES_READY = True


def insert_incident_response_units(traci, step):
    global RESPONSE_INSERTED

    if RESPONSE_INSERTED:
        return

    if step < DEMO_RESPONSE_INSERT_STEP:
        return

    active = set(traci.vehicle.getIDList())

    try:
        if "ambulance_response_4x4_01" not in active:
            traci.vehicle.add(
                vehID="ambulance_response_4x4_01",
                routeID="runtime_ambulance_priority_alt",
                typeID="emergency_vehicle",
                depart="now",
            )
            apply_priority_colour(traci, "ambulance_response_4x4_01")
            print("[EMERGENCY] ambulance_response_4x4_01 inserted on priority alternative route.")
    except Exception as e:
        print(f"[WARNING] Could not insert ambulance_response_4x4_01: {e}")

    try:
        if "police_response_4x4_01" not in active:
            traci.vehicle.add(
                vehID="police_response_4x4_01",
                routeID="runtime_police_priority_alt",
                typeID="police_vehicle",
                depart="now",
            )
            apply_priority_colour(traci, "police_response_4x4_01")
            print("[EMERGENCY] police_response_4x4_01 inserted on priority alternative route.")
    except Exception as e:
        print(f"[WARNING] Could not insert police_response_4x4_01: {e}")

    try:
        if "fire_response_4x4_02" not in active:
            traci.vehicle.add(
                vehID="fire_response_4x4_02",
                routeID="runtime_fire_priority_alt",
                typeID="fire_vehicle",
                depart="now",
            )
            apply_priority_colour(traci, "fire_response_4x4_02")
            print("[EMERGENCY] fire_response_4x4_02 inserted on priority alternative route.")
    except Exception as e:
        print(f"[WARNING] Could not insert fire_response_4x4_02: {e}")

    RESPONSE_INSERTED = True


def initialise_vehicle_behaviour(traci):
    for veh_id in traci.vehicle.getIDList():
        if is_priority_vehicle(veh_id):
            try:
                # Priority vehicles model emergency pre-emption.
                traci.vehicle.setSpeedMode(veh_id, 0)
            except Exception:
                pass

            apply_priority_colour(traci, veh_id)

        elif "malicious" in veh_id.lower():
            try:
                traci.vehicle.setSpeedMode(veh_id, 31)
            except Exception:
                pass
            set_vehicle_colour(traci, veh_id, (255, 0, 0, 255))

        elif "flagged" in veh_id.lower():
            try:
                traci.vehicle.setSpeedMode(veh_id, 31)
            except Exception:
                pass
            set_vehicle_colour(traci, veh_id, (255, 200, 0, 255))

        else:
            try:
                # Normal vehicles obey traffic lights and safety logic.
                traci.vehicle.setSpeedMode(veh_id, 31)
            except Exception:
                pass


def activate_incident(traci):
    try:
        traci.edge.setMaxSpeed(BLOCKED_EDGE, 1.0)

        try:
            traci.edge.adaptTraveltime(BLOCKED_EDGE, 9999.0, INCIDENT_START_STEP, INCIDENT_END_STEP)
        except Exception:
            pass

        print(f"[EVENT] Road incident activated on edge {BLOCKED_EDGE}")
        print("[EVENT] DENM warning generated and sent to RSU.")
        print("[EVENT] TrustV2X decision: ACCEPT genuine road blockage warning.")
        print("[EVENT] RSU broadcasts verified hazard alert.")
        print("[EVENT] Vehicles approaching the central incident should use alternatives.")

    except Exception as e:
        print(f"[WARNING] Could not restrict blocked edge {BLOCKED_EDGE}: {e}")


def clear_incident(traci):
    try:
        traci.edge.setMaxSpeed(BLOCKED_EDGE, 10.0)

        try:
            traci.edge.adaptTraveltime(BLOCKED_EDGE, 1.0, INCIDENT_END_STEP, SIMULATION_STEPS)
        except Exception:
            pass

        print(f"[EVENT] Road incident cleared on edge {BLOCKED_EDGE}")
    except Exception:
        pass


def normal_alternative_route_for_current_edge(current_edge, use_north=True):
    """
    Alternative route must begin with the vehicle's current edge.
    """
    if current_edge == "W1_J01":
        if use_north:
            return ["W1_J01", "J01_J02", "J02_J12", "J12_J22", "J22_J32", "J32_E2"]
        return ["W1_J01", "J01_J00", "J00_J10", "J10_J20", "J20_J30", "J30_E0"]

    if current_edge == "J01_J11":
        if use_north:
            return ["J01_J11", "J11_J12", "J12_J22", "J22_J32", "J32_E2"]
        return ["J01_J11", "J11_J10", "J10_J20", "J20_J30", "J30_E0"]

    if current_edge == "J11_J12":
        return ["J11_J12", "J12_J22", "J22_J32", "J32_E2"]

    if current_edge == "J11_J10":
        return ["J11_J10", "J10_J20", "J20_J30", "J30_E0"]

    return None


def priority_route_for_current_edge(veh_id, current_edge):
    if is_ambulance(veh_id):
        mapping = {
            "S0_J00": ["S0_J00", "J00_J01", "J01_J02", "J02_J03", "J03_N0"],
            "J00_J01": ["J00_J01", "J01_J02", "J02_J03", "J03_N0"],
            "J01_J02": ["J01_J02", "J02_J03", "J03_N0"],
            "J02_J03": ["J02_J03", "J03_N0"],
        }
        return mapping.get(current_edge)

    if is_police(veh_id):
        mapping = {
            "E3_J33": ["E3_J33", "J33_J23", "J23_J13", "J13_J03", "J03_W3"],
            "J33_J23": ["J33_J23", "J23_J13", "J13_J03", "J03_W3"],
            "J23_J13": ["J23_J13", "J13_J03", "J03_W3"],
            "J13_J03": ["J13_J03", "J03_W3"],
        }
        return mapping.get(current_edge)

    if is_fire(veh_id):
        mapping = {
            "W0_J00": ["W0_J00", "J00_J10", "J10_J20", "J20_J30", "J30_E0"],
            "J00_J10": ["J00_J10", "J10_J20", "J20_J30", "J30_E0"],
            "J10_J20": ["J10_J20", "J20_J30", "J30_E0"],
            "J20_J30": ["J20_J30", "J30_E0"],
        }
        return mapping.get(current_edge)

    return None


def reroute_normal_vehicles(traci, step):
    for veh_id in traci.vehicle.getIDList():
        if is_priority_vehicle(veh_id):
            continue

        if veh_id in REROUTED_NORMAL:
            continue

        try:
            route = list(traci.vehicle.getRoute(veh_id))
            current_edge = traci.vehicle.getRoadID(veh_id)
        except Exception:
            continue

        if current_edge.startswith(":"):
            continue

        if BLOCKED_EDGE not in route:
            continue

        # Try SUMO dynamic travel-time rerouting first.
        try:
            traci.vehicle.rerouteTraveltime(veh_id)
            updated_route = list(traci.vehicle.getRoute(veh_id))

            if BLOCKED_EDGE not in updated_route:
                REROUTED_NORMAL.add(veh_id)
                set_vehicle_colour(traci, veh_id, (0, 180, 0, 255))
                print(f"[REROUTE] Step {step}: {veh_id} rerouted by travel-time logic.")
                continue

        except Exception:
            pass

        # Fallback manual alternative
        use_north = len(REROUTED_NORMAL) % 2 == 0
        new_route = normal_alternative_route_for_current_edge(current_edge, use_north)

        if not new_route:
            continue

        try:
            traci.vehicle.setRoute(veh_id, new_route)
            REROUTED_NORMAL.add(veh_id)

            route_name = "north alternative" if use_north else "south alternative"
            print(f"[REROUTE] Step {step}: {veh_id} rerouted via {route_name}.")
        except Exception:
            pass


def reroute_priority_vehicles(traci, step):
    for veh_id in traci.vehicle.getIDList():
        if not is_priority_vehicle(veh_id):
            continue

        if veh_id in REROUTED_PRIORITY:
            continue

        try:
            current_edge = traci.vehicle.getRoadID(veh_id)
        except Exception:
            continue

        if current_edge.startswith(":"):
            continue

        try:
            dist = vehicle_distance_to_hazard(traci, veh_id)
        except Exception:
            dist = 999

        if dist > 45 and BLOCKED_EDGE not in list(traci.vehicle.getRoute(veh_id)):
            continue

        new_route = priority_route_for_current_edge(veh_id, current_edge)

        if not new_route:
            continue

        try:
            traci.vehicle.setRoute(veh_id, new_route)
            REROUTED_PRIORITY.add(veh_id)
            apply_priority_colour(traci, veh_id)
            print(f"[PRIORITY REROUTE] Step {step}: {veh_id} assigned safe priority route.")
        except Exception:
            pass


def same_lane_and_ahead(traci, vehicle_id, priority_id):
    try:
        veh_lane = traci.vehicle.getLaneID(vehicle_id)
        pri_lane = traci.vehicle.getLaneID(priority_id)

        if veh_lane != pri_lane:
            return False

        veh_pos = traci.vehicle.getLanePosition(vehicle_id)
        pri_pos = traci.vehicle.getLanePosition(priority_id)

        return veh_pos > pri_pos

    except Exception:
        return False


def vehicle_distance(traci, veh_a, veh_b):
    ax, ay = traci.vehicle.getPosition(veh_a)
    bx, by = traci.vehicle.getPosition(veh_b)

    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def emergency_yielding_logic(traci):
    active = set(traci.vehicle.getIDList())
    priority_units = [v for v in active if is_priority_vehicle(v)]

    if not priority_units:
        for veh_id in list(YIELDED_FOR_PRIORITY):
            if veh_id in active:
                try:
                    traci.vehicle.setSpeed(veh_id, -1)
                    set_vehicle_colour(traci, veh_id, (0, 180, 0, 255))
                except Exception:
                    pass
        YIELDED_FOR_PRIORITY.clear()
        return

    for priority_id in priority_units:
        for veh_id in active:
            if veh_id == priority_id:
                continue

            if is_priority_vehicle(veh_id):
                continue

            try:
                dist = vehicle_distance(traci, veh_id, priority_id)
            except Exception:
                continue

            if dist <= 14:
                try:
                    if same_lane_and_ahead(traci, veh_id, priority_id):
                        # Let the vehicle ahead clear the lane.
                        traci.vehicle.setSpeed(veh_id, 7.0)
                    else:
                        # Cross-traffic yields.
                        traci.vehicle.setSpeed(veh_id, 0)

                    set_vehicle_colour(traci, veh_id, (255, 200, 0, 255))
                    YIELDED_FOR_PRIORITY.add(veh_id)

                except Exception:
                    pass

            elif veh_id in YIELDED_FOR_PRIORITY and dist > 24:
                try:
                    traci.vehicle.setSpeed(veh_id, -1)
                    set_vehicle_colour(traci, veh_id, (0, 180, 0, 255))
                    YIELDED_FOR_PRIORITY.remove(veh_id)
                except Exception:
                    pass


def save_screenshot(traci, step):
    screenshot_path = SCREENSHOT_DIR / f"trustv2x_4x4_100m_cits_step_{step}.png"

    try:
        traci.gui.screenshot(VIEW_ID, str(screenshot_path))
        print(f"[OK] Screenshot saved: {screenshot_path}")
    except Exception as e:
        print(f"[WARNING] Screenshot failed at step {step}: {e}")


def main():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Config not found: {CONFIG_FILE}\n"
            "Run scripts/12_create_4x4_100m_cits_signalised_network.py first."
        )

    traci = import_traci()
    sumo_gui = get_sumo_gui_binary()

    print("=" * 70)
    print("TrustV2X 4x4 100m C-ITS TraCI Behavioural Demo")
    print("=" * 70)
    print(f"[INFO] Config: {CONFIG_FILE}")

    traci.start([
        sumo_gui,
        "-c",
        str(CONFIG_FILE),
        "--start",
        "--delay",
        "150",
    ])

    time.sleep(1)
    set_gui_view(traci)
    setup_runtime_routes(traci)

    incident_active = False
    incident_cleared = False

    tls_ids = traci.trafficlight.getIDList()
    print(f"[INFO] Traffic lights detected: {tls_ids}")

    for step in range(SIMULATION_STEPS + 1):
        traci.simulationStep()

        initialise_vehicle_behaviour(traci)

        if step == INCIDENT_START_STEP and not incident_active:
            activate_incident(traci)
            incident_active = True

        if step >= DEMO_RESPONSE_INSERT_STEP:
            insert_incident_response_units(traci, step)

        if incident_active and INCIDENT_START_STEP <= step <= INCIDENT_END_STEP:
            reroute_normal_vehicles(traci, step)
            reroute_priority_vehicles(traci, step)

        if step == INCIDENT_END_STEP and not incident_cleared:
            clear_incident(traci)
            incident_cleared = True

        emergency_yielding_logic(traci)

        if step in SCREENSHOT_STEPS:
            save_screenshot(traci, step)

        if step % 50 == 0:
            print(f"[INFO] Simulation step: {step}")

        time.sleep(0.01)

    print("\n[INFO] Simulation completed.")
    print("[INFO] SUMO-GUI will remain open until you press Enter here.")
    input("Press Enter to close SUMO-GUI...")

    traci.close()

    print("\n" + "=" * 70)
    print("[OK] TrustV2X 4x4 100m C-ITS behavioural demo complete")
    print("=" * 70)
    print(f"[OK] Screenshots saved in: {SCREENSHOT_DIR}")

    print("\nDemo interpretation:")
    print("- Green vehicles: trusted/normal vehicles.")
    print("- Yellow vehicles: yielding/cautious vehicles.")
    print("- Cyan vehicles: ambulance/emergency response.")
    print("- Blue vehicles: police/priority response.")
    print("- Orange vehicles: fire response.")
    print("- Red zone/marker: road incident / blockage / rejected-warning context.")
    print("- Transparent blue circle: RSU communication range.")
    print("- Purple marker: MEC / TrustV2X edge node.")


if __name__ == "__main__":
    main()