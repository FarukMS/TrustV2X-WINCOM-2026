import os
import math
import time
from pathlib import Path


# ============================================================
# TrustV2X - Step 11
# Run 2x2 C-ITS Behavioural TraCI Demo - Final Corrected Version
# ============================================================
#
# Demonstrates:
# - Road incident activation
# - TrustV2X-accepted DENM warning
# - RSU-assisted rerouting
# - Normal vehicles using alternative routes
# - Ambulance, police and fire response vehicles
# - Emergency vehicles avoid the incident route
# - Vehicles ahead of emergency vehicles clear the lane
# - Cross-traffic yields/slows
#
# This is a behavioural C-ITS demonstration.
# It is NOT the 4x4 quantitative experiment.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = (
    PROJECT_ROOT
    / "sumo"
    / "realistic_2x2_cits_signalised_demo"
    / "config"
    / "trustv2x_2x2_signalised.sumocfg"
)

SCREENSHOT_DIR = (
    PROJECT_ROOT
    / "sumo"
    / "realistic_2x2_cits_signalised_demo"
    / "screenshots"
)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

VIEW_ID = "View #0"

SIMULATION_STEPS = 360
INCIDENT_START_STEP = 110
INCIDENT_END_STEP = 300

# Additional response vehicles inserted during incident for clearer demo.
DEMO_RESPONSE_INSERT_STEP = 125

SCREENSHOT_STEPS = [20, 80, 120, 150, 180, 240, 320]

# Incident edge: east exit from roundabout.
BLOCKED_EDGE = "R0_E"

# Hazard marker location, used for proximity checks.
HAZARD_X = 76
HAZARD_Y = 50

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
    """
    True when vehicle_id is ahead of the priority vehicle on the same lane.
    Such vehicles should keep moving to clear the lane, not stop.
    """
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
    """
    Adds runtime routes used by dynamic response vehicles.
    """
    global RUNTIME_ROUTES_READY

    if RUNTIME_ROUTES_READY:
        return

    add_route_if_missing(
        traci,
        "runtime_ambulance_priority_alt",
        ["SI_S", "S_SW", "SW_W", "W_NW", "NW_N", "N_NI"],
    )

    add_route_if_missing(
        traci,
        "runtime_police_priority_alt",
        ["EO_E", "E_SE", "SE_S", "S_SW", "SW_W", "W_WI"],
    )

    add_route_if_missing(
        traci,
        "runtime_fire_priority_alt",
        ["WI_W", "W_NW", "NW_N", "N_NE", "NE_E", "E_EO"],
    )

    add_route_if_missing(
        traci,
        "runtime_west_to_east_north_alt",
        ["WI_W", "W_NW", "NW_N", "N_NE", "NE_E", "E_EO"],
    )

    add_route_if_missing(
        traci,
        "runtime_west_to_east_south_alt",
        ["WI_W", "W_SW", "SW_S", "S_SE", "SE_E", "E_EO"],
    )

    RUNTIME_ROUTES_READY = True


def insert_incident_response_units(traci, step):
    """
    Inserts extra emergency-response vehicles during the incident.
    This ensures the demo clearly shows priority vehicles using alternatives.
    """
    global RESPONSE_INSERTED

    if RESPONSE_INSERTED:
        return

    if step < DEMO_RESPONSE_INSERT_STEP:
        return

    active = set(traci.vehicle.getIDList())

    try:
        if "ambulance_response_01" not in active:
            traci.vehicle.add(
                vehID="ambulance_response_01",
                routeID="runtime_ambulance_priority_alt",
                typeID="emergency_vehicle",
                depart="now",
            )
            apply_priority_colour(traci, "ambulance_response_01")
            print("[EMERGENCY] ambulance_response_01 inserted on clear priority alternative route.")
    except Exception as e:
        print(f"[WARNING] Could not insert ambulance_response_01: {e}")

    try:
        if "police_response_01" not in active:
            traci.vehicle.add(
                vehID="police_response_01",
                routeID="runtime_police_priority_alt",
                typeID="police_vehicle",
                depart="now",
            )
            apply_priority_colour(traci, "police_response_01")
            print("[EMERGENCY] police_response_01 inserted on clear priority alternative route.")
    except Exception as e:
        print(f"[WARNING] Could not insert police_response_01: {e}")

    try:
        if "fire_response_02" not in active:
            traci.vehicle.add(
                vehID="fire_response_02",
                routeID="runtime_fire_priority_alt",
                typeID="fire_vehicle",
                depart="now",
            )
            apply_priority_colour(traci, "fire_response_02")
            print("[EMERGENCY] fire_response_02 inserted on clear priority alternative route.")
    except Exception as e:
        print(f"[WARNING] Could not insert fire_response_02: {e}")

    RESPONSE_INSERTED = True


def initialise_vehicle_behaviour(traci):
    """
    Normal vehicles obey SUMO rules.
    Priority vehicles receive special visual and priority behaviour.
    """
    for veh_id in traci.vehicle.getIDList():
        if is_priority_vehicle(veh_id):
            try:
                # Priority vehicles may pass through signals in this C-ITS demo,
                # while other vehicles yield. This models emergency pre-emption.
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
            # Normal vehicles obey traffic lights and safety rules.
            try:
                traci.vehicle.setSpeedMode(veh_id, 31)
            except Exception:
                pass


def activate_incident(traci):
    """
    Restricts the incident edge but avoids total gridlock.
    """
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
        print("[EVENT] Affected vehicles should use alternative routes.")
        print("[EVENT] Emergency vehicles use priority alternatives.")
    except Exception as e:
        print(f"[WARNING] Could not restrict blocked edge {BLOCKED_EDGE}: {e}")


def clear_incident(traci):
    try:
        traci.edge.setMaxSpeed(BLOCKED_EDGE, 13.89)

        try:
            traci.edge.adaptTraveltime(BLOCKED_EDGE, 1.0, INCIDENT_END_STEP, SIMULATION_STEPS)
        except Exception:
            pass

        print(f"[EVENT] Road incident cleared on edge {BLOCKED_EDGE}")
    except Exception:
        pass


def normal_alternative_route_for_current_edge(current_edge, use_north=True):
    """
    Hard-coded safe alternatives from common positions.
    The returned route always starts with the vehicle's current edge.
    """

    if current_edge == "WI_W":
        if use_north:
            return ["WI_W", "W_NW", "NW_N", "N_NE", "NE_E", "E_EO"]
        return ["WI_W", "W_SW", "SW_S", "S_SE", "SE_E", "E_EO"]

    if current_edge == "W_R4":
        return ["W_R4", "R4_R3", "R3_R2", "R2_N", "N_NE", "NE_E", "E_EO"]

    if current_edge == "R4_R3":
        return ["R4_R3", "R3_R2", "R2_N", "N_NE", "NE_E", "E_EO"]

    if current_edge == "R3_R2":
        return ["R3_R2", "R2_N", "N_NE", "NE_E", "E_EO"]

    if current_edge == "R2_R1":
        return ["R2_R1", "R1_R0", "R0_R7", "R7_R6", "R6_S", "S_SE", "SE_E", "E_EO"]

    if current_edge == "R1_R0":
        return ["R1_R0", "R0_R7", "R7_R6", "R6_S", "S_SE", "SE_E", "E_EO"]

    return None


def priority_route_for_current_edge(veh_id, current_edge):
    """
    Reroutes priority vehicles only if they accidentally approach congestion.
    Most should already be on safe alternative routes from the route file.
    """

    if is_ambulance(veh_id):
        mapping = {
            "SI_S": ["SI_S", "S_SW", "SW_W", "W_NW", "NW_N", "N_NI"],
            "S_SW": ["S_SW", "SW_W", "W_NW", "NW_N", "N_NI"],
            "SW_W": ["SW_W", "W_NW", "NW_N", "N_NI"],
            "W_NW": ["W_NW", "NW_N", "N_NI"],
            "NW_N": ["NW_N", "N_NI"],
            "S_R6": ["S_R6", "R6_R5", "R5_R4", "R4_W", "W_NW", "NW_N", "N_NI"],
            "R6_R5": ["R6_R5", "R5_R4", "R4_W", "W_NW", "NW_N", "N_NI"],
            "R5_R4": ["R5_R4", "R4_W", "W_NW", "NW_N", "N_NI"],
            "R4_W": ["R4_W", "W_NW", "NW_N", "N_NI"],
        }

        return mapping.get(current_edge)

    if is_police(veh_id):
        mapping = {
            "EO_E": ["EO_E", "E_SE", "SE_S", "S_SW", "SW_W", "W_WI"],
            "E_SE": ["E_SE", "SE_S", "S_SW", "SW_W", "W_WI"],
            "SE_S": ["SE_S", "S_SW", "SW_W", "W_WI"],
            "S_SW": ["S_SW", "SW_W", "W_WI"],
            "SW_W": ["SW_W", "W_WI"],
            "E_R0": ["E_R0", "R0_R7", "R7_R6", "R6_S", "S_SW", "SW_W", "W_WI"],
            "R0_R7": ["R0_R7", "R7_R6", "R6_S", "S_SW", "SW_W", "W_WI"],
            "R7_R6": ["R7_R6", "R6_S", "S_SW", "SW_W", "W_WI"],
            "R6_S": ["R6_S", "S_SW", "SW_W", "W_WI"],
        }

        return mapping.get(current_edge)

    if is_fire(veh_id):
        mapping = {
            "WI_W": ["WI_W", "W_NW", "NW_N", "N_NE", "NE_E", "E_EO"],
            "W_NW": ["W_NW", "NW_N", "N_NE", "NE_E", "E_EO"],
            "NW_N": ["NW_N", "N_NE", "NE_E", "E_EO"],
            "N_NE": ["N_NE", "NE_E", "E_EO"],
            "W_R4": ["W_R4", "R4_R3", "R3_R2", "R2_N", "N_NE", "NE_E", "E_EO"],
        }

        return mapping.get(current_edge)

    return None


def reroute_normal_vehicles(traci, step):
    """
    Reroutes normal vehicles affected by the incident.
    """
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

        # First try SUMO reroute with adapted travel time.
        try:
            traci.vehicle.rerouteTraveltime(veh_id)
            updated_route = list(traci.vehicle.getRoute(veh_id))

            if BLOCKED_EDGE not in updated_route:
                REROUTED_NORMAL.add(veh_id)
                set_vehicle_colour(traci, veh_id, (0, 180, 0, 255))
                print(f"[REROUTE] Step {step}: {veh_id} rerouted by SUMO travel-time logic.")
                continue

        except Exception:
            pass

        # Fallback hard-coded route.
        use_north = (len(REROUTED_NORMAL) % 2 == 0)
        new_route = normal_alternative_route_for_current_edge(current_edge, use_north=use_north)

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
    """
    Priority vehicles should not be trapped by the incident.
    This catches any priority vehicle that approaches the hazard area.
    """
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

        # Most are already on safe routes. Only reroute if they drift near hazard or roundabout.
        if dist > 45 and BLOCKED_EDGE not in list(traci.vehicle.getRoute(veh_id)):
            continue

        new_route = priority_route_for_current_edge(veh_id, current_edge)

        if not new_route:
            continue

        try:
            traci.vehicle.setRoute(veh_id, new_route)
            REROUTED_PRIORITY.add(veh_id)
            apply_priority_colour(traci, veh_id)
            print(f"[PRIORITY REROUTE] Step {step}: {veh_id} assigned safe priority alternative.")
        except Exception:
            pass


def emergency_yielding_logic(traci):
    """
    C-ITS yielding behaviour:
    - vehicles ahead of emergency vehicle on same lane keep moving to clear path.
    - cross-traffic slows/stops briefly.
    - stopped/yielded vehicles are released after the emergency has passed.
    """
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
                        # Clear the lane; do not stop in front of ambulance/police/fire.
                        traci.vehicle.setSpeed(veh_id, 7.0)
                    else:
                        # Cross traffic yields.
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
    screenshot_path = SCREENSHOT_DIR / f"trustv2x_2x2_cits_behaviour_step_{step}.png"

    try:
        traci.gui.screenshot(VIEW_ID, str(screenshot_path))
        print(f"[OK] Screenshot saved: {screenshot_path}")
    except Exception as e:
        print(f"[WARNING] Screenshot failed at step {step}: {e}")


def main():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Config not found: {CONFIG_FILE}\n"
            "Run scripts/10_create_2x2_cits_signalised_network.py first."
        )

    traci = import_traci()
    sumo_gui = get_sumo_gui_binary()

    print("=" * 70)
    print("TrustV2X 2x2 C-ITS TraCI Behavioural Demo - Final Corrected")
    print("=" * 70)
    print(f"[INFO] Config: {CONFIG_FILE}")

    traci.start([
        sumo_gui,
        "-c",
        str(CONFIG_FILE),
        "--start",
        "--delay",
        "160",
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
    print("[OK] TrustV2X 2x2 C-ITS behavioural demo complete")
    print("=" * 70)
    print(f"[OK] Screenshots saved in: {SCREENSHOT_DIR}")

    print("\nDemo interpretation:")
    print("- Green vehicles: trusted/normal vehicles.")
    print("- Yellow vehicles: yielding/cautious vehicles responding to C-ITS priority or warnings.")
    print("- Cyan vehicles: ambulance/emergency response.")
    print("- Blue vehicles: police/priority vehicle.")
    print("- Orange vehicles: fire response.")
    print("- Red zone/marker: incident, blockage or rejected/malicious warning context.")
    print("- Transparent blue circle: RSU communication range.")
    print("- Purple marker: MEC / TrustV2X edge node.")


if __name__ == "__main__":
    main()