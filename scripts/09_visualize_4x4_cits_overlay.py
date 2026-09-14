import os
import math
import time
from pathlib import Path


# ============================================================
# TrustV2X - Step 09
# Visual C-ITS Overlay for Existing 4x4 SUMO Experiment
# ============================================================
#
# Purpose:
# - Keep the original 4x4 experiment untouched.
# - Improve visual realism for screenshots/video.
# - Add C-ITS components: RSU, MEC/TrustV2X, hazard marker,
#   coloured vehicles, communication range and decision markers.
#
# IMPORTANT:
# - This is a VISUAL-ONLY demonstration.
# - It does NOT retrain models.
# - It does NOT regenerate the dataset.
# - It does NOT change vehicle route, speed, acceleration, lane,
#   vehicle class, network topology or traffic flow.
#
# Safe visual-only actions used:
# - traci.vehicle.setColor(...)
# - traci.vehicle.setShapeClass(...) where supported
# - traci.poi.add(...)
# - traci.polygon.add(...)
# - traci.gui.screenshot(...)
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUMO_CONFIG = PROJECT_ROOT / "sumo" / "config" / "trustv2x.sumocfg"

SCREENSHOT_DIR = PROJECT_ROOT / "results" / "figures" / "cits_4x4_visual_demo"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

SIMULATION_STEPS = 670
SCREENSHOT_STEPS = [50, 150, 300, 500, 660]

VIEW_ID = "View #0"


def get_sumo_gui_binary():
    """
    Finds SUMO-GUI from SUMO_HOME if available.
    Falls back to 'sumo-gui' if SUMO is already in PATH.
    """
    sumo_home = os.environ.get("SUMO_HOME")

    if sumo_home:
        candidate = Path(sumo_home) / "bin" / "sumo-gui.exe"
        if candidate.exists():
            return str(candidate)

    return "sumo-gui"


def import_traci():
    """
    Imports TraCI using SUMO_HOME/tools if required.
    """
    sumo_home = os.environ.get("SUMO_HOME")

    if sumo_home:
        tools = Path(sumo_home) / "tools"
        if tools.exists():
            import sys
            sys.path.append(str(tools))

    import traci
    return traci


def circle_points(cx, cy, radius, n=60):
    """
    Generates circle points for RSU communication-zone polygon.
    """
    points = []

    for i in range(n):
        angle = 2 * math.pi * i / n
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))

    return points


def add_or_update_poi(traci, poi_id, x, y, color, poi_type, layer=10):
    """
    Adds or updates a SUMO POI marker.
    Visual only.
    """
    try:
        existing = traci.poi.getIDList()

        if poi_id in existing:
            traci.poi.setPosition(poi_id, x, y)
            traci.poi.setColor(poi_id, color)
            traci.poi.setType(poi_id, poi_type)
        else:
            traci.poi.add(
                poiID=poi_id,
                x=x,
                y=y,
                color=color,
                poiType=poi_type,
                layer=layer,
            )

    except Exception as e:
        print(f"[WARNING] Could not add/update POI {poi_id}: {e}")


def add_or_update_polygon(traci, polygon_id, shape, color, polygon_type, layer=1, fill=True):
    """
    Adds or updates a SUMO polygon.
    Used for visual communication range only.
    """
    try:
        existing = traci.polygon.getIDList()

        if polygon_id in existing:
            traci.polygon.setShape(polygon_id, shape)
            traci.polygon.setColor(polygon_id, color)
        else:
            traci.polygon.add(
                polygonID=polygon_id,
                shape=shape,
                color=color,
                fill=fill,
                polygonType=polygon_type,
                layer=layer,
            )

    except Exception as e:
        print(f"[WARNING] Could not add/update polygon {polygon_id}: {e}")

def read_net_boundary(traci):
    """
    Handles different SUMO/TraCI net boundary formats.

    Some versions return:
        ((xmin, ymin), (xmax, ymax))

    Others may return:
        (xmin, ymin, xmax, ymax)
    """
    boundary = traci.simulation.getNetBoundary()

    if len(boundary) == 2:
        (xmin, ymin), (xmax, ymax) = boundary
    elif len(boundary) == 4:
        xmin, ymin, xmax, ymax = boundary
    else:
        raise ValueError(f"Unexpected SUMO net boundary format: {boundary}")

    return float(xmin), float(ymin), float(xmax), float(ymax)

def set_gui_view(traci):
    """
    Adjusts SUMO-GUI camera/view for clearer screenshots.
    Visual only.
    """
    try:
        xmin, ymin, xmax, ymax = read_net_boundary(traci)

        pad_x = (xmax - xmin) * 0.05
        pad_y = (ymax - ymin) * 0.05

        traci.gui.setBoundary(
            VIEW_ID,
            xmin - pad_x,
            ymin - pad_y,
            xmax + pad_x,
            ymax + pad_y,
        )

        try:
            traci.gui.setZoom(VIEW_ID, 900)
        except Exception:
            pass

        try:
            traci.gui.setScheme(VIEW_ID, "real world")
        except Exception:
            pass

    except Exception as e:
        print(f"[WARNING] Could not set GUI view: {e}")


def get_network_centre(traci):
    """
    Returns centre and boundary of current SUMO network.
    """
    xmin, ymin, xmax, ymax = read_net_boundary(traci)

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    return cx, cy, xmin, ymin, xmax, ymax

def add_cits_components(traci):
    """
    Adds visual C-ITS components:
    - RSU
    - MEC / TrustV2X Edge
    - C-ITS Control Centre
    - Road hazard / obstacle marker
    - RSU communication range circle
    """
    cx, cy, xmin, ymin, xmax, ymax = get_network_centre(traci)

    width = xmax - xmin
    height = ymax - ymin

    # Main RSU near centre of 4x4 grid
    rsu_x = cx
    rsu_y = cy

    # MEC/TrustV2X slightly offset from RSU
    mec_x = cx + width * 0.08
    mec_y = cy + height * 0.08

    # Road hazard/obstacle marker
    hazard_x = cx + width * 0.18
    hazard_y = cy

    # C-ITS control marker
    control_x = cx - width * 0.10
    control_y = cy + height * 0.10

    # Communication range around RSU
    rsu_range = min(width, height) * 0.20
    rsu_circle = circle_points(rsu_x, rsu_y, rsu_range, n=72)

    add_or_update_polygon(
        traci,
        "RSU_1_COMMUNICATION_RANGE",
        rsu_circle,
        color=(0, 90, 255, 35),
        polygon_type="RSU communication range",
        layer=0,
        fill=True,
    )

    add_or_update_poi(
        traci,
        "RSU_1",
        rsu_x,
        rsu_y,
        color=(0, 80, 255, 255),
        poi_type="RSU-1",
        layer=20,
    )

    add_or_update_poi(
        traci,
        "MEC_TRUSTV2X_EDGE",
        mec_x,
        mec_y,
        color=(160, 0, 255, 255),
        poi_type="MEC / TrustV2X Edge",
        layer=20,
    )

    add_or_update_poi(
        traci,
        "CITS_CONTROL_CENTRE",
        control_x,
        control_y,
        color=(255, 140, 0, 255),
        poi_type="C-ITS Control Centre",
        layer=20,
    )

    add_or_update_poi(
        traci,
        "ROAD_HAZARD_OBSTACLE",
        hazard_x,
        hazard_y,
        color=(255, 0, 0, 255),
        poi_type="Road Hazard / Obstacle",
        layer=20,
    )

    return {
        "rsu": (rsu_x, rsu_y),
        "mec": (mec_x, mec_y),
        "hazard": (hazard_x, hazard_y),
        "control": (control_x, control_y),
        "range": rsu_range,
    }


def choose_demo_vehicles(active_vehicles, remembered):
    """
    Selects stable demo vehicle roles:
    - one malicious/rejected vehicle
    - one suspicious/flagged vehicle
    - one emergency/priority vehicle

    This only assigns visual roles, not behavioural roles.
    """
    active = list(active_vehicles)

    if not active:
        return remembered

    if remembered.get("malicious") not in active and len(active) >= 1:
        remembered["malicious"] = active[0]

    if remembered.get("flagged") not in active and len(active) >= 2:
        remembered["flagged"] = active[1]

    if remembered.get("emergency") not in active and len(active) >= 3:
        remembered["emergency"] = active[2]

    return remembered


def style_vehicle(traci, veh_id, color, shape_class=None):
    """
    Strict visual-only styling.

    Safe:
    - setColor changes only the vehicle colour.
    - setShapeClass changes only the visual GUI shape where supported.

    Avoided:
    - setVehicleClass is NOT used because it can affect permissions/behaviour.
    """
    try:
        traci.vehicle.setColor(veh_id, color)
    except Exception:
        pass

    if shape_class:
        try:
            traci.vehicle.setShapeClass(veh_id, shape_class)
        except Exception:
            # Some SUMO/TraCI versions may not support this.
            # Colour styling will still work.
            pass


def apply_vehicle_visual_roles(traci, step, roles):
    """
    Colours active vehicles according to visual C-ITS roles.

    Green  = trusted/safe vehicle
    Grey   = background traffic
    Cyan   = emergency/priority vehicle
    Yellow = suspicious/flagged vehicle
    Red    = malicious/rejected vehicle
    """
    active = traci.vehicle.getIDList()

    roles = choose_demo_vehicles(active, roles)

    # Default: trusted/safe vehicles
    for veh_id in active:
        style_vehicle(
            traci,
            veh_id,
            color=(0, 180, 0, 255),
            shape_class="passenger",
        )

    # Background traffic: grey visual variation
    for i, veh_id in enumerate(active):
        if i % 7 == 0:
            style_vehicle(
                traci,
                veh_id,
                color=(160, 160, 160, 255),
                shape_class="passenger",
            )

    # Emergency / priority vehicle
    emergency_id = roles.get("emergency")
    if emergency_id in active:
        style_vehicle(
            traci,
            emergency_id,
            color=(0, 220, 255, 255),
            shape_class="emergency",
        )

    # Suspicious / flagged vehicle appears after step 120
    flagged_id = roles.get("flagged")
    if flagged_id in active and step >= 120:
        style_vehicle(
            traci,
            flagged_id,
            color=(255, 200, 0, 255),
            shape_class="passenger",
        )

    # Malicious / rejected vehicle appears after step 180
    malicious_id = roles.get("malicious")
    if malicious_id in active and step >= 180:
        style_vehicle(
            traci,
            malicious_id,
            color=(255, 0, 0, 255),
            shape_class="passenger",
        )

    return roles


def add_dynamic_message_markers(traci, step):
    """
    Adds simple POI markers to show the TrustV2X message-decision story.
    These markers are visual only.
    """
    cx, cy, xmin, ymin, xmax, ymax = get_network_centre(traci)

    width = xmax - xmin
    height = ymax - ymin

    if step >= 120:
        add_or_update_poi(
            traci,
            "CAM_MESSAGES_ACTIVE",
            cx - width * 0.15,
            cy - height * 0.12,
            color=(0, 180, 255, 255),
            poi_type="CAM Broadcast Active",
            layer=25,
        )

    if step >= 180:
        add_or_update_poi(
            traci,
            "TRUSTV2X_FLAG_DECISION",
            cx + width * 0.12,
            cy - height * 0.10,
            color=(255, 200, 0, 255),
            poi_type="TrustV2X: FLAG",
            layer=25,
        )

    if step >= 240:
        add_or_update_poi(
            traci,
            "TRUSTV2X_REJECT_DECISION",
            cx + width * 0.18,
            cy - height * 0.14,
            color=(255, 0, 0, 255),
            poi_type="TrustV2X: REJECT",
            layer=25,
        )

    if step >= 300:
        add_or_update_poi(
            traci,
            "TRUSTV2X_ACCEPT_DECISION",
            cx - width * 0.20,
            cy + height * 0.14,
            color=(0, 220, 0, 255),
            poi_type="TrustV2X: ACCEPT",
            layer=25,
        )


def save_screenshot(traci, step):
    """
    Saves SUMO-GUI screenshot for paper/demo use.
    """
    try:
        screenshot_path = SCREENSHOT_DIR / f"trustv2x_4x4_cits_step_{step}.png"
        traci.gui.screenshot(VIEW_ID, str(screenshot_path))
        print(f"[OK] Screenshot saved: {screenshot_path}")
    except Exception as e:
        print(f"[WARNING] Screenshot failed at step {step}: {e}")


def main():
    if not SUMO_CONFIG.exists():
        raise FileNotFoundError(
            f"SUMO config not found: {SUMO_CONFIG}\n"
            "Check the path to your existing 4x4 trustv2x.sumocfg file."
        )

    traci = import_traci()
    sumo_gui = get_sumo_gui_binary()

    print("=" * 70)
    print("TrustV2X 4x4 C-ITS Visual Overlay")
    print("=" * 70)
    print(f"[INFO] SUMO config: {SUMO_CONFIG}")
    print(f"[INFO] SUMO-GUI: {sumo_gui}")
    print("[INFO] This is a visual-only demonstration.")
    print("[INFO] It does not retrain models or overwrite main TrustV2X results.")
    print("[INFO] Vehicle class, route, speed, lane and network topology are not changed.")

    traci.start([
        sumo_gui,
        "-c",
        str(SUMO_CONFIG),
        "--start",
        "--quit-on-end",
        "--delay",
        "80",
        "--output-prefix",
        "visual_",
    ])

    set_gui_view(traci)

    roles = {}
    cits_info_added = False

    for step in range(SIMULATION_STEPS + 1):
        traci.simulationStep()

        if not cits_info_added:
            add_cits_components(traci)
            cits_info_added = True

        roles = apply_vehicle_visual_roles(traci, step, roles)

        add_dynamic_message_markers(traci, step)

        if step in SCREENSHOT_STEPS:
            save_screenshot(traci, step)

        if step % 100 == 0:
            print(f"[INFO] Visual simulation step: {step}")

        time.sleep(0.01)

    traci.close()

    print("\n" + "=" * 70)
    print("[OK] TrustV2X 4x4 C-ITS visual demo complete")
    print("=" * 70)
    print(f"[OK] Screenshots saved in: {SCREENSHOT_DIR}")

    print("\nSuggested figure caption:")
    print(
        "SUMO-based 4x4 C-ITS visual scenario showing RSU/MEC placement, "
        "CAM-style message exchange, road-hazard context, and TrustV2X "
        "accept/flag/reject decision outcomes."
    )


if __name__ == "__main__":
    main()