import os
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path


# ============================================================
# TrustV2X - Step 09
# Prepare and Run 4x4 C-ITS Visual Demo
# ============================================================
#
# Purpose:
# - Keep the original 4x4 quantitative experiment untouched.
# - Create a NEW visual SUMO configuration file.
# - Add C-ITS visual infrastructure:
#   RSU, MEC/TrustV2X Edge, C-ITS control centre,
#   road hazard marker, communication zone and decision markers.
# - Run SUMO-GUI with clear vehicle colours.
#
# This does NOT modify:
# - the original network
# - the original route file
# - the dataset
# - ML/DL results
# - TrustV2X scoring results
#
# It creates a separate visual demo config:
# sumo/visual_4x4_cits_demo/config/trustv2x_4x4_cits_visual.sumocfg
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_SUMO_CONFIG = PROJECT_ROOT / "sumo" / "config" / "trustv2x.sumocfg"

VISUAL_ROOT = PROJECT_ROOT / "sumo" / "visual_4x4_cits_demo"
VISUAL_CONFIG_DIR = VISUAL_ROOT / "config"
VISUAL_ADDITIONAL_DIR = VISUAL_ROOT / "additional"
VISUAL_SCREENSHOT_DIR = VISUAL_ROOT / "screenshots"

VISUAL_CONFIG_FILE = VISUAL_CONFIG_DIR / "trustv2x_4x4_cits_visual.sumocfg"
VISUAL_ADDITIONAL_FILE = VISUAL_ADDITIONAL_DIR / "trustv2x_4x4_cits_visual.add.xml"

SIMULATION_STEPS = 670
SCREENSHOT_STEPS = [50, 150, 300, 500, 660]

VIEW_ID = "View #0"


def ensure_directories():
    for folder in [VISUAL_CONFIG_DIR, VISUAL_ADDITIONAL_DIR, VISUAL_SCREENSHOT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


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


def resolve_path(base_dir, path_text):
    path_obj = Path(path_text)

    if path_obj.is_absolute():
        return path_obj

    return (base_dir / path_obj).resolve()


def parse_original_sumocfg():
    """
    Reads the original 4x4 SUMO config and extracts net-file and route-files.
    """
    if not ORIGINAL_SUMO_CONFIG.exists():
        raise FileNotFoundError(
            f"Original SUMO config not found: {ORIGINAL_SUMO_CONFIG}"
        )

    tree = ET.parse(ORIGINAL_SUMO_CONFIG)
    root = tree.getroot()

    input_node = root.find("input")

    if input_node is None:
        raise RuntimeError("Could not find <input> section in original SUMO config.")

    net_node = input_node.find("net-file")
    route_node = input_node.find("route-files")
    additional_node = input_node.find("additional-files")

    if net_node is None:
        raise RuntimeError("Could not find <net-file> in original SUMO config.")

    if route_node is None:
        raise RuntimeError("Could not find <route-files> in original SUMO config.")

    original_config_dir = ORIGINAL_SUMO_CONFIG.parent

    net_file = resolve_path(original_config_dir, net_node.attrib["value"])

    route_files = []
    for route_text in route_node.attrib["value"].split(","):
        route_files.append(resolve_path(original_config_dir, route_text.strip()))

    existing_additional_files = []
    if additional_node is not None:
        for add_text in additional_node.attrib["value"].split(","):
            add_text = add_text.strip()
            if add_text:
                existing_additional_files.append(resolve_path(original_config_dir, add_text))

    return net_file, route_files, existing_additional_files


def get_net_boundary_from_net_file(net_file):
    """
    Reads network boundary from the SUMO .net.xml file.
    """
    tree = ET.parse(net_file)
    root = tree.getroot()

    location = root.find("location")

    if location is not None and "convBoundary" in location.attrib:
        values = location.attrib["convBoundary"].split(",")
        xmin, ymin, xmax, ymax = [float(v) for v in values]
        return xmin, ymin, xmax, ymax

    # Fallback: estimate boundary from lane shapes
    xs = []
    ys = []

    for lane in root.findall(".//lane"):
        shape = lane.attrib.get("shape", "")
        if not shape:
            continue

        for point in shape.split():
            if "," in point:
                x, y = point.split(",")[:2]
                xs.append(float(x))
                ys.append(float(y))

    if not xs or not ys:
        raise RuntimeError("Could not determine network boundary from .net.xml file.")

    return min(xs), min(ys), max(xs), max(ys)


def circle_shape(cx, cy, radius, n=96):
    points = []

    for i in range(n):
        angle = 2 * math.pi * i / n
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append(f"{x:.2f},{y:.2f}")

    return " ".join(points)


def rect_shape(cx, cy, size):
    half = size / 2
    x1 = cx - half
    y1 = cy - half
    x2 = cx + half
    y2 = cy + half

    return f"{x1:.2f},{y1:.2f} {x2:.2f},{y1:.2f} {x2:.2f},{y2:.2f} {x1:.2f},{y2:.2f}"


def rectangle_between(x1, y1, x2, y2, width):
    """
    Simple visual rectangle for hazard/route marker.
    """
    return (
        f"{x1:.2f},{y1 - width:.2f} "
        f"{x2:.2f},{y1 - width:.2f} "
        f"{x2:.2f},{y2 + width:.2f} "
        f"{x1:.2f},{y2 + width:.2f}"
    )


def write_visual_additional_file(net_file):
    """
    Creates a proper SUMO additional file for visible C-ITS infrastructure.
    """
    xmin, ymin, xmax, ymax = get_net_boundary_from_net_file(net_file)

    width = xmax - xmin
    height = ymax - ymin

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    marker_size = max(min(width, height) * 0.025, 8)
    rsu_range = min(width, height) * 0.22

    rsu_x = cx
    rsu_y = cy

    mec_x = cx + width * 0.08
    mec_y = cy + height * 0.08

    control_x = cx - width * 0.12
    control_y = cy + height * 0.12

    hazard_x = cx + width * 0.18
    hazard_y = cy

    accept_x = cx - width * 0.20
    accept_y = cy + height * 0.16

    flag_x = cx + width * 0.12
    flag_y = cy - height * 0.13

    reject_x = cx + width * 0.20
    reject_y = cy - height * 0.16

    cam_x = cx - width * 0.16
    cam_y = cy - height * 0.12

    with open(VISUAL_ADDITIONAL_FILE, "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<additional>

    <!-- ===================================================== -->
    <!-- TrustV2X 4x4 C-ITS visual infrastructure overlay       -->
    <!-- Visual only: does not change routes or traffic data     -->
    <!-- ===================================================== -->

""")

        # Background context polygon
        f.write(
            f'    <poly id="CITS_DEMO_BACKGROUND" type="C-ITS simulation area" '
            f'color="220,235,255,30" fill="true" layer="-5" '
            f'shape="{xmin:.2f},{ymin:.2f} {xmax:.2f},{ymin:.2f} {xmax:.2f},{ymax:.2f} {xmin:.2f},{ymax:.2f}"/>\n'
        )

        # RSU communication zone
        f.write(
            f'    <poly id="RSU_1_COMMUNICATION_RANGE" type="RSU communication range" '
            f'color="0,90,255,45" fill="true" layer="0" '
            f'shape="{circle_shape(rsu_x, rsu_y, rsu_range)}"/>\n'
        )

        # Larger visible infrastructure markers as polygons
        f.write(
            f'    <poly id="RSU_1_MARKER" type="RSU-1" color="0,80,255,255" '
            f'fill="true" layer="40" shape="{rect_shape(rsu_x, rsu_y, marker_size)}"/>\n'
        )

        f.write(
            f'    <poly id="MEC_TRUSTV2X_MARKER" type="MEC / TrustV2X Edge" color="160,0,255,255" '
            f'fill="true" layer="40" shape="{rect_shape(mec_x, mec_y, marker_size)}"/>\n'
        )

        f.write(
            f'    <poly id="CITS_CONTROL_MARKER" type="C-ITS Control Centre" color="255,140,0,255" '
            f'fill="true" layer="40" shape="{rect_shape(control_x, control_y, marker_size)}"/>\n'
        )

        f.write(
            f'    <poly id="ROAD_HAZARD_MARKER" type="Road Hazard / Obstacle" color="255,0,0,180" '
            f'fill="true" layer="45" shape="{rect_shape(hazard_x, hazard_y, marker_size * 1.4)}"/>\n'
        )

        # Decision markers
        f.write(
            f'    <poly id="TRUSTV2X_ACCEPT_MARKER" type="TrustV2X ACCEPT" color="0,220,0,200" '
            f'fill="true" layer="45" shape="{rect_shape(accept_x, accept_y, marker_size)}"/>\n'
        )

        f.write(
            f'    <poly id="TRUSTV2X_FLAG_MARKER" type="TrustV2X FLAG" color="255,200,0,220" '
            f'fill="true" layer="45" shape="{rect_shape(flag_x, flag_y, marker_size)}"/>\n'
        )

        f.write(
            f'    <poly id="TRUSTV2X_REJECT_MARKER" type="TrustV2X REJECT" color="255,0,0,220" '
            f'fill="true" layer="45" shape="{rect_shape(reject_x, reject_y, marker_size)}"/>\n'
        )

        f.write(
            f'    <poly id="CAM_BROADCAST_MARKER" type="CAM Broadcast Active" color="0,180,255,180" '
            f'fill="true" layer="45" shape="{rect_shape(cam_x, cam_y, marker_size)}"/>\n'
        )

        # Visual guide line / corridor from hazard to RSU
        f.write(
            f'    <poly id="DENM_WARNING_VISUAL_FLOW" type="DENM warning to RSU" color="255,0,0,90" '
            f'fill="true" layer="5" '
            f'shape="{rectangle_between(rsu_x, rsu_y, hazard_x, hazard_y, marker_size * 0.25)}"/>\n'
        )

        # Small POIs for selectable labels in SUMO-GUI
        f.write(f'    <poi id="RSU_1" type="RSU-1" color="0,80,255" x="{rsu_x:.2f}" y="{rsu_y:.2f}" layer="60"/>\n')
        f.write(f'    <poi id="MEC_TRUSTV2X_EDGE" type="MEC / TrustV2X Edge" color="160,0,255" x="{mec_x:.2f}" y="{mec_y:.2f}" layer="60"/>\n')
        f.write(f'    <poi id="CITS_CONTROL_CENTRE" type="C-ITS Control Centre" color="255,140,0" x="{control_x:.2f}" y="{control_y:.2f}" layer="60"/>\n')
        f.write(f'    <poi id="ROAD_HAZARD_OBSTACLE" type="Road Hazard / Obstacle" color="255,0,0" x="{hazard_x:.2f}" y="{hazard_y:.2f}" layer="60"/>\n')
        f.write(f'    <poi id="CAM_ACTIVE" type="CAM Broadcast Active" color="0,180,255" x="{cam_x:.2f}" y="{cam_y:.2f}" layer="60"/>\n')
        f.write(f'    <poi id="TRUSTV2X_ACCEPT" type="TrustV2X ACCEPT" color="0,220,0" x="{accept_x:.2f}" y="{accept_y:.2f}" layer="60"/>\n')
        f.write(f'    <poi id="TRUSTV2X_FLAG" type="TrustV2X FLAG" color="255,200,0" x="{flag_x:.2f}" y="{flag_y:.2f}" layer="60"/>\n')
        f.write(f'    <poi id="TRUSTV2X_REJECT" type="TrustV2X REJECT" color="255,0,0" x="{reject_x:.2f}" y="{reject_y:.2f}" layer="60"/>\n')

        f.write("\n</additional>\n")

    print(f"[OK] Visual additional file created: {VISUAL_ADDITIONAL_FILE}")


def write_visual_sumocfg(net_file, route_files, existing_additional_files):
    """
    Creates a new SUMO config file for the visual 4x4 C-ITS demo.
    """
    all_additional = existing_additional_files + [VISUAL_ADDITIONAL_FILE]

    route_value = ",".join(str(path) for path in route_files)
    additional_value = ",".join(str(path) for path in all_additional)

    with open(VISUAL_CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>

    <input>
        <net-file value="{net_file}"/>
        <route-files value="{route_value}"/>
        <additional-files value="{additional_value}"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="{SIMULATION_STEPS}"/>
        <step-length value="1"/>
    </time>

    <processing>
        <time-to-teleport value="-1"/>
        <collision.action value="warn"/>
    </processing>

    <gui_only>
        <start value="true"/>
    </gui_only>

    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
    </report>

</configuration>
""")

    print(f"[OK] Visual SUMO config created: {VISUAL_CONFIG_FILE}")


def read_net_boundary_from_traci(traci):
    """
    Handles SUMO boundary output format:
    either ((xmin, ymin), (xmax, ymax)) or (xmin, ymin, xmax, ymax).
    """
    boundary = traci.simulation.getNetBoundary()

    if len(boundary) == 2:
        (xmin, ymin), (xmax, ymax) = boundary
    elif len(boundary) == 4:
        xmin, ymin, xmax, ymax = boundary
    else:
        raise RuntimeError(f"Unexpected boundary format: {boundary}")

    return float(xmin), float(ymin), float(xmax), float(ymax)


def set_gui_view(traci):
    """
    Sets the full network boundary.
    Important: no forced high zoom, because that caused blank-looking screenshots.
    """
    try:
        xmin, ymin, xmax, ymax = read_net_boundary_from_traci(traci)

        width = xmax - xmin
        height = ymax - ymin

        pad_x = width * 0.08
        pad_y = height * 0.08

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


def style_vehicle(traci, veh_id, color, shape_class=None):
    """
    Strict visual-only vehicle styling.
    Does not change vehicle class, speed, route, lane or behaviour.
    """
    try:
        traci.vehicle.setColor(veh_id, color)
    except Exception:
        pass

    if shape_class:
        try:
            traci.vehicle.setShapeClass(veh_id, shape_class)
        except Exception:
            pass


def choose_vehicle_roles(active_vehicles, roles):
    active = list(active_vehicles)

    if not active:
        return roles

    if roles.get("malicious") not in active and len(active) >= 1:
        roles["malicious"] = active[0]

    if roles.get("flagged") not in active and len(active) >= 2:
        roles["flagged"] = active[1]

    if roles.get("emergency") not in active and len(active) >= 3:
        roles["emergency"] = active[2]

    return roles


def apply_vehicle_colours(traci, step, roles):
    active = traci.vehicle.getIDList()
    roles = choose_vehicle_roles(active, roles)

    for i, veh_id in enumerate(active):
        # Default safe/trusted vehicles
        style_vehicle(traci, veh_id, color=(0, 180, 0, 255), shape_class="passenger")

        # Some background traffic
        if i % 6 == 0:
            style_vehicle(traci, veh_id, color=(160, 160, 160, 255), shape_class="passenger")

    emergency_id = roles.get("emergency")
    if emergency_id in active:
        style_vehicle(traci, emergency_id, color=(0, 220, 255, 255), shape_class="emergency")

    flagged_id = roles.get("flagged")
    if flagged_id in active and step >= 120:
        style_vehicle(traci, flagged_id, color=(255, 200, 0, 255), shape_class="passenger")

    malicious_id = roles.get("malicious")
    if malicious_id in active and step >= 180:
        style_vehicle(traci, malicious_id, color=(255, 0, 0, 255), shape_class="passenger")

    return roles


def save_screenshot(traci, step):
    screenshot_path = VISUAL_SCREENSHOT_DIR / f"trustv2x_4x4_cits_visual_step_{step}.png"

    try:
        traci.gui.screenshot(VIEW_ID, str(screenshot_path))
        print(f"[OK] Screenshot saved: {screenshot_path}")
    except Exception as e:
        print(f"[WARNING] Screenshot failed at step {step}: {e}")


def run_visual_demo():
    traci = import_traci()
    sumo_gui = get_sumo_gui_binary()

    print("\n[INFO] Opening SUMO-GUI with new 4x4 C-ITS visual config...")
    print(f"[INFO] Config: {VISUAL_CONFIG_FILE}")

    traci.start([
        sumo_gui,
        "-c",
        str(VISUAL_CONFIG_FILE),
        "--start",
        "--delay",
        "120",
    ])

    time.sleep(1)
    set_gui_view(traci)

    roles = {}

    for step in range(SIMULATION_STEPS + 1):
        traci.simulationStep()

        roles = apply_vehicle_colours(traci, step, roles)

        if step in SCREENSHOT_STEPS:
            save_screenshot(traci, step)

        if step % 100 == 0:
            print(f"[INFO] Visual simulation step: {step}")

        time.sleep(0.01)

    print("\n[INFO] Simulation completed.")
    print("[INFO] SUMO-GUI will remain open until you press Enter here.")
    input("Press Enter to close SUMO-GUI...")

    traci.close()


def main():
    print("=" * 70)
    print("TrustV2X 4x4 C-ITS Visual Demo Preparation")
    print("=" * 70)

    ensure_directories()

    net_file, route_files, existing_additional_files = parse_original_sumocfg()

    print(f"[INFO] Original config: {ORIGINAL_SUMO_CONFIG}")
    print(f"[INFO] Existing net file: {net_file}")
    print(f"[INFO] Existing route files: {route_files}")

    write_visual_additional_file(net_file)
    write_visual_sumocfg(net_file, route_files, existing_additional_files)

    print("\n" + "=" * 70)
    print("[OK] New 4x4 C-ITS visual SUMO config created")
    print("=" * 70)
    print(f"[OK] New visual config: {VISUAL_CONFIG_FILE}")
    print(f"[OK] New visual overlay: {VISUAL_ADDITIONAL_FILE}")

    print("\nYou can open it manually with:")
    print(f'& "{get_sumo_gui_binary()}" -c "{VISUAL_CONFIG_FILE}"')

    print("\nNow starting preview...")
    run_visual_demo()

    print("\n" + "=" * 70)
    print("[OK] 4x4 C-ITS visual demo complete")
    print("=" * 70)
    print(f"[OK] Screenshots saved in: {VISUAL_SCREENSHOT_DIR}")


if __name__ == "__main__":
    main()