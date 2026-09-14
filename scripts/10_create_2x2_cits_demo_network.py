import os
import math
import subprocess
from pathlib import Path


# ============================================================
# TrustV2X - Step 10
# Create a New 2x2 C-ITS SUMO Demo Network
# ============================================================
#
# Purpose:
# - Create a separate realistic-looking SUMO network.
# - Keep the existing 4x4 experiment untouched.
# - Build a compact C-ITS visual demo with:
#   central roundabout, junctions, traffic lights, bridge route,
#   alternative routes, RSU, MEC/TrustV2X, hazard marker,
#   emergency vehicle and coloured vehicle types.
#
# New corrections included:
# - Overhead bridge / bypass route is clearly marked.
# - Bridge vehicles have separate colour.
# - Side legend panel is added.
# - C-ITS elements are labelled using POIs.
# - Hazard/blockage zone is visually separated from TrustV2X reject markers.
#
# This does NOT modify the existing 4x4 SUMO experiment.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEMO_ROOT = PROJECT_ROOT / "sumo" / "realistic_2x2_cits_demo"

NETWORK_DIR = DEMO_ROOT / "network"
ROUTES_DIR = DEMO_ROOT / "routes"
CONFIG_DIR = DEMO_ROOT / "config"
ADDITIONAL_DIR = DEMO_ROOT / "additional"
OUTPUT_DIR = DEMO_ROOT / "outputs"

NODES_FILE = NETWORK_DIR / "trustv2x_2x2_cits.nod.xml"
EDGES_FILE = NETWORK_DIR / "trustv2x_2x2_cits.edg.xml"
NET_FILE = NETWORK_DIR / "trustv2x_2x2_cits.net.xml"
ROUTE_FILE = ROUTES_DIR / "trustv2x_2x2_cits.rou.xml"
ADDITIONAL_FILE = ADDITIONAL_DIR / "trustv2x_2x2_cits.add.xml"
CONFIG_FILE = CONFIG_DIR / "trustv2x_2x2_cits.sumocfg"


def ensure_directories():
    for folder in [NETWORK_DIR, ROUTES_DIR, CONFIG_DIR, ADDITIONAL_DIR, OUTPUT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def get_sumo_binary(binary_name):
    """
    Finds SUMO binaries from SUMO_HOME or PATH.
    """
    sumo_home = os.environ.get("SUMO_HOME")

    if sumo_home:
        candidate = Path(sumo_home) / "bin" / binary_name
        if candidate.exists():
            return str(candidate)

    return binary_name


def write_nodes():
    """
    Creates a compact 2x2-style road network:
    - Central roundabout
    - Four approach roads
    - Outer alternative routes
    - Overhead bridge/bypass route
    - Traffic-light-controlled junction approach nodes
    """

    cx, cy = 50, 50
    r = 10

    roundabout_nodes = {
        "R0": (cx + r, cy),          # east
        "R1": (cx + 7, cy + 7),      # north-east
        "R2": (cx, cy + r),          # north
        "R3": (cx - 7, cy + 7),      # north-west
        "R4": (cx - r, cy),          # west
        "R5": (cx - 7, cy - 7),      # south-west
        "R6": (cx, cy - r),          # south
        "R7": (cx + 7, cy - 7),      # south-east
    }

    other_nodes = {
        # Main approach junctions
        "W": (0, 50),
        "E": (100, 50),
        "N": (50, 100),
        "S": (50, 0),

        # Outer 2x2 alternative network
        "NW": (0, 100),
        "NE": (100, 100),
        "SW": (0, 0),
        "SE": (100, 0),

        # Overhead bridge / bypass route
        "BW": (0, 72),
        "BE": (100, 72),
    }

    with open(NODES_FILE, "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<nodes>
""")

        for node_id, (x, y) in other_nodes.items():
            if node_id in ["W", "E", "N", "S"]:
                node_type = "traffic_light"
            else:
                node_type = "priority"

            f.write(f'    <node id="{node_id}" x="{x}" y="{y}" type="{node_type}"/>\n')

        for node_id, (x, y) in roundabout_nodes.items():
            f.write(f'    <node id="{node_id}" x="{x}" y="{y}" type="priority"/>\n')

        f.write("</nodes>\n")

    print(f"[OK] Nodes written: {NODES_FILE}")


def edge_xml(edge_id, from_node, to_node, speed=13.89, lanes=1, priority=1):
    return (
        f'    <edge id="{edge_id}" from="{from_node}" to="{to_node}" '
        f'priority="{priority}" numLanes="{lanes}" speed="{speed}"/>\n'
    )


def write_edges():
    """
    Creates:
    - Roundabout circulation
    - Main roads
    - Outer alternative routes
    - Overhead bridge / bypass route

    Note:
    The bridge route visually crosses the upper part of the network,
    but it is not connected to lower crossing roads. This represents
    grade separation, not a flat traffic conflict.
    """

    edges = []

    # Clockwise roundabout circulation
    ring = ["R0", "R7", "R6", "R5", "R4", "R3", "R2", "R1", "R0"]

    for i in range(len(ring) - 1):
        edge_id = f"{ring[i]}_{ring[i+1]}"
        edges.append(edge_xml(edge_id, ring[i], ring[i+1], speed=8.0, lanes=1, priority=3))

    # Main approaches into and out of the roundabout
    approaches = [
        ("W_R4", "W", "R4"),
        ("R4_W", "R4", "W"),

        ("E_R0", "E", "R0"),
        ("R0_E", "R0", "E"),

        ("N_R2", "N", "R2"),
        ("R2_N", "R2", "N"),

        ("S_R6", "S", "R6"),
        ("R6_S", "R6", "S"),
    ]

    for edge_id, src, dst in approaches:
        edges.append(edge_xml(edge_id, src, dst, speed=13.89, lanes=1, priority=2))

    # Outer alternative 2x2 road network
    outer_edges = [
        ("W_NW", "W", "NW"), ("NW_W", "NW", "W"),
        ("NW_NE", "NW", "NE"), ("NE_NW", "NE", "NW"),
        ("NE_E", "NE", "E"), ("E_NE", "E", "NE"),

        ("W_SW", "W", "SW"), ("SW_W", "SW", "W"),
        ("SW_SE", "SW", "SE"), ("SE_SW", "SE", "SW"),
        ("SE_E", "SE", "E"), ("E_SE", "E", "SE"),

        ("N_NW", "N", "NW"), ("NW_N", "NW", "N"),
        ("N_NE", "N", "NE"), ("NE_N", "NE", "N"),

        ("S_SW", "S", "SW"), ("SW_S", "SW", "S"),
        ("S_SE", "S", "SE"), ("SE_S", "SE", "S"),
    ]

    for edge_id, src, dst in outer_edges:
        edges.append(edge_xml(edge_id, src, dst, speed=12.0, lanes=1, priority=1))

    # Overhead bridge / bypass route
    bridge_edges = [
        ("W_BW", "W", "BW"),
        ("BW_BE_BRIDGE", "BW", "BE"),
        ("BE_E", "BE", "E"),

        ("E_BE", "E", "BE"),
        ("BE_BW_BRIDGE", "BE", "BW"),
        ("BW_W", "BW", "W"),
    ]

    for edge_id, src, dst in bridge_edges:
        edges.append(edge_xml(edge_id, src, dst, speed=16.67, lanes=1, priority=4))

    with open(EDGES_FILE, "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<edges>
""")
        for edge in edges:
            f.write(edge)
        f.write("</edges>\n")

    print(f"[OK] Edges written: {EDGES_FILE}")


def circle_shape(cx, cy, radius, n=72):
    points = []

    for i in range(n):
        angle = 2 * math.pi * i / n
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append(f"{x:.2f},{y:.2f}")

    return " ".join(points)


def rect_shape(x1, y1, x2, y2):
    return f"{x1},{y1} {x2},{y1} {x2},{y2} {x1},{y2}"


def box_shape(x, y, w=3.5, h=3.5):
    return f"{x},{y} {x+w},{y} {x+w},{y+h} {x},{y+h}"


def write_additional():
    """
    Adds visual C-ITS infrastructure:
    - RSU
    - MEC/TrustV2X Edge
    - C-ITS control centre
    - CAM broadcast marker
    - DENM hazard warning marker
    - Road hazard / obstacle zone
    - Overhead bridge / bypass label
    - Side legend panel
    """

    rsu_range_shape = circle_shape(50, 50, 32)

    with open(ADDITIONAL_FILE, "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<additional>

    <!-- ===================================================== -->
    <!-- TrustV2X 2x2 C-ITS visual environment                 -->
    <!-- ===================================================== -->

""")

        # =====================================================
        # RSU communication zone
        # =====================================================
        f.write(
            f'    <poly id="RSU_1_COMMUNICATION_RANGE" type="RSU communication range" '
            f'color="0,90,255,45" fill="true" layer="0" shape="{rsu_range_shape}"/>\n'
        )

        # =====================================================
        # Core C-ITS infrastructure markers
        # =====================================================
        f.write("""
    <poi id="RSU_1" type="RSU-1" color="0,80,255" x="50" y="50" layer="20"/>
    <poi id="MEC_TRUSTV2X_EDGE" type="MEC / TrustV2X Edge" color="160,0,255" x="62" y="62" layer="20"/>
    <poi id="CITS_CONTROL_CENTRE" type="C-ITS Control Centre" color="255,140,0" x="35" y="68" layer="20"/>

    <poi id="CAM_ACTIVE" type="CAM Broadcast Zone" color="0,180,255" x="42" y="40" layer="20"/>
    <poi id="DENM_WARNING" type="DENM Hazard Warning" color="255,0,0" x="82" y="54" layer="20"/>

    <poi id="TRUSTV2X_ACCEPT" type="TrustV2X ACCEPT" color="0,220,0" x="30" y="72" layer="20"/>
    <poi id="TRUSTV2X_FLAG" type="TrustV2X FLAG" color="255,200,0" x="67" y="35" layer="20"/>
    <poi id="TRUSTV2X_REJECT" type="TrustV2X REJECT" color="255,0,0" x="80" y="35" layer="20"/>

    <poi id="ROAD_HAZARD_OBSTACLE" type="Road Hazard / Obstacle" color="255,0,0" x="76" y="50" layer="20"/>
""")

        # =====================================================
        # Road hazard / blockage zone
        # =====================================================
        f.write(
            f'    <poly id="ROAD_BLOCKAGE_ZONE" type="road blockage / accident" '
            f'color="255,0,0,120" fill="true" layer="10" shape="72,47 82,47 82,53 72,53"/>\n'
        )

        # =====================================================
        # Overhead bridge / bypass route
        # This explicitly explains the non-conflict crossing.
        # =====================================================
        f.write(
            f'    <poly id="OVERHEAD_BRIDGE_DECK" type="Overhead Bridge / Alternative Route" '
            f'color="140,140,140,90" fill="true" layer="2" shape="0,69 100,69 100,75 0,75"/>\n'
        )

        f.write("""
    <poi id="OVERHEAD_BRIDGE_LABEL" type="Overhead Bridge / Alternative Route" color="90,90,90" x="50" y="78" layer="25"/>
""")

        # =====================================================
        # Optional roadside / pedestrian markers
        # =====================================================
        f.write("""
    <poi id="PEDESTRIAN_CROSSING_W" type="Pedestrian Crossing" color="255,255,255" x="12" y="50" layer="20"/>
    <poi id="PEDESTRIAN_CROSSING_E" type="Pedestrian Crossing" color="255,255,255" x="88" y="50" layer="20"/>
""")

        # =====================================================
        # Side legend panel
        # Placed to the right side of the network.
        # The coloured boxes remain visible even if SUMO labels are disabled.
        # To show text, enable POI labels/name display in SUMO-GUI settings.
        # =====================================================
        f.write(
            f'    <poly id="LEGEND_PANEL_BG" type="TrustV2X / C-ITS Legend" '
            f'color="245,245,245,230" fill="true" layer="100" '
            f'shape="106,8 148,8 148,96 106,96"/>\n'
        )

        f.write("""
    <poi id="LEGEND_TITLE" type="TrustV2X / C-ITS Legend" color="0,0,0" x="127" y="93" layer="111"/>
""")

        legend_items = [
            ("LEG_TRUSTED",      "Trusted Vehicle",                  "0,180,0",     126, 86),
            ("LEG_BACKGROUND",   "Background Vehicle",               "160,160,160", 126, 80),
            ("LEG_BRIDGE",       "Bridge / Bypass Vehicle",          "190,190,210", 126, 74),
            ("LEG_EMERGENCY",    "Emergency / Ambulance",            "0,220,255",   126, 68),
            ("LEG_FLAGGED",      "Suspicious / Flagged",             "255,200,0",   126, 62),
            ("LEG_MALICIOUS",    "Malicious / Rejected",             "255,0,0",     126, 56),
            ("LEG_RSU",          "RSU",                              "0,80,255",    126, 50),
            ("LEG_MEC",          "MEC / TrustV2X Edge",              "160,0,255",   126, 44),
            ("LEG_CONTROL",      "C-ITS Control Centre",             "255,140,0",   126, 38),
            ("LEG_CAM",          "CAM Broadcast",                    "0,180,255",   126, 32),
            ("LEG_DENM",         "DENM Hazard Warning",              "255,0,0",     126, 26),
            ("LEG_BLOCKAGE",     "Road Blockage / Obstacle",         "200,0,0",     126, 20),
            ("LEG_RANGE",        "RSU Communication Range",          "0,90,255",    126, 14),
        ]

        for item_id, item_type, color, label_x, label_y in legend_items:
            r, g, b = color.split(",")

            # Coloured box
            f.write(
                f'    <poly id="{item_id}_BOX" type="{item_type}" '
                f'color="{r},{g},{b},255" fill="true" layer="110" '
                f'shape="{box_shape(label_x - 14, label_y - 1.6, 3.2, 3.2)}"/>\n'
            )

            # Label anchor as POI
            f.write(
                f'    <poi id="{item_id}_LABEL" type="{item_type}" color="0,0,0" '
                f'x="{label_x}" y="{label_y}" layer="111"/>\n'
            )

        f.write("\n</additional>\n")

    print(f"[OK] Additional C-ITS visual elements written: {ADDITIONAL_FILE}")


def write_routes():
    """
    Creates vehicle types, flows and routes for the visual C-ITS scenario.
    """

    with open(ROUTE_FILE, "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<routes>

    <!-- ===================================================== -->
    <!-- Vehicle visual types                                  -->
    <!-- ===================================================== -->

    <vType id="trusted_car" vClass="passenger" guiShape="passenger" color="0,180,0" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.89"/>
    <vType id="background_car" vClass="passenger" guiShape="passenger" color="160,160,160" accel="2.4" decel="4.2" sigma="0.6" length="5" maxSpeed="13.89"/>
    <vType id="bridge_vehicle" vClass="passenger" guiShape="passenger" color="190,190,210" accel="2.2" decel="4.0" sigma="0.4" length="5" maxSpeed="14.50"/>
    <vType id="flagged_car" vClass="passenger" guiShape="passenger" color="255,200,0" accel="2.3" decel="4.1" sigma="0.7" length="5" maxSpeed="12.00"/>
    <vType id="malicious_car" vClass="passenger" guiShape="passenger" color="255,0,0" accel="3.0" decel="4.5" sigma="0.9" length="5" maxSpeed="16.00"/>
    <vType id="emergency_vehicle" vClass="emergency" guiShape="emergency" color="0,220,255" accel="3.5" decel="5.0" sigma="0.3" length="6" maxSpeed="18.00"/>
    <vType id="bus_vehicle" vClass="bus" guiShape="bus" color="80,80,255" accel="1.5" decel="3.5" sigma="0.5" length="10" maxSpeed="11.00"/>

    <!-- ===================================================== -->
    <!-- Roundabout routes                                     -->
    <!-- ===================================================== -->

    <route id="west_to_east_roundabout" edges="W_R4 R4_R3 R3_R2 R2_R1 R1_R0 R0_E"/>
    <route id="east_to_west_roundabout" edges="E_R0 R0_R7 R7_R6 R6_R5 R5_R4 R4_W"/>
    <route id="north_to_south_roundabout" edges="N_R2 R2_R1 R1_R0 R0_R7 R7_R6 R6_S"/>
    <route id="south_to_north_roundabout" edges="S_R6 R6_R5 R5_R4 R4_R3 R3_R2 R2_N"/>

    <!-- ===================================================== -->
    <!-- Alternative routes around the outer network            -->
    <!-- ===================================================== -->

    <route id="west_to_east_north_alt" edges="W_NW NW_NE NE_E"/>
    <route id="west_to_east_south_alt" edges="W_SW SW_SE SE_E"/>
    <route id="east_to_west_north_alt" edges="E_NE NE_NW NW_W"/>
    <route id="east_to_west_south_alt" edges="E_SE SE_SW SW_W"/>

    <!-- ===================================================== -->
    <!-- Overhead bridge / bypass route                         -->
    <!-- ===================================================== -->

    <route id="west_to_east_bridge" edges="W_BW BW_BE_BRIDGE BE_E"/>
    <route id="east_to_west_bridge" edges="E_BE BE_BW_BRIDGE BW_W"/>

    <!-- ===================================================== -->
    <!-- Regular traffic flows                                  -->
    <!-- ===================================================== -->

    <flow id="flow_trusted_we" type="trusted_car" route="west_to_east_roundabout" begin="0" end="360" period="4"/>
    <flow id="flow_trusted_ew" type="trusted_car" route="east_to_west_roundabout" begin="5" end="360" period="5"/>
    <flow id="flow_trusted_ns" type="trusted_car" route="north_to_south_roundabout" begin="10" end="360" period="6"/>
    <flow id="flow_trusted_sn" type="trusted_car" route="south_to_north_roundabout" begin="12" end="360" period="6"/>

    <!-- Background and alternative-route traffic -->
    <flow id="flow_background_north_alt" type="background_car" route="west_to_east_north_alt" begin="20" end="360" period="9"/>
    <flow id="flow_background_south_alt" type="background_car" route="east_to_west_south_alt" begin="25" end="360" period="10"/>

    <!-- Bridge traffic uses a distinct bridge vehicle colour -->
    <flow id="flow_bridge_alt" type="bridge_vehicle" route="west_to_east_bridge" begin="30" end="360" period="12"/>
    <flow id="flow_bridge_alt_reverse" type="bridge_vehicle" route="east_to_west_bridge" begin="45" end="360" period="15"/>

    <!-- ===================================================== -->
    <!-- Special scenario vehicles                              -->
    <!-- ===================================================== -->

    <vehicle id="ambulance_01" type="emergency_vehicle" route="south_to_north_roundabout" depart="35" departLane="best"/>
    <vehicle id="bus_01" type="bus_vehicle" route="west_to_east_north_alt" depart="50" departLane="best"/>

    <vehicle id="flagged_vehicle_01" type="flagged_car" route="east_to_west_roundabout" depart="80" departLane="best"/>
    <vehicle id="malicious_vehicle_01" type="malicious_car" route="west_to_east_roundabout" depart="110" departLane="best"/>

    <!-- These vehicles represent vehicles that would use alternative routes -->
    <vehicle id="rerouted_vehicle_01" type="trusted_car" route="west_to_east_bridge" depart="140" departLane="best"/>
    <vehicle id="rerouted_vehicle_02" type="trusted_car" route="west_to_east_south_alt" depart="150" departLane="best"/>
    <vehicle id="rerouted_vehicle_03" type="trusted_car" route="west_to_east_north_alt" depart="160" departLane="best"/>

</routes>
""")

    print(f"[OK] Routes written: {ROUTE_FILE}")


def write_config():
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>

    <input>
        <net-file value="../network/{NET_FILE.name}"/>
        <route-files value="../routes/{ROUTE_FILE.name}"/>
        <additional-files value="../additional/{ADDITIONAL_FILE.name}"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="360"/>
        <step-length value="1"/>
    </time>

    <processing>
        <time-to-teleport value="-1"/>
        <collision.action value="warn"/>
    </processing>

    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
    </report>

</configuration>
""")

    print(f"[OK] SUMO config written: {CONFIG_FILE}")


def build_network_with_netconvert():
    netconvert = get_sumo_binary("netconvert.exe")

    command = [
        netconvert,
        "-n",
        str(NODES_FILE),
        "-e",
        str(EDGES_FILE),
        "-o",
        str(NET_FILE),
        "--no-turnarounds",
        "true",
        "--junctions.join",
        "false",
        "--tls.guess",
        "true",
        "--default.junctions.keep-clear",
        "true",
    ]

    print("\n[INFO] Building SUMO network with netconvert...")
    print(" ".join(command))

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("\n[ERROR] netconvert failed.")
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("SUMO netconvert failed. Check generated nodes/edges.")

    print(f"[OK] SUMO network created: {NET_FILE}")

    if result.stderr.strip():
        print("\n[netconvert messages]")
        print(result.stderr)


def main():
    print("=" * 70)
    print("TrustV2X 2x2 C-ITS Demo Network Creator")
    print("=" * 70)

    ensure_directories()

    write_nodes()
    write_edges()
    write_additional()
    write_routes()
    build_network_with_netconvert()
    write_config()

    print("\n" + "=" * 70)
    print("[OK] TrustV2X 2x2 C-ITS demo network created")
    print("=" * 70)

    print("\nOpen the new demo in SUMO-GUI using:")
    print(f'sumo-gui -c "{CONFIG_FILE}"')

    print("\nOr with full Windows SUMO path:")
    print(f'& "C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\sumo-gui.exe" -c "{CONFIG_FILE}"')

    print("\nWhat you should see:")
    print("- compact 2x2 road network")
    print("- central roundabout")
    print("- traffic-light approach junctions")
    print("- outer alternative routes")
    print("- overhead bridge / bypass route clearly marked")
    print("- RSU marker and communication zone")
    print("- MEC / TrustV2X Edge marker")
    print("- C-ITS Control Centre marker")
    print("- hazard / road blockage marker")
    print("- side legend panel")
    print("- green trusted vehicles")
    print("- grey background vehicles")
    print("- grey-blue bridge/bypass vehicles")
    print("- red malicious vehicle")
    print("- yellow flagged vehicle")
    print("- cyan emergency vehicle")
    print("- blue bus/background traffic")

    print("\nNote:")
    print("If text labels are not visible in SUMO-GUI, enable POI/object labels in the SUMO visual settings.")
    print("For final paper figures, use the SUMO screenshot and add clean typed labels externally if needed.")


if __name__ == "__main__":
    main()
    