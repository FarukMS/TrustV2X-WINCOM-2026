import os
import math
import subprocess
from pathlib import Path


# ============================================================
# TrustV2X - Step 10
# Create Corrected Signalised 2x2 C-ITS SUMO Demo Network
# ============================================================
#
# Purpose:
# - Build a corrected 2x2 C-ITS visual/behavioural demo network.
# - Remove the overhead bridge conflict.
# - Replace the top crossing with a real traffic-light junction.
# - Include central roundabout, traffic lights, alternative routes,
#   road incident zone, RSU, MEC/TrustV2X edge, emergency vehicle,
#   police vehicle, CAM/DENM markers.
#
# Legend has been removed.
# You can add the legend manually to the final screenshot.
#
# This does NOT modify the existing 4x4 quantitative experiment.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEMO_ROOT = PROJECT_ROOT / "sumo" / "realistic_2x2_cits_signalised_demo"

NETWORK_DIR = DEMO_ROOT / "network"
ROUTES_DIR = DEMO_ROOT / "routes"
CONFIG_DIR = DEMO_ROOT / "config"
ADDITIONAL_DIR = DEMO_ROOT / "additional"
OUTPUT_DIR = DEMO_ROOT / "outputs"

NODES_FILE = NETWORK_DIR / "trustv2x_2x2_signalised.nod.xml"
EDGES_FILE = NETWORK_DIR / "trustv2x_2x2_signalised.edg.xml"
NET_FILE = NETWORK_DIR / "trustv2x_2x2_signalised.net.xml"
ROUTE_FILE = ROUTES_DIR / "trustv2x_2x2_signalised.rou.xml"
ADDITIONAL_FILE = ADDITIONAL_DIR / "trustv2x_2x2_signalised.add.xml"
CONFIG_FILE = CONFIG_DIR / "trustv2x_2x2_signalised.sumocfg"


def ensure_directories():
    for folder in [NETWORK_DIR, ROUTES_DIR, CONFIG_DIR, ADDITIONAL_DIR, OUTPUT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def get_sumo_binary(binary_name):
    sumo_home = os.environ.get("SUMO_HOME")

    if sumo_home:
        candidate = Path(sumo_home) / "bin" / binary_name
        if candidate.exists():
            return str(candidate)

    return binary_name


def write_nodes():
    """
    Creates:
    - 100m x 100m core road network
    - External spawn/exit nodes
    - Central roundabout
    - Four signalised junctions: W, E, N, S
    """

    cx, cy = 50, 50
    r = 10

    roundabout_nodes = {
        "R0": (cx + r, cy),
        "R1": (cx + 7, cy + 7),
        "R2": (cx, cy + r),
        "R3": (cx - 7, cy + 7),
        "R4": (cx - r, cy),
        "R5": (cx - 7, cy - 7),
        "R6": (cx, cy - r),
        "R7": (cx + 7, cy - 7),
    }

    other_nodes = {
        # External source/sink nodes
        "WI": (-20, 50),
        "EO": (120, 50),
        "NI": (50, 120),
        "SI": (50, -20),

        # Main signalised junctions
        "W": (0, 50),
        "E": (100, 50),
        "N": (50, 100),
        "S": (50, 0),

        # 2x2 corner network
        "NW": (0, 100),
        "NE": (100, 100),
        "SW": (0, 0),
        "SE": (100, 0),
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
    Important correction:
    The old direct top bypass/bridge crossing is removed.

    Top road now passes through:
        NW -> N -> NE

    This makes the top crossing a real signalised junction at N.
    Vehicles must interact with the traffic-light system.
    """

    edges = []

    # External source/sink links
    external_edges = [
        ("WI_W", "WI", "W"), ("W_WI", "W", "WI"),
        ("E_EO", "E", "EO"), ("EO_E", "EO", "E"),
        ("NI_N", "NI", "N"), ("N_NI", "N", "NI"),
        ("SI_S", "SI", "S"), ("S_SI", "S", "SI"),
    ]

    for edge_id, src, dst in external_edges:
        edges.append(edge_xml(edge_id, src, dst, speed=13.89, lanes=1, priority=2))

    # Clockwise roundabout circulation
    ring = ["R0", "R7", "R6", "R5", "R4", "R3", "R2", "R1", "R0"]

    for i in range(len(ring) - 1):
        edge_id = f"{ring[i]}_{ring[i + 1]}"
        edges.append(edge_xml(edge_id, ring[i], ring[i + 1], speed=8.0, lanes=1, priority=4))

    # Main approaches between signalised junctions and roundabout
    approaches = [
        ("W_R4", "W", "R4"), ("R4_W", "R4", "W"),
        ("E_R0", "E", "R0"), ("R0_E", "R0", "E"),
        ("N_R2", "N", "R2"), ("R2_N", "R2", "N"),
        ("S_R6", "S", "R6"), ("R6_S", "R6", "S"),
    ]

    for edge_id, src, dst in approaches:
        edges.append(edge_xml(edge_id, src, dst, speed=13.89, lanes=1, priority=3))

    # Outer 2x2 alternative network.
    # All top/bottom paths pass through signalised N/S nodes.
    outer_edges = [
        # Left vertical side via W
        ("W_NW", "W", "NW"), ("NW_W", "NW", "W"),
        ("W_SW", "W", "SW"), ("SW_W", "SW", "W"),

        # Right vertical side via E
        ("E_NE", "E", "NE"), ("NE_E", "NE", "E"),
        ("E_SE", "E", "SE"), ("SE_E", "SE", "E"),

        # Top road via N traffic light
        ("NW_N", "NW", "N"), ("N_NW", "N", "NW"),
        ("N_NE", "N", "NE"), ("NE_N", "NE", "N"),

        # Bottom road via S traffic light
        ("SW_S", "SW", "S"), ("S_SW", "S", "SW"),
        ("S_SE", "S", "SE"), ("SE_S", "SE", "S"),
    ]

    for edge_id, src, dst in outer_edges:
        edges.append(edge_xml(edge_id, src, dst, speed=12.0, lanes=1, priority=2))

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


def write_additional():
    """
    Adds visual elements only:
    - RSU communication zone
    - RSU marker
    - MEC/TrustV2X marker
    - C-ITS control marker
    - CAM marker
    - DENM marker
    - TrustV2X accept/flag/reject markers
    - Road blockage marker
    - Top signalised junction marker

    Legend has been removed.
    """

    rsu_range_shape = circle_shape(50, 50, 32)

    with open(ADDITIONAL_FILE, "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<additional>

    <!-- RSU communication zone -->
""")

        f.write(
            f'    <poly id="RSU_1_COMMUNICATION_RANGE" type="RSU Communication Range" '
            f'color="0,90,255,45" fill="true" layer="0" shape="{rsu_range_shape}"/>\n'
        )

        f.write("""
    <!-- C-ITS infrastructure markers -->
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

        # Road blockage marker on the east roundabout exit.
        f.write(
            f'    <poly id="ROAD_BLOCKAGE_ZONE" type="Road Blockage / Accident" '
            f'color="255,0,0,120" fill="true" layer="10" shape="72,47 82,47 82,53 72,53"/>\n'
        )

        # Mark top road as a signalised junction route, not a bridge.
        f.write(
            f'    <poly id="TOP_SIGNALISED_JUNCTION_ZONE" type="Top Signalised Junction" '
            f'color="255,220,0,55" fill="true" layer="1" shape="44,94 56,94 56,106 44,106"/>\n'
        )

        f.write("""
    <poi id="TOP_SIGNAL_LABEL" type="Signalised Top Junction" color="255,220,0" x="50" y="108" layer="25"/>

    <!-- Pedestrian / roadside markers -->
    <poi id="PEDESTRIAN_CROSSING_W" type="Pedestrian Crossing" color="255,255,255" x="12" y="50" layer="20"/>
    <poi id="PEDESTRIAN_CROSSING_E" type="Pedestrian Crossing" color="255,255,255" x="88" y="50" layer="20"/>

</additional>
""")

    print(f"[OK] Additional C-ITS visual elements written without legend: {ADDITIONAL_FILE}")


def write_routes():
    """
    Creates sorted vehicle departures and routes.
    """

    with open(ROUTE_FILE, "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<routes>

    <!-- Vehicle types -->
    <vType id="trusted_car" vClass="passenger" guiShape="passenger" color="0,180,0" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.89"/>
    <vType id="background_car" vClass="passenger" guiShape="passenger" color="160,160,160" accel="2.4" decel="4.2" sigma="0.6" length="5" maxSpeed="13.89"/>
    <vType id="flagged_car" vClass="passenger" guiShape="passenger" color="255,200,0" accel="2.3" decel="4.1" sigma="0.7" length="5" maxSpeed="12.00"/>
    <vType id="malicious_car" vClass="passenger" guiShape="passenger" color="255,0,0" accel="3.0" decel="4.5" sigma="0.9" length="5" maxSpeed="16.00"/>
    <vType id="emergency_vehicle" vClass="emergency" guiShape="emergency" color="0,220,255" accel="3.5" decel="5.0" sigma="0.3" length="6" maxSpeed="18.00"/>
    <vType id="police_vehicle" vClass="authority" guiShape="passenger" color="0,80,255" accel="3.2" decel="5.0" sigma="0.3" length="5.5" maxSpeed="17.00"/>
    <vType id="bus_vehicle" vClass="bus" guiShape="bus" color="80,80,255" accel="1.5" decel="3.5" sigma="0.5" length="10" maxSpeed="11.00"/>

    <!-- Main roundabout routes -->
    <route id="west_to_east_roundabout" edges="WI_W W_R4 R4_R3 R3_R2 R2_R1 R1_R0 R0_E E_EO"/>
    <route id="east_to_west_roundabout" edges="EO_E E_R0 R0_R7 R7_R6 R6_R5 R5_R4 R4_W W_WI"/>
    <route id="north_to_south_roundabout" edges="NI_N N_R2 R2_R1 R1_R0 R0_R7 R7_R6 R6_S S_SI"/>
    <route id="south_to_north_roundabout" edges="SI_S S_R6 R6_R5 R5_R4 R4_R3 R3_R2 R2_N N_NI"/>

    <!-- Alternative routes through signalised top/bottom junctions -->
    <route id="west_to_east_north_alt" edges="WI_W W_NW NW_N N_NE NE_E E_EO"/>
    <route id="west_to_east_south_alt" edges="WI_W W_SW SW_S S_SE SE_E E_EO"/>
    <route id="east_to_west_north_alt" edges="EO_E E_NE NE_N N_NW NW_W W_WI"/>
    <route id="east_to_west_south_alt" edges="EO_E E_SE SE_S S_SW SW_W W_WI"/>

    <!-- Vehicle departures sorted by time -->
    <flow id="flow_trusted_we" type="trusted_car" route="west_to_east_roundabout" begin="0" end="360" period="5"/>
    <flow id="flow_trusted_ew" type="trusted_car" route="east_to_west_roundabout" begin="5" end="360" period="6"/>
    <flow id="flow_trusted_ns" type="trusted_car" route="north_to_south_roundabout" begin="10" end="360" period="7"/>
    <flow id="flow_trusted_sn" type="trusted_car" route="south_to_north_roundabout" begin="12" end="360" period="7"/>

    <flow id="flow_background_north_alt" type="background_car" route="west_to_east_north_alt" begin="20" end="360" period="10"/>
    <flow id="flow_background_south_alt" type="background_car" route="east_to_west_south_alt" begin="25" end="360" period="11"/>

    <vehicle id="ambulance_01" type="emergency_vehicle" route="south_to_north_roundabout" depart="35" departLane="best"/>
    <vehicle id="police_01" type="police_vehicle" route="east_to_west_roundabout" depart="45" departLane="best"/>
    <vehicle id="bus_01" type="bus_vehicle" route="west_to_east_north_alt" depart="50" departLane="best"/>

    <vehicle id="flagged_vehicle_01" type="flagged_car" route="east_to_west_roundabout" depart="80" departLane="best"/>
    <vehicle id="malicious_vehicle_01" type="malicious_car" route="west_to_east_roundabout" depart="110" departLane="best"/>

    <vehicle id="rerouted_vehicle_01" type="trusted_car" route="west_to_east_north_alt" depart="140" departLane="best"/>
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
        "--default.junctions.keep-clear",
        "true",
    ]

    print("\n[INFO] Building SUMO network with netconvert...")
    print(" ".join(command))

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("\n[ERROR] netconvert failed.")
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("SUMO netconvert failed.")

    print(f"[OK] SUMO network created: {NET_FILE}")

    if result.stderr.strip():
        print("\n[netconvert messages]")
        print(result.stderr)


def main():
    print("=" * 70)
    print("TrustV2X Corrected Signalised 2x2 C-ITS Network Creator")
    print("=" * 70)

    ensure_directories()
    write_nodes()
    write_edges()
    write_additional()
    write_routes()
    build_network_with_netconvert()
    write_config()

    print("\n" + "=" * 70)
    print("[OK] Corrected signalised 2x2 C-ITS network created without legend")
    print("=" * 70)

    print("\nOpen with:")
    print(f'& "C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\sumo-gui.exe" -c "{CONFIG_FILE}"')


if __name__ == "__main__":
    main()