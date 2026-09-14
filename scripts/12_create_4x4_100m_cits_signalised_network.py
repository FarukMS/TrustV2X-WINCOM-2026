import os
import math
import subprocess
from pathlib import Path


# ============================================================
# TrustV2X - Step 12
# Create 4x4 100m x 100m Signalised C-ITS SUMO Demo Network
# ============================================================
#
# Purpose:
# - Build a separate 4x4 grid-like C-ITS visual/behavioural demo.
# - Keep the original 4x4 quantitative experiment untouched.
# - Network size: approximately 100m x 100m.
# - Add traffic lights, RSU, MEC/TrustV2X edge, C-ITS control,
#   CAM/DENM markers, road incident zone, emergency vehicles,
#   police/fire priority vehicles and alternative routes.
#
# This does NOT modify the existing quantitative TrustV2X results.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEMO_ROOT = PROJECT_ROOT / "sumo" / "visual_4x4_100m_cits_demo"

NETWORK_DIR = DEMO_ROOT / "network"
ROUTES_DIR = DEMO_ROOT / "routes"
CONFIG_DIR = DEMO_ROOT / "config"
ADDITIONAL_DIR = DEMO_ROOT / "additional"
OUTPUT_DIR = DEMO_ROOT / "outputs"

NODES_FILE = NETWORK_DIR / "trustv2x_4x4_100m.nod.xml"
EDGES_FILE = NETWORK_DIR / "trustv2x_4x4_100m.edg.xml"
NET_FILE = NETWORK_DIR / "trustv2x_4x4_100m.net.xml"
ROUTE_FILE = ROUTES_DIR / "trustv2x_4x4_100m.rou.xml"
ADDITIONAL_FILE = ADDITIONAL_DIR / "trustv2x_4x4_100m.add.xml"
CONFIG_FILE = CONFIG_DIR / "trustv2x_4x4_100m.sumocfg"

GRID_SIZE = 4
NETWORK_LENGTH = 100.0
SPACING = NETWORK_LENGTH / (GRID_SIZE - 1)


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


def node_id(i, j):
    return f"J{i}{j}"


def west_ext(j):
    return f"W{j}"


def east_ext(j):
    return f"E{j}"


def south_ext(i):
    return f"S{i}"


def north_ext(i):
    return f"N{i}"


def write_nodes():
    """
    Creates 4x4 signalised grid intersections plus external entry/exit points.
    Coordinates span approximately 0..100m in x and y.
    """
    with open(NODES_FILE, "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<nodes>
""")

        # Internal 4x4 signalised intersections
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x = i * SPACING
                y = j * SPACING
                f.write(
                    f'    <node id="{node_id(i, j)}" x="{x:.2f}" y="{y:.2f}" type="traffic_light"/>\n'
                )

        # External entry/exit nodes
        for j in range(GRID_SIZE):
            y = j * SPACING
            f.write(f'    <node id="{west_ext(j)}" x="-20" y="{y:.2f}" type="priority"/>\n')
            f.write(f'    <node id="{east_ext(j)}" x="120" y="{y:.2f}" type="priority"/>\n')

        for i in range(GRID_SIZE):
            x = i * SPACING
            f.write(f'    <node id="{south_ext(i)}" x="{x:.2f}" y="-20" type="priority"/>\n')
            f.write(f'    <node id="{north_ext(i)}" x="{x:.2f}" y="120" type="priority"/>\n')

        f.write("</nodes>\n")

    print(f"[OK] Nodes written: {NODES_FILE}")


def edge_xml(edge_id, from_node, to_node, speed=10.0, lanes=1, priority=1):
    return (
        f'    <edge id="{edge_id}" from="{from_node}" to="{to_node}" '
        f'priority="{priority}" numLanes="{lanes}" speed="{speed}"/>\n'
    )


def write_edges():
    """
    Creates a bidirectional 4x4 grid.
    """
    edges = []

    # Horizontal internal edges
    for j in range(GRID_SIZE):
        for i in range(GRID_SIZE - 1):
            a = node_id(i, j)
            b = node_id(i + 1, j)

            edges.append(edge_xml(f"{a}_{b}", a, b, speed=10.0, priority=2))
            edges.append(edge_xml(f"{b}_{a}", b, a, speed=10.0, priority=2))

    # Vertical internal edges
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE - 1):
            a = node_id(i, j)
            b = node_id(i, j + 1)

            edges.append(edge_xml(f"{a}_{b}", a, b, speed=10.0, priority=2))
            edges.append(edge_xml(f"{b}_{a}", b, a, speed=10.0, priority=2))

    # External west/east entry/exit edges
    for j in range(GRID_SIZE):
        w = west_ext(j)
        e = east_ext(j)
        left = node_id(0, j)
        right = node_id(GRID_SIZE - 1, j)

        edges.append(edge_xml(f"{w}_{left}", w, left, speed=11.0, priority=3))
        edges.append(edge_xml(f"{left}_{w}", left, w, speed=11.0, priority=3))

        edges.append(edge_xml(f"{right}_{e}", right, e, speed=11.0, priority=3))
        edges.append(edge_xml(f"{e}_{right}", e, right, speed=11.0, priority=3))

    # External south/north entry/exit edges
    for i in range(GRID_SIZE):
        s = south_ext(i)
        n = north_ext(i)
        bottom = node_id(i, 0)
        top = node_id(i, GRID_SIZE - 1)

        edges.append(edge_xml(f"{s}_{bottom}", s, bottom, speed=11.0, priority=3))
        edges.append(edge_xml(f"{bottom}_{s}", bottom, s, speed=11.0, priority=3))

        edges.append(edge_xml(f"{top}_{n}", top, n, speed=11.0, priority=3))
        edges.append(edge_xml(f"{n}_{top}", n, top, speed=11.0, priority=3))

    with open(EDGES_FILE, "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<edges>
""")
        for edge in edges:
            f.write(edge)
        f.write("</edges>\n")

    print(f"[OK] Edges written: {EDGES_FILE}")


def circle_shape(cx, cy, radius, n=96):
    points = []

    for i in range(n):
        angle = 2 * math.pi * i / n
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append(f"{x:.2f},{y:.2f}")

    return " ".join(points)


def write_additional():
    """
    Adds visual C-ITS markers without a side legend.
    Manual labels/legend can be added to final screenshots externally.
    """
    rsu_range = circle_shape(50, 50, 40)

    with open(ADDITIONAL_FILE, "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<additional>

    <!-- RSU communication range -->
""")

        f.write(
            f'    <poly id="RSU_1_COMMUNICATION_RANGE" type="RSU Communication Range" '
            f'color="0,90,255,45" fill="true" layer="0" shape="{rsu_range}"/>\n'
        )

        f.write("""
    <!-- C-ITS / TrustV2X infrastructure markers -->
    <poi id="RSU_1" type="RSU-1" color="0,80,255" x="50" y="50" layer="30"/>
    <poi id="MEC_TRUSTV2X_EDGE" type="MEC / TrustV2X Edge" color="160,0,255" x="58" y="58" layer="30"/>
    <poi id="CITS_CONTROL_CENTRE" type="C-ITS Control Centre" color="255,140,0" x="42" y="58" layer="30"/>

    <poi id="CAM_ACTIVE" type="CAM Broadcast Zone" color="0,180,255" x="42" y="42" layer="30"/>
    <poi id="DENM_WARNING" type="DENM Hazard Warning" color="255,0,0" x="53" y="34" layer="30"/>

    <poi id="TRUSTV2X_ACCEPT" type="TrustV2X ACCEPT" color="0,220,0" x="25" y="75" layer="30"/>
    <poi id="TRUSTV2X_FLAG" type="TrustV2X FLAG" color="255,200,0" x="75" y="25" layer="30"/>
    <poi id="TRUSTV2X_REJECT" type="TrustV2X REJECT" color="255,0,0" x="85" y="25" layer="30"/>

    <poi id="ROAD_HAZARD_OBSTACLE" type="Road Hazard / Obstacle" color="255,0,0" x="50" y="33.3" layer="30"/>

    <!-- Road incident / blockage zone on central eastbound edge J11_J21 -->
    <poly id="ROAD_BLOCKAGE_ZONE" type="Road Blockage / Accident" color="255,0,0,120" fill="true" layer="10" shape="45,30 57,30 57,37 45,37"/>

    <!-- Pedestrian / roadside markers -->
    <poi id="PEDESTRIAN_CROSSING_W" type="Pedestrian Crossing" color="255,255,255" x="2" y="50" layer="20"/>
    <poi id="PEDESTRIAN_CROSSING_E" type="Pedestrian Crossing" color="255,255,255" x="98" y="50" layer="20"/>

</additional>
""")

    print(f"[OK] Additional C-ITS visual elements written: {ADDITIONAL_FILE}")


def route_west_to_east(row):
    return f"{west_ext(row)}_{node_id(0, row)} {node_id(0, row)}_{node_id(1, row)} {node_id(1, row)}_{node_id(2, row)} {node_id(2, row)}_{node_id(3, row)} {node_id(3, row)}_{east_ext(row)}"


def route_east_to_west(row):
    return f"{east_ext(row)}_{node_id(3, row)} {node_id(3, row)}_{node_id(2, row)} {node_id(2, row)}_{node_id(1, row)} {node_id(1, row)}_{node_id(0, row)} {node_id(0, row)}_{west_ext(row)}"


def route_south_to_north(col):
    return f"{south_ext(col)}_{node_id(col, 0)} {node_id(col, 0)}_{node_id(col, 1)} {node_id(col, 1)}_{node_id(col, 2)} {node_id(col, 2)}_{node_id(col, 3)} {node_id(col, 3)}_{north_ext(col)}"


def route_north_to_south(col):
    return f"{north_ext(col)}_{node_id(col, 3)} {node_id(col, 3)}_{node_id(col, 2)} {node_id(col, 2)}_{node_id(col, 1)} {node_id(col, 1)}_{node_id(col, 0)} {node_id(col, 0)}_{south_ext(col)}"


def write_routes():
    """
    Creates regular, emergency, police, fire and malicious/flagged routes.
    """
    with open(ROUTE_FILE, "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<routes>

    <!-- Vehicle types -->
    <vType id="trusted_car" vClass="passenger" guiShape="passenger" color="0,180,0" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="10.0"/>
    <vType id="background_car" vClass="passenger" guiShape="passenger" color="160,160,160" accel="2.4" decel="4.2" sigma="0.6" length="5" maxSpeed="10.0"/>

    <vType id="flagged_car" vClass="passenger" guiShape="passenger" color="255,200,0" accel="2.3" decel="4.1" sigma="0.7" length="5" maxSpeed="9.0"/>
    <vType id="malicious_car" vClass="passenger" guiShape="passenger" color="255,0,0" accel="3.0" decel="4.5" sigma="0.9" length="5" maxSpeed="11.0"/>

    <vType id="emergency_vehicle" vClass="emergency" guiShape="emergency" color="0,220,255" accel="3.5" decel="5.0" sigma="0.3" length="6" maxSpeed="12.0"/>
    <vType id="police_vehicle" vClass="emergency" guiShape="passenger" color="0,80,255" accel="3.2" decel="5.0" sigma="0.3" length="5.5" maxSpeed="12.0"/>
    <vType id="fire_vehicle" vClass="emergency" guiShape="emergency" color="255,80,0" accel="3.0" decel="5.0" sigma="0.3" length="7" maxSpeed="11.0"/>

    <vType id="bus_vehicle" vClass="bus" guiShape="bus" color="80,80,255" accel="1.5" decel="3.5" sigma="0.5" length="10" maxSpeed="8.0"/>

""")

        # Main row routes
        for row in range(GRID_SIZE):
            f.write(f'    <route id="we_r{row}" edges="{route_west_to_east(row)}"/>\n')
            f.write(f'    <route id="ew_r{row}" edges="{route_east_to_west(row)}"/>\n')

        # Main column routes
        for col in range(GRID_SIZE):
            f.write(f'    <route id="sn_c{col}" edges="{route_south_to_north(col)}"/>\n')
            f.write(f'    <route id="ns_c{col}" edges="{route_north_to_south(col)}"/>\n')

        f.write("""
    <!-- Alternative routes around central incident edge J11_J21 -->
    <route id="we_r1_north_alt" edges="W1_J01 J01_J02 J02_J12 J12_J22 J22_J32 J32_E2"/>
    <route id="we_r1_south_alt" edges="W1_J01 J01_J00 J00_J10 J10_J20 J20_J30 J30_E0"/>

    <!-- Priority routes avoiding the central incident area -->
    <route id="ambulance_priority_alt" edges="S0_J00 J00_J01 J01_J02 J02_J03 J03_N0"/>
    <route id="police_priority_alt" edges="E3_J33 J33_J23 J23_J13 J13_J03 J03_W3"/>
    <route id="fire_priority_alt" edges="W0_J00 J00_J10 J10_J20 J20_J30 J30_E0"/>

    <!-- Normal traffic flows -->
    <flow id="flow_trusted_we_r1" type="trusted_car" route="we_r1" begin="0" end="420" period="6"/>
    <flow id="flow_trusted_ew_r2" type="trusted_car" route="ew_r2" begin="4" end="420" period="7"/>
    <flow id="flow_trusted_sn_c1" type="trusted_car" route="sn_c1" begin="8" end="420" period="8"/>
    <flow id="flow_trusted_ns_c2" type="trusted_car" route="ns_c2" begin="10" end="420" period="8"/>

    <flow id="flow_background_we_r0" type="background_car" route="we_r0" begin="15" end="420" period="10"/>
    <flow id="flow_background_ew_r3" type="background_car" route="ew_r3" begin="18" end="420" period="11"/>
    <flow id="flow_background_sn_c0" type="background_car" route="sn_c0" begin="22" end="420" period="12"/>
    <flow id="flow_background_ns_c3" type="background_car" route="ns_c3" begin="25" end="420" period="12"/>

    <!-- Special C-ITS vehicles -->
    <vehicle id="ambulance_4x4_01" type="emergency_vehicle" route="ambulance_priority_alt" depart="35" departLane="best"/>
    <vehicle id="police_4x4_01" type="police_vehicle" route="police_priority_alt" depart="50" departLane="best"/>
    <vehicle id="fire_4x4_01" type="fire_vehicle" route="fire_priority_alt" depart="65" departLane="best"/>
    <vehicle id="bus_4x4_01" type="bus_vehicle" route="we_r0" depart="70" departLane="best"/>

    <vehicle id="flagged_vehicle_4x4_01" type="flagged_car" route="ew_r2" depart="90" departLane="best"/>
    <vehicle id="malicious_vehicle_4x4_01" type="malicious_car" route="we_r1" depart="115" departLane="best"/>

    <!-- Vehicles already using C-ITS alternative guidance -->
    <vehicle id="rerouted_vehicle_4x4_01" type="trusted_car" route="we_r1_north_alt" depart="145" departLane="best"/>
    <vehicle id="rerouted_vehicle_4x4_02" type="trusted_car" route="we_r1_south_alt" depart="155" departLane="best"/>

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
        <end value="420"/>
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

    print("\n[INFO] Building 4x4 100m SUMO network with netconvert...")
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
    print("TrustV2X 4x4 100m C-ITS Network Creator")
    print("=" * 70)

    ensure_directories()
    write_nodes()
    write_edges()
    write_additional()
    write_routes()
    build_network_with_netconvert()
    write_config()

    print("\n" + "=" * 70)
    print("[OK] 4x4 100m C-ITS visual/behavioural network created")
    print("=" * 70)

    print("\nOpen with:")
    print(f'& "C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\sumo-gui.exe" -c "{CONFIG_FILE}"')


if __name__ == "__main__":
    main()