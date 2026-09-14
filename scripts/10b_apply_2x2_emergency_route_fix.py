from pathlib import Path


# ============================================================
# TrustV2X - Step 10b
# Apply Emergency and Priority Route Fix for 2x2 C-ITS Demo
# ============================================================
#
# Purpose:
# - Keep the existing 2x2 signalised network.
# - Replace only the route file.
# - Put ambulance and police vehicles on alternative priority routes
#   from the beginning, instead of sending them into the congested
#   roundabout/incident area.
#
# This does NOT affect the 4x4 quantitative experiment.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROUTE_FILE = (
    PROJECT_ROOT
    / "sumo"
    / "realistic_2x2_cits_signalised_demo"
    / "routes"
    / "trustv2x_2x2_signalised.rou.xml"
)


def write_corrected_routes():
    ROUTE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(ROUTE_FILE, "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<routes>

    <!-- ===================================================== -->
    <!-- Vehicle types                                         -->
    <!-- ===================================================== -->

    <vType id="trusted_car" vClass="passenger" guiShape="passenger" color="0,180,0" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.89"/>
    <vType id="background_car" vClass="passenger" guiShape="passenger" color="160,160,160" accel="2.4" decel="4.2" sigma="0.6" length="5" maxSpeed="13.89"/>

    <vType id="flagged_car" vClass="passenger" guiShape="passenger" color="255,200,0" accel="2.3" decel="4.1" sigma="0.7" length="5" maxSpeed="12.00"/>
    <vType id="malicious_car" vClass="passenger" guiShape="passenger" color="255,0,0" accel="3.0" decel="4.5" sigma="0.9" length="5" maxSpeed="16.00"/>

    <vType id="emergency_vehicle" vClass="emergency" guiShape="emergency" color="0,220,255" accel="3.5" decel="5.0" sigma="0.3" length="6" maxSpeed="18.00"/>
    <vType id="police_vehicle" vClass="authority" guiShape="passenger" color="0,80,255" accel="3.2" decel="5.0" sigma="0.3" length="5.5" maxSpeed="17.00"/>
    <vType id="fire_vehicle" vClass="emergency" guiShape="emergency" color="255,80,0" accel="3.0" decel="5.0" sigma="0.3" length="7" maxSpeed="16.00"/>

    <vType id="bus_vehicle" vClass="bus" guiShape="bus" color="80,80,255" accel="1.5" decel="3.5" sigma="0.5" length="10" maxSpeed="11.00"/>


    <!-- ===================================================== -->
    <!-- Main roundabout routes                                -->
    <!-- ===================================================== -->

    <route id="west_to_east_roundabout" edges="WI_W W_R4 R4_R3 R3_R2 R2_R1 R1_R0 R0_E E_EO"/>
    <route id="east_to_west_roundabout" edges="EO_E E_R0 R0_R7 R7_R6 R6_R5 R5_R4 R4_W W_WI"/>
    <route id="north_to_south_roundabout" edges="NI_N N_R2 R2_R1 R1_R0 R0_R7 R7_R6 R6_S S_SI"/>
    <route id="south_to_north_roundabout" edges="SI_S S_R6 R6_R5 R5_R4 R4_R3 R3_R2 R2_N N_NI"/>


    <!-- ===================================================== -->
    <!-- Alternative routes through signalised junctions        -->
    <!-- ===================================================== -->

    <route id="west_to_east_north_alt" edges="WI_W W_NW NW_N N_NE NE_E E_EO"/>
    <route id="west_to_east_south_alt" edges="WI_W W_SW SW_S S_SE SE_E E_EO"/>

    <route id="east_to_west_north_alt" edges="EO_E E_NE NE_N N_NW NW_W W_WI"/>
    <route id="east_to_west_south_alt" edges="EO_E E_SE SE_S S_SW SW_W W_WI"/>


    <!-- ===================================================== -->
    <!-- Emergency and priority routes                          -->
    <!-- These avoid the roundabout/incident zone from start.   -->
    <!-- ===================================================== -->

    <route id="ambulance_priority_alt" edges="SI_S S_SW SW_W W_NW NW_N N_NI"/>
    <route id="police_priority_alt" edges="EO_E E_SE SE_S S_SW SW_W W_WI"/>
    <route id="fire_priority_alt" edges="WI_W W_NW NW_N N_NE NE_E E_EO"/>


    <!-- ===================================================== -->
    <!-- Normal traffic flows                                  -->
    <!-- ===================================================== -->

    <flow id="flow_trusted_we" type="trusted_car" route="west_to_east_roundabout" begin="0" end="360" period="5"/>
    <flow id="flow_trusted_ew" type="trusted_car" route="east_to_west_roundabout" begin="5" end="360" period="6"/>
    <flow id="flow_trusted_ns" type="trusted_car" route="north_to_south_roundabout" begin="10" end="360" period="7"/>
    <flow id="flow_trusted_sn" type="trusted_car" route="south_to_north_roundabout" begin="12" end="360" period="7"/>

    <flow id="flow_background_north_alt" type="background_car" route="west_to_east_north_alt" begin="20" end="360" period="10"/>
    <flow id="flow_background_south_alt" type="background_car" route="east_to_west_south_alt" begin="25" end="360" period="11"/>


    <!-- ===================================================== -->
    <!-- Special vehicles                                      -->
    <!-- Ambulance/police/fire now start on safe alternatives. -->
    <!-- ===================================================== -->

    <vehicle id="ambulance_01" type="emergency_vehicle" route="ambulance_priority_alt" depart="35" departLane="best"/>
    <vehicle id="police_01" type="police_vehicle" route="police_priority_alt" depart="45" departLane="best"/>
    <vehicle id="fire_response_01" type="fire_vehicle" route="fire_priority_alt" depart="65" departLane="best"/>

    <vehicle id="bus_01" type="bus_vehicle" route="west_to_east_north_alt" depart="50" departLane="best"/>

    <vehicle id="flagged_vehicle_01" type="flagged_car" route="east_to_west_roundabout" depart="80" departLane="best"/>
    <vehicle id="malicious_vehicle_01" type="malicious_car" route="west_to_east_roundabout" depart="110" departLane="best"/>

    <!-- These represent vehicles already using C-ITS alternative guidance. -->
    <vehicle id="rerouted_vehicle_01" type="trusted_car" route="west_to_east_north_alt" depart="140" departLane="best"/>
    <vehicle id="rerouted_vehicle_02" type="trusted_car" route="west_to_east_south_alt" depart="150" departLane="best"/>
    <vehicle id="rerouted_vehicle_03" type="trusted_car" route="west_to_east_north_alt" depart="160" departLane="best"/>

</routes>
""")

    print("=" * 70)
    print("[OK] Emergency/priority route fix applied")
    print("=" * 70)
    print(f"[OK] Updated route file: {ROUTE_FILE}")
    print("\nKey correction:")
    print("- ambulance_01 now uses ambulance_priority_alt")
    print("- police_01 now uses police_priority_alt")
    print("- fire_response_01 added using fire_priority_alt")
    print("- emergency vehicles avoid the congested roundabout/incident route from departure")


if __name__ == "__main__":
    write_corrected_routes()