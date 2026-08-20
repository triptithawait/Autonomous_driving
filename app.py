import streamlit as st
import cv2
import numpy as np
from PIL import Image
from models.fuzzy_logic import SmartNavFuzzySystem
from models.cnn_model import RoadWidthCNN
from core.graph_engine import RoadNetworkEngine
from core.osm_engine import OSMMapEngine
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import folium

# Page Config
st.set_page_config(page_title="SmartNav AI - Vehicle-Size-Aware Navigation", layout="wide")

# Initialize Systems
@st.cache_resource
def load_systems():
    return SmartNavFuzzySystem(), RoadWidthCNN(), RoadNetworkEngine(), OSMMapEngine()

fuzzy_sys, cnn_sys, road_net, osm_net = load_systems()

# CSS for better styling
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .suitability-high { color: green; font-weight: bold; }
    .suitability-low { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title(" SmartNav AI: Vehicle-Size-Aware Navigation System")
st.subheader("Optimizing Indian Road Routes using CNN & Fuzzy Logic")

# Sidebar - Vehicle Inputs
with st.sidebar:
    st.header("1. Vehicle Profile")
    v_type = st.selectbox("Select Vehicle Type", ["Small Car (WagonR)", "Sedan (City)", "SUV (Innova)", "Truck/Bus"])
    
    dimensions = {
        "Small Car (WagonR)": 1.5,
        "Sedan (City)": 1.8,
        "SUV (Innova)": 2.0,
        "Truck/Bus": 2.8
    }
    v_width = st.number_input("Vehicle Width (meters)", value=dimensions[v_type], step=0.1)
    
    st.header("2. Navigation Nodes")
    nodes = list(road_net.graph.nodes())
    start_node = st.selectbox("Start Point", nodes, index=0)
    end_node = st.selectbox("Destination", nodes, index=len(nodes)-1)

    if st.button("Calculate Optimal Route"):
        st.session_state['calculate'] = True

# Main Content - Tabs
tab1, tab2, tab3 = st.tabs([" Simulated Network", " Real-World Map (Live)", " CNN Width Analysis"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header(" Comparison Graph")
        road_net.update_road_costs(v_width, fuzzy_sys)
        
        path_smart = None
        path_std = None
        if st.session_state.get('calculate'):
            path_smart = road_net.find_optimal_route(start_node, end_node)
            path_std = road_net.find_standard_route(start_node, end_node)
            
            if path_smart:
                st.info(f"🔵 Blue Line: Smart AI Route")
                st.write(f"🔴 Red Dashed: Standard Route (Shortest)")
            else:
                st.error("No suitable route found for this vehicle size!")

        fig = road_net.get_network_plot(smart_path=path_smart, standard_path=path_std)
        st.pyplot(fig)

    with col2:
        st.header(" AI vs Standard GPS")
        
        if st.session_state.get('calculate') and path_smart:
            # Comparison Analysis
            def analyze_path(path):
                dist = 0
                min_suit = 100
                for u, v in zip(path, path[1:]):
                    data = road_net.graph[u][v]
                    s = data['suitability']
                    if data['width'] < v_width + 0.4: s = 5.0
                    dist += data['length']
                    min_suit = min(min_suit, s)
                return dist, min_suit

            smart_dist, smart_suit = analyze_path(path_smart)
            std_dist, std_suit = analyze_path(path_std)

            st.write("### Route Comparison Table")
            comparison_data = {
                "Metric": ["Total Distance (m)", "Min Suitability (%)", "Road Width Constraint"],
                "Standard GPS": [f"{std_dist:.1f}m", f"{std_suit:.1f}%", "❌ Ignored"],
                "Smart AI": [f"{smart_dist:.1f}m", f"{smart_suit:.1f}%", "✅ Respected"]
            }
            st.table(comparison_data)

            if smart_suit > std_suit:
                st.success(f" Smart AI avoided a dangerous road (Suitability improved by {smart_suit - std_suit:.1f}%)!")
            
            if smart_dist > std_dist:
                st.warning(f"Note: AI added {smart_dist - std_dist}m to your trip to ensure safety.")
        else:
            st.info("Select vehicle and destination to see AI route optimization.")

with tab2:
    st.header(" Real-World Smart Navigation")
    st.write("Live data from OpenStreetMap. Routes are calculated by avoiding narrow roads for your vehicle size.")
    
    city_col1, city_col2 = st.columns([1, 1])
    with city_col1:
        city_name = st.text_input("Enter City/Location", "New Delhi, India")
        search_btn = st.button("Download Road Network")
        
        if search_btn or 'osm_graph' in st.session_state:
            with st.spinner("Fetching city map data..."):
                if 'osm_graph' not in st.session_state or search_btn:
                    st.session_state['osm_graph'] = osm_net.get_city_graph(city_name)
                
                st.success(f"Map Loaded: {len(st.session_state['osm_graph'].nodes)} intersections found.")
                osm_net.apply_suitability(st.session_state['osm_graph'], v_width, fuzzy_sys)
    
    with city_col2:
        if 'osm_graph' in st.session_state:
            st.subheader("Set Route Points")
            
            addr_start = st.text_input("Start Address", "Connaught Place, Delhi")
            addr_end = st.text_input("End Address", "India Gate, Delhi")
            
            if st.button("Resolve Addresses"):
                # Geocode address to Lat/Lon
                try:
                    loc_start = osm_net.geocoder.geocode(addr_start)
                    loc_end = osm_net.geocoder.geocode(addr_end)
                    if loc_start and loc_end:
                        st.session_state['start_pos'] = (loc_start.latitude, loc_start.longitude)
                        st.session_state['end_pos'] = (loc_end.latitude, loc_end.longitude)
                        st.success("Addresses resolved successfully!")
                    else:
                        st.error("Could not find one of the addresses. Please be more specific.")
                except:
                    st.error("Geocoding service busy. Try again or enter Lat/Lon manually.")

            # Display coordinates (manual edit possible)
            s_lat, s_lon = st.session_state.get('start_pos', (28.6328, 77.2197))
            e_lat, e_lon = st.session_state.get('end_pos', (28.6129, 77.2295))
            
            start_lat = st.number_input("Start Lat", value=s_lat, format="%.4f")
            start_lon = st.number_input("Start Lon", value=s_lon, format="%.4f")
            end_lat = st.number_input("End Lat", value=e_lat, format="%.4f")
            end_lon = st.number_input("End Lon", value=e_lon, format="%.4f")
            
            if st.button("Generate Real-World Route"):
                route = osm_net.find_route(st.session_state['osm_graph'], (start_lat, start_lon), (end_lat, end_lon))
                if route:
                    result = osm_net.plot_folium_route(st.session_state['osm_graph'], route)
                    # Handle both old (cached) and new return formats
                    if isinstance(result, tuple):
                        m, dist_km = result
                        st.metric("Total Route Distance", f"{dist_km} km")
                    else:
                        m = result
                        st.write("💡 Note: Restart app to see distance metrics.")
                    
                    st.write("💡 Hover over road segments to see Width and Suitability.")
                    st_folium(m, width=900, height=600)
                else:
                    st.error("Could not find a safe route for your vehicle dimensions in this area.")

with tab3:
    # CNN Road Width Estimation Section
    st.header(" Real-time Road Width Estimation (CNN)")
    image_col, pred_col = st.columns([1, 1])

    with image_col:
        img_file = st.file_uploader("Upload Image of Road", type=["jpg", "png", "jpeg"])
        camera_input = st.camera_input("Or Capture from Camera")
        
        input_source = None
        if img_file: 
            # Save temp file for CV2 to read if on local
            with open("temp_road.jpg", "wb") as f:
                f.write(img_file.getbuffer())
            input_source = "temp_road.jpg"
        elif camera_input: 
            input_source = camera_input

    with pred_col:
        # Manual Correction Sidebar/Option
        manual_override = st.checkbox("Manual Correction (Override AI)")
        manual_class = st.selectbox("Select Actual Road Class", ["Narrow", "Medium", "Wide"], disabled=not manual_override)

        if input_source:
            if not camera_input: st.image(input_source, width=400)
            
            label, est_width = cnn_sys.predict_width(image_path=input_source if not camera_input else None, frame=None)
            
            if manual_override:
                label = manual_class
                est_width = cnn_sys.width_map[label]

            st.success(f"CNN Prediction: **{label} Road**")
            st.write(f"Estimated Width: **{est_width} meters**")
            
            # Buffer-aware Suitability
            suit_score = fuzzy_sys.compute_suitability(est_width, v_width)
            if est_width < v_width + 0.4:
                suit_score = 5.0
                st.error("🚨 VEHICLE TOO WIDE FOR THIS ROAD!")
            
            st.progress(suit_score/100)
            st.write(f"Calculated Suitability for current vehicle: **{suit_score:.1f}%**")
        else:
            st.write("Upload or capture a road image to test width estimation.")

st.sidebar.markdown("---")
st.sidebar.caption("SmartNav AI v1.0 - Advanced Indian Road Navigation")
