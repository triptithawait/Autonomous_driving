import osmnx as ox
import networkx as nx
import folium
from geopy.geocoders import Nominatim

class OSMMapEngine:
    def __init__(self):
        self.geocoder = Nominatim(user_agent="smart_nav_ai")
        self.road_widths = {
            'motorway': 12.0,
            'trunk': 10.0,
            'primary': 8.5,
            'secondary': 7.0,
            'tertiary': 5.5,
            'residential': 3.8,
            'living_street': 3.0,
            'service': 2.5,
            'pedestrian': 2.0,
            'unclassified': 4.0
        }

    def get_city_graph(self, city_name):
        """Downloads the road network for a city."""
        try:
            # Using 'drive' network type
            G = ox.graph_from_place(city_name, network_type='drive')
            return G
        except Exception as e:
            print(f"Error fetching city graph: {e}")
            return None

    def apply_suitability(self, G, vehicle_width, fuzzy_system):
        """Applies fuzzy suitability to every edge in the real map."""
        for u, v, k, data in G.edges(keys=True, data=True):
            # Estimate width based on OSM tags
            highway_type = data.get('highway', 'unclassified')
            if isinstance(highway_type, list): highway_type = highway_type[0]
            
            # Use actual 'width' tag if exists, else estimate from type
            raw_width = data.get('width', self.road_widths.get(highway_type, 4.0))
            try:
                if isinstance(raw_width, str):
                    road_width = float(raw_width.split(' ')[0])
                elif isinstance(raw_width, list):
                    road_width = float(raw_width[0])
                else:
                    road_width = float(raw_width)
            except:
                road_width = self.road_widths.get(highway_type, 4.0)

            suitability = fuzzy_system.compute_suitability(road_width, vehicle_width)
            
            # Calculate weighted cost
            length = data.get('length', 100)
            score_factor = max(suitability / 100, 0.05) # Prevent division by zero
            
            # CRITICAL BUFFER for real roads
            if road_width < vehicle_width + 0.3:
                score_factor = 0.001 

            data['suitability'] = suitability
            data['estimated_width'] = road_width
            data['weight'] = length / score_factor

    def find_route(self, G, start_coords, end_coords):
        """Finds path between two GPS coordinates."""
        orig_node = ox.nearest_nodes(G, start_coords[1], start_coords[0])
        dest_node = ox.nearest_nodes(G, end_coords[1], end_coords[0])
        
        try:
            route = nx.shortest_path(G, orig_node, dest_node, weight='weight')
            return route
        except nx.NetworkXNoPath:
            return None

    def plot_folium_route(self, G, route):
        """Creates an interactive Folium map with color-coded segments and popups."""
        if not route: return None
        
        start_node = G.nodes[route[0]]
        m = folium.Map(location=[start_node['y'], start_node['x']], zoom_start=14)
        
        total_km = 0
        
        # Iterate through segments to color them individually
        for u, v in zip(route[:-1], route[1:]):
            # OSM edges could have multiple keys, we take the one with smallest weight/dist
            edge_data = G.get_edge_data(u, v)
            if not edge_data: continue
            
            # Take first key if multi-graph
            if 0 in edge_data: data = edge_data[0]
            else: data = list(edge_data.values())[0]

            u_data = G.nodes[u]
            v_data = G.nodes[v]
            coords = [(u_data['y'], u_data['x']), (v_data['y'], v_data['x'])]
            
            # Distance accumulation
            dist = data.get('length', 0)
            total_km += dist / 1000.0
            
            suit = data.get('suitability', 50)
            width = data.get('estimated_width', 3.0)
            
            # Determine color
            color = "#2ecc71" if suit > 70 else "#f39c12" if suit > 40 else "#e74c3c"
            
            # Add segment to map with popup
            folium.PolyLine(
                coords, 
                color=color, 
                weight=6, 
                opacity=0.9,
                tooltip=f"Width: {width}m | Suit: {suit:.1f}%"
            ).add_to(m)

        # Markers
        folium.Marker((G.nodes[route[0]]['y'], G.nodes[route[0]]['x']), popup="Start", icon=folium.Icon(color='green')).add_to(m)
        folium.Marker((G.nodes[route[-1]]['y'], G.nodes[route[-1]]['x']), popup="Destination", icon=folium.Icon(color='red')).add_to(m)
        
        return m, round(total_km, 2)
