import networkx as nx
import matplotlib.pyplot as plt

class RoadNetworkEngine:
    def __init__(self):
        self.graph = nx.Graph()
        self._initialize_sample_network()

    def _initialize_sample_network(self):
        """
        Nodes: Intersections
        Edges: Roads with attributes (length, estimated_width, type)
        """
        # (Node1, Node2, {attr})
        roads = [
            ("A", "B", {"width": 8.0, "length": 500, "type": "Main Road"}),
            ("B", "C", {"width": 3.0, "length": 300, "type": "Narrow Street"}),
            ("C", "D", {"width": 9.0, "length": 400, "type": "Highway"}),
            ("A", "C", {"width": 4.5, "length": 700, "type": "Street"}),
            ("B", "D", {"width": 2.5, "length": 600, "type": "Alley"}),
            ("D", "E", {"width": 10.0, "length": 200, "type": "Highway"}),
            ("C", "E", {"width": 5.0, "length": 500, "type": "Street"}),
        ]
        self.graph.add_edges_from(roads)

    def update_road_costs(self, vehicle_width, fuzzy_system):
        """
        Recalculate edge weights based on suitability.
        Cost = Length / (Suitability/100)
        """
        for u, v, data in self.graph.edges(data=True):
            road_width = data['width']
            suitability = fuzzy_system.compute_suitability(road_width, vehicle_width)
            
            # Penalize roads where suitability is low
            # If suitability is 0, cost is infinity (effectively blocked)
            score_factor = max(suitability / 100, 0.01) 
            data['suitability'] = suitability
            data['weight'] = data['length'] / score_factor

    def find_optimal_route(self, source, target):
        """AI Route - Weighted by Suitability"""
        try:
            path = nx.dijkstra_path(self.graph, source, target, weight='weight')
            return path
        except nx.NetworkXNoPath:
            return None

    def find_standard_route(self, source, target):
        """Standard Route - Ignoring width (Purely Shortest Distance)"""
        try:
            path = nx.dijkstra_path(self.graph, source, target, weight='length')
            return path
        except nx.NetworkXNoPath:
            return None

    def get_network_plot(self, smart_path=None, standard_path=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Color edges by suitability
        edges = self.graph.edges(data=True)
        edge_colors = []
        for u, v, d in edges:
            suit = d.get('suitability', 50)
            if suit > 70: edge_colors.append('#2ecc71') # Green
            elif suit > 40: edge_colors.append('#f39c12') # Orange
            else: edge_colors.append('#e74c3c') # Red

        nx.draw_networkx_nodes(self.graph, pos, node_size=700, node_color='#3498db', ax=ax)
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_weight='bold', font_color='white', ax=ax)
        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, width=2, ax=ax, alpha=0.3)
        
        # Highlight Standard Path (Dashed Red)
        if standard_path:
            std_edges = list(zip(standard_path, standard_path[1:]))
            nx.draw_networkx_edges(self.graph, pos, edgelist=std_edges, edge_color='red', width=3, style='dashed', ax=ax, label="Standard Route")
            
        # Highlight Smart Path (Solid Blue)
        if smart_path:
            smart_edges = list(zip(smart_path, smart_path[1:]))
            nx.draw_networkx_edges(self.graph, pos, edgelist=smart_edges, edge_color='#2980b9', width=5, ax=ax, label="Smart AI Route")
            
        edge_labels = {(u, v): f"{d['width']}m" for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=9, ax=ax)
        
        plt.legend()
        plt.axis('off')
        return fig
