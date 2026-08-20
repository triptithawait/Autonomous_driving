import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl

class SmartNavFuzzySystem:
    def __init__(self):
        # Antecedents (Inputs)
        self.road_width = ctrl.Antecedent(np.arange(0, 11, 1), 'road_width')  # 0-10 meters
        self.vehicle_width = ctrl.Antecedent(np.arange(0, 5, 0.1), 'vehicle_width') # 0-5 meters
        
        # Consequent (Output)
        self.suitability = ctrl.Consequent(np.arange(0, 101, 1), 'suitability') # 0-100 score

        # Membership Functions for Road Width
        self.road_width['narrow'] = fuzzy.trimf(self.road_width.universe, [0, 0, 4])
        self.road_width['medium'] = fuzzy.trimf(self.road_width.universe, [3, 5, 7])
        self.road_width['wide'] = fuzzy.trimf(self.road_width.universe, [6, 10, 10])

        # Membership Functions for Vehicle Width
        self.vehicle_width['small'] = fuzzy.trimf(self.vehicle_width.universe, [0, 0, 2])
        self.vehicle_width['medium'] = fuzzy.trimf(self.vehicle_width.universe, [1.5, 2.5, 3.5])
        self.vehicle_width['large'] = fuzzy.trimf(self.vehicle_width.universe, [3, 5, 5])

        # Membership Functions for Suitability
        self.suitability['very_low'] = fuzzy.trimf(self.suitability.universe, [0, 0, 25])
        self.suitability['low'] = fuzzy.trimf(self.suitability.universe, [20, 40, 60])
        self.suitability['medium'] = fuzzy.trimf(self.suitability.universe, [50, 65, 80])
        self.suitability['high'] = fuzzy.trimf(self.suitability.universe, [70, 85, 100])
        self.suitability['very_high'] = fuzzy.trimf(self.suitability.universe, [90, 100, 100])

        # Fuzzy Rules
        self.rules = [
            ctrl.Rule(self.road_width['narrow'] & self.vehicle_width['large'], self.suitability['very_low']),
            ctrl.Rule(self.road_width['narrow'] & self.vehicle_width['medium'], self.suitability['low']),
            ctrl.Rule(self.road_width['narrow'] & self.vehicle_width['small'], self.suitability['medium']),
            
            ctrl.Rule(self.road_width['medium'] & self.vehicle_width['large'], self.suitability['low']),
            ctrl.Rule(self.road_width['medium'] & self.vehicle_width['medium'], self.suitability['high']),
            ctrl.Rule(self.road_width['medium'] & self.vehicle_width['small'], self.suitability['very_high']),
            
            ctrl.Rule(self.road_width['wide'], self.suitability['very_high']),
        ]

        # Control System
        self.routing_ctrl = ctrl.ControlSystem(self.rules)
        self.simulator = ctrl.ControlSystemSimulation(self.routing_ctrl)

    def compute_suitability(self, r_width, v_width):
        """
        Computes suitability score 0-100.
        r_width: estimated road width in meters
        v_width: vehicle width in meters
        """
        try:
            self.simulator.input['road_width'] = r_width
            self.simulator.input['vehicle_width'] = v_width
            self.simulator.compute()
            return self.simulator.output['suitability']
        except Exception as e:
            # Fallback for boundary conditions
            if r_width < v_width: return 5
            return 50
