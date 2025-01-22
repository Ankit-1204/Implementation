from lux.utils import direction_to
import sys
import numpy as np
class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
   
        unit_mask = np.array(obs["units_mask"][self.team_id]) 
        unit_positions = np.array(obs["units"]["position"][self.team_id])  
        unit_energys = np.array(obs["units"]["energy"][self.team_id])  
        observed_relic_node_positions = np.array(obs["relic_nodes"])  
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) 
        team_points = np.array(obs["team_points"])  
    
        
        available_unit_ids = np.where(unit_mask)[0]
    
      
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
    
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
    
    
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])
    
   
        def fitness(unit_id):
            unit_pos = unit_positions[unit_id]
            energy_collected = unit_energys[unit_id]
            if len(self.relic_node_positions) > 0:
                nearest_relic_node = self.relic_node_positions[0]
                distance_to_relic = abs(unit_pos[0] - nearest_relic_node[0]) + abs(unit_pos[1] - nearest_relic_node[1])
                return energy_collected - 0.5 * distance_to_relic  
            return energy_collected
    
       
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            if len(self.relic_node_positions) > 0:
                nearest_relic_node = self.relic_node_positions[0]
                distance_to_relic = abs(unit_pos[0] - nearest_relic_node[0]) + abs(unit_pos[1] - nearest_relic_node[1])
    
                
                if distance_to_relic <= 4:
                    random_direction = np.random.randint(0, 5)
                    actions[unit_id] = [random_direction, 0, 0]
                else:
                    actions[unit_id] = [direction_to(unit_pos, nearest_relic_node), 0, 0]
            else:
      
                if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                    rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                    self.unit_explore_locations[unit_id] = rand_loc
                actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
    
     
        total_fitness = sum(fitness(unit_id) for unit_id in available_unit_ids)
        if total_fitness <= 0:
            total_fitness = 1e-6
        for unit_id in available_unit_ids:
            prob = (fitness(unit_id)) / total_fitness
            if np.random.rand() < prob:
                if len(self.relic_node_positions) > 0:
                    actions[unit_id] = [direction_to(unit_positions[unit_id], self.relic_node_positions[0]), 0, 0]
                else:
                    random_direction = np.random.randint(0, 5)
                    actions[unit_id] = [random_direction, 0, 0]
    
     
        for unit_id in available_unit_ids:
            if fitness(unit_id) < 1: 
                rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                self.unit_explore_locations[unit_id] = rand_loc
                actions[unit_id] = [direction_to(unit_positions[unit_id], self.unit_explore_locations[unit_id]), 0, 0]
        return actions
