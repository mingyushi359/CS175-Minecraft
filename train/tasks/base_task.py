import math

import numpy as np

class BaseTask:
    # default ObservationFromGrid
    GRID_MIN = {"x": -12, "y": -1, "z": -12}
    GRID_MAX = {"x": 12, "y": 1, "z": 12}

    # shared methods for any Task class
    def __init__(self, GRID_MIN=None, GRID_MAX=None):
        # ObservationFromGrid coordinate range, change accordingly
        if GRID_MIN:
            self.GRID_MIN = GRID_MIN
        if GRID_MAX:
            self.GRID_MAX = GRID_MAX

    def find_nearest_entity(self, info_dict, entity_name: str):
        # find the nearest target entity from info_dict given state
        if not info_dict:
            return None

        agent_name = info_dict.get("Name")
        agent_x = float(info_dict.get("XPos", 0.0))
        agent_y = float(info_dict.get("YPos", 0.0))
        agent_z = float(info_dict.get("ZPos", 0.0))
        agent_yaw = float(info_dict.get("Yaw", 0.0))

        best = None
        for entity in info_dict.get("entities", []):
            if entity.get("name") == agent_name:
                continue
            if entity.get("name") != entity_name:
                continue

            dx = float(entity.get("x", 0.0)) - agent_x
            dy = float(entity.get("y", 0.0)) - agent_y
            dz = float(entity.get("z", 0.0)) - agent_z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            yaw_error = self.yaw_error_to_target(agent_yaw, dx, dz)

            if best is None or distance < best["distance"]:
                # returns the nearest entity's distance and camera angle difference
                best = {
                    "dx": dx,
                    "dy": dy,
                    "dz": dz,
                    "distance": distance, 
                    "yaw_error": yaw_error,
                    }
        return best
    
    def find_nearest_block(self, info_dict, block_name: str):
        # find the nearest block from info_dict given a state
        if not info_dict:
            return None
        
        board = info_dict.get("board", [])
        if not board:
            return None
        
        x_count = self.GRID_MAX["x"] - self.GRID_MIN["x"] + 1
        y_count = self.GRID_MAX["y"] - self.GRID_MIN["y"] + 1
        z_count = self.GRID_MAX["z"] - self.GRID_MIN["z"] + 1

        expected_board_len = x_count * y_count * z_count
        if len(board) != expected_board_len:
            # skip if board length doesn't match
            return None
        
        board_3d = np.array(board, dtype=object).reshape(y_count, z_count, x_count)
        block_indices = np.argwhere(board_3d == block_name)
        if len(block_indices) == 0:
            return None
        
        agent_yaw = float(info_dict.get("Yaw", 0.0))
        
        best = None
        for y_idx, z_idx, x_idx in block_indices:
            dx = self.GRID_MIN["x"] + int(x_idx)
            dy = self.GRID_MIN["y"] + int(y_idx)
            dz = self.GRID_MIN["z"] + int(z_idx)
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            yaw_error = self.yaw_error_to_target(agent_yaw, dx, dz)

            if best is None or distance < best["distance"]:
                best = {
                    "dx": dx,
                    "dy": dy,
                    "dz": dz,
                    "distance": distance, 
                    "yaw_error": yaw_error,
                    }
        return best

    def yaw_error_to_target(self, agent_yaw, dx, dz):
        target_yaw = math.degrees(math.atan2(-dx, dz))
        return abs(self.angle_difference_degrees(agent_yaw, target_yaw))

    def angle_difference_degrees(self, a, b):
        return (a - b + 180.0) % 360.0 - 180.0

    def clamp(self, value, low, high):
        return max(low, min(high, value))