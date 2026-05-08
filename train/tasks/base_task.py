import math

class BaseTask:
    # shared methods for any Task class
    def __init__(self):
        pass

    def find_nearest_entity(self, info_dict, entity_name):
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

    def yaw_error_to_target(self, agent_yaw, dx, dz):
        target_yaw = math.degrees(math.atan2(-dx, dz))
        return abs(self.angle_difference_degrees(agent_yaw, target_yaw))

    def angle_difference_degrees(self, a, b):
        return (a - b + 180.0) % 360.0 - 180.0

    def clamp(self, value, low, high):
        return max(low, min(high, value))