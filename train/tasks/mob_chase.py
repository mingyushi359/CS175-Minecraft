# defines task-specific reward shaping for mob chase
import math

class Task:
    # task specific class for chasing a mob (pig)
    TASK_ID = 1
    TARGET_ENTITY = "Pig"

    # custom actions, required for train_ppo.py, else actions will be inferred from the XML only
    CUSTOM_ACTIONS = ["move 1", "turn 1", "turn -1", "strafe 1", "strafe -1"]

    # custom rewards
    REACH_DISTANCE = 2.0
    REACH_REWARD = 25.0
    DISTANCE_PROGRESS_SCALE = 3.0
    LOOK_PROGRESS_SCALE = 0.03

    GOOD_FACING_DEGREES = 45.0
    OK_FACING_DEGREES = 60.0
    BAD_FACING_DEGREES = 90.0
    GOOD_FACING_REWARD = 0.03

    TURN_ACTIONS = [1, 2]  # left, right
    UNNECESSARY_TURN_PENALTY = -0.08

    FORWARD_ACTION = 0
    BAD_FORWARD_PENALTY = -0.10
    NO_PROGRESS_PENALTY = -0.20

    MOVEMENT_ACTIONS = [0, 3, 4]  # forward, strafe left/right

    def __init__(self):
        self.stuck_movement_counter = 0

    def reset(self):
        self.stuck_movement_counter = 0

    def build_state(self, info_dict):
        # builds necessary state for mob chase

        # agent camera feature
        yaw = float(info_dict.get("Yaw", 0))
        pitch = float(info_dict.get("Pitch", 0))
        agent_features = [yaw / 180.0, pitch / 90.0]

        target = self.find_target(info_dict)  # finds the nearest pig
        if target is not None:
            entity_features = [
                target["dx"] / 12.0,
                target["dy"] / 5.0,
                target["dz"] / 12.0,
                target["distance"] / 17.0,
                target["yaw_error"] / 180.0,
            ]
        else:
            entity_features = [0.0, 0.0, 0.0, 0.0, 0.0]

        return agent_features + entity_features + [float(self.TASK_ID)]

    def shape_reward(self, raw_reward, prev_info, curr_info, action, step):
        # additional reward logic
        reward = float(raw_reward)
        done = False
        metrics = {}  # saving agent stats for debug purposes, not used

        prev_target = self.find_target(prev_info)
        curr_target = self.find_target(curr_info)

        if curr_target is None:
            return reward, done, metrics

        if prev_target is not None:
            distance_progress = prev_target["distance"] - curr_target["distance"]
            look_progress = prev_target["yaw_error"] - curr_target["yaw_error"]

            # reward += self.DISTANCE_PROGRESS_SCALE * clamp(distance_progress, -1.0, 1.0)
            reward += self.LOOK_PROGRESS_SCALE * self.clamp(look_progress, -10.0, 10.0)

            if curr_target["yaw_error"] <= self.OK_FACING_DEGREES:
                # give distance reward only when roughly facing the target
                reward += self.DISTANCE_PROGRESS_SCALE * self.clamp(distance_progress, -1.0, 1.0)

            if curr_target["yaw_error"] <= self.GOOD_FACING_DEGREES:
                # extra reward for facing the target
                reward += self.GOOD_FACING_REWARD

            if action == self.FORWARD_ACTION and curr_target["yaw_error"] <= self.GOOD_FACING_DEGREES:
                # reward for moving forward when facing the target
                reward += 0.08

            if action in self.TURN_ACTIONS and curr_target["yaw_error"] <= self.GOOD_FACING_DEGREES:
                # penalty for turning when already facing the target
                reward += self.UNNECESSARY_TURN_PENALTY

            if action == self.FORWARD_ACTION and curr_target["yaw_error"] > self.BAD_FACING_DEGREES:
                # penalty for moving forward when not facing the target
                reward += self.BAD_FORWARD_PENALTY

            if action in self.MOVEMENT_ACTIONS and distance_progress <= 0.01:
                # penalty for not making no progress when moving forward (like moving against a wall)
                reward += self.NO_PROGRESS_PENALTY
                self.stuck_movement_counter += 1
                if self.stuck_movement_counter >= 5:
                    reward -= 0.40
            else:
                self.stuck_movement_counter = 0

        if curr_target["distance"] <= self.REACH_DISTANCE and curr_target["yaw_error"] <= self.GOOD_FACING_DEGREES:
            # reward for reaching and facing the target
            reward += self.REACH_REWARD
            done = True
            # print(f"REACHED target at step={step}, distance={curr_target['distance']:.2f}")

        return reward, done, metrics


    def find_target(self, info_dict):
        # find the nearest target entity (pig) from info_dict given state
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
            if entity.get("name") != self.TARGET_ENTITY:
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
