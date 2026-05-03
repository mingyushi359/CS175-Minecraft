# defines task-specific reward shaping for mob chase
import math

TASK_ID = 2
TARGET_ENTITY = "Pig"
# custom actions, required for train_ppo.py, else actions will be inferred from the XML only
CUSTOM_ACTIONS = ["move 1", "turn 1", "turn -1", "strafe 1", "strafe -1"]

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

# BACKWARD_ACTION = 1
# BACKWARD_PENALTY = -0.1

# ATTACK_ACTION = 4
# FAR_ATTACK_PENALTY = -0.05

stuck_movement_counter = 0

def reset():
    global stuck_movement_counter
    stuck_movement_counter = 0


def shape_reward(raw_reward, prev_info, curr_info, action, step):
    global stuck_movement_counter

    reward = float(raw_reward)
    done = False
    metrics = {}

    prev_target = find_target(prev_info)
    curr_target = find_target(curr_info)

    if curr_target is None:
        metrics["target_visible"] = 0.0
        return reward, done, metrics

    metrics["target_visible"] = 1.0
    metrics["target_distance"] = curr_target["distance"]
    metrics["target_yaw_error"] = curr_target["yaw_error"]

    if prev_target is not None:
        distance_progress = prev_target["distance"] - curr_target["distance"]
        look_progress = prev_target["yaw_error"] - curr_target["yaw_error"]

        # reward += DISTANCE_PROGRESS_SCALE * clamp(distance_progress, -1.0, 1.0)
        reward += LOOK_PROGRESS_SCALE * clamp(look_progress, -10.0, 10.0)

        if curr_target["yaw_error"] <= OK_FACING_DEGREES:
            # give distance reward only when roughly facing the target
            reward += DISTANCE_PROGRESS_SCALE * clamp(distance_progress, -1.0, 1.0)

        if curr_target["yaw_error"] <= GOOD_FACING_DEGREES:
            # extra reward for facing the target
            reward += GOOD_FACING_REWARD

        if action == FORWARD_ACTION and curr_target["yaw_error"] <= GOOD_FACING_DEGREES:
            # reward for moving forward when facing the target
            reward += 0.08

        if action in TURN_ACTIONS and curr_target["yaw_error"] <= GOOD_FACING_DEGREES:
            # penalty for turning when already facing the target
            reward += UNNECESSARY_TURN_PENALTY

        if action == FORWARD_ACTION and curr_target["yaw_error"] > BAD_FACING_DEGREES:
            # penalty for moving forward when not facing the target
            reward += BAD_FORWARD_PENALTY

        if action in MOVEMENT_ACTIONS and distance_progress <= 0.01:
            # penalty for not making no progress when moving forward (e.g. moving against a wall)
            reward += NO_PROGRESS_PENALTY
            stuck_movement_counter += 1
            if stuck_movement_counter >= 5:
                reward -= 0.40
        else:
            stuck_movement_counter = 0
                

    if curr_target["distance"] <= REACH_DISTANCE and curr_target["yaw_error"] <= GOOD_FACING_DEGREES:
        # reward for reaching and facing the target
        reward += REACH_REWARD
        done = True
        # print(f"REACHED target at step={step}, distance={curr_target['distance']:.2f}")

    # if action == ATTACK_ACTION and (curr_target["distance"] > REACH_DISTANCE + 0.5 or curr_target["yaw_error"] > BAD_FACING_DEGREES):
    #     # penalty for attacking when not close to and facing the target
    #     reward += FAR_ATTACK_PENALTY

    # if action == BACKWARD_ACTION:
    #     # penalty for moving backwards
    #     reward += BACKWARD_PENALTY

    return reward, done, metrics


def find_target(info_dict):
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
        if entity.get("name") != TARGET_ENTITY:
            continue

        dx = float(entity.get("x", 0.0)) - agent_x
        dy = float(entity.get("y", 0.0)) - agent_y
        dz = float(entity.get("z", 0.0)) - agent_z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        yaw_error = yaw_error_to_target(agent_yaw, dx, dz)

        if best is None or distance < best["distance"]:
            best = {
                "distance": distance,
                "yaw_error": yaw_error,
            }

    return best


def yaw_error_to_target(agent_yaw, dx, dz):
    target_yaw = math.degrees(math.atan2(-dx, dz))
    return abs(angle_difference_degrees(agent_yaw, target_yaw))


def angle_difference_degrees(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def clamp(value, low, high):
    return max(low, min(high, value))
