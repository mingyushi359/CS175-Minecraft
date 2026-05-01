import math

TASK_ID = 2
TARGET_ENTITY = "Pig"

REACH_DISTANCE = 2.0
REACH_REWARD = 25.0
DISTANCE_PROGRESS_SCALE = 5.0
LOOK_PROGRESS_SCALE = 0.02

# GOOD_FACING_DEGREES = 30.0
BAD_FACING_DEGREES = 90.0

FORWARD_ACTION = 0
BAD_FORWARD_PENALTY = -0.02

# BACKWARD_ACTION = 1
# BACKWARD_PENALTY = -0.1

# ATTACK_ACTION = 4
# FAR_ATTACK_PENALTY = -0.05

def reset():
    pass


def shape_reward(raw_reward, prev_info, curr_info, action, step):
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

        reward += DISTANCE_PROGRESS_SCALE * clamp(distance_progress, -1.0, 1.0)
        reward += LOOK_PROGRESS_SCALE * clamp(look_progress, -10.0, 10.0)

    if curr_target["distance"] <= REACH_DISTANCE:
        # reward for reaching the target
        reward += REACH_REWARD
        done = True
        # print(f"REACHED target at step={step}, distance={curr_target['distance']:.2f}")

    if action == FORWARD_ACTION and curr_target["yaw_error"] > BAD_FACING_DEGREES:
        # penalty for moving forward when not facing the target
        reward += BAD_FORWARD_PENALTY

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
