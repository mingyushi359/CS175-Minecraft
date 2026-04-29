# train/tasks/mob_chase.py

import math

TASK_ID = 2


def shape_reward(raw_reward, prev_info, curr_info):
    reward = float(raw_reward)
    done = False

    prev_dist = nearest_pig_distance(prev_info)
    curr_dist = nearest_pig_distance(curr_info)
    prev_yaw_error = pig_yaw_error(prev_info)
    curr_yaw_error = pig_yaw_error(curr_info)

    if prev_dist is not None and curr_dist is not None:
        distance_progress = prev_dist - curr_dist
        distance_progress = max(-1.0, min(1.0, distance_progress))
        reward += 1.0 * distance_progress

    if prev_yaw_error is not None and curr_yaw_error is not None:
        yaw_progress = prev_yaw_error - curr_yaw_error
        yaw_progress = max(-10.0, min(10.0, yaw_progress))
        reward += 0.005 * yaw_progress

    if curr_yaw_error is not None and curr_yaw_error <= 10.0:
        reward += 0.01

    if curr_dist is not None and curr_dist <= 1.5:
        reward += 50.0
        done = True

    return reward, done


def nearest_pig_distance(info_dict):
    if not info_dict:
        return None

    x = float(info_dict.get("XPos", 0.0))
    y = float(info_dict.get("YPos", 0.0))
    z = float(info_dict.get("ZPos", 0.0))
    agent_name = info_dict.get("Name")

    best = None
    for entity in info_dict.get("entities", []):
        if entity.get("name") == agent_name:
            continue
        if entity.get("name") != "Pig":
            continue

        dx = float(entity.get("x", 0.0)) - x
        dy = float(entity.get("y", 0.0)) - y
        dz = float(entity.get("z", 0.0)) - z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if best is None or dist < best:
            best = dist

    return best


def pig_yaw_error(info_dict):
    if not info_dict:
        return None

    x = float(info_dict.get("XPos", 0.0))
    z = float(info_dict.get("ZPos", 0.0))
    agent_yaw = float(info_dict.get("Yaw", 0.0))
    agent_name = info_dict.get("Name")

    closest_entity = None
    closest_dist2 = None

    for entity in info_dict.get("entities", []):
        if entity.get("name") == agent_name:
            continue
        if entity.get("name") != "Pig":
            continue

        dx = float(entity.get("x", 0.0)) - x
        dz = float(entity.get("z", 0.0)) - z
        dist2 = dx * dx + dz * dz

        if closest_dist2 is None or dist2 < closest_dist2:
            closest_entity = entity
            closest_dist2 = dist2

    if closest_entity is None:
        return None

    dx = float(closest_entity.get("x", 0.0)) - x
    dz = float(closest_entity.get("z", 0.0)) - z

    target_yaw = math.degrees(math.atan2(-dx, dz))
    return abs(angle_difference_degrees(agent_yaw, target_yaw))


def angle_difference_degrees(a, b):
    return (a - b + 180.0) % 360.0 - 180.0
