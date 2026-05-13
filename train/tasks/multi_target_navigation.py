import random
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from tasks.base_task import BaseTask

class Task(BaseTask):
    TASK_ID = 2
    TARGETS = ["pig", "emerald", "log"]

    CUSTOM_ACTIONS = ["move 1", "turn 1", "turn -1", "strafe 1", "strafe -1"]

    INSTRUCTION_TEMPLATE = [
        "go to the {}",
        "find the {}",
        "move to the {}",
    ]

    INSTRUCTION_TARGET_ALIASES = {
        "pig": ["pig", "animal"],
        "emerald": ["emerald", "green block"],
        "log": ["log", "wood", "wooden block"],
    }

    # sentence transformers
    TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    TEXT_MODEL_DIM = 384
    TEXT_OUT_DIM = 64
    STATE_DIM = 22  # state dim produced from your build_state, excluding text dim

    # ObservationFromGrid, same as the one in xml
    GRID_MIN = {"x": -12, "y": -1, "z": -12}
    GRID_MAX = {"x": 12, "y": 1, "z": 12}

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
        super().__init__()
        self.stuck_movement_counter = 0
        self.current_instruction = None
        self.current_target = None
        self.text_model = SentenceTransformer(self.TEXT_MODEL_NAME)
        self.current_instruction_embedding = [0.0] * self.TEXT_MODEL_DIM
        self.embedding_cache = {}

    def reset(self, instruction=None, eval_mode=False):
        self.stuck_movement_counter = 0

        if instruction is None:  # randomly samples an instruction during trianing
            self.current_target = random.choice(self.TARGETS)
            template = random.choice(self.INSTRUCTION_TEMPLATE)
            target_alias = random.choice(self.INSTRUCTION_TARGET_ALIASES[self.current_target])
            self.current_instruction = template.format(target_alias)
        else:  # specify instruction during eval
            self.current_instruction = instruction
            self.current_target = self.parse_target(instruction)

        self.current_instruction_embedding = self.encode_instruction(self.current_instruction)

    def encode_instruction(self, text):
        # sentence transformer
        if text in self.embedding_cache:
            embedding = self.embedding_cache[text]
        else:  # cache embeddings
            embedding = self.text_model.encode(text, normalize_embeddings=True).astype("float32")
            self.embedding_cache[text] = embedding
        return embedding.tolist()

    def parse_target(self, instruction):
        # manually parse instruction for the purpose of calculating eval reward
        text = instruction.lower()

        if "pig" in text or "chase" in text:
            return "pig"
        if "emerald" in text or "green" in text:
            return "emerald"
        if "log" in text or "wood" in text or "wooden" in text:
            return "log"

        raise ValueError(f"Unknown instruction: {instruction}")

    def build_state(self, info_dict):
        # build state for multi target

        # agent camera feature
        yaw = float(info_dict.get("Yaw", 0))
        pitch = float(info_dict.get("Pitch", 0))
        agent_features = [yaw / 180.0, pitch / 90.0]

        pig = self.find_nearest_entity(info_dict, "Pig")
        emerald = self.find_nearest_block(info_dict, "emerald_block")
        log = self.find_nearest_block(info_dict, "log")

        target_features = []
        for target in [pig, emerald, log]:
            if target is not None:
                target_features.extend([
                    target["dx"] / 12.0,
                    target["dy"] / 5.0,
                    target["dz"] / 12.0,
                    target["distance"] / 17.0,
                    target["yaw_error"] / 180.0,
                ])
            else:
                target_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # direct obstacle relative to the camera angle
        fx, fz = self.yaw_to_direction(yaw)
        lx, lz = fz, -fx
        rx, rz = -fz, fx

        front = float(self.is_block(self.get_board_block(info_dict, fx, 0, fz)))
        back = float(self.is_block(self.get_board_block(info_dict, -fx, 0, -fz)))
        left = float(self.is_block(self.get_board_block(info_dict, lx, 0, lz)))
        right = float(self.is_block(self.get_board_block(info_dict, rx, 0, rz)))

        obstacle_features = [front, back, left, right]

        return agent_features + target_features + obstacle_features + [float(self.TASK_ID)] + self.current_instruction_embedding

    def shape_reward(self, raw_reward, prev_info, curr_info, action, step):
        # additional reward logic
        reward = float(raw_reward)
        done = False
        metrics = {}  # saving agent stats for debug purposes, not used

        prev_target = self.find_nearest_target(prev_info)
        curr_target = self.find_nearest_target(curr_info)

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
    
    def find_nearest_target(self, info_dict):
        # picks the correct find_nearest_x function based on current target
        if self.current_target == "pig":
            return self.find_nearest_entity(info_dict, "Pig")
        if self.current_target == "emerald":
            return self.find_nearest_block(info_dict, "emerald_block")
        if self.current_target == "log":
            return self.find_nearest_block(info_dict, "log")
        return None
