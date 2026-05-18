import random
from sentence_transformers import SentenceTransformer
from tasks.base_task import BaseTask

class Task(BaseTask):
    TASK_ID = 3
    TARGETS = ["diamond_ore","log","sand"]

    CUSTOM_ACTIONS = [
        "move 1",
        "turn 1",
        "turn -1",
        "strafe 1",
        "strafe -1",
        "hotbar.1 1",  # pickaxe
        "hotbar.2 1",  # axe
        "hotbar.3 1",  # shovel
        "attack 1",
    ]
    INSTRUCTION_TEMPLATE = [
        "break the {}",
        # "mine the {}",
        # "destroy the {}",
    ]

    INSTRUCTION_TARGET_ALIASES = {
        "diamond_ore": ["diamond ore", 
                        # "diamond block", 
                        # "ore"
                        ],
        "log": ["log", 
                # "wood", 
                # "wooden block"
                ],
        "sand": ["sand"],
    }

    REQUIRED_TOOL_ACTION = {
        "diamond_ore": 5,  # hotbar.1 / pickaxe
        "log": 6,          # hotbar.2 / axe
        "sand": 7,       # hotbar.3 / shovel
    }

    TARGET_DROP_ITEM = {
        "diamond_ore": "diamond",
        "log": "log",
        "sand": "sand",
    }

    # sentence transformers
    TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    TEXT_MODEL_DIM = 384
    TEXT_OUT_DIM = 64
    STATE_DIM = 25  # state dim produced from your build_state, excluding text dim

    # ObservationFromGrid, same as the one in xml
    GRID_MIN = {"x": -12, "y": -1, "z": -12}
    GRID_MAX = {"x": 12, "y": 1, "z": 12}

    # custom rewards
    REACH_DISTANCE = 2.0
    REACH_REWARD = 80.0
    DISTANCE_PROGRESS_SCALE = 1.0
    LOOK_PROGRESS_SCALE = 0.03

    MAX_ATTACK_RANGE = 3.3
    MIN_ATTACK_RANGE = 0.0

    GOOD_FACING_DEGREES = 45.0
    OK_FACING_DEGREES = 60.0
    BAD_FACING_DEGREES = 90.0
    GOOD_FACING_REWARD = 0.03

    TURN_ACTIONS = [1, 2]  # left, right
    UNNECESSARY_TURN_PENALTY = -0.08

    FORWARD_ACTION = 0
    BAD_FORWARD_PENALTY = -0.10
    NO_PROGRESS_PENALTY = -0.20

    ATTACK_ACTION = 8
    MOVEMENT_ACTIONS = [0, 3, 4]  # forward, strafe left/right
    TOOL_ACTIONS = [5, 6, 7]

    def __init__(self):
        super().__init__()
        self.stuck_movement_counter = 0
        self.current_instruction = None
        self.current_target = None
        self.current_tool_action = 5
        self.text_model = SentenceTransformer(self.TEXT_MODEL_NAME)
        self.current_instruction_embedding = [0.0] * self.TEXT_MODEL_DIM
        self.select_correct_tool_once = False
        self.reached_once = False
        self.embedding_cache = {}

    def reset(self, instruction=None, eval_mode=False):
        self.stuck_movement_counter = 0
        self.current_tool_action = 5
        self.select_correct_tool_once = False
        self.reached_once = False

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

        if "diamond" in text or "ore" in text:
            return "diamond_ore"
        if "log" in text or "wood" in text or "wooden" in text:
            return "log"
        if "sand" in text :
            return "sand"

        raise ValueError(f"Unknown instruction: {instruction}")

    def build_state(self, info_dict):
        # build state for multi target

        # agent camera feature
        yaw = float(info_dict.get("Yaw", 0))
        pitch = float(info_dict.get("Pitch", 0))
        agent_features = [yaw / 180.0, pitch / 90.0]

        diamond_ore = self.find_nearest_block(info_dict, "diamond_ore")
        log = self.find_nearest_block(info_dict, "log")
        sand = self.find_nearest_block(info_dict, "sand")

        target_features = []
        for target in [diamond_ore, log, sand]:
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

        # pit in either direction also represents obstacle
        front = float(self.is_block(self.get_board_block(info_dict, fx, 0, fz)) or self.pit_in_direction(info_dict, fx, fz))
        back = float(self.is_block(self.get_board_block(info_dict, -fx, 0, -fz)) or self.pit_in_direction(info_dict, -fx, -fz))
        left = float(self.is_block(self.get_board_block(info_dict, lx, 0, lz)) or self.pit_in_direction(info_dict, lx, lz))
        right = float(self.is_block(self.get_board_block(info_dict, rx, 0, rz)) or self.pit_in_direction(info_dict, rx, rz))

        obstacle_features = [front, back, left, right]

        tool_features = [
            1.0 if self.current_tool_action == 5 else 0.0,  # pickaxe
            1.0 if self.current_tool_action == 6 else 0.0,  # axe
            1.0 if self.current_tool_action == 7 else 0.0,  # shovel
        ]

        return agent_features + target_features + obstacle_features + tool_features + [float(self.TASK_ID)] + self.current_instruction_embedding

    def shape_reward(self, raw_reward, prev_info, curr_info, action, step):
        # additional reward logic
        reward = float(raw_reward)
        reward -= 0.01  # global penalty
        done = False
        metrics = {}  # saving agent stats for debug purposes, not used

        if not curr_info or not prev_info:  # skip if info_dict is empty
            return reward, done, metrics 
        
        if float(curr_info.get("YPos", 4.0)) < 3.5:
            # terminate if y level changes (fall into the pit)
            reward -= 10.0
            done = True
            return reward, done, metrics

        prev_target = self.find_nearest_block(prev_info, self.current_target)
        curr_target = self.find_nearest_block(curr_info, self.current_target)

        target_item = self.TARGET_DROP_ITEM[self.current_target]
        if self.inventory_contains(curr_info, target_item):
            # win condition of collecting target drop item
            reward += self.REACH_REWARD

            done = True
            return reward, done, metrics

        if curr_target is None or prev_target is None:
            return reward, done, metrics
        
        # prev_distance = prev_target["distance"]
        # in_attack_range = self.MIN_ATTACK_RANGE < prev_distance <= self.MAX_ATTACK_RANGE

        target_ahead = self.block_ahead_matches_target(prev_info, max_dist=int(self.MAX_ATTACK_RANGE + 1))
        if action == self.ATTACK_ACTION:
            # reward for attacking while near and facing the target block
            if target_ahead:
                reward += 0.3
            else:
                reward -= 0.3

        # avoid pit
        yaw = float(prev_info.get("Yaw", 0))
        fx, fz = self.yaw_to_direction(yaw)
        lx, lz = fz, -fx
        rx, rz = -fz, fx
        if action == self.FORWARD_ACTION and self.pit_in_direction(prev_info, fx, fz):
            # forward into pit penalty
            reward -= 1.0
        if action == 3 and self.pit_in_direction(prev_info, lx, lz):
            # strafe into pit penalty
            reward -= 1.0
        if action == 4 and self.pit_in_direction(prev_info, rx, rz):
            reward -= 1.0

        # navigation rewards
        distance_progress = prev_target["distance"] - curr_target["distance"]

        if prev_target["yaw_error"] <= self.GOOD_FACING_DEGREES:
            # give distance reward when roughly facing the target
            reward += self.DISTANCE_PROGRESS_SCALE * self.clamp(distance_progress, -1.0, 1.0)
            # more reward for moving towards to the object
            if action == self.FORWARD_ACTION:
                reward += 0.08

            if action in self.TURN_ACTIONS:
                # penalty for turning when already facing the target
                reward += self.UNNECESSARY_TURN_PENALTY

        if action == self.FORWARD_ACTION and prev_target["yaw_error"] > self.BAD_FACING_DEGREES:
            # penalty for moving forward when not facing the target
            reward += self.BAD_FORWARD_PENALTY

        prev_x = float(prev_info.get("XPos", 0.0))
        prev_z = float(prev_info.get("ZPos", 0.0))
        curr_x = float(curr_info.get("XPos", 0.0))
        curr_z = float(curr_info.get("ZPos", 0.0))
        pos_delta = ((curr_x - prev_x) ** 2 + (curr_z - prev_z) ** 2) ** 0.5
        if action in self.MOVEMENT_ACTIONS and pos_delta < 0.03:
            # penalty for not making no progress when moving forward (like moving against a wall)
            reward += self.NO_PROGRESS_PENALTY

        if curr_target["distance"] <= self.MAX_ATTACK_RANGE and curr_target["yaw_error"] <= self.GOOD_FACING_DEGREES:
            # reward for reaching and facing the target
            if not self.reached_once:
                reward += 10.0
                self.reached_once = True
            # done = True

        return reward, done, metrics

    def inventory_contains(self, info_dict, item_name):
        # check if inventory contains item
        for i in range(9):
            item = info_dict.get(f"InventorySlot_{i}_item", "")
            if item == item_name:
                return True
        return False 
    
    def block_ahead_matches_target(self, info_dict, max_dist=3):
        # check if target block is in front
        yaw = float(info_dict.get("Yaw", 0))
        fx, fz = self.yaw_to_direction(yaw)

        for d in range(1, max_dist + 1):
            block = self.get_board_block(info_dict, fx * d, 0, fz * d)
            if block == self.current_target:
                return True

        return False 
    
    def pit_in_direction(self, info_dict, dx, dz):
        # check if there's a pit in either direction
        below = self.get_board_block(info_dict, dx, -1, dz)
        return below == "air"