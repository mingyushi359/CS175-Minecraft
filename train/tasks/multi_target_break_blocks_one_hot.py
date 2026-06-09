import random
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tasks.base_task import BaseTask

class Task(BaseTask):
    TASK_ID = 3
    TARGETS = ["diamond_ore", "log", "clay"]
    TARGET_EPISODE_BATCH = 3  # run 3 consecutive episodes for each target-pair combination

    CUSTOM_ACTIONS = [
        "move 1",
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
        "clay": ["clay"],
    }

    # balanced target-layout combination for better generalization and even learning
    TARGET_LAYOUT_PAIRS = [
        ("clay", 0),
        ("diamond_ore", 1),
        ("log", 2),
        ("diamond_ore", 0),
        ("log", 1),
        ("clay", 2),
        ("log", 0),
        ("clay", 1),
        ("diamond_ore", 2),
    ]

    REQUIRED_TOOL_ACTION = {
        "diamond_ore": 3,  # hotbar.1 / pickaxe
        "log": 4,          # hotbar.2 / axe
        "clay": 5,       # hotbar.3 / shovel
    }

    TARGET_DROP_ITEM = {
        "diamond_ore": "diamond",
        "log": "log",
        "clay": "clay_ball",
    }

    # sentence transformers
    TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    TEXT_MODEL_DIM = 3
    TEXT_OUT_DIM = 3
    STATE_DIM = 33  # state dim produced from your build_state, excluding text dim

    # ObservationFromGrid, same as the one in xml
    GRID_MIN = {"x": -12, "y": -1, "z": -12}
    GRID_MAX = {"x": 12, "y": 1, "z": 12}

    # custom rewards
    CORRECT_BREAK_REWARD = 50.0
    WRONG_BREAK_PENALTY = -25.0

    MAX_ATTACK_RANGE = 4.0
    MIN_ATTACK_RANGE = 0.0

    TURN_ACTIONS = []  # left, right
    FORWARD_ACTION = 0
    ATTACK_ACTION = 6
    MOVEMENT_ACTIONS = [0, 1, 2]  # forward, strafe left/right
    TOOL_ACTIONS = [3, 4, 5]

    def __init__(self):
        super().__init__()
        self.current_instruction = None
        self.current_target = None
        self.current_tool_action = None
        # self.text_model = SentenceTransformer(self.TEXT_MODEL_NAME)
        self.current_instruction_embedding = [0.0] * self.TEXT_MODEL_DIM
        self.current_episode = 0  # manaully adjust starting episode if continuing training from checkpoint (check # of rows in monitor.csv)
        self.select_correct_tool_once = False
        self.reached_once = False
        self.penalized_wrong_breaks = set()
        self.embedding_cache = {}
        self.eval = False
        self.current_layout_idx = 0
        self.pair_bag = []

    def reset(self, instruction=None, eval_mode=False):
        self.current_tool_action = None
        self.select_correct_tool_once = False
        self.reached_once = False
        self.penalized_wrong_breaks = set()
        self.eval = eval_mode

        if instruction is None:  # randomly samples an instruction during trianing
            self.current_target = self.choose_target()
            self.current_episode += 1
            template = random.choice(self.INSTRUCTION_TEMPLATE)
            target_alias = random.choice(self.INSTRUCTION_TARGET_ALIASES[self.current_target])
            self.current_instruction = template.format(target_alias)
        else:  # specify instruction during eval
            self.current_instruction = instruction
            self.current_target = self.parse_target(instruction)

        if self.current_target == "diamond_ore":
            self.current_instruction_embedding = [1.0, 0.0, 0.0]
        elif self.current_target == "log":
            self.current_instruction_embedding = [0.0, 1.0, 0.0]
        elif self.current_target == "clay":
            self.current_instruction_embedding = [0.0, 0.0, 1.0]
        # self.current_instruction_embedding = self.encode_instruction(self.current_instruction)

    def choose_target(self):
        # let each target run for some consecutive episodes
        # if self.current_episode < 4000:
        #     self.TARGET_EPISODE_BATCH = 20
        # elif self.current_episode < 5000:
        #     self.TARGET_EPISODE_BATCH = 10
        # elif self.current_episode < 6000:
        #     self.TARGET_EPISODE_BATCH = 5
        # else:
        #     self.TARGET_EPISODE_BATCH = 3

        # target_idx = (self.current_episode // self.TARGET_EPISODE_BATCH) % len(self.TARGETS)
        # return self.TARGETS[target_idx]
        offset = self.current_episode // (self.TARGET_EPISODE_BATCH * len(self.TARGET_LAYOUT_PAIRS)) % len(self.TARGET_LAYOUT_PAIRS)
        pair_idx = ((self.current_episode // self.TARGET_EPISODE_BATCH) + offset) % len(self.TARGET_LAYOUT_PAIRS)

        if len(self.pair_bag) == 0:
            pairs = list(range(len(self.TARGET_LAYOUT_PAIRS)))
            random.shuffle(pairs)

            self.pair_bag = []
            for p in pairs:
                self.pair_bag.extend([p] * self.TARGET_EPISODE_BATCH)
        pair_idx = self.pair_bag.pop()

        self.current_layout_idx = self.TARGET_LAYOUT_PAIRS[pair_idx][1]
        return self.TARGET_LAYOUT_PAIRS[pair_idx][0]

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
        if "clay" in text :
            return "clay"

        raise ValueError(f"Unknown instruction: {instruction}")
    
    def make_random_mission_xml(self, mission_path):
        # randomly shuffles the popsition of the target
        base_xml = Path(mission_path).read_text()

        # zs = [1, 6, 11]
        # random.shuffle(zs)
        # layout = [
        #     [1, 6, 11],
        #     [11, 1, 6],
        #     [6, 11, 1],
        # ]
        # zs = layout[self.current_episode % len(layout)]

        x_z_layout = [
            [(11, 3),
            (11, 6),
            (11, 9),],
            [(11, 9),
            (11, 3),
            (11, 6),],
            [(11, 6),
            (11, 9),
            (11, 3),],
        ]
        xz = x_z_layout[self.current_layout_idx]
        xz = self.get_eval_layout()

        agent_x = random.choice([4.5])
        agent_z = random.choice([6.5])
        agent_yaw = random.choice([270])

        if self.eval:
            xz = self.get_eval_layout()
            # self.current_episode += 1
            # xz = x_z_layout[self.current_episode % len(x_z_layout)]  # fixed layout order for eval
            # xz = random.choice(x_z_layout)  # random layout for eval
            agent_x = 4.5

        return (
                base_xml
                .replace("__DIAMOND_Z__", str(xz[0][1]))
                .replace("__LOG_Z__", str(xz[1][1]))
                .replace("__CLAY_Z__", str(xz[2][1]))
                .replace("__DIAMOND_X__", str(xz[0][0]))
                .replace("__LOG_X__", str(xz[1][0]))
                .replace("__CLAY_X__", str(xz[2][0]))
                .replace("__AGENT_X__", str(agent_x))
                .replace("__AGENT_Z__", str(agent_z))
                .replace("__AGENT_YAW__", str(agent_yaw))
            )
    
    def get_eval_layout(self):
        candidates = [1, 3, 5, 7, 9, 11]

        valid_layouts = []
        for d in candidates:
            for l in candidates:
                for c in candidates:
                    zs = [d, l, c]
                    if len(set(zs)) < 3:
                        continue
                    if min(abs(zs[i] - zs[j]) for i in range(3) for j in range(i + 1, 3)) < 2:
                        continue
                    valid_layouts.append([(11, d), (11, l), (11, c)])

        # deterministic pseudo-random order
        idx = (self.current_episode * 7 + 3) % len(valid_layouts)
        self.current_episode += 1
        return valid_layouts[idx]

    def build_state(self, info_dict):
        # build state for multi target

        # agent camera feature
        yaw = float(info_dict.get("Yaw", 0))
        pitch = float(info_dict.get("Pitch", 0))
        agent_x = float(info_dict.get("XPos", 0))
        agent_z = float(info_dict.get("ZPos", 0))
        agent_features = [yaw / 180.0, pitch / 90.0, agent_x / 12.0, agent_z / 12.0]

        diamond_ore = self.find_nearest_block(info_dict, "diamond_ore")
        log = self.find_nearest_block(info_dict, "log")
        clay = self.find_nearest_block(info_dict, "clay")

        target_features = []
        for target in [diamond_ore, log, clay]:
            if target is not None:
                target_features.extend([
                    target["x"] / 12.0,
                    target["z"] / 12.0,
                    target["dx"] / 12.0,
                    target["dz"] / 12.0,
                    target["distance"] / 17.0,
                    target["yaw_error"] / 180.0,
                ])
            else:
                target_features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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
            1.0 if self.current_tool_action == 3 else 0.0,  # pickaxe
            1.0 if self.current_tool_action == 4 else 0.0,  # axe
            1.0 if self.current_tool_action == 5 else 0.0,  # shovel
        ]

        # line of sight
        los_features = [
            1.0 if self.los_matches_target(info_dict) else 0.0,
            1.0 if self.los_matches_wrong_target(info_dict) else 0.0,
            1.0 if self.los_in_range(info_dict) else 0.0,
        ]

        # target_spatial_features = self.build_spatial_grid(info_dict, ["diamond_ore", "log", "clay"], radius=6)

        compact_features = (
            agent_features + 
            target_features + 
            obstacle_features + 
            tool_features + 
            los_features + 
            [float(self.TASK_ID)]
        )

        return compact_features + self.current_instruction_embedding

    def shape_reward(self, raw_reward, prev_info, curr_info, action, step):
        # additional reward logic
        reward = float(raw_reward)
        reward -= 0.05  # global penalty
        done = False
        metrics = {}  # saving agent stats for debug purposes, not used

        if not curr_info or not prev_info:  # skip if info_dict is empty
            return reward, done, metrics 
        
        # early terminate if y level changes (fall into the pit)
        # if float(curr_info.get("YPos", 4.0)) < 3.5:
        #     reward -= 20.0
        #     done = True
        #     return reward, done, metrics

        # win condition (collect drop item)
        target_item = self.TARGET_DROP_ITEM[self.current_target]
        if self.inventory_contains(curr_info, target_item):
            if self.current_tool_action == self.REQUIRED_TOOL_ACTION[self.current_target]:
                reward += 10.0
            steps_bonus = max(0.0, 30.0 - step) * 1.5  # bonus for finishing early
            reward += steps_bonus
            reward += self.CORRECT_BREAK_REWARD

            done = True
            return reward, done, metrics

        # reward for making an attacking when it should
        if self.los_matches_target(prev_info):
            if action == self.ATTACK_ACTION:
                if not self.reached_once:  
                    reward += 5.0
                    self.reached_once = True
            else:
                reward -= 0.5
        
        # attack reward/penalty
        if action == self.ATTACK_ACTION:
            if self.los_matches_target(prev_info):  # reward already applied
                reward += 0.0
            elif self.los_matches_wrong_target(prev_info):  # breaking the wrong target
                reward -= 2.0
            else:
                reward -= 0.3

        # penalty for breaking other target block
        for other_target in [t for t in self.TARGETS if t != self.current_target]:
            if other_target in self.penalized_wrong_breaks:
                continue

            other_drop = self.TARGET_DROP_ITEM[other_target]

            if self.inventory_contains(curr_info, other_drop):
                reward += self.WRONG_BREAK_PENALTY
                self.penalized_wrong_breaks.add(other_target)

        # tool selection reward
        if action in self.TOOL_ACTIONS:
            self.current_tool_action = action

            if action == self.REQUIRED_TOOL_ACTION[self.current_target]:
                if not self.select_correct_tool_once:
                    reward += 10.0
                    self.select_correct_tool_once = True
                else:
                    reward -= 0.1
            else:
                reward -= 0.5

        prev_target = self.find_nearest_block(prev_info, self.current_target)
        curr_target = self.find_nearest_block(curr_info, self.current_target)
        if curr_target is None or prev_target is None:
            return reward, done, metrics

        # navigation rewards
        prev_dist = prev_target["distance"]
        curr_dist = curr_target["distance"]
        prev_yaw = abs(prev_target["yaw_error"])
        curr_yaw = abs(curr_target["yaw_error"])

        distance_progress = prev_dist - curr_dist
        look_progress = prev_yaw - curr_yaw

        # reward for recovering from the wrong target
        if self.los_matches_wrong_target(prev_info):
            if action in self.TOOL_ACTIONS:
                reward -= 0.8
            elif action in [1, 2]:  # strafe left/right
                if distance_progress > 0:  # moving closer to the correct target
                    reward += 0.3
                else:
                    reward -= 0.3
            elif action == self.FORWARD_ACTION:
                reward -= 0.3

        # reward for camera angle
        if action in self.TURN_ACTIONS:
            reward += 0.08 * self.clamp(look_progress / 30.0, -1.0, 1.0)

        # reward for moving closer
        if action in self.MOVEMENT_ACTIONS:
            if distance_progress > 0:
                reward += 1.2 * self.clamp(distance_progress, -1.0, 1.0)
            else:
                reward -= 0.25

        # penalty for moving away
        if action == self.FORWARD_ACTION and prev_yaw > 95.0:
            reward -= 0.15

        # penalty for turning/strafing when already facing the target
        if self.get_los_type(prev_info) == self.current_target:
            # if already facing the target, penalize turning/strafing
            if action in [1, 2]:
                reward -= 0.3

            # if already facing the target, penalize non move forward action
            if not self.los_in_range(prev_info):
                if action == self.FORWARD_ACTION:
                    reward += 0.1
                else:  
                    reward -= 0.3

        # if action in self.TURN_ACTIONS and prev_yaw < 25.0:
        #     reward -= 0.10


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
            block = self.get_board_block(info_dict, fx * d, 1, fz * d)
            if block == self.current_target:
                return True

        return False 
    
    def pit_in_direction(self, info_dict, dx, dz):
        # check if there's a pit in either direction
        below = self.get_board_block(info_dict, dx, -1, dz)
        return below == "air"
    
    def get_los(self, info):
        return info.get("LineOfSight", {})

    def get_los_type(self, info):
        return self.get_los(info).get("type", None)

    def los_in_range(self, info):
        # is target interactable (left/right click)
        return self.get_los(info).get("inRange", False)

    def los_matches_target(self, info):
        # crosshair facing the target and in range for interaction
        return (
            self.get_los_type(info) == self.current_target
            and self.los_in_range(info)
        )

    def los_matches_wrong_target(self, info):
        # crosshair 
        block = self.get_los_type(info)

        return (
            block in self.TARGETS
            and block != self.current_target
            and self.los_in_range(info)
        )