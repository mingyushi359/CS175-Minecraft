# Note: Current version is only tested for the farming task, but should work for other place_item tasks as well.
import time
import random
import math
from sentence_transformers import SentenceTransformer
from tasks.base_task import BaseTask


class Task(BaseTask):

    # Number of steps to confirm a farmland block to store in memory.
    FARMLAND_CONFIRM_STEPS = 3

    # Minimum overlap ratio required before accepting a new local-grid-to-world
    # anchor once memory already exists. This uses only observation consistency,
    # not fixed farmland count or fixed layout.
    ANCHOR_MIN_OVERLAP_RATIO = 0.8

    # debug section code - does not affect agent state or reward
    DEBUG_PRINT = False
    DEBUG_PRINT_EVERY_N_STEPS = 100
    SLOW_STEP_WARN_SECONDS = 0.20

    TASK_ID = 3

    TARGETS = [
        "wheat",
        "carrot", 
        "potato"
        ]

    TARGET_EPISODE_BATCH = 3

    CUSTOM_ACTIONS = [
        "move 1",      # 1: forward
        # "turn 1",      # 2: turn right
        # "turn -1",     # 3: turn left
        "strafe 1",    # 4: strafe right
        "strafe -1",   # 5: strafe left
        "hotbar.1 1",  # 6: select wheat seeds
        "hotbar.2 1",  # 7: select carrot
        "hotbar.3 1",  # 8: select potato
        "use 1",       # 9: use/place crop
    ]

    FORWARD_ACTION = 0
    TURN_ACTIONS = [] # 1, 2
    STRAFE_ACTIONS = [1, 2] # 3, 4
    MOVEMENT_ACTIONS = [0, 1, 2]
    HOTBAR_ACTIONS = {3: "wheat", 4: "carrot", 5: "potato"}
    USE_ACTION = 6



    INSTRUCTION_TEMPLATE = [
        "plant {}",
        "place {} seeds",
        "grow {}",
    ]

    TARGET_ALIASES = {
        "wheat": ["wheat", "wheat seeds"],
        "carrot": ["carrot", "carrots"],
        "potato": ["potato", "potatoes"],
    }

    CROP_BLOCKS = {
        "wheat": ["wheat"],
        "carrot": ["carrots", "carrot"],
        "potato": ["potatoes", "potato"],
    }
    ALL_CROP_BLOCKS = ["wheat", "carrots", "carrot", "potatoes", "potato"]

    WATER_BLOCKS = ["water", "flowing_water"]

    TARGET_SLOT = {
        "wheat": 0,
        "carrot": 1,
        "potato": 2,
    }

    EXPECTED_HOTBAR_ITEM = {
        "wheat": "wheat_seeds",
        "carrot": "carrot",
        "potato": "potato",
    }

    TEXT_MODEL_NAME = None
    TEXT_MODEL_DIM = 3
    TEXT_OUT_DIM = 3

    # Non-text state:
    # agent(2) + current-goal farmland(5) + memory progress(3)
    # + selected slot(3) + hotbar items(3) + obstacles(4)
    # + los_feature(3) + task id(1) = 24
    STATE_DIM = 24

    GRID_MIN = {"x": -8, "y": -1, "z": -8}
    GRID_MAX = {"x": 8, "y": 1, "z": 8}

    # Hidden evaluator value from the fixed XML farm layout.
    # This is used for reward/done, not directly exposed as a remaining-count state.
    EXPECTED_FARMLAND_COUNT = 3

    # Used only to normalize memory-count features. This does not need to equal the true total.
    MEMORY_COUNT_SCALE = 32.0

    USE_INVALID_TARGET_PENALTY = -0.3
    BAD_USE_NO_CONFIRM_PENALTY = -0.2

    STEP_PENALTY = -0.03
    COMPLETE_ALL_REWARD = 100.0
    BECAME_CAN_USE_REWARD = 0.1
    READY_WITH_CORRECT_SLOT_REWARD = 0.0

    # In your environment, once the agent enters water, movement actions are no longer useful.
    END_ON_WATER = True
    WATER_PENALTY = -5.0

    WRONG_PLANT_PENALTY = -10.0
    FAILED_FULL_FARM_PENALTY = -20.0

    WRONG_SLOT_USE_PENALTY = -2.0

    CORRECT_SLOT_SELECT_REWARD = 10.0
    REPEAT_CORRECT_SLOT_PENALTY = -0.1
    INCORRECT_SLOT_SELECT_PENALTY = -0.5
    NO_PROGRESS_PENALTY_HOTBAR = -0.1
    SWITCH_AWAY_FROM_CORRECT_SLOT_PENALTY = -0.5
    HOLDING_CORRECT_SLOT_REWARD = 0.0
    HOLDING_WRONG_SLOT_PENALTY = -0.0

    CAN_USE_CORRECT_SLOT_REWARD = 0.0

    # Lower distance/look shaping because movement is discrete.
    DISTANCE_PROGRESS_SCALE = 1.2
    LOOK_PROGRESS_SCALE = 0.04

    GOOD_FACING_DEGREES = 45.0
    OK_FACING_DEGREES = 60.0
    BAD_FACING_DEGREES = 90.0

    NO_PROGRESS_PENALTY = -0.25
    BAD_FORWARD_PENALTY = -0.05
    UNNECESSARY_TURN_PENALTY = -0.08
    STUCK_EXTRA_PENALTY = -0.0

    MISSED_USE_CHANCE_PENALTY = -0.8

    USE_WHEN_READY_REWARD = 10.0

    TURN_SPAM_PENALTY = -0.12
    TURN_SPAM_THRESHOLD = 4
    BAD_TURN_NO_PROGRESS_PENALTY = -0.08
    BAD_STRAFE_NO_PROGRESS_PENALTY = -0.0

    PROGRESSIVE_PLANT_REWARD_BASE = 3.0
    PROGRESSIVE_PLANT_REWARD_SCALE = 2.0

    MIN_SEEN_FARMLAND_TO_ALLOW_MEMORY_DONE = 3

    USE_WRONG_GOAL_PENALTY = -2.0

    def __init__(self):
        super().__init__(GRID_MIN=self.GRID_MIN, GRID_MAX=self.GRID_MAX)
        self.current_instruction = None
        self.current_target = None
        self.text_model = None
        self.current_instruction_embedding = [0.0] * self.TEXT_MODEL_DIM
        self.embedding_cache = {}

        self.selected_slot = 0
        self.prev_success_count = 0
        self.current_goal_world = None

        self.consecutive_turn_counter = 0

        self._step_obs_cache = {}
        self.select_correct_slot_once = False

        # Agent memory / internal progress tracking.
        # seen/memory_empty are based on observation history.
        # planted/wrong_planted are based on actual use + planting confirmation + targeted world coordinate.
        self.seen_farmland_world = set()
        self.memory_empty_farmland_world = set()
        self.planted_farmland_world = set()
        self.wrong_planted_farmland_world = set()

        self.last_visible_farmland_world = set()
        self.last_grid_anchor = None
        self.pending_farmland_world_counts = {}
        self.pending_anchor_worldset_counts = {}

        self.current_episode = 0
        self.target_bag = []
        self.eval = False

        # debug section code - does not affect agent state or reward
        self.debug_action_counts = {i: 0 for i in range(len(self.CUSTOM_ACTIONS))}
        self.debug_event_counts = {}
        self.debug_last_step_time = 0.0

    def reset(self, instruction=None, eval_mode=False):

        # debug section code - does not affect agent state or reward
        self.debug_action_counts = {i: 0 for i in range(len(self.CUSTOM_ACTIONS))}
        self.debug_event_counts = {}
        self.debug_last_step_time = 0.0
        if hasattr(self, "_printed_local_grid_debug"):
            del self._printed_local_grid_debug

        self.selected_slot = 0
        self.prev_success_count = 0
        self.current_goal_world = None

        self.consecutive_turn_counter = 0

        self._step_obs_cache = {}
        self.select_correct_slot_once = False

        self.seen_farmland_world = set()
        self.memory_empty_farmland_world = set()
        self.planted_farmland_world = set()
        self.wrong_planted_farmland_world = set()

        self.last_visible_farmland_world = set()
        self.last_grid_anchor = None
        self.pending_farmland_world_counts = {}
        self.pending_anchor_worldset_counts = {}

        if instruction is None:
            self.eval = eval_mode
            self.current_target = self.choose_target()
            self.current_episode += 1

            template = random.choice(self.INSTRUCTION_TEMPLATE)
            target_alias = random.choice(self.TARGET_ALIASES[self.current_target])
            self.current_instruction = template.format(target_alias)
        else:
            self.eval = eval_mode
            self.current_instruction = instruction
            self.current_target = self.parse_target(instruction)

        self.current_instruction_embedding = self.encode_target_one_hot(self.current_target)
    
    def choose_target(self):
        if self.eval:
            return random.choice(self.TARGETS)

        if len(self.target_bag) == 0:
            targets = list(self.TARGETS)
            random.shuffle(targets)

            self.target_bag = []
            for t in targets:
                self.target_bag.extend([t] * self.TARGET_EPISODE_BATCH)

        return self.target_bag.pop()
    
    def encode_target_one_hot(self, target):
        if target == "wheat":
            return [1.0, 0.0, 0.0]
        if target == "carrot":
            return [0.0, 1.0, 0.0]
        if target == "potato":
            return [0.0, 0.0, 1.0]
        raise ValueError(f"Unknown target for one-hot encoding: {target}")

    def encode_instruction(self, text):
        if text in self.embedding_cache:
            embedding = self.embedding_cache[text]
        else:
            embedding = self.text_model.encode(text, normalize_embeddings=True).astype("float32")
            self.embedding_cache[text] = embedding
        return embedding.tolist()

    def parse_target(self, instruction):
        text = instruction.lower()
        if "wheat" in text:
            return "wheat"
        if "carrot" in text:
            return "carrot"
        if "potato" in text:
            return "potato"
        raise ValueError(f"Unknown planting instruction: {instruction}")

    def build_state(self, info_dict):
        self.clear_step_cache()
        state_t0 = time.time()

        if not info_dict:
            info_dict = {}

        # Update agent memory from only the currently observed local grid.
        # This does not give the agent hidden global world state.
        # self.update_farmland_memory(info_dict) ---temperarily comment out the memory update in build state since the shape_reward already have it.

        yaw = float(info_dict.get("Yaw", 0))
        pitch = float(info_dict.get("Pitch", 0))
        agent_features = [yaw / 180.0, pitch / 90.0]

        goal = self.get_or_update_goal_farmland(info_dict)
        farmland_features = self.goal_to_features(goal)

        goal_world = goal["world"] if goal is not None else self.current_goal_world
        los_features = [
            1.0 if self.los_matches_goal(info_dict, goal_world) else 0.0,
            1.0 if self.los_matches_wrong_goal(info_dict, goal_world) else 0.0,
            1.0 if self.can_use_available_farmland_now(info_dict) > 0.0 else 0.0,
        ]

        scale = max(1.0, self.MEMORY_COUNT_SCALE)
        farm_memory_features = [
            min(len(self.memory_empty_farmland_world) / scale, 1.0),
            min(len(self.planted_farmland_world) / scale, 1.0),
            min(len(self.wrong_planted_farmland_world) / scale, 1.0),
        ]

        selected_features = [
            1.0 if self.selected_slot == 0 else 0.0,
            1.0 if self.selected_slot == 1 else 0.0,
            1.0 if self.selected_slot == 2 else 0.0,
        ]

        hotbar_items = self.get_hotbar_items(info_dict)
        hotbar_item_features = [
            1.0 if hotbar_items[0] == "wheat_seeds" else 0.0,
            1.0 if hotbar_items[1] == "carrot" else 0.0,
            1.0 if hotbar_items[2] == "potato" else 0.0,
        ]

        fx, fz = self.yaw_to_direction(yaw)
        lx, lz = fz, -fx
        rx, rz = -fz, fx

        front = float(self.is_block(self.get_board_block(info_dict, fx, 0, fz)))
        back = float(self.is_block(self.get_board_block(info_dict, -fx, 0, -fz)))
        left = float(self.is_block(self.get_board_block(info_dict, lx, 0, lz)))
        right = float(self.is_block(self.get_board_block(info_dict, rx, 0, rz)))

        obstacle_features = [front, back, left, right]
        
        state = (
            agent_features
            + farmland_features
            + farm_memory_features
            + selected_features
            + hotbar_item_features
            + obstacle_features
            + los_features
            + [float(self.TASK_ID)]
            + self.current_instruction_embedding
        )

        state_dt = time.time() - state_t0
        if self.DEBUG_PRINT and state_dt > self.SLOW_STEP_WARN_SECONDS:
            self.debug_event("slow_build_state")
            self.debug_print(
                f"[SLOW_BUILD_STATE] time={state_dt:.4f}s "
                f"seen={len(self.seen_farmland_world)} "
                f"memory_empty={len(self.memory_empty_farmland_world)} "
                f"goal={self.current_goal_world}"
            )

        return state

    def shape_reward(self, raw_reward, prev_info, curr_info, action, step):
        self.clear_step_cache()
        debug_t0 = time.time()

        if 0 <= action < len(self.CUSTOM_ACTIONS):
            self.debug_action_counts[action] += 1
        else:
            self.debug_event("unknown_action")

        reward = float(raw_reward) + self.STEP_PENALTY
        done = False
        metrics = {}

        has_prev_info = prev_info is not None
        target_slot = self.TARGET_SLOT[self.current_target]

        # 1. Hotbar selection reward / penalty
        old_slot = self.selected_slot

        if action in self.HOTBAR_ACTIONS:
            selected_target = self.HOTBAR_ACTIONS[action]
            new_slot = self.TARGET_SLOT[selected_target]
            target_slot = self.TARGET_SLOT[self.current_target]

            self.selected_slot = new_slot
            if new_slot == target_slot:
                if not self.select_correct_slot_once:
                    reward += self.CORRECT_SLOT_SELECT_REWARD
                    self.select_correct_slot_once = True
                    self.debug_event("first_correct_slot_select")
                else:
                    reward += self.REPEAT_CORRECT_SLOT_PENALTY
                    self.debug_event("repeat_correct_slot_select")
            else:
                reward += self.INCORRECT_SLOT_SELECT_PENALTY
                self.debug_event("wrong_slot_select")
        
        # 2. Water handling
        if self.is_agent_in_water(curr_info):
            self.debug_event("water")
            reward += self.WATER_PENALTY
            metrics["in_water"] = True
            metrics["task_success"] = False

            if self.END_ON_WATER:
                self.debug_event("done_water")

                debug_dt = time.time() - debug_t0
                self.debug_last_step_time = debug_dt

                self.debug_print(
                    f"[DEBUG_WATER_END] step={step} "
                    f"last_shape_time={debug_dt:.4f}s "
                    f"reward={reward:.2f} "
                    f"target={self.current_target} "
                    f"slot={self.selected_slot}/{target_slot} "
                    f"planted={len(self.planted_farmland_world)} "
                    f"wrong={len(self.wrong_planted_farmland_world)} "
                    f"seen={len(self.seen_farmland_world)} "
                    f"memory_empty={len(self.memory_empty_farmland_world)} "
                    f"events={self.debug_event_counts}"
                )
                done = True
                return reward, done, metrics

        # 3. Fixed-goal movement shaping
        goal_world_for_step = self.current_goal_world

        if goal_world_for_step is None:
            initial_goal = self.get_or_update_goal_farmland(prev_info if has_prev_info else curr_info)
            if initial_goal is not None:
                goal_world_for_step = initial_goal["world"]
                self.current_goal_world = goal_world_for_step

        prev_goal = self.get_goal_relative(prev_info, goal_world_for_step) if has_prev_info else None
        curr_goal = self.get_goal_relative(curr_info, goal_world_for_step)

        prev_can_use = self.can_use_available_farmland_now(prev_info) if has_prev_info else 0.0
        curr_can_use = self.can_use_available_farmland_now(curr_info)

        prev_target_goal_usable = (has_prev_info and self.is_target_goal_usable_now(prev_info, goal_world_for_step))
        curr_target_goal_usable = self.is_target_goal_usable_now(curr_info, goal_world_for_step)


        # Debug only: any usable farmland, not necessarily target goal.
        if curr_can_use > 0.0:
            self.debug_event("can_use_seen_any_farmland")

        if curr_can_use > 0.0 and self.selected_slot == target_slot:
            self.debug_event("can_use_correct_slot_seen_any_farmland")

        # Reward: only target goal became usable.
        if (not prev_target_goal_usable) and curr_target_goal_usable:
            reward += self.BECAME_CAN_USE_REWARD
            self.debug_event("became_target_goal_can_use")

        # Penalty: only target goal was usable but agent did not use.
        if (prev_target_goal_usable and action != self.USE_ACTION and old_slot == target_slot):
            reward += self.MISSED_USE_CHANCE_PENALTY
            self.debug_event("missed_use_chance_target_goal")

        # Ready event/reward: only target goal ready.
        if curr_target_goal_usable and self.selected_slot == target_slot:
            reward += self.READY_WITH_CORRECT_SLOT_REWARD
            self.debug_event("ready_correct_slot_target_goal")

        if prev_goal is not None and curr_goal is not None:
            prev_los_dist = self.get_los_distance_to_goal(prev_info, goal_world_for_step) if has_prev_info else None
            curr_los_dist = self.get_los_distance_to_goal(curr_info, goal_world_for_step)

            if prev_los_dist is not None and curr_los_dist is not None:
                distance_progress = prev_los_dist - curr_los_dist
            else:
                distance_progress = prev_goal["distance"] - curr_goal["distance"]

            if action in self.STRAFE_ACTIONS and distance_progress <= 0.01:
                reward += self.BAD_STRAFE_NO_PROGRESS_PENALTY
                self.debug_event("bad_strafe_no_progress")

            look_progress = prev_goal["yaw_error"] - curr_goal["yaw_error"]

            # Penalize turning that does not improve aim.
            if action in self.TURN_ACTIONS:
                self.consecutive_turn_counter += 1
                if prev_goal["yaw_error"] <= self.GOOD_FACING_DEGREES:
                    reward += self.UNNECESSARY_TURN_PENALTY
                    self.debug_event("turn_when_already_facing")
                elif look_progress <= 0.5:
                    reward += self.BAD_TURN_NO_PROGRESS_PENALTY
                    self.debug_event("bad_turn_no_look_progress")
                else:
                    reward += self.LOOK_PROGRESS_SCALE * self.clamp(look_progress, 0.0, 10.0)

                if self.consecutive_turn_counter >= self.TURN_SPAM_THRESHOLD:
                    reward += self.TURN_SPAM_PENALTY
                    self.debug_event("turn_spam")
            else:
                self.consecutive_turn_counter = 0

            # Movement progress, block-break style.
            if action in self.MOVEMENT_ACTIONS:
                if distance_progress > 0:
                    reward += self.DISTANCE_PROGRESS_SCALE * self.clamp(distance_progress, -1.0, 1.0)
                    self.debug_event("move_closer_to_goal")
                else:
                    reward += self.NO_PROGRESS_PENALTY
                    self.debug_event("move_no_goal_progress")
            # Disabled in no-turn curriculum.
            # if action == self.FORWARD_ACTION and curr_goal["yaw_error"] > self.BAD_FACING_DEGREES:
            #     reward += self.BAD_FORWARD_PENALTY


        # 4. Planting reward
        # Do NOT use curr_stats - prev_stats here because those are local observations only.
        if action == self.USE_ACTION:
            self.debug_event("use_attempt")

            if not has_prev_info:
                metrics["missing_prev_info_on_use"] = True
            else:
                can_use_before = prev_can_use
                target_world = self.get_targeted_farmland_world(prev_info)
                target_is_goal = target_world == goal_world_for_step

                prev_selected_count = self.get_inventory_count_by_slot(prev_info, self.selected_slot)
                curr_selected_count = self.get_inventory_count_by_slot(curr_info, self.selected_slot)

                inventory_decreased = (
                    prev_selected_count is not None
                    and curr_selected_count is not None
                    and curr_selected_count == prev_selected_count - 1
                )

                crop_after = None
                crop_appeared = False

                if target_world is not None:
                    crop_after = self.get_block_above_world(curr_info, target_world)
                    crop_appeared = self.is_any_crop_block(crop_after)

                ready_before = (
                    can_use_before > 0.0
                    and target_world is not None
                )

                actual_planted = (
                    can_use_before > 0.0
                    and target_world is not None
                    and (inventory_decreased or crop_appeared)
                )

                if not ready_before:
                    # If the agent tried to use when it not a available farmland, give a penalty. This encourages the agent to learn the correct timing of use, rather than just spamming use and hoping for a lucky hit.
                    reward += self.USE_INVALID_TARGET_PENALTY
                    if can_use_before <= 0.0:
                        self.debug_event("use_fail_not_usable")
                    elif target_world is None:
                        self.debug_event("use_fail_no_target_world")
                    else:
                        self.debug_event("use_fail_wrong_slot")

                elif not actual_planted:
                    reward += self.BAD_USE_NO_CONFIRM_PENALTY
                    self.debug_event("use_fail_no_plant_confirmed")

                else:
                    self.debug_event("successful_plant_use")

                    if not target_is_goal:
                    # penalty for plant on a non-goal farmland, to encourage the agent to prioritize the current goal rather than just filling any farmland. This is important for learning the correct order of planting when there are multiple farmlands.
                        reward += self.USE_WRONG_GOAL_PENALTY
                        self.debug_event("use_wrong_goal_penalty")

                    if crop_appeared:
                        planted_target_crop = self.is_target_crop_block(crop_after, self.current_target)
                    else:
                        planted_target_crop = self.selected_slot == target_slot

                    was_already_planted = target_world in self.planted_farmland_world
                    was_already_wrong = target_world in self.wrong_planted_farmland_world

                    if planted_target_crop:
                        # Memory: this farmland is now correctly filled, even if it was not the current goal.
                        self.planted_farmland_world.add(target_world)
                        self.wrong_planted_farmland_world.discard(target_world)
                        self.memory_empty_farmland_world.discard(target_world)

                        if target_is_goal:
                            reward += self.USE_WHEN_READY_REWARD
                            self.debug_event("use_when_ready_target_goal")

                            if not was_already_planted:
                                self.debug_event("correct_target_goal_plant")
                                self.prev_success_count = len(self.planted_farmland_world)

                                progressive_reward = (
                                    self.PROGRESSIVE_PLANT_REWARD_BASE
                                    + self.PROGRESSIVE_PLANT_REWARD_SCALE * self.prev_success_count
                                )
                                reward += progressive_reward
                                self.debug_event(f"progressive_correct_plant_{self.prev_success_count}")

                                metrics["new_plants"] = 1
                                self.current_goal_world = None
                            else:
                                self.debug_event("duplicate_correct_target_goal_plant")

                        else:
                            # Correct crop, but not the current goal.
                            # Do not mark as wrong. Just update memory.
                            self.debug_event("off_goal_correct_crop_plant")
                    else:
                        # Wrong crop, regardless of whether it was target goal or off-goal.
                        self.debug_event("wrong_crop_plant")

                        self.wrong_planted_farmland_world.add(target_world)
                        self.planted_farmland_world.discard(target_world)
                        self.memory_empty_farmland_world.discard(target_world)

                        if not was_already_wrong:
                            reward += self.WRONG_PLANT_PENALTY
                            metrics["wrong_plants"] = 1
                        
                        reward += self.WRONG_SLOT_USE_PENALTY

                        if target_is_goal:
                            self.current_goal_world = None
        # Keep memory updated from the latest visible local observation.
        before_correct = len(self.planted_farmland_world)
        before_wrong = len(self.wrong_planted_farmland_world)

        self.update_farmland_memory(curr_info)

        after_correct = len(self.planted_farmland_world)
        after_wrong = len(self.wrong_planted_farmland_world)

        if after_correct > before_correct:
            self.debug_event("memory_detected_new_correct_after_update")

        if after_wrong > before_wrong:
            self.debug_event("memory_detected_new_wrong_after_update")

        # 5. Completion check based on memory-empty condition.
        filled_count = self.get_filled_farmland_count()
        correct_count = len(self.planted_farmland_world)
        wrong_count = len(self.wrong_planted_farmland_world)

        seen_count = len(self.seen_farmland_world)
        empty_count = len(self.memory_empty_farmland_world)

        memory_done = (
            seen_count >= self.MIN_SEEN_FARMLAND_TO_ALLOW_MEMORY_DONE
            and empty_count == 0
            and filled_count >= seen_count
        )
        if memory_done:
            done = True
            if wrong_count == 0 and correct_count >= seen_count:
                reward += self.COMPLETE_ALL_REWARD
                metrics["task_success"] = True
                print("done_memory_all_correct")
                self.debug_event("done_memory_all_correct")
            else:
                # reward += self.FAILED_FULL_FARM_PENALTY
                metrics["task_success"] = False
                metrics["farm_filled_but_wrong"] = True
                print("done_memory_has_wrong")
                self.debug_event("done_memory_has_wrong")

        # 6. Metrics. get_farm_stats is local observation only.
        if (
            self.DEBUG_PRINT
            and self.DEBUG_PRINT_EVERY_N_STEPS > 0
            and step % self.DEBUG_PRINT_EVERY_N_STEPS == 0
        ):
            curr_stats = self.get_farm_stats(curr_info, self.current_target)
        else:
            curr_stats = {
                "empty": -1,
                "target_planted": -1,
                "wrong_planted": -1,
                "total": -1,
            }

        metrics["target"] = self.current_target
        metrics["selected_slot"] = self.selected_slot
        metrics["curr_can_use"] = curr_can_use

        metrics["success_count"] = correct_count
        metrics["target_planted"] = correct_count
        metrics["wrong_planted"] = wrong_count
        metrics["filled_farmland"] = filled_count
        metrics["total_farmland"] = seen_count

        metrics["seen_farmland"] = seen_count
        metrics["memory_empty_farmland"] = empty_count
        metrics["memory_done"] = memory_done

        metrics["observed_empty_farmland"] = curr_stats["empty"]
        metrics["observed_target_planted"] = curr_stats["target_planted"]
        metrics["observed_wrong_planted"] = curr_stats["wrong_planted"]
        metrics["observed_total_farmland"] = curr_stats["total"]


        debug_dt = time.time() - debug_t0
        self.debug_last_step_time = debug_dt

        if debug_dt > self.SLOW_STEP_WARN_SECONDS:
            self.debug_event("slow_shape_reward")

        if (
            self.DEBUG_PRINT
            and self.DEBUG_PRINT_EVERY_N_STEPS > 0
            and step % self.DEBUG_PRINT_EVERY_N_STEPS == 0
        ):
            top_actions = sorted(
                self.debug_action_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            top_actions_str = ", ".join(
                f"{a}:{self.CUSTOM_ACTIONS[a]}={c}"
                for a, c in top_actions
            )

            events_str = ", ".join(
                f"{k}={v}"
                for k, v in sorted(self.debug_event_counts.items())
            )

            self.debug_print(
                f"[DEBUG_SUMMARY] step={step} "
                f"last_shape_time={debug_dt:.4f}s "
                f"reward={reward:.2f} "
                f"target={self.current_target} "
                f"slot={self.selected_slot}/{target_slot} "
                f"goal={self.current_goal_world} "
                f"planted={correct_count} "
                f"wrong={wrong_count} "
                f"filled={filled_count} "
                f"seen={seen_count} "
                f"memory_empty={empty_count} "
                f"memory_done={memory_done} "
                f"top_actions=[{top_actions_str}] "
                f"events=[{events_str}]"
            )
        return reward, done, metrics

    def update_farmland_memory(self, info_dict):
        """
        Update memory from the currently observed local grid only.
        This is agent memory, not hidden global world state.
        """
        if not info_dict:
            return 0

        new_seen = 0

        for c in self.get_farmland_candidates(info_dict, confirm_new_cluster=True):
            world = c.get("world")
            above = c.get("above")

            if world is None:
                continue

            self.pending_farmland_world_counts[world] = (self.pending_farmland_world_counts.get(world, 0) + 1)

            if (world not in self.seen_farmland_world and self.pending_farmland_world_counts[world] >= self.FARMLAND_CONFIRM_STEPS):
                self.seen_farmland_world.add(world)
                new_seen += 1
            
            if world not in self.seen_farmland_world:
                continue

            if self.is_empty_above_farmland(above):
                if world not in self.planted_farmland_world and world not in self.wrong_planted_farmland_world:
                    self.memory_empty_farmland_world.add(world)
            elif self.is_any_crop_block(above):
                # The block is no longer empty.
                # Update memory counts from visible world state.
                self.memory_empty_farmland_world.discard(world)

                if self.is_target_crop_block(above, self.current_target):
                    self.planted_farmland_world.add(world)
                    self.wrong_planted_farmland_world.discard(world)
                else:
                    self.wrong_planted_farmland_world.add(world)
                    self.planted_farmland_world.discard(world)

        # Debug-only sanity check. This does not change reward, state, memory, or done logic.
        # The existing DEBUG_SUMMARY format is intentionally unchanged.
        if self.DEBUG_PRINT and len(self.seen_farmland_world) > self.EXPECTED_FARMLAND_COUNT:
            self.debug_print(
                f"[BUG_MEMORY_OVER_EXPECTED] seen={len(self.seen_farmland_world)} "
                f"expected={self.EXPECTED_FARMLAND_COUNT} "
                f"seen_farmland_world={sorted(self.seen_farmland_world)}"
            )

        return new_seen

    def get_filled_farmland_count(self):
        """Count unique farmland blocks filled by either correct or wrong crops."""
        return len(self.planted_farmland_world | self.wrong_planted_farmland_world)

    def get_targeted_farmland_world(self, info_dict):
        """Return world coordinate of the farmland currently targeted by LineOfSight."""
        if not info_dict:
            return None

        los = info_dict.get("LineOfSight", {})
        if not los:
            return None

        if los.get("hitType", "") != "block":
            return None
        if los.get("type", "") != "farmland":
            return None
        if not bool(los.get("inRange", False)):
            return None

        los_world = self.get_los_block_world(info_dict)
        if los_world is None:
            return None
        
        x, y, z = los_world
        return (x, y, z)

    def get_inventory_count_by_slot(self, info_dict, slot):
        """Return item count from a specific hotbar slot."""
        if not info_dict:
            return None

        size_key = f"Hotbar_{slot}_size"
        size = info_dict.get(size_key, None)

        if size is None:
            return None

        try:
            return int(size)
        except (TypeError, ValueError):
            return None

    def get_or_update_goal_farmland(self, info_dict):
        if not info_dict:
            return None

        current = self.get_goal_relative(info_dict, self.current_goal_world)
        if current is not None and current["world"] in self.memory_empty_farmland_world and self.is_empty_above_farmland(current.get("above")):
            return current

        nearest_memory = self.find_nearest_memory_empty_farmland(info_dict)
        if nearest_memory is not None:
            self.current_goal_world = nearest_memory["world"]
            return nearest_memory

        nearest_visible = self.find_nearest_empty_farmland(info_dict)
        if nearest_visible is None:
            self.current_goal_world = None
            return None
        
        self.current_goal_world = nearest_visible["world"]
        return nearest_visible

    def get_goal_relative(self, info_dict, goal_world):
        if not info_dict or goal_world is None:
            return None

        agent_x, agent_y, agent_z = self.get_agent_block_pos(info_dict)
        wx, wy, wz = goal_world
        dx = wx - agent_x
        dy = wy - agent_y
        dz = wz - agent_z

        if not self.offset_in_grid(dx, dy, dz):
            return None

        block = self.get_board_block(info_dict, dx, dy, dz)
        above = self.get_board_block(info_dict, dx, dy + 1, dz)
        if block != "farmland":
            return None

        yaw = float(info_dict.get("Yaw", 0.0))
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        yaw_error = self.yaw_error_to_target(yaw, dx, dz)

        return {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "distance": distance,
            "yaw_error": yaw_error,
            "world": goal_world,
            "above": above,
        }

    def goal_to_features(self, goal):
        x_scale = max(abs(self.GRID_MIN["x"]), abs(self.GRID_MAX["x"]))
        y_scale = max(abs(self.GRID_MIN["y"]), abs(self.GRID_MAX["y"]))
        z_scale = max(abs(self.GRID_MIN["z"]), abs(self.GRID_MAX["z"]))
        distance_scale = math.sqrt(x_scale ** 2 + y_scale ** 2 + z_scale ** 2)

        if goal is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        return [
            goal["dx"] / x_scale,
            goal["dy"] / y_scale,
            goal["dz"] / z_scale,
            goal["distance"] / distance_scale,
            goal["yaw_error"] / 180.0,
        ]

    def find_nearest_empty_farmland(self, info_dict):
        candidates = self.get_farmland_candidates(info_dict)
        empty_candidates = [c for c in candidates if (c["world"] in self.seen_farmland_world and self.is_empty_above_farmland(c["above"]))]
        if not empty_candidates:
            return None

        return min(empty_candidates, key=lambda c: c["distance"])

    def get_farmland_candidates(self, info_dict, confirm_new_cluster=False):
        if not info_dict:
            return []

        cache = self.get_step_cache(info_dict)
        cache_key = ("farmland_candidates", confirm_new_cluster)

        if cache is not None and cache_key in cache:
            return cache[cache_key]

        board = info_dict.get("board", [])
        if not board:
            return []

        x_count = self.GRID_MAX["x"] - self.GRID_MIN["x"] + 1
        y_count = self.GRID_MAX["y"] - self.GRID_MIN["y"] + 1
        z_count = self.GRID_MAX["z"] - self.GRID_MIN["z"] + 1
        expected_len = x_count * y_count * z_count
        if len(board) != expected_len:
            return []

        local_offsets = self.get_local_farmland_offsets(info_dict)

        if self.DEBUG_PRINT and not hasattr(self, "_printed_local_grid_debug"):
            self._printed_local_grid_debug = True
            self.debug_print(
                f"[DEBUG_LOCAL_FARMLAND_OFFSETS] count={len(local_offsets)} "
                f"offsets={local_offsets}"
            )

        if not local_offsets:
            return []

        anchor = self.get_observation_anchor(
            info_dict,
            allow_fallback=False,
            confirm_new_cluster=confirm_new_cluster,
        )

        if anchor is None:
            self.debug_event("skip_memory_uncertain_anchor")
            return []

        self.last_grid_anchor = anchor

        ax, ay, az = anchor
        yaw = float(info_dict.get("Yaw", 0.0))
        candidates = []

        for dx, dy, dz in local_offsets:
            above = self.get_board_block(info_dict, dx, dy + 1, dz)
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            yaw_error = self.yaw_error_to_target(yaw, dx, dz)
            world = (ax + dx, ay + dy, az + dz)

            candidates.append({
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "distance": distance,
                "yaw_error": yaw_error,
                "world": world,
                "above": above,
            })

        if cache is not None:
            cache[cache_key] = candidates
        return candidates

    def get_farm_stats(self, info_dict, target):
        """Local observation stats only. Do not use these stats for full task completion."""
        stats = {
            "total": 0,
            "empty": 0,
            "target_planted": 0,
            "wrong_planted": 0,
            "other_occupied": 0,
        }

        for c in self.get_farmland_candidates(info_dict):
            stats["total"] += 1
            above = c.get("above", None)

            if self.is_empty_above_farmland(above):
                stats["empty"] += 1
            elif self.is_target_crop_block(above, target):
                stats["target_planted"] += 1
            elif self.is_any_crop_block(above):
                stats["wrong_planted"] += 1
            else:
                stats["other_occupied"] += 1

        return stats

    def is_any_crop_block(self, block):
        return block in self.ALL_CROP_BLOCKS

    def is_empty_above_farmland(self, block):
        # In your actual read, empty space may be returned as None instead of "air".
        return block is None or block == "air"

    def get_hotbar_items(self, info_dict):
        if not info_dict:
            return {
                0: "air",
                1: "air",
                2: "air",
            }
        return {
            0: info_dict.get("Hotbar_0_item", "air"),
            1: info_dict.get("Hotbar_1_item", "air"),
            2: info_dict.get("Hotbar_2_item", "air"),
        }

    def can_use_available_farmland_now(self, info_dict):
        if not info_dict:
            return 0.0

        los = info_dict.get("LineOfSight", {})
        if not los:
            return 0.0

        hit_type = los.get("hitType", "")
        block_type = los.get("type", "")
        in_range = bool(los.get("inRange", False))

        if not (hit_type == "block" and block_type == "farmland" and in_range):
            return 0.0

        if not self.is_los_hitting_top_of_block(info_dict, los):
            return 0.0

        block_above = self.get_block_above_targeted_farmland(info_dict, los)
        if self.is_empty_above_farmland(block_above):
            return 1.0

        return 0.0

    def get_block_above_targeted_farmland(self, info_dict, los=None):
        """
        Return the block above the farmland currently targeted by LineOfSight.
        This assumes ObservationFromGrid uses absoluteCoords="false",
        so get_board_block(info_dict, dx, dy, dz) expects local offsets.
        """
        if not info_dict:
            return None

        if los is None:
            los = info_dict.get("LineOfSight", {})
        if not los:
            return None

        if los.get("hitType", "") != "block":
            return None
        if los.get("type", "") != "farmland":
            return None
        if not bool(los.get("inRange", False)):
            return None
        if "x" not in los or "y" not in los or "z" not in los:
            return None

        los_world = self.get_los_block_world(info_dict)
        if los_world is None:
            return None

        target_x, target_y, target_z = los_world

        agent_x, agent_y, agent_z = self.get_agent_block_pos(info_dict)

        # Convert world block position to local ObservationFromGrid offset.
        dx = target_x - agent_x
        dy = target_y - agent_y
        dz = target_z - agent_z

        if not self.offset_in_grid(dx, dy + 1, dz):
            return None

        return self.get_board_block(info_dict, dx, dy + 1, dz)

    def get_target_inventory_count(self, info_dict, target):
        """
        Return the inventory count for the target crop item from ObservationFromHotBar.
        Kept for compatibility/debug. The updated reward uses get_inventory_count_by_slot().
        """
        if not info_dict:
            return None

        slot = self.TARGET_SLOT[target]

        size_key = f"Hotbar_{slot}_size"
        item_key = f"Hotbar_{slot}_item"

        item = info_dict.get(item_key, "air")
        size = info_dict.get(size_key, None)

        expected_item = self.EXPECTED_HOTBAR_ITEM[target]
        if item != expected_item:
            print(
                f"Inventory warning: target={target}, expected_item={expected_item}, "
                f"but slot {slot} has item={item}"
            )

        if size is None:
            return None

        try:
            return int(size)
        except (TypeError, ValueError):
            return None

    def is_agent_in_water(self, info_dict):
        if not info_dict:
            return False

        current_block = self.get_board_block(info_dict, 0, 0, 0)
        below_block = self.get_board_block(info_dict, 0, -1, 0)
        return current_block in self.WATER_BLOCKS or below_block in self.WATER_BLOCKS

    def get_agent_block_pos(self, info_dict):
        anchor = self.get_observation_anchor(info_dict, allow_fallback=True)
        if anchor is None:
            return 0, 0, 0
        return anchor

    def offset_in_grid(self, dx, dy, dz):
        return (
            self.GRID_MIN["x"] <= dx <= self.GRID_MAX["x"]
            and self.GRID_MIN["y"] <= dy <= self.GRID_MAX["y"]
            and self.GRID_MIN["z"] <= dz <= self.GRID_MAX["z"]
        )

    def find_nearest_memory_empty_farmland(self, info_dict):
        """
        Find the nearest remembered empty farmland that is currently inside
        the observation grid. This makes the agent use its memory instead of
        only choosing from newly scanned farmland.
        """
        if not info_dict:
            return None

        best = None

        for world in list(self.memory_empty_farmland_world):
            rel = self.get_goal_relative(info_dict, world)
            if rel is None:
                continue

            if best is None or rel["distance"] < best["distance"]:
                best = rel

        return best

    def get_block_above_world(self, info_dict, world):
        """Return the observed block above a world farmland coordinate, if visible."""
        if not info_dict or world is None:
            return None

        agent_x, agent_y, agent_z = self.get_agent_block_pos(info_dict)
        wx, wy, wz = world

        dx = wx - agent_x
        dy = wy - agent_y
        dz = wz - agent_z

        above_dy = dy + 1

        if not self.offset_in_grid(dx, above_dy, dz):
            return None

        return self.get_board_block(info_dict, dx, above_dy, dz)


    def is_target_crop_block(self, block, target):
        """Return True if block is the crop block for the current target."""
        if block is None:
            return False
        return block in self.CROP_BLOCKS.get(target, [])

    def get_los_block_world(self, info_dict):
        """Return LineOfSight block world coordinate if available."""
        if not info_dict:
            return None

        los = info_dict.get("LineOfSight", {})
        if not los:
            return None

        if los.get("hitType", "") != "block":
            return None

        if "x" not in los or "y" not in los or "z" not in los:
            return None

        # LOS block coordinates should be treated as block coordinates.
        return (
            int(math.floor(float(los["x"]) + 1e-6)),
            int(math.floor(float(los["y"]) + 1e-6)),
            int(math.floor(float(los["z"]) + 1e-6)),
        )


    def get_local_farmland_offsets(self, info_dict):
        """
        Return local grid offsets where the board says the block is farmland.
        This does not convert to world coordinates.
        """
        if not info_dict:
            return []
        
        cache = self.get_step_cache(info_dict)
        if cache is not None and "local_farmland_offsets" in cache:
            return cache["local_farmland_offsets"]

        board = info_dict.get("board", [])
        if not board:
            return []

        x_count = self.GRID_MAX["x"] - self.GRID_MIN["x"] + 1
        y_count = self.GRID_MAX["y"] - self.GRID_MIN["y"] + 1
        z_count = self.GRID_MAX["z"] - self.GRID_MIN["z"] + 1
        expected_len = x_count * y_count * z_count

        if len(board) != expected_len:
            return []

        offsets = []

        for dy in range(self.GRID_MIN["y"], self.GRID_MAX["y"]):
            for dz in range(self.GRID_MIN["z"], self.GRID_MAX["z"] + 1):
                for dx in range(self.GRID_MIN["x"], self.GRID_MAX["x"] + 1):
                    block = self.get_board_block(info_dict, dx, dy, dz)
                    if block == "farmland":
                        offsets.append((dx, dy, dz))
        
        if cache is not None:
            cache["local_farmland_offsets"] = offsets

        return offsets

    def get_anchor_candidates(self, info_dict):
        x_pos = float(info_dict.get("XPos", 0.0))
        y_pos = float(info_dict.get("YPos", 0.0))
        z_pos = float(info_dict.get("ZPos", 0.0))

        y_anchor = math.floor(y_pos + 1e-6)

        x_floor = math.floor(x_pos + 1e-6)
        z_floor = math.floor(z_pos + 1e-6)
        x_round = int(round(x_pos))
        z_round = int(round(z_pos))

        raw = [
            (x_floor, y_anchor, z_floor),
            (x_round, y_anchor, z_round),
            (x_floor, y_anchor, z_round),
            (x_round, y_anchor, z_floor),
        ]

        anchors = []
        for a in raw:
            if a not in anchors:
                anchors.append(a)

        return anchors

    def choose_consistent_anchor(self, info_dict, local_offsets, confirm_new_cluster=False):
        """
        Choose a stable local-grid-to-world anchor using only observation
        consistency. This does not use fixed farmland count or layout.

        When memory already exists, a candidate anchor must overlap enough with
        known/last visible farmland. Otherwise we skip memory updates for that
        frame instead of adding likely fake shifted coordinates.

        LineOfSight is used only as an observation-based calibration signal:
        if the LOS world block maps back to a local grid cell, that local cell
        should contain the same block type.
        """
        anchors = self.get_anchor_candidates(info_dict)

        known_world = (
            self.seen_farmland_world
            | self.planted_farmland_world
            | self.wrong_planted_farmland_world
            | self.memory_empty_farmland_world
        )

        los_world = self.get_los_block_world(info_dict)
        los_type = None
        los = info_dict.get("LineOfSight", {})
        if los and los.get("hitType", "") == "block":
            los_type = los.get("type", None)

        best_anchor = None
        best_worlds = set()
        best_score = -10**9
        best_overlap = 0
        best_visible_known_match = 0
        best_visible_known_mismatch = 0

        for ax, ay, az in anchors:
            worlds = {
                (ax + dx, ay + dy, az + dz)
                for dx, dy, dz in local_offsets
            }

            overlap_known = len(worlds & known_world)
            overlap_last = len(worlds & self.last_visible_farmland_world)

            # New coordinates are allowed, but they should not dominate
            # when we already have an overlapping map.
            new_count = len(worlds - known_world)

            # Inverse consistency check: if a remembered farmland world coordinate
            # is inside the current local grid under this anchor, the local grid
            # should still show farmland at that base coordinate. This uses only
            # observation and memory, not fixed layout/count.
            visible_known_match = 0
            visible_known_mismatch = 0
            for wx, wy, wz in known_world:
                kdx = wx - ax
                kdy = wy - ay
                kdz = wz - az
                if self.offset_in_grid(kdx, kdy, kdz):
                    observed_known = self.get_board_block(info_dict, kdx, kdy, kdz)
                    if observed_known == "farmland":
                        visible_known_match += 1
                    else:
                        visible_known_mismatch += 1

            score = 0.0
            score += 5.0 * overlap_known
            score += 3.0 * overlap_last
            score += 4.0 * visible_known_match
            score -= 20.0 * visible_known_mismatch
            score -= 1.0 * new_count

            # Strong observation-based calibration from LineOfSight when it
            # can be mapped into the local grid.
            if los_world is not None and los_type is not None:
                lx, ly, lz = los_world
                ldx = lx - ax
                ldy = ly - ay
                ldz = lz - az

                if self.offset_in_grid(ldx, ldy, ldz):
                    observed = self.get_board_block(info_dict, ldx, ldy, ldz)
                    if observed == los_type:
                        score += 100.0
                    else:
                        score -= 100.0

            # Prefer continuity with previous anchor, but do not hard-lock it.
            if self.last_grid_anchor is not None:
                pax, pay, paz = self.last_grid_anchor
                score -= 0.25 * (abs(ax - pax) + abs(ay - pay) + abs(az - paz))

            if score > best_score:
                best_score = score
                best_anchor = (ax, ay, az)
                best_worlds = worlds
                best_overlap = max(overlap_known, overlap_last)
                best_visible_known_match = visible_known_match
                best_visible_known_mismatch = visible_known_mismatch

        if best_anchor is None:
            return None, set()

        # If this anchor contradicts known farmland that is currently visible,
        # it is likely a shifted mapping and should not update memory.
        if best_visible_known_mismatch > 0:
            self.debug_event("skip_memory_anchor_known_mismatch")
            return None, set()

        # Confidence rule:
        # - If memory is empty, accept the first mapping.
        # - If memory exists, require enough overlap with previous/known memory.
        #   This prevents shifted versions of the same observed farmland pattern
        #   from entering memory as fake new world positions.
        if len(known_world) > 0:
            required_overlap = max(
                1,
                int(math.ceil(self.ANCHOR_MIN_OVERLAP_RATIO * len(local_offsets)))
            )

            if best_overlap < required_overlap:
                # Low-overlap mappings are not accepted immediately. They can be
                # real expansion into a larger/new farm, or they can be a shifted
                # fake mapping. Require the exact observed world set to be stable
                # across memory-update calls before accepting it.
                if not confirm_new_cluster:
                    if best_overlap > 0:
                        self.debug_event("skip_memory_low_anchor_overlap")
                    return None, set()

                worldset_key = frozenset(best_worlds)
                self.pending_anchor_worldset_counts[worldset_key] = (
                    self.pending_anchor_worldset_counts.get(worldset_key, 0) + 1
                )

                if self.pending_anchor_worldset_counts[worldset_key] < self.FARMLAND_CONFIRM_STEPS:
                    if best_overlap > 0:
                        self.debug_event("skip_memory_low_overlap_unconfirmed")
                    else:
                        self.debug_event("skip_memory_new_cluster_unconfirmed")
                    return None, set()

                if best_overlap > 0:
                    self.debug_event("accept_memory_low_overlap_stable")
                else:
                    self.debug_event("accept_memory_new_cluster")

        return best_anchor, best_worlds


    def get_observation_anchor(self, info_dict, allow_fallback=True, confirm_new_cluster=False):
        """
        Return the current local-grid-to-world anchor.

        If allow_fallback=False, return None when the anchor is uncertain.
        This mode is used for memory creation to avoid fake farmland coordinates.

        If allow_fallback=True, use deterministic floor fallback.
        This mode is used for relative checks such as goal/crop lookup.
        """
        if not info_dict:
            return None
    
        cache = self.get_step_cache(info_dict)
        cache_key = ("observation_anchor", allow_fallback, confirm_new_cluster)

        if cache is not None and cache_key in cache:
            return cache[cache_key]

        local_offsets = self.get_local_farmland_offsets(info_dict)

        if local_offsets:
            anchor, visible_worlds = self.choose_consistent_anchor(
                info_dict,
                local_offsets,
                confirm_new_cluster=confirm_new_cluster,
            )
            if anchor is not None:
                self.last_grid_anchor = anchor
                self.last_visible_farmland_world = visible_worlds

                if cache is not None:
                    cache[cache_key] = anchor

                return anchor

        if not allow_fallback:
            if cache is not None:
                cache[cache_key] = None
            return None

        x_pos = float(info_dict.get("XPos", 0.0))
        y_pos = float(info_dict.get("YPos", 0.0))
        z_pos = float(info_dict.get("ZPos", 0.0))

        fallback = (
            math.floor(x_pos + 1e-6),
            math.floor(y_pos + 1e-6),
            math.floor(z_pos + 1e-6),
        )

        if cache is not None:
            cache[cache_key] = fallback

        return fallback
    
    def is_los_hitting_top_of_block(self, info_dict, los=None, tolerance=0.08):
        """
        Return True only if LineOfSight appears to hit the top face of the targeted block.

        For farmland at block y = 3, the top surface is around y = 4.0.
        This assumes los["y"] is the hit/intersection coordinate, not just block y.
        """
        if not info_dict:
            return False

        if los is None:
            los = info_dict.get("LineOfSight", {})

        if not los:
            return False

        if "y" not in los:
            return False

        los_world = self.get_los_block_world(info_dict)
        if los_world is None:
            return False

        _, block_y, _ = los_world

        try:
            hit_y = float(los["y"])
        except (TypeError, ValueError):
            return False

        top_y = block_y + 1.0

        return abs(hit_y - top_y) <= tolerance

    # memory cache helper functions.
    def clear_step_cache(self):
        """
        Clear per-step observation cache.
        This cache must not persist across environment steps because memory/anchor
        logic has side effects.
        """
        self._step_obs_cache = {}


    def get_step_cache(self, info_dict):
        """
        Return cache dictionary for one observation dict.
        Uses id(info_dict), but only within the current step.
        """
        if not hasattr(self, "_step_obs_cache"):
            self._step_obs_cache = {}

        if info_dict is None:
            return None

        key = id(info_dict)
        if key not in self._step_obs_cache:
            self._step_obs_cache[key] = {}

        return self._step_obs_cache[key]
    
    def get_los_distance_to_goal(self, info_dict, goal_world):
        """
        Return distance from currently targeted LineOfSight block to goal farmland.
        Lower means the crosshair is closer to the target block.
        """
        if not info_dict or goal_world is None:
            return None

        los_world = self.get_los_block_world(info_dict)
        if los_world is None:
            return None

        gx, gy, gz = goal_world
        lx, ly, lz = los_world

        dx = lx - gx
        dy = ly - gy
        dz = lz - gz

        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def is_target_goal_usable_now(self, info_dict, goal_world):
        if not info_dict or goal_world is None:
            return False

        if self.can_use_available_farmland_now(info_dict) <= 0.0:
            return False

        targeted_world = self.get_targeted_farmland_world(info_dict)
        return targeted_world == goal_world

    def los_matches_goal(self, info_dict, goal_world):
        return self.is_target_goal_usable_now(info_dict, goal_world)


    def los_matches_wrong_goal(self, info_dict, goal_world):
        if not info_dict or goal_world is None:
            return False

        if self.can_use_available_farmland_now(info_dict) <= 0.0:
            return False

        targeted_world = self.get_targeted_farmland_world(info_dict)
        if targeted_world is None:
            return False

        return targeted_world != goal_world

    # Debug functions - do not affect agent state or reward
    def debug_event(self, name, amount=1):
        if name not in self.debug_event_counts:
            self.debug_event_counts[name] = 0
        self.debug_event_counts[name] += amount

    def debug_print(self, message):
        if self.DEBUG_PRINT:
            print(message, flush=False)