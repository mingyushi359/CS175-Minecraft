import argparse
from ast import While
import importlib.util
import json
import math
import time
from pathlib import Path

import imageio.v2 as imageio
import malmoenv
from malmoenv.core import ActionSpace
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

BLOCK_TO_ID = {
    "air": 0,
    "grass": 1,
    "dirt": 2,
    "clay": 3,
    "brick_block": 4,
    "log": 5,
    "leaves": 6,
    "stone": 7,
    "planks": 8,
    "emerald_block": 9,
}

ENTITY_TO_ID = {
    "Chicken": 1,
    "Pig": 2,
    "Cow": 3,
    "Sheep": 4,
}


def load_task_module(task_py):
    if task_py is None:
        return None

    spec = importlib.util.spec_from_file_location("task_module", task_py)
    task_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task_module)
    return task_module

def build_state(info_dict, task_id=0, max_entities=1, max_board_size=200):
    # extracts basic state infos
    x = float(info_dict.get("XPos", 0))
    y = float(info_dict.get("YPos", 0))
    z = float(info_dict.get("ZPos", 0))
    yaw = float(info_dict.get("Yaw", 0))
    pitch = float(info_dict.get("Pitch", 0))
    life = float(info_dict.get("Life", 0))
    food = float(info_dict.get("Food", 0))
    air = float(info_dict.get("Air", 0))

    agent_features = [
        yaw / 180.0, pitch / 90.0
    ]

    entities = []
    for entity in info_dict.get("entities", []):
        # skip agent itself
        if entity["name"] == info_dict.get("Name"):
            continue

        dx = float(entity["x"]) - x
        dy = float(entity["y"]) - y
        dz = float(entity["z"]) - z
        distance = np.sqrt(dx * dx + dy * dy + dz * dz)
        target_yaw = math.degrees(math.atan2(-dx, dz))
        signed_yaw_error = (target_yaw - yaw + 180.0) % 360.0 - 180.0
        entity_type = float(ENTITY_TO_ID.get(entity["name"], -1))

        dist2 = dx * dx + dy * dy + dz * dz
        entities.append((dist2, [dx / 12.0, dy / 5.0, dz / 12.0, distance / 17.0, signed_yaw_error / 180.0, entity_type / 10.0]))
 
    entities.sort(key=lambda x: x[0])

    entity_features = []
    # sort and only consider the closest max_entities entities
    for _, entity in entities[:max_entities]:
        entity_features.extend(entity)

    # pad with zeros if there are less than max_entities entities
    while len(entity_features) < max_entities * 6:
        entity_features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    board_features = []
    for block in info_dict.get("board", []):
        board_features.append(float(BLOCK_TO_ID.get(block, -1)))

    if len(board_features) < max_board_size:
        board_features.extend([0.0] * (max_board_size - len(board_features)))
    else:
        board_features = board_features[:max_board_size]

    # state = agent_features + entity_features + board_features + [float(task_id)]
    # skip board features for now
    state = agent_features + entity_features + [float(task_id)]
    return np.array(state, dtype=np.float32)

class MalmoStructuredEnv(gym.Env):
    # Malmo env wrapper
    def __init__(self, args, task_module=None):
        super().__init__()

        self.args = args
        self.task_module = task_module
        self.task_id = getattr(task_module, "TASK_ID", 0) if task_module else 0
        self.steps = 0
        self.prev_info_dict = None
        self.last_frame = None

        xml = Path(args.mission).read_text()
        self.env = malmoenv.make()
        
        # custom actions can be defined in the task specific reward module and mission XML
        custom_actions = getattr(task_module, "CUSTOM_ACTIONS", None) if task_module else None

        if custom_actions is not None:
            self.env.init(
                xml,
                args.port,
                server=args.server,
                server2=args.server2,
                port2=args.port2,
                role=args.role,
                exp_uid=args.experimentUniqueId,
                episode=args.episode,
                resync=args.resync,
                action_space = ActionSpace(custom_actions),  
            )
        else:  # default action space will be inferred from the mission XML only
            self.env.init(
                xml,
                args.port,
                server=args.server,
                server2=args.server2,
                port2=args.port2,
                role=args.role,
                exp_uid=args.experimentUniqueId,
                episode=args.episode,
                resync=args.resync,
            )

        print({i: self.env.action_space[i] for i in range(self.env.action_space.n)})

        sample_state = build_state({}, task_id=self.task_id)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample_state.shape,
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(self.env.action_space.n)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.env.reset()
        self.steps = 0
        self.prev_info_dict = None
        if self.task_module and hasattr(self.task_module, "reset"):
            self.task_module.reset()

        # dummy step to get observation space dimensions
        obs, reward, done, info = self.env.step(0)
        self.last_frame = obs
        self.obs_shape = self.env.observation_space.shape
        info_dict = json.loads(info) if info else {}
        if info_dict:
            state = build_state(info_dict, task_id=self.task_id)
            self.prev_info_dict = info_dict
        else:
            state = np.zeros(self.observation_space.shape, dtype=np.float32)
        return state, {}
    
    def step(self, action):
        self.steps += 1

        obs, reward, done, info = self.env.step(int(action))
        self.last_frame = obs
        info_dict = json.loads(info) if info else {}

        reward = float(reward)
        task_done = False

        if self.task_module and hasattr(self.task_module, "shape_reward"):
            reward, task_done, metrics = self.task_module.shape_reward(
                raw_reward=reward,
                prev_info=self.prev_info_dict,
                curr_info=info_dict,
                action=int(action),
                step=self.steps,
            )

        if info_dict:
            state = build_state(info_dict, task_id=self.task_id)
        else:
            state = np.zeros(self.observation_space.shape, dtype=np.float32)

        self.prev_info_dict = info_dict if info_dict else self.prev_info_dict

        terminated = bool(done or task_done)
        truncated = bool(self.args.episodemaxsteps > 0 and self.steps >= self.args.episodemaxsteps)

        return state, reward, terminated, truncated, info_dict
    
    def close(self):
        self.env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='malmovnv test')
    parser.add_argument('--mission', type=str, default='missions/mobchase_single_agent.xml', help='the mission xml')
    parser.add_argument('--port', type=int, default=9000, help='the mission server port')
    parser.add_argument('--server', type=str, default='127.0.0.1', help='the mission server DNS or IP address')
    parser.add_argument('--port2', type=int, default=9000, help="(Multi-agent) role N's mission port")
    parser.add_argument('--server2', type=str, default=None, help="(Multi-agent) role N's server DNS or IP")
    parser.add_argument('--episodes', type=int, default=1, help='the number of resets to perform - default is 1')
    parser.add_argument('--episode', type=int, default=0, help='the start episode - default is 0')
    parser.add_argument('--role', type=int, default=0, help='the agent role - defaults to 0')
    parser.add_argument('--episodemaxsteps', type=int, default=0, help='max number of steps per episode')
    parser.add_argument('--saveimagesteps', type=int, default=0, help='save an image every N steps')
    parser.add_argument('--resync', type=int, default=0, help='exit and re-sync every N resets'
                                                              ' - default is 0 meaning never.')
    parser.add_argument('--experimentUniqueId', type=str, default='test1', help="the experiment's unique id.")
    parser.add_argument('--total-timesteps', type=int, default=100000, help='number of PPO training timesteps')
    parser.add_argument('--model-path', type=str, default='ppo_model', help='path to save/load PPO model')
    parser.add_argument('--eval', action='store_true', help='run trained PPO model instead of training')
    parser.add_argument('--task-py', type=str, default=None, help='optional Python task reward file')
    parser.add_argument('--record', action='store_true', help='record videos during evaluation')

    args = parser.parse_args()
    if args.server2 is None:
        args.server2 = args.server

    task_module = load_task_module(args.task_py)
    env = MalmoStructuredEnv(args, task_module=task_module)

    if args.eval:
        model = PPO.load(args.model_path, env=env, device="cpu")
        record_dir = Path(args.model_path).parent / "ppo_eval_records"
        record_dir.mkdir(exist_ok=True)

        for i in range(args.episodes):
            obs, info = env.reset()

            terminated = False
            truncated = False
            episode_reward = 0.0
            steps = 0
            action_counts = {}
            
            frames = []
            if env.last_frame is not None:
                frames.append(np.flipud(env.last_frame.reshape(env.obs_shape)))

            while not terminated and not truncated:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)

                action_counts[action] = action_counts.get(action, 0) + 1

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1

                if env.last_frame is not None:
                    frames.append(np.flipud(env.last_frame.reshape(env.obs_shape)))

                time.sleep(0.25)

            if args.record and frames:
                record_path = record_dir / f"episode_{i}_reward_{episode_reward:.2f}.gif"
                imageio.mimsave(record_path, frames, fps=4)

            print(
                f"EVAL episode={i}, steps={steps}, "
                f"reward={episode_reward:.2f}, actions={action_counts}"
            )

    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            gamma=0.99,
            ent_coef=0.01,
            device="cpu",
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=args.model_path,
            name_prefix="ppo"
        )

        model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)
        model.save(Path(args.model_path) / "ppo_final.zip")
        print(f"Saved PPO model to {args.model_path}/ppo_final.zip")

    env.close()
