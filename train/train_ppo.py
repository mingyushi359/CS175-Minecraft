import argparse
import copy
import importlib.util
import json
import time
from pathlib import Path
import utility

import imageio.v2 as imageio
import malmoenv
from malmoenv.core import ActionSpace
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

def load_task_module(task_py):
    # loads the task specific module
    if task_py is None:
        return None

    spec = importlib.util.spec_from_file_location("task_module", task_py)
    task_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task_module)

    if hasattr(task_module, "Task"):
        return task_module.Task()

    return task_module

def build_state(info_dict, task_module=None):
    # calls task specific module's build state
    if task_module is not None:
        state = task_module.build_state(info_dict)
        return np.array(state, dtype=np.float32)
    
    raise ValueError("Need to define build_state() in task class")

class StateTextFeatureExtractor(BaseFeaturesExtractor):
    # custom feature extrator that processes texts separately before PPO
    def __init__(self, observation_space, state_dim=18, text_dim=384, text_out_dim=16):
        super().__init__(observation_space, features_dim=state_dim + text_out_dim)

        self.state_dim = state_dim
        self.text_dim = text_dim

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, text_out_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        state = observations[:, :self.state_dim]
        text = observations[:, self.state_dim:self.state_dim + self.text_dim]

        text_features = self.text_proj(text)

        return torch.cat([state, text_features], dim=1)

class PlotCallback(BaseCallback):
    # save plots periodically during PPO training
    def __init__(self, log_dir, save_freq=2000):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.save_freq = save_freq

    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:
            utility.save_monitor_plots(self.log_dir)
        return True

class MalmoStructuredEnv(gym.Env):
    # Malmo env wrapper
    def __init__(self, args, task_module=None):
        super().__init__()
        self.args = args
        self.task_module = task_module
        self.task_id = task_module.TASK_ID if task_module is not None else 0
        self.steps = 0
        self.prev_info_dict = None
        self.prev_state = None
        self.last_frame = None

        self.env = None
        self.init_malmo_env()

        print({i: self.env.action_space[i] for i in range(self.env.action_space.n)})

        sample_state = build_state({}, task_module=self.task_module)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample_state.shape,
            dtype=np.float32,
        )

        if self.task_module is not None:
            # initialize PPO without the quit action
            self.action_space = spaces.Discrete(len(self.task_module.CUSTOM_ACTIONS))
        else:
            self.action_space = spaces.Discrete(self.env.action_space.n)

    def init_malmo_env(self):
        # initialize malmo env

        # use random generated raw mission xml
        if getattr(self.args, "random", False) and hasattr(self.task_module, "make_random_mission_xml"):
            xml = self.task_module.make_random_mission_xml(self.args.mission)
        else:  # else use default xml
            xml = Path(self.args.mission).read_text()

        self.env = malmoenv.make()
        # custom_actions = self.task_module.CUSTOM_ACTIONS if self.task_module else None

        # task actions provided from task-py
        task_actions = self.task_module.CUSTOM_ACTIONS if self.task_module else None
        self.quit_action_index = None

        if task_actions is not None:  # add quit to custom actions
            custom_actions = list(task_actions) + ["quit"]
            self.quit_action_index = len(task_actions)
        else:
            custom_actions = None

        init_kwargs = dict(
            server=self.args.server,
            server2=self.args.server2,
            port2=self.args.port2,
            role=self.args.role,
            exp_uid=self.args.experimentUniqueId,
            episode=self.args.episode,
            resync=self.args.resync,
        )

        if custom_actions is not None:  # initalize malmo with additioanl quit action
            init_kwargs["action_space"] = ActionSpace(custom_actions)

        self.env.init(xml, self.args.port, **init_kwargs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # re-initialize malmo env per episode for randomized mission xml
        if getattr(self.args, "random", False):
            if self.env is not None:
                self.env.close()
            self.init_malmo_env()

        self.env.reset()
        self.steps = 0
        self.prev_info_dict = None
        self.prev_state = None
        if self.task_module:
            self.task_module.reset(instruction=self.args.instruction, eval_mode=self.args.eval)

        # dummy step to get observation space dimensions
        obs, reward, done, info = self.env.step(0)
        self.last_frame = obs
        self.obs_shape = self.env.observation_space.shape
        info_dict = json.loads(info) if info else {}
        if info_dict:
            state = build_state(info_dict, task_module=self.task_module)
            self.prev_info_dict = info_dict
        else:
            state = np.zeros(self.observation_space.shape, dtype=np.float32)

        self.prev_state = state
        return state, {}
    
    def step(self, action):
        self.steps += 1

        obs, reward, done, info = self.env.step(int(action))
        self.last_frame = obs
        info_dict = json.loads(info) if info else {}

        reward = float(reward)
        task_done = False

        if info_dict:
            if self.task_module:
                reward, task_done, metrics = self.task_module.shape_reward(
                    raw_reward=reward,
                    prev_info=self.prev_info_dict,
                    curr_info=info_dict,
                    action=int(action),
                    step=self.steps,
                )

            state = build_state(info_dict, task_module=self.task_module)

            self.prev_info_dict = info_dict
            self.prev_state = state
        else:  # when info_dict is empty
            if self.prev_state is not None:  # reuse previous state if available
                state = self.prev_state
            else:  # else return zero
                state = np.zeros(self.observation_space.shape, dtype=np.float32)

        terminated = bool(done or task_done)
        truncated = bool(self.args.episodemaxsteps > 0 and self.steps >= self.args.episodemaxsteps)
        if (task_done or truncated) and not done and self.quit_action_index is not None:
            try:  # manually calling quit action when done=True
                self.env.step(self.quit_action_index)
                # gives time for malmo to quit before reset
                # you might want to increase delay if the minecraft window ever gets frozen or unresponsive
                time.sleep(0.3)  
            except Exception as e:
                print(f"Warning: failed to send Malmo quit command: {e}")

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
    parser.add_argument('--instruction', type=str, default=None, help='text instruction for eval')
    parser.add_argument('--projection', action='store_true', help='enable the projection layer for multi-target instruction training, else defaults to raw etxt embedding concact')
    parser.add_argument('--random', action='store_true', help='randomize xml for each episode, need to define "make_random_mission_xml" in task-py and add <MissionQuitCommands/> to xml')
    parser.add_argument('--load-model', type=str, default=None, help='path to a PPO checkpoint (.zip) to continue training')
    parser.add_argument('--lr', type=float, default=3e-4, help='PPO learning rate')
    parser.add_argument('--n-steps', type=int, default=512, help='PPO n_steps before policy update')
    parser.add_argument('--batch-size', type=int, default=64, help='PPO batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='PPO gamma')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='PPO entropy')


    args = parser.parse_args()
    if args.server2 is None:
        args.server2 = args.server

    task_module = load_task_module(args.task_py)
    env = MalmoStructuredEnv(args, task_module=task_module)

    if args.eval:
        model = PPO.load(args.model_path, env=env, device="cpu")
        if args.record:
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

                if env.last_frame is not None and env.last_frame.size != 0:
                    frames.append(np.flipud(env.last_frame.reshape(env.obs_shape)))

                # time.sleep(0.15)  # delay in eval might cause weird actions due to delayed state

            if args.record and frames:
                record_path = record_dir / f"episode_{i}_reward_{episode_reward:.2f}.gif"
                imageio.mimsave(record_path, frames, fps=8, loop=0)

            print(
                f"EVAL episode={i}, steps={steps}, "
                f"reward={episode_reward:.2f}, actions={action_counts}"
            )

    else:
        log_dir = Path(args.model_path)
        env = Monitor(env, filename=str(log_dir / "monitor.csv"))  # Monitor warpper for logging rewards

        policy_kwargs=dict(  # larger PPO network
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )

        if args.projection:  # additional projection layer for text
            policy_kwargs["features_extractor_class"] = StateTextFeatureExtractor
            policy_kwargs["features_extractor_kwargs"] = dict(
                state_dim=task_module.STATE_DIM,
                text_dim=task_module.TEXT_MODEL_DIM,
                text_out_dim=task_module.TEXT_OUT_DIM,
            )
            print(f"Projection layer - " \
                  f"state_dim: {task_module.STATE_DIM}, " \
                  f"text_dim: {task_module.TEXT_MODEL_DIM}, " \
                  f"text_out_dim: {task_module.TEXT_OUT_DIM}"
                  )
            
        if args.load_model:  # load existing checkpoint
            model = PPO.load(args.load_model, env=env, device="cpu")
            model.learning_rate = args.lr
        else:
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=args.lr,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                gamma=args.gamma,
                ent_coef=args.ent_coef,
                policy_kwargs=policy_kwargs,
                device="cpu",
            )

        checkpoint_callback = CheckpointCallback(
            save_freq=2500,
            save_path=args.model_path,
            name_prefix="ppo"
        )

        plot_callback = PlotCallback(
            log_dir=log_dir,
            save_freq=512,
        )

        callback = CallbackList([checkpoint_callback, plot_callback])

        try:
            model.learn(total_timesteps=args.total_timesteps, callback=callback)
            model.save(Path(args.model_path) / "ppo_final.zip")
            print(f"Saved PPO model to {args.model_path}/ppo_final.zip")
        finally:
            if not args.eval:
                utility.save_monitor_plots(log_dir)  # call save plots before exiting

    env.close()
