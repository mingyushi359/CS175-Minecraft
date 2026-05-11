import argparse
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
        self.last_frame = None

        xml = Path(args.mission).read_text()
        self.env = malmoenv.make()
        
        # custom actions can be defined in the task specific reward module and mission XML
        custom_actions = task_module.CUSTOM_ACTIONS if task_module else None

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

        sample_state = build_state({}, task_module=self.task_module)
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
        return state, {}
    
    def step(self, action):
        self.steps += 1

        obs, reward, done, info = self.env.step(int(action))
        self.last_frame = obs
        info_dict = json.loads(info) if info else {}

        reward = float(reward)
        task_done = False

        if self.task_module:
            reward, task_done, metrics = self.task_module.shape_reward(
                raw_reward=reward,
                prev_info=self.prev_info_dict,
                curr_info=info_dict,
                action=int(action),
                step=self.steps,
            )

        if info_dict:
            state = build_state(info_dict, task_module=self.task_module)
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
    parser.add_argument('--instruction', type=str, default=None, help='text instruction for eval')


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

                if env.last_frame is not None and env.last_frame.size != 0:
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
        log_dir = Path(args.model_path)
        env = Monitor(env, filename=str(log_dir / "monitor.csv"))  # Monitor warpper for logging rewards

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            gamma=0.99,
            ent_coef=0.01,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256])
            ),
            device="cpu",
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=2500,
            save_path=args.model_path,
            name_prefix="ppo"
        )

        plot_callback = PlotCallback(
            log_dir=log_dir,
            save_freq=2000,
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
