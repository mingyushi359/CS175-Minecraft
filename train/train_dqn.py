from collections import deque
import importlib.util
import json
import random

import malmoenv
import argparse
from pathlib import Path
import time
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

BLOCK_TO_ID = {
    "air": 0,
    "grass": 1,
    "dirt": 2,
    "clay": 3,
    "brick_block": 4,
    "log": 5,
    "log2": 5,
    "leaves": 6,
    "leaves2": 6,
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

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)
    
def train_step(q_network, target_network, optimizer, loss_fn, replay_buffer, batch_size, gamma, device):
    # trains the q network using a batch of transitions from the replay buffer
    if len(replay_buffer) < batch_size:
        return

    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

    # select q values for the actions taken for each transition
    q_values = q_network(states).gather(1, actions)
    with torch.no_grad():  # compute target q values using the Bellman equation
        next_actions = q_network(next_states).argmax(dim=1, keepdim=True)
        next_q_values = target_network(next_states).gather(1, next_actions)
        target_q_values = rewards + gamma * next_q_values * (1 - dones)
        # next_q_values = target_network(next_states).max(dim=1, keepdim=True)[0]
        # target_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = loss_fn(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
    
def choose_action(state, env, q_network, epsilon, device):
    # random exploration
    if state is None or q_network is None or random.random() < epsilon:
        return env.action_space.sample()
    
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = q_network(state_tensor)
    return int(torch.argmax(q_values, dim=1).item())

def build_state(info_dict, task_id=0, max_entities=1, max_board_size=200):
    # extracts basic state infos
    x = float(info_dict.get("XPos", 0))
    y = float(info_dict.get("YPos", 0))
    z = float(info_dict.get("ZPos", 0))
    yaw = float(info_dict.get("Yaw", 0))
    pitch = float(info_dict.get("Pitch", 0))

    agent_features = [
        x / 12.0, y / 10.0, z / 12.0,
        yaw / 180.0,
        pitch / 90.0,
        # float(info_dict.get("Life", 0)),
        # float(info_dict.get("Food", 0)),
        # float(info_dict.get("Air", 0))
    ]

    entities = []
    for entity in info_dict.get("entities", []):
        # skip agent itself
        if entity["name"] == info_dict.get("Name"):
            continue

        dx = float(entity["x"]) - x
        dy = float(entity["y"]) - y
        dz = float(entity["z"]) - z
        entity_type = float(ENTITY_TO_ID.get(entity["name"], -1))

        dist2 = dx * dx + dy * dy + dz * dz
        entities.append((dist2, [dx / 12.0, dy / 5.0, dz / 12.0, entity_type]))
 
    entities.sort(key=lambda x: x[0])

    entity_features = []
    # sort and only consider the closest max_entities entities
    for _, entity in entities[:max_entities]:
        entity_features.extend(entity)

    # pad with zeros if there are less than max_entities entities
    while len(entity_features) < max_entities * 4:
        entity_features.extend([0.0, 0.0, 0.0, 0.0])

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

def load_task_module(task_py):
    if task_py is None:
        return None

    spec = importlib.util.spec_from_file_location("task_module", task_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

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
    parser.add_argument('--eval', action='store_true', help='run inference only')
    parser.add_argument('--model-path', type=str, default='q_model.pt', help='path to save/load model')
    parser.add_argument('--task-py', type=str, default=None, help='optional task reward module')

    args = parser.parse_args()
    if args.server2 is None:
        args.server2 = args.server

    xml = Path(args.mission).read_text()
    env = malmoenv.make()

    env.init(xml, args.port,
             server=args.server,
             server2=args.server2, port2=args.port2,
             role=args.role,
             exp_uid=args.experimentUniqueId,
             episode=args.episode, resync=args.resync)
    
    epsilon_start = 1.0
    epsilon_min = 0.05
    epsilon_decay_steps = 50000
    total_steps = 0

    replay_buffer = deque(maxlen=5000)
    
    q_network = None  # TODO: need to determine the state_dim
    target_network = None
    optimizer = None
    loss_fn = nn.SmoothL1Loss()  # Huber loss
    gamma = 0.99
    batch_size = 64
    target_update_freq = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    task_module = load_task_module(args.task_py)
    task_id = getattr(task_module, "TASK_ID", 0) if task_module else 0

    for i in range(args.episodes):
        # print("reset " + str(i))
        obs = env.reset()  # observations in pixels (ignore for now)

        steps = 0
        state = None
        done = False
        
        episode_reward = 0.0
        episode_loss = 0.0
        loss_count = 0
        prev_info_dict = None
        action_counts = {}
        while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):
            if args.eval:
                epsilon = 0.0
            else:
                epsilon = max(epsilon_min,
                              epsilon_start - (epsilon_start - epsilon_min) * total_steps / epsilon_decay_steps)

            action = choose_action(state, env, q_network, epsilon, device)
            action_counts[action] = action_counts.get(action, 0) + 1

            obs, reward, done, info = env.step(action)
            if info:
                # build state from into dict
                info_dict = json.loads(info)

                # apply task-specific reward shaping if available
                if task_module and hasattr(task_module, "shape_reward"):
                    reward, task_done = task_module.shape_reward(reward, prev_info_dict, info_dict)
                    done = done or task_done

                next_state = build_state(info_dict, task_id=task_id)
                prev_info_dict = info_dict

                if q_network is None:
                    # initialize q network after knowing the state_dim for now
                    state_dim = len(next_state)
                    action_dim = env.action_space.n
                    q_network = QNetwork(state_dim, action_dim).to(device)
                    target_network = QNetwork(state_dim, action_dim).to(device)
                    if args.eval:
                        q_network.load_state_dict(torch.load(args.model_path, map_location=device))
                        q_network.eval()
                    else:
                        q_network.train()

                    target_network.load_state_dict(q_network.state_dict())
                    target_network.eval()
                if optimizer is None and not args.eval:
                    # same for optimizer
                    optimizer = optim.Adam(q_network.parameters(), lr=3e-4)
                
            else:
                next_state = None

            if state is not None and next_state is not None:
                # store transition in replay buffer
                if not args.eval:
                    replay_buffer.append((state, action, reward, next_state, done))
                    loss = train_step(q_network, target_network, optimizer, loss_fn, replay_buffer, batch_size, gamma, device)
                    if loss is not None:
                        episode_loss += loss
                        loss_count += 1

            if not args.eval and q_network is not None and target_network is not None:
                # periodically update target network
                if total_steps > 0 and total_steps % target_update_freq == 0:
                    target_network.load_state_dict(q_network.state_dict())

            state = next_state

            steps += 1
            total_steps += 1
            episode_reward += reward

            if args.eval:
                time.sleep(.1)
                

        avg_loss = episode_loss / loss_count if loss_count > 0 else 0.0
        # print(f"Episode {i}: reward={episode_reward:.2f}, avg_loss={avg_loss:.4f}, epsilon={epsilon:.4f}, buffer={len(replay_buffer)}")
        print(
            f"Time={time.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"Episode={i+1}, steps={steps}, reward={episode_reward:.2f}, "
            f"avg_loss={avg_loss:.4f}, epsilon={epsilon:.4f}, "
            f"buffer={len(replay_buffer)}, actions={action_counts}"
        )

        if not args.eval and q_network is not None and (i + 1) % 100 == 0:
            # save every 100 episodes
            torch.save(q_network.state_dict(), args.model_path)
            # print(f"Saved model to {args.model_path}")

    if q_network is not None and not args.eval:
        torch.save(q_network.state_dict(), args.model_path)
        print(f"Saved model to {args.model_path}")
    
    env.close()
