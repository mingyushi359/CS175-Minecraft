# CS175-Minecraft

Running Minecraft simulation using [`MalmoEnv`](https://github.com/Microsoft/malmo/tree/master/MalmoEnv)

---

## Java Requirements (Windows)

### 1. Install Java8 JDK ([AdoptOpenJDK](https://adoptopenjdk.net/))

### 2. Set `JAVA_HOME` and add to `PATH`

* Create new **Environment Variables** `JAVA_HOME` with your JDK path (e.g. `C:\program Files\Eclipse Adoptium\jdk-8\`)

* Add path `<your JDK path>\bin` to system system `PATH` variable

### 3. Verify

Make sure `java -version` shows the correct version in `cmd`

---

## Getting Started

### 1. Create virtual environment (~python3.9) and git clone this repo

### 2. Install dependencies

```bash
pip install malmoenv gymnasium numpy pillow lxml stable-baselines3 imageio matplotlib pandas transformers sentence-transformers
```
You might need to install GPU-compatible versions of `torch` if you wanna use GPU for training, current default uses CPU

### 3. Download MalmoPlatform

From the root directory (of this repo), run:

```bash
python -c "import malmoenv.bootstrap; malmoenv.bootstrap.download()"
```


### 4. Run Minecraft

Start a new instance of Minecraft

```bash
python -c "import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft(9000)"
```

### 5. Run sample missions

Start a new terminal and  run:
```bash
python sample_scripts/run.py --mission sample_missions/mobchase_single_agent.xml
```

The original sample scripts and missions are located in `MalmoPlatform/MalmoEnv/` and `MalmoPlatform/MalmoEnv/Missions`

---


## Training PPO Agent
### Single Target

```bash
python -u train/train_ppo.py --mission missions/mob_chase_single_agent.xml --episodemaxsteps 100 --total-timesteps 100000 --model-path ppo_logs/out1/ --task-py train/tasks/mob_chase.py > ppo_logs/out1/out.txt
```
`--mission` is the mission.xml file

`--episodemaxsteps` is the max steps the agent can take per episode

`--timesteps` is the total number of steps the agent will take during the entire training. Currently set to 100000 as a safe upperbound. Single target tasks like `mob_chase` will usually require less thank 50k timesteps to train, so just end training early as needed

`--model-path` is the path where the trained models and logs will be saved

`--task-py` is an additional (and required) custom task file that handles building state and shapping reward. See `train/tasks/mob_chase.py` as an example


### Multi-Target
```bash
python -u train/train_ppo.py --mission missions/multi_target_single_agent.xml --episodemaxsteps 100 --total-timesteps 150000 --model-path ppo_logs/out1/ --task-py train/tasks/multi_target_navigation.py --projection > ppo_logs/out1/out.txt
```

The training command for multi-target tasks is the mostly same. Just change the `--task-py` and `--mission` accordingly

`--projection` is an optional argument for enabling the projection layer that projects the text instruction through trainable MLP instead of passing the raw encoded embedding into the PPO network

Note that current multi-target setup might requires more timesteps. It's recommended to set `--total-timesteps` to 150000 for tasks like `multi_target_navigation.py` just in case it takes more steps to train, but you can always end early as needed

---

## Evaluating PPO Agent
### Single Target

```bash
python train/train_ppo.py --mission missions/mob_chase_single_agent.xml --episodemaxsteps 100 --task-py train/tasks/mob_chase.py --model-path ppo_logs/out1/ppo_final.zip --task-py train/tasks/mob_chase.py --eval --episodes 5
```

`--mission`, `--task-py`, and `--episodemaxsteps 100` are the same as training

`--model-path` need to be exact path to a saved checkpoint (.zip)

`--eval` indicates evluation instead of training

`--episodes` is the number of episodes to run for evluation

`--record` optionally records the evluation and saves as GIF

In cases where the agent performs poorly due to bad policy learning given imperfect reward shaping, you can try changing `deterministic` to `False` in `train_ppo.py` to allow the agent to take some random actions during evaluation. Not ideal, but performs better is some cases

### Multi-Target

```bash
python train/train_ppo.py --mission missions/multi_target_single_agent.xml --episodemaxsteps 100 --task-py train/tasks/multi_target_navigation.py --model-path ppo_logs/out13/ppo_65000_steps.zip --projection --eval --episodes 5 --instruction "go to the pig" 
```
Mostly the same as single target eval, except:

`--projection` need to be consistent with training

`--instruction` need to pass in the text instruction for multi-target evaluation

## Example Recording
### mob_chase
<img width="160" height="120" alt="episode_0_reward_51 46" src="https://github.com/user-attachments/assets/da12f77a-0249-4188-a54c-99509444537f" />
<img width="160" height="120" alt="episode_1_reward_47 32" src="https://github.com/user-attachments/assets/31c95dcf-080a-4e96-9f0f-fa0b80999bb6" />

### multi-target_navigation
|"go to the pig"|"go to the log"|"go to the emerald"|
| -------- | -------- | -------- |
|<img width="160" height="120" alt="episode_0_reward_48 05" src="https://github.com/user-attachments/assets/b13984b9-0120-4780-ba5a-03c337b70104" />|<img width="160" height="120" alt="episode_0_reward_51 02" src="https://github.com/user-attachments/assets/a3f11098-c3dc-4157-89d9-142b889c8799" />|<img width="160" height="120" alt="episode_0_reward_41 58" src="https://github.com/user-attachments/assets/51485fcb-8668-44c0-9889-7afb225e43d3" />|

## Example learning plot
### multi-target_navigation

<img width="1500" height="750" alt="reward_curve" src="https://github.com/user-attachments/assets/8dbd5fe0-482e-4537-a261-0c19e195bdca" />
<img width="1500" height="750" alt="steps_curve" src="https://github.com/user-attachments/assets/3b2fc9f8-9191-44db-9f8f-59e6736a1284" />

This run was from the "Add obstacle states" commit. The best model appears to be the 57500_steps checkpoint, and the learning started to drift away after that

---

## Training DQN Agent

`train_dqn.py` still have issues and not properly learning, only use it as a reference

```bash
python train/train_dqn.py --mission missions/reach_target_single_agent.xml --task train/tasks/mob_chase.py --episodes 700 --episodemaxsteps 100 --model-path q_model
```

## Evaluating DQN Agent

```bash
python train/train_dqn.py --mission missions/reach_target_single_agent.xml --eval --task train/tasks/mob_chase.py --episodes 5 --episodemaxsteps 100 --model-path q_model
```

---

## Miscellaneous

To change Minecraft memory allocation, go to the downloaded `MalmoPlatform/Minecraft/build.gradle`, locate the `exec.jvmArgs` on line 51, and change the `"-Xmx2G"` value to whatever (e.g. `"-Xmx4G"` for 4GB of memory)
