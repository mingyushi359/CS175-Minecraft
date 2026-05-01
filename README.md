# CS175-Minecraft

Running Minecraft simulation using [`MalmoEnv`](https://github.com/Microsoft/malmo/tree/master/MalmoEnv)

---

## Java Requirements (Windows)

### 1) Install Java8 JDK ([AdoptOpenJDK](https://adoptopenjdk.net/))

### 2) Set `JAVA_HOME` and add to `PATH`

* Create new **Environment Variables** `JAVA_HOME` with your JDK path (e.g. `C:\program Files\Eclipse Adoptium\jdk-8\`)

* Add path `<your JDK path>\bin` to system system `PATH` variable

### 3) Verify

Make sure `java -version` shows the correct version in `cmd`

---

## Getting Started

### 1. Create virtual environment (~python3.9)

### 2. Install dependencies

```bash
pip install malmoenv gymnasium numpy pillow lxml
```

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

```bash
python -u train/train_ppo.py --mission missions/mob_chase_single_agent.xml --episodemaxsteps 100 --total-timesteps 100000 --model-path ppo_logs/out1/ --task-py train/tasks/mob_chase.py > ppo_logs/out1/out.txt
```

## Evaluating PPO Agent

```bash
python train/train_ppo.py --mission missions/mob_chase_single_agent.xml --task-py train/tasks/mob_chase.py --eval --model-path ppo_logs/out1/ppo_final.zip --episodes 5 --episodemaxsteps 100 --task-py train/tasks/mob_chase.py
```

## Training DQN Agent

```bash
python train/train_dqn.py --mission missions/reach_target_single_agent.xml --task train/tasks/mob_chase.py --episodes 700 --episodemaxsteps 100 --model-path q_model
```

--task defines the specific file that contains the additional reward function for the task (e.g., train/tasks/mob_chase.py)

## Evaluating DQN Agent

```bash
python train/train_dqn.py --mission missions/reach_target_single_agent.xml --eval --task train/tasks/mob_chase.py --episodes 5 --episodemaxsteps 100 --model-path q_model
```
