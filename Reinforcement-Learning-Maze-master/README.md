# Reinforcement Learning Maze Final Project

This project compares reinforcement learning agents in a maze environment with
walls, traps, reward shaping, fixed maps, and dynamic maps.

The main comparison used in this project is:

- Q-learning
- DDQN

The demo code shows the learned walking path of each agent in real time.

## Project Goals

The agent starts from a valid cell and tries to reach the exit while avoiding
walls and traps.

The project includes two maze settings:

1. Fixed maze
   - The map does not change.
   - Traps are fixed.
   - Q-learning and DDQN are compared on the same maze.

2. Dynamic maze
   - The map/trap/reward can change between rounds.
   - This is harder than the fixed maze.
   - Q-learning only observes position, while DDQN observes map information.

## Environment Rules

The maze is defined in `environment/maze.py`.

Cell types:

| Value | Meaning |
| --- | --- |
| `0` | empty cell |
| `1` | wall |
| `2` | trap / hole |
| `3` | current agent position for rendering |

Actions:

| Action | Meaning |
| --- | --- |
| `0` | move left |
| `1` | move right |
| `2` | move up |
| `3` | move down |

Current reward design:

| Event | Reward |
| --- | --- |
| reach exit | `+20.0` |
| step into trap | `-20.0` |
| normal move | `-0.05` |
| revisit cell | `-0.30` |
| impossible move / wall | `-1.00` |
| move closer to exit | `+0.05` |
| move farther from exit | `-0.05` |

The closer/farther reward is reward shaping. It helps the agent learn a useful
direction without changing the final goal.

## File Overview

| File | Purpose |
| --- | --- |
| `environment/maze.py` | Maze environment, rewards, traps, terminal states, rendering |
| `train_maze.py` | Train tabular models such as Q-learning |
| `train_dqn_maze.py` | Train neural-network DQN/DDQN-style agent on the fixed maze |
| `dynamic_maze_compare.py` | Train and compare Q-learning/DDQN in dynamic maze settings |
| `demo_agents.py` | Evaluate trained fixed-maze agents and save path plots |
| `dynamic_trap_demo.py` | Demo dynamic trap rounds with live or saved visualization |
| `showcase_demo.py` | Main presentation demo for Q-learning vs DDQN |
| `models/` | Saved trained models |
| `plots/` | Training curves, best-move plots, demo images, CSV results |

## Requirements

Recommended packages:

```bash
pip install numpy matplotlib torch
```

If you use Conda:

```bash
conda activate new_env
pip install numpy matplotlib torch
```

TensorFlow is not required for the current Q-learning/DDQN demos.

## Important: Run From The Project Folder

You can run scripts from inside this folder:

```bash
cd "Reinforcement-Learning-Maze-master"
```

Or you can run the file by absolute path. The newer demo scripts automatically
resolve model paths relative to the script folder.

## Main Live Demo

Use this for presentation.

It shows:

- Fixed maze: 13 rounds
- Dynamic maze: 13 rounds
- Q-learning and DDQN side by side
- Real-time inference path rendering

```bash
python showcase_demo.py --mode both --rounds 13
```

If you are running from outside the folder:

```bash
/Users/iris/anaconda3/envs/new_env/bin/python \
"/Users/iris/Library/CloudStorage/GoogleDrive-iriscdrive@gmail.com/我的雲端硬碟/NCKU/碩一下/RL/RL_Final_Project/Reinforcement-Learning-Maze-master/showcase_demo.py" \
--mode both \
--rounds 13
```

Only fixed maze:

```bash
python showcase_demo.py --mode fixed --rounds 13
```

Only dynamic maze:

```bash
python showcase_demo.py --mode dynamic --rounds 13
```

Animation speed controls:

```bash
python showcase_demo.py \
  --mode both \
  --rounds 13 \
  --live-delay 0.02 \
  --live-step-stride 5
```

If the animation is too fast, use a larger `--live-delay` or smaller
`--live-step-stride`.

If the animation is too slow, use a smaller `--live-delay` or larger
`--live-step-stride`.

## Expected Demo Behavior

The terminal should print something like:

```text
Opening live walking-path demo window with matplotlib backend: MacOSX
```

Then a Matplotlib window opens.

For `--mode both`:

1. The fixed maze demo appears first.
2. After the 13 fixed rounds finish, close the window.
3. The dynamic maze demo then appears.
4. Close the second window when finished.

If the backend says `Agg`, then the current Python environment cannot open an
interactive GUI window. Try running from Terminal with Conda environment
activated, or use the `python` from your Conda environment directly.

## Train Fixed-Maze Q-learning

```bash
python train_maze.py \
  --model qtable \
  --episodes 300 \
  --stop-at-convergence \
  --render nothing \
  --save-plot plots/qtable_trap_training_curve.png \
  --save-bestmove plots/qtable_trap_bestmove.png \
  --save-model models/qtable_trap_model.pkl
```

Outputs:

- `models/qtable_trap_model.pkl`
- `plots/qtable_trap_training_curve.png`
- `plots/qtable_trap_bestmove.png`

## Train Fixed-Maze DDQN

```bash
python train_dqn_maze.py \
  --episodes 1500 \
  --epsilon-decay 0.996 \
  --reward-scale 20 \
  --stop-at-convergence \
  --render nothing \
  --save-plot plots/dqn_trap_training_curve.png \
  --save-bestmove plots/dqn_trap_bestmove.png \
  --save-model models/dqn_trap_model.pt
```

Outputs:

- `models/dqn_trap_model.pt`
- `plots/dqn_trap_training_curve.png`
- `plots/dqn_trap_bestmove.png`

Note: the file is named `dqn_trap_model.pt`, but the current implementation uses
Double-DQN style target selection in the update logic.

## Evaluate Fixed-Maze Agents

Evaluate all legal start positions:

```bash
python demo_agents.py --agent both --episodes 0
```

Save a path plot:

```bash
python demo_agents.py \
  --agent ddqn \
  --episodes 0 \
  --save-path-plot plots/ddqn_demo_path.png
```

## Train Dynamic Maze Agents

Dynamic maze training changes map/trap/reward between episodes.

```bash
python dynamic_maze_compare.py \
  --agent both \
  --episodes 2000 \
  --eval-episodes 300 \
  --save-plot plots/dynamic_compare_curve.png \
  --save-csv plots/dynamic_compare_metrics.csv \
  --save-q-model models/dynamic_qlearning.pkl \
  --save-ddqn-model models/dynamic_ddqn.pt
```

Recommended:

- Q-learning: around `2000` episodes
- DDQN: around `2000` to `3000` episodes

Dynamic maze is much harder than fixed maze, so short training runs may perform
poorly.

## Dynamic Trap Demo

This demo changes trap positions every round and compares agents.

```bash
python dynamic_trap_demo.py \
  --rounds 10 \
  --agent both \
  --live-render
```

Fast playback:

```bash
python dynamic_trap_demo.py \
  --rounds 10 \
  --agent both \
  --live-render \
  --live-delay 0.02 \
  --live-step-stride 8
```

Save summary plots instead of live rendering:

```bash
python dynamic_trap_demo.py \
  --rounds 10 \
  --agent both \
  --save-plot plots/dynamic_trap_10round_demo.png \
  --save-csv plots/dynamic_trap_10round_demo.csv
```

## Model Files

Important saved models:

| Model | File |
| --- | --- |
| Fixed maze Q-learning | `models/qtable_trap_model.pkl` |
| Fixed maze DDQN | `models/dqn_trap_model.pt` |
| Dynamic maze Q-learning | `models/dynamic_qlearning.pkl` |
| Dynamic maze DDQN | `models/dynamic_ddqn.pt` |

## Plot Files

Important output plots:

| Plot | File |
| --- | --- |
| Q-learning training curve | `plots/qtable_trap_training_curve.png` |
| Q-learning best move | `plots/qtable_trap_bestmove.png` |
| DDQN training curve | `plots/dqn_trap_training_curve.png` |
| DDQN best move | `plots/dqn_trap_bestmove.png` |
| Dynamic comparison curve | `plots/dynamic_compare_curve.png` |
| Dynamic trap demo | `plots/dynamic_trap_10round_demo.png` |

## Q-learning vs DDQN

Q-learning:

- Uses a Q-table.
- State is mainly the agent position.
- Very effective in small fixed mazes.
- Does not generalize well when trap/map changes.

DDQN:

- Uses a neural network.
- Uses replay buffer.
- Uses target network.
- Uses valid-action masking.
- Can encode richer state information.
- More suitable for dynamic map/trap/reward settings.

For the report, a clear comparison is:

| Setting | Expected Result |
| --- | --- |
| Fixed maze | Q-learning is fast and stable; DDQN can also learn |
| Dynamic maze | Q-learning struggles because state is incomplete; DDQN is more suitable |

## Troubleshooting

### `FileNotFoundError: models/qtable_trap_model.pkl`

Run from this folder:

```bash
cd "Reinforcement-Learning-Maze-master"
python showcase_demo.py --mode both --rounds 13
```

The latest `showcase_demo.py` also supports absolute-path execution and resolves
model paths relative to the script folder.

### No rendering window appears

Make sure you are not using `--no-live-render`.

Run:

```bash
python showcase_demo.py --mode fixed --rounds 13
```

The terminal should show a GUI backend such as:

```text
Opening live walking-path demo window with matplotlib backend: MacOSX
```

If it shows `Agg`, your environment is non-interactive.

### Round looks frozen

The agent may be taking many steps before timeout.

Use faster playback:

```bash
python showcase_demo.py \
  --mode fixed \
  --rounds 13 \
  --live-delay 0.02 \
  --live-step-stride 8
```

### Dynamic DQN model missing

The current main demo compares Q-learning and DDQN only. It does not show DQN.

## Suggested Presentation Flow

1. Explain the maze rules, traps, and rewards.
2. Show fixed maze live demo:

```bash
python showcase_demo.py --mode fixed --rounds 13
```

3. Show dynamic maze live demo:

```bash
python showcase_demo.py --mode dynamic --rounds 13
```

4. Show training curves in `plots/`.
5. Explain why Q-learning works well in fixed maze but DDQN is more suitable for
   dynamic settings.
