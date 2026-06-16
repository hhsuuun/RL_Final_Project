# Automated Ball-in-Maze Control System

This project implements the final-project proposal:

- State: `[x, y, vx, vy, d_trap]`
- DQN action space: `Up`, `Down`, `Left`, `Right`, `Stay`
- DDPG action space: continuous board tilt `(theta_x, theta_y)`
- Rewards: goal `+100`, trap `-50`, wall collision `-5`, time penalty `-0.01`,
  plus distance-to-goal shaping.

## Files

- `config.py`: maze size, traps, walls, rewards, physics constants
- `maze_env.py`: reusable headless environment for RL
- `game.py`: Pygame visualization and manual/model demo
- `train_dqn.py`: discrete-control DQN training
- `train_ddpg.py`: continuous-control DDPG training

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run the Game

Manual control:

```bash
python3 game.py --mode manual
```

Random agent:

```bash
python3 game.py --mode random
```

After training:

```bash
python3 game.py --mode dqn --model models/dqn_ball_maze.pt
python3 game.py --mode ddpg --model models/ddpg_ball_maze.pt
```

## Train

```bash
python3 train_dqn.py --episodes 600
python3 train_ddpg.py --episodes 900
```

Show the game view and a live training curve while training:

```bash
python3 train_dqn.py --episodes 600 --render-training
python3 train_ddpg.py --episodes 900 --render-training
```

If rendering slows training too much, render fewer episodes or lower the monitor FPS:

```bash
python3 train_dqn.py --episodes 600 --render-training --render-every 5 --render-fps 20
```

The best checkpoints are saved under `models/`.
Training curves and CSV logs are saved under `plots/`.

Default training outputs:

- `plots/dqn_training_curve.png`
- `plots/dqn_training_log.csv`
- `plots/ddpg_training_curve.png`
- `plots/ddpg_training_log.csv`
