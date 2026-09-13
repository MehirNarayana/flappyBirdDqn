# Flappy Bird DQN

I built this project after reading Maxim Lapan's ```Deep Reinforcement Learning Hands-On```, to put what I'd learned into practice. The goal was to teach an agent to play Flappy Bird.

## Example run

<p align="center">
  <a href="https://www.youtube.com/watch?v=hueP16FkSwY">
    <img src="https://img.youtube.com/vi/hueP16FkSwY/hqdefault.jpg" alt="Watch the Flappy Bird agent play" width="480">
  </a>
  <br>
  <a href="https://www.youtube.com/watch?v=hueP16FkSwY">Watch the agent run on YouTube</a>
</p>

## Model architecture

The agent uses a fully connected PyTorch network with a 12-value game-state vector as input:

```text
12 state features → Normalization → Linear(12, 128) → ReLU
                                 → Linear(128, 64) → ReLU
                                 → Linear(64, 2)
```

The two outputs are Q-values: estimated future returns for **flap** and **do nothing**. During inference, the agent picks the action with the higher value. Observation normalization collects running mean and variance for the first 20,000 observations, then freezes them.

Training uses **Double DQN**: the policy network selects the next action, while a target network with the same architecture evaluates it. The replay buffer holds 50,000 transitions, sampled in batches of 32.

## Running

Use Python 3.11 and run from the project root:

```sh
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -e .

python bird.py --mode train
python bird.py --mode inference
```

Inference uses `flappy_bird_dqn.pth` and `flappy_bird_normalization.pth`. The pretrained model and its normalization statistics are included so you can try inference without training first. Use matching model weights and normalization statistics. Training writes its best weights to `best.dat` and can overwrite the normalization file.

Select files or a device explicitly:

```sh
python bird.py --mode inference --checkpoint best.dat --rms flappy_bird_normalization.pth --device cpu
```

Device selection otherwise prefers CUDA, then MPS, then CPU. View training curves with `tensorboard --logdir runs`.

## Formatting

Dependencies are defined in `pyproject.toml`. Install the `dev` extra to include Ruff:

```sh
python -m pip install -e '.[dev]'
python -m ruff format .
python -m ruff format --check .
```
