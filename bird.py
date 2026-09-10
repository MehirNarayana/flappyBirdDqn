from Wrappers import construct
from Wrappers import constructNormalizer
from Wrappers import constructNormalizerHuman

from dqn import DQN

import argparse
import time
import numpy as np
from collections import namedtuple
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


gamma = 0.99
epsilon = 0.01
tau = 0.005
decayRate = 50000
minEpsilon = 0.01
batchSize = 32
replaySize = 50000
updateRate = 1000
rewardCutoff = 100
Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)

playFree = False


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0
        self.eps = 1e-5  # small constant to avoid zero priority

    def reset(self):
        self.buffer.clear()
        self.priorities.clear()
        self.pos = 0

    def add(self, exp):
        # exp = (state, action, reward, done, next_state)
        max_prio = max(self.priorities, default=1.0)

        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = exp
            self.priorities[self.pos] = max_prio

        self.pos = (self.pos + 1) % self.capacity

    def full(self):
        return len(self.buffer) == self.capacity

    def sample(self, batch_size=32, beta=0.4):
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from an empty buffer")

        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize for stability

        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, dones, next_states = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(next_states),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(abs(prio) + self.eps)


class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=replaySize)

    def reset(self):
        self.buffer.clear()

    def add(self, exp):
        self.buffer.append(exp)

    def full(self):
        return len(self.buffer) == replaySize

    def sample(self):
        indices = np.random.choice(len(self.buffer), batchSize, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices]
        )
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(next_states),
        )


class Agent:
    def __init__(self, env, buffer):
        self.env = env
        self.buffer = buffer
        self.reset()

    def reset(self):
        self.state, _ = self.env.reset()  # Fixed: use self.env instead of global env
        self.episode_reward = 0

    def step(self, net, device, epsilon):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()

        else:
            torch_state = torch.tensor(np.array([self.state]), dtype=torch.float32).to(
                device
            )
            pred = net(torch_state)
            _, action = torch.max(pred, dim=1)
            action = int(action.item())

        nextState, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        done = truncated or terminated
        if done:
            saveReward = self.episode_reward

            currExp = Experience(self.state, action, reward, True, nextState)
            self.buffer.add(currExp)

        else:
            currExp = Experience(self.state, action, reward, False, nextState)
            self.buffer.add(currExp)
            saveReward = None
        self.state = nextState

        if done:
            self.reset()

        return saveReward

    def playFree(self, net, device):
        with torch.no_grad():
            torch_state = torch.tensor([self.state], dtype=torch.float32).to(device)
            q_values = net(torch_state)
            print("Q-values:", q_values.cpu().numpy())
            action = q_values.argmax(dim=1).item()

        nextState, reward, terminated, truncated, info = self.env.step(action)
        self.state = nextState

        if terminated or truncated:
            self.reset()

        return terminated or truncated


def calculateLoss(pNet, tNet, buffer, device):
    states, actions, rewards, dones, nextstates, indices, weights = buffer.sample()
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.bool).to(device)
    nextstates = torch.tensor(nextstates, dtype=torch.float32).to(device)

    qActionsTaken = pNet(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    # dont compute any gradients since we dont need to update the target network here

    with torch.no_grad():
        next_q_policy = pNet(nextstates)
        next_actions = next_q_policy.argmax(dim=1)

        next_q_target = tNet(nextstates)
        q_next_states = next_q_target.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)

        q_next_states[dones] = 0.0

    target = rewards + gamma * q_next_states
    loss_per_sample = torch.nn.functional.smooth_l1_loss(
        qActionsTaken, target, reduction="none"
    )

    weights_t = torch.tensor(weights, dtype=torch.float32).to(device)
    loss = (weights_t * loss_per_sample).mean()
    td_errors = (target - qActionsTaken).abs().detach().cpu().numpy()

    return loss, td_errors, indices


def main(defaultMode="train"):
    parser = argparse.ArgumentParser(description="Train or run the Flappy Bird DQN")
    parser.add_argument("--mode", choices=["train", "inference"], default=defaultMode)
    parser.add_argument(
        "--checkpoint", default="flappy_bird_dqn.pth", help="Inference model weights"
    )
    parser.add_argument(
        "--rms",
        default="flappy_bird_normalization.pth",
        help="Inference normalization statistics",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    args = parser.parse_args()
    playFree = args.mode == "inference"

    if playFree:
        env = constructNormalizerHuman(args.rms)
    else:
        env = constructNormalizer()

    writer = None
    try:
        if args.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(args.device)
        buffer = PrioritizedReplayBuffer(replaySize)

        agent = Agent(env, buffer)
        frames = 0
        rewards = []
        accumReward = 0

        policyNet = DQN(env.observation_space.shape, env.action_space.n).to(device)
        if playFree:
            weights = torch.load(
                args.checkpoint, map_location=device, weights_only=True
            )

            policyNet.load_state_dict(weights)
            policyNet.eval()
        else:
            targetNet = DQN(env.observation_space.shape, env.action_space.n).to(device)
            optimizer = torch.optim.Adam(policyNet.parameters(), lr=1e-4)
            writer = SummaryWriter(comment="-flappy-bird-dqn")
        # targetNet.load_state_dict(weights)
        best_reward = None

        print(f"Starting {args.mode} on {device}...")

        while True:
            if playFree:
                if agent.playFree(policyNet, device):
                    break
            else:
                frames += 1

                epsilon = max(minEpsilon, 1 - frames / decayRate)
                reward = agent.step(policyNet, device, epsilon)

                if reward is not None:
                    rewards.append(reward)
                    mean_reward = np.mean(rewards[-100:])

                    if len(rewards) % 10 == 0:
                        print(
                            f"Episode: {len(rewards)} | Frame: {frames} | Reward: {reward:.2f} | Mean Reward (100): {mean_reward:.2f} | Epsilon: {epsilon:.4f}"
                        )

                    if best_reward is None or best_reward < reward:
                        torch.save(policyNet.state_dict(), "best.dat")
                        if best_reward is not None:
                            print(
                                "Best reward updated %.3f -> %.3f"
                                % (best_reward, reward)
                            )

                        best_reward = reward
                        if env.steps > env.warmup:
                            torch.save(env.rms, "flappy_bird_normalization.pth")

                    if mean_reward > rewardCutoff:
                        msg = "done in" + str(frames) + "frames"
                        print(msg)
                        torch.save(policyNet.state_dict(), "solved.dat")
                        torch.save(env.rms, "solved_rms.pth")
                        break

                    writer.add_scalar("epsilon", epsilon, frames)
                    writer.add_scalar(
                        "mean reward over last 100 epsiodes", mean_reward, frames
                    )
                    writer.add_scalar("reward", reward, frames)

                    # Added print statement to verify writing is happening
                    if len(rewards) % 1 == 0:  # Print every episode to confirm logic
                        print(
                            f"Logged to TensorBoard - Frame: {frames} | Reward: {reward}"
                        )

                for tp, p in zip(targetNet.parameters(), policyNet.parameters()):
                    tp.data.copy_(tp.data * (1 - tau) + p.data * tau)

                if not buffer.full():
                    if frames % 200 == 0:
                        print(
                            f"Filling replay buffer: {len(buffer.buffer)} / {replaySize}"
                        )
                    continue

                optimizer.zero_grad()

                loss, td_errors, indices = calculateLoss(
                    policyNet, targetNet, buffer, device
                )
                loss.backward()
                nn.utils.clip_grad_norm_(policyNet.parameters(), 1.0)
                optimizer.step()
                buffer.update_priorities(indices, td_errors)
                if frames % 1000 == 0:
                    print(f"Frame: {frames} | Loss: {loss.item():.6f}")
    finally:
        env.close()
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
