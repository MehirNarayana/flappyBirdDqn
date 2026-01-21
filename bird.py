from Wrappers import construct
from dqn import DQN

import argparse
import time
import numpy as np
from collections import namedtuple  
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter


gamma = 0.99
epsilon = 0.01

decayRate = 150000
minEpsilon = 0.01
batchSize  = 32
replaySize = 50000
updateRate = 10000
rewardCutoff = 100
Experience = namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])

playFree = False
class ReplayBuffer():
    
    def __init__(self):
        self.buffer = deque(maxlen=replaySize)
    
    def reset(self):
        self.buffer.clear()
         
    
    def add(self, exp):
        self.buffer.append(exp)

    def full(self):
        return len(self.buffer) == replaySize
        

    def sample(self):
        indices = np.random.choice(len(self.buffer), batchSize,
                                   replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)






class RunningMeanStd:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x):  # x: np.array shape (N, features) or (features,)
        x = np.atleast_2d(x).astype(np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        # combine stats
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta**2) * (self.count * batch_count / tot_count)
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x, clip=5.0):
        x = np.array(x, dtype=np.float64)
        std = np.sqrt(self.var + 1e-8)
        return np.clip((x - self.mean) / std, -clip, clip)

class Agent():
    def __init__(self, env, buffer):
        self.env = env
        self.buffer = buffer
        self.reset()
        
    
    def reset(self):
        self.state, _ = self.env.reset() # Fixed: use self.env instead of global env
        self.episode_reward = 0

    def step(self, net, device, epsilon):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
            
        else:
            torch_state = torch.tensor(np.array([self.state]), dtype=torch.float32).to(device)
            pred = net(torch_state)
            _, action = torch.max(pred, dim=1)
            action = int(action.item())
            
        
        nextState, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward +=reward 
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
    
    def playFree(self, net):
        with torch.no_grad():
            torch_state = torch.tensor(np.array([self.state]), dtype=torch.float32).to(device)
            pred = net(torch_state)
            _, action = torch.max(pred, dim=1)
            action = int(action.item())
        nextState, reward, terminated, truncated, info = self.env.step(action)
        return terminated or truncated



def calculateLoss(pNet, tNet, buffer, device):
    states, actions, rewards, dones, nextstates = buffer.sample()
    states  = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.bool).to(device)
    nextstates = torch.tensor(nextstates, dtype=torch.float32).to(device)
    #no need to add reward because this is the neural net is estimating the q value for the state already
    # qActionsTaken = pNet(states).gather(    1, actions.unsqueeze(-1)).squeeze(-1)
    # #dont compute any gradients since we dont need to update the target network here
    
    # with torch.no_grad():
        
    #     next_q_policy = pNet(nextstates) 
    #     next_actions = next_q_policy.argmax(dim=1)
        
    #     next_q_target = tNet(nextstates) 
    #     q_next_states = next_q_target.gather(1, next_actions.unsqueeze(-1)).squeeze(-1) 
        
    #     q_next_states[dones] = 0.0
        
        
    #     # q_next_states = tNet(nextstates).to(device).max(1).values

    #     # # for i in range(len(dones)):
    #     # #     if dones[i]:
    #     # #         q_next_states[i] = 0
    #     # #this line basically replaced the line above, but is more optimized
    #     # q_next_states[dones] = 0.0
    #     # #freeze backpropogation on this tensor because its not being used

    #     # q_next_states.detach()

    qActionsTaken = pNet(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    #dont compute any gradients since we dont need to update the target network here
    
    with torch.no_grad():
        
        next_q_policy = pNet(nextstates) 
        next_actions = next_q_policy.argmax(dim=1)
        
        next_q_target = tNet(nextstates) 
        q_next_states = next_q_target.gather(1, next_actions.unsqueeze(-1)).squeeze(-1) 
        
        q_next_states[dones] = 0.0

        

    target = rewards + gamma * q_next_states
    return torch.nn.functional.smooth_l1_loss(qActionsTaken, target)
    



             







if __name__ == "__main__":
    
    
    env = construct()
    rms = RunningMeanStd(shape=(12,)) 
    device = torch.device("mps")
    buffer = ReplayBuffer()
    writer = SummaryWriter(comment="-flappy-bird-dqn")
    agent = Agent(env, buffer)
    frames = 0
    rewards = []
    accumReward = 0
    
    policyNet = DQN(env.observation_space.shape, env.action_space.n).to(device)
    targetNet = DQN(env.observation_space.shape, env.action_space.n).to(device)
    if playFree:
        weights = torch.load("best_119.dat", map_location=device)

        policyNet.load_state_dict(weights)
    #targetNet.load_state_dict(weights)
    optimizer = torch.optim.Adam(policyNet.parameters(), lr=1e-4)
    best_reward = None

    print(f"Starting training on {device}...")

    while True:
        if playFree:
            if agent.playFree(policyNet):
                break
        else:
            frames+=1
            
            
            



            
            epsilon = max(minEpsilon, 1 - frames/decayRate)
            reward = agent.step(policyNet, device, epsilon)
            
            
            
            if reward is not None:
                
                rewards.append(reward)
                mean_reward = np.mean(rewards[-100:])

                if len(rewards) % 10 == 0:
                    print(f"Episode: {len(rewards)} | Frame: {frames} | Reward: {reward:.2f} | Mean Reward (100): {mean_reward:.2f} | Epsilon: {epsilon:.4f}")

                if best_reward is None or best_reward < reward:
                    torch.save(policyNet.state_dict(),
                            "best_%.0f.dat" % reward)
                    if best_reward is not None:
                        print("Best reward updated %.3f -> %.3f" % (
                            best_reward, reward))
                    best_reward = reward
                
                if mean_reward>rewardCutoff:
                    
                    msg = "done in" + str(frames) + "frames"
                    print(msg)
                    torch.save(policyNet.state_dict(), "solved.dat")
                    break
                    
                    
                writer.add_scalar("epsilon", epsilon, frames)
                writer.add_scalar("mean reward over last 100 epsiodes", mean_reward, frames)
                writer.add_scalar("reward", reward, frames)
                
                # Added print statement to verify writing is happening
                if len(rewards) % 1 == 0: # Print every episode to confirm logic
                    print(f"Logged to TensorBoard - Frame: {frames} | Reward: {reward}")
                
            


            if frames%updateRate == 0:
                #update model weights
                targetNet.load_state_dict(policyNet.state_dict()) 
                print(f"Frame {frames}: Target network synced")


            if not buffer.full():
                if frames % 200 == 0:
                    print(f"Filling replay buffer: {len(buffer.buffer)} / {replaySize}")
                continue
        
            
            
            optimizer.zero_grad()

            loss = calculateLoss(policyNet, targetNet, buffer, device)
            loss.backward()
            nn.utils.clip_grad_norm_(policyNet.parameters(), 1.0)
            optimizer.step()

            if frames % 1000 == 0:
                print(f"Frame: {frames} | Loss: {loss.item():.6f}")

            



