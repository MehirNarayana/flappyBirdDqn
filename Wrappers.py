import cv2
import gymnasium as gym
import gymnasium.spaces
import numpy as np

import flappy_bird_gymnasium
from collections import deque
from gymnasium.wrappers import ResizeObservation


from gymnasium.wrappers import FrameStackObservation






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



   

class StateNormalizer(gym.ObservationWrapper):
    """Keeps all features (no reduction). Always uses running mean/std normalization."""
    def __init__(self, env, warmup=20000):
        super().__init__(env)
        self.rms = RunningMeanStd(shape=(12,))
        low = -np.inf * np.ones((12,), dtype=np.float32)
        high = np.inf * np.ones((12,), dtype=np.float32)
        self.steps = 0 
        self.warmup = warmup
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        obs = np.array(obs, dtype=np.float32)
        normalized = self.rms.normalize(obs)   # normalize using previous stats
        if self.steps > self.warmup: 
            self.rms.update(obs)
        self.steps+=1                  # then incorporate this obs into stats
        return normalized
    





class StateMinMaxNormalizer(gym.ObservationWrapper):
    """
    Drops features [0,1,2,11] and min-max normalizes the remaining 8 features to [0,1].
    Kept features (original indices): [3,4,5,6,7,8,9,10]
    """
    def __init__(self, env):
        super().__init__(env)
        width = 288.0
        height = 512.0
        max_v = 10.0
        min_v = -9

        mins = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, min_v], dtype=np.float32)
        maxs = np.array([width, height, height, width, height, height, height, max_v], dtype=np.float32)

        self._mins = mins
        self._maxs = maxs
        low = np.zeros((8,), dtype=np.float32)
        high = np.ones((8,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        obs = np.array(obs, dtype=np.float32)
        kept = obs[[3,4,5,6,7,8,9,10]].astype(np.float32)
        denom = (self._maxs - self._mins)
        denom[denom == 0.0] = 1.0
        scaled = (kept - self._mins) / denom
        return np.clip(scaled, 0.0, 1.0).astype(np.float32)




    
def construct():
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    #env = StateNormalizer(env)
    #env = StateMinMaxNormalizer(env)
    env.reset()
    
    print(env.observation_space.shape)
    
    return env
    

    #print(env.res])
    
    
    
    
    
    
    #return new_state






