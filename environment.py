import gymnasium as gym
import cv2
import numpy as np
from collections import deque
import torch


class BreakoutWrapper(gym.Wrapper):
    def __init__(self,env, frame_skip: int = 4, k_stack: int = 4):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.frame_stack = deque(maxlen=k_stack)

    def step(self,action):
        total_reward, obs, info, terminated,truncated = 0.0, None, None, False, False
        for skip in range(self.frame_skip):
            obs, reward, terminated, truncated , info = self.env.step(action)
            
            total_reward+= reward
            if terminated or truncated:
                break
        obs = self._preprocess(obs=obs)
        self.frame_stack.append(obs)
        return (torch.tensor(np.array(self.frame_stack)).to(torch.float32) / 255.0,
                total_reward,
                terminated,
                truncated,
                info)
    
    def _preprocess(self, obs):
        """
        This method will take a raw frame (obs) from the environment and perform those transformations using cv2.
        """
        downsampling_size = (84,110)
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        downsampled_obs = cv2.resize(gray_obs, downsampling_size, interpolation=cv2.INTER_AREA)
        return downsampled_obs[:84,:]
    

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for i in range(self.frame_stack.maxlen):
            self.frame_stack.append(self._preprocess(obs))
        return torch.tensor(np.array(self.frame_stack)).to(torch.float32) / 255.0, info





