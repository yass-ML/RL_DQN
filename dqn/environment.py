import gymnasium as gym
import numpy as np
import torch
from collections import deque
from PIL import Image

class BreakoutWrapper(gym.Wrapper):
    def __init__(self,game: str | gym.Env = "ALE/Breakout-v5", 
                 render_mode: str | None = None, 
                 frame_skip: int = 4,
                 device: str = "cpu", 
                 lives_penalty: bool = False,
                 max_frame: bool = False,
                 crop_region: tuple | None = (18,102)):
        
        if isinstance(game, gym.Env):
            env = game
            env.frameskip = 1
            env.repeat_action_probability = 0.0
        else:
            env = gym.make(id=game, render_mode=render_mode, frameskip=1, repeat_action_probability=0.0)
        super().__init__(env)
        self.frame_skip = frame_skip
        self.frame_buffer = deque(maxlen=frame_skip)
        # flag to perform or not a penalty on reward when losing a life
        self.lives_penalty = lives_penalty
        # flag to perform or not pointwise max of the last two frames in the skipping process: From paper Nature on DQN (2015)
        self.max_frame = max_frame
        self.dtype = np.uint8

        self.device = device
        self.lives = env.unwrapped.ale.lives()
        self.crop_region = crop_region

    def step(self,action):
        total_reward = 0.0
        done = False
        for i in range(self.frame_skip):
            obs, reward, done, truncated, info = self.env.step(action=action)
            total_reward+=reward

            if self.lives_penalty:
                current_lives = info["lives"]
                if current_lives < self.lives:
                    total_reward -=1
                    self.lives = current_lives

            self.frame_buffer.append(obs)

            if done or truncated:
                break
        

        if self.max_frame:
            # the pointwise maximum of the two last frames in the skip phase make one observation
            observation = np.max(list(self.frame_buffer)[-2:], axis=0) 
        else:
            observation = self.frame_buffer[-1]

        # Preprocess and return SINGLE frame (no stacking here)
        observation = self._preprocess(observation)  # Shape: (84, 84) as uint8

        total_reward = float(total_reward)
        done = bool(done or truncated)

        return observation, total_reward, done, info


    
    def _preprocess(self, obs):
        """
        This method will take a raw frame (obs) from the environment and resize and crop.
        Returns a single frame as numpy array (84, 84) uint8.
        """
        downsampling_size = (84,110)
        img = Image.fromarray(obs)
        img = img.resize(downsampling_size)
        img = img.convert("L") # grayscale
        img = np.array(img, dtype=self.dtype)
        if self.crop_region:
            lower_bound,upper_bound = self.crop_region
            img = img[lower_bound:upper_bound,:]
        else:
            cropped = img.shape[0] - 84
            img = img[cropped:,:]
        
        return img  # Shape: (84, 84) uint8



    

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()

        self.frame_buffer = deque(maxlen=self.frame_skip)

        # Preprocess and return SINGLE frame (no stacking here)
        obs = self._preprocess(obs)  # Shape: (84, 84) as uint8

        return obs, info


