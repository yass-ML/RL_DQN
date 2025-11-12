import gymnasium as gym
import numpy as np
from collections import deque
import torch
from PIL import Image

class BreakoutWrapper(gym.Wrapper):
    def __init__(self,game="ALE/Breakout-v5", render_mode="human", frame_skip: int = 4,frame_stack:int=4, device="cpu"):
        env = gym.make(id=game, render_mode=render_mode) if render_mode else gym.make(id=game)
        super().__init__(env)
        self.frame_skip = frame_skip
        self.frame_stack_len = frame_stack
        self.frame_buffer = deque(maxlen=frame_skip)
        self.frame_stack = deque(maxlen=frame_stack)
        for _ in range(frame_stack):
            self.frame_stack.append(torch.zeros((1,1,84,84), dtype=torch.float32).to(device=device))
        self.device = device
        self.lives = env.unwrapped.ale.lives()

    def step(self,action):
        total_reward = 0.0
        done = False
        for i in range(self.frame_skip):
            obs, reward, done, truncated, info = self.env.step(action=action)
            total_reward+=reward

            current_lives = info["lives"]
            if current_lives < self.lives:
                total_reward -=1
                self.lives = current_lives

            self.frame_buffer.append(obs)

            if done or truncated:
                break
        
        # the pointwise maximum of the two last frames in the skip phase make one observation => 
        # we will stack 4 of those observations to make a state for the dqn
        max_frame = np.max(list(self.frame_buffer)[-2:], axis=0) 
        max_frame = self._preprocess(max_frame)

        self.frame_stack.append(max_frame)
        observation = torch.cat(list(self.frame_stack), dim=1) # concatenate along the channel dimension

        total_reward = torch.tensor(total_reward).view(1,-1).float().to(device=self.device)
        done = torch.tensor(done or truncated).view(1,-1).to(device=self.device)

        return observation, total_reward, done, info


    
    def _preprocess(self, obs):
        """
        This method will take a raw frame (obs) from the environment and resize and crop
        """
        downsampling_size = (84,110) # for now, we resize then crop, we'll try directly resizing to 84 x 84 if poor perfs
        img = Image.fromarray(obs)
        img = img.resize(downsampling_size)
        img = img.convert("L") # grayscale
        img = np.array(img)
        cropped = img.shape[0] - 84
        img = img[cropped:,:]
        img = torch.tensor(img)
        img = img.unsqueeze(0).unsqueeze(0) # One additional dimension for the image channel (we'll stack 4 images to make a state), the other for batch size
        img = img / 255.0

        return img.to(device=self.device)

    

    def reset(self, **kwargs):
        self.frame_buffer = deque(maxlen=self.frame_skip)
        self.frame_stack = deque(maxlen=self.frame_stack_len)
        for _ in range(self.frame_stack_len):
            self.frame_stack.append(torch.zeros((1,1,84,84), dtype=torch.float32).to(device=self.device))
        obs, info = self.env.reset(**kwargs)

        self.lives = self.env.unwrapped.ale.lives()

        obs = self._preprocess(obs)
        self.frame_stack.append(obs)
        obs = torch.cat(list(self.frame_stack), dim=1) # concatenate along the channel dimension

        return obs, info


