from agent import Agent
from dqn_model import DQN
from environment import BreakoutWrapper
import torch
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import ale_py
gym.register_envs(ale_py)
import json



device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
model_path = "models/run2/latest.pth"

if __name__ == "__main__":
    env = BreakoutWrapper(game="ALE/Breakout-v5", device=device, render_mode=None)
    n_actions = env.env.action_space.n

    model = DQN(n_actions=n_actions,device=device)
    model.load(model_path)

    agent = Agent(model=model,device=device)
    test_stats = agent.test(env, episodes=30, render=False)
    with open("test_stats.json", "w") as f:
        json.dump(test_stats, f, indent=4)

