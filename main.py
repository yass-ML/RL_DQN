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

if __name__ == "__main__":
    env = BreakoutWrapper(game="ALE/Breakout-v5", device=device, render_mode=None)
    n_actions = env.env.action_space.n

    model = DQN(n_actions=n_actions,device=device)
    agent = Agent(model=model, 
                  start_eps=1.0, 
                  min_eps=0.1, 
                  nb_warmup=1_000_000, 
                  memory_capacity=140_000, 
                  batch_size=32, 
                  device=device, 
                  learning_rate=0.00025,
                  gamma=0.99)
    
    stats = agent.train(env, training_steps=9_000_000)
    with open("stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    test_env = BreakoutWrapper(game="ALE/Breakout-v5", device=device, render_mode="human")
    test_stats = agent.test(test_env, episodes=5, render=True)
    with open("test_stats.json", "w") as f:
        json.dump(test_stats, f, indent=4)
