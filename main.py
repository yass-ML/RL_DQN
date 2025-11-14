import json
import ale_py
import torch
import gymnasium as gym

from dqn.agent import Agent
from dqn.model import DQN
from dqn.environment import BreakoutWrapper

from gymnasium.wrappers import RecordVideo

gym.register_envs(ale_py)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
game_id = "ALE/Breakout-v5"

if __name__ == "__main__":
    game = gym.make(id=game_id)
    n_actions = game.action_space.n

    model = DQN(n_actions=n_actions,device=device)
    agent = Agent(model=model, 
                  start_eps=1.0, 
                  min_eps=0.1, 
                  nb_warmup=1_000_000, 
                  memory_capacity=160_000, 
                  batch_size=32, 
                  device=device, 
                  learning_rate=0.00025,
                  gamma=0.95)
    
    env = BreakoutWrapper(game=game, device=device, zeros_init=agent.zeros_init, crop_region=agent.crop_region)
    
    stats = agent.train(env, training_steps=9_000_000)
    with open("stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    test_env = BreakoutWrapper(game="ALE/Breakout-v5", device=device, render_mode="human", zeros_init=agent.zeros_init, crop_region=agent.crop_region)
    test_stats = agent.test(test_env, episodes=5, render=True)
    with open("test_stats.json", "w") as f:
        json.dump(test_stats, f, indent=4)
