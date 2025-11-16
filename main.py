import json
import ale_py
import torch
import gymnasium as gym

from datetime import datetime
from dqn.agent import Agent
from dqn.model import DQN
from dqn.environment import BreakoutWrapper

from gymnasium.wrappers import RecordVideo

gym.register_envs(ale_py)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
game_id = "ALE/Breakout-v5"

experiment_name = f"DQN_Breakout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
log_dir = f"runs/{experiment_name}"

if __name__ == "__main__":
    game = gym.make(id=game_id)
    n_actions = game.action_space.n

    model = DQN(n_actions=n_actions,device=device)
    agent = Agent(model=model, 
                  start_eps=1.0, 
                  min_eps=0.1, 
                  nb_warmup=1_000_000, 
                  memory_capacity=330_000, 
                  batch_size=32, 
                  device=device, 
                  learning_rate=0.00025,
                  gamma=0.95,
                  log_dir=None)
    
    env = BreakoutWrapper(game=game, device=device, zeros_init=agent.zeros_init, crop_region=agent.crop_region)
    
    stats = agent.train(env, training_steps=9_000_000, eval_episodes=10, eval_interval=50, end_ep_on_life_loss=True)
    with open("stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    # save hyperparameters and settings
    settings = {
        "game_id": game_id,
        "device": device,
        "learning_rate": agent.learning_rate,
        "gamma": agent.gamma,
        "batch_size": agent.batch_size,
        "memory_capacity": agent.memory.capacity,
        "start_eps": agent.start_eps,
        "min_eps": agent.min_eps,
        "nb_warmup": agent.nb_warmup,
        "zeros_init": agent.zeros_init,
        "crop_region": agent.crop_region,
        "use_target_model": agent.use_target_model
    }
    with open("settings.json", "w") as f:
        json.dump(settings, f, indent=4)


    test_env = BreakoutWrapper(game="ALE/Breakout-v5", device=device, render_mode="human", zeros_init=agent.zeros_init, crop_region=agent.crop_region)
    test_stats = agent.test(test_env, episodes=5, render=True)
    with open("test_stats.json", "w") as f:
        json.dump(test_stats, f, indent=4)
