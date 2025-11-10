from agent import DQNAgent
from environment import BreakoutWrapper
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from torch.optim import RMSprop
import torch
from tqdm import tqdm

import ale_py
gym.register_envs(ale_py)





def main():
    print("Hello from tp4!")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    print(f"Device is {device}")




    total_frames = 10_000_000
    annealing_frames = 1_000_000
    start_eps = 1.0
    end_eps = 0.1
    current_frame = 0
    current_episode = 1

    env = gym.make("ALE/Breakout-v5")
    breakout_env = BreakoutWrapper(env=env,frame_skip=4,k_stack=4)
    agent = DQNAgent(eps=start_eps, gamma=0.9, n_actions=env.action_space.n, batch_size=32, optimizer=RMSprop)

    current_state, _ = breakout_env.reset()
    episode_reward = 0.0
    episode_rewards = []
    pbar = tqdm(total=total_frames, initial=current_frame)

    while current_frame < total_frames:
        if current_frame >= total_frames:
            print("Finishing training")
            break

        # --- Update frame and Epsilon ---
        current_frame+=breakout_env.frame_skip
        agent.eps = start_eps + min(1.0, current_frame / annealing_frames) * (end_eps - start_eps)
        pbar.update(breakout_env.frame_skip)

        # --- Act ---
        action = agent.act(current_state)
        
        # --- Step ---
        obs, total_reward, terminated,truncated,info = breakout_env.step(action=action)
        episode_reward +=total_reward

        # --- Store & Learn ---
        done = terminated or truncated
        transition = current_state,action,total_reward,obs, done
        agent.memory.store(transition=transition)
        agent.learn()
        
        # --- End of Step ---
        current_state = obs
        if done:
            # --- This is the key change ---
            # Log the completed episode
            print(f"Episode: {current_episode}, Episode Reward: {episode_reward}")
            
            # Reset the environment for the *next* episode
            current_state, _ = breakout_env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            current_episode+=1
        else:
            current_state = obs

    pbar.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
