from agent import DQNAgent
from environment import BreakoutWrapper
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from torch.optim import RMSprop
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import os

import ale_py
gym.register_envs(ale_py)


# (Place this in main.py, outside the main() function)

def evaluate(agent:DQNAgent, env: BreakoutWrapper, num_episodes=10, fixed_epsilon=0.05):
    """
    Runs the agent for a fixed number of episodes with a fixed epsilon
    and NO learning.
    """
    print("--- Starting Evaluation ---")
    agent.model.eval() # Set model to evaluation mode (good practice)
    total_rewards = []

    for i in range(num_episodes):
        current_state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            # 1. Get action with the fixed, small epsilon
            # We temporarily set the agent's epsilon for this step
            original_eps = agent.eps
            agent.eps = fixed_epsilon 
            action = agent.act(current_state)
            agent.eps = original_eps # (or just pass eps into act)

            # 2. Step the environment (same wrapper, same frame skip/stack)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            current_state = obs

        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {i+1}/{num_episodes}, Reward: {episode_reward}")

    agent.model.train() # Set model back to training mode
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"--- Evaluation Finished. Average Reward: {avg_reward} ---")
    return avg_reward, total_rewards



def main():
    print("Hello from tp4!")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    print(f"Device is {device}")

    # Create output directory for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    EVALUATION_INTERVAL = 25000
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
    
    # Track evaluation metrics
    evaluation_frames = []
    evaluation_avg_rewards = []
    evaluation_all_rewards = []
    
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

        if (current_frame % EVALUATION_INTERVAL) < breakout_env.frame_skip:
            # --- This is the key change ---
            # We are *pausing* training to run evaluation
            avg_score, eval_rewards = evaluate(agent, breakout_env, num_episodes=10, fixed_epsilon=0.05)
            print(f"Evaluated on 10 real episodes - avg_score: {avg_score}")
            
            # Store evaluation metrics
            evaluation_frames.append(current_frame)
            evaluation_avg_rewards.append(avg_score)
            evaluation_all_rewards.append(eval_rewards)
        
        # --- End of Step ---
        current_state = obs
        if done:
            # --- This is the key change ---
            # Log the completed episode
            print(f"Episode: {current_episode}, Episode Reward: {episode_reward} - eps = {agent.eps}")
            
            # Reset the environment for the *next* episode
            current_state, _ = breakout_env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            current_episode+=1
        else:
            current_state = obs

    pbar.close()
    print("Training finished.")
    
    # ========== SAVE MODEL ==========
    print("\n" + "="*50)
    print("Saving trained model...")
    model_path = os.path.join(output_dir, "dqn_model.pth")
    torch.save({
        'model_state_dict': agent.model.state_dict(),
        'target_model_state_dict': agent.target_model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'total_frames': current_frame,
        'total_episodes': current_episode,
    }, model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # ========== SAVE TRAINING METRICS ==========
    print("\n" + "="*50)
    print("Saving training metrics...")
    
    # Create training episode dataframe
    train_df = pd.DataFrame({
        'episode': range(1, len(episode_rewards) + 1),
        'reward': episode_rewards
    })
    
    # Calculate rolling statistics
    window_size = min(100, len(episode_rewards))
    if window_size > 0:
        train_df['rolling_mean_100'] = train_df['reward'].rolling(window=window_size, min_periods=1).mean()
        train_df['rolling_std_100'] = train_df['reward'].rolling(window=window_size, min_periods=1).std()
    
    # Save training metrics to CSV
    train_csv_path = os.path.join(output_dir, "training_episodes.csv")
    train_df.to_csv(train_csv_path, index=False)
    print(f"✓ Training episodes saved to: {train_csv_path}")
    
    # ========== SAVE EVALUATION METRICS ==========
    print("\n" + "="*50)
    print("Saving evaluation metrics...")
    
    # Create evaluation dataframe
    eval_data = []
    for i, (frame, avg_reward, rewards) in enumerate(zip(evaluation_frames, evaluation_avg_rewards, evaluation_all_rewards)):
        for j, reward in enumerate(rewards):
            eval_data.append({
                'evaluation_step': i + 1,
                'frame': frame,
                'episode_in_eval': j + 1,
                'reward': reward,
                'avg_reward_this_eval': avg_reward
            })
    
    if eval_data:
        eval_df = pd.DataFrame(eval_data)
        eval_csv_path = os.path.join(output_dir, "evaluation_episodes.csv")
        eval_df.to_csv(eval_csv_path, index=False)
        print(f"✓ Evaluation episodes saved to: {eval_csv_path}")
        
        # Summary evaluation metrics
        eval_summary_df = pd.DataFrame({
            'evaluation_step': range(1, len(evaluation_avg_rewards) + 1),
            'frame': evaluation_frames,
            'avg_reward': evaluation_avg_rewards,
            'std_reward': [np.std(rewards) for rewards in evaluation_all_rewards],
            'min_reward': [np.min(rewards) for rewards in evaluation_all_rewards],
            'max_reward': [np.max(rewards) for rewards in evaluation_all_rewards]
        })
        eval_summary_csv_path = os.path.join(output_dir, "evaluation_summary.csv")
        eval_summary_df.to_csv(eval_summary_csv_path, index=False)
        print(f"✓ Evaluation summary saved to: {eval_summary_csv_path}")
    
    # ========== GENERATE GRAPHS ==========
    print("\n" + "="*50)
    print("Generating graphs...")
    
    # 1. Training Episode Rewards
    plt.figure(figsize=(12, 6))
    plt.plot(train_df['episode'], train_df['reward'], alpha=0.3, label='Episode Reward')
    if 'rolling_mean_100' in train_df.columns:
        plt.plot(train_df['episode'], train_df['rolling_mean_100'], label='Rolling Mean (100 episodes)', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    train_plot_path = os.path.join(output_dir, "training_rewards.png")
    plt.savefig(train_plot_path, dpi=300)
    plt.close()
    print(f"✓ Training rewards plot saved to: {train_plot_path}")
    
    # 2. Evaluation Average Rewards
    if eval_data:
        plt.figure(figsize=(12, 6))
        plt.plot(eval_summary_df['frame'], eval_summary_df['avg_reward'], marker='o', linewidth=2, markersize=6)
        plt.fill_between(eval_summary_df['frame'], 
                         eval_summary_df['avg_reward'] - eval_summary_df['std_reward'],
                         eval_summary_df['avg_reward'] + eval_summary_df['std_reward'],
                         alpha=0.3)
        plt.xlabel('Training Frames')
        plt.ylabel('Average Reward')
        plt.title('Evaluation Performance Over Training')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        eval_plot_path = os.path.join(output_dir, "evaluation_rewards.png")
        plt.savefig(eval_plot_path, dpi=300)
        plt.close()
        print(f"✓ Evaluation rewards plot saved to: {eval_plot_path}")
        
        # 3. Combined plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top: Training rewards
        ax1.plot(train_df['episode'], train_df['reward'], alpha=0.3, label='Episode Reward')
        if 'rolling_mean_100' in train_df.columns:
            ax1.plot(train_df['episode'], train_df['rolling_mean_100'], label='Rolling Mean (100 episodes)', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom: Evaluation rewards
        ax2.plot(eval_summary_df['frame'], eval_summary_df['avg_reward'], marker='o', linewidth=2, markersize=6)
        ax2.fill_between(eval_summary_df['frame'], 
                         eval_summary_df['avg_reward'] - eval_summary_df['std_reward'],
                         eval_summary_df['avg_reward'] + eval_summary_df['std_reward'],
                         alpha=0.3)
        ax2.set_xlabel('Training Frames')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Evaluation Performance Over Training')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        combined_plot_path = os.path.join(output_dir, "combined_metrics.png")
        plt.savefig(combined_plot_path, dpi=300)
        plt.close()
        print(f"✓ Combined metrics plot saved to: {combined_plot_path}")
    
    # ========== GENERATE SUMMARY TABLE ==========
    print("\n" + "="*50)
    print("Generating summary table...")
    
    summary_stats = {
        'Metric': [
            'Total Training Frames',
            'Total Training Episodes',
            'Mean Training Reward',
            'Std Training Reward',
            'Max Training Reward',
            'Min Training Reward',
            'Final 100 Episodes Mean Reward',
        ],
        'Value': [
            current_frame,
            current_episode - 1,
            np.mean(episode_rewards) if episode_rewards else 0,
            np.std(episode_rewards) if episode_rewards else 0,
            np.max(episode_rewards) if episode_rewards else 0,
            np.min(episode_rewards) if episode_rewards else 0,
            np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards) if episode_rewards else 0,
        ]
    }
    
    if eval_data:
        summary_stats['Metric'].extend([
            'Number of Evaluations',
            'Final Evaluation Avg Reward',
            'Best Evaluation Avg Reward',
        ])
        summary_stats['Value'].extend([
            len(evaluation_avg_rewards),
            evaluation_avg_rewards[-1] if evaluation_avg_rewards else 0,
            np.max(evaluation_avg_rewards) if evaluation_avg_rewards else 0,
        ])
    
    summary_df = pd.DataFrame(summary_stats)
    summary_table_path = os.path.join(output_dir, "summary_statistics.csv")
    summary_df.to_csv(summary_table_path, index=False)
    print(f"✓ Summary statistics saved to: {summary_table_path}")
    
    # Print summary to console
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*50)
    print(f"✓ All results saved to: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
