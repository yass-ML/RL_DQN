from dqn_model import DQN
from replay_memory import ReplayMemory
from environment import BreakoutWrapper as Breakout
import torch
from torch.optim import RMSprop, Adam
import random
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm



class Agent:
    def __init__(self, 
                 model,
                 device: str = "cpu", 
                 start_eps: float = 1.0, 
                 min_eps: float = 0.1,
                 gamma: float = 0.9,
                 nb_warmup: int = 10000,
                 nb_actions: int | None = None,
                 memory_capacity: int = 20_000,
                 batch_size: int = 32, 
                 learning_rate: float = 0.00025,
                 use_target_model: bool = False,
                 optimizer: torch.optim.Optimizer = RMSprop):
        
        assert nb_actions is not None
        self.nb_actions = nb_actions
        self.nb_warmup = nb_warmup
        self.batch_size = batch_size
        self.lr = learning_rate

        self.memory = ReplayMemory(capacity=memory_capacity, device=device)
        self.model = model
        self.model = self.model.to(device)

        if use_target_model:
            self.target_model = DQN(n_actions=nb_actions,device=device)
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model = self.target_model.eval()
        else:
            self.target_model = None

        self.start_eps, self.eps, self.min_eps = start_eps, start_eps ,min_eps
        self.eps_decay = (min_eps - start_eps) / self.nb_warmup

    
        self.gamma = gamma
        self.nb_actions
        self.optimizer = optimizer(self.model.parameters(), lr = self.lr)

        print(f"Starting: eps = {self.eps} - eps decay = {self.eps_decay}")


    
    def act(self, state: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.eps:
            return torch.randint(self.nb_actions, (1,1))
        else:
            av = self.model(state).detach()
            return torch.argmax(av, dim=1, keepdim=True)
        
        
        
    def train(self, 
              env: Breakout,
              training_steps: int | None = None):
        
        stats = {
            "Returns": [], 
            "AvgReturns": [], 
            "EpsCheckpoint": [],
            "Losses": [],           
            "AvgLosses": [], 
            "Q_Values": [], 
            "Episode_Lengths": [], 
            "Max_Q": [], 
            "Min_Q": [], 
            "Reward_Per_Step": [], 
            "Memory_Utilization": [], 
            "Learning_Steps_Total": 0,
            "Total_Frames": 0
        }

        if not training_steps:
            training_steps = 9 * self.nb_warmup # 10% warmup, 90% training

        episode_length = 0
        n_frame = 0
        pbar = tqdm(total=self.nb_warmup + training_steps)
        ep = 0
        while n_frame < self.nb_warmup + training_steps:
            episode_length = 0
            state, _ = env.reset()
            done  = False
            ep_return = 0
            while not done:
                action = self.act(state)
                stats["Total_Frames"] = n_frame
                n_frame +=1

                next_state,reward,done,info = env.step(action)
                reward = torch.clip(reward, min=-1.0, max=1.0) # reward clipping for training

                if self.eps > self.min_eps:
                    self.eps += self.eps_decay

                self.memory.insert([state,action,reward,done,next_state])
                done = done.item()

                if self.memory.can_sample(batch_size=self.batch_size):

                    state_b, action_b, reward_b, done_b, next_state_b = self.memory.sample(self.batch_size)
                    reward_b = torch.clip(reward_b, min=-1.0, max=1.0) # reward clipping for training
                    q_states_b = self.model(state_b).gather(1, action_b)

                    with torch.no_grad():
                        next_q_states_b = self.target_model(next_state_b) if self.target_model else self.model(next_state_b)

                    next_q_states_b = torch.max(next_q_states_b,dim=-1, keepdim=True)[0]
                    target_b = reward_b + self.gamma * (1 - done_b.float()) * next_q_states_b
                    loss = F.mse_loss(target=target_b, input=q_states_b)

                    stats["Losses"].append(loss.item())
                    stats["Q_Values"].append(q_states_b.mean().item())
                    stats["Max_Q"].append(q_states_b.max().item())
                    stats["Min_Q"].append(q_states_b.min().item())
                    stats["Learning_Steps_Total"] += 1

                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                state = next_state
                ep_return += reward.item()
                episode_length += 1
            
            stats["Returns"].append(ep_return)
            stats["Episode_Lengths"].append(episode_length)
            stats["Reward_Per_Step"].append(ep_return / max(episode_length, 1))
            stats["Memory_Utilization"].append(len(self.memory) / self.memory.capacity)

            average_returns = np.mean(stats["Returns"][-100:]) if stats["Returns"] else None
            avg_loss = np.mean(stats["Losses"][-1000:]) if stats["Losses"] else None
            avg_q_value = np.mean(stats["Q_Values"][-1000:]) if stats["Q_Values"] else None
            avg_episode_length = np.mean(stats["Episode_Lengths"][-100:]) if stats["Episode_Lengths"] else None

            stats["AvgReturns"].append(average_returns)
            stats["AvgLosses"].append(avg_loss)
            stats["EpsCheckpoint"].append(self.eps)

            # Compact postfix with essential info
            pbar.set_postfix({
                "Ep": ep,
                "Ret": f"{ep_return:.2f}",
                "ε": f"{self.eps:.3f}"
            })
            pbar.update(episode_length)
            
            if ep % 10 == 0:
                self.model.save()

                avg_ret_str = f"{average_returns:.2f}" if average_returns is not None else "N/A"
                avg_loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "N/A"
                avg_q_str = f"{avg_q_value:.2f}" if avg_q_value is not None else "N/A"
                avg_eplen_str = f"{avg_episode_length:.1f}" if avg_episode_length is not None else "N/A"
                
                tqdm.write(f"\n{'='*70}")
                tqdm.write(f"Episode {ep} | Return: {ep_return:.2f} | AvgReturn (100 ep): {avg_ret_str}")
                tqdm.write(f" AvgLoss (1000 steps): {avg_loss_str} | AvgQ (1000 steps): {avg_q_str} | AvgEpLen (100 ep): {avg_eplen_str}")
                tqdm.write(f" Epsilon: {self.eps:.3f} | Replay Memory Usage: {stats['Memory_Utilization'][-1]*100:.1f}% | Learning Steps: {stats['Learning_Steps_Total']}")
                tqdm.write(f"{'='*70}\n")

            if  self.target_model and n_frame % 10_000 == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            if ep % 1000 == 0:
                self.model.save(save_path=f"models/dqn_model_ep{ep}.pth")
            
            ep+=1
            
        pbar.close()
        return stats
    
    def test(self, 
             env: Breakout, 
             episodes: int = 5, 
             force_fire: bool = False,
             render: bool = False, 
             save_video: bool = False, 
             video_folder: str = "videos"):
        """
        Test the agent and return comprehensive evaluation metrics.
        
        Args:
            env: Environment to test on
            episodes: Number of test episodes
            render: Whether to render during testing
            save_video: Whether to save video recordings
            video_folder: Folder to save videos
        
        Returns:
            dict: Comprehensive test statistics
        """
        test_stats = {
            "Returns": [],
            "Episode_Lengths": [],
            "Action_Distribution": np.zeros(self.nb_actions),
            "Reward_Per_Step": [],
        }
        
        original_eps = self.eps
        self.eps = 0.05 # same eps as in paper during evaluation
        self.model.eval()
        
        for ep in range(episodes):
            state, info = env.reset()
            done = False
            ep_return = 0
            ep_length = 0
            ep_q_values = []
            current_lives = info.get('lives', 5)
            steps_since_reward = 0
            max_idle_steps = 1000  # Timeout if no reward for 1000 steps
            
            while not done:
                if render:
                    env.render()
                
                # Check if we lost a life - force FIRE to restart
                if force_fire and'lives' in info and info['lives'] < current_lives:
                    action = torch.tensor([[1]], dtype=torch.long)  # FIRE action
                    current_lives = info['lives']
                else:
                    with torch.no_grad():
                        action = self.act(state)
                
                test_stats["Action_Distribution"][action.item()] += 1
                
                next_state, reward, done, info = env.step(action)
                done = done.item()
                state = next_state
                ep_return += reward.item()
                ep_length += 1
                
                # Timeout check: if idle for too long, break
                if reward.item() != 0:
                    steps_since_reward = 0
                else:
                    steps_since_reward += 1
                    if steps_since_reward >= max_idle_steps:
                        print(f"  Episode {ep} timed out after {max_idle_steps} idle steps")
                        break
            
            test_stats["Returns"].append(ep_return)
            test_stats["Episode_Lengths"].append(ep_length)
            test_stats["Reward_Per_Step"].append(ep_return / max(ep_length, 1))
            
            print(f"Test Episode {ep} - Return: {ep_return:.2f} - Length: {ep_length}")
        
        # Restore original epsilon
        self.eps = original_eps
        self.model.train()
        
        # Calculate summary statistics
        test_stats["Average_Return"] = np.mean(test_stats["Returns"])
        test_stats["Std_Return"] = np.std(test_stats["Returns"])
        test_stats["Average_Episode_Length"] = np.mean(test_stats["Episode_Lengths"])
        test_stats["Action_Distribution"] = test_stats["Action_Distribution"] / test_stats["Action_Distribution"].sum()
        
        print(f"\n{"="*60}")
        print(f"Test Results over {episodes} episodes:")
        print(f"  Average Return: {test_stats["Average_Return"]:.2f} ± {test_stats["Std_Return"]:.2f}")
        print(f"  Average Episode Length: {test_stats["Average_Episode_Length"]:.1f}")
        print(f"  Action Distribution: {test_stats["Action_Distribution"]}")
        print(f"{"="*60}\n")
        
        return test_stats










