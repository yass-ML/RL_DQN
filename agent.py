from dqn_model import DQN
from replay_memory import ReplayMemory
import torch
from torch.optim import RMSprop, Adam
import random
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class DQNAgent():
    def __init__(self, eps,gamma, n_actions,mem_capacity=1000000,batch_size=32, optimizer=RMSprop,device="cpu"):
        self.memory = ReplayMemory(capacity=mem_capacity)
        self.model = DQN(n_actions=n_actions,device=device)
        self.target_model = DQN(n_actions=n_actions,device=device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.n_actions = n_actions
        self.eps = eps
        self.gamma = gamma
        self.batch_size = batch_size
        self.optimizer = optimizer(self.model.parameters())

        

    def act(self,state):
        if random.random() <= self.eps:
            action = random.sample(range(self.n_actions), 1)[0]
        else:
            action_probs = self.model(state.unsqueeze(0))
            action = torch.argmax(action_probs, dim=1).item()

        return action
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            print(f"Not enough replay memory to learn: {len(self.memory)} < {self.batch_size}")
            return #Not enough memory to learn
        
        minibatch = self.memory.sample(batch_size=self.batch_size)

        state_batch = torch.stack([t[0] for t in minibatch], dim=0)
        actions_batch = torch.tensor([t[1] for t in minibatch], dtype=torch.long)
        rewards_batch = torch.tensor([t[2] for t in minibatch], dtype=torch.float32)
        next_state_batch = torch.stack([t[3] for t in minibatch], dim=0)
        done_batch = torch.tensor([t[4] for t in minibatch], dtype=torch.float32)

        # Update the i-1 target model every learning step i
        self.target_model.load_state_dict(self.model.state_dict())

        with torch.no_grad():
            next_q_values = self.target_model(next_state_batch)
            next_max_q, _ = torch.max(next_q_values,dim=1)
            y = rewards_batch + self.gamma * next_max_q * (1 - done_batch)

        q_values = self.model(state_batch)
        predicted_q_values = torch.gather(q_values,1,index=actions_batch.unsqueeze(1)).squeeze(1)

        loss = F.mse_loss(input = predicted_q_values,target=y)

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()





class Agent:
    def __init__(self,model, device="cpu", start_eps=1.0, min_eps=0.1, nb_warmup=10000,nb_actions=None,memory_capacity=20_000,
                 batch_size=32, learning_rate=0.00025):
        
        assert nb_actions is not None
        self.nb_actions = nb_actions
        self.nb_warmup = nb_warmup
        self.batch_size = batch_size
        self.lr = learning_rate

        self.memory = ReplayMemory(capacity=memory_capacity, device=device)
        self.model = model
        self.model = self.model.to(device)

        self.target_model = DQN(n_actions=nb_actions,device=device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model = self.target_model.eval()

        self.start_eps, self.eps, self.min_eps = start_eps, start_eps ,min_eps
        self.eps_decay = (min_eps - start_eps) / self.nb_warmup

    
        self.gamma = 0.9
        self.nb_actions
        self.optimizer = Adam(self.model.parameters(), lr = self.lr)

        print(f"Starting: eps = {self.eps} - eps decay = {self.eps_decay}")


    
    def act(self,state):
        if torch.rand(1) < self.eps:
            return torch.randint(self.nb_actions, (1,1))
        else:
            av = self.model(state).detach()
            return torch.argmax(av, dim=1, keepdim=True)
        
        
        
    def train(self, env, epochs, training_steps=None):
        stats = {
            "Returns": [], 
            "AvgReturns": [], 
            "EpsCheckpoint": [],
            "Losses": [],  # NEW: Track TD loss
            "AvgLosses": [],  # NEW: Average loss every N episodes
            "Q_Values": [],  # NEW: Average Q-values
            "Episode_Lengths": [],  # NEW: Steps per episode
            "Max_Q": [],  # NEW: Max Q-value seen
            "Min_Q": [],  # NEW: Min Q-value seen
            "Reward_Per_Step": [],  # NEW: Reward efficiency
            "Memory_Utilization": [],  # NEW: % of memory used
            "Learning_Steps_Total": 0,  # NEW: Total optimization steps
            "Total_Frames": 0
        }

        if not training_steps or  training_steps < self.nb_warmup:
            raise ValueError(f"Training steps {training_steps} should be higher than warmup steps {self.nb_warmup} ")

        episode_length = 0
        n_frame = 0
        pbar = tqdm(total=self.nb_warmup + training_steps)
        ep = 0
        while n_frame < self.nb_warmup + training_steps:
            episode_length = 0
            state, info = env.reset()
            done  = False
            ep_return = 0
            while not done:
                action = self.act(state)
                stats["Total_Frames"] +=1
                n_frame +=1
                pbar.update(1)

                next_state,reward,done,info = env.step(action)
                reward = torch.clip(reward, min=-1.0, max=1.0) # reward clipping for training

                if self.eps > self.min_eps:
                    self.eps += self.eps_decay

                self.memory.insert([state,action,reward,done,next_state])

                if self.memory.can_sample(batch_size=self.batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = self.memory.sample(self.batch_size)
                    reward_b = torch.clip(reward_b, min=-1.0, max=1.0) # reward clipping for training

                    q_states_b = self.model(state_b).gather(1, action_b)
                    next_q_states_b = self.target_model(next_state_b)
                    next_q_states_b = torch.max(next_q_states_b,dim=-1, keepdim=True)[0]
                    target_b = reward_b + ~done_b* self.gamma* next_q_states_b
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

            # if self.eps > self.min_eps:
            #     self.eps += self.eps_decay
            
            if ep % 10 == 0:
                self.model.save()
                print(" ")
                average_returns = np.mean(stats["Returns"][-100:])
                avg_loss = np.mean(stats["Losses"][-1000:]) if stats["Losses"] else 0  # NEW
                avg_q_value = np.mean(stats["Q_Values"][-1000:]) if stats["Q_Values"] else 0  # NEW
                avg_episode_length = np.mean(stats["Episode_Lengths"][-100:])  # NEW

                stats["AvgReturns"].append(average_returns)
                stats["AvgLosses"].append(avg_loss)  # NEW
                stats["EpsCheckpoint"].append(self.eps)

                if len(stats["Returns"]) > 100:
                    print(f"Episode {ep} - Return: {ep_return:.2f} - AvgReturn (last 100): {average_returns:.2f}")
                    print(f"  AvgLoss: {avg_loss:.4f} - AvgQ: {avg_q_value:.2f} - AvgEpLen: {avg_episode_length:.1f}")
                    print(f"  Eps: {self.eps:.3f} - Memory: {stats['Memory_Utilization'][-1]*100:.1f}% - Steps: {stats['Learning_Steps_Total']}")
                else:
                    print(f"Episode {ep} - Return: {ep_return:.2f} - Eps: {self.eps:.3f}")

            if  n_frame % 10_000 == 0:   #ep % 100 == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            if ep % 1000 == 0:
                self.model.save(save_path=f"models/dqn_model_ep{ep}.pth")
            
            ep+=1
            
        pbar.close()
        return stats
    
    def test(self, env, episodes=5, render=False, save_video=False, video_folder="videos"):
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
            "Q_Values": [],
            "Action_Distribution": np.zeros(self.nb_actions),
            "Max_Q_Per_Episode": [],
            "Min_Q_Per_Episode": [],
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
            
            while not done:
                if render:
                    env.render()
                
                # Get Q-values and action
                with torch.no_grad():
                    q_vals = self.model(state)
                    action = torch.argmax(q_vals, dim=1, keepdim=True)
                    
                ep_q_values.append(q_vals.max().item())
                test_stats["Action_Distribution"][action.item()] += 1
                
                next_state, reward, done, info = env.step(action)
                state = next_state
                ep_return += reward.item()
                ep_length += 1
            
            test_stats["Returns"].append(ep_return)
            test_stats["Episode_Lengths"].append(ep_length)
            test_stats["Q_Values"].extend(ep_q_values)
            test_stats["Max_Q_Per_Episode"].append(max(ep_q_values))
            test_stats["Min_Q_Per_Episode"].append(min(ep_q_values))
            test_stats["Reward_Per_Step"].append(ep_return / max(ep_length, 1))
            
            print(f"Test Episode {ep} - Return: {ep_return:.2f} - Length: {ep_length} - Avg Q: {np.mean(ep_q_values):.2f}")
        
        # Restore original epsilon
        self.eps = original_eps
        self.model.train()
        
        # Calculate summary statistics
        test_stats["Average_Return"] = np.mean(test_stats["Returns"])
        test_stats["Std_Return"] = np.std(test_stats["Returns"])
        test_stats["Average_Episode_Length"] = np.mean(test_stats["Episode_Lengths"])
        test_stats["Average_Q_Value"] = np.mean(test_stats["Q_Values"])
        test_stats["Action_Distribution"] = test_stats["Action_Distribution"] / test_stats["Action_Distribution"].sum()
        
        print(f"\n{'='*60}")
        print(f"Test Results over {episodes} episodes:")
        print(f"  Average Return: {test_stats['Average_Return']:.2f} ± {test_stats['Std_Return']:.2f}")
        print(f"  Average Episode Length: {test_stats['Average_Episode_Length']:.1f}")
        print(f"  Average Q-Value: {test_stats['Average_Q_Value']:.2f}")
        print(f"  Action Distribution: {test_stats['Action_Distribution']}")
        print(f"{'='*60}\n")
        
        return test_stats










