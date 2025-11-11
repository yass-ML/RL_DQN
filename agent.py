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
    def __init__(self,model, device="cpu", start_eps=1.0, min_eps=0.1, nb_warmup=10000,nb_actions=None,memory_capacity=10000,
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
        
        
        
    def train(self, env, epochs):
        stats = {"Returns": [], "AvgReturns": [], "EpsCheckpoint": []}



        for ep in tqdm(range(epochs)):
            state, info = env.reset()
            done  = False
            ep_return = 0
            while not done:
                action = self.act(state)

                next_state,reward,done,info = env.step(action)

                self.memory.insert([state,action,reward,done,next_state])

                if self.memory.can_sample(batch_size=self.batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = self.memory.sample(self.batch_size)
                    q_states_b = self.model(state_b).gather(1, action_b)
                    next_q_states_b = self.target_model(next_state_b)
                    next_q_states_b = torch.max(next_q_states_b,dim=-1, keepdim=True)[0]
                    target_b = reward_b + ~done_b* self.gamma* next_q_states_b
                    loss = F.mse_loss(target=target_b, input=q_states_b)

                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                state = next_state
                ep_return += reward.item()
            
            stats["Returns"].append(ep_return)
            if self.eps > self.min_eps:
                self.eps += self.eps_decay
            
            if ep % 10 == 0:
                self.model.save()
                print(" ")
                average_returns = np.mean(stats["Returns"][-100:])
                stats["AvgReturns"].append(average_returns)
                stats["EpsCheckpoint"].append(self.eps)

                if len(stats["Returns"]) > 100:
                    print(f"Episode {ep} - Return: {ep_return} - AvgReturn (last 100): {average_returns} - Eps: {self.eps}")
                else:
                    print(f"Episode {ep} - Return: {ep_return} - Eps: {self.eps}")

            if ep % 100 == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            if ep % 1000 == 0:
                self.model.save(save_path=f"models/dqn_model_ep{ep}.pth")

        return stats
    
    def test(self, env, episodes=5, render=False):
        average_returns = 0.0
        for ep in range(episodes):
            state, info = env.reset()
            done = False
            ep_return = 0
            while not done:
                if render:
                    env.render()
                action = self.act(state)
                next_state, reward, done, info = env.step(action)
                state = next_state
                ep_return += reward.item()
            print(f"Test Episode {ep} - Return: {ep_return}")
            average_returns += ep_return
        
        average_returns /= episodes
        print(f"Average Test Return over {episodes} episodes: {average_returns}")
        return average_returns










