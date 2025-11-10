from dqn_model import DQN
from replay_memory import ReplayMemory
import torch
from torch.optim import RMSprop
import random
import torch.nn.functional as F

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

        
        








