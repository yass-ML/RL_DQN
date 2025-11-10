import torch
from collections import deque
import random

class ReplayMemory:
    def __init__(self,capacity=1000000):
        self.data = deque(maxlen=capacity)

    def store(self, transition):
        obs_t,a_t,r_t,obs_t_1, done = transition
        if not isinstance(obs_t, torch.Tensor) or not isinstance(obs_t_1, torch.Tensor):
            raise ValueError("Observation should be a tensor of size (84x84x4)")
        
        self.data.append(transition)
    
    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size):
        n_transitions = len(self.data)
        n_sample = min(batch_size,n_transitions)
        indices = random.sample(range(n_transitions), n_sample)
        return [self.data[idx] for idx in indices]