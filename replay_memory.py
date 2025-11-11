import torch
from collections import deque
import random

class ReplayMemory:
    def __init__(self,capacity=1000000, device="cpu"):
        # self.data = deque(maxlen=capacity)
        self.memory = []
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.memory_max_report = 0


    # def store(self, transition):
    #     transition = (item.to("cpu") for item in transition) # DONE BECAUSE GPU MEMORY FILLS OUT TO RAPIDLY. WILL GO BACK ON DEVICE WHEN SAMPLING

    #     obs_t,a_t,r_t,obs_t_1, done = transition
    #     if not isinstance(obs_t, torch.Tensor) or not isinstance(obs_t_1, torch.Tensor):
    #         raise ValueError("Observation should be a tensor of size (84x84x4)")
        
    #     self.data.append(transition)

    def insert(self,transition):
        transition = [item.to("cpu") for item in transition]
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory.remove(self.memory[0])
            self.memory.append(transition)
    
    def __len__(self) -> int:
        return len(self.memory)

    def sample(self, batch_size=32):
        # n_transitions = len(self.data)
        # n_sample = min(batch_size,n_transitions)
        # indices = random.sample(range(n_transitions), n_sample)
        # return [self.data[idx] for idx in indices]

        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        sample = [torch.cat(items).to(device=self.device) for items in batch]
        #print(f"sample len: {len(sample)}, sample shapes: {[s.shape for s in sample]}")
        return sample
    

    def can_sample(self,batch_size, factor=10):
        return len(self.memory) >= batch_size * factor