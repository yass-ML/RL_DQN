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
        
        print(f"Replay Memory: {capacity} capacity on {device}")
        if device == "cuda":
            print(f"  Estimated VRAM usage: {capacity * 55 / 1024:.1f} MB (uint8 storage)")


    # def store(self, transition):
    #     transition = (item.to("cpu") for item in transition) # DONE BECAUSE GPU MEMORY FILLS OUT TO RAPIDLY. WILL GO BACK ON DEVICE WHEN SAMPLING

    #     obs_t,a_t,r_t,obs_t_1, done = transition
    #     if not isinstance(obs_t, torch.Tensor) or not isinstance(obs_t_1, torch.Tensor):
    #         raise ValueError("Observation should be a tensor of size (84x84x4)")
        
    #     self.data.append(transition)

    def insert(self, transition):
        """Store transition as uint8 to save 75% memory"""
        state, action, reward, done, next_state = transition
        
        # Convert float32 [0,1] to uint8 [0,255] and move to device
        state_uint8 = (state * 255).byte().to(self.device)
        next_state_uint8 = (next_state * 255).byte().to(self.device)
        
        # Store optimized transition
        optimized = [
            state_uint8,
            action.to(self.device),
            reward.to(self.device),
            done.to(self.device),
            next_state_uint8
        ]
        
        if len(self.memory) < self.capacity:
            self.memory.append(optimized)
        else:
            # Circular buffer - overwrite oldest
            self.memory[self.position] = optimized
            self.position = (self.position + 1) % self.capacity
    
    def __len__(self) -> int:
        return len(self.memory)

    def sample(self, batch_size=32):
        """Sample batch and convert uint8 back to float32"""
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)
        batch = list(zip(*batch))
        
        # Convert uint8 back to float32 [0,1] - already on correct device
        states = torch.cat([b for b in batch[0]]).float() / 255.0
        actions = torch.cat([b for b in batch[1]])
        rewards = torch.cat([b for b in batch[2]])
        dones = torch.cat([b for b in batch[3]])
        next_states = torch.cat([b for b in batch[4]]).float() / 255.0
        
        return [states, actions, rewards, dones, next_states]
    

    def can_sample(self,batch_size, factor=10):
        return len(self.memory) >= batch_size * factor