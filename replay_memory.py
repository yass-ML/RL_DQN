import torch
from collections import deque
import random

class ReplayMemory:
    def __init__(self,capacity=1000000, device="cpu"):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity
        self.device = device
        
        print(f"Replay Memory: {capacity} capacity on {device}")
        if device == "cuda":
            print(f"  Estimated VRAM usage: {capacity * 55 / 1024:.1f} MB (uint8 storage)")


    def insert(self, transition):
        """Store transition as uint8 to save GPU memory"""
        state, action, reward, done, next_state = transition
        
        # Convert float32 [0,1] to uint8 [0,255] and move to device
        state_uint8 = (state * 255).byte().to(self.device)
        next_state_uint8 = (next_state * 255).byte().to(self.device)
        
        optimized_transition = [
            state_uint8,
            action.to(self.device),
            reward.to(self.device),
            done.to(self.device),
            next_state_uint8
        ]

        self.memory.append(optimized_transition)


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