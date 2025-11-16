import torch
from collections import deque
import random
import numpy as np

class ReplayMemory:
    def __init__(self,capacity=1000000, device="cpu"):
        self.capacity = capacity
        self.device = device

        # Frame storage
        self.frames = torch.zeros((capacity, 1, 84, 84), dtype=torch.uint8, device=device)

        # Transition data (aligned with frames)
        self.actions = torch.zeros((capacity, 1), dtype=torch.int, device=device) #action[i]: action taken to go from frame[i-1] to frame[i]
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device) #reward[i]: reward received after taking action at frame[i-1]
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool, device=device) #done[i]: whether episode ended at frame[i]

        # Circular buffer state
        self.write_pos = 0  # Current write position
        self.size = 0      # Number of valid entries (up to capacity)    



    def insert_frame(self,frame: torch.Tensor) -> int:
        assert frame.shape == (84,84), f"Game frame not of right shape: got {frame.shape}, expected (84,84)"
        frame_idx = self.write_pos
        self.frames[self.write_pos] = frame
        self.write_pos = (self.write_pos + 1) % self.capacity
        if self.size < self.capacity:
            self.size +=1

        return frame_idx
    

    def insert_transition(self, next_frame_idx, action, reward, done):
        assert next_frame_idx < self.capacity, f"index {next_frame_idx} above capacity {self.capacity}"
        self.actions[next_frame_idx] = action
        self.rewards[next_frame_idx] = reward
        self.dones[next_frame_idx] = done
            
        

    def get_stacked_state(self, frame_indices: list, stack_size=4) -> torch.Tensor:
        """
        Get a stacked state from a list of frame indices, handling episode boundaries.
        
        Args:
            frame_indices: List of 4 frame indices [idx-3, idx-2, idx-1, idx]
            stack_size: Number of frames to stack (default 4)
        
        Returns:
            Stacked state tensor of shape (1, stack_size, 84, 84) as float32 [0, 1]
        """
        assert len(frame_indices) == stack_size, f"Expected {stack_size} indices, got {len(frame_indices)}"
        
        # Check for episode boundaries in the list
        # Find if any of the first 3 frames are terminal
        num_to_pad = 0
        for i in range(stack_size - 1):  # Don't check the last frame
            if self.dones[frame_indices[i]]:  # Episode ended at this frame
                num_to_pad = i + 1  # Pad all frames up to and including this one
                break
        
        # Get the valid frames (after episode boundary)
        if num_to_pad > 0:
            valid_indices = frame_indices[num_to_pad:]  # Frames from current episode
            valid_frames = self.frames[valid_indices]  # Shape: (stack_size - num_to_pad, 1, 84, 84)
            
            # Create zero padding
            padding = torch.zeros((num_to_pad, 1, 84, 84), dtype=torch.uint8, device=self.device)
            
            # Concatenate: padding first, then valid frames
            stacked = torch.cat([padding, valid_frames], dim=0)  # Shape: (stack_size, 1, 84, 84)
        else:
            # No padding needed, use all frames
            stacked = self.frames[frame_indices]  # Shape: (stack_size, 1, 84, 84)
        
        # Reshape and convert to float32 [0, 1]
        stacked = stacked.squeeze(1).unsqueeze(0)  # Shape: (1, stack_size, 84, 84)
        stacked = stacked.float() / 255.0
        
        return stacked

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size=32):
        """Sample batch and convert uint8 back to float32"""
        assert self.can_sample(batch_size)

        # Can only sample indices where next frame exists (i+1 must be valid)
        # AND the current frame is not terminal (so i+1 is meaningful)
        frame_indices = torch.arange(self.size - 1, device=self.device)  # Can't sample last frame
        dones_mask = self.dones[:self.size - 1].squeeze(-1)
        non_terminal_indices = frame_indices[~dones_mask]

        indices = non_terminal_indices[torch.randint(0, len(non_terminal_indices), (batch_size,), device=self.device)]
        
        # Vectorized state stacking
        offsets = torch.tensor([-3, -2, -1, 0], device=self.device)
        state_indices = (indices.unsqueeze(1) + offsets) % self.capacity
        
        next_offsets = torch.tensor([-2, -1, 0, 1], device=self.device)
        next_state_indices = (indices.unsqueeze(1) + next_offsets) % self.capacity
        
        # Vectorized episode boundary detection
        check_offsets = torch.tensor([-3, -2, -1], device=self.device)
        check_indices = (indices.unsqueeze(1) + check_offsets) % self.capacity
        dones_check = self.dones[check_indices].squeeze(-1)  # (batch_size, 3)
        
        dones_flipped = dones_check.flip(dims=[1])
        has_boundary = dones_check.any(dim=1)
        
        boundary_pos = torch.where(has_boundary,
                                dones_flipped.long().argmax(dim=1),
                                torch.full((batch_size,), 3, dtype=torch.long, device=self.device))
        
        num_pad = torch.clamp(3 - boundary_pos, min=0, max=3)
        
        positions = torch.arange(4, device=self.device).unsqueeze(0)
        pad_mask_state = positions < num_pad.unsqueeze(1)
        
        num_pad_next = torch.clamp(2 - boundary_pos, min=0, max=3)
        pad_mask_next = positions < num_pad_next.unsqueeze(1)
        
        # Gather frames
        states_frames = self.frames[state_indices]
        next_states_frames = self.frames[next_state_indices]
        
        # Apply padding
        pad_mask_state_expanded = pad_mask_state.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        pad_mask_next_expanded = pad_mask_next.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        
        states_frames = torch.where(pad_mask_state_expanded, 
                                    torch.zeros_like(states_frames), 
                                    states_frames)
        next_states_frames = torch.where(pad_mask_next_expanded,
                                        torch.zeros_like(next_states_frames),
                                        next_states_frames)
        
        states = states_frames.squeeze(2).float() / 255.0
        next_states = next_states_frames.squeeze(2).float() / 255.0
        
        # CORRECTED: Get transition data at indices + 1
        # Because actions[i+1] is the action FROM frame[i] TO frame[i+1]
        transition_indices = (indices + 1) % self.capacity
        actions = self.actions[transition_indices]
        rewards = self.rewards[transition_indices]
        dones = self.dones[transition_indices]
        
        return states, actions, rewards, dones, next_states

    

    def can_sample(self,batch_size, factor=50):
        return self.size >= batch_size * factor