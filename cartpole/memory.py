import pdb
import torch
import random
import numpy as np

class Memory:
    def __init__(self, capacity):
        self.capacity   = capacity
        self.states     = [None] * capacity
        self.new_states = [None] * capacity
        self.actions    = np.zeros(capacity, dtype=int)
        self.rewards    = np.zeros(capacity)
        self.dones      = np.zeros(capacity)

        self.idx = 0
        self.size = 0

    def add(self, state, action, new_state, reward, done):
        idx = self.idx
        self.states[idx] = torch.FloatTensor(state)
        self.new_states[idx] = torch.FloatTensor(new_state)
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = [random.randint(0, self.size - 1) for _ in range(batch_size)]
        return {
            'states':       torch.stack([self.states[i] for i in indices]),
            'new_states':   torch.stack([self.new_states[i] for i in indices]),
            'actions':      torch.LongTensor(self.actions[indices]),
            'rewards':      torch.FloatTensor(self.rewards[indices]),
            'dones':        torch.FloatTensor(self.dones[indices]),
        }
