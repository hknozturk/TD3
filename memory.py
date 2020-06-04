from __future__ import division
from collections import namedtuple
import numpy as np
import torch

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "not_done"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False # Track memory capacity
        self.sum_tree = np.zeros((2 * size - 1, ), dtype=np.float32) # Init fixed size tree with all (prioritiy) zero
        self.data = np.array([None] * size) # Wrap-around cyclic buffer
        self.max = 1 # Init max value to return (1 = 1^w)

    # Propagates value up tree given a tree index
    def _propagate(self, index, value):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)

    # Updates value given a tree index
    def update(self, index, value):
        self.sum_tree[index] = value # Set new value
        self._propagate(index, value) # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data # Store data in underlying data structure
        self.update(self.index + self.size - 1, value) # Update tree
        self.index = (self.index + 1) % self.size # Update index
        self.full = self.full or self.index == 0 # Save when capacity reached
        self.max = max(value, self.max)

        if self.full:
            print("Memory is full")

    # Searches for the location of a value in sum tree
    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_tree[left])

    # Searches for a value in sum tree and returns value, data index and tree index
    def find(self, value):
        index = self._retrieve(0, value) # Search for index of item from root
        data_index = index - self.size + 1
        return (self.sum_tree[index], data_index, index) # Return value, data index, tree index

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]

    def __len__(self):
        return len(self.data)

class ReplayMemory():
    def __init__(self, args, capacity):
        self.capacity = capacity
        self.discount = args.discount
        self.priority_weight = args.priority_weight
        self.priority_exponent = args.priority_exponent
        self.absolute_error_upper = args.absolute_error_upper
        self.t = 0 # Internal episode timestep counter
        self.tree = SegmentTree(capacity) # Store experiences in a wrap-around cyclic buffer within a sum tree for querying priorities

    # Adds state and action at time t, reward and done at time t + 1
    def append(self, state, action, reward, next_state, done):
        self.tree.append(Experience(state, action, reward, next_state, 1. - done), self.tree.max) # Store new transition with maximum priority
        self.t = 0 if done else self.t + 1 # Start new episodes with t = 0

    def _get_sample_from_segment(self, segment, i):
        valid = False
        while not valid:
            sample = np.random.uniform(i * segment, (i + 1) * segment) # Uniformly sample an element from within a segment
            prob, idx, tree_idx = self.tree.find(sample) # Retrieve sample from tree with un-normalised probability
            # Resample if transition straddled current index or probability 0
            if prob != 0:
                valid = True # Note that conditions are valid but extra conservative around buffer index 0

        experience = self.tree.get(idx)

        return prob, idx, tree_idx, experience

    def sample(self, batch_size):
        p_total = self.tree.total() # Retrieve sum of all priorities (used to create a normalised probability distribution)
        segment = p_total / batch_size # Batch size number of segments, based on sum over all probabilities

        batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)] # Get batch of valid samples
        probs, idxs, tree_idxs, experiences = zip(*batch)
        
        states = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).to(device=device).to(dtype=torch.float32)
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).to(device=device).to(dtype=torch.float32)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).to(device=device).to(dtype=torch.float32)
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences if exp is not None])).to(device=device).to(dtype=torch.float32)
        not_dones = torch.from_numpy(np.vstack([exp.not_done for exp in experiences if exp is not None])).to(device=device).to(dtype=torch.float32)

        probs = np.array(probs, dtype=np.float32) / p_total # Calculate normalised probabilities
        capacity = self.capacity if self.tree.full else self.tree.index
        weights = (capacity * probs) ** -self.priority_weight # Compute importance-sampling weights w
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=device) # Normalise by max importance-sampling weight from batch
        return tree_idxs, states, actions, rewards, next_states, not_dones, weights

    def update_priorities(self, idxs, priorities):
        # priorities = np.power(priorities, self.priority_exponent)
        clipped_errors = np.minimum(priorities, self.absolute_error_upper)
        clipped_errors = np.power(clipped_errors, self.priority_exponent)
        for idx, priority in zip(idxs, clipped_errors):
            self.tree.update(idx, priority)

    def __len__(self):
        return len(self.tree)