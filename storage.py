import torch
from random import shuffle


class RolloutStorage:
    def __init__(self, num_steps, num_envs, multi_frames):
        super().__init__()
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.multi_frames = multi_frames

        self.values = []
        self.rewards = []
        self.actions = []
        self.log_probs = []
        self.observations = []
        self.masks = []
        self.targets = []

    def add(self, obs, value, reward, action, log_prob, mask):
        self.observations.append(obs.cpu())
        self.values.append(value.cpu())
        self.rewards.append(reward.cpu())
        self.actions.append(action.cpu())
        self.log_probs.append(log_prob.cpu())
        self.masks.append(mask.cpu())

    def reset(self):
        self.values = []
        self.rewards = []
        self.actions = []
        self.log_probs = []
        self.observations = self.observations[-1:]
        self.masks = self.masks[-1:]
        self.values = self.values[-1:]

    def generator(self, gamma, batch_size=16, device='cpu'):
        targets = [None] * (self.num_steps + 1)
        for i in range(self.num_steps+1):
            if i == 0:
                targets[-i-1] = self.values[-i-1]
            else:
                targets[-i-1] = self.rewards[-i] + gamma * targets[-i] * self.masks[-i-1]
        index = list(range(self.num_steps * self.observations[0].shape[0]))
        shuffle(index)
        
        observations = torch.cat(self.observations[:-1])
        targets = torch.cat(targets[:-1])
        actions = torch.cat(self.actions)
        log_probs = torch.cat(self.log_probs)
        for i in range(len(index) // batch_size):
            j = index[i * batch_size: (i + 1) * batch_size]
            yield observations[j].to(device), targets[j].to(device), actions[j].to(device), log_probs[j].to(device)