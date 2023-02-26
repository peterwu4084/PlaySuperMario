import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOLoss(nn.Module):
    def __init__(self, clip_param=0.1, value_loss_weight=1, action_loss_weight=1, entropy_loss_weight=0.01):
        super().__init__()
        self.clip_param = clip_param
        self.value_loss_weight = value_loss_weight
        self.action_loss_weight = action_loss_weight
        self.entropy_loss_weight = entropy_loss_weight

    def forward(self,
                values,
                actions,
                cur_log_probs,
                old_log_probs,
                target_values):
        advantages = target_values - values
        # if advantages.shape[0] > 1:
        #     advantages = (advantages - advantages.mean()) / (
        #         advantages.std() + 1e-5)
        value_loss = advantages.pow(2).mean()

        cur_action_log_probs = cur_log_probs.gather(1, actions)
        old_action_log_probs = old_log_probs.gather(1, actions)
        ratio = torch.exp(cur_action_log_probs - old_action_log_probs)
        clamped_ratio = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param)
        action_loss = -torch.min(ratio * advantages.detach(),
                                 clamped_ratio * advantages.detach()).mean()

        dist = torch.distributions.Categorical(logits=cur_log_probs)
        entropy_loss = -dist.entropy().mean()

        total_loss = value_loss * self.value_loss_weight + \
                     action_loss * self.action_loss_weight + \
                     entropy_loss * self.entropy_loss_weight
        return total_loss, (value_loss.item(), action_loss.item(), entropy_loss.item(), total_loss.item())