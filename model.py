import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k//2, bias=True)
        # self.bn = nn.LayerNorm()
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        # return self.act(self.bn(self.conv(x)))
        return self.act(self.conv(x))


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_shape, feature_sharing=True):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.feature_sharing = feature_sharing
        self.actor_backbone = self.build_backbone()
        self.actor_neck = self.build_neck()
        if not feature_sharing:
            self.critic_backbone = self.build_backbone()
            self.critic_neck = self.build_neck()
        else:
            self.critic_backbone = self.actor_backbone
            self.critic_neck = self.actor_neck
        self.actor_head, self.critic_head = self.build_head() 
        self.init_weight()

    def build_backbone(self):
        # backbone = nn.Sequential(
        #     Conv(self.obs_shape[0], 32, 8, 4),
        #     Conv(32, 64, 4, 2),
        #     Conv(64, 32, 3, 1),
        # )
        backbone = nn.Sequential(
            Conv(self.obs_shape[0], 32, 4, 4),
            Conv(32, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 64, 3, 2),
            Conv(64, 128, 3, 2),
        )
        return backbone

    def build_neck(self):
        in_features = self.actor_backbone(torch.zeros(1, *self.obs_shape)).view(1, -1).size(1)
        return nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU()
        )

    def build_head(self):
        return nn.Linear(512, self.action_shape), nn.Linear(512, 1)

    def init_weight(self):
        # init backbone and neck
        gain = nn.init.calculate_gain('relu') 
        for conv in self.actor_backbone:
            nn.init.orthogonal_(conv.conv.weight.data, gain=gain)
            nn.init.constant_(conv.conv.bias.data, 0)
        nn.init.orthogonal_(self.actor_neck[0].weight.data, gain=gain)
        nn.init.constant_(self.actor_neck[0].bias.data, 0)

        if not self.feature_sharing:
            for conv in self.critic_backbone:
                nn.init.orthogonal_(conv.conv.weight.data, gain=gain)
                nn.init.constant_(conv.conv.bias.data, 0)
            nn.init.orthogonal_(self.critic_neck[0].weight.data, gain=gain)
            nn.init.constant_(self.critic_neck[0].bias.data, 0)

        # init head
        nn.init.orthogonal_(self.actor_head.weight.data, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight.data, gain=1)
        nn.init.constant_(self.actor_head.bias.data, 0)
        nn.init.constant_(self.critic_head.bias.data, 0)

    def forward(self, x):
        bs = x.shape[0]
        actor_feature = self.actor_neck(self.actor_backbone(x).reshape(bs, -1))
        if self.training:
            if self.feature_sharing:
                critic_feature = actor_feature
            else:
                critic_feature = self.critic_neck(self.critic_backbone(x).view(bs, -1))
            return self.critic_head(critic_feature), self.actor_head(actor_feature)
        else:
            return None, self.actor_head(actor_feature)
