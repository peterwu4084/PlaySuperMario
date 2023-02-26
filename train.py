import argparse
import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from multiprocessing import cpu_count
from data import preprocess
from env import build_env
from model import ActorCritic
from loss import PPOLoss
from storage import RolloutStorage
from collections import deque


def get_action(model, obs):
    values, logits = model(obs)
    dist = torch.distributions.Categorical(logits=logits)
    actions = dist.sample().reshape(-1, 1)
    log_probs = F.log_softmax(logits, dim=1)
    return values, actions, log_probs


def main(env,
         weights='',
         num_envs=None,
         action_complexity=3,
         clip_param=0.1,
         value_loss_weight=1,
         action_loss_weight=1,
         entropy_loss_weight=0.01,
         lr=7e-4,
         num_frames=1e7,
         num_steps=128,
         gamma=0.99,
         num_reuse=3,
         grad_norm_clip=0.5,
         save_dir='project',
         device='cpu',
         multi_frames=4):
    num_envs = cpu_count() if num_envs is None else num_envs
    envs = [build_env(env, action_complexity=action_complexity) for _ in range(num_envs)]
    obs_shape = envs[0].observation_space.shape
    action_shape = envs[0].action_space.n
    try:
        device = torch.device(int(device))
    except:
        device = torch.device('cpu')
    obs_shape = (multi_frames, *obs_shape[:-1])
    model = ActorCritic(obs_shape, action_shape).to(device)
    if weights and os.path.exists(weights):
        weights = torch.load(weights, map_location=device)
        model.load_state_dict(weights['weights'])
    model.train()
    ppo_loss = PPOLoss(clip_param, value_loss_weight, action_loss_weight, entropy_loss_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    storage = RolloutStorage(num_steps, num_envs, multi_frames)

    epochs = int(num_frames / num_envs / num_steps)
    lf = lambda x: (1 - x / epochs) * (1.0 - 0.01) + 0.01  # linear
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    obs = torch.from_numpy(np.stack([preprocess(e.reset()) for e in envs]))
    mask = torch.BoolTensor([True] * num_envs).reshape(-1, 1)
    if multi_frames > 1:
        history = deque([torch.zeros(obs.shape) for _ in range(multi_frames)], maxlen=multi_frames)
        history.append(obs)
        obs = torch.cat(tuple(history), dim=1)
    storage.masks.append(mask)
    storage.observations.append(obs)
    logger = open(os.path.join(save_dir, 'log.csv'), 'w')

    for epoch in range(epochs):
        with torch.no_grad():
            for step in range(num_steps):
                values, actions, log_probs = get_action(model, obs.to(device))
                outputs = [envs[env_idx].step(actions[env_idx].item())[:3] for env_idx in range(num_envs)]
                for i, output in enumerate(outputs):
                    if output[2]:
                        outputs[i] = envs[i].reset(), output[1], output[2]
                        if multi_frames > 1:
                            for j in range(multi_frames):
                                history[j][i] = 0
                obs = torch.from_numpy(np.stack([preprocess(_[0]) for _ in outputs]))
                if multi_frames > 1:
                    history.append(obs)
                    obs = torch.cat(tuple(history), dim=1)
                rewards = torch.from_numpy(np.array([_[1] for _ in outputs])).reshape(-1, 1) / 15
                masks = torch.from_numpy(np.array([not _[2] for _ in outputs])).reshape(-1, 1)
                storage.add(obs, values, rewards, actions, log_probs, masks)
            else:
                values = get_action(model, obs.to(device))[0]
                storage.values.append(values.cpu())
        for i in range(num_reuse):
            mean_value_loss, mean_action_loss, mean_entropy_loss, total_loss = 0, 0, 0, 0
            for j, (obs, targets, actions, old_log_probs) in enumerate(storage.generator(gamma, device=device)):
                values, _, log_probs = get_action(model, obs)
                loss, loss_items = ppo_loss(values, actions, log_probs, old_log_probs, targets)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
                optimizer.step()

                mean_value_loss += loss_items[0]
                mean_action_loss += loss_items[1]
                mean_entropy_loss += loss_items[2]
                total_loss += loss.item()

            mean_value_loss /= (j + 1)
            mean_action_loss /= (j + 1)
            mean_entropy_loss /= (j + 1)
            total_loss /= (j + 1)
            lr = optimizer.param_groups[0]['lr']
            s = '{:10d}/{:<10d}, {:10.5g}, {:10.5g}, {:10.5g}, {:10.5g}, {:10.5g}'.format(epoch * num_reuse + i, 
                                                                                     epochs * num_reuse,
                                                                                     lr,
                                                                                     mean_value_loss,
                                                                                     mean_action_loss,
                                                                                     mean_entropy_loss,
                                                                                     total_loss)
            print(s)
            logger.write(s+'\n')
        scheduler.step()
        storage.reset()
    state_dict = model.state_dict()
    torch.save(dict(weights=state_dict, env=env, action_complexity=action_complexity),
               os.path.join(save_dir, 'weights.pt'))
    logger.close()
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='SuperMarioBros-1-1-v0')
    parser.add_argument('--weights', type=str, default='project/weights.pt')
    parser.add_argument('--action-complexity', type=int, default=0)
    parser.add_argument('--clip-param', type=float, default=0.1)
    parser.add_argument('--value-loss-weight', type=float, default=1)
    parser.add_argument('--action-loss-weight', type=float, default=1)
    parser.add_argument('--entropy-loss-weight', type=float, default=0.01)
    parser.add_argument('--num-envs', type=int, default=None)
    # parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--num-frames', type=int, default=5e6)
    # parser.add_argument('--num-frames', type=int, default=128)
    parser.add_argument('--num-steps', type=int, default=512)
    parser.add_argument('--gamma', type=float, default=0.90)
    parser.add_argument('--num-reuse', type=int, default=6)
    parser.add_argument('--grad-norm-clip', type=float, default=0.5)
    parser.add_argument('--save-dir', type=str, default='project')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--multi-frames', type=int, default=4)

    args = vars(parser.parse_args())
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])
    with open(os.path.join(args['save_dir'], 'arguments.json'), 'w') as f:
        json.dump(args, f, indent=2)
    main(**args)