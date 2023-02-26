import argparse
import torch
from data import preprocess
from env import build_env
from model import ActorCritic
from collections import deque


def main(weights,
         env,
         action_complexity=3,
         device='cpu',
         multi_frames=4):
    env = build_env(env, action_complexity=action_complexity)
    obs_shape = env.observation_space.shape
    obs_shape = (multi_frames, *obs_shape[:-1])
    action_shape = env.action_space.n
    try:
        device = torch.device(int(device))
    except:
        device = torch.device('cpu')
    model = ActorCritic(obs_shape, action_shape).to(device)
    weights = torch.load(weights, map_location=device)
    model.load_state_dict(weights['weights'])
    model.eval()

    obs = torch.from_numpy(preprocess(env.reset())).unsqueeze(dim=0)
    if multi_frames > 1:
        history = deque([torch.zeros(obs.shape) for _ in range(multi_frames)], maxlen=multi_frames)
        history.append(obs)
        obs = torch.cat(tuple(history), dim=1)
    while True:
        # obs = torch.from_numpy(preprocess(obs).unsqueeze(dim=0).to(device))
        # if multi_frames > 1:
        #     history.append(obs)
        #     obs = torch.stack(tuple(history), dim=1)
        _, logits = model(obs.to(device))
        action = logits.argmax(dim=1).cpu().item()
        obs, reward, done, _ = env.step(action)
        env.render()
        if done: break
        obs = torch.from_numpy(preprocess(obs)).unsqueeze(dim=0)
        if multi_frames > 1:
            history.append(obs)
            obs = torch.cat(tuple(history), dim=1)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='project/weights.pt')
    parser.add_argument('--env', type=str, default='SuperMarioBros-1-1-v0')
    parser.add_argument('--action-complexity', type=int, default=0)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--multi-frames', type=int, default=4)

    args = vars(parser.parse_args())
    main(**args)