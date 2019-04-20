import argparse
import pdb
import numpy as np
import random
import gym
from dqn import MLP
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from memory import Memory

def get_action(model, state):
    # arg max action
    action_vals = model(torch.FloatTensor(state).unsqueeze(0))
    action = torch.argmax(action_vals)
    return action.item()

def update(model, target, batch, discount, loss_function, opt):
    q_s = model.forward(batch['states'])
    q_sa = torch.stack([q_s[idx, i] for idx, i in enumerate(batch['actions'])]).unsqueeze(-1)

    q_next = target.forward(batch['new_states'])
    q_next_max = torch.max(q_next, dim=1)[0].detach()
    q_onestep = batch['rewards'] + (1 - batch['dones']) * (discount * q_next_max) # size batch x 2

    opt.zero_grad()
    loss = loss_function(q_sa, q_onestep)
    loss.backward()
    opt.step()

def main(args):
    env = gym.make('CartPole-v0')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    env.seed(args.seed)
    n_in = 4 # should really get this from env feature space somehow
    n_hid = 32
    n_out = 2
    discount = 1
    lr = 0.1
    episodes = 2000
    exp_epochs = 1000
    exp_min = 0.05
    max_steps = 200
    capacity = 10000
    batch_size = 64
    update_int = 25
    log_int = 100
    target_int = 25

    model = MLP(n_in, n_hid, n_out)
    target = MLP(n_in, n_hid, n_out)
    target.load_state_dict(model.state_dict())

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    ep_len = np.zeros(episodes)
    memory = Memory(capacity)
    eps = []
    idx = 0

    for e in range(episodes):
        state = env.reset()
        for i in range(max_steps):
            exp_rate = max(0.05, 1 - e / exp_epochs)
            if random.random() < exp_rate:
                action = random.randint(0, env.action_space.n - 1)
            else:
                action = get_action(model, state)

            new_state, reward, done, _ = env.step(action)
            memory.add(state, action, new_state, reward, done)

            state = new_state
            idx += 1

            if idx % update_int == 0 and idx > 0 and memory.size >= batch_size:
            #if memory.size >= batch_size:
                batch = memory.sample(batch_size)
                update(model, target, batch, discount, loss_func, opt)

            if idx % target_int == 0 and idx > 0:
                target.load_state_dict(model.state_dict())

            if done:
                break

        ep_len[e] = i
        eps.append(i)
        if e % log_int == 0 and e > 0:
            lastk = eps[-log_int:]
            print('Episode {:4} | Last {} ep avg {:5.2f}, max {:5.2f}, min {:5.2f} | exp rate: {:.2f}'.format(
                e, log_int, np.mean(lastk), np.max(lastk), np.min(lastk), exp_rate
            ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    main(args)
