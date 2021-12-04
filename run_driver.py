from driver_env import DriverEnv
from utils import read_input, construct_graph
import gym
from gym import spaces
import torch
import tianshou as ts
import numpy as np 
import pickle

def train_model(args, riders: dict, hourly_time, allday_time, isbaseline):
    env = DriverEnv(riders, hourly_time, allday_time, 608, baseline=isbaseline)
    # See https://github.com/thu-ml/tianshou/blob/master/README.md
    lr, epoch, batch_size = 1e-3, args.epoch, 64
    from torch.utils.tensorboard import SummaryWriter
    logger = ts.utils.TensorboardLogger(SummaryWriter(args.logger))
    train_num, test_num = 100, 100
    gamma, n_step, target_freq = 0.9, 3, 320
    buffer_size = 2000000
    eps_train, eps_test = 0.1, 0.05
    step_per_epoch, step_per_collect = args.step_per_epoch, args.step_per_collect

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    from tianshou.utils.net.common import Net
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    train_envs = ts.env.DummyVectorEnv([lambda: DriverEnv(riders, hourly_time, allday_time, 608, baseline=isbaseline) for _ in range(train_num)])
    test_envs = ts.env.DummyVectorEnv([lambda: DriverEnv(riders, hourly_time, allday_time, 608, baseline=isbaseline) for _ in range(test_num)])

    policy = ts.policy.DQNPolicy(net, optim, gamma, n_step, target_update_freq=target_freq)
    print(type(policy))
    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num), exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method

    print(policy)
    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect,
        test_num, batch_size, update_per_step=1 / step_per_collect,
        train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
        test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
        logger=logger)
    print(f'Finished training! Use {result["duration"]}')
    return policy


def eval_model(policy: ts.policy, dev_riders: dict, hourly_time, allday_time, isbaseline):
    test_env = DriverEnv(dev_riders, hourly_time, allday_time, 608, baseline=isbaseline)
    buf = ts.data.ReplayBuffer(2000000)
    test_collector = ts.data.Collector(policy, test_env, buf)
    result = test_collector.collect(n_episode=1)
    print(result)
    print(buf)
    for item in dir(buf):
        print(item)
    print('Sum rewards %f' % np.sum(buf.rew)) 
    for cnt, item in enumerate(buf.rew):
        pass
        #print('%d,%f' % (cnt, item))

import argparse
import time
import random
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--logger', type=str, default='./log/train')
    parser.add_argument('--seed', type=int, default=int(time.time()))
    parser.add_argument('--test_baseline', dest='test_baseline', action='store_true')
    parser.add_argument('--step_per_epoch', type=int, default=100000)
    parser.add_argument('--step_per_collect', type=int, default=10)
    parser.add_argument('--model_prefix', type=str, default='./model/')
    args = parser.parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    riders_train, riders_dev = read_input('download/Gridwise/gridwise_trips_sample_pit.csv')
    #graphs, weights = construct_graph('download/pittsburgh-censustracts-2020-1-All-HourlyAggregate.csv')
    hourly_time = pickle.load(open('download/hourly_time', 'rb'))
    allday_time = pickle.load(open('download/allday_time', 'rb'))
    policy = train_model(args, riders_train, hourly_time, allday_time, isbaseline=False)

    torch.save(policy.state_dict(), args.model_prefix + 'dqn_train.pth')
    eval_model(policy, riders_dev, hourly_time, allday_time, isbaseline=False)

    if args.test_baseline:
        print("Testing baseline")
        policy = train_model(args, riders_train, hourly_time, allday_time, isbaseline=True)
        eval_model(policy, riders_dev, hourly_time, allday_time, isbaseline=True)



