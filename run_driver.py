from driver_env import DriverEnv
from utils import read_input
import gym
from gym import spaces
import torch
import tianshou as ts
 

def train_model(riders: dict):
    env = DriverEnv(riders)
    # See https://github.com/thu-ml/tianshou/blob/master/README.md
    lr, epoch, batch_size = 1e-3, 1, 64
    train_num, test_num = 10, 100
    gamma, n_step, target_freq = 0.9, 3, 320
    buffer_size = 2000000
    eps_train, eps_test = 0.1, 0.05
    step_per_epoch, step_per_collect = 10000, 10

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    from tianshou.utils.net.common import Net
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    train_envs = ts.env.DummyVectorEnv([lambda: DriverEnv(riders) for _ in range(train_num)])
    test_envs = ts.env.DummyVectorEnv([lambda: DriverEnv(riders) for _ in range(test_num)])

    policy = ts.policy.DQNPolicy(net, optim, gamma, n_step, target_update_freq=target_freq)
    print(type(policy))
    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num), exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method

    print(policy)
    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect,
        test_num, batch_size, update_per_step=1 / step_per_collect,
        train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
        test_fn=lambda epoch, env_step: policy.set_eps(eps_test))
    print(f'Finished training! Use {result["duration"]}')
    return policy


def eval_model(policy: ts.policy, dev_riders: dict):
    test_env = DriverEnv(dev_riders)
    buf = ts.data.ReplayBuffer(2000000)
    test_collector = ts.data.Collector(policy, test_env, buf)
    result = test_collector.collect(n_episode=1)
    print(result)
    print(buf)
    for item in dir(buf):
        print(item)


if __name__ == '__main__':
    riders_train, riders_dev = read_input('download/Gridwise/gridwise_trips_sample_pit.csv')
    policy = train_model(riders_train)
    eval_model(policy, riders_dev)

