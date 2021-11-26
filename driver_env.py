import numpy as np
import gym
from gym import spaces
import math
import random
import torch

class DriverEnv(gym.Env):

    def __init__(self, order_dict, hourly_time, allday_time, max_repositions, max_ride_requests=2, baseline=False, interval=300):
        super(DriverEnv, self).__init__()
        self.total_time = 24 * 3600
        self.timestamp = 0
        self.baseline = baseline
        self.max_repositions = max_repositions
        self.interval = interval # interval of the orders inside a period
        if self.baseline == False:
            self.locations = max_repositions
        else:
            # if baseline is True
            # never reposition at all
            self.locations = 0
        self.max_ride_requests = max_ride_requests  # maximum rider requests at a timestamp
        # Total possible action
        self.action_space = spaces.Discrete(self.locations + self.max_ride_requests)
        obs_space = 2 + self.max_ride_requests * 3  # locations id + timestep + number of orders
        self.observation_space = spaces.Box(low=-1, high=100000, shape=(obs_space,), dtype=np.float32)
        self.trip_map = order_dict  # Mapping timestamp to trip orders
        self.cur_location = random.randint(1, max_repositions) # TODO should be sampled from several locations
        self.hourly_time = hourly_time
        self.allday_time = allday_time
        self.distri = torch.distributions.Normal(0, 1000)

    def reset(self):
        self.timestamp = 0
        self.cur_location = random.randint(1, self.max_repositions)
        return self.get_obs()

    def estimate_time(self, dst, src):
        """
        Dummy estimate the travel time, distance from src to dst
        Both dst and src are integers
        return seconds
        """
        dur = 0
        hod = self.timestamp // 3600
        if src == dst:
            dur =  60
        elif src in self.hourly_time and dst in self.hourly_time[src] and hod in self.hourly_time[src][dst][hod]:
            dur = self.hourly_map[src][dst][hod]
        else:
            dur = self.allday_time[src][dst]
        assert dur > 0
        return dur


    def get_obs(self):
        """
        return an observation at a timestamp
        """
        
        trip_orders = self.trip_map.get(self.timestamp // self.interval, [])
        trip_orders = self.sample_trip(trip_orders) # Sample from trip orders
        self.trip_orders_cache = trip_orders
        riders = [[trip_orders[i]['src'], trip_orders[i]['dst'], trip_orders[i]['earnings']] if i < len(trip_orders) else [0, 0, 0] for i in range(self.max_ride_requests)]
        riders = np.array(riders, dtype=np.int32).flatten()
        world_env = np.array([self.cur_location, self.timestamp])
        riders = np.concatenate((world_env, riders))
        return riders

    def parse_action_old(self, action):
        dst_loc = action / (self.max_ride_requests * self.max_food_requests)
        request_trip = (action % (self.max_ride_requests * self.max_food_requests)) / self.max_food_requests
        request_food = action % self.max_food_requests
        return dst_loc, request_trip, request_food


    def sample_trip(self, trip_orders):
        """
        sample orders based on normal distributions
        """
        if len(trip_orders) == 0:
            return trip_orders

        rc_orders = []
        for trip in trip_orders:
            dur = self.estimate_time(trip['src'], self.cur_location)
            if dur < torch.abs(self.distri.sample()).item():
                rc_orders.append(trip)
        return rc_orders

    def step(self, action):
        """
        action (dst_block_id, trip, food)
        """
        trip_orders = self.trip_orders_cache

        order_succ = False
        repo_succ = False
        reward = 0
         
        if action >= self.locations or self.baseline == True:
            if len(trip_orders) == 0:
                # No order
                # the driver stays the same location
                self.timestamp += 60
                reward = -0.0005
            else:
                # There is a order
                # driver take the order
                if action - self.locations < len(trip_orders):
                    # Try to take a ride requests based on driver's preference
                    trip = trip_orders[action - self.locations]
                else:
                    # just take the first order
                    trip = trip_orders[0]

                duration = self.estimate_time(trip['src'], self.cur_location)
                order_succ = True
                reward = (-0.0015 * (duration + trip['dur']) + trip['earnings'])
                self.timestamp += (duration + trip['dur'])
                self.cur_location = trip['dst']
        
        else:
            # driver move to some other places
            # Action represents the locations
            repo_loc = action + 1
            duration =  self.estimate_time(repo_loc, self.cur_location)
            repo_succ = True
            self.timestamp += duration
            reward = -0.0015 * duration
            self.cur_location = repo_loc
    

        done = False
        if self.timestamp >= self.total_time:
            done = True
        info = {'loc': self.cur_location, 'time': self.timestamp, 'order_s': order_succ, 'repo_s': repo_succ}
        return self.get_obs(), reward, done, info

    def render(self, mode='console'):
        pass

    def close(self):
        pass

if __name__ == '__main__':
    rider = {}
    rider[1] = [{}]
    rider[1][0]['earnings'] = 300
    rider[1][0]['src'] = 100
    rider[1][0]['dst'] = 200

    env = DriverEnv(rider)

    print(env.observation_space.shape)
    print(env.action_space.shape)
    
    for _ in range(2):
        action = env.action_space.sample()
        print(type(action))
        obs, reward, done, info = env.step(action)
        print(obs)
        print(obs.shape, reward, done, info)


    env.reset()

    # dummy program for testing based on tianshou demo
    # See https://github.com/thu-ml/tianshou/blob/master/README.md
    lr, epoch, batch_size = 1e-3, 2, 64
    train_num, test_num = 10, 100
    gamma, n_step, target_freq = 0.9, 3, 320
    buffer_size = 2000000
    eps_train, eps_test = 0.1, 0.05
    step_per_epoch, step_per_collect = 10000, 10

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    import torch
    import tianshou as ts
    from torch import nn
    from tianshou.utils.net.common import Net
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    train_envs = ts.env.DummyVectorEnv([lambda: DriverEnv(rider) for _ in range(train_num)])
    test_envs = ts.env.DummyVectorEnv([lambda: DriverEnv(rider) for _ in range(test_num)])

    policy = ts.policy.DQNPolicy(net, optim, gamma, n_step, target_update_freq=target_freq)
    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num), exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method

    print(policy)
    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect,
        test_num, batch_size, update_per_step=1 / step_per_collect,
        train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
        test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
        stop_fn=lambda mean_rewards: mean_rewards >= 50)
    print(f'Finished training! Use {result["duration"]}')
   
    policy.eval()
    policy.set_eps(eps_test)

    buffer = ts.data.ReplayBuffer(size=10000)
    env.reset()
    collector = ts.data.Collector(policy, env, buffer=buffer, exploration_noise=True)
    
    from time import time
    begin = time()
    collector.collect(n_episode=1)
    print("Collect 1 episode %.2f " % (time() - begin))

    # Use pickle save binary
    # See https://stackoverflow.com/questions/1047318/easiest-way-to-persist-a-data-structure-to-a-file-in-python
    with open('debug.out', 'w') as f:
        for item in buffer.obs:
            f.write(str(list(item))+ '\n')
    



