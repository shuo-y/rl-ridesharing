import numpy as np
import gym
from gym import spaces
import math

class DriverEnv(gym.Env):

    IDLE = 0
    TAKE_RIDER = 1
    TAKE_FOOD = 2
    def __init__(self, total_time, order_dict, max_repositions = 4, max_ride_requests=2):
        super(DriverEnv, self).__init__()
        self.total_time = total_time
        self.timestamp = 0
        self.locations = max_repositions
        self.max_ride_requests = max_ride_requests  # maximum rider requests at a timestamp
        # Total possible action
        self.action_space = spaces.Discrete(self.locations + self.max_ride_requests)
        obs_space = self.max_ride_requests * 3  
        self.observation_space = spaces.Box(low=-1, high=10000, shape=(obs_space,), dtype=np.float32)
        self.trip_map = order_dict  # Mapping timestamp to trip orders
        self.cur_available_rider = 1
        self.cur_location = 1 # TODO should be sampled from several locations


    def reset(self):
        self.timestamp = 0
        self.cur_available_rider = 1
        self.cur_location = 0
        return self.get_obs()

    def estimate_time(self, dst, src):
        """
        Dummy estimate the travel time, distance from src to dst
        Both dst and src are integers
        """
        return 10

    def estimate_trip(self, trip, src):
        """
        A trip
        {'src': int request_location,
         'dst_list': a list of int, 
             if a rider order, only a single destination_location,
             if a food deliever order, multiple destinations
         'earning:'  float trip fee,
        }
        Estimate the duration, reward, and return the final location
        after a rider order or a food deliver order
        """
        time = 0
        dur = estimate_time(trip['src'], src)
        time += dur
        reward = -2.5 * dur
        time += trip['dur']
        reward += -2.5 * dur
        reward += trip['earnings']
        return time, reward, trip['dst']

    def get_obs(self):
        """
        return an observation at a timestamp
        """

        trip_orders = self.trip_map.get(self.timestamp, [])

        riders = [[trip_orders[i]['src'], trip_orders[i]['dst'], trip_orders[i]['earnings']] if i < len(trip_orders) else [0, 0, 0] for i in range(self.max_ride_requests)]
        riders = np.array(riders, dtype=np.int32).flatten()

        
        return riders

    def parse_action_old(self, action):
        dst_loc = action / (self.max_ride_requests * self.max_food_requests)
        request_trip = (action % (self.max_ride_requests * self.max_food_requests)) / self.max_food_requests
        request_food = action % self.max_food_requests
        return dst_loc, request_trip, request_food


    def sample_trip(self, trip_orders):
        """
        add randomness for the trip
        """
        return trip_orders

    def step(self, action):
        """
        action (dst_block_id, trip, food)
        """
        trip_orders = self.trip_map.get(self.timestamp, [])

        trip_orders = self.sample_trip(trip_orders) # Sample from trip orders
        idle = True
        reward = 0
        
        if (action > self.locations and action - self.locations < len(trip_orders)):
            # Take a ride requests
            idle = False
            trip = trip_orders[action - self.max_repositions]
            duration, reward, dst = self.estimate_trip(trip, self.cur_location)
            self.timestamp += duration
            self.cur_location = dst
        else:
            # idle for one timestamp
            # Action represents the locations
            repo_loc = action + 1
            duration =  self.estimate_time(repo_loc, self.cur_location)
            self.timestamp +=  duration
            reward = -2.5 * duration
            self.cur_location = repo_loc

        done = False
        if self.timestamp >= self.total_time:
            done = True
        info = {'loc': self.cur_location, 'time': self.timestamp}
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

    env = DriverEnv(24 * 6, rider)

    print(env.observation_space.shape)
    print(env.action_space.shape)
    
    for _ in range(2):
        action = env.action_space.sample()
        print(type(action))
        obs, reward, done, info = env.step(action)
        print(obs)
        print(obs.shape, reward, done, info)

    env.reset()
    print(env.step(50))
    print(env.step(50))
    print(env.step(50))
    print(env.step(50))

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

    train_envs = ts.env.DummyVectorEnv([lambda: DriverEnv(24 * 6, rider) for _ in range(train_num)])
    test_envs = ts.env.DummyVectorEnv([lambda: DriverEnv(24 * 6, rider) for _ in range(test_num)])

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
    



