import numpy as np
import gym
from gym import spaces
import math

class DriverEnv(gym.Env):

    IDLE = 0
    TAKE_RIDER = 1
    TAKE_FOOD = 2
    def __init__(self, total_time, rider_dict, food_dict, max_ride_requests=10, max_food_requests=10, max_food_serv=5):
        super(DriverEnv, self).__init__()
        self.total_time = total_time
        self.timestamp = 0
        self.locations = 6
        self.max_ride_requests = max_ride_requests  # maximum rider requests at a timestamp
        self.max_food_requests = max_food_requests  # maximum food requests at a timestamp
        self.max_food_serve =  max_food_serv # maximum food deliver destinations of a food requests
        # Total possible action
        # action_dim = 1 + self.max_ride_requests + math.comb(self.max_food_requests, self.max_food_serve)
        self.action_space = spaces.Discrete(self.locations * self.max_ride_requests * self.max_food_requests)
        obs_space = self.max_ride_requests * 3 + self.max_food_requests * (2 + self.max_food_serve)
        self.observation_space = spaces.Box(low=-1, high=1000, shape=(obs_space,), dtype=np.float32)
        self.world = np.zeros((200, 200))
        self.has_rider = False
        self.trip_map = rider_dict  # Mapping timestamp to trip orders
        self.food_map = food_dict # Mapping timestamp to food orders
        self.cur_available_rider = 1
        self.cur_location = 0


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
        return dst - src

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
        src = trip['src']
        for dst in trip['dst_list']:
           dur = estimate_time(dst, src)
           time += dur
           reward += -2.5 * dur
           src = dst

        reward += trip['earning']
        return time, reward, src

    def get_obs(self):
        """
        return an observation at a timestamp
        """

        trip_orders = self.trip_map.get(self.timestamp, [])
        food_orders = self.food_map.get(self.timestamp, [])

        riders = [[trip_orders[i]['src'], trip_orders[i]['dst_list'][0], trip_orders[i]['earning']] if i < len(trip_orders) else [0, 0, 0] for i in range(self.max_ride_requests)]
        riders = np.array(riders, dtype=np.int32).flatten()

        #print(riders)
        food = [([food_orders[i]['src'], food_orders[i]['earning']] +
                    [food_orders[i]['dst_list'][k] if k < len(food_orders[i]['dst_list']) else 0 for k in range(self.max_food_serve)])
                         if i < len(food_orders) else ([0, 0] + [0 for k in range(self.max_food_serve)])
                            for i in range(self.max_ride_requests)]
        #print(food)
        food = np.array(food, dtype=np.int32).flatten()

        ret = np.concatenate((riders, food), axis=0)
        
        return ret

    def parse_action(self, action):
        dst_loc = action / (self.max_ride_requests * self.max_food_requests)
        request_trip = (action % (self.max_ride_requests * self.max_food_requests)) / self.max_food_requests
        request_food = action % self.max_food_requests
        return dst_loc, request_trip, request_food

    def step(self, action):
        """
        action (dst_block_id, trip, food)
        """
        trip_orders = self.trip_map.get(self.timestamp, [])
        food_orders = self.food_map.get(self.timestamp, [])

        idle = True
        reward = 0
        dst_loc, request_trip, request_food = self.parse_action(action)
        if len(trip_orders) > 0 and request_trip > 0 and request_trip < len(trip_orders):
            idle = False
            trip = trip_orders[request_trip]
            duration, reward, dst = self.estimate_trip(trip, self.cur_location)
            self.timestamp += duration
            self.cur_location = dst
            loc = self.cur_location
        if len(food_orders) > 0 and request_food > 0 and request_food < len(food_orders):
            idle = False
            food = food_orders[request_food]
            duration, reward, dst = self.estimate_trip(food, self.cur_location)
            self.timestamp += duration
            self.cur_location = dst
            loc = self.cur_location
        else:
            # idle for one timestamp
            duration =  self.estimate_time(dst_loc, self.cur_location)
            self.timestamp += 1 + duration
            reward = -2.5 * duration
            self.cur_location = dst_loc

        done = False
        if self.timestamp >= self.total_time:
            done = True
        info = {}
        return self.get_obs(), reward, done, info

    def render(self, mode='console'):
        pass

    def close(self):
        pass

if __name__ == '__main__':
    rider = {}
    rider[1] = [{}]
    rider[1][0]['earning'] = 300
    rider[1][0]['src'] = 100
    rider[1][0]['dst_list'] = [200,]

    food = {}
    food[1] = [{}]
    food[1][0]['earning'] = 200
    food[1][0]['src'] = 100
    food[1][0]['dst_list'] = [200, 300]

    env = DriverEnv(24 * 6, rider, food)

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

    train_envs = ts.env.DummyVectorEnv([lambda: DriverEnv(24 * 6, rider, food) for _ in range(train_num)])
    test_envs = ts.env.DummyVectorEnv([lambda: DriverEnv(24 * 6, rider, food) for _ in range(test_num)])

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
    



