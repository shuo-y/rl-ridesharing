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
        self.max_ride_requests = max_ride_requests  # maximum rider requests at a timestamp
        self.max_food_requests = max_food_requests  # maximum food requests at a timestamp
        self.max_food_serve =  max_food_serv # maximum food deliver destinations of a food requests
        # Total possible action
        # action_dim = 1 + self.max_ride_requests + math.comb(self.max_food_requests, self.max_food_serve)
        self.action_space = spaces.Box(low=-10, high=10, shape=(3,), dtype=np.int32)
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

    def estimate_time_dist(self, dst, src):
        """
        Dummy estimate the travel time, distance from src to dst
        """
        if dst == src:
            return 0, 1
        else:
            return 5, 10

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
        dur, dist = estimate_time_dist(trip['src'], src)
        time += dur
        reward = -2.5 * dist
        src = trip['src']
        for dst in trip['dst_list']:
           dur, dist = estimate_time_dist(dst, src)
           time += dur
           reward += -2.5 * dist
           src = dst

        reward += trip['earning']
        return time, reward, src

    def get_obs(self):
        """
        return an observation at a timestamp
        """

        trip_orders = self.trip_map.get(self.timestamp, [])
        food_orders = self.food_map.get(self.timestamp, [])

        

    def step(self, action):
        """
        action (dst_block_id, trip, food)
        """
        trip_orders = self.trip_map.get(self.timestamp, [])
        food_orders = self.food_map.get(self.timestamp, [])

        idle = True
        reward = 0
        dst_loc = action[0]
        request_trip = action[1]
        request_food = action[2]
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
            distance, duration =  self.estimate_time_dist(dst_loc, self.cur_location)
            self.timestamp += duration
            reward -= -2.5 * distance
            self.cur_location = dst_loc

        done = False
        if self.timestamp >= self.total_time:
            done = True
        info = {}
        return np.array([-1, 10, 100]).astype(np.float32), reward, done, info

    def render(self, mode='console'):
        pass

    def close(self):
        pass

if __name__ == '__main__':
    env = DriverEnv(24 * 60 * 60, {}, {})
    action = env.action_space.sample()
    print(env.step(action))
