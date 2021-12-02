# a greedy agent interact with the driver env
import pickle
from driver_env import DriverEnv
from utils import read_input
from operator import itemgetter

class Driver:

    def __init__(self, env, locations, max_ride_requests):
        self.env = env
        self.locations = locations
        self.max_ride_requests = max_ride_requests

    
    def roundout(self):
        obs = self.env.reset()
        done = False
        total_rewards = 0.0
        print(obs)
        while done == False:
            cur_location = int(obs[0])
            #print(obs)
            trips = obs[- self.max_ride_requests * 3:] # get the trips
            action = self.locations
            #print(trips)
            scores = [-0.0015 * self.env.estimate_time(int(trips[ind * 3]), int(cur_location)) + trips[ind * 3 + 2] 
                        if trips[ind * 3] > 0 else 0.0 for ind in range(self.max_ride_requests)]
            #print(scores)
            if len(scores) > 0:
                pick_ind = max(range(len(scores)), key=scores.__getitem__)
                # https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
                action = action + pick_ind
                print("should move to %d "  % trips[pick_ind * 3 + 1])
            
            obs, reward, done, info = self.env.step(action)
            print(obs)
            total_rewards += reward
        
        print("Total_rewards %lf" % total_rewards)

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()

    hourly_time = pickle.load(open('download/hourly_time', 'rb'))
    allday_time = pickle.load(open('download/allday_time', 'rb'))

    riders_train, riders_dev = read_input('download/Gridwise/gridwise_trips_sample_pit.csv')
    env = DriverEnv(riders_dev, hourly_time, allday_time, 608, baseline=False)

    for i in range(args.repeat):
        driver = Driver(env, 608, 2)
        driver.roundout()