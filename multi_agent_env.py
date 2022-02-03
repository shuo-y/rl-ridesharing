import numpy as np
import torch
import gym
from gym import spaces


class MultiDriverEnv(gym.Env):
    def __init__(self, arms, num_agents, interval=300):
        super(MultiDriverEnv, self).__init__()
        self.total_time = 24 * 3600
        self.arms = arms
        self.num_agents = num_agents
        actions = [spaces.Discrete(arms) for _ in range(num_agents)]
        self.action_space = spaces.Tuple(actions)
        obs_space = num_agents # First consider bandit case each agent has a single loss
        self.observation_space = spaces.Box(low=-1, high=100000, shape=(obs_space,), dtype=np.float32)
        # TODO check what kinds of distribution makes sense
        alpha_n = [0.2 for _ in range(self.arms)]
        beta_n = [0.5 for _ in range(self.arms)]
        self.total_loss_samples = [torch.distributions.beta.Beta(torch.tensor(alpha_n[i]), torch.tensor(beta_n[i])) 
                                   for i in range(self.arms)]
        # TODO the loss should between [0, 1]
        # current loss on each arm not depends on specific agents
        # even only assume loss on each arm only depends on num
        # it is a different case

    def compute_loss_arm(self, arm_id, agent_list, loss):
        # Compute the total loss on a pulled arm
        # and then compute the loss of each agent on that arm
        arm_sample = self.total_loss_samples[arm_id]
        arm_loss = arm_sample.sample()
        # Assume each agent divides them
        loss_val = arm_loss / len(agent_list)
        for agent_id in agent_list:
            loss[agent_id] = loss_val.item()

    def step(self, action_n):
        # Multiple agents 
        # action_n a list of actions
        # actions https://stackoverflow.com/questions/44369938/openai-gym-environment-for-multi-agent-games
        loss = np.zeros(self.num_agents)
         
        arm_pulllist = [[] for _ in range(self.arms)]

        # action is the index of the arm
        for agent_id, action in enumerate(action_n):
            idx = action
            arm_pulllist[idx].append(agent_id)
        
        for arm_id, pull_agent_list in enumerate(arm_pulllist):
            if len(pull_agent_list) > 0:
                self.compute_loss_arm(arm_id, pull_agent_list, loss)
        
        return loss

class FIBagent:
    """
    A class represents a full information procedure
    external regret algorithm
    using polynomial weights algorithm
    """

    def __init__(self, arms, eta = 0.5):
        self.arms = arms
        self.weights = np.ones(self.arms)
        self.step = 0
        self.eta = eta

    def update_weights(self, loss):
        """
        loss is a self.arms vector
        """
        self.weights = self.weights - self.weights * loss
        self.step += 1
        

class Agent:

    def __init__(self, arms, block_size):
        self.arms = arms
        self.block_size = block_size 
        assert self.block_size > self.arms
        # Let block size equals the arm size
        self.step = 0
        self.isExplore = True
        self.pwagent = FIBagent(arms)
        self.prev_act = 0
        self.sample_loss = np.zeros(arms)
        # First explore and them commit
        # Compute the weights and then average them

    def pick_action(self):
        action_ps = self.pwagent.weights / (np.sum(self.pwagent.weights))
        m = torch.distributions.categorical.Categorical(torch.tensor(action_ps))
        action = m.sample().item()
        return action
    
    def act(self, loss):
        action = 0
        if self.step % self.block_size == 0:
            # sample the first arm 
            # no need to use the loss
            action = 0
            self.prev_act = action
        elif self.step % self.block_size < self.arms:
            # sample 1, ..., N - 1 arm
            self.sample_loss[self.prev_act] = loss
            action = self.step % self.block_size
            self.prev_act = action
        elif self.step % self.block_size == self.arms:
            # First action after exploration
            # Need to update the PW agent 
            self.sample_loss[self.prev_act] = loss
            self.pwagent.update_weights(self.sample_loss)

            action = self.pick_action()
            self.prev_act = action
        elif self.step % self.block_size > self.arms:
            # Only commit
            # no need to record loss
            action = self.pick_action()
            self.prev_act = action
        self.step += 1        
        return action

    def reset(self):
        selt.step = 0
        self.pwagent = FIBagent(self.arms)

if __name__ == '__main__':
    num_arms = 10
    num_agents = 2
    round = 100
    commit_rounds = 5
    env = MultiDriverEnv(num_arms, num_agents)

    # print(env.step([0, 2, 2]))
    # What if agents sample in the same order
    # or agents sample in different order

    agents = [Agent(num_arms, num_arms + commit_rounds) for _ in range(num_agents)]

    actions = [agent.act(1.0) for agent in agents]
    print(actions)
    for step in range(1, round):
        loss_n = env.step(actions)
        print(loss_n)
        print(" ")
        actions = [agents[i].act(loss_n[i]) for i in range(num_agents)]
        print(actions)
    