import numpy as np
import pandas as pd
import torch
import gym
from gym.spaces import MultiDiscrete, Box, Discrete
from stream_rl.registry import register_env, create_reward
from ray.rllib.env.env_context import EnvContext
from collections import deque
import random
import sys

gym.logger.set_level(40)


@register_env("EdgeRIC_simulator")
class EdgeRIC_simulator(gym.Env):
    """EdgeRIC Env: Simulation of the realtime RIC setup"""

    def __init__(self, config: EnvContext):
        self.seed = config["seed"]
        self.app = config["app"] 
        if self.seed != -1:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.T = config["T"]
        self.t = None
        self.num_UEs = config["num_UEs"]
        self.total_rbgs = config["num_RBGs"]
        self.cqi_map = config["cqi_map"]
        self.augment_state_space = config["augment_state_space"]

        # Delay mechanism
        self.state_delay = config["delay_state"]
        self.action_delay = config["delay_action"]
        self.state_history = deque(
            [np.array([0, 1] * self.num_UEs, dtype=np.float32)]
            * (self.state_delay + 1),
            maxlen=self.state_delay + 1,
        )
        if self.augment_state_space:
            self.state_history = deque(
                [np.array([0, 1, 0] * self.num_UEs, dtype=np.float32)]
                * (self.state_delay + 1),
                maxlen=self.state_delay + 1,
            )
        self.action_history = deque(
            [np.zeros(shape=(self.num_UEs,), dtype=np.float32)]
            * (self.action_delay + 1),
            maxlen=self.action_delay + 1,
        )
        # print(self.action_history)
        # a=input("check")

        # Backlog Buffer Elements
        self.max_len_backlog = int(config["base_station"]["max_len"])
        self.backlog_lens = []
        self.backlog_population_params = config["backlog_population"]
        if (
            len(self.backlog_population_params) != self.num_UEs
        ):  # same params to all UEs backlog population
            self.backlog_population_params = [
                self.backlog_population_params
            ] * self.num_UEs

        # CQI Elements
        self.cqis = []
        self.cqi_traces = [
            pd.read_csv(config["cqi_traces"][ue]).squeeze().tolist()
            for ue in range(self.num_UEs)
        ]
        self.cqi_timesteps = [None] * self.num_UEs

        # Action and Observation Space Definitions
        self.action_space_type = config["action_space_type"]
        if self.action_space_type == "continuous":
            self.action_space = Box(
                low=0.0, high=1.0, shape=(self.num_UEs,), dtype=np.float32
            )
        elif self.action_space_type == "discrete":
            self.action_space = Discrete(config["num_RBGs"])
            assert (
                self.num_UEs == 1
            ), "Whittle learning supoported for only one UE currently!!"
        elif self.action_space_type == "binary":
            #self.action_space = Discrete(2)
            self.action_space = Discrete(3)
            self.binary_high_RBGs = config["binary_high_RBGs"]
            self.binary_low_RBGs = config["binary_low_RBGs"]
            self.binary_zero_RBGs = config["binary_zero_RBGs"]
            self.num_RBGs = config["num_RBGs"]
        self.cost_high_action = config["cost_high_action"]
        # assert (
        #     self.num_UEs == 1
        # ), "Whittle learning supoported for only one UE currently!!"
        self.num_state_variables = 2
        self.observation_space = Box(
            low=np.array([0, 1] * self.num_UEs),
            high=np.array([self.max_len_backlog, 15] * self.num_UEs),
            dtype=np.float32,
        )
        if self.augment_state_space:
            self.num_state_variables = 3
            self.observation_space = Box(
                low=np.array([0, 1, 0] * self.num_UEs),
                high=np.array(
                    [self.max_len_backlog, 15, 15 * self.max_len_backlog] * self.num_UEs
                ),
                dtype=np.float32,
            )

        self.normalize_state_space = config["normalize_state_space"]
        self.normalizer = (
            (
                np.array(
                    [
                        self.max_len_backlog ,
                        15,
                        self.max_len_backlog * 15,
                    ]
                    * self.num_UEs,
                    dtype=np.float32,
                )
                if self.augment_state_space
                else np.array(
                    [
                        self.max_len_backlog,
                        15,
                    ]
                    * self.num_UEs,
                    dtype=np.float32,
                )
            )
            if self.normalize_state_space
            else 1
        )

        self.reward_func = create_reward(config["reward"])
        #self.observation_space.seed(self.seed)
        self.tsls = 0
        self.interarrivaltimebursty = int(np.random.choice([50,100,150,200],size=1,p=None))

    def reset(self):
        self.interarrivaltimebursty = int(np.random.choice([50,100,150,200],size=1,p=None))
        self.t = 0
        self.tsls = 0
        self.backlog_lens = [0] * self.num_UEs
        self.cqi_timesteps = [
            random.randint(0, len(self.cqi_traces[ue]) - 1)
            for ue in range(self.num_UEs)
        ]
        self.cqis = [
            self.cqi_traces[ue][self.cqi_timesteps[ue]] for ue in range(self.num_UEs)
        ]
        self.back_pressures = [
            cqi * backlog_len for cqi, backlog_len in zip(self.cqis, self.backlog_lens)
        ]

        if self.augment_state_space:
            init_state = np.array(
                [
                    param[ue]
                    for ue in range(self.num_UEs)
                    for param in (self.backlog_lens, self.cqis, self.back_pressures)
                ],
                dtype=np.float32,
            )  # [BL1, CQI1, BP1, BL2, CQI2, BP2.....]
        else:
            init_state = np.array(
                [
                    param[ue]
                    for ue in range(self.num_UEs)
                    for param in (self.backlog_lens, self.cqis)
                ],
                dtype=np.float32,
            )  # [BL1, CQI1, BL2, CQI2,.....]

        self.state_history.append(init_state)
        # print(f"sate history:{self.state_history}")

        # print(f"s0 = {(self.state_history[0] / self.normalizer)}")
        return self.state_history[0] / self.normalizer, self.backlog_lens[0], self.cqis[0]

    def step(self, action):
        """Order of operations within a step - transfers from :
        1.) Backlog buffer to playback buffer
        2.) Cloud to backlog buffer
        """
        # print(f"\n\nstate of system at start of time {self.t+1}\n {self.backlog_lens}, {self.cqis}")
        # print(f"action recommended by agent:{action}")
        # Add delay to action
        self.action_history.append(action)
        action = self.action_history[0]
        if self.action_space_type == "continuous":
            action = np.clip(
                action, a_min=0.00000001, a_max=1.0
            )  # Project action back to action space + add epsilon to prevent divide by zero error
        if self.action_space_type == "discrete":
            raise NotImplementedError(
                "Haven't figured out the specifics of this action space!"
            )

        # Update time
        self.t += 1

        # Update TSLS
        #self.tsls = 0 if action == self.action_space.n-1 else self.tsls + 1

        # Update CQI for all UEs according to trace
        if self.app == "mmtc":
            if self.backlog_lens[0] != 0:
                self.tsls = self.tsls + 1
            elif self.backlog_lens[0] == 0:
                self.tsls = 0

        elif self.app == "embb" or self.app == "XR" or self.app == "urllc":            
            if action != self.action_space.n-1 and self.backlog_lens[0] != 0:
                #self.tsls = 0  
                self.tsls = self.tsls + 1
                #print("action is high " + str(self.tsls))
            #elif BL == 0:
            #    self.tsls = 0 
                #print("Buffer is zero " + str(self.tsls))   
            else:
                self.tsls = 0

        total_bytes_transferred = 0
        for ue in range(self.num_UEs):
            self.cqi_timesteps[ue] += 1
            self.cqi_timesteps[ue] %= len(self.cqi_traces[ue])
            self.cqis[ue] = self.cqi_traces[ue][self.cqi_timesteps[ue]]

            # Compute RBGs allocated for this UE
            if self.action_space_type == "continuous":
                percentage_RBG = action[ue] / sum(action)
                allocated_RBG = np.round(percentage_RBG * self.total_rbgs)
            elif self.action_space_type == "discrete":
                allocated_RBG = action
            elif self.action_space_type == "binary":
                # allocated_RBG = (
                #     self.binary_high_RBGs if action else self.binary_low_RBGs
                # )
                if action == self.action_space.n-1:
                    allocated_RBG = self.binary_high_RBGs
                elif action == self.action_space.n-2:
                    allocated_RBG = self.binary_low_RBGs
                else:
                    allocated_RBG = self.binary_zero_RBGs


            # Transfer data from BL to UE
            mean, std = self.cqi_map[self.cqis[ue]]
            bytes_transferred = (
                allocated_RBG
                * np.random.normal(mean, std)
                # * np.random.binomial(1, 0.9)  # BLER 10%
                * 1000
            ) // 8
            bytes_transferred = min(bytes_transferred, self.backlog_lens[ue])
            # print(bytes_transferred)
            total_bytes_transferred += bytes_transferred
            self.backlog_lens[ue] -= bytes_transferred
            # Update BLs
            inter_arrival_time = 0
            chunk_size = 0
            if self.app == "embb":
                inter_arrival_time = 10 
                chunk_size = 6250
            elif self.app == "XR":
                inter_arrival_time = 10
                chunk_size = 9534.9

            else:
                inter_arrival_time = self.interarrivaltimebursty
                if self.app == "urllc":
                    chunk_size = 250000
                else:
                    chunk_size = 375000
            if self.t % inter_arrival_time == 0:
                if self.app == "urllc" or self.app == "mmtc":
                    self.interarrivaltimebursty = self.t+int(np.random.choice([500,1000,1500,2000],size=1,p=None))
                self.backlog_lens[ue] += chunk_size
                self.backlog_lens[ue] = min(self.backlog_lens[ue], self.max_len_backlog)
        
        if action == self.action_space.n-1:
            reward = (
                self.reward_func(total_bytes_transferred, self.backlog_lens)/1000
                - (self.cost_high_action/1000)#*(self.binary_high_RBGs/self.num_RBGs)
                )
        # elif action == self.action_space.n-2:
        #     reward = (
        #         self.reward_func(total_bytes_transferred, self.backlog_lens)/1000
        #         - (self.cost_high_action/1000)*(self.binary_low_RBGs/self.num_RBGs)
        #         )
        else:
            reward = self.reward_func(total_bytes_transferred, self.backlog_lens)/1000
        

        
        #print(reward,action,self.cost_high_action)
        if self.augment_state_space:
            self.back_pressures = [
                cqi * backlog_len
                for cqi, backlog_len in zip(self.cqis, self.backlog_lens)
            ]
            next_state = np.array(
                [
                    param[ue]
                    for ue in range(self.num_UEs)
                    for param in (self.backlog_lens, self.cqis, self.back_pressures)
                ],
                dtype=np.float32,
            )  # [BL1, CQI1, BP1, BL2, CQI2, BP2,.....]
        else:
            next_state = np.array(
                [
                    param[ue]
                    for ue in range(self.num_UEs)
                    for param in (self.backlog_lens, self.cqis)
                ],
                dtype=np.float32,
            )  # [BL1, CQI1, BL2, CQI2,.....]

        done = self.t == self.T
        info = {}
        self.state_history.append(next_state)  # Add delay to state observation
        # print(f"state of system at end of time {self.t}\n {self.backlog_lens}, {self.cqis}")
        
        # print(f"action : {self.action_history[0]}")
        # print(f"action_history: {self.action_history}")
        # print(f"reward: {reward}")
        # a=input(f"{self.t} continue?")
        # print(reward)
        return self.state_history[0] / self.normalizer, reward, done, info, allocated_RBG, self.tsls, self.backlog_lens[0], self.cqis[0]
