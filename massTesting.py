import os
import argparse
import yaml

from awet_rl.core import Trainer

def run_job(params):
    print('=========== Job Started !!!')
    Trainer(params)
    print('=========== Job Finished !!!')

configs = ["awet_td3","awet_ddpg","awet_sac","basic_td3","basic_ddpg","basic_sac"]
#configs = ["awet_sac","awet_ddpg","awet_td3"]
#configs = ["awet_ddpg","awet_td3"]
#configs = ["awet_td3"]

configLocation = "configs/humanoid/"
if __name__ == '__main__':
    print(configs)
    for x in range(len(configs)):
        print(configs[x])
        arg = configLocation + configs[x] + ".yml"
        print(arg)
        with open(arg) as f:
            params = yaml.safe_load(f)  # params is dict
        run_job(params)
    

"""
--params_path configs/humanoid/awet_td3.yml
--params_path configs/pendulum/awet_td3.yml
--params_path configs/pusher/awet_td3.yml
--params_path configs/reacher/awet_td3.yml
--params_path configs/humanoidStandup/awet_td3.yml

awet_sac
awet_td3
awet_ddpg
"""