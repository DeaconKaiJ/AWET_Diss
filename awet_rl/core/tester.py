import os
import yaml
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
import custom_gym
import time

from stable_baselines3 import DDPG, TD3, SAC
from awet_rl import AWET_DDPG, AWET_TD3, AWET_SAC
from awet_rl.common.util import listdirs

def test_envs(env, model, num_episodes=100, render=False):
    reward = [] 
    eps_len = []
    start_time = time.time()
    for _ in range(num_episodes):
        r = 0
        obs,info = env.reset()
        for i in range(100):
            action, _states = model.predict(obs)
            obs, new_r, term,dones, info = env.step(action)
            if render:
                    env.render()
            r += new_r
            if (term or dones) == True:
                eps_len.append(i)
                break
            
        reward.append(r)
    reward = np.array(reward)

    eps_len = np.array(eps_len)
    test_time = time.time() - start_time
    return reward, eps_len, test_time

def Test(env_name, exp_path, model_name, num_episodes=100, render=False):
    path = f'{exp_path}/{model_name}/'
    seeds = listdirs(path)
    seeds.sort()
    print(env_name)
    results_df = pd.DataFrame(columns = ['env_name', 'model_name', 'seed', 'rew_avg', 'test_time', 'eps_len'])

    for seed in seeds:
        env = gym.make(env_name)
        if model_name.startswith("DDPG"): 
            model = DDPG.load(f'{exp_path}/{model_name}/{seed}/best_model.zip', env=env)
        elif model_name.startswith("TD3"): 
            model = TD3.load(f'{exp_path}/{model_name}/{seed}/best_model.zip', env=env)
        elif model_name.startswith("SAC"):  
            model = SAC.load(f'{exp_path}/{model_name}/{seed}/best_model.zip', env=env)
        elif model_name.startswith("AWET_DDPG"): 
            model = AWET_DDPG.load(f'{exp_path}/{model_name}/{seed}/best_model.zip', env=env)
        elif model_name.startswith("AWET_TD3"): 
            model = AWET_TD3.load(f'{exp_path}/{model_name}/{seed}/best_model.zip', env=env)
        elif model_name.startswith("AWET_SAC"):  
            model = AWET_SAC.load(f'{exp_path}/{model_name}/{seed}/best_model.zip', env=env)
        else:
            raise ValueError(f"The agent name must starts with 'DDPG', 'TD3', 'SAC', 'AWET_DDPG', 'AWET_TD3', or 'AWET_SAC' and not {model_name}")
   
        reward, eps_len, test_time = test_envs(env, model, num_episodes, render)
        
        results = [env_name, model_name, seed, round(reward.mean(),4),test_time, round(eps_len.mean(),4)]
        print(f'{env_name}, {model_name}, {seed}:')
        print(f'rew_avg = {results[3]}, test_time = {results[4]}, eps_len = {results[5]}')
        results_df = results_df.append({'env_name': results[0],
                                        'model_name': results[1],
                                        'seed': results[2],
                                        'rew_avg': results[3],
                                        'test_time': results[4],
                                        'eps_len': results[5],
                                        },
                                        ignore_index = True)

    return results_df

def Tester(params):
    print('=========== Testing Started !!!')
    env_name = params['general_params']['env_name']
    exp_path = f"experiments/{params['general_params']['env_name']}/{params['general_params']['exp_name']}"
    models = listdirs(exp_path)
    print(models)
    if 'tensorboard_logs' in models:
        models.remove('tensorboard_logs') # Ignore the tensorboard logs folder

    results_df = pd.DataFrame(columns = ['env_name', 'model_name', 'seed', 'res_mean', 'res_std', 'rew_avg', 'success_rate', 'test_time', 'eps_len'])

    for model in models:
        results_dict = Test(env_name, 
                            exp_path, 
                            model,
                            num_episodes=params['tester_params']['num_episodes'],
                            render=params['tester_params']['render'],
                            )
        results_df = results_df.append(results_dict, ignore_index = True)

    results_df.to_csv(f'{exp_path}/Test_results.csv', index=False)
    # print(Test_results)
    print('=========== Testing Finished !!!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an experiment using AWET algorithm')
    parser.add_argument('--params_path', type=str,   default='configs/humanoid/awet_td3.yml', help='parameters directory for training')
    args = parser.parse_args()

    # load paramaters:
    with open(args.params_path) as f:
        params = yaml.safe_load(f)  # params is dict

    Tester(params)