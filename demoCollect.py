import numpy as np
import gymnasium as gym
import custom_gym
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import argparse


def demoCollection(params):
    path = "ppoDemo_"+params.envs
    # Parallel environments
    env = gym.make(params.envs)#, terminate_when_unhealthy=False)

    model = PPO("MlpPolicy", env, verbose=1)
    print("Starting Learn")
    model.learn(total_timesteps=100000)
    model.save("ppoModelSave_"+params.envs)

    del model # remove to demonstrate saving and loading

    model = PPO.load("ppoModelSave_"+params.envs)

    count=0

    obsList,nextObsList,actions,rewards,dones =[],[],[],[],[]
    tempDemoOut,demoOut ={},{}
    print("Starting Demo Collect")
    while count !=99:
        obs,info= env.reset()
        while True:
            action, _states = model.predict(obs)
            new_obs, reward,truncate, terminate, info = env.step(action)
            #print(new_obs, reward,term, done, info)
            #env.render()
            obsList.append(obs)
            nextObsList.append(new_obs)
            actions.append(action)
            rewards.append([reward])
            obs = new_obs.copy()
            if (truncate or terminate) == True:
                dones.append([1.])
                tempDemoOut['obs'] = np.array(obsList)
                tempDemoOut['next_obs'] = np.array(nextObsList)
                tempDemoOut['actions'] = np.array(actions)
                tempDemoOut['rewards'] = np.array(rewards)
                tempDemoOut['dones'] = np.array(dones)
                demoOut["demo"+str(count)] = tempDemoOut
                count+=1
                #print(len(np.array(obsList)))
                obsList,nextObsList,actions,rewards,dones =[],[],[],[],[]
                tempDemoOut ={}
                break
            else:
                dones.append([0.])

    savePickle(path,demoOut)
    if params.load_pickle == True:
        loadPickle(path)

        

def savePickle(path,demoOut):
    with open(path+'.pkl', 'wb') as handle:
        pickle.dump(demoOut, handle, protocol=pickle.HIGHEST_PROTOCOL)
           

def loadPickle(path):
    with open(path+'.pkl', 'rb') as f:
        loaded_obj = pickle.load(f)
    print('loaded_obj is', loaded_obj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an experiment using AWET algorithm')
    parser.add_argument('--envs', type=str,   default='CustomHumanoid-v1', help='Environment to collect for e.g CustomHumanoid-v1')
    parser.add_argument('--load_pickle', type=bool, default=False, help="show save pickle to user")
    args = parser.parse_args()

    demoCollection(args)

"""
CustomHumanoid-v1
CustomHumanoid-v2

CustomHumanoidStandUp-v1
CustomHumanoidStandUp-v2
"""