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
    env = gym.make(params.envs)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("ppoModelSave_"+params.envs)
    vec_env = model.get_env()

    del model # remove to demonstrate saving and loading

    model = PPO.load("ppoModelSave_"+params.envs)

    obs = vec_env.reset()

    n=1
    count=0

    obsList,nextObsList,actions,rewards,dones =[],[],[],[],[]
    tempDemoOut,demoOut ={},{}

    while True:
        action, _states = model.predict(obs)
        new_obs, reward, done, info = vec_env.step(action)
        
        obsList.append(list(obs[0]))
        nextObsList.append(list(new_obs[0]))
        actions.append(list(action[0]))
        rewards.append([reward[0]])
        obs = new_obs.copy()
        if done[0] == True:
            if n ==50:
                dones.append([1.])
                tempDemoOut['obs'] = np.array(obsList)
                tempDemoOut['next_obs'] = np.array(nextObsList)
                tempDemoOut['actions'] = np.array(actions)
                tempDemoOut['rewards'] = np.array(rewards)
                tempDemoOut['dones'] = np.array(dones)
                demoOut["demo"+str(count)] = tempDemoOut
                count+=1

            obsList,nextObsList,actions,rewards,dones =[],[],[],[],[]
            tempDemoOut ={}
            n=0
        else:
            dones.append([0.])
        if count==99:
            savePickle(path,demoOut)
            break
        n=n+1
    if params.load_pickle == True:
        loadPickle(path)

        #vec_env.render("human")

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