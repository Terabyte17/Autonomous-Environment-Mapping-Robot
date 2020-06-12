import gym
from baselines import deepq
import balance_bot

def callback(lcl, glb):
    is_solved = lcl['t']>100 and sum(lcl['episode_rewards'][-101:-1])/100 >= 50    #callback to terminate the learning if the mean 100 episode reward becomes greater than 50 
    return is_solved

env = gym.make("balancebot-v0") 
model = deepq.models.mlp([16, 16])
act = deepq.learn(env,q_func=model,lr=1e-3,max_timesteps=200000,buffer_size=50000,exploration_fraction=0.1,exploration_final_eps=0.02,print_freq=10,callback=callback)
act.save("balance.pkl")
