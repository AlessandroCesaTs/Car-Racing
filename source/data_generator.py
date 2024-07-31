from functions.gym_wrapper import *
from functions.image_processing import *
import numpy as np
import pickle
from functions.distance import *
import time
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--num_of_transitions',type=int,default=55000)

args = parser.parse_args()
num_of_transitions = args.num_of_transitions

start_time=time.time()


env = gym.make('CarRacing-v2', continuous=False,max_episode_steps=2000)
env = ImageEnv(env)

transitions=[]

state = env.reset()

steps_without_gas=0
episode_step=0

for i in range(num_of_transitions):
    episode_step+=1

    if steps_without_gas>3:
        action=3
        steps_without_gas=0
    else:
        if i%10<5:
            probabilities=[0.15,0.4,0.15,0.15,0.15]
        else:
            probabilities=[0.15,0.15,0.4,0.15,0.15]
        action=np.random.choice([0,1,2,3,4],p=probabilities)
        if action !=3 :
            steps_without_gas+=1
        else:
            steps_without_gas=0

    next_state, over = env.step(action)
    
    transitions.append([state,action,next_state[3]])

    state=next_state

    if over or distance(next_state[3])[0]>=74:
        state = env.reset()
        episode_step=0

with open('data/simulated_data.pickle', 'wb') as f:
    pickle.dump(transitions, f)

end_time=time.time()
print(f"Time: {(end_time-start_time)/60}")

