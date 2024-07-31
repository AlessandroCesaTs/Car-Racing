import gymnasium as gym
import matplotlib.pyplot as plt
from functions.gym_wrapper import *
from functions.system_predicting_nn import *
from functions.agent import *
from functions.video_making import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--time_horizon',type=int,default=5)

args = parser.parse_args()
time_horizon = args.time_horizon


env = gym.make('CarRacing-v2', continuous=False, render_mode='rgb_array',max_episode_steps=1000)
env = ImageEnv(env)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=systemPredictingNN()
model.load_state_dict(torch.load('./models/system_predictor_0.001.pt',map_location=device))

agent=Agent(model,time_horizon)

frames=[]
distances_from_street=[]
distances_from_grass=[]
on_street=[]
street_on_sight=[]

state, over = env.reset(), False
state=torch.tensor(state,dtype=torch.float32)
while not over:
    frames.append(env.render())
    action=agent.step(state)
    state, over = env.step(action)
    state=torch.tensor(state,dtype=torch.float32)
    
    distance_from_street,distance_from_grass,_=distance(state[3])
    distances_from_street.append(distance_from_street)
    distances_from_grass.append(distance_from_grass)
    on_street.append(distance_from_street==0.0)
    street_on_sight.append(distance_from_street<74)


make_video(frames,'videos/output_video_'+str(time_horizon)+'.mp4')

plt.plot(distances_from_street)
plt.savefig('plots/distances_from_street_'+str(time_horizon)+'.png')
plt.close()

plt.plot(distances_from_grass)
plt.savefig('plots/distances_from_grass_'+str(time_horizon)+'.png')
plt.close()

plt.plot(on_street)
plt.savefig('plots/on_street_'+str(time_horizon)+'.png')
plt.close()

plt.plot(street_on_sight)
plt.savefig('plots/street_on_sight_'+str(time_horizon)+'.png')
plt.close()
