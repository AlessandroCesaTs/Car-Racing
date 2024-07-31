import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from functions.distance import *


class Agent:
    def __init__(self,model,time_horizon=5):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
        self.time_horizon=time_horizon
        self.model=model.to(self.device)
        
        model.eval()

        self.possible_actions=torch.arange(0, 5, dtype=torch.float32,device=self.device)
        self.steps_without_gas=0

        self.predicted_state=None
    
    def get_best_next_action(self,state):
        states=state.repeat(5,1,1,1)
        predicted_next_frames=self.model(states,self.possible_actions)
        distances=[distance(predicted_frame)[2] for predicted_frame in predicted_next_frames]
        best_action=np.argmax(distances)
        predicted_next_state=torch.cat((state[1:],predicted_next_frames[best_action].unsqueeze(0)))
        return predicted_next_state,distances[best_action]
    
    def get_best_next_action_horizon(self,state):
        states=state.repeat(5,1,1,1)
        predicted_frames=self.model(states,self.possible_actions)
        tot_distances=[distance(predicted_frame)[2] for predicted_frame in predicted_frames]
        for action in self.possible_actions:
            state=states[int(action)]
            predicted_frame=predicted_frames[int(action)].unsqueeze(0)
            predicted_next_state=torch.cat((state[1:],predicted_frame),axis=0)
            for t in range(self.time_horizon):
                predicted_next_state,timestep_distance=self.get_best_next_action(predicted_next_state)
                tot_distances[int(action)]+=timestep_distance
        best_action=np.argmax(tot_distances)
        return best_action
    
    def step(self,state):
        state=state.to(self.device)
        
        if self.steps_without_gas>3:
            self.steps_without_gas=0
            return 3
        else:
            action=self.get_best_next_action_horizon(state)
            if action !=3 :
                self.steps_without_gas+=1
            else:
                self.steps_without_gas=0
            return action
