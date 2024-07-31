import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import time
from functions.system_predicting_nn import *

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate',type=float,default=0.0005)
parser.add_argument('--epochs',type=int,default=100)

args = parser.parse_args()
learning_rate = args.learning_rate
EPOCHS = args.epochs

class CustomDataset(Dataset):
    def __init__(self,data):
        self.data=data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        transition=self.data[idx]
        state_action=transition[:2]
        next_state=transition[2]
        return state_action,next_state
    
def extract_from_batched_data(batched_data):
    state_action,next_state=batched_data

    state=state_action[0].to(torch.float).to(device)
    action=state_action[1].to(torch.float).to(device)
    next_state=next_state.to(torch.float).to(device)

    return state,action,next_state

device= torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

batch_size=128
validation_batch_size=int(1e4)
loss_function=nn.MSELoss()

model=systemPredictingNN().to(device)



optimizer=optim.Adam(model.parameters(),learning_rate)

dataset=pd.read_pickle('data/simulated_data.pickle')
training_set,validation_set=train_test_split(dataset,test_size=0.2)
training_dataset=CustomDataset(training_set)
validation_dataset=CustomDataset(validation_set)
training_loader=DataLoader(training_dataset,batch_size=batch_size)
validation_loader=DataLoader(validation_dataset,batch_size=validation_batch_size)


training_losses=[]
validation_losses=[]
lowest_validation_loss=float('inf')

start_time=time.time()

for epoch in range(EPOCHS+1):
    model.train()
    for _,batched_data in enumerate(training_loader):
        state,action,next_state=extract_from_batched_data(batched_data)
        
        predicted_next_state=model(state,action)

        loss=loss_function(next_state,predicted_next_state)

        optimizer.zero_grad()
        
        loss.backward()

        optimizer.step()

    model.eval()

    with torch.no_grad():
        training_loss=0
        validation_loss=0
        num_elem_training=0
        num_elem_validation=0

        for _,batched_data in enumerate(training_loader):
            state,action,next_state=extract_from_batched_data(batched_data)
            
            predicted_next_state=model(state,action)

            training_loss+=loss_function(next_state,predicted_next_state).item()

        training_losses.append(training_loss/len(training_loader))

        for _,batched_data in enumerate(validation_loader):
            state,action,next_state=extract_from_batched_data(batched_data)

            predicted_next_state=model(state,action)

            validation_loss+=loss_function(next_state,predicted_next_state).item()

        validation_loss_avg=validation_loss/len(validation_loader)
        validation_losses.append(validation_loss_avg)

        if validation_loss_avg<lowest_validation_loss:
            torch.save(model.state_dict(),'models/system_predictor_'+str(learning_rate)+'.pt')
            lowest_validation_loss=validation_loss_avg
            print(f"Saved best model at epoch {epoch}, whith loss {lowest_validation_loss}")

end_time=time.time()
print(f"Time: {(end_time-start_time)/60}")
plt.plot(training_losses,label='training')
plt.plot(validation_losses,label='validation')
plt.legend()
plt.savefig('plots/learn_system_plot_'+str(learning_rate)+'.png')
