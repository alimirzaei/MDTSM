import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from datasets import SyntethicData
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
# Here we define our model as a class

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=2,latent_dim=2):
        super(LSTM, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear1 = nn.Linear(self.hidden_dim, latent_dim)
        self.linear2 = nn.Linear(latent_dim, output_dim)
    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.float().view((-1, self.batch_size, self.input_dim)))
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        latent = F.relu(self.linear1(lstm_out[-1].view(self.batch_size, -1)))
        output = self.linear2(latent)
        return output.view(self.batch_size, self.input_dim, -1)

input_dim = 1
hidden_dim = 50
batch_size = 32
num_layers = 2
output_dim = 40
latent_dim = 2

model = LSTM(input_dim, hidden_dim, latent_dim=latent_dim,batch_size=batch_size, output_dim=output_dim, num_layers=num_layers)

learning_rate = 0.01
num_epochs = 100

loss_fn = torch.nn.MSELoss(size_average=False)
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

#####################
# Train model
#####################

hist = np.zeros(num_epochs)
dataset=SyntethicData()
for t in range(num_epochs):
    # Clear stored gradient
    for batch_x, batch_y in DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True):
        model.zero_grad()
        
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        model.hidden = model.init_hidden()
        
        # Forward pass
        y_pred = model(batch_x)
        loss = loss_fn(y_pred.float(), batch_y.float())
        hist[t] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        print("Epoch ", t, "MSE: ", loss.item())
        optimiser.step()