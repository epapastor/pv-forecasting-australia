import torch
import torch.nn as nn

class GRU_two_layers(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size = input_size , hidden_size= hidden_size , num_layers=2, batch_first=True, bidirectional = False, dropout = dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        #Passes the sequences complete by the GRU layer
        # Out: hidden states of all the times and h_n es the last hidden state
        out, h_n =self.gru(x)
        # extract the last hidden state that contains information from the whole sequence
        last_hidden = out[:, -1, :]
        y_hat = self.fc(last_hidden)
        return y_hat
