
from torch import nn


class LSTM_two_layers(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()
        #definimos la capa LSTM: proceses sequential data
        self.lstm = nn.LSTM(input_size = input_size, #size of the vector for temporal instance
                            hidden_size = hidden_size , #size of the intern state
                            num_layers = 2, #number of stacked LSTM layers
                            batch_first=True,
                            bidirectional= False,
                            dropout = dropout) # the entry will have the following form (batch, seq_len, features)
        #capa densa final:maps hidden state output to predictions
        self.fc = nn.Linear(hidden_size, output_size) 

    #Define how the information goes through the model
    def forward(self, x): 
        # Self represents the instance of the class
        # x tensor with shape: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x) # out: all ocult states, h Is the last state and c is the last cell state
        last_hidden = out[:, -1, :]
        y_hat = self.fc(last_hidden) #take last time step
        return y_hat
