import torch.nn as nn
import torch
from network.computations import categoryFromOutput

class RNN(nn.Module):
    def __init__(self, input_size, output_size, n_layers, hidden_dim, dropout=0.2):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size=batch_size)
        lstm_out, hidden = self.lstm(x, hidden)
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.softmax(out)
        out = out[:,-1]
        # out = categoryFromOutput(out)
        return out
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden


if __name__ == "__main__":
    input_t = torch.randn((4,10,7))
    rnn = RNN(input_size=7, output_size=7, n_layers=1, hidden_dim=256, dropout=0.2)
    hidden = rnn.init_hidden(batch_size=4)
    print(rnn(input_t, hidden)[0])
    #print(rnn(input_t, hidden))
   
    
    