import torch.nn as nn
import torch
from computations import categoryFromOutput

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


# if __name__ == "__main__":
#     input_t = torch.randn((10,1,7))
#     rnn = RNN(7, 256, 7)
#     hidden = rnn.initHidden()
    
#     print(categoryFromOutput(rnn(input_t[0], hidden)[0]))
    
    