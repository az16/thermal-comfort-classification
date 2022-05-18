import torch.nn as nn
import torch
#from network.computations import categoryFromOutput

class MLP(nn.Module):
    def __init__(self, input_size, num_categories):
        super(MLP, self).__init__()

        """
        Simple linear regression classifier without activation layer
        """
        
        self.lin_1 = nn.Linear(input_size, 1)
        

    def forward(self, x):
        #b,s,f = x.size()
        #x = x.view(b, f, s)
        x = torch.squeeze(x)
        # print(x.shape)
        # print(x)
        x = self.lin_1(x)
        #x = x.view(b,1,f)
        return torch.squeeze(x)

class RNN(nn.Module):
    def __init__(self, in_features, num_classes, n_layers=3, hidden_dim=256, dropout=0.75):
        super(RNN, self).__init__()
        """
        LSTM classifier without activation layer
        """
        
        self.lstm = nn.LSTM(in_features, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        self.lstm.flatten_parameters() #use multi GPU capabilities
        _, (h_t, _) = self.lstm(x)
        x = h_t[-1]
        
        return self.fc(x)
    
    # def init_hidden(self, batch_size):
    #     weight = next(self.parameters()).data
    #     hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
    #               weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
    #     return hidden


class CNN(nn.Module):
    def __init__(self, input_size, num_categories):
        super(CNN, self).__init__()

        """
        CNN classifier for image/keypoint input
        """
        
        

    def forward(self, x):
        pass