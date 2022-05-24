import torch.nn as nn
import torch
from sklearn.ensemble import RandomForestClassifier
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
    def __init__(self, in_features, num_classes, n_layers=1, hidden_dim=10, dropout=0.2):
        super(RNN, self).__init__()
        """
        LSTM classifier without activation layer
        """
        
        self.lstm = nn.LSTM(in_features, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dp = nn.Dropout(dropout)
        self.tan = nn.Tanh()
        
    def forward(self, x):
        self.lstm.flatten_parameters() #use multi GPU capabilities
        _, (h_t, _) = self.lstm(x)
        x = h_t[-1]
        #x = self.dp(x)
        x = self.fc(x)
        #x = self.tan(x)
        #x *= 3 #scale to [-3,3]
        return x #x.float()
    
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


class RandomForest():
    def __init__(self, n_estimators=None, max_depth=None, critirion='gini', bootstrap=True, cv=True, max_features="log2"):
        self.rf = RandomForestClassifier()
        if not cv:
            self.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, random_state=0, criterion=critirion, bootstrap=bootstrap, verbose=2, min_samples_leaf=6)
        
    
    def fit(self, train_inputs, train_labels):
        print("Fitting random forest classifier to data..")
        self.rf.fit(train_inputs, train_labels)
    
    def feature_importances(self):
        return self.rf.feature_importances_
        
    def predict(self, x):
        return self.rf.predict(x)