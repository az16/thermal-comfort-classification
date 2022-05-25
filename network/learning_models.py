import torch.nn as nn
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
    def __init__(self, in_features, num_classes, n_layers=1, hidden_dim=256, dropout=0.2):
        super(RNN, self).__init__()
        """
        LSTM classifier without activation layer
        """
        self.n_layers = n_layers
        self.lstm = nn.LSTM(in_features, hidden_dim, n_layers, batch_first=True)
        self.dp_layer = nn.Dropout(dropout)
        if n_layers > 1:
            self.lstm = self.lstm = nn.LSTM(in_features, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        self.lstm.flatten_parameters() #use multi GPU capabilities
        _, (h_t, _) = self.lstm(x)
        x = h_t[-1]
        if self.n_layers > 1:
            x = self.dp_layer(x)
        x = self.fc(x)
        x = self.activation(x)
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
            self.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, criterion=critirion, bootstrap=bootstrap, verbose=2, class_weight=self.weight_dict())
        
    
    def fit(self, train_inputs, train_labels):
        print("Fitting random forest classifier to data..")
        self.rf.fit(train_inputs, train_labels)
    
    def feature_importances(self):
        return self.rf.feature_importances_
        
    def predict(self, x):
        return self.rf.predict(x)
    
    def weight_func(self, x):
        return 0.1*x^2+1
    
    def weight_dict(self):
        weights = {-3: self.weight_func(-3),
                   -2: self.weight_func(-2),
                   -1: self.weight_func(-1),
                   0: self.weight_func(0),
                   1: self.weight_func(1),
                   2: self.weight_func(2),
                   3: self.weight_func(3)}
        return weights

class RandomForestRegressor():
    def __init__(self, n_estimators=None, max_depth=None, criterion='squared_error', bootstrap=True, cv=True, max_features="log2"):
        self.rf = RandomForestClassifier()
        if not cv:
            self.rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, bootstrap=bootstrap)
        
    
    def fit(self, train_inputs, train_labels):
        print("Fitting random forest classifier to data..")
        self.rf.fit(train_inputs, train_labels)
    
    def feature_importances(self):
        return self.rf.feature_importances_
        
    def predict(self, x):
        return self.rf.predict(x)

if __name__ == "__main__":
    t = torch.randn((4,10,4))
    model = RNN(4,7)
    print(torch.sum(model(t), dim=1))