from sklearn import feature_extraction
import torch.nn as nn
import torch
import torchvision as tv
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
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, num_classes)
        self.activation = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        #self.lstm.flatten_parameters() #use multi GPU capabilities
        _, (h_t, _) = self.lstm(x)
        x = h_t[-1]
        if self.n_layers == 1:
            x = self.dp_layer(x)
        x = self.fc1(x)
        x = self.dp_layer(x)
        x = self.fc2(x)
        #x = self.activation(x)
        #x *= 3 #scale to [-3,3]
        return x #x.float()


class RCNN(nn.Module):
    def __init__(self, num_in_features, num_classes=7, hidden=512, n_layers=2, dropout=0.2):
        super(RCNN, self).__init__()
        
        self.n_layers = n_layers
        self.feature_extractor = tv.models.resnet18(pretrained=True, progress=True) #use pretrained resnet
        num_features = self.feature_extractor.fc.in_features + num_in_features
        
        self.feature_extractor.fc = Skip() #take resnet fully connected out
        
        self.dp_layer= nn.Dropout(dropout) #regularization
        self.lstm = nn.LSTM(num_features, hidden, n_layers, batch_first=True)#rnn
        if n_layers > 1:
            self.lstm = nn.LSTM(num_features, hidden, n_layers, batch_first=True, dropout=dropout)
            
        self.fc1 = nn.Linear(hidden, hidden//2)
        self.fc2 = nn.Linear(hidden//2, num_classes)
        
        #self.activation = nn.Softmax(dim=1) #Softmax activation with 7 classes 
        
    def forward(self, x):
        rgb, numeric = x
        _, S, _, _, _ = rgb.shape
        
        #Use resnet feature extractor to create sequence of features --> shape: B, S, self.feature_extractor.output_size
        tmp = [torch.unsqueeze(torch.cat((self.feature_extractor(rgb[:,i]), numeric[:,i]), dim=1), dim=1) for i in range(0,S)]
        tmp = torch.cat(tmp, dim=1)

        self.lstm.flatten_parameters() #use multi GPU capabilities for lstm
        _, (h_t, _) = self.lstm(tmp)
        x = h_t[-1]
        if self.n_layers == 1:
            x = self.dp_layer(x)
        x = self.fc1(x)
        x = self.dp_layer(x)
        x = self.fc2(x)
        #x = self.activation(x)
        #x *= 3 #scale to [-3,3]
        return x #x.float()
    
class Skip(nn.Module):
    def __init__(self):
        super(Skip, self).__init__()
    def forward(self, x):
        return x    

class RandomForest():
    def __init__(self, n_estimators=500, max_depth=6, critirion='gini', bootstrap=True, cv=True, max_features="auto", min_samples_leaf=3, min_samples_split=2):
        self.rf = RandomForestClassifier()
        if not cv:
            self.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, criterion=critirion, bootstrap=bootstrap, verbose=0, min_samples_leaf=3, min_samples_split=2,
                                             random_state=0, max_samples=525)
        
    
    def fit(self, train_inputs, train_labels):
        print("Fitting random forest classifier to data..")
        self.rf.fit(train_inputs, train_labels)
    
    def feature_importances(self):
        return self.rf.feature_importances_
        
    def predict(self, x):
        return self.rf.predict(x)
    
    def weight_func(self, x):
        return 0.1*(x**2)+1.0
        #r = [2.0, 1.5, 1.0, 1.0, 1.0, 1.25, 1.5]
        #return r[x+3]
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
    # t1 = torch.Tensor([0.8, 0.1, 0.05, 0.])
    # t2 = torch.randn((16,5,4))
    # t = (t1,t2)
    # # t = torch.Tensor([1.9, 1.9, 1.1, 1.0, 1.4, 1.1, 1.4])
    # # print(t.shape)
    # # print(t)
    # m = RCNN(num_in_features=4)
    # r = m(t)
    # print(r.shape)
    # #print(r)
    # Example of target with class indices
    loss = nn.CrossEntropyLoss()
    # input = torch.randn(3, 5, requires_grad=True)
    # target = torch.empty(3, dtype=torch.long).random_(5)
    # print(input)
    # print(target)
    # output = loss(input, target)
    # print(output)
    # output.backward()
    # # Example of target with class probabilities
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5).softmax(dim=1)
    print(input)
    print(target)
    output = loss(input, target)
    print(output)