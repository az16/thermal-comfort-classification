from turtle import forward
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
        
        print(input_size)
        self.layers = nn.Sequential(
            nn.Linear(input_size, input_size*2),
            #nn.BatchNorm1d(num_features=10),
            nn.ReLU(),
            nn.Linear(input_size*2, input_size*2),
            #nn.BatchNorm1d(num_features=10),
            nn.ReLU(),
            nn.Linear(input_size*2, num_categories),
            nn.Sigmoid()
        )
        # self.input_layer = nn.Linear(input_size, input_size*2)
        # self.hidden_1 = nn.Linear(input_size*2, input_size*2)
        # self.hidden_2 = nn.Linear(input_size*2, num_categories)
        # self.activation = nn.Sigmoid()
        

    def forward(self, x):
        x = self.layers(x)
        return torch.squeeze(x, dim=1)

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
        self.activation = nn.Sigmoid()
        #self.discretizer = Discretizer(num_classes)
        
    def forward(self, x):
        self.lstm.flatten_parameters() #use multi GPU capabilities
        _, (h_t, _) = self.lstm(x)
        x = h_t[-1]
        if self.n_layers == 1:
            x = self.dp_layer(x)
        x = self.fc1(x)
        x = self.dp_layer(x)
        x = self.fc2(x)
        x = self.activation(x)
        #x = self.discretizer(x)
        #print(x)
        
        return x #x.float()
    

class Discretizer(nn.Module):
    def __init__(self, categories) -> None:
        super().__init__()
        self.n = categories
        self.b = torch.arange(0.0,1.0, 1/(categories*2))[1:]
        self.b = self.b[1::2]
    
    def forward(self, x):
        B, _ = x.size()
        #cut_points = torch.repeat_interleave(torch.unsqueeze(self.b,dim=0), B, dim=0)
        #values = torch.ones((B,1))*x
        #return torch.squeeze(torch.bucketize(values, self.b))
        pad = torch.ones((B,5))
        if x.is_cuda:
            pad = pad.cuda()
            self.b = self.b.cuda()
        pad.requires_grad=True
        t = torch.multiply(pad,x)
        # print(t)
        c = torch.cat((x,t), dim=1)
        # print(c)
        # print(self.b)
        x = torch.sum((c[::]>self.b[::]), dim=1).float()
        x.requires_grad=True
        # print(x.requires_grad)
        return x

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
        self.rf = RandomForestClassifier(random_state=0)
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
    def __init__(self, n_estimators=500, max_depth=6, criterion='squared_error', bootstrap=True, cv=True, max_features="log2"):
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
    d = Discretizer(7)
    b = torch.arange(0.0,1.0, 1/14)[1:]
    b = b[1::2]
    c = torch.sigmoid(torch.randn((4,1)))
    B, _ = c.size() 
    t = torch.ones(B,5)*c
    c = torch.cat((c,t), dim=1)
    print(c)
    tmp = torch.sum((c[::]>b[::]), dim=1)
    print(tmp)

    #print(c)
    #print(torch.squeeze(d(c)))
    # b = torch.arange(0.0,1.0, 1/14)[1:]
    # #b = b[0::2]
    # c = b[1::2].contiguous()
    # print(c)
    # #b = torch.repeat_interleave(torch.unsqueeze(b,dim=0), 1, dim=0)
    # #print(b)
    # v = torch.ones((1,7))*0.8571
    # print(torch.bucketize(v, c))
    # print(v)
    # distances = torch.abs(torch.subtract(b,v))
    # print(distances)
    # min = torch.min(distances, dim=1)
    # print(min)
    # min = (distances == min).nonzero(as_tuple=True)[0]
    # print(min)
    