import sys
import random
import torch
from dataloaders.path import Path
from argparse import ArgumentParser
from network.learning_models import RandomForest
from dataloaders.sklearn_dataloader import TC_Dataloader
from metrics import visualize_feature_importance, accuracy_score

if __name__ == "__main__":
    
    
    parser = ArgumentParser('Trains thermal comfort estimation models')
    parser.add_argument('--estimators', default=100, type=int, help='Minimum number of epochs.')
    parser.add_argument('--sequence_window', type=int, default=0, help="Use thermal comfort dataset sequentially.")
    parser.add_argument('--module', default='', help='The network module to be used for training')
    parser.add_argument('--columns', default=[], help='The number of variables used for training')
    
    args = parser.parse_args()
    
    params_grid = {
                 'n_estimators': [10, 20, 30, 40, 50, 60],
                 'max_depth': [2, 4, 8, 16, 32, 64],
                 'max_features': []}
    
    dataset = TC_Dataloader()
    
    x_t, y_t, x_v, y_v = dataset.splits()
    
    feature_names = dataset.independent
    
    #print("Fitting model to training data..")
    model = RandomForest(args.estimators)
    
    model.fit(x_t,y_t)
    
    print("Model fitting done. Testing ..")
    
    preds = model.predict(x_v)
    
    print("Test accuracy: {0}".format(accuracy_score(preds,y_v)))
    
    print("Computing feature importance")
    feature_importance = model.feature_importances()
    print(feature_importance)
    visualize_feature_importance(feature_importance, feature_names)
    
    