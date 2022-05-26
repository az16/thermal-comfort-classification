from cv2 import correctMatches
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from dataloaders.path import Path
from argparse import ArgumentParser
from network.learning_models import RandomForest, RandomForestRegressor
from dataloaders.sklearn_dataloader import TC_Dataloader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from metrics import visualize_feature_importance, accuracy_score
from random import randint


def clf_tree(x,y,label_names, name):
    #print(preds, labels)
    cfm = confusion_matrix(y, x, labels=label_names, normalize='true')
    #print(cfm)
    df = pd.DataFrame(cfm, index=label_names, columns=label_names)

    #visualization
    m_val = sns.heatmap(df, annot=True, fmt=".1%", cmap="Blues")
    m_val.set_yticklabels(m_val.get_yticklabels(), rotation=0, ha='right', size=11)
    m_val.set_xticklabels(m_val.get_xticklabels(), rotation=30, ha='right', size=11)
    plt.ylabel('Target Labels')
    plt.xlabel('Predicted Label')
    #plt.figure(figsize=(15, 15))
    fig = m_val.get_figure()
    fig.savefig("sklearn_logs/media/{0}.png".format(name))
    plt.close(fig)

def get_random_prediciton_input(x, y, samples = 5):
    prediction_input = []
    correct_label = []
    r,c = x.shape[0], x.shape[1]
    for i in range(samples):
        idx = randint(0,r)
        prediction_input.append(x[idx:idx+1])
        correct_label.append(y[idx:idx+1])
    
    return prediction_input, correct_label
    

if __name__ == "__main__":
    
    
    parser = ArgumentParser('Trains thermal comfort estimation models')
    parser.add_argument('--estimators', default=5, type=int, help='Number of estimators.')
    parser.add_argument('--depth', default=16, type=int, help='Max depth for tree descend.')
    parser.add_argument('--module', default='', help='The network module to be used for training')
    parser.add_argument('--columns', default=[], help='The number of variables used for training')
    
    args = parser.parse_args()
    
    params_grid = {
                 'n_estimators': [20],
                 'max_depth': [16],
                 'max_features': ["log2"]}
    
    label_mapping = {-3:"Cold",
                     -2:"Cool",
                     -1:"Slightly Cool",
                     0:"Comfortable", 
                     1:"Sligthly Warm",
                     2:"Warm",
                     3:"Hot"}
    
    
    dataset = TC_Dataloader()
    
    x_t, y_t, x_v, y_v = dataset.splits()
    x_v, x_test = x_v[:int(x_v.shape[0]/2)], x_v[int(x_v.shape[0]/2):]
    y_v, y_test = y_v[:int(y_v.shape[0]/2)], y_v[int(y_v.shape[0]/2):]
    print(x_v.shape, x_test.shape)
    feature_names = dataset.independent
    #label_names = ["Cold", "Cool", "Slightly Cool", "Comfortable", "Slightly Warm", "Warm", "Hot"]
    label_names = [-3,-2,-1,0,1,2,3]    
    
    print("Features: {0}".format(x_t.shape[1]))
    model = RandomForest(n_estimators=args.estimators, max_depth=args.depth, max_features=x_t.shape[1], cv=False)
    clf = model.rf
    
    # print(cross_val_score(clf, x_t, y_t, scoring="accuracy", cv=10))
    # grid_clf = GridSearchCV(clf, params_grid, cv=10, return_train_score=True, verbose=5)
    # grid_clf.fit(x_t, y_t)  
    model.fit(x_t,y_t)
    
    # print("Model fitting done. Testing ..")
    
    # # model = grid_clf.best_estimator_
    # # print("Best params: {0}".format(grid_clf.best_params_))
    # # print("Grid search results: {0}".format(grid_clf.cv_results_))
    
    preds_train = model.predict(x_t)
    preds_val = model.predict(x_v)
    
    #print("Train accuracy: {0}".format(accuracy_score(preds_train,y_t)))
    print("Test accuracy: {0}".format(accuracy_score(preds_val,y_v)))
    clf_tree(preds_train, y_t, label_names, "train classifier")
    clf_tree(preds_val, y_v, label_names, "test classifier")
    print("Computing feature importance")
    feature_importance = model.feature_importances()
    print(feature_importance)
    visualize_feature_importance(feature_importance, feature_names)
    
    r_i, r_l = get_random_prediciton_input(x_test, y_test)
    
    print("Testing on random inputs from dataset..")
    for i in range(0,len(r_i)):
        tmp_i, tmp_l = r_i[i], label_mapping[r_l[i].values[0]]
        pred = label_mapping[model.predict(tmp_i)[0]]
        print("Input: {0}\r\nModel prediction: {1}\r\nActual label: {2}".format(tmp_i, pred, tmp_l))
    
    