from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from dataloaders.path import Path
from sklearn.pipeline import make_pipeline
from argparse import ArgumentParser
from network.learning_models import RandomForest
from dataloaders.sklearn_dataloader import TC_Dataloader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from metrics import visualize_feature_importance, accuracy_score, top_k_accuracy_score
from random import randint
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
# from sktime.datasets import load_basic_motions

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

def get_random_prediciton_input(x, y, samples = 10):
    prediction_input = []
    correct_label = []
    r,c = x.shape[0], x.shape[1]
    for i in range(samples):
        idx = randint(0,r)
        prediction_input.append(x[idx:idx+1])
        correct_label.append(y[idx:idx+1])
    
    return prediction_input, correct_label

def cross_validate(x_splits_train, y_splits_train, x_splits_val, y_splits_val, feature_names=[]):
    model = RandomForest(n_estimators=args.estimators, max_depth=None, cv=False)
    val_scores = []
    for i in range(len(x_splits_train)):
        x_t, y_t, x_v, y_v = x_splits_train[i], y_splits_train[i], x_splits_val[i], y_splits_val[i]
        model.fit(x_t,y_t)
        #preds_train = model.predict(x_t)
        preds_val = model.predict(x_v)
        # # # print(y_v.values)
        # # # print(preds_val)
        val_scores.append(accuracy_score(preds_val, y_v))
        preds_val = model.predict(x_v)
        # # # print(y_v.values)
        # # # print(preds_val)
        #print("Validation top_2_accuracy: {0}".format(top_k_accuracy_score(y_v, preds_val, k=2)))
        clf_tree(preds_val, y_v, [-1,0,1], "test classifier")
        #print("Train accuracy: {0}".format(accuracy_score(preds_train,y_t)))
        #print("Validation top_1_accuracy: {0}".format(accuracy_score(preds_val, y_v)))
        feature_importance = model.feature_importances()
        #print(feature_importance)
        visualize_feature_importance(feature_importance, feature_names, i=i)
    
    n = len(val_scores)
    print("Mean validation accuracy: {0}".format(sum(val_scores)/n))
            

if __name__ == "__main__":
    
    
    parser = ArgumentParser('Trains thermal comfort estimation models')
    parser.add_argument('--estimators', default=100, type=int, help='Number of estimators.')
    parser.add_argument('--depth', default=12, type=int, help='Max depth for tree descend.')
    parser.add_argument('--module', default='', help='The network module to be used for training')
    parser.add_argument('--columns', default=[], help='The number of variables used for training')
    
    args = parser.parse_args()
    
    params_grid = {
                 'n_estimators': [50, 100],
                 'max_depth': [None,32]}
                #  'min_samples_split':[2,4,6,8],
                # 'min_samples_leaf':[2,4,6,8]}
    
    label_mapping = {-3:"Cold",
                     -2:"Cool",
                     -1:"Slightly Cool",
                     0:"Comfortable", 
                     1:"Sligthly Warm",
                     2:"Warm",
                     3:"Hot"}
    
    
    dataset = TC_Dataloader(cv=True)
    #x, y = dataset.splits()
    x_t,y_t,x_v,y_v = dataset.splits()
    cross_validate(x_t,y_t,x_v,y_v, feature_names=dataset.independent)
    #x_t, y_t, x_v, y_v, x_test, y_test = dataset.splits()
    #x, y = pd.concat([x_t, x_v]), pd.concat([y_t, y_v])
    # y_t, y_v, y_test = y_t.to_numpy(), y_v.to_numpy(), y_test.to_numpy()
    # feature_names = dataset.independent
    # #label_names = ["Cold", "Cool", "Slightly Cool", "Comfortable", "Slightly Warm", "Warm", "Hot"]
    # label_names = [-3,-2,-1,0,1,2,3]    
        
    # #print("Features: {0}".format(x_t.shape[1]))
    # model = RandomForest(n_estimators=args.estimators, max_depth=args.depth, cv=False)
    # # scaler = StandardScaler()
    # clf = model.rf
    # # model = make_pipeline(scaler, clf)
    # # #print(cross_val_score(clf, x_v, y_v, scoring="accuracy", cv=5))
    # # grid_clf = GridSearchCV(clf, params_grid, cv=5, return_train_score=True, verbose=5)
    # # grid_clf.fit(x, y)  
    # model.fit(x_t,y_t)
    
    # #print("Model fitting done. Testing ..")
    
    # # model = grid_clf.best_estimator_
    # # print("Best params: {0}".format(grid_clf.best_params_))
    # # print("Grid search results: {0}".format(grid_clf.cv_results_))
    
    # preds_train = model.predict(x_t)
    # preds_val = model.predict(x_v)
    # # # # print(y_v.values)
    # # # # print(preds_val)
    # print("Train accuracy: {0}".format(accuracy_score(preds_train,y_t)))
    # print("Validation top_1_accuracy: {0}".format(accuracy_score(preds_val, y_v)))
    # #print("Validation top_2_accuracy: {0}".format(top_k_accuracy_score(y_v, preds_val, k=2)))
    # clf_tree(preds_train, y_t, label_names, "train classifier")
    # clf_tree(preds_val, y_v, label_names, "test classifier")
    # print("Computing feature importance")
    # feature_importance = model.feature_importances()
    # #print(feature_importance)
    # visualize_feature_importance(feature_importance, feature_names)
    
    # r_i, r_l = get_random_prediciton_input(x_test, y_test)
    
    # print("Testing on random inputs from dataset..")
    # print("Random Testing Result Mapping: (y_hat, y)")
    # for i in range(0,len(r_i)):
    #     tmp_i, tmp_l = r_i[i], label_mapping[r_l[i].values[0]]
    #     pred = label_mapping[model.predict(tmp_i)[0]]
    #     print("({0}, {1})".format(pred, tmp_l))
    
    