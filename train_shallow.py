from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, cross_validate
from dataloaders.path import Path
from sklearn.pipeline import make_pipeline
from argparse import ArgumentParser
from network.learning_models import RandomForest
from dataloaders.sklearn_dataloader import TC_Dataloader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
from metrics import visualize_feature_importance, accuracy_score, mean_absolute_error, mean_accuracy
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

def get_random_prediciton_input(x, y, samples = 100):
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
    
    
    dataset = TC_Dataloader(by_file=True)
    #x, y = dataset.splits()
    x_t, y_t, x_v, y_v, x_test, y_test = dataset.splits()
    #x_t,x_v,y_t,y_v = train_test_split(x,y, test_size=0.2, random_state=1, stratify=y)
    #cross_validate(x_t,y_t,x_v,y_v, feature_names=dataset.independent)
    feature_names = dataset.independent
    label_names = ["Cold", "Cool", "Slightly Cool", "Comfortable", "Slightly Warm", "Warm", "Hot"]
    labels = [-3,-2,-1,0,1,2,3]  
    model = RandomForest(n_estimators=200, max_depth=16, cv=False)
    clf = model.rf  
    model.fit(x_t,y_t)
    
    preds_val = model.predict(x_v)
   
    print("OOB score: {0}".format(clf.oob_score_))
    print("Validation accuracy: {0}".format(accuracy_score(preds_val, y_v)))
    #clf_tree(preds_train, y_t, label_names, "train classifier")
    clf_tree(preds_val, y_v, labels, "test classifier")
    print(classification_report(y_v, preds_val, labels=labels, target_names=label_names))
    print("Computing feature importance")
    feature_importance = model.feature_importances()
    #print(feature_importance)
    visualize_feature_importance(feature_importance, feature_names)
    
    r_i, r_l = get_random_prediciton_input(x_v, y_v)
    
    print("Testing on random inputs from dataset..")
    #print("Random Testing Result Mapping: (y_hat, y)")
    test_lines = ["i;y_hat;y;correct\n"]
    test_preds, test_labels = [], []
    correct = ""
    for i in range(0,len(r_i)):
        correct = "FALSE"
        tmp_i, tmp_l = r_i[i], label_mapping[r_l[i].values[0]]
        pred_num = model.predict(tmp_i)[0]
        pred = label_mapping[pred_num]
        if pred == tmp_l:
            correct = "TRUE"
        test_lines.append(";".join([tmp_i.to_string(header=None), pred, tmp_l, correct])+"\n")
        test_preds.append(pred_num)
        test_labels.append(r_l[i].values[0])
    with open('D:/thermal-comfort-classification/sklearn_logs/test_results.csv', 'w') as fp:
        for item in test_lines:
            fp.write(item)
    print("Mean test accuracy: {0}, mean absolute error: {1}".format(mean_accuracy(test_preds, test_labels), mean_absolute_error(test_preds, test_labels)))
    
    