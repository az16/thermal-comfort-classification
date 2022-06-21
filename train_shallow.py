from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, cross_validate
from dataloaders.path import Path
from sklearn.pipeline import make_pipeline
from argparse import ArgumentParser
from network.learning_models import RandomForest
from dataloaders.sklearn_dataloader import TC_Dataloader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
import pandas as pd
import numpy as np
from metrics import top_k_accuracy_score, visualize_feature_importance, accuracy_score, mean_absolute_error, mean_accuracy
from random import randint
from dataloaders.utils import feature_permutations, weight_dict
from sklearn.inspection import permutation_importance


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

def cross_validate_local(x_splits_train, y_splits_train, x_splits_val, y_splits_val, feature_names=[]):
    model = RandomForest(cv=False)
    val_scores = []
    importance_imp = []
    importance_perm = []
    for i in range(len(x_splits_train)):
        x_t, y_t, x_v, y_v = x_splits_train[i], y_splits_train[i], x_splits_val[i], y_splits_val[i]
        model.fit(x_t,y_t)
        #preds_train = model.predict(x_t)
        preds_val = model.predict(x_v)
        # # # print(y_v.values)
        # # # print(preds_val)
        print(accuracy_score(preds_val, y_v))
        val_scores.append(accuracy_score(preds_val, y_v))
        preds_val = model.predict(x_v)
      
        #print("Validation top_2_accuracy: {0}".format(top_k_accuracy_score(y_v, preds_val, k=2)))
        clf_tree(preds_val, y_v, [-3, -2, -1, 0, 1, 2, 3], "test classifier {0}".format(i))
        #print("Train accuracy: {0}".format(accuracy_score(preds_train,y_t)))
        #print("Validation top_1_accuracy: {0}".format(accuracy_score(preds_val, y_v)))
        feature_importance_imp = model.feature_importances()
        feature_importance_perm = permutation_importance(model.rf, x_v, y_v, random_state=0)["importances_mean"]
        importance_imp.append(feature_importance_imp)
        importance_perm.append(feature_importance_perm)
        # print(feature_importance_imp)
        # print(feature_importance_perm)
    
        
        #print(feature_importance)
    
    n = len(val_scores)
    print("Mean validation accuracy: {0}".format(sum(val_scores)/n))
    visualize_feature_importance(np.mean(np.array(importance_imp), axis=0), feature_names, i="impurity_based")
    visualize_feature_importance(np.mean(np.array(importance_perm), axis=0), feature_names, i="permutation_based")
    
def feature_selection(in_features):
    performances = []
    feature_log = []
    permutations = [list(x) for x in feature_permutations(in_features, max_size=len(in_features))]
    model = RandomForest(n_estimators=75, max_depth=12, cv=False)
    clf = model.rf
    dataset = TC_Dataloader(full=True)
    for p in permutations:
        print(p)
        x, y = dataset.splits(mask=p)
        cv = cross_validate(clf, x, y, return_train_score=True, verbose=3, cv=10)
        feature_log.append(p)
        performances.append(np.mean(np.array(cv["test_score"])))
        
        if len(performances)>0:
            print("Current best performance: {0}".format(max(performances)))
            max_index = performances.index(max(performances))
            print("Current best feature combination: {0}".format(feature_log[max_index]))
    
    print("Best performance: {0}".format(max(performances)))
    max_index = performances.index(max(performances))
    print("Best feature combination: {0}".format(feature_log[max_index]))

def grid_search(param_grid):
    dataset = TC_Dataloader(full=True)
    x,y = dataset.splits(mask=['Ambient_Humidity', 'Ambient_Temperature', 'Radiation-Temp'])
    model = RandomForest(n_estimators=75, max_depth=12, cv=True)
    clf = model.rf
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, verbose=5)
    gs.fit(x,y)
    print("Best parameters: {0}".format(gs.best_params_))
    print("Best score: {0}".format(gs.best_score_))

def downsampling_selection(param_grid, limit=30):
    dataset = TC_Dataloader(full=True)
    mask=['Ambient_Humidity', 'Ambient_Temperature', 'Radiation-Temp']
    x, y = dataset.splits(mask=mask)
    df = x 
    df["Label"] = y 
    #df.Label = df.Label.astype(float)
    #print(df)
    max = 0
    best_sampling_rate = 0
    for i in range(29,2*limit):
        tmp = df[df.index % i == 0]
        #print(df)
        x = tmp[mask]
        y = tmp["Label"]
        model = RandomForest(cv=True)
        clf = model.rf
        gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, verbose=5)
        gs.fit(x,y)
        #print("Best parameters: {0}".format(gs.best_params_))
        print("score: {0}".format(gs.best_score_))
        print(i)
        if gs.best_score_ > max:
            max = gs.best_score_
            best_sampling_rate = i
    
    print("Best performance: {0}".format(max))
    print("Best sampling rate: {0}".format(best_sampling_rate))
    
def fit_random_forest(mask=['Ambient_Humidity', 'Ambient_Temperature', 'Radiation-Temp']):
    model = RandomForest(cv=False)
    dataset = TC_Dataloader(by_file=True) 
    X, Y, val_X, val_Y, test_X, test_Y = dataset.splits(mask=mask)
    model.fit(X, Y)
    #print(val_Y)
    #roc = roc_auc_score(val_Y, model.rf.predict_proba(val_X), multi_class='ovr')
    #print("ROC score: {0}".format(roc))
    val_preds = model.predict(val_X)
    clf_tree(val_preds, val_Y, [-1,0,1], "test classifier")
    print("Top 0 accuracy: {0}".format(accuracy_score(val_preds, val_Y)))
    print("Top 1 accuracy: {0}".format(top_k_accuracy_score(val_preds, val_Y)))
    print("Top 2 accuracy: {0}".format(top_k_accuracy_score(val_preds, val_Y, k=3)))
    # RocCurveDisplay.from_estimator(model.rf, val_X, val_Y)
    # plt.show()
    print(classification_report(val_Y, val_preds))
    print("MAE for validation: {0}".format(mean_absolute_error(val_preds, val_Y)))

def sk_cross_validate():
    model = RandomForest(cv=False)
    dataset = TC_Dataloader(full=True)
    x, y = dataset.splits()
    cv=cross_validate(model.rf, x, y, cv=10, verbose=5)
    print(cv)
            

if __name__ == "__main__":
    
    
    parser = ArgumentParser('Trains thermal comfort estimation models')
    parser.add_argument('--estimators', default=50, type=int, help='Number of estimators.')
    parser.add_argument('--depth', default=12, type=int, help='Max depth for tree descend.')
    parser.add_argument('--module', default='', help='The network module to be used for training')
    parser.add_argument('--columns', default=[], help='The number of variables used for training')
    
    args = parser.parse_args()
    
    in_features = [ "Gender","Weight","Bodytemp",
                    "Bodyfat",
                    "GSR",
                    "Radiation-Temp",	
                    "Wrist_Skin_Temperature",
                    "Sport-Last-Hour",
                    "Ambient_Humidity",
                    "Ambient_Temperature"]
    
    params_grid = {
                'n_estimators': [500],
                'max_depth': [6],
                'min_samples_split':[2],
                'min_samples_leaf':[3],
                'max_leaf_nodes': [None],
                'max_samples':[425,525,625],
                'random_state': [0],
                'max_features': ['auto'],
                #'class_weight': [None, 'balanced'],
                'criterion': ['gini'],
                'bootstrap':[True]}
    
    label_mapping = {-3:"Cold",
                     -2:"Cool",
                     -1:"Slightly Cool",
                     0:"Comfortable", 
                     1:"Sligthly Warm",
                     2:"Warm",
                     3:"Hot"}
    
    # dataset = TC_Dataloader(cv=True) #cols=["Ambient_Humidity", "Ambient_Temperature", "Gender", "Radiation-Temp"])
    # x_t, y_t, x_v, y_v = dataset.splits()
    #feature_names = dataset.independent
    label_names = ["Cold", "Cool", "Slightly Cool", "Comfortable", "Slightly Warm", "Warm", "Hot"]
    labels = [-3,-2,-1,0,1,2,3]
    #feature_selection(in_features=in_features)
    #grid_search(params_grid)
    #downsampling_selection(params_grid)
    fit_random_forest()
    #cross_validate_local(x_t, y_t, x_v, y_v, feature_names=dataset.independent)
    #sk_cross_validate()
    
     
    # r_i, r_l = get_random_prediciton_input(x_test, y_test)
    
    # print("Testing on random inputs from dataset..")
    # #print("Random Testing Result Mapping: (y_hat, y)")
    # test_lines = ["i;y_hat;y;correct\n"]
    # test_preds, test_labels = [], []
    # correct = ""
    # for i in range(0,len(r_i)):
    #     correct = "FALSE"
    #     tmp_i, tmp_l = r_i[i], label_mapping[r_l[i].values[0]]
    #     pred_num = model.predict(tmp_i)[0]
    #     pred = label_mapping[pred_num]
    #     if pred == tmp_l:
    #         correct = "TRUE"
    #     test_lines.append(";".join([tmp_i.to_string(header=None), pred, tmp_l, correct])+"\n")
    #     test_preds.append(pred_num)
    #     test_labels.append(r_l[i].values[0])
    # with open('D:/thermal-comfort-classification/sklearn_logs/test_results.csv', 'w') as fp:
    #     for item in test_lines:
    #         fp.write(item)
    # print("Mean test accuracy: {0}, mean absolute error: {1}".format(mean_accuracy(test_preds, test_labels), mean_absolute_error(test_preds, test_labels)))
    
    