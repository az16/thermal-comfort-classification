from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from dataloaders.path import Path
from argparse import ArgumentParser
from network.learning_models import RandomForest
from dataloaders.sklearn_dataloader import TC_Dataloader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from metrics import visualize_feature_importance, accuracy_score


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

if __name__ == "__main__":
    
    
    parser = ArgumentParser('Trains thermal comfort estimation models')
    parser.add_argument('--estimators', default=20, type=int, help='Number of estimators.')
    parser.add_argument('--depth', default=8, type=int, help='Max depth for tree descend.')
    parser.add_argument('--module', default='', help='The network module to be used for training')
    parser.add_argument('--columns', default=[], help='The number of variables used for training')
    
    args = parser.parse_args()
    
    params_grid = {
                 'n_estimators': [100],
                 'max_depth': [32],
                 'max_features': ["log2"]}
                #  'bootstrap': [True, False]}
    
    
    dataset = TC_Dataloader(full=True)
    
    #x_t, y_t, x_v, y_v = dataset.splits()
    x_t, y_t = dataset.splits()
    # x_t, x_v, y_t, y_v = train_test_split(x_t, y_t, test_size=0.3, random_state=0, shuffle=True)
    
    feature_names = dataset.independent
    #label_names = ["Cold", "Cool", "Slightly Cool", "Comfortable", "Slightly Warm", "Warm", "Hot"]
    label_names = [-3,-2,-1,0,1,2,3]
    # model = svm.SVC(decision_function_shape='ovo')
    # model.fit(x_t, y_t)
    # preds = model.predict(x_v)
    
    # print("Test accuracy: {0}".format(accuracy_score(preds, y_v)))
    
    
    #print("Fitting model to training data..")
    print("Features: {0}".format(x_t.shape[1]))
    model = RandomForest(n_estimators=args.estimators, max_depth=args.depth, max_features=x_t.shape[1], cv=False)
    clf = model.rf
    
    print(cross_val_score(clf, x_t, y_t, scoring="accuracy", cv=10))
    # grid_clf = GridSearchCV(clf, params_grid, cv=10, return_train_score=True, verbose=5)
    # grid_clf.fit(x_t, y_t)  
    # model.fit(x_t,y_t)
    
    # print("Model fitting done. Testing ..")
    
    # # model = grid_clf.best_estimator_
    # # print("Best params: {0}".format(grid_clf.best_params_))
    # # print("Grid search results: {0}".format(grid_clf.cv_results_))
    # preds_train = model.predict(x_t)
    # preds_val = model.predict(x_v)
    
    # print("Train accuracy: {0}".format(accuracy_score(preds_train,y_t)))
    # print("Test accuracy: {0}".format(accuracy_score(preds_val,y_v)))
    # clf_tree(preds_train, y_t, label_names, "train classifier")
    # clf_tree(preds_val, y_v, label_names, "test classifier")
    # print("Computing feature importance")
    # feature_importance = model.feature_importances()
    # #print(feature_importance)
    # visualize_feature_importance(feature_importance, feature_names)
    