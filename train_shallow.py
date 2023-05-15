from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split, cross_val_score, cross_validate
from dataloaders.path import Path
from sklearn.pipeline import make_pipeline
from argparse import ArgumentParser
from network.learning_models import RandomForest
from dataloaders.sklearn_dataloader import TC_Dataloader
from dataloaders.ashrae import ASHRAE_Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
import pandas as pd
import numpy as np
from metrics import visualize_feature_importance, accuracy_score, mean_absolute_error, mean_accuracy
from random import randint
from dataloaders.utils import feature_permutations
from sklearn.inspection import permutation_importance
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def clf_tree(x,y,label_names, name):
    """
        Creates a confusion matrix given model predictions, labels, label names and file name

        Args:
            x (nparray): the model predictions
            y (nparray): the actual labels
            label_names (list): the label names (for axis creation)
            name (str): the file name to save the img of the confusion matrix
    """
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
    fig.savefig("sklearn_logs/media/{0}.pdf".format(name))
    plt.close(fig)

def cross_validate_local(x_splits_train, y_splits_train, x_splits_val, y_splits_val, feature_names=[]):
    """
        This method uses a cross validation scheme that uses dataframe permutation by participant rather than
        by csv line count. 
        
        Example: Particpant files 1-15 in training and 16-20 in validation rather than lines 1-1500000 in trainging and rest in validation

        Args:
            x_splits_train (list): the training splits for every fold
            y_splits_train (list): the training labels for every fold
            x_splits_val (list): the validation split for every fold
            y_splits_val (list): the validation labels for every fold
            feature_names (list, optional): If only certain features are supposed to be tested put them here. Defaults to [].
    """
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
        clf_tree(preds_val, y_v, [0, 1], "test classifier {0}".format(i))
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
    
def feature_selection(in_features, test=None):
    """
        Tests different features combinations of varying length for prediction performance on an unoptimzed
        random forest model.

        Args:
            in_features (list): _description_
            test (bool, optional): debugging flag to get more prints. Defaults to None.
    """
    performances = []
    feature_log = []
    permutations = [list(x) for x in tqdm(feature_permutations(in_features, max_size=len(in_features)))]
    if not test is None:
        permutations = test
    model = RandomForest(cv=True)
    clf = model.rf
    dataset = TC_Dataloader(full=True)
    for p in permutations:
        print(p)
        x, y = dataset.splits(mask=p)
        cv = cross_validate(clf, x, y, return_train_score=True, verbose=3, cv=5)
        feature_log.append(p)
        performances.append(np.mean(np.array(cv["test_score"])))
        
        if len(performances)>0:
            print("Current best performance: {0}".format(max(performances)))
            max_index = performances.index(max(performances))
            print("Current best feature combination: {0}".format(feature_log[max_index]))
    
    print("Best performance: {0}".format(max(performances)))
    max_index = performances.index(max(performances))
    print("Best feature combination: {0}".format(feature_log[max_index]))

def sk_feature_selection():
    """
        Does feature selection using only sklearn models and functions.
    """
    dataset = TC_Dataloader(full=True)
    x, y = dataset.splits()
    model = RandomForest(cv=False)
    model.fit(x,y)
    print(x)
    model = SelectFromModel(model.rf, prefit=True)
    #print(model.get_support())
    x_new = model.transform(x)
    print(x_new.shape)
    print(x_new)    

def grid_search(param_grid):
    """
        This method performs grid search with a fixed set of features and tests different model
        params.

        Args:
            param_grid (dict): the parameter dict that defines the parameter ranges to test during grid search.
    """
    dataset = TC_Dataloader(full=True)
    x,y = dataset.splits(mask=['Ambient_Humidity', 'Ambient_Temperature', 'Radiation-Temp'])
    model = RandomForest(cv=True)
    clf = model.rf
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, verbose=5)
    gs.fit(x,y)
    print("Best parameters: {0}".format(gs.best_params_))
    print("Best score: {0}".format(gs.best_score_))

def downsampling_selection(param_grid, limit=30):
    """
        Tests different downsampling rates and finds the optimum.

        Args:
            param_grid (_type_): the random forest parameter dict
            limit (int, optional): up to which sampling rate the method is meant to test. Defaults to 30.
    """
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
        gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, verbose=5)
        gs.fit(x,y)
        #print("Best parameters: {0}".format(gs.best_params_))
        print("score: {0}".format(gs.best_score_))
        print(i)
        if gs.best_score_ > max:
            max = gs.best_score_
            best_sampling_rate = i
    
    print("Best performance: {0}".format(max))
    print("Best sampling rate: {0}".format(best_sampling_rate))
    
def fit_random_forest(train="thermal_comfort", eval="thermal_comfort"):
    """
        Given a set of input features this method loads the necessary training data and fits a random forest classifier.
    

        Args:
            mask (list, optional): the features to used. Defaults to ['Ambient_Humidity', 'Ambient_Temperature', 'Radiation-Temp'].
    """
    model = RandomForest(cv=False)#RandomForest(cv=False)
    if train == "thermal_comfort":
        train_dataset = TC_Dataloader(by_file=True, cols=["Ambient_Humidity", "Ambient_Temperature", "Radiation-Temp"]) 
    elif train == "ashrae":
        train_dataset = ASHRAE_Dataset(path="H:/data/ASHRAE", split="all", cols=[49,30,40,14], scale=7)
    if eval == "thermal_comfort":
        eval_dataset = TC_Dataloader(by_file=True, cols=["Ambient_Humidity", "Ambient_Temperature", "Radiation-Temp"]) 
    elif eval == "ashrae":
        eval_dataset = ASHRAE_Dataset(path="H:/data/ASHRAE", split="all", cols=[49,30,40,14], scale=7)
    
    train_X, train_Y, _, _, _, _ = train_dataset.splits()
    _, _, val_X, val_Y, _, _ = eval_dataset.splits()

    train_X= np.asarray(train_X, dtype=np.float32)
    train_Y= np.asarray(train_Y, dtype=int)
    val_X= np.asarray(val_X, dtype=np.float32)
    val_Y= np.asarray(val_Y, dtype=int)

    #print(X.shape)
    #print(np.mean(cross_val_score(model.rf, X,Y, cv=10, verbose=3)))
    model.fit(train_X,train_Y)
    preds = model.predict(val_X)
    #print(val_Y)
    #roc = roc_auc_score(val_Y, model.rf.predict_proba(val_X), multi_class='ovr')
    #print("ROC score: {0}".format(roc))
    #val_preds = cross_val_predict(model.rf, X, Y, cv=10, verbose=3)
    #print(len(val_preds))
    #print(val_preds)
    #clf_tree(val_preds, Y, [0,1], "Test_x_Point")
    #print("Top 0 accuracy: {0}".format(accuracy_score(preds, val_Y)))
    # print("Top 1 accuracy: {0}".format(top_k_accuracy_score(val_preds, val_Y)))
    # print("Top 2 accuracy: {0}".format(top_k_accuracy_score(val_preds, val_Y, k=3)))
    
    feature_importance_imp = model.feature_importances()
    feature_importance_perm = permutation_importance(model.rf, val_X, val_Y, random_state=0)["importances_mean"]
    visualize_feature_importance(feature_importance_imp, eval_dataset.independent, i="impurity_based")
    visualize_feature_importance(feature_importance_perm, eval_dataset.independent, i="permutation_based")
    
    # RocCurveDisplay.from_estimator(model.rf, val_X, val_Y)
    # plt.show()
    #print(classification_report(val_Y, val_preds))
    #print("MAE for validation: {0}".format(mean_absolute_error(val_preds, val_Y)))
    np.save("rf_pred.npy", preds)
    np.save("rf_gt.npy", val_Y)

def sk_cross_validate():
    """
        Does cross validation using the sklearn methods instead of self-implemented ones
    """
    model = RandomForest(cv=False)
    dataset = TC_Dataloader(full=True)
    x, y = dataset.splits()
    cv=cross_validate(model.rf, x, y, cv=10, verbose=5)
    print(cv)
            

if __name__ == "__main__":
    import torch
    from torchmetrics import Accuracy
    from dataloaders.utils import label2idx, class7To2, class7To3
    
    parser = ArgumentParser('Trains thermal comfort estimation models')
    parser.add_argument('--estimators', default=50, type=int, help='Number of estimators.')
    parser.add_argument('--depth', default=12, type=int, help='Max depth for tree descend.')
    parser.add_argument('--module', default='', help='The network module to be used for training')
    parser.add_argument('--columns', default=[], help='The number of variables used for training')
    parser.add_argument('--train', default="thermal_comfort")
    parser.add_argument('--eval', default="thermal_comfort")
    
    
    args = parser.parse_args()
    
    in_features = [ "Gender","Weight","Height",
                    "Wrist_Skin_Temperature",
                    "Radiation-Temp",
                    "Sport-Last-Hour",
                    "Ambient_Humidity",
                    "Ambient_Temperature"]
    
    params_grid = {
                'n_estimators': [100,200,300,400],
                'max_depth': [6,8],
                'min_samples_split':[3],
                'min_samples_leaf':[2],
                'max_leaf_nodes': [None],
                'max_samples':[400,425,450,475,500],
                'random_state': [0]
                #'max_features': ['auto']}
                #'class_weight': [None, 'balanced'],
                #'criterion': ['gini', 'entropy'],
                #'bootstrap':[True]
    }
    
    label_mapping = {-3:"Cold",
                     -2:"Cool",
                     -1:"Slightly Cool",
                     0:"Comfortable", 
                     1:"Sligthly Warm",
                     2:"Warm",
                     3:"Hot"}
    
    #dataset = TC_Dataloader(cv=True) #cols=["Ambient_Humidity", "Ambient_Temperature", "Gender", "Radiation-Temp"])
    #x_t, y_t, x_v, y_v = dataset.splits(mask=['Ambient_Humidity', 'Ambient_Temperature', 'Radiation-Temp', 'Sport-Last-Hour'])
    #feature_names = dataset.independent
    label_names = ["Cold", "Cool", "Slightly Cool", "Comfortable", "Slightly Warm", "Warm", "Hot"]
    labels = [-3,-2,-1,0,1,2,3]
    #feature_selection(in_features=in_features)#, test=[["Ambient_Temperature","Ambient_Humidity","Radiation-Temp"]])
    #sk_feature_selection()
    #grid_search(params_grid)
    fit_random_forest(train=args.train, eval=args.eval)
    #downsampling_selection(params_grid)
    #fit_random_forest()
    #cross_validate_local(x_t, y_t, x_v, y_v, feature_names=dataset.independent)
    #sk_cross_validate()

    accuracy = Accuracy(num_classes=7, task="multiclass")
    accuracy3 = Accuracy(num_classes=3, task="multiclass")
    accuracy2 = Accuracy(num_classes=2, task="binary")

    preds = np.load("rf_pred.npy")
    gt = np.load("rf_gt.npy")

    gt = np.array([label2idx(v) for v in gt], dtype=int)
    preds = np.array([label2idx(v) for v in preds], dtype=int)

    fig, ax = plt.subplots()
            
    ax.spines["left"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["bottom"].set_color("white")   
    
    ConfusionMatrixDisplay.from_predictions(gt, preds, normalize='true', display_labels=["cold", "cool", "slightly cool", "comfortable", "slightly warm", "warm", "hot"], cmap=plt.cm.Blues, values_format=".1%", ax=ax, xticks_rotation="vertical")
    plt.savefig("rf_cm.pdf",bbox_inches='tight', pad_inches=0)

    
    fig, ax = plt.subplots()
            
    ax.spines["left"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["bottom"].set_color("white")
    ConfusionMatrixDisplay.from_predictions(class7To3(gt), class7To3(preds), normalize='true', display_labels=["cool", "comfortable", "warm"], cmap=plt.cm.Blues, values_format=".1%", ax=ax, xticks_rotation="vertical")
    plt.savefig("rf_3_cm.pdf",bbox_inches='tight', pad_inches=0)

    fig, ax = plt.subplots()
            
    ax.spines["left"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["bottom"].set_color("white")
    ConfusionMatrixDisplay.from_predictions(class7To2(gt), class7To2(preds), normalize='true', display_labels=["uncomfortable", "comfortable"], cmap=plt.cm.Blues, values_format=".1%", ax=ax, xticks_rotation="vertical")
    plt.savefig("rf_2_cm.pdf",bbox_inches='tight', pad_inches=0)

    gt = torch.from_numpy(gt)
    preds = torch.from_numpy(preds)

    acc = accuracy(preds, gt)
    acc3 = accuracy3(class7To3(preds), class7To3(gt))
    acc2 = accuracy2(class7To2(preds), class7To2(gt))

    print("Accuracy: {:.1f}\%".format(acc.item() * 100))
    print("Accuracy3: {:.1f}\%".format(acc3.item() * 100))
    print("Accuracy2: {:.1f}\%".format(acc2.item() * 100))