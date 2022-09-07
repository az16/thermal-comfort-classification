from utils import narrow_labels
from path import Path
from utils import *
import pandas as pd 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class PMV_Results():
    """
    Loads .csv and computes pmv index to compare pmv labels with model labels
    """
    def __init__(self): 
        self.columns = ["Radiation-Temp", "Ambient_Temperature", "Ambient_Humidity", "Clothing-Level", "Label"] #first for input features needed for pmv calculation, labels needed for comparison
        self.load_file_contents()
    
    def load_file_contents(self):
        """
        Loads .csv data as np.array and splits input signals/labels.


        Args:
            root (str): path of the root directory
            split (str): training or validation split string
        """
        
        file_names = []#os.listdir("./dataloaders/splits/")
        #print(file_names)
        with open("./dataloaders/splits/{0}_{1}.txt".format("validation", 60)) as file:
            lines = file.readlines()
            file_names = [line.rstrip() for line in lines]
        assert len(file_names) > 0; "No files found at {0}".format(Path.db_root_dir("tcs"))
    
        #load .csv contents as list
        print("Calculating PMV results..")
        self.df = pd.concat([pd.DataFrame(pd.read_csv(Path.db_root_dir("tcs")+x, delimiter=";"), columns = self.columns) for x in file_names])
        print("Adding to dataframe..")
        
        #contiuous preds
        self.df["PMV"] = pmv_calc(self.df["Radiation-Temp"], self.df["Clothing-Level"], self.df["Ambient_Temperature"], self.df["Ambient_Humidity"]) #calls pythermalcomfort api to calculate PMV based on ASHRAE standard
        #round preds to compare for accuracy 
        self.df["PMV_rounded"] = np.round(self.df["PMV"], 0).astype(np.int64)
        
        #gt is at self.df["Label"]

        #clf_tree(self.df["PMV_rounded"].values, self.df["Label"].values, [-3, -2, -1, 0, 1, 2, 3], "PMV_cm") #Use this to generate a confusion matrix
        print("Computing accuracy..")
        correct_7 = np.sum(self.df["PMV_rounded"].values==self.df["Label"].values)
        correct_3 = np.sum(reduce_scale(self.df, "PMV_rounded", scale=3)==reduce_scale(self.df, "Label", scale=3))
        correct_2 = np.sum(reduce_scale(self.df, "PMV_rounded", scale=2)==reduce_scale(self.df, "Label", scale=2))
        total = self.df.shape[0]

        self.pmv_accuracy_7 = correct_7/total
        self.pmv_accuracy_3 = correct_3/total
        self.pmv_accuracy_2 = correct_2/total

def reduce_scale(df, column, scale=2):
    if scale==2:
        #print("2-Point scale transformation.")
        #df.loc[(df[column] == 0), column] = 0
        #df.loc[(df[column] == 1), column] = 1
        df.loc[(df[column] == -1), column] = 1
        df.loc[(df[column] == -3), column] = 1
        df.loc[(df[column] == -2), column] = 1
        df.loc[(df[column] == 2), column] = 1
        df.loc[(df[column] == 3), column] = 1
        print("{0}: all 1 {1}, all 0 {2}".format(column, (df[column].values==1).all(), (df[column].values==0).all()))
        return df[column].values
    elif scale==3:
        #print("3-Point scale transformation.")

        df.loc[(df[column] == -1), column] = 0
        df.loc[(df[column] == -3),column] = -1
        df.loc[(df[column] == -2), column] = -1
        #df.loc[(df[column] == 0), column] = 0
        df.loc[(df[column] == 1), column] = 0
        df.loc[(df[column] == 2), column] = 1
        df.loc[(df[column] == 3), column] = 1
        print("{0}: all 1 {1}, all 0 {2}".format(column, (df[column].values==1).all(), (df[column].values==0).all()))
        return df[column].values
    return df[column].values
   
def clf_tree(x,y,label_names, name):
    """
       Creates a confusion matrix given labels and prediction.


        Args:
            x: preds
            y: targets
            label_names: target label names
            names: pred label names
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

if __name__ == "__main__":
    
    pmv_res = PMV_Results()
    print("7-point accuracy: {0}".format(pmv_res.pmv_accuracy_7))    
    print("3-point accuracy: {0}".format(pmv_res.pmv_accuracy_3))    
    print("2-point accuracy: {0}".format(pmv_res.pmv_accuracy_2))    


    