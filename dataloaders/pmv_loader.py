#from dataloaders.dataset import *
from dataloaders.path import Path
from dataloaders.utils import *
# from path import Path
# from utils import *
import pandas as pd 

import numpy as np


class PMV_Results():
    """
    Loads .csv and computes pmv index to compare pmv labels with model labels
    """
    def __init__(self): 
        self.columns = ["Radiation-Temp", "PCE-Ambient-Temp", "Ambient_Humidity", "Clothing-Level", "Label"]
        self.cut_points = np.array([-2.5,-1.5,-0.5,0.5,1.5,2.5])
        self.load_file_contents()
    
    def load_file_contents(self):
        """
        Loads .csv data as np.array and splits input signals/labels.


        Args:
            root (str): path of the root directory
            split (str): training or validation split string
        """
        
        #find files
        #print("Searching for {0} files..".format(self.split))
        file_names = [] #os.listdir(Path.db_root_dir("tcs"))
        with open("./dataloaders/splits/{0}_{1}.txt".format("validation", 60)) as file:
            lines = file.readlines()
            file_names = [line.rstrip() for line in lines]
        assert len(file_names) > 0; "No files found at {0}".format(Path.db_root_dir("tcs"))
            
        file_names = [Path.db_root_dir("tcs")+x for x in file_names]
        #print("Found {0} {1} files at {2}".format(len(file_names),self.split,Path.db_root_dir("tcs")))

        #load .csv contents as list
        print("Calculating PMV results..")
        self.df = pd.concat([pd.DataFrame(pd.read_csv(x, delimiter=";"), columns = self.columns) for x in file_names])
        #print("Creating data frames..")
        self.df["PMV"] = pmv(self.df["Radiation-Temp"], self.df["Clothing-Level"], self.df["PCE-Ambient-Temp"], self.df["Ambient_Humidity"])
        x =  np.expand_dims(self.df["PMV"].values, axis=1)
        x = np.repeat(x, 6, axis=1)
        pad = np.ones((x.shape[0],6))
        c = np.multiply(pad,x[:])
        # print(t)
        #c = torch.cat((x,t), dim=1)
        #tmp = (c[::]>self.cut_points[::])
        idx = np.sum((c[::]>self.cut_points[::]), axis=1)
        predictions = idx-3
        #self.df["PMV_rounded"] = np.round(self.df["PMV"], 0).astype(np.int64)
        correct = np.sum(predictions==self.df["Label"].values)
        total = self.df.shape[0]
        print(np.concatenate((np.expand_dims(predictions, axis=1), np.expand_dims(self.df["Label"].values, axis=1)), axis=1))
        self.pmv_accuracy = correct/total
        self.mse = np.mean((predictions-self.df["Label"].values)**2)
        self.mae = np.mean(np.abs(predictions-self.df["Label"].values))
        #print("File contents loaded!")
        
if __name__ == "__main__":
    
    pmv_res = PMV_Results()
    print(pmv_res.pmv_accuracy)    
    print(pmv_res.mse)
    print(pmv_res.mae)
    #print(pmv_res.df["PMV_rounded"], pmv_res.df["Label"])

    