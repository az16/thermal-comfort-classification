import glob
import os
import torch 

"""
    This file contains some quick access functions for training modules
"""

label_names = ["Cold", "Cool", "Slightly Cool", "Comfortable", "Slightly Warm", "Warm", "Hot"]

def categoryFromOutput(output):
    """
        Given a numerical label mapping, this method returns its string representation.

        Args:
            output(tensor): the model output

    """
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return tc_categories(category_i), category_i

def tc_categories(index=-1):
    """
        Given a string label, this method returns its list index

        Args:
            index (int): skip mapping for one label if -1.
            

    """
    categories = ["Cold", "Cool", "Slightly Cool", "Comfortable", "Slightly Warm", "Warm", "Hot"]
    if index == -1:
        return categories
    return categories[index]


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    
    data = np.load("all_real.npy", allow_pickle=True)
    labels = data[-1]
    ambient_temp = data[-3]
    cats = [[],[],[],[],[],[],[]]
    for temp, cat in zip(ambient_temp, labels):
        cats[cat].append(temp)
    # Visualize petal length distribution for all species
    fig, ax = plt.subplots(figsize=(12, 7))
    # Remove top and right border
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Remove y-axis tick marks
    ax.yaxis.set_ticks_position('none')
    # Add major gridlines in the y-axis
    ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
    # Set plot title
    ax.set_title('Distribution of petal length by species')
    # Set species names as labels for the boxplot
    setosa_petal_length = np.random.rand(50)
    versicolor_petal_length = np.random.rand(50)
    virginica_petal_length =  np.random.rand(50)
    dataset = cats
    labels = ["Cold", "Cool", "Slightly Cool", "Comfortable", "Slightly Warm", "Warm", "Hot"]
    ax.boxplot(dataset, labels=labels)
    plt.savefig("real.pdf")