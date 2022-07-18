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
