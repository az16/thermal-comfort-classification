
from numpy import int8
import torch

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    #top_i = torch.squeeze(top_i)
    # print(top_i)
    # category_i = top_i[0].item()
    # print(category_i)
    return tc_categories(top_i)

def tc_categories(prediction_idx):
    categories = [-3, -2, -1, 0, 1, 2, 3]
    # if index == -1:
    #     return categories
    out = []
    for i in prediction_idx:
        out.append([categories[i[0]]])
    
    return torch.LongTensor(out)

