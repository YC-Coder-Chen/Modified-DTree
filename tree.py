import pandas as pd
import copy
import numpy as np
import collections
import warnings
warnings.filterwarnings("ignore")
from functions import *

class tree_model:
    def __init__(self, ctg_col, ctn_col, tar_col, path):
        self.ctg_col = ctg_col
        self.ctn_col = ctn_col
        self.tar_col = tar_col
        self.path = path
    
    def pre_process(self, weight=1):
        data = pd.read_csv(self.path)
        data['weight'] = weight
        data_dummy = pd.get_dummies(data,columns=self.ctg_col, dummy_na=False)
        for i in self.ctg_col:
            try:
                index = data[data[i].isna()].index
                col = []
                for j in data_dummy.columns:
                    if j.startswith(i):
                        col.append(j)
                data_dummy.loc[index,col]=np.nan
            except:
                None
        self.dataset = data_dummy
    
    def build_tree(self, max_depth, min_size, min_improvement):
        root = get_split(self.dataset,self.tar_col)
        tar_col = self.tar_col
        split(root, max_depth, min_size, min_improvement, 1, tar_col)
        self.root = root
        
    def print_tree(self, depth=0):
        node = self.root
        def print_tree_inner(node, depth): 
            if isinstance(node, dict):
                print('%s[On%s < %.3f]' % ((depth*' ', (node['split_fet']), node['best_value'])))
                print_tree_inner(node['left'], depth+1)
                print_tree_inner(node['right'], depth+1)
            else:
                print('%s[%s]' % ((depth*' ', node)))
        print_tree_inner(node,depth)   
    
    def predict(self,dataset):
        tree=self.root
        return predict_with_na(tree, dataset)
     
    # fix the weight bug    
    def prune(self, validset, threshold):
        tree = self.root
        tar_col = self.tar_col 
        return prune_with_valid(tree, validset, threshold, tar_col)   



