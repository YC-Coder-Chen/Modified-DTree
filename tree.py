import pandas as pd
import copy
import numpy as np
import collections
import warnings
warnings.filterwarnings("ignore")
from functions import *

class tree_model:
    def __init__(self, ctg_col, ctn_col, tar_col, path=''):
        self.ctg_col = ctg_col
        self.ctn_col = ctn_col
        self.tar_col = tar_col
        self.path = path
    
    def pre_process(self, data=pd.DataFrame([]),weight=1):
        return_ind = 1
        if data.empty:
            return_ind = 0
            try:
                data = pd.read_csv(self.path)
                self.original_data = data
            except:
                raise ValueError('If do not provide dataset, then must initial self.path')
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
        if return_ind==1:
            return data_dummy
    
    def build_tree(self, max_depth, min_size, min_improvement, dataset=pd.DataFrame([])):
        if dataset.empty:
            root = get_split(self.dataset,self.tar_col)
        else:
            if 'weight' in dataset.columns:
                print('You have defined your own weight!')
            else:
                dataset['weight']=1
            root = get_split(dataset,self.tar_col)
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
        if 'weight' in dataset.columns:
            print('You have defined your own weight!')
        else:
            dataset['weight']=1
        return predict_with_na(tree, dataset)
     
    # fix the weight bug    
    def prune(self, validset, threshold):
        tree = self.root
        tar_col = self.tar_col 
        return prune_with_valid(tree, validset, threshold, tar_col)   

    def cross_validation_split(self,df, k_folds,random_states = int(np.random.randint(low = 0, high = 10000, size =1))):
        dataset_split = list()
        dataset_copy = df.copy()
        fold_size = int(len(df) / k_folds)
        for i in range(k_folds):
            fold = dataset_copy.sample(n = fold_size,random_state = random_states)
            dataset_copy = dataset_copy.drop(fold.index)
            dataset_split.append(fold)
        return dataset_split

    def accuracy(self,actual,pred):
        if len(actual) != len(pred):
            raise Exception('Actual and prediction have different length')
        actual = actual.reshape(len(actual),1)
        pred = pred.reshape(len(actual),1)
        score = (actual == pred).sum()
        return score/len(actual)
    
    def k_fold(self, k_fold,threshold = 0.5,max_depth = 5, min_size =1, min_improvement = 0.005):
        folds = self.cross_validation_split(self.dataset, k_fold)
        full_dataset = self.dataset
        scores = list()
        i = 1
        for fold in folds:
            print('Working on the ' + str(i) + ' fold!')
            trainset = full_dataset.drop(fold.index)
            testset = fold
            new_tree = tree_model(self.ctg_col,self.ctn_col,self.tar_col)
            
            new_tree.dataset = trainset
            new_tree.build_tree(max_depth, min_size, min_improvement)
            testset['preds'] = new_tree.predict(testset)
            testset['preds'] = testset['preds'] >= threshold
            accu = self.accuracy(testset[self.tar_col].values,testset['preds'].values)
            print(accu)
            scores.append(accu)
            i = i + 1
        return scores
    

