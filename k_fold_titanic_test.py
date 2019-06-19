import pandas as pd
import copy
import numpy as np
import collections
import warnings
warnings.filterwarnings("ignore")
from functions import *
from tree import tree_model

# UDF inputs
ctg_col = ['Pclass','Sex','Embarked']
ctn_col = ['Age','SibSp', 'Parch','Fare']
tar_col = ['Survived']
path = './Testdata/Titanic_train.csv'

tree = tree_model(ctg_col,ctn_col,tar_col,path)
tree.pre_process()

# tree.dataset
a = tree.k_fold(5, 0.5, 10, 1, 0.005, 'entropy', 1234)
print(a)
print(np.mean(a))

a = tree.k_fold(5, 0.5, 10, 1, 0.005, 'gini', 1234)
print(a)
print(np.mean(a))
