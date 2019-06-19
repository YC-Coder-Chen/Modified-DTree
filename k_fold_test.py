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
path = './Testdata/train1.csv'

tree = tree_model(ctg_col,ctn_col,tar_col,path)
tree.pre_process()

# tree.dataset
tree.k_fold(5)