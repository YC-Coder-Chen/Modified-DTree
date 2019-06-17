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
tree.build_tree(100, 1, 0.0000000005)
# # tree.root
# tree.print_tree()
# tree.predict(tree.dataset)
# df_prune = tree.dataset
# df_prune['Density'] = 1
# tree.prune(df_prune, 0.5)
# print_tree_inner(tree.prune(df_prune, 0.5),0)