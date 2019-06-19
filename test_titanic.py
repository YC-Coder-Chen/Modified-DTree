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
train_path = './Testdata/Titanic_train.csv'
test_path = './Testdata/Titanic_test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

tree = tree_model(ctg_col,ctn_col,tar_col,train_path)
tree.pre_process()
tree.build_tree(5, 1, 0.005, 'entropy')
test = tree.pre_process(test)
test['Survived'] = tree.predict(test)

####tree = tree_model(ctg_col,ctn_col,tar_col,train_path)
####tree.pre_process()

# tree.dataset
#tree.build_tree(5, 1, 0.005, 'gini')
#### testset = pd.read_csv(test_path)
# testset = tree.pre_process(testset)
# prediction = tree.predict(testset)
# # tree.root
# tree.print_tree()
# tree.predict(tree.dataset)
# df_prune = tree.dataset
# df_prune['Density'] = 1
# tree.prune(df_prune, 0.5)
# print_tree_inner(tree.prune(df_prune, 0.5),0)