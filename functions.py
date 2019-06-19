import numpy as np
import pandas as pd
import collections
import copy
 
def test_split(col_name, value, dataset):
    left = dataset[(dataset[col_name]<value)|(dataset[col_name].isna())]
    right = dataset[(dataset[col_name]>=value) | (dataset[col_name].isna())]
    p_left = dataset[(dataset[col_name]<value)]['weight'].sum()/dataset[~dataset[col_name].isnull()]['weight'].sum()
    p_right = dataset[(dataset[col_name]>=value)]['weight'].sum()/dataset[~dataset[col_name].isnull()]['weight'].sum()
    left.loc[dataset[dataset[col_name].isna()].index,'weight'] = left.loc[dataset[dataset[col_name].isna()].index,'weight'] * p_left
    right.loc[dataset[dataset[col_name].isna()].index,'weight'] = right.loc[dataset[dataset[col_name].isna()].index,'weight'] * p_right
    return left,right

def cpu_median(li):
    t = sorted(li)
    median_value = []
    for index in range(len(t)-1):
        median_value.append((t[index]+t[index+1])/2)
    return median_value

def gini(dataset, tar_col):
    split_fet = None
    best_value = None
    best_gini = 999999

    for i in list(set(dataset.columns)-set(['weight'] + tar_col)):
        temp_df = dataset[[i,'weight'] + tar_col].copy()
        temp_df_clean = temp_df.dropna()
        p = temp_df_clean['weight'].sum()/temp_df['weight'].sum()
        if np.isnan(p):
            print(temp_df,p)


        actual_value = temp_df_clean[i].unique()
        for med in cpu_median(actual_value):
            left = temp_df_clean[temp_df_clean[i]<med].copy()
            right = temp_df_clean[temp_df_clean[i]>=med].copy()
            def check_na(a):
                if np.isnan(a):
                    return 0
                else:
                    return a

            left_true = left[left[tar_col[0]]==1]
            left_true = left_true['weight'].sum()/left['weight'].sum()
            left_false = left[left[tar_col[0]]==0]
            left_false = left_false['weight'].sum()/left['weight'].sum()
            
            right_true = right[right[tar_col[0]]==1]
            right_true = right_true['weight'].sum()/right['weight'].sum()
            right_false = right[right[tar_col[0]]==0]
            right_false = right_false['weight'].sum()/right['weight'].sum()
            
            if len(left)==0:
                gini_left = 9999999
            else:
                gini_left = 1 - (left_true)**2 - (left_false)**2
            if len(right)==0:
                gini_right = 9999999
            else:
                gini_right = 1 - (right_true)**2 - (right_false)**2
            
            overall_gini = (check_na(left['weight'].sum())/temp_df_clean['weight'].sum())*gini_left + (check_na(right['weight'].sum())/temp_df_clean['weight'].sum())*gini_right
            weighted_gini = overall_gini/p
            
            
            if weighted_gini<best_gini:
                split_fet = i
                best_value = med
                best_gini = weighted_gini
    
    return split_fet,best_value,best_gini

def entropy(dataset, tar_col):
    
    split_fet = None
    best_value = None
    best_IG = 0
    
    def cal_entropy(subset, tar_col):
        ent_0 = subset[subset[tar_col[0]]==0]['weight'].sum()/subset['weight'].sum()
        ent_1 = subset[subset[tar_col[0]]==1]['weight'].sum()/subset['weight'].sum()
        if ent_0 == 0:
            ent_0 = 0
        else:
            ent_0 = -ent_0*(np.log2(ent_0))
        if ent_1 == 0:
            ent_1 = 0
        else:
            ent_1 = -ent_1*(np.log2(ent_1))
        return ent_0 + ent_1

    for i in list(set(dataset.columns)-set(['weight'] + tar_col)):
        temp_df = dataset[[i,'weight'] + tar_col].copy()
        temp_df_clean = temp_df.dropna()
        p = temp_df_clean['weight'].sum()/temp_df['weight'].sum()
        if np.isnan(p):
            print(temp_df,p)
            
        # calculate parent node entropy 
        entropy_parent = cal_entropy(temp_df_clean, tar_col)
        
        attr_values = temp_df_clean[i].unique()
        for med_value in cpu_median(attr_values):
            left = temp_df_clean[temp_df_clean[i]<med_value].copy()  
            right = temp_df_clean[temp_df_clean[i]>=med_value].copy()
            
            def check_na(a):
                if np.isnan(a):
                    return 0
                else:
                    return a
                
            entropy_left = cal_entropy(left, tar_col)
            entropy_right = cal_entropy(right, tar_col)
            
            entropy_children = (check_na(left['weight'].sum())/temp_df_clean['weight'].sum())*entropy_left + (check_na(right['weight'].sum())/temp_df_clean['weight'].sum())*entropy_right
            information_gain = entropy_parent - entropy_children
            weighted_IG = information_gain/p
            
            if weighted_IG > best_IG:
                split_fet = i
                best_value = med_value
                best_IG = weighted_IG
                
    return split_fet, best_value, best_IG
   
def get_split(dataset, tar_col, benchmark):
    if benchmark == 'gini':
        (split_fet,best_value,best_gini) = gini(dataset, tar_col)
    elif benchmark == 'entropy':
        (split_fet,best_value,best_gini) = entropy(dataset, tar_col)
    else:
        raise ValueError('wrong split benchmark input')    
    (left, right) = test_split(split_fet,best_value,dataset)
    return {'split_fet':split_fet, 'best_value':best_value, 'best_gini': best_gini,
            'left_node':left, 'right_node':right}   

def to_terminal(subset, tar_col):
    return ((subset[tar_col[0]]*subset['weight']).sum())/subset['weight'].sum()

def split(node, max_depth, min_size, min_improvement, depth, tar_col, benchmark):
    left, right = node['left_node'], node['right_node']
    previous_gini = node['best_gini']
    del(node['left_node'])
    del(node['right_node']) 

    # check for a no split, cannot make any improvement
    if left.empty or right.empty:
        if left.empty:
            node['left'] = node['right'] = to_terminal(right, tar_col)
        if right.empty:
            node['left'] = node['right'] = to_terminal(left, tar_col)
        return

    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left, tar_col), to_terminal(right, tar_col)
        return
    # process left child
    
    if len(left) <= min_size:
        node['left'] = to_terminal(left, tar_col)
    else:
        node['left'] = get_split(left, tar_col, benchmark)
        #if previous_gini - node['left']['best_gini']<min_improvement:
        if False:
            node['left'] = to_terminal(left, tar_col)
        else:
            node['Previous_left'] = to_terminal(left, tar_col)
            try:
                split(node['left'], max_depth, min_size, min_improvement,depth+1, tar_col, benchmark)
            except:
                node['left'] = to_terminal(left, tar_col)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right, tar_col)
    else:
        node['right'] = get_split(right, tar_col, benchmark)
        #if previous_gini - node['right']['best_gini']<min_improvement:
        if False:
            node['right'] = to_terminal(right, tar_col)
        else:
            node['Previous_right'] = to_terminal(right, tar_col)
            try:
                split(node['right'], max_depth, min_size, min_improvement,depth+1, tar_col, benchmark)
            except:
                node['right'] = to_terminal(right, tar_col)

def print_tree_inner(node, depth): 
    if isinstance(node, dict):
        print('%s[On%s < %.3f]' % ((depth*' ', (node['split_fet']), node['best_value'])))
        print_tree_inner(node['left'], depth+1)
        print_tree_inner(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))

def outerfunc(tree, dataset):
    import copy
    node = copy.deepcopy(tree)
    def predict(tree, dataset):
        node = tree
        if isinstance(node['left'], dict):
            left = dataset[(dataset[node['split_fet']]<node['best_value'])|(dataset[node['split_fet']].isna())].copy()
            p_left = dataset[(dataset[node['split_fet']]<node['best_value'])]['weight'].sum()/dataset[~dataset[node['split_fet']].isnull()]['weight'].sum()
            left.loc[dataset[dataset[node['split_fet']].isna()].index,'weight'] = left.loc[dataset[dataset[node['split_fet']].isna()].index,'weight']  * p_left
            predict(node['left'], left)
        else:
            node['predict'] = dataset
        if isinstance(node['right'], dict):
            right = dataset[(dataset[node['split_fet']]>=node['best_value']) | (dataset[node['split_fet']].isna())].copy()
            p_right = dataset[(dataset[node['split_fet']]>=node['best_value'])]['weight'].sum() /dataset[~dataset[node['split_fet']].isnull()]['weight'].sum()
            right.loc[dataset[dataset[node['split_fet']].isna()].index,'weight'] =  right.loc[dataset[dataset[node['split_fet']].isna()].index,'weight'] * p_right
            predict(node['right'], right)
        else:
            node['predict'] = dataset
    predict(node, dataset)
    return node

def flatten_dot(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dot(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def nest_dict(flat):
    result = {}
    for k, v in flat.items():
        _nest_dict_rec(k, v, result)
    return result

def _nest_dict_rec(k, v, out):
    k, *rest = k.split('.', 1)
    if rest:
        _nest_dict_rec(rest[0], v, out.setdefault(k, {}))
    else:
        out[k] = v

def predict_with_na(tree, dataset):   
    tree = outerfunc(tree, dataset)
    flattten_tree = flatten_dot(tree)
    result_df = pd.DataFrame([])
    for i in flattten_tree.keys():
        if '.predict' in i:
            df = flattten_tree[i]
            split_fet = flattten_tree[i.replace('predict','') + 'split_fet']
            split_value = flattten_tree[i.replace('predict','') + 'best_value']
            try:
                left_prediction = flattten_tree[i.replace('predict','') + 'left']
                df_left = df[(df[split_fet]<split_value) | (df[split_fet].isnull())].copy()
                p_left = df[(df[split_fet]<split_value)]['weight'].sum()/df[~df[split_fet].isnull()]['weight'].sum()
                df_left.loc[df[df[split_fet].isna()].index,'weight'] = df_left.loc[df[df[split_fet].isna()].index,'weight'] * p_left               
                df_left['Prediction'] = df_left['weight'] * left_prediction
                result_df = result_df.append(df_left['Prediction'].reset_index())
            except:
                None

            try:
                right_prediction = flattten_tree[i.replace('predict','') + 'right']
                df_right = df[(df[split_fet]>=split_value)|(df[split_fet].isnull())].copy()
                p_right = df[(df[split_fet]>=split_value)]['weight'].sum()/df[~df[split_fet].isnull()]['weight'].sum()
                df_right.loc[df[df[split_fet].isna()].index,'weight'] = df_left.loc[df[df[split_fet].isna()].index,'weight'] * p_right              
                df_right['Prediction'] = df_right['weight'] * right_prediction
                result_df = result_df.append(df_right['Prediction'].reset_index())
            except:
                None
    return result_df.groupby(['index']).sum()

def prune_with_valid(tree, validset, threshold, tar_col):
    flatten_tree = flatten_dot(tree)
    while True:
        deep_code = copy.deepcopy(flatten_tree) 
        sort_key = sorted(flatten_tree)
        sort_key.sort(key = lambda x: -len(x))
        for i in sort_key:
            if i.endswith('.left'):
                if i[-9:-5] == 'left':
                    pre_node = i[:-9]
                else:
                    pre_node = i[:-10]

                previous_value = flatten_tree[pre_node+'Previous_left']
                ######precision_after
                candidate_tree = copy.deepcopy(flatten_tree)
                remove_list = []
                for j in candidate_tree.keys():
                    if j.startswith(i[:-5]):
                        remove_list.append(j)
                for rm in remove_list:
                    del(candidate_tree[rm])
                candidate_tree[i[:-5]] = previous_value
                pre_true = pd.merge(predict_with_na(nest_dict(candidate_tree),validset).reset_index(),
                                    validset[tar_col].reset_index(), on =['index'])
                pre_true['Prediction'] = np.where(pre_true['Prediction']>=threshold,1,0)
                precision_after = len(pre_true[pre_true['Prediction']==pre_true['Label']])/len(pre_true)
                ######precision_before
                before_true = pd.merge(predict_with_na(nest_dict(flatten_tree),validset).reset_index(),
                                    validset[tar_col].reset_index(), on =['index'])
                before_true['Prediction'] = np.where(before_true['Prediction']>=threshold,1,0)
                precision_before = len(before_true[before_true['Prediction']==before_true['Label']])/len(before_true)
                if precision_before<=precision_after:
                    flatten_tree = copy.deepcopy(candidate_tree)

            if i.endswith('.right'):
                if i[-10:-6] == 'left':
                    pre_node = i[:-10]
                if i[-11:-6] == 'right':
                    pre_node = i[:-11]
                previous_value = flatten_tree[pre_node+'Previous_right']
                ######precision_after
                candidate_tree = copy.deepcopy(flatten_tree)
                remove_list = []
                for j in candidate_tree.keys():
                    if j.startswith(i[:-6]):
                        remove_list.append(j)
                for rm in remove_list:
                    del(candidate_tree[rm])
                candidate_tree[i[:-6]] = previous_value
                pre_true = pd.merge(predict_with_na(nest_dict(candidate_tree),validset).reset_index(),
                                    validset[tar_col].reset_index(), on =['index'])
                pre_true['Prediction'] = np.where(pre_true['Prediction']>=threshold,1,0)
                precision_after = len(pre_true[pre_true['Prediction']==pre_true['Label']])/len(pre_true)
                ######precision_before
                before_true = pd.merge(predict_with_na(nest_dict(flatten_tree),validset).reset_index(),
                                    validset[tar_col].reset_index(), on =['index'])
                before_true['Prediction'] = np.where(before_true['Prediction']>=threshold,1,0)
                precision_before = len(before_true[before_true['Prediction']==before_true['Label']])/len(before_true)
                if precision_before<=precision_after:
                    flatten_tree = copy.deepcopy(candidate_tree)

        if deep_code==flatten_tree:
            break
    return nest_dict(flatten_tree) 







