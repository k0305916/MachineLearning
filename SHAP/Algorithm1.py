import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor, plot_tree
import shap
import tabulate
from itertools import combinations
import scipy

d = load_boston()
df = pd.DataFrame(d['data'], columns=d['feature_names'])
y = pd.Series(d['target'])
df = df[['RM', 'LSTAT', 'DIS', 'NOX']]
clf = DecisionTreeRegressor(max_depth=3)
clf.fit(df, y)
fig = plt.figure(figsize=(20, 5))
ax = fig.add_subplot(111)
_ = plot_tree(clf, ax=ax, feature_names=df.columns)

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(df[:1])
print(tabulate.tabulate(pd.DataFrame(
    {'shap_value': shap_values.squeeze(),
     'feature_value': df[:1].values.squeeze()}, index=df.columns),
                        tablefmt="github", headers="keys"))


def pred_tree(clf, coalition, row, node=0):
    left_node = clf.tree_.children_left[node]
    right_node = clf.tree_.children_right[node]
    is_leaf = left_node == right_node

    if is_leaf:
        print('clf.tree_.value[node].squeeze()')
        print(clf.tree_.value[node].squeeze())
        return clf.tree_.value[node].squeeze()

    feature = row.index[clf.tree_.feature[node]]
    print('feature')
    print(feature)
    if feature in coalition:
        print('***************')
        print('row.loc[feature]')
        print(row.loc[feature])
        print('clf.tree_.threshold[node]')
        print(clf.tree_.threshold[node])
        if row.loc[feature] <= clf.tree_.threshold[node]:
            print('go left')
            # go left
            wl = pred_tree(clf, coalition, row, node=left_node)
            print('wl: {}'.format(wl))
            return wl
        # go right
        print('go right')
        wr = pred_tree(clf, coalition, row, node=right_node)
        print('wr: {}'.format(wr))
        return wr
    print('feature not in coalition------------')
    # take weighted average of left and right
    wl = clf.tree_.n_node_samples[left_node] / clf.tree_.n_node_samples[node]
    print('wl: leftnode {}, parentnode {}. {}'.format(clf.tree_.n_node_samples[left_node],clf.tree_.n_node_samples[node],wl))
    wr = clf.tree_.n_node_samples[right_node] / clf.tree_.n_node_samples[node]
    print('wr: leftnode {}, parentnode {}. {}'.format(clf.tree_.n_node_samples[right_node],clf.tree_.n_node_samples[node],wr))
    print('go left 1')
    value = wl * pred_tree(clf, coalition, row, node=left_node)
    print('wl: {}, left value: {}'.format(wl,value))
    print('go right 1')
    value += wr * pred_tree(clf, coalition, row, node=right_node)
    print('wr: {}, shap value: {}'.format(wr,value))
    return value

# value = pred_tree(clf, coalition=['LSTAT', 'NOX', 'RM'], row=df[:1].T.squeeze())
# print(df[:1].T.squeeze())
# value = pred_tree(clf, coalition=['LSTAT','NOX'], row=df[:1].T.squeeze())
# value = pred_tree(clf, coalition=['NOX', 'RM'], row=df[:1].T.squeeze())
# print(value)

def make_value_function(clf, row, col):
    def value(c):
        marginal_gain = pred_tree(clf, c + [col], row) - pred_tree(clf, c, row)
        num_coalitions = scipy.special.comb(len(row) - 1, len(c))
        return marginal_gain / num_coalitions
    return value

def make_coalitions(row, col):
    rest = [x for x in row.index if x != col]
    for i in range(len(rest) + 1):
        for x in combinations(rest, i):
            yield list(x)

def compute_shap(clf, row, col):
    v = make_value_function(clf, row, col)
    return sum([v(coal) / len(row) for coal in make_coalitions(row, col)])

print(tabulate.tabulate(pd.DataFrame(
    {'shap_value': shap_values.squeeze(),
     'my_shap': [compute_shap(clf, df[:1].T.squeeze(), x) for x in df.columns],
     'feature_value': df[:1].values.squeeze()
    }, index=df.columns),tablefmt="github", headers="keys"))