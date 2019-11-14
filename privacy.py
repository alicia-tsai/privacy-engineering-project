import numpy as np
import pandas as pd


# k anonymity
def get_k_anonymity(data, quasi_ids):
    result = data.groupby(quasi_ids).count()
    result.columns = ["k_anonymous"]
    result = result.sort_values(by="k_anonymous")
    display(result)

# distinct l-diversity
def get_l_diversity(data, quasi_ids, sen_attr):
    result = pd.DataFrame(data.groupby(quasi_ids).apply(lambda x: len(x[sen_attr].unique())))
    result.columns = ["l_diverse"]
    result = result.sort_values(by="l_diverse")
    display(result)

# recursive (c, l)-diversity
def get_c(group, l, sen_attr, quasi_ids):
    result = group.groupby(sen_attr).count().sort_values(by=quasi_ids, ascending=False)
    #display(result)
    r1 = result.iloc[0, 0]
    rl_sum = result.iloc[l-1:, 0].sum()
    return r1 / rl_sum

def get_l(group, c, sen_attr, quasi_ids):
    result = group.groupby(sen_attr).count().sort_values(by=quasi_ids, ascending=False)
    #display(result)
    r1 = result.iloc[0, 0]
    l = None
    for i in range(len(result)):
        rl_sum = result.iloc[i:, 0].sum()
        if r1 / rl_sum <= c:
            l = i+1
        else:
            return l
    return l

def get_c_l_diversity(data, quasi_ids, sen_attr, l=None, c=None):
    if l:
        result = pd.DataFrame(data.groupby(quasi_ids).apply(get_c, l, sen_attr, quasi_ids))
    if c:
        result = pd.DataFrame(data.groupby(quasi_ids).apply(get_l, c, sen_attr, quasi_ids))
    result.columns = ["c_l_diverse"]
    #result = result.sort_values(by="c_l_diverse")
    display(result)


# entropy l-diversity
def entropy(group, sen_attr):
    entropy = 0
    for i in group[sen_attr].unique():
        p = np.sum(group[sen_attr] == i) / len(group[sen_attr])
        entropy -= p * np.log(p)

    return np.exp(entropy)


def get_entropy_l_diversity(data, quasi_ids, sen_attr):
    result = pd.DataFrame(data.groupby(quasi_ids).apply(entropy, sen_attr))
    result.columns = ["l_diverse"]
    result = result.sort_values(by="l_diverse")
    display(result)

# t-closeness
def order_ground_dist(group, q, sens):
    p = group.groupby([sens]).count().iloc[:, 0:1] / len(group)
    result = pd.merge(p, q, how='outer', on=sens)
    result = result.fillna(0)
    result.columns = ["p", "q"]
    result.sort_index(inplace=True)

    return np.sum(np.abs(np.cumsum(result.p - result.q))) / (len(q) - 1)

def equal_ground_dist(group, q, sens):
    p = group.groupby([sens]).count().iloc[:, 0:1] / len(group)
    result = pd.merge(p, q, how='outer', on=sens)
    result = result.fillna(0)
    result.columns = ["p", "q"]

    return 0.5 * np.sum(np.abs(result.p - result.q))


def get_t_closeness(data, quasi, sens, ground_dist):
    q = data.groupby([sens]).count().iloc[:, 1] / len(data)
    result = pd.DataFrame(data.groupby(quasi).apply(ground_dist, q, sens))
    result.columns = ["t_closeness"]
    result = result.sort_values(by="t_closeness")
    display(result)
