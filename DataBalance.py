import math
from Generate_Samples import generate_samples

"""
    data: dataset_orig_train
    datatype: see if statement in Generate_Sample.py
    key: attribute key, e.g. 'race'
    pattr: protected attribute
    mapping: a dict using combination as key, ratio as value, e.g. {'zero_one' : 0.25}
"""
def rebalance(data, datatype, key, pattr, mapping):
    df_map = {}
    out_map = {}

    total_len = 0

    for t in mapping.keys():
        if t == 'zero_zero':
            df_map[t] = data[(data['Probability'] == 0) & (data[pattr] == 0)]
        if t == 'zero_one':
            df_map[t] = data[(data['Probability'] == 0) & (data[pattr] == 1)]
        if t == 'one_one':
            df_map[t] = data[(data['Probability'] == 1) & (data[pattr] == 1)]
        if t == 'one_zero':
            df_map[t] = data[(data['Probability'] == 1) & (data[pattr] == 0)]

    for t in df_map.keys():
        for k in key:
            df_map[t][k] = df_map[t][k].astype(str)

    for t in df_map.keys():
        total_len += len(df_map[t])

    for t in df_map.keys():
        r_len = int(math.ceil(total_len * mapping[t]))
        d_len = len(df_map[t])
        if r_len > d_len:
            out_map[t] = generate_samples(r_len - d_len ,df_map[t], datatype)
        else:
            out_map[t] = df_map[t].sample(n = r_len)

    out = []
    for i, o in enumerate(out_map.keys()):
        if i == 0:
            out = out_map[o]
        else:
            out = out.append(out_map[o])
        #print(out)

    for k in key:
        out[k] = out[k].astype(float)

    return out
