import math
from Generate_Samples import generate_samples

str_bin = ['zero', 'one']

ratio_dist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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

    # Dynamic data division.
    pattr_len = len(pattr)
    bin_digit = pattr_len + 1
    bin_pattr_len = 2**bin_digit

    for i in range(0, bin_pattr_len):
        c = 0
        bstr = format(i, "b").zfill(bin_digit)
        kc = []
        temp_data = None
        while(c < bin_digit):
            bol = int(bstr[c])
            if c == 0:
                temp_data = data[data['Probability'] == bol]
            else:
                temp_data = temp_data[temp_data[pattr[c - 1]] == bol]
            kc.append(str_bin[bol])
            c += 1
        df_map["_".join(kc)] = temp_data

    for t in df_map.keys():
        for k in key:
            df_map[t][k] = df_map[t][k].astype(str)

    for t in df_map.keys():
        total_len += len(df_map[t])

    for t in mapping.keys():
        r_len = int(math.ceil(total_len * mapping[t]))
        kc = []
        for k in list(t):
            kc.append(str_bin[int(k)])
        kw = "_".join(list(kc))
        d_len = len(df_map[kw])
        if r_len > d_len:
            out_map[kw] = generate_samples(r_len - d_len, df_map[kw], datatype)
        else:
            out_map[kw] = df_map[kw].sample(n=r_len)

    out = []
    for i, o in enumerate(out_map.keys()):
        if i == 0:
            out = out_map[o]
        else:
            out = out.append(out_map[o])

    for k in key:
        out[k] = out[k].astype(float)

    return out


def buildRatioMap(pattr):

    # Dynamic data division.
    pattr_len = len(pattr)
    bin_digit = pattr_len + 1
    bin_pattr_len = 2**bin_digit

    bstr = []
    out = []
    # list = [ [] for i in range(0, bin_digit)]
    raw = []
    final = []
    recr_helper([], 1, bin_pattr_len, raw)

    for r in raw:
        if round(sum(r), 1) == 1.0:
            final.append(r)

    for i in range(0, bin_pattr_len):
        bstr.append(format(i, "b").zfill(bin_digit))
        
    for r in final:
        temp = {}
        for i, element in enumerate(r):
            temp[bstr[i]] = element
        out.append(temp)
            
    return out


def recr_helper(list, target, limit, out=[], track=[]):

    if len(track) < limit:
        for i in ratio_dist:
            recr_helper(list, target, limit, out, track + [i])
    else:
        if track not in out:
            out.append(track)
