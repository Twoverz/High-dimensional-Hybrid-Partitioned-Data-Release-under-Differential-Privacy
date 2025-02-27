import os
import math
import operator
import numpy as np
import itertools
from tqdm import tqdm
from collections.abc import Iterable
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz

import sys
sys.path.append("../private-pgm/src")
import heapq
import pdb
from mbi_mwem import Dataset, Domain

def _get_outer_edges(a, range):
    """
    Determine the outer bin edges to use, from either the data or the range
    argument
    """
    if range is not None:
        first_edge, last_edge = range
        if first_edge > last_edge:
            raise ValueError(
                'max must be larger than min in range parameter.')
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "supplied range of [{}, {}] is not finite".format(first_edge, last_edge))
    elif a.size == 0:
        # handle empty arrays. Can't determine range, so use 0-1.
        first_edge, last_edge = 0, 1
    else:
        first_edge, last_edge = a.min(), a.max()
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "autodetected range of [{}, {}] is not finite".format(first_edge, last_edge))

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge

_range = range
def get_xy(sample, bins):
    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        N, D = sample.shape

    nbin = np.empty(D, int)
    edges = D * [None] #[None, None, None]
    dedges = D * [None] #[None, None, None]

    try:
        M = len(bins)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                ' sample x.')
    except TypeError:
        # bins is an integer
        bins = D * [bins]

    # normalize the range argument
    range = (None,) * D #(None, None, None)

    # Create edge arrays
    for i in _range(D):
        if np.ndim(bins[i]) == 0: #bins[0]=range(0, 3)
            if bins[i] < 1:
                raise ValueError(
                    '`bins[{}]` must be positive, when an integer'.format(i))
            smin, smax = _get_outer_edges(sample[:, i], range[i])
            try:
                n = operator.index(bins[i])

            except TypeError as e:
                raise TypeError(
                    "`bins[{}]` must be an integer, when a scalar".format(i)
                ) from e

            edges[i] = np.linspace(smin, smax, n + 1)
        elif np.ndim(bins[i]) == 1:
            edges[i] = np.asarray(bins[i]) #range(0,3)become[0 1 2]
            if np.any(edges[i][:-1] > edges[i][1:]):
                raise ValueError(
                    '`bins[{}]` must be monotonically increasing, when an array'
                        .format(i))
        else:
            raise ValueError(
                '`bins[{}]` must be a scalar or 1d array'.format(i))

        nbin[i] = len(edges[i]) + 1  # includes an outlier on each end
        dedges[i] = np.diff(edges[i]) #用于计算数组中相邻元素之间的差值。

    # Compute the bin number each sample falls into.
    Ncount = tuple(            #包含了所有每个属性的所有值的排序3×19994
        # avoid np.digitize to work around gh-11022
        np.searchsorted(edges[i], sample[:, i], side='right') #针对二维数组 sample 中的所有行，提取第 i 列数据。
        for i in _range(D) #用于在已排序的数组中查找元素的插入位置，以便维持数组的有序性。
    )

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in _range(D): #排除异常值，上面为异常值添加了位置，现在把这个位置的值都变成本属性的最大值
        # Find which points are on the rightmost edge.
        on_edge = (sample[:, i] == edges[i][-1])
        # Shift these points one bin to the left.
        Ncount[i][on_edge] -= 1

    # Compute the sample indices in the flattened histogram matrix.
    # This raises an error if the array is too large.

    xy = np.ravel_multi_index(Ncount, nbin) #nbin=[ 4 18 12]  这一步将多维数组展平为一维数组后，对应位置的索引。相当于求个体属于哪个具体的查询
    "例如现在维度是4×18×12，【1，25，10】对应哪个索引"
    return xy, nbin

def convert_query(query, domain):
    query = query.copy()
    query_attrs = []
    for size in domain.shape:
        if len(query) > 0 and query[0] < size:
            attr, query = query[0], query[1:]
            query_attrs.append(attr)
        else:
            query_attrs.append(-1)
        query -= size
    query_attrs = np.array(query_attrs)

    # domain_values = np.array(list(domain.config.values()))
    # domain_values_cumsum = np.cumsum(domain_values)
    # x = query_orig[:, np.newaxis] - domain_values_cumsum[np.newaxis, :] + domain_values
    # mask = (x < domain_values) & (x >= 0)
    # x[~mask] = -1
    # x = x.max(axis=0)
    # assert((query_attrs == x).mean() == 1)

    return query_attrs

def histogramdd(xy, nbin, weights):
    hist = np.bincount(xy, weights, minlength=nbin.prod())#用于统计非负整数数组中每个整数出现的次数。 第一个大查询有864组合，这里就是864个具体查询的次数

    # Shape into a proper matrix
    hist = hist.reshape(nbin) #(4, 18, 12)

    # This preserves the (bad) behavior observed in gh-7845, for now.
    hist = hist.astype(float, casting='safe')

    # Remove outliers (indices 0 and -1 for each dimension).
    D = nbin.shape[0]
    core = D * (slice(1, -1),)
    hist = hist[core] #(2, 16, 10)

    if (hist.shape != nbin - 2).any():
        raise RuntimeError(
            "Internal Shape Error")
    return hist

def min_int_dtype(arr):
    max_val_abs = np.abs(arr).max()
    for dtype in [np.int8, np.int16, np.int32, np.int64]:
        if max_val_abs < np.iinfo(dtype).max:
            return dtype

def add_row_convert_dtype(array, row, idx):
    max_val_abs = np.abs(row).max()
    if max_val_abs > np.iinfo(array.dtype).max:
        dtype = min_int_dtype(row)
        array = array.astype(dtype)
    array[idx, :len(row)] = row
    return array

def get_num_queries(domain, workloads):
    col_map = {}
    for i, col in enumerate(domain.attrs):
        col_map[col] = i
    feat_pos = []
    cur = 0
    for f, sz in enumerate(domain.shape):
        feat_pos.append(list(range(cur, cur + sz)))
        cur += sz

    num_queries = 0

    query_idx = []
    for feat in workloads: #feat = {a b c}
        positions = []
        "针对每个查询，把它涉及到的所有属性的所有类别放在positions中"
        for col in feat: #col从a开始循环
            i = col_map[col] #输出该属性对应的编号
            positions.append(feat_pos[i]) #第i个属性的所有类的编号，例如sex=[[0, 1]]
        x = list(itertools.product(*positions)) #三个属性每个类别的组合(0, 46, 136), (0, 46, 137), (0, 46, 138)。。。
        query_idx.append(x)
        num_queries += len(x)

    #query_idx = np.concatenate(query_idx)

    return num_queries, query_idx


def closest_indices(lst, x, k):
    # 使用堆来存储差值最小的元素索引
    min_heap = []

    # 遍历列表
    for i, val in enumerate(lst):
        # 计算差值的绝对值
        diff = abs(val - x)
        diff = np.sum(diff)

        # 将索引和差值加入堆中
        heapq.heappush(min_heap, (diff, i))

        # 如果堆的大小超过 k，则弹出堆顶元素
        if len(min_heap) > k:
            heapq.heappop(min_heap)

    # 提取堆中的索引
    closest_indices = [index for _, index in min_heap]

    return closest_indices


class QueryManager():
    """< 1e-9
    K-marginal queries manager
    """
    def __init__(self, domain, workloads):
        self.domain = domain #domain.attrs 所有属性名
        self.workloads = workloads
        self.att_id = {} #{'sex': 0, 'income>50K': 1, 'race': 2, 'relationship': 3, 'marital-status': 4,
        col_map = {} #{'sex': 0, 'income>50K': 1, 'race': 2, 'relationship': 3, 'marital-status': 4,
        for i,col in enumerate(self.domain.attrs):
            col_map[col] = i
            self.att_id[col] = i

        "对每个属性的类别个数编号"
        feat_pos = [] #[0, 1], [2, 3], [4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14],
        cur = 0
        for f, sz in enumerate(domain.shape):#(2, 2, 5, 6, 7, 9, 15, 16, 42, 16, 6, 10, 10)
            feat_pos.append(list(range(cur, cur + sz)))
            cur += sz

        self.dim = np.sum(self.domain.shape) #总共属性×类别个数
        self.num_queries, self.query_idx = get_num_queries(self.domain, self.workloads)
        self.max_marginal = np.array([len(x) for x in self.workloads]).max()  #3

        dtype = min_int_dtype([self.dim])
        self.queries = -1 * np.ones((self.num_queries, self.max_marginal), dtype=dtype) #[-1,-1,-1]
        idx = 0
        print("Initializing self.queries...")
        for feat in tqdm(self.workloads):
            positions = []
            for col in feat:
                i = col_map[col]
                positions.append(feat_pos[i]) #[[0, 1], [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61], [136, 137, 138, 139, 140, 141, 142, 143, 144, 145]]
            x = list(itertools.product(*positions))#[(0, 46, 136), (0, 46, 137), (0, 46, 138), (0, 46, 139), (0, 46, 140),
            x = np.array(x)
            self.queries[idx:idx+x.shape[0], :x.shape[1]] = x #把初始化的self.queries都赋对应的值 [  0  46 136] [  0  46 137] [  0  46 138]
            idx += x.shape[0]


        self.feat_pos = feat_pos #[[0,1],[2,3,4,5,6]]
        self.xy = None
        self.nbin = None
        self.query_attrs = None
        self.q_x = None

    def get_small_separator_workload(self):
        W = []
        for i in range(self.dim):
            w = np.zeros(self.dim)
            w[i] = 1
            W.append(w)
        return np.array(W)

    def get_query_workload(self, q_ids):
        if not isinstance(q_ids, Iterable):
            q_ids = [q_ids]
        W = []
        for q_id in q_ids:
            w = np.zeros(self.dim)
            for p in self.queries[q_id]:
                if p < 0:
                    break
                w[p] = 1
            W.append(w)
        if len(W) == 1:
            W = np.array(W).reshape(1, -1)
        else:
            W = np.array(W)
        return W

    def get_query_workload_weighted(self, q_ids):
        if not isinstance(q_ids, Iterable):
            q_ids = [q_ids]
        wei = {}
        for q_id in q_ids:
            wei[q_id] = 1 + wei[q_id] if q_id in wei else 1
        W = []
        weights = []
        for q_id in wei:
            w = np.zeros(self.dim)
            for p in self.queries[q_id]:
                w[p] = 1
            W.append(w)
            weights.append(wei[q_id])
        if len(W) == 1:
            W = np.array(W).reshape(1,-1)
            weights = np.array(weights)
        else:
            W = np.array(W)
            weights = np.array(weights)
        return W, weights

    def get_answer(self, data, weights=None, concat=True, debug=False):
        ans_vec = [] #128 ×（三个属性构造的所有具体查询的值）  每个具体查询是一个值
        N_sync = data.df.shape[0] #43944

        # for proj, W in self.workloads:
        for proj in self.workloads:#('sex', 'education-num', 'age_10')
            # weights let's you do a weighted sum
            x = data.project(proj).datavector(weights=weights)#最终这个直方图输出每个bin中有多少个个体
            if weights is None:
                x = x / N_sync #求分布， x=320个数字 三个属性组合所得的所有查询结果
            ans_vec.append(x)

        if concat:
            ans_vec = np.concatenate(ans_vec)
        return ans_vec

    def get_answer_app(self, data_ori, data1, data2, data3, weights=None, concat=True, debug=False):

        N_sync = np.sum((data1.df.shape[0], data2.df.shape[0], data3.df.shape[0]))  # 43944

        data_en = [data1, data2, data3]

        "计算权重"
        weight = {} #存放所有权重的字典
        weight_value = []
        num_data = 3
        for d in data_en:
            value = np.float16((d.df.shape[0]/N_sync) * len(d.domain.config.keys())/len(data_ori.domain.config.keys()))
            weight_value.append(value)
        weight_value_a = np.asarray(weight_value)
        weight_value_a = np.float16(weight_value_a/sum(weight_value)*len(data_en)) #[1.055  0.9736 0.9736]
        for i in range(num_data):
            weight.update({data_en[i]:weight_value_a[i]})

        "数据集分类"
        true_ans = [data1] #将数据集分类 根据是否满足查询
        false_ans = [data2, data3]

        final_ans = np.zeros(159034)

        "满足查询属性的数据"
        for data in true_ans:
            print(data)
            ans_vec = []
            for proj in self.workloads:
                # weights let's you do a weighted sum
                x = data.project(proj).datavector(weights=weights)  # 最终这个直方图输出每个bin中有多少个个体
                ans_vec.append(x) #最终输出15000+个查询对应的个体数

            if concat:
                ans_vec = np.concatenate(ans_vec)

            ans_arr = np.asarray(ans_vec) * weight.get(data)
            final_ans = final_ans + ans_arr


        "不满足查询属性的数据"
        i = 1
        for data in false_ans:
            print("number of false data = ", i)
            ans_vec = []
            for proj in self.workloads: #self.workloads:列表中存放了128个元组
                null_att = []
                "检查属性是否被包含在当前数据集中"
                for attr in proj:
                    "寻找缺失的属性"
                    if attr not in data.domain.config.keys():
                        null_att.append(attr)

                if len(null_att) != 0:
                    print("appromiate query alter")

                    "根据已有属性和缺失属性在整表寻找proj的替代：首先找所有符合的替补proj，再通过主表计算最相近的"
                    proj_list = list(proj)
                    for att in null_att:
                        proj_list.remove(att)

                    "分别针对1 2 3个属性不满足查询的情况，找所有符合的替补proj"
                    sub_workloads = [] #存放所有替补查询

                    if len(null_att)==1: #有一个属性不满足
                        print("number of mismatched attribute =", len(null_att))
                        for sub_proj in self.workloads:
                            if null_att[0] not in sub_proj:
                                if proj_list[0] in sub_proj and proj_list[1] in sub_proj:
                                    reattr = list(sub_proj)  # 找到不满足的那个属性
                                    reattr.remove(proj_list[0])
                                    reattr.remove(proj_list[1])
                                    if data_ori.domain.config.get(null_att[0]) == data_ori.domain.config.get(reattr[0]):
                                        sub_workloads.append(sub_proj)

                    elif len(null_att)==2: #有2个属性不满足
                        print("number of mismatched attribute =", len(null_att))
                        for sub_proj in self.workloads:
                            if (null_att[0] not in sub_proj) and (null_att[1] not in sub_proj):
                                if proj_list[0] in sub_proj:
                                    reattr = list(sub_proj)  # 找到不满足的那2个属性
                                    reattr.remove(proj_list[0])
                                    if (data_ori.domain.config.get(null_att[0]) == data_ori.domain.config.get(reattr[0]) and data_ori.domain.config.get(null_att[1]) == data_ori.domain.config.get(reattr[1]))\
                                        or (data_ori.domain.config.get(null_att[1]) == data_ori.domain.config.get(reattr[0]) and data_ori.domain.config.get(null_att[0]) == data_ori.domain.config.get(reattr[1])):
                                        sub_workloads.append(sub_proj)

                    elif len(null_att)==3: #有三个个属性不满足
                        print("number of mismatched attribute =", len(null_att))
                        for sub_proj in self.workloads:
                            if (null_att[0] not in sub_proj) and (null_att[1] not in sub_proj) and (null_att[2] not in sub_proj):
                                value_dom = [data_ori.domain.config.get(sub_proj[0]), data_ori.domain.config.get(sub_proj[1]), data_ori.domain.config.get(sub_proj[2])]
                                if (data_ori.domain.config.get(null_att[0]) in value_dom) and (data_ori.domain.config.get(null_att[1]) in value_dom) and (data_ori.domain.config.get(null_att[2]) in value_dom):
                                    sub_workloads.append(sub_proj)

                    "通过主表计算最相近的近似proj"
                    app_proj = [] #存放与proj最相近的5个sub_proj
                    compare_ans = data_ori.project(proj).datavector(weights=weights)
                    sub_ans = []
                    for sub_pro in sub_workloads:
                        x = data_ori.project(sub_pro).datavector(weights=weights)
                        sub_ans.append(x)
                    sub_idx = closest_indices(sub_ans, compare_ans, k=5) #存放了sub_workloads中最相似的5个索引
                    for x in sub_idx:
                        app_proj.append(sub_workloads[x])


                    "替换proj"
                    "如果app_proj里面有满足情况的查询，将当前的大查询proj用app_proj中的替换"
                    for sub_app_proj in app_proj:
                        null_att1 = []
                        for attr in sub_app_proj:
                            "寻找缺失的属性"
                            if attr not in data.domain.config.keys():
                                null_att1.append(attr)
                        if len(null_att1) == 0:
                            print("transform attribute {} into {}".format(proj, sub_app_proj))
                            x = data.project(sub_app_proj).datavector(weights=weights)
                            ans_vec.append(x)
                            break


                    else:
                        print("data imputation")
                        for loss_attr in null_att:
                            ranval = data_ori.df[loss_attr].sample(data.df.shape[0])
                            data.df[loss_attr] = ranval.values
                            data.domain.config.update({loss_attr: data_ori.domain.config.get(loss_attr)})

                        domain_new = Domain(data.domain.config.keys(), data.domain.config.values())
                        data_new = Dataset(data.df, domain_new)

                        x = data_new.project(proj).datavector(weights=weights)
                        ans_vec.append(x)

                else:
                    x = data.project(proj).datavector(weights=weights)  # 最终这个直方图输出每个bin中有多少个个体 array
                    ans_vec.append(x) #最终输出15000+个查询对应的个体数  'list'

            i=i+1

            if concat:
                ans_vec = np.concatenate(ans_vec)

            ans_arr = np.asarray(ans_vec) * weight.get(data)
            final_ans = final_ans + ans_arr

        final_ans = final_ans / N_sync
        final_ans = final_ans.tolist()

        return final_ans

    def get_answer_core(self, data_ori, data1, data2, data3, weights=None, concat=True, debug=False):

        N_sync = np.sum((data1.df.shape[0], data2.df.shape[0], data3.df.shape[0]))  # 43944

        data_en = [data1, data2, data3]

        "计算权重"
        weight = {} #存放所有权重的字典
        weight_value = []
        num_data = 3
        for d in data_en:
            value = np.float16((d.df.shape[0]/N_sync) * len(d.domain.config.keys())/len(data_ori.domain.config.keys()))
            weight_value.append(value)
        weight_value_a = np.asarray(weight_value)
        weight_value_a = np.float16(weight_value_a/sum(weight_value)*len(data_en)) #[1.055  0.9736 0.9736]
        for i in range(num_data):
            weight.update({data_en[i]:weight_value_a[i]})

        "数据集分类"
        true_ans = [data1] #将数据集分类 根据是否满足查询
        false_ans = [data2, data3]

        final_ans = np.zeros(33800)

        "满足查询属性的数据"
        for data in true_ans:
            print(data)
            ans_vec = []
            for proj in self.workloads:
                # weights let's you do a weighted sum
                x = data.project(proj).datavector(weights=weights)  # 最终这个直方图输出每个bin中有多少个个体
                ans_vec.append(x) #最终输出15000+个查询对应的个体数

            if concat:
                ans_vec = np.concatenate(ans_vec)

            ans_arr = np.asarray(ans_vec) * weight.get(data)
            final_ans = final_ans + ans_arr


        "不满足查询属性的数据"
        i = 1
        for data in false_ans:
            print("number of false data = ", i)
            ans_vec = []
            for proj in self.workloads: #self.workloads:列表中存放了128个元组
                null_att = []
                "检查属性是否被包含在当前数据集中"
                for attr in proj:
                    "寻找缺失的属性"
                    if attr not in data.domain.config.keys():
                        null_att.append(attr)

                if len(null_att) != 0:
                    print("appromiate query alter")

                    "根据已有属性和缺失属性在整表寻找proj的替代：首先找所有符合的替补proj，再通过主表计算最相近的"
                    proj_list = list(proj)
                    for att in null_att:
                        proj_list.remove(att)

                    "分别针对1 2 3个属性不满足查询的情况，找所有符合的替补proj"
                    sub_workloads = [] #存放所有替补查询

                    if len(null_att)==1: #有一个属性不满足
                        print("number of mismatched attribute =", len(null_att))
                        for sub_proj in self.workloads:
                            if null_att[0] not in sub_proj:
                                if proj_list[0] in sub_proj and proj_list[1] in sub_proj:
                                    reattr = list(sub_proj)  # 找到不满足的那个属性
                                    reattr.remove(proj_list[0])
                                    reattr.remove(proj_list[1])
                                    if data_ori.domain.config.get(null_att[0]) == data_ori.domain.config.get(reattr[0]):
                                        sub_workloads.append(sub_proj)

                    elif len(null_att)==2: #有2个属性不满足
                        print("number of mismatched attribute =", len(null_att))
                        for sub_proj in self.workloads:
                            if (null_att[0] not in sub_proj) and (null_att[1] not in sub_proj):
                                if proj_list[0] in sub_proj:
                                    reattr = list(sub_proj)  # 找到不满足的那2个属性
                                    reattr.remove(proj_list[0])
                                    if (data_ori.domain.config.get(null_att[0]) == data_ori.domain.config.get(reattr[0]) and data_ori.domain.config.get(null_att[1]) == data_ori.domain.config.get(reattr[1]))\
                                        or (data_ori.domain.config.get(null_att[1]) == data_ori.domain.config.get(reattr[0]) and data_ori.domain.config.get(null_att[0]) == data_ori.domain.config.get(reattr[1])):
                                        sub_workloads.append(sub_proj)

                    elif len(null_att)==3: #有三个个属性不满足
                        print("number of mismatched attribute =", len(null_att))
                        for sub_proj in self.workloads:
                            if (null_att[0] not in sub_proj) and (null_att[1] not in sub_proj) and (null_att[2] not in sub_proj):
                                value_dom = [data_ori.domain.config.get(sub_proj[0]), data_ori.domain.config.get(sub_proj[1]), data_ori.domain.config.get(sub_proj[2])]
                                if (data_ori.domain.config.get(null_att[0]) in value_dom) and (data_ori.domain.config.get(null_att[1]) in value_dom) and (data_ori.domain.config.get(null_att[2]) in value_dom):
                                    sub_workloads.append(sub_proj)

                    "通过主表计算最相近的近似proj"
                    app_proj = [] #存放与proj最相近的5个sub_proj
                    compare_ans = data_ori.project(proj).datavector(weights=weights)
                    sub_ans = []
                    for sub_pro in sub_workloads:
                        x = data_ori.project(sub_pro).datavector(weights=weights)
                        sub_ans.append(x)
                    sub_idx = closest_indices(sub_ans, compare_ans, k=5) #存放了sub_workloads中最相似的5个索引
                    for x in sub_idx:
                        app_proj.append(sub_workloads[x])


                    "替换proj"
                    "如果app_proj里面有满足情况的查询，将当前的大查询proj用app_proj中的替换"
                    for sub_app_proj in app_proj:
                        null_att1 = []
                        for attr in sub_app_proj:
                            "寻找缺失的属性"
                            if attr not in data.domain.config.keys():
                                null_att1.append(attr)
                        if len(null_att1) == 0:
                            print("transform attribute {} into {}".format(proj, sub_app_proj))
                            x = data.project(sub_app_proj).datavector(weights=weights)
                            ans_vec.append(x)
                            break


                    else:
                        print("data imputation")
                        for loss_attr in null_att:
                            ranval = data_ori.df[loss_attr].sample(data.df.shape[0], replace=True) #从核心集中寻找当前数据集丢失属性对应的值
                            data.df[loss_attr] = ranval.values
                            data.domain.config.update({loss_attr: data_ori.domain.config.get(loss_attr)})

                        domain_new = Domain(data.domain.config.keys(), data.domain.config.values())
                        data_new = Dataset(data.df, domain_new)

                        x = data_new.project(proj).datavector(weights=weights)
                        ans_vec.append(x)

                else:
                    x = data.project(proj).datavector(weights=weights)  # 最终这个直方图输出每个bin中有多少个个体 array
                    ans_vec.append(x) #最终输出15000+个查询对应的个体数  'list'

            i=i+1

            if concat:
                ans_vec = np.concatenate(ans_vec)

            ans_arr = np.asarray(ans_vec) * weight.get(data)
            final_ans = final_ans + ans_arr

        final_ans = final_ans / N_sync
        #final_ans = final_ans.tolist()

        return final_ans

    def setup_query_workload(self):
        domain_values = np.array(list(self.domain.config.values()))
        domain_values_cumsum = np.cumsum(domain_values)
        domain_values = domain_values.astype(min_int_dtype(domain_values))
        domain_values_cumsum = domain_values_cumsum.astype(min_int_dtype(domain_values_cumsum))

        shape = (len(self.queries), len(domain_values))
        self.query_attrs = -1 * np.ones(shape, dtype=np.int8)

        idx = 0
        num_chunks = math.ceil(shape[0] / int(1e7))
        for queries in tqdm(np.array_split(self.queries, num_chunks)):
            x = queries[:, :, np.newaxis] - domain_values_cumsum[np.newaxis, np.newaxis, :] + domain_values
            mask = (x < domain_values) & (x >= 0)
            x[~mask] = -1
            x = x.max(axis=1)

            dtype = min_int_dtype(x)
            self.query_attrs = self.query_attrs.astype(dtype, copy=False)
            self.query_attrs[idx:idx+x.shape[0]] = x
            idx += x.shape[0]

    def setup_query_attr(self, save_dir=None):
        path = None
        if save_dir is not None:
            path = os.path.join(save_dir, 'query_attr.npy')
            if os.path.exists(path):
                self.query_attrs = np.load(path)
                self.queries = None
                return

        print('running init_query_attr...')
        self.setup_query_workload()

        if path is not None:
            np.save(path, self.query_attrs)
        # when using init_query_attr, I don't think you will need self.queries anymore. Change this later if we run into a situation where it's still needed
        self.queries = None

    def setup_xy(self, data, save_dir=None, overwrite=False):
        path_xy, path_nbin = None, None
        if save_dir is not None:
            path_xy = os.path.join(save_dir, 'xy.npy')
            path_nbin = os.path.join(save_dir, 'nbin.npy')
            if not overwrite and os.path.exists(path_xy) and os.path.exists(path_nbin):
                self.xy = np.load(path_xy)
                self.nbin = np.load(path_nbin)
                return

        print('running set_up_xy...')
        self.xy = None
        self.nbin = None
        for i, proj in enumerate(tqdm(self.workloads)):
            _data = data.project(proj) #proj=('sex', 'education-num', 'age_10') _data.domain((sex: 2, education-num: 16, age_10: 10) df:[19994 rows x 3 columns]
            bins = [range(n + 1) for n in _data.domain.shape] #[range(0, 3), range(0, 17), range(0, 11)]
            xy, nbin = get_xy(_data.df.values, bins) #查询索引 和 查询维度   xy是每个个体位于3-way margnal的第几个查询

            if self.xy is None:
                shape = (len(self.workloads), xy.shape[0])#(128, 19994)
                self.xy = -1 * np.ones(shape, dtype=np.int8)
            if self.nbin is None:
                shape = (len(self.workloads), nbin.shape[0])#(128, 3)
                self.nbin = -1 * np.ones(shape, dtype=np.int8)

            "把每个个体针对每个大查询的所有索引放入数组,128个大查询"
            self.xy = add_row_convert_dtype(self.xy, xy, i)
            self.nbin = add_row_convert_dtype(self.nbin, nbin, i)

        # if path_xy is not None and path_nbin is not None:
        #     np.save(path_xy, self.xy)
        #     np.save(path_nbin, self.nbin)

    def get_answer_weights(self, weights, concat=True):
        assert(self.xy is not None) # otherwise set_up_xy hasn't been run
        ans_vec = []
        xy_neg = -1 in self.xy
        nbin_neg = -1 in self.nbin

        for i in range(len(self.workloads)):
            xy = self.xy[i] #查询编号
            nbin = self.nbin[i] #查询编号对应的的值
            if xy_neg:
                xy = xy[xy != -1]
            if nbin_neg:
                nbin = nbin[nbin != -1]
            x = histogramdd(xy, nbin, weights).flatten() #计算所有个体在当前查询下的结果

            ans_vec.append(x)
        if concat:
            ans_vec = np.concatenate(ans_vec)
        return ans_vec

    # doesn't speed things
    def setup_q_x(self, data, save_dir=None):
        path = None
        if save_dir is not None:
            path = os.path.join(save_dir, 'q_x.npz')
            if os.path.exists(path):
                self.q_x = load_npz(path)
                return

        self.q_x = []
        GB_TOTAL = 3
        gb = 1e-9 * data.df.shape[0]
        chunk_size = int(GB_TOTAL / gb)
        num_chunks = math.ceil(len(self.query_attrs) / chunk_size)
        for query_attrs_chunk in np.array_split(self.query_attrs, num_chunks):
            shape = (query_attrs_chunk.shape[0], data.df.shape[0])
            q_x = np.zeros(shape, dtype=np.byte)
            for idx, query_attrs in enumerate(tqdm(query_attrs_chunk)):
                query_mask = query_attrs != -1
                q_t_x = data.df.values[:, query_mask] - query_attrs[query_mask]
                q_t_x = np.abs(q_t_x).sum(axis=1)
                q_t_x = (q_t_x == 0).astype(np.byte)
                q_x[idx] = q_t_x
            q_x = csr_matrix(q_x, dtype=np.byte)
            self.q_x.append(q_x)
        self.q_x = vstack(self.q_x)

        if path is not None:
            save_npz(path, self.q_x)





















