# import sys
# sys.path.append('../mbi')
import os
import itertools
import numpy as np
import pandas as pd
import random

import json

from mbi_mwem import Dataset, Domain

import pdb

def save_results(df_results, results_path):
    if os.path.exists(results_path):
        df_existing_results = pd.read_csv(results_path)
        df_results = pd.concat((df_existing_results, df_results))
    df_results.to_csv(results_path, index=False)

def get_proj(dataset):
    proj = None
    if dataset == 'adult_orig':
        proj = ['age', 'workclass', 'fnlwgt', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                'capital-loss', 'hours-per-week', 'native-country', 'income>50K']
    elif dataset == 'loans':
        proj = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'installment', 'annual_inc', 'dti',
                'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc',
                'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
                'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt',
                'delinq_amnt', 'pub_rec_bankruptcies', 'settlement_amount', 'settlement_percentage', 'settlement_term',
                'term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status', 'issue_d',
                'loan_status', 'purpose', 'zip_code', 'addr_state', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d',
                'last_credit_pull_d', 'debt_settlement_flag', 'debt_settlement_flag_date', 'settlement_status',
                'settlement_date']
    elif dataset.startswith('adult'):
        if dataset.endswith('-small'):
            proj = ['sex', 'income>50K', 'race', 'marital-status',
                    'occupation', 'education-num',
                    'age_10'
                    ]
        elif dataset.endswith('multi_1'):
            proj = ['sex', 'income>50K', 'race', 'relationship', 'marital-status', 'workclass',
                    'occupation', 'education-num', 'native-country',
                    'capital-gain', 'capital-loss', 'hours-per-week',
                    'age_10'
                    ]
        elif dataset.endswith('multi_2'):
            proj = ['sex', 'race', 'relationship', 'marital-status', 'workclass','income>50K',
                    'occupation', 'education-num', 'native-country',
                    'capital-gain', 'capital-loss', #'hours-per-week',
                    'age_10'
                    ]
        elif dataset.endswith('multi_3'):
            proj = ['sex', 'income>50K', 'relationship', 'workclass','race', 'marital-status',
                    'occupation', 'education-num', 'native-country', #'capital-gain',
                     'capital-loss', 'hours-per-week',
                    'age_10'
                    ]
        else:
            proj = ['sex', 'income>50K', 'race', 'relationship', 'marital-status', 'workclass',
                    'occupation', 'education-num', 'native-country',
                    'capital-gain', 'capital-loss', 'hours-per-week',
                    'age_10'
                    ]
    elif dataset.startswith('acs'):
        if dataset.endswith('-small'):
            # for reference
            ##############
            proj2 = ['SEX', 'FOODSTMP'
                     'RACWHT', 'RACASIAN', 'RACBLK', 'RACAMIND', 'RACPACIS', 'RACOTHER'
                     'DIFFEYE', 'DIFFHEAR', 'DIFFSENS'
                     'HCOVANY', 'HCOVPRIV', 'HINSCAID', 'HINSCARE', 'HINSVA'
                    ]
            proj3 = ['SCHOOL', 'CLASSWKR', 'ACREHOUS', 'OWNERSHP', 'LABFORCE'
                     'DIFFCARE', 'DIFFREM', 'DIFFMOB', 'DIFFPHYS'
                     'VETSTAT', 'VETWWII', 'VET90X01', 'VETVIETN', 'VET47X50', 'VET55X64', 'VET01LTR', 'VETKOREA', 'VET75X90'
                     'WIDINYR', 'MARRINYR', 'FERTYR'
                    ]
            proj4 = ['MORTGAGE', 'EMPSTAT', 'SCHLTYPE', 'LOOKING', 'CITIZEN', 'WORKEDYR'
                     'DIVINYR', 'MARRNO',
                     'MULTGEN'
                     ]
            proj5 = ['HISPAN', 'AVAILBLE', 'METRO']
            proj6 = ['MARST']
            proj_ = ['AGE', 'DEGFIELD', 'OCCSCORE', 'LANGUAGE' ]
            ##############

            proj = ['SEX', 'FOODSTMP',
                    'RACWHT', 'RACASIAN', 'RACBLK', 'RACAMIND', 'RACPACIS', 'RACOTHER',
                    'DIFFEYE', 'DIFFHEAR', # 'DIFFPHYS', 'DIFFSENS',
                    'HCOVPRIV', 'HINSCAID', 'HINSCARE', # 'HCOVANY',
                    'OWNERSHP', # 'VETSTAT', 'CLASSWKR', 'ACREHOUS'
                    'EMPSTAT', # 'SCHLTYPE',
                    ]
        else:
            proj = ['VETWWII', 'AVAILBLE', 'MIGRATE1', 'MARRNO', 'GRADEATT', 'RACE', 'MARRINYR', 'EDUC', 'DIFFREM',
                    'VET75X90', 'EMPSTAT', 'VET47X50', 'MORTGAGE', 'VETVIETN', 'DIFFSENS', 'HCOVANY', 'LABFORCE',
                    'FOODSTMP', 'NCHILD', 'NSIBS', 'VETKOREA', 'VET90X01', 'RACWHT', 'RELATE', 'SEX', #'ROOMS',
                    'NMOTHERS', 'SCHLTYPE', 'DIFFEYE', 'VET55X64', 'SCHOOL', 'WIDINYR', 'MARST', 'VET01LTR', #'FAMSIZE',
                    'VEHICLES', 'WORKEDYR', 'VETDISAB', 'METRO', 'DIFFMOB', 'ACREHOUS', 'NFATHERS', #'LANGUAGE',
                    'NCHLT5', 'SPEAKENG', 'CLASSWKR', 'CITIZEN', 'VACANCY', 'RACASIAN', 'DIFFCARE', #'SEI', 'DEGFIELD',
                    'AGE', 'LOOKING', 'RACBLK', 'RACAMIND', 'DIFFPHYS', 'HINSCARE', # 'OCCSCORE', 'BUILTYR2', 'BEDROOMS',
                    'VETSTAT', 'MIGTYPE1', 'NCOUPLES', 'HISPAN', 'MULTGEN', 'DIFFHEAR', 'RACOTHER', 'HINSCAID', 'HINSVA',
                    'OWNERSHP', 'FERTYR', 'HCOVPRIV', 'DIVINYR', 'RACPACIS' # 'ELDCH', 'YNGCH', 'NFAMS',
                    ]

    return proj

def get_filters(args):
    filter_pub, filter_private = None, None
    if args.dataset == 'adult':
        filter_private = ('_split', 1)
        filter_pub = ('_split', 0)
    elif args.dataset == 'baltimore_911_calls':
        filter_private = ('Year', int(args.state))
        if hasattr(args, 'state_pub'):
            filter_pub = ('Year', int(args.state))
    elif args.dataset.startswith("acs_"):
        filter_private = ('STATE', args.state)
        if hasattr(args, 'state_pub'):
            filter_pub = ('STATE', args.state_pub)
    return filter_private, filter_pub

def get_min_dtype(int_val):
    for dtype in [np.int8, np.int16, np.int32]:
        if np.iinfo(dtype).max > int_val:
            return dtype
    return np.int64

def randomKway(name, number, marginal, proj=None, seed=0, filter=None, root_path='./', args=None):
    check_size = name in ['adult_orig', 'loans']
    path = os.path.join(root_path, "Datasets/{}.csv".format(name))
    df = pd.read_csv(path) #48813 16

    domain = os.path.join(root_path, "Datasets/{}-domain.json".format(name)) #返回一个路径
    config = json.load(open(domain))# 'education': 16, 'education-num': 16, 'marital-status': 7,
    domain = Domain(config.keys(), config.values())  #字典形式的数据 键：值 15个

    if name == 'adult':
        if args.adult_seed is not None:
            prng = np.random.RandomState(args.adult_seed)
            mask = prng.binomial(1, 0.9, size=len(df))
            df.loc[:, '_split'] = mask
        else:
            df.loc[:, '_split'] = 1

    if filter is not None:
        col, val = filter
        df = df[df[col] == val].reset_index(drop=True)
        del df[col]

    domain_max = max(domain.config.values())
    dtype = get_min_dtype(domain_max) #确定数据类型
    df = df.astype(dtype)#对整体数据的数据类型规范

    data = Dataset(df, domain)#把数据和domain（键 值）封装起来了
    #print(data.domain) #15个

    if proj is not None:
        data = data.project(proj)#13个attribution

    return data, randomKwayData(data, number, marginal, seed, check_size=check_size)

def randomKwayData(data, number, marginal, seed=0, check_size=False):
    prng = np.random.RandomState(seed)
    total = data.df.shape[0] #个体总数
    dom = data.domain

    if check_size:
        proj = [p for p in itertools.combinations(data.domain.attrs, marginal) if dom.size(p) <= total]
    else:
        proj = [p for p in itertools.combinations(data.domain.attrs, marginal)] #13个选3个

    if len(proj) > number: #286
        proj = [proj[i] for i in prng.choice(len(proj), number, replace=False)]

    return proj


def randomKway_multi(name,  proj1=None, proj2=None, proj3=None,  filter=None, root_path='./', args=None):
    path = os.path.join(root_path, "Datasets/{}.csv".format(name))
    df = pd.read_csv(path)  # 48813 16
    domain = os.path.join(root_path, "Datasets/{}-domain.json".format(name))  # 返回一个路径
    config = json.load(open(domain))  # 'education': 16, 'education-num': 16, 'marital-status': 7,
    domain = Domain(config.keys(), config.values())  # 字典形式的数据 键：值 15个

    if name == 'adult':
        if args.adult_seed is not None:
            prng = np.random.RandomState(args.adult_seed)
            mask = prng.binomial(1, 0.9, size=len(df))
            df.loc[:, '_split'] = mask
        else:
            df.loc[:, '_split'] = 1

    if filter is not None:
        col, val = filter
        df = df[df[col] == val].reset_index(drop=True)
        del df[col]

    domain_max = max(domain.config.values())
    dtype = get_min_dtype(domain_max)  # 确定数据类型
    df = df.astype(dtype)  # 对整体数据的数据类型规范 [43944 rows x 16 columns]

    data = Dataset(df, domain)  # 把数据和domain（键 值）封装起来了 print(data.domain) #15个

    df_index = data.df.index.tolist()  #得到数据的索引并将其打乱
    random.shuffle(df_index)
    split_3 = len(df_index) #得到总长度 平分为三份
    df_index1 = df_index[:(split_3//3)]
    df_half = df_index[(split_3//3):]
    df_index2 = df_half[:(split_3//3)]
    df_index3 = df_half[(split_3//3):]

    if proj1 is not None:
        data_mu1 = data.project(proj1)
        data_mu1.df = data_mu1.df.loc[df_index1]
    if proj2 is not None:
        data_mu2 = data.project(proj2)
        data_mu2.df = data_mu2.df.loc[df_index2]
    if proj3 is not None:
        data_mu3 = data.project(proj3)
        data_mu3.df = data_mu3.df.loc[df_index3]

    return data_mu1, data_mu2, data_mu3



def select_workload(data, workloads):
    data_att = data.domain.config.keys()
    ex = []
    for i in range(len(workloads)):
        for j in workloads[i]:
            if j not in data_att:
                ex.append(i)
                continue

    workloads = [workloads[i] for i in range(len(workloads)) if i not in ex]
    return workloads


def get_support(data):
    df_support = []
    for val in list(data.domain.config.values()):#dict_values([2, 2, 5, 6, 7, 9, 15, 16, 42, 16, 6, 10, 10])
        df_support.append(np.arange(val))#array([0, 1]), array([0, 1]), array([0, 1, 2, 3, 4]),

    df_support = list(itertools.product(*df_support))
    df_support = np.array(df_support)
    df_support = pd.DataFrame(df_support, columns=data.df.columns)
    data_support = Dataset(df_support, data.domain)
    A_init = np.ones(len(df_support))
    A_init /= len(A_init)

    return data_support, A_init

def get_A_init(data, df):
    cols = list(df.columns)
    df = df.groupby(cols).size().reset_index(name='Count')
    A_init = df['Count'].values
    A_init = A_init / A_init.sum()
    del df['Count']
    data_pub = Dataset(df, data.domain)

    # A_init = df.groupby(cols, sort=False).size().values
    # A_init = A_init / A_init.sum()
    # df = df.drop_duplicates()
    # data_pub = Dataset(df, data.domain)

    return data_pub, A_init

# create a fake dataset with just the unique entries of the real data
def get_pub_dataset(data, pub_frac, frac_seed):
    df_pub = data.df.copy()
    pub_data_size = int(pub_frac * df_pub.shape[0])

    prng = np.random.RandomState(frac_seed)
    idxs = prng.choice(df_pub.index, size=pub_data_size, replace=False)
    df_pub = df_pub.loc[idxs].reset_index(drop=True)
    data_pub, A_init = get_A_init(data, df_pub)

    return data_pub, A_init

# create a fake dataset with just the unique entries of the real data
def get_pub_dataset_biased(data, pub_frac, frac_seed, bias_attr, perturb):
    prng = np.random.RandomState(frac_seed)

    df_priv = data.df
    N = data.df.shape[0]

    attr_distr = np.bincount(df_priv[bias_attr]) #sex:[16182 32631]

    attr_distr = attr_distr / attr_distr.sum() #sex:[[0.33151005 0.66848995]]
    orig_attr_distr = attr_distr.copy()

    attr_distr[0] += perturb
    attr_distr[1] = 1 - attr_distr[0]

    df_pub = []
    "根据sex属性的值0/1，对表中两类数据进行了重复抽样，与原始数据大小仍一致"
    for i in range(attr_distr.shape[0]):
        mask = df_priv[bias_attr] == i
        df_attr = df_priv[mask].reset_index(drop=True) #df_priv[mask]是所有sex=i的个体数据
        size = int(pub_frac * N * attr_distr[i]) #根据上面的比例 计算sex=i 的个数
        idxs = prng.choice(df_attr.index, size=size, replace=True)#用于从给定的一维数组或整数中随机抽取样本。   相当于100个抽100次 可重复 增加了随机性
        df_pub.append(df_attr.loc[idxs]) #.loc选择固定的行和列

    df_pub = pd.concat(df_pub).reset_index(drop=True) #[48813 rows x 13 columns]

    cols = list(df_pub.columns)
    df_pub = df_pub.reset_index().groupby(cols).count() #[19994 rows x 1 columns] 对数据进行分组并计算每个分组中非空值的数量。 可以理解为找出48813中有几种不同的，得出19994，以及对应的权重

    df_pub.reset_index(inplace=True)

    A_init = df_pub['index'].values
    A_init = A_init / A_init.sum()#48813

    data_pub = Dataset(df_pub, data.domain)

    return data_pub, A_init, orig_attr_distr

def get_pub_dataset_nobiased(data, pub_frac, frac_seed, bias_attr, perturb):
    df_priv = data.df

    attr_distr = np.bincount(df_priv[bias_attr])  # sex:[16182 32631]

    attr_distr = attr_distr / attr_distr.sum()  # sex:[[0.33151005 0.66848995]]

    orig_attr_distr = attr_distr.copy()

    cols = list(data.df.columns)
    df_pub = data.df.reset_index().groupby(
        cols).count()  # [19994 rows x 1 columns] 对数据进行分组并计算每个分组中非空值的数量。 可以理解为找出48813中有几种不同的，得出19994，以及对应的权重

    df_pub.reset_index(inplace=True)
    A_init = df_pub['index'].values
    A_init = A_init / A_init.sum()  # 48813

    data_pub = Dataset(df_pub, data.domain)

    return data_pub, A_init, orig_attr_distr


def get_pub_dataset_nopub(data, pub_frac, frac_seed, bias_attr, perturb):
    df_priv = data.df

    attr_distr = np.bincount(df_priv[bias_attr])  # sex:[16182 32631]

    attr_distr = attr_distr / attr_distr.sum()  # sex:[[0.33151005 0.66848995]]

    orig_attr_distr = attr_distr.copy()

    cols = list(data.df.columns)
    df_pub = data.df.reset_index().groupby(cols).count()  # [19994 rows x 1 columns] 对数据进行分组并计算每个分组中非空值的数量。 可以理解为找出48813中有几种不同的，得出19994，以及对应的权重

    df_pub.reset_index(inplace=True)

    A_init = df_pub['index'].values
    A_init = A_init / A_init.sum()  # 48813

    data_pub = Dataset(df_pub, data.domain)

    return data_pub, A_init, orig_attr_distr


# create a fake dataset with just the unique entries of the real data
def get_pub_dataset_corrupt(data, pub_frac, frac_seed, perturb, perturb_seed=0, asymmetric=False):
    prng_frac = np.random.RandomState(frac_seed)
    prng_perturb = np.random.RandomState(perturb_seed)

    df_pub = data.df.copy()
    pub_data_size = int(pub_frac * df_pub.shape[0])

    idxs = prng_frac.choice(df_pub.index, size=pub_data_size, replace=False)
    df_pub = df_pub.loc[idxs].reset_index(drop=True)

    mask = prng_perturb.binomial(1, p=perturb, size=df_pub.shape).astype(bool)
    
    domain = data.domain
    for i, attr in enumerate(df_pub.columns):
        mask_attr = mask[:, i]
        if asymmetric:
            perturbation = 1
        else:
            perturbation = prng_perturb.choice(np.arange(1, domain[attr]), size=mask_attr.sum(), replace=True)
        df_pub.loc[mask_attr, attr] += perturbation
        df_pub.loc[mask_attr, attr] %= domain[attr]

    cols = list(df_pub.columns)
    df_pub = df_pub.groupby(cols).size().reset_index(name='Count')
    A_init = df_pub['Count'].values
    A_init = A_init / A_init.sum()

    data_pub = Dataset(df_pub, data.domain)

    return data_pub, A_init

def getdata(df, domain):
    data = Dataset(df, domain)
    return data