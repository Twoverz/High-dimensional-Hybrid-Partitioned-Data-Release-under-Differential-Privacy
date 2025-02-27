import numpy as np
import pandas as pd
import os
import json
from src.mbi import Domain


class Dataset:
    def __init__(self, df, domain, weights=None):
        """ create a Dataset object

        :param df: a pandas dataframe
        :param domain: a domain object
        :param weight: weight for each row
        """
        assert set(domain.attrs) <= set(
            df.columns
        ), "data must contain domain attributes"
        assert weights is None or df.shape[0] == weights.size
        self.domain = domain
        self.df = df.loc[:, domain.attrs]
        self.weights = weights

    @staticmethod
    def synthetic(domain, N):
        """ Generate synthetic data conforming to the given domain

        :param domain: The domain object 
        :param N: the number of individuals
        """
        arr = [np.random.randint(low=0, high=n, size=N) for n in domain.shape]
        values = np.array(arr).T
        df = pd.DataFrame(values, columns=domain.attrs)
        return Dataset(df, domain)

    @staticmethod
    def load(path, domain):
        """ Load data into a dataset object

        :param path: path to csv file
        :param domain: path to json file encoding the domain information
        """
        df = pd.read_csv(path)
        config = json.load(open(domain))
        domain = Domain(config.keys(), config.values())
        return Dataset(df, domain)


    @staticmethod
    def divide_v(path, domain, n):
        """ Load data into multiple dataset objects

        :param path: path to csv file
        :param domain: path to json file encoding the domain information
        """
        df = pd.read_csv(path)
        config = json.load(open(domain))
        keys = list(config.keys())
        mulconfig = []
        number = len(config) // n
        for i in range(n):
            if i == (n-1) and (i+1)*number < len(config):
                mulconfig.append({key: config[key] for key in keys[i*number:]})
            else:
                mulconfig.append({key: config[key] for key in keys[i*number: (i+1)*number]})

        multi_data = []
        for k in mulconfig:
            sdomain = Domain(k.keys(), k.values())
            multi_data.append(Dataset(df, sdomain))

        return multi_data

    @staticmethod
    def divide_p(path, domain, n):
        """ Load data into multiple dataset objects

        :param path: path to csv file
        :param domain: path to json file encoding the domain information
        """
        df = pd.read_csv(path)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        groups = np.array_split(df, n)

        config = json.load(open(domain))
        domain = Domain(config.keys(), config.values())
        multi_data = []
        for k in groups:
            multi_data.append(Dataset(k, domain))

        return multi_data

    @staticmethod
    def divide_random(path, domain, pn, vn):
        """ paritation pa datasets"""
        df = pd.read_csv(path)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        groups = np.array_split(df, pn)
        config = json.load(open(domain))
        domain = Domain(config.keys(), config.values())
        multi_p_data = []
        for k in groups:
            multi_p_data.append(Dataset(k, domain))

        """ paritation ve datasets after pa datasets"""
        keys = list(config.keys())
        mulconfig = []
        number = len(config) // vn
        for i in range(vn):
            if i == (vn - 1) and (i + 1) * number < len(config):
                mulconfig.append({key: config[key] for key in keys[i * number:]})
            else:
                mulconfig.append({key: config[key] for key in keys[i * number: (i + 1) * number]})
        multi_r_data = []
        for pdate in multi_p_data:
            multi_v_data = []
            for c in mulconfig:
                sdomain = Domain(c.keys(), c.values())
                multi_v_data.append(Dataset(pdate.df, sdomain))
            multi_r_data.append(multi_v_data)

        return multi_r_data

    def project(self, cols):
        """ project dataset onto a subset of columns """
        if type(cols) in [str, int]:
            cols = [cols]
        data = self.df.loc[:, cols]
        domain = self.domain.project(cols)
        return Dataset(data, domain, self.weights)

    def drop(self, cols):
        proj = [c for c in self.domain if c not in cols]
        return self.project(proj)

    @property
    def records(self):
        return self.df.shape[0]

    def datavector(self, flatten=True):
        """ return the database in vector-of-counts form """
        bins = [range(n + 1) for n in self.domain.shape]
        ans = np.histogramdd(self.df.values, bins, weights=self.weights)[0]
        return ans.flatten() if flatten else ans

    def dataonehot(self,):
        print(self.domain)
        if len(self.domain) == 1:
            print('domain:', len(self.domain))
            result = []
            for v in range(self.domain.shape[0]):
                vector = (self.df[self.domain.attrs[0]] == v).astype(int).tolist()
                result.append(vector)
        elif len(self.domain) == 2:
            print('domain:', len(self.domain))
            result = []
            for a1 in range(self.domain.shape[0]):
                for a2 in range(self.domain.shape[1]):
                    # 创建布尔向量，判断每条记录是否满足 (attr1 == a1) 和 (attr2 == a2)
                    vector = ((self.df[self.domain.attrs[0]] == a1) & (self.df[self.domain.attrs[1]] == a2)).astype(int).tolist()
                    result.append(vector)
        # elif len(self.domain) == 3:
        #     print('domain:', len(self.domain))
        #     result = []
        #     for a1 in range(self.domain.shape[0]):
        #         for a2 in range(self.domain.shape[1]):
        #             for a3 in range(self.domain.shape[2]):
        #                 # 创建布尔向量，判断每条记录是否满足 (attr1 == a1) 和 (attr2 == a2)
        #                 vector = ((self.df[self.domain.attrs[0]] == a1) & (self.df[self.domain.attrs[1]] == a2) & (self.df[self.domain.attrs[2]] == a3)).astype(int).tolist()
        #                 result.append(vector)
        else:
            print('size of marginal is too large')


        return result

