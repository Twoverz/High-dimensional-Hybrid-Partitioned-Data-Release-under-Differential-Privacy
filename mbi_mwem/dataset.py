import numpy as np
import pandas as pd
import os
import json
from mbi_mwem import Domain

class Dataset:
    def __init__(self, df, domain):
        """ create a Dataset object

        :param df: a pandas dataframe
        :param domain: a domain object
        """
        assert set(domain.attrs) <= set(df.columns), 'data must contain domain attributes'
        self.domain = domain
        self.df = df.loc[:, domain.attrs]

    @staticmethod
    def synthetic(domain, N):
        """ Generate synthetic data conforming to the given domain

        :param domain: The domain object 
        :param N: the number of individuals
        """
        arr = [np.random.randint(low=0, high=n, size=N) for n in domain.shape]
        values = np.array(arr).T
        df = pd.DataFrame(values, columns = domain.attrs)
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
    
    def project(self, cols):
        """ project dataset onto a subset of columns """
        if type(cols) in [str, int]:
            cols = [cols]
        data = self.df.loc[:,cols]
        domain = self.domain.project(cols)
        return Dataset(data, domain)

    def drop(self, cols):
        proj = [c for c in self.domain if c not in cols]
        return self.project(proj)

    def datavector(self, flatten=True, weights=None):
        """ return the database in vector-of-counts form """
        bins = [range(n+1) for n in self.domain.shape] #每个属性的大小[range(0, 3), range(0, 17), range(0, 11)] 2 16 10 总共有320个组合，320个直方图
        ans = np.histogramdd(self.df.values, bins, weights=weights)[0]  #43944=len(self.df.values)=得到每个个体上述三个属性的值  最终这个直方图输出每个直方图中有多少个个体
        return ans.flatten() if flatten else ans
