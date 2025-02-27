import torch
import jax.numpy as jnp
from jax import vjp
import time
from jax.nn import softmax as jax_softmax
from copy import deepcopy
from scipy.special import logsumexp
from scipy.stats import norm, laplace
import numpy as np
from cvxopt import solvers, matrix
import json
from collections import OrderedDict
from scipy.sparse.linalg import LinearOperator, eigsh, lsmr, aslinearoperator
from scipy import optimize, sparse
from collections import defaultdict
from functools import reduce
import pickle
import networkx as nx
import itertools
import pandas as pd
from scipy._lib._disjoint_set import DisjointSet
class CallBack:
    """ A CallBack is a function called after every iteration of an iterative optimization procedure
    It is useful for tracking loss and other metrics over time.
    """

    def __init__(self, engine, frequency=50):
        """ Initialize the callback objet

        :param engine: the FactoredInference object that is performing the optimization
        :param frequency: the number of iterations to perform before computing the callback function
        """
        self.engine = engine
        self.frequency = frequency
        self.calls = 0

    def run(self, marginals):
        pass

    def __call__(self, marginals):
        if self.calls == 0:
            self.start = time.time()
        if self.calls % self.frequency == 0:
            self.run(marginals)
        self.calls += 1


class Logger(CallBack):
    """ Logger is the default callback function.  It tracks the time, L1 loss, L2 loss, and
        optionally the total variation distance to the true query answers (when available).
        The last is for debugging purposes only - in practice the true answers can not  be observed.
    """

    def __init__(self, engine, true_answers=None, frequency=50):
        """ Initialize the callback objet

        :param engine: the FactoredInference object that is performing the optimization
        :param true_answers: a dictionary containing true answers to the measurement queries.
        :param frequency: the number of iterations to perform before computing the callback function
        """
        CallBack.__init__(self, engine, frequency)
        self.true_answers = true_answers
        self.idx = 0

    def setup(self):
        model = self.engine.model
        total = sum(model.domain.size(cl) for cl in model.cliques)
        print('Total clique size:', total, flush=True)
        # cl = max(model.cliques, key=lambda cl: model.domain.size(cl))
        # print('Maximal clique', cl, model.domain.size(cl), flush=True)
        cols = ['iteration', 'time', 'l1_loss', 'l2_loss', 'feasibility']
        if self.true_answers is not None:
            cols.append('variation')
        self.results = pd.DataFrame(columns=cols)
        print('\t\t'.join(cols), flush=True)

    def variational_distances(self, marginals):
        errors = []
        for Q, y, proj in self.true_answers:
            for cl in marginals:
                if set(proj) <= set(cl):
                    mu = marginals[cl].project(proj)
                    x = mu.values.flatten()
                    diff = Q.dot(x) - y
                    err = 0.5 * np.abs(diff).sum() / y.sum()
                    errors.append(err)
                    break
        return errors

    def primal_feasibility(self, mu):
        ans = 0
        count = 0
        for r in mu:
            for s in mu:
                if r == s: break
                d = tuple(set(r) & set(s))
                if len(d) > 0:
                    x = mu[r].project(d).datavector()
                    y = mu[s].project(d).datavector()
                    err = np.linalg.norm(x - y, 1)
                    ans += err
                    count += 1
        try:
            return ans / count
        except:
            return 0

    def run(self, marginals):
        if self.idx == 0:
            self.setup()

        t = time.time() - self.start
        l1_loss = self.engine._marginal_loss(marginals, metric='L1')[0]
        l2_loss = self.engine._marginal_loss(marginals, metric='L2')[0]
        feasibility = self.primal_feasibility(marginals)
        row = [self.calls, t, l1_loss, l2_loss, feasibility]
        if self.true_answers is not None:
            variational = np.mean(self.variational_distances(marginals))
            row.append(100 * variational)
        self.results.loc[self.idx] = row
        self.idx += 1

        print('\t\t'.join(['%.2f' % v for v in row]), flush=True)


class CliqueVector(dict):
    """ This is a convenience class for simplifying arithmetic over the
        concatenated vector of marginals and potentials.

        These vectors are represented as a dictionary mapping cliques (subsets of attributes)
        to marginals/potentials (Factor objects)
    """

    def __init__(self, dictionary):
        self.dictionary = dictionary
        dict.__init__(self, dictionary)

    @staticmethod
    def zeros(domain, cliques):
        return CliqueVector({cl: Factor.zeros(domain.project(cl)) for cl in cliques})

    @staticmethod
    def ones(domain, cliques):

        return CliqueVector({cl: Factor.ones(domain.project(cl)) for cl in cliques})

    @staticmethod
    def uniform(domain, cliques):

        return CliqueVector({cl: Factor.uniform(domain.project(cl)) for cl in cliques})

    @staticmethod
    def random(domain, cliques, prng=np.random):

        return CliqueVector({cl: Factor.random(domain.project(cl), prng) for cl in cliques})

    @staticmethod
    def normal(domain, cliques, prng=np.random):

        return CliqueVector({cl: Factor.normal(domain.project(cl), prng) for cl in cliques})

    @staticmethod
    def from_data(data, cliques):

        ans = {}
        for cl in cliques:
            mu = data.project(cl)
            ans[cl] = Factor(mu.domain, mu.datavector())
        return CliqueVector(ans)

    def combine(self, other):
        # combines this CliqueVector with other, even if they do not share the same set of factors
        # used for warm-starting optimization
        # Important note: if other contains factors not defined within this CliqueVector, they
        # are ignored and *not* combined into this CliqueVector
        for cl in other:
            for cl2 in self:
                if set(cl) <= set(cl2):
                    self[cl2] += other[cl]
                    break

    def __mul__(self, const):
        ans = {cl: const * self[cl] for cl in self}
        return CliqueVector(ans)

    def __rmul__(self, const):
        return self.__mul__(const)

    def __add__(self, other):

        if np.isscalar(other):
            ans = {cl: self[cl] + other for cl in self}
        else:
            ans = {cl: self[cl] + other[cl] for cl in self}
        return CliqueVector(ans)

    def __sub__(self, other):
        return self + -1 * other

    def exp(self):
        ans = {cl: self[cl].exp() for cl in self}
        return CliqueVector(ans)

    def log(self):
        ans = {cl: self[cl].log() for cl in self}
        return CliqueVector(ans)

    def dot(self, other):
        return sum((self[cl] * other[cl]).sum() for cl in self)

    def size(self):
        return sum(self[cl].domain.size() for cl in self)


class Dataset:
    def __init__(self, df, domain, weights=None):
        """ create a Dataset object

        :param df: a pandas dataframe
        :param domain: a domain object
        :param weight: weight for each row
        """
        assert set(domain.attrs) <= set(df.columns), 'data must contain domain attributes'
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

    # def dataonehot(self, ):
    #     print(self.domain)
    #     if len(self.domain) == 1:
    #         print('domain:', len(self.domain))
    #         result = []
    #         for v in range(self.domain.shape[0]):
    #             vector = (self.df[self.domain.attrs[0]] == v).astype(int).tolist()
    #             result.append(vector)
    #     elif len(self.domain) == 2:
    #         print('domain:', len(self.domain))
    #         result = []
    #         for a1 in range(self.domain.shape[0]):
    #             for a2 in range(self.domain.shape[1]):
    #                 # 创建布尔向量，判断每条记录是否满足 (attr1 == a1) 和 (attr2 == a2)
    #                 vector = ((self.df[self.domain.attrs[0]] == a1) & (self.df[self.domain.attrs[1]] == a2)).astype(
    #                     int).tolist()
    #                 result.append(vector)
    #     elif len(self.domain) == 3:
    #         print('domain:', len(self.domain))
    #         result = []
    #         for a1 in range(self.domain.shape[0]):
    #             for a2 in range(self.domain.shape[1]):
    #                 for a3 in range(self.domain.shape[2]):
    #                     # 创建布尔向量，判断每条记录是否满足 (attr1 == a1) 和 (attr2 == a2)
    #                     vector = ((self.df[self.domain.attrs[0]] == a1) & (self.df[self.domain.attrs[1]] == a2) & (
    #                                 self.df[self.domain.attrs[2]] == a3)).astype(int).tolist()
    #                     result.append(vector)
    #     else:
    #         print('size of marginal is too large')
    #
    #     return result


class Domain:
    def __init__(self, attrs, shape):
        """ Construct a Domain object

        :param attrs: a list or tuple of attribute names
        :param shape: a list or tuple of domain sizes for each attribute
        """
        assert len(attrs) == len(shape), 'dimensions must be equal'
        self.attrs = tuple(attrs)
        self.shape = tuple(shape)
        self.config = dict(zip(attrs, shape))

    @staticmethod
    def fromdict(config):
        """ Construct a Domain object from a dictionary of { attr : size } values """
        return Domain(config.keys(), config.values())

    def project(self, attrs):
        """ project the domain onto a subset of attributes

        :param attrs: the attributes to project onto
        :return: the projected Domain object
        """
        # return the projected domain
        if type(attrs) is str:
            attrs = [attrs]
        shape = tuple(self.config[a] for a in attrs)
        return Domain(attrs, shape)

    def marginalize(self, attrs):
        """ marginalize out some attributes from the domain (opposite of project)

        :param attrs: the attributes to marginalize out
        :return: the marginalized Domain object
        """
        proj = [a for a in self.attrs if not a in attrs]
        return self.project(proj)

    def axes(self, attrs):
        """ return the axes tuple for the given attributes

        :param attrs: the attributes
        :return: a tuple with the corresponding axes
        """
        return tuple(self.attrs.index(a) for a in attrs)

    def transpose(self, attrs):
        """ reorder the attributes in the domain object """
        return self.project(attrs)

    def invert(self, attrs):
        """ returns the attributes in the domain not in the list """
        return [a for a in self.attrs if a not in attrs]

    def merge(self, other):
        """ merge this domain object with another

        :param other: another Domain object
        :return: a new domain object covering the full domain

        Example:
        >>> D1 = Domain(['a','b'], [10,20])
        >>> D2 = Domain(['b','c'], [20,30])
        >>> D1.merge(D2)
        Domain(['a','b','c'], [10,20,30])
        """
        extra = other.marginalize(self.attrs)
        return Domain(self.attrs + extra.attrs, self.shape + extra.shape)

    def contains(self, other):
        """ determine if this domain contains another

        """
        return set(other.attrs) <= set(self.attrs)

    def size(self, attrs=None):
        """ return the total size of the domain """
        if attrs == None:
            return reduce(lambda x, y: x * y, self.shape, 1)
        return self.project(attrs).size()

    def sort(self, how='size'):
        """ return a new domain object, sorted by attribute size or attribute name """
        if how == 'size':
            attrs = sorted(self.attrs, key=self.size)
        elif how == 'name':
            attrs = sorted(self.attrs)
        return self.project(attrs)

    def canonical(self, attrs):
        """ return the canonical ordering of the attributes """
        return tuple(a for a in self.attrs if a in attrs)

    def __contains__(self, attr):
        return attr in self.attrs

    def __getitem__(self, a):
        """ return the size of an individual attribute
        :param a: the attribute
        """
        return self.config[a]

    def __iter__(self):
        """ iterator for the attributes in the domain """
        return self.attrs.__iter__()

    def __len__(self):
        return len(self.attrs)

    def __eq__(self, other):
        return self.attrs == other.attrs and self.shape == other.shape

    def __repr__(self):
        inner = ', '.join(['%s: %d' % x for x in zip(self.attrs, self.shape)])
        return 'Domain(%s)' % inner

    def __str__(self):
        return self.__repr__()


class Factor:
    def __init__(self, domain, values):
        """ Initialize a factor over the given domain

        :param domain: the domain of the factor
        :param values: the ndarray of factor values (for each element of the domain)

        Note: values may be a flattened 1d array or a ndarray with same shape as domain
        """
        assert domain.size() == values.size, 'domain size does not match values size'
        assert values.ndim == 1 or values.shape == domain.shape, 'invalid shape for values array'
        self.domain = domain
        self.values = values.reshape(domain.shape)

    @staticmethod
    def zeros(domain):
        return Factor(domain, np.zeros(domain.shape))

    @staticmethod
    def ones(domain):
        return Factor(domain, np.ones(domain.shape))

    @staticmethod
    def random(domain):
        return Factor(domain, np.random.rand(*domain.shape))

    @staticmethod
    def uniform(domain):
        return Factor.ones(domain) / domain.size()

    @staticmethod
    def active(domain, structural_zeros):
        """ create a factor that is 0 everywhere except in positions present in
            'structural_zeros', where it is -infinity

        :param: domain: the domain of this factor
        :param: structural_zeros: a list of values that are not possible
        """
        idx = tuple(np.array(structural_zeros).T)
        vals = np.zeros(domain.shape)
        vals[idx] = -np.inf
        return Factor(domain, vals)

    def expand(self, domain):
        assert domain.contains(self.domain), 'expanded domain must contain current domain'
        dims = len(domain) - len(self.domain)
        values = self.values.reshape(self.domain.shape + tuple([1] * dims))
        ax = domain.axes(self.domain.attrs)
        values = np.moveaxis(values, range(len(ax)), ax)
        values = np.broadcast_to(values, domain.shape)
        return Factor(domain, values)

    def transpose(self, attrs):
        assert set(attrs) == set(self.domain.attrs), 'attrs must be same as domain attributes'
        newdom = self.domain.project(attrs)
        ax = newdom.axes(self.domain.attrs)
        values = np.moveaxis(self.values, range(len(ax)), ax)
        return Factor(newdom, values)

    def project(self, attrs, agg='sum'):
        """
        project the factor onto a list of attributes (in order)
        using either sum or logsumexp to aggregate along other attributes
        """
        assert agg in ['sum', 'logsumexp'], 'agg must be sum or logsumexp'
        marginalized = self.domain.marginalize(attrs)
        if agg == 'sum':
            ans = self.sum(marginalized.attrs)
        elif agg == 'logsumexp':
            ans = self.logsumexp(marginalized.attrs)
        return ans.transpose(attrs)

    def sum(self, attrs=None):
        if attrs is None:
            return np.sum(self.values)
        axes = self.domain.axes(attrs)
        values = np.sum(self.values, axis=axes)
        newdom = self.domain.marginalize(attrs)
        return Factor(newdom, values)

    def logsumexp(self, attrs=None):
        if attrs is None:
            return logsumexp(self.values)
        axes = self.domain.axes(attrs)
        values = logsumexp(self.values, axis=axes)
        newdom = self.domain.marginalize(attrs)
        return Factor(newdom, values)

    def logaddexp(self, other):
        newdom = self.domain.merge(other.domain)
        factor1 = self.expand(newdom)
        factor2 = self.expand(newdom)
        return Factor(newdom, np.logaddexp(factor1.values, factor2.values))

    def max(self, attrs=None):
        if attrs is None:
            return self.values.max()
        axes = self.domain.axes(attrs)
        values = np.max(self.values, axis=axes)
        newdom = self.domain.marginalize(attrs)
        return Factor(newdom, values)

    def condition(self, evidence):
        """ evidence is a dictionary where
                keys are attributes, and
                values are elements of the domain for that attribute """
        slices = [evidence[a] if a in evidence else slice(None) for a in self.domain]
        newdom = self.domain.marginalize(evidence.keys())
        values = self.values[tuple(slices)]
        return Factor(newdom, values)

    def copy(self, out=None):
        if out is None:
            return Factor(self.domain, self.values.copy())
        np.copyto(out.values, self.values)
        return out

    def __mul__(self, other):
        if np.isscalar(other):
            new_values = np.nan_to_num(other * self.values)
            return Factor(self.domain, new_values)
        # print(self.values.max(), other.values.max(), self.domain, other.domain)
        newdom = self.domain.merge(other.domain)
        factor1 = self.expand(newdom)
        factor2 = other.expand(newdom)
        return Factor(newdom, factor1.values * factor2.values)

    def __add__(self, other):
        if np.isscalar(other):
            return Factor(self.domain, other + self.values)

        newdom = self.domain.merge(other.domain)
        factor1 = self.expand(newdom)
        factor2 = other.expand(newdom)
        return Factor(newdom, factor1.values + factor2.values)

    def __iadd__(self, other):
        if np.isscalar(other):
            self.values += other
            return self
        factor2 = other.expand(self.domain)
        self.values += factor2.values
        return self

    def __imul__(self, other):
        if np.isscalar(other):
            self.values *= other
            return self
        factor2 = other.expand(self.domain)
        self.values *= factor2.values
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if np.isscalar(other):
            return Factor(self.domain, self.values - other)
        other = Factor(other.domain, np.where(other.values == -np.inf, 0, -other.values))
        return self + other

    def __truediv__(self, other):
        # assert np.isscalar(other), 'divisor must be a scalar'
        if np.isscalar(other):
            new_values = self.values / other
            new_values = np.nan_to_num(new_values)
            return Factor(self.domain, new_values)
        tmp = other.expand(self.domain)
        vals = np.divide(self.values, tmp.values, where=tmp.values > 0)
        vals[tmp.values <= 0] = 0.0
        return Factor(self.domain, vals)

    def exp(self, out=None):
        if out is None:
            return Factor(self.domain, np.exp(self.values))
        np.exp(self.values, out=out.values)
        return out

    def log(self, out=None):
        if out is None:
            return Factor(self.domain, np.log(self.values + 1e-100))
        np.log(self.values, out=out.values)
        return out

    def datavector(self, flatten=True):
        """ Materialize the data vector """
        if flatten:
            return self.values.flatten()
        return self.values


class FactorGraph():
    def __init__(self, domain, cliques, total=1.0, convex=False, iters=25):
        self.domain = domain
        self.cliques = cliques
        self.total = total
        self.convex = convex
        self.iters = iters

        if convex:
            self.counting_numbers = self.get_counting_numbers()
            self.belief_propagation = self.convergent_belief_propagation
        else:
            counting_numbers = {}
            for cl in cliques:
                counting_numbers[cl] = 1.0
            for a in domain:
                counting_numbers[a] = 1.0 - len([cl for cl in cliques if a in cl])
            self.counting_numbers = None, None, counting_numbers
            self.belief_propagation = self.loopy_belief_propagation

        self.potentials = None
        self.marginals = None
        self.messages = self.init_messages()
        self.beliefs = {i: Factor.zeros(domain.project(i)) for i in domain}

    def datavector(self, flatten=True):
        """ Materialize the explicit representation of the distribution as a data vector. """
        logp = sum(self.potentials[cl] for cl in self.cliques)
        ans = np.exp(logp - logp.logsumexp())
        wgt = ans.domain.size() / self.domain.size()
        return ans.expand(self.domain).datavector(flatten) * wgt * self.total

    def init_messages(self):
        mu_n = defaultdict(dict)
        mu_f = defaultdict(dict)
        for cl in self.cliques:
            for v in cl:
                mu_f[cl][v] = Factor.zeros(self.domain.project(v))
                mu_n[v][cl] = Factor.zeros(self.domain.project(v))
        return mu_n, mu_f

    def primal_feasibility(self, mu):
        ans = 0
        count = 0
        for r in mu:
            for s in mu:
                if r == s: break
                d = tuple(set(r) & set(s))
                if len(d) > 0:
                    x = mu[r].project(d).datavector()
                    y = mu[s].project(d).datavector()
                    err = np.linalg.norm(x - y, 1)
                    ans += err
                    count += 1
        try:
            return ans / count
        except:
            return 0

    def project(self, attrs):
        if type(attrs) is list:
            attrs = tuple(attrs)

        if self.marginals is not None:
            # we will average all ways to obtain the given marginal,
            # since there may be more than one
            ans = Factor.zeros(self.domain.project(attrs))
            terminate = False
            for cl in self.cliques:
                if set(attrs) <= set(cl):
                    ans += self.marginals[cl].project(attrs)
                    terminate = True
            if terminate: return ans * (self.total / ans.sum())

        belief = sum(self.beliefs[i] for i in attrs)
        belief += np.log(self.total) - belief.logsumexp()
        return belief.transpose(attrs).exp()

    def loopy_belief_propagation(self, potentials, callback=None):

        mu_n, mu_f = self.messages
        self.potentials = potentials

        for i in range(self.iters):
            # factor to variable BP
            for cl in self.cliques:
                pre = sum(mu_n[c][cl] for c in cl)
                for v in cl:
                    complement = [var for var in cl if var is not v]
                    mu_f[cl][v] = potentials[cl] + pre - mu_n[v][cl]
                    mu_f[cl][v] = mu_f[cl][v].logsumexp(complement)
                    mu_f[cl][v] -= mu_f[cl][v].logsumexp()

            # variable to factor BP
            for v in self.domain:
                fac = [cl for cl in self.cliques if v in cl]
                pre = sum(mu_f[cl][v] for cl in fac)
                for f in fac:
                    complement = [var for var in fac if var is not f]
                    # mu_n[v][f] = Factor.zeros(self.domain.project(v))
                    mu_n[v][f] = pre - mu_f[f][v]  # sum(mu_f[c][v] for c in complement)
                    # mu_n[v][f] += sum(mu_f[c][v] for c in complement)
                    # mu_n[v][f] -= mu_n[v][f].logsumexp()

            if callback is not None:
                mg = self.clique_marginals(mu_n, mu_f, potentials)
                callback(mg)

        self.beliefs = {v: sum(mu_f[cl][v] for cl in self.cliques if v in cl) for v in self.domain}
        self.messages = mu_n, mu_f
        self.marginals = self.clique_marginals(mu_n, mu_f, potentials)
        return self.marginals

    def convergent_belief_propagation(self, potentials, callback=None):
        # Algorithm 11.2 in Koller & Friedman (modified to work in log space)

        v, vhat, k = self.counting_numbers
        sigma, delta = self.messages
        # sigma, delta = self.init_messages()

        for it in range(self.iters):

            # pre = {}
            # for r in self.cliques:
            #    pre[r] = sum(sigma[j][r] for j in r)

            for i in self.domain:
                nbrs = [r for r in self.cliques if i in r]
                for r in nbrs:
                    comp = [j for j in r if i != j]
                    delta[r][i] = potentials[r] + sum(sigma[j][r] for j in comp)
                    # delta[r][i] = potentials[r] + pre[r] - sigma[i][r]
                    delta[r][i] /= vhat[i, r]
                    delta[r][i] = delta[r][i].logsumexp(comp)
                belief = Factor.zeros(self.domain.project(i))
                belief += sum(delta[r][i] * vhat[i, r] for r in nbrs) / vhat[i]
                belief -= belief.logsumexp()
                self.beliefs[i] = belief
                for r in nbrs:
                    comp = [j for j in r if i != j]
                    A = -v[i, r] / vhat[i, r]
                    B = v[r]
                    sigma[i][r] = A * (potentials[r] + sum(sigma[j][r] for j in comp))
                    # sigma[i][r] = A*(potentials[r] + pre[r] - sigma[i][r])
                    sigma[i][r] += B * (belief - delta[r][i])
            if callback is not None:
                mg = self.clique_marginals(sigma, delta, potentials)
                callback(mg)

        self.messages = sigma, delta
        return self.clique_marginals(sigma, delta, potentials)

    def clique_marginals(self, mu_n, mu_f, potentials):
        if self.convex: v, _, _ = self.counting_numbers
        marginals = {}
        for cl in self.cliques:
            belief = potentials[cl] + sum(mu_n[n][cl] for n in cl)
            if self.convex: belief *= 1.0 / v[cl]
            belief += np.log(self.total) - belief.logsumexp()
            marginals[cl] = belief.exp()
        return CliqueVector(marginals)

    def mle(self, marginals):
        return -self.bethe_entropy(marginals)[1]

    def bethe_entropy(self, marginals):
        """
        Return the Bethe Entropy and the gradient with respect to the marginals

        """
        _, _, weights = self.counting_numbers
        entropy = 0
        dmarginals = {}
        attributes = set()
        for cl in self.cliques:
            mu = marginals[cl] / self.total
            entropy += weights[cl] * (mu * mu.log()).sum()
            dmarginals[cl] = weights[cl] * (1 + mu.log()) / self.total
            for a in set(cl) - set(attributes):
                p = mu.project(a)
                entropy += weights[a] * (p * p.log()).sum()
                dmarginals[cl] += weights[a] * (1 + p.log()) / self.total
                attributes.update(a)

        return -entropy, -1 * CliqueVector(dmarginals)

    def get_counting_numbers(self):

        solvers.options['show_progress'] = False
        index = {}
        idx = 0

        for i in self.domain:
            index[i] = idx
            idx += 1
        for r in self.cliques:
            index[r] = idx
            idx += 1

        for r in self.cliques:
            for i in r:
                index[r, i] = idx
                idx += 1

        vectors = {}
        for r in self.cliques:
            v = np.zeros(idx)
            v[index[r]] = 1
            for i in r:
                v[index[r, i]] = 1
            vectors[r] = v

        for i in self.domain:
            v = np.zeros(idx)
            v[index[i]] = 1
            for r in self.cliques:
                if i in r:
                    v[index[r, i]] = -1
            vectors[i] = v

        constraints = []
        for i in self.domain:
            con = vectors[i].copy()
            for r in self.cliques:
                if i in r:
                    con += vectors[r]
            constraints.append(con)
        A = np.array(constraints)
        b = np.ones(len(self.domain))

        X = np.vstack([vectors[r] for r in self.cliques])
        y = np.ones(len(self.cliques))
        P = X.T @ X
        q = -X.T @ y
        G = -np.eye(q.size)
        h = np.zeros(q.size)
        minBound = 1.0 / len(self.domain)
        for r in self.cliques:
            h[index[r]] = -minBound

        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)

        ans = solvers.qp(P, q, G, h, A, b)
        x = np.array(ans['x']).flatten()
        # for p in vectors: print(p, vectors[p] @ x)

        counting_v = {}
        for r in self.cliques:
            counting_v[r] = x[index[r]]
            for i in r:
                counting_v[i, r] = x[index[r, i]]
        for i in self.domain:
            counting_v[i] = x[index[i]]

        counting_vhat = {}
        counting_k = {}
        for i in self.domain:
            nbrs = [r for r in self.cliques if i in r]
            counting_vhat[i] = counting_v[i] + sum(counting_v[r] for r in nbrs)
            counting_k[i] = counting_v[i] - sum(counting_v[i, r] for r in nbrs)
            for r in nbrs:
                counting_vhat[i, r] = counting_v[r] + counting_v[i, r]
        for r in self.cliques:
            counting_k[r] = counting_v[r] + sum(counting_v[i, r] for i in r)

        return counting_v, counting_vhat, counting_k


class GraphicalModel:
    def __init__(self, domain, cliques, total=1.0, elimination_order=None):
        """ Constructor for a GraphicalModel

        :param domain: a Domain object
        :param total: the normalization constant for the distribution
        :param cliques: a list of cliques (not necessarilly maximal cliques)
            - each clique is a subset of attributes, represented as a tuple or list
        :param elim_order: an elimination order for the JunctionTree algorithm
            - Elimination order will impact the efficiency by not correctness.
              By default, a greedy elimination order is used
        """
        self.domain = domain
        self.total = total
        tree = JunctionTree(domain, cliques, elimination_order)
        self.junction_tree = tree

        self.cliques = tree.maximal_cliques()  # maximal cliques
        self.message_order = tree.mp_order()
        self.sep_axes = tree.separator_axes()
        self.neighbors = tree.neighbors()
        self.elimination_order = tree.elimination_order

        self.size = sum(domain.size(cl) for cl in self.cliques)
        if self.size * 8 > 4 * 10 ** 9:
            import warnings
            message = 'Size of parameter vector is %.2f GB. ' % (self.size * 8 / 10 ** 9)
            message += 'Consider removing some measurements or finding a better elimination order'
            warnings.warn(message)

    @staticmethod
    def save(model, path):
        pickle.dump(model, open(path, 'wb'))

    @staticmethod
    def load(path):
        return pickle.load(open(path, 'rb'))

    def project(self, attrs):
        """ Project the distribution onto a subset of attributes.
            I.e., compute the marginal of the distribution

        :param attrs: a subset of attributes in the domain, represented as a list or tuple
        :return: a Factor object representing the marginal distribution
        """
        # use precalculated marginals if possible
        if type(attrs) is list:
            attrs = tuple(attrs)
        if hasattr(self, 'marginals'):
            for cl in self.cliques:
                if set(attrs) <= set(cl):
                    return self.marginals[cl].project(attrs)

        elim = self.domain.invert(attrs)
        elim_order = greedy_order(self.domain, self.cliques + [attrs], elim)
        pots = list(self.potentials.values())
        ans = variable_elimination_logspace(pots, elim_order, self.total)

        return ans.project(attrs)

    def krondot(self, matrices):
        """ Compute the answer to the set of queries Q1 x Q2 X ... x Qd, where
            Qi is a query matrix on the ith attribute and "x" is the Kronecker product
        This may be more efficient than computing a supporting marginal then multiplying that by Q.
        In particular, if each Qi has only a few rows.

        :param matrices: a list of matrices for each attribute in the domain
        :return: the vector of query answers
        """
        assert all(M.shape[1] == n for M, n in zip(matrices, self.domain.shape)), \
            'matrices must conform to the shape of the domain'
        logZ = self.belief_propagation(self.potentials, logZ=True)
        factors = [self.potentials[cl].exp() for cl in self.cliques]
        Factor = type(factors[0])  # infer the type of the factors
        elim = self.domain.attrs
        for attr, Q in zip(elim, matrices):
            d = Domain(['%s-answer' % attr, attr], Q.shape)
            factors.append(Factor(d, Q))
        result = variable_elimination(factors, elim)
        result = result.transpose(['%s-answer' % a for a in elim])
        return result.datavector(flatten=False) * self.total / np.exp(logZ)

    def calculate_many_marginals(self, projections):
        """ Calculates marginals for all the projections in the list using
            Algorithm for answering many out-of-clique queries (section 10.3 in Koller and Friedman)

        This method may be faster than calling project many times

        :param projections: a list of projections, where
            each projection is a subset of attributes (represented as a list or tuple)
        :return: a list of marginals, where each marginal is represented as a Factor
        """

        self.marginals = self.belief_propagation(self.potentials)
        sep = self.sep_axes
        neighbors = self.neighbors
        # first calculate P(Cj | Ci) for all neighbors Ci, Cj
        conditional = {}
        for Ci in neighbors:
            for Cj in neighbors[Ci]:
                Sij = sep[(Cj, Ci)]
                Z = self.marginals[Cj]
                conditional[(Cj, Ci)] = Z / Z.project(Sij)

        # now iterate through pairs of cliques in order of distance
        pred, dist = nx.floyd_warshall_predecessor_and_distance(self.junction_tree.tree, weight=False)
        results = {}
        for Ci, Cj in sorted(itertools.combinations(self.cliques, 2), key=lambda X: dist[X[0]][X[1]]):
            Cl = pred[Ci][Cj]
            Y = conditional[(Cj, Cl)]
            if Cl == Ci:
                X = self.marginals[Ci]
                results[(Ci, Cj)] = results[(Cj, Ci)] = X * Y
            else:
                X = results[(Ci, Cl)]
                S = set(Cl) - set(Ci) - set(Cj)
                results[(Ci, Cj)] = results[(Cj, Ci)] = (X * Y).sum(S)

        results = {self.domain.canonical(key[0] + key[1]): results[key] for key in results}

        answers = {}
        for proj in projections:
            for attr in results:
                if set(proj) <= set(attr):
                    answers[proj] = results[attr].project(proj)
                    break
            if proj not in answers:
                # just use variable elimination
                answers[proj] = self.project(proj)

        return answers

    def datavector(self, flatten=True):
        """ Materialize the explicit representation of the distribution as a data vector. """
        logp = sum(self.potentials[cl] for cl in self.cliques)
        ans = np.exp(logp - logp.logsumexp())
        wgt = ans.domain.size() / self.domain.size()
        return ans.expand(self.domain).datavector(flatten) * wgt * self.total

    def belief_propagation(self, potentials, logZ=False):
        """ Compute the marginals of the graphical model with given parameters

        Note this is an efficient, numerically stable implementation of belief propagation

        :param potentials: the (log-space) parameters of the graphical model
        :param logZ: flag to return logZ instead of marginals
        :return marginals: the marginals of the graphical model
        """
        beliefs = {cl: potentials[cl].copy() for cl in potentials}
        messages = {}
        for i, j in self.message_order:
            sep = beliefs[i].domain.invert(self.sep_axes[(i, j)])
            if (j, i) in messages:
                tau = beliefs[i] - messages[(j, i)]
            else:
                tau = beliefs[i]
            messages[(i, j)] = tau.logsumexp(sep)
            beliefs[j] += messages[(i, j)]

        cl = self.cliques[0]
        if logZ: return beliefs[cl].logsumexp()

        logZ = beliefs[cl].logsumexp()
        for cl in self.cliques:
            beliefs[cl] += np.log(self.total) - logZ
            beliefs[cl] = beliefs[cl].exp(out=beliefs[cl])

        return CliqueVector(beliefs)

    def mle(self, marginals):
        """ Compute the model parameters from the given marginals

        :param marginals: target marginals of the distribution
        :param: the potentials of the graphical model with the given marginals
        """
        potentials = {}
        variables = set()
        for cl in self.cliques:
            new = tuple(variables & set(cl))
            # factor = marginals[cl] / marginals[cl].project(new)
            variables.update(cl)
            potentials[cl] = marginals[cl].log() - marginals[cl].project(new).log()
        return CliqueVector(potentials)

    def fit(self, data):
        assert data.domain.contains(self.domain), 'model domain not compatible with data domain'
        marginals = {}
        for cl in self.cliques:
            x = data.project(cl).datavector()
            dom = self.domain.project(cl)
            marginals[cl] = Factor(dom, x)
        self.potentials = self.mle(marginals)

    def synthetic_data(self, rows=None, method='round'):
        """ Generate synthetic tabular data from the distribution.
            Valid options for method are 'round' and 'sample'."""

        total = int(self.total) if rows is None else rows #48822
        cols = self.domain.attrs
        data = np.zeros((total, len(cols)), dtype=int)
        df = pd.DataFrame(data, columns=cols) #14 attributes with 3 attributes as a group
        cliques = [set(cl) for cl in self.cliques] #14 attributes with 3 attributes as a group

        def synthetic_col(counts, total):
            "Generate a col of synthetic tabular data from the distribution"
            if method == 'sample':
                probas = counts / counts.sum()
                return np.random.choice(counts.size, total, True, probas)
            counts *= total / counts.sum()
            frac, integ = np.modf(counts)#divide float as frac and integ
            integ = integ.astype(int)
            extra = total - integ.sum()
            if extra > 0:#allocate extra individual into a group
                idx = np.random.choice(counts.size, extra, False, frac / frac.sum())
                integ[idx] += 1
            vals = np.repeat(np.arange(counts.size), integ)
            np.random.shuffle(vals)
            return vals

        order = self.elimination_order[::-1] #14 attributes
        col = order[0] #a attribute

        marg = self.project([col]).datavector(flatten=False) #m每个属性对应的每个值的个数
        df.loc[:, col] = synthetic_col(marg, total) # get the first col in synthetic tabular data
        used = {col}#a attribute

        for col in order[1:]:#residual all attributes
            relevant = [cl for cl in cliques if col in cl] #marginal including a attribute called col
            relevant = used.intersection(set.union(*relevant))
            proj = tuple(relevant)
            used.add(col)
            marg = self.project(proj + (col,)).datavector(flatten=False)


            def foo(group):
                idx = group.name
                vals = synthetic_col(marg[idx], group.shape[0])
                group[col] = vals
                return group

            if len(proj) >= 1:
                df = df.groupby(list(proj), group_keys=False).apply(foo)
            else:
                df[col] = synthetic_col(marg, df.shape[0])

        return Dataset(df, self.domain)


class MWModel:
    def __init__(self, factors, domain, cliques, total=1.0, elimination_order=None):
        """ Constructor for a GraphicalModel

        :param domain: a Domain object
        :param total: the normalization constant for the distribution
        :param cliques: a list of cliques (not necessarilly maximal cliques)
            - each clique is a subset of attributes, represented as a tuple or list
        :param elim_order: an elimination order for the JunctionTree algorithm
            - Elimination order will impact the efficiency by not correctness.
              By default, a greedy elimination order is used
        """
        self.domain = domain
        self.total = total
        tree = JunctionTree(domain, cliques, elimination_order)
        self.junction_tree = tree

        self.cliques = tree.maximal_cliques()  # maximal cliques
        self.message_order = tree.mp_order()
        self.sep_axes = tree.separator_axes()
        self.neighbors = tree.neighbors()
        self.elimination_order = tree.elimination_order

        self.size = sum(domain.size(cl) for cl in self.cliques)
        if self.size * 8 > 4 * 10 ** 9:
            import warnings
            message = 'Size of parameter vector is %.2f GB. ' % (self.size * 8 / 10 ** 9)
            message += 'Consider removing some measurements or finding a better elimination order'
            warnings.warn(message)

        self.factors = factors
        for a in domain:
            if not any(a in f.domain for f in factors):
                sub = domain.project([a])
                x = np.ones(domain[a]) / domain[a]
                factors.append(Factor(sub, x))

    @staticmethod
    def save(model, path):
        pickle.dump(model, open(path, 'wb'))

    @staticmethod
    def load(path):
        return pickle.load(open(path, 'rb'))

    def project(self, attrs):
        """ Project the distribution onto a subset of attributes.
            I.e., compute the marginal of the distribution

        :param attrs: a subset of attributes in the domain, represented as a list or tuple
        :return: a Factor object representing the marginal distribution
        """
        # use precalculated marginals if possible
        if type(attrs) is list:
            attrs = tuple(attrs)
        if hasattr(self, 'marginals'):
            for cl in self.cliques:
                if set(attrs) <= set(cl):
                    return self.marginals[cl].project(attrs)

        elim = self.domain.invert(attrs)
        elim_order = greedy_order(self.domain, self.cliques + [attrs], elim)
        pots = list(self.potentials.values())
        ans = variable_elimination_logspace(pots, elim_order, self.total)#<src.mbi.mbiii.Factor object at 0x0000025DFEB6BD90>
        return ans.project(attrs)#<src.mbi.mbiii.Factor object at 0x000001D79186EC50>

    def krondot(self, matrices):
        """ Compute the answer to the set of queries Q1 x Q2 X ... x Qd, where
            Qi is a query matrix on the ith attribute and "x" is the Kronecker product
        This may be more efficient than computing a supporting marginal then multiplying that by Q.
        In particular, if each Qi has only a few rows.

        :param matrices: a list of matrices for each attribute in the domain
        :return: the vector of query answers
        """
        assert all(M.shape[1] == n for M, n in zip(matrices, self.domain.shape)), \
            'matrices must conform to the shape of the domain'
        logZ = self.belief_propagation(self.potentials, logZ=True)
        factors = [self.potentials[cl].exp() for cl in self.cliques]
        Factor = type(factors[0])  # infer the type of the factors
        elim = self.domain.attrs
        for attr, Q in zip(elim, matrices):
            d = Domain(['%s-answer' % attr, attr], Q.shape)
            factors.append(Factor(d, Q))
        result = variable_elimination(factors, elim)
        result = result.transpose(['%s-answer' % a for a in elim])
        return result.datavector(flatten=False) * self.total / np.exp(logZ)

    def calculate_many_marginals(self, projections):
        """ Calculates marginals for all the projections in the list using
            Algorithm for answering many out-of-clique queries (section 10.3 in Koller and Friedman)

        This method may be faster than calling project many times

        :param projections: a list of projections, where
            each projection is a subset of attributes (represented as a list or tuple)
        :return: a list of marginals, where each marginal is represented as a Factor
        """

        self.marginals = self.belief_propagation(self.potentials)
        sep = self.sep_axes
        neighbors = self.neighbors
        # first calculate P(Cj | Ci) for all neighbors Ci, Cj
        conditional = {}
        for Ci in neighbors:
            for Cj in neighbors[Ci]:
                Sij = sep[(Cj, Ci)]
                Z = self.marginals[Cj]
                conditional[(Cj, Ci)] = Z / Z.project(Sij)

        # now iterate through pairs of cliques in order of distance
        pred, dist = nx.floyd_warshall_predecessor_and_distance(self.junction_tree.tree, weight=False)
        results = {}
        for Ci, Cj in sorted(itertools.combinations(self.cliques, 2), key=lambda X: dist[X[0]][X[1]]):
            Cl = pred[Ci][Cj]
            Y = conditional[(Cj, Cl)]
            if Cl == Ci:
                X = self.marginals[Ci]
                results[(Ci, Cj)] = results[(Cj, Ci)] = X * Y
            else:
                X = results[(Ci, Cl)]
                S = set(Cl) - set(Ci) - set(Cj)
                results[(Ci, Cj)] = results[(Cj, Ci)] = (X * Y).sum(S)

        results = {self.domain.canonical(key[0] + key[1]): results[key] for key in results}

        answers = {}
        for proj in projections:
            for attr in results:
                if set(proj) <= set(attr):
                    answers[proj] = results[attr].project(proj)
                    break
            if proj not in answers:
                # just use variable elimination
                answers[proj] = self.project(proj)

        return answers

    def datavector(self, flatten=True):
        """ Materialize the explicit representation of the distribution as a data vector. """
        logp = sum(self.potentials[cl] for cl in self.cliques)
        ans = np.exp(logp - logp.logsumexp())
        wgt = ans.domain.size() / self.domain.size()
        return ans.expand(self.domain).datavector(flatten) * wgt * self.total


    def belief_propagation(self, potentials, logZ=False):
        """ Compute the marginals of the graphical model with given parameters

        Note this is an efficient, numerically stable implementation of belief propagation

        :param potentials: the (log-space) parameters of the graphical model
        :param logZ: flag to return logZ instead of marginals
        :return marginals: the marginals of the graphical model
        """
        beliefs = {cl: potentials[cl].copy() for cl in potentials}
        messages = {}
        for i, j in self.message_order:
            sep = beliefs[i].domain.invert(self.sep_axes[(i, j)])
            if (j, i) in messages:
                tau = beliefs[i] - messages[(j, i)]
            else:
                tau = beliefs[i]
            messages[(i, j)] = tau.logsumexp(sep)
            beliefs[j] += messages[(i, j)]

        cl = self.cliques[0]
        if logZ: return beliefs[cl].logsumexp()

        logZ = beliefs[cl].logsumexp()
        for cl in self.cliques:
            beliefs[cl] += np.log(self.total) - logZ
            beliefs[cl] = beliefs[cl].exp(out=beliefs[cl])

        return CliqueVector(beliefs)

    def mle(self, marginals):
        """ Compute the model parameters from the given marginals

        :param marginals: target marginals of the distribution
        :param: the potentials of the graphical model with the given marginals
        """
        potentials = {}
        variables = set()
        for cl in self.cliques:
            new = tuple(variables & set(cl))
            # factor = marginals[cl] / marginals[cl].project(new)
            variables.update(cl)
            potentials[cl] = marginals[cl].log() - marginals[cl].project(new).log()
        return CliqueVector(potentials)

    def fit(self, data):
        assert data.domain.contains(self.domain), 'model domain not compatible with data domain'
        marginals = {}
        for cl in self.cliques:
            x = data.project(cl).datavector()
            dom = self.domain.project(cl)
            marginals[cl] = Factor(dom, x)
        self.potentials = self.mle(marginals)

    def synthetic_data(self, rows=None, method='round'):
        """ Generate synthetic tabular data from the distribution.
            Valid options for method are 'round' and 'sample'."""

        total = int(self.total) if rows is None else rows #48822
        cols = self.domain.attrs
        data = np.zeros((total, len(cols)), dtype=int)
        df = pd.DataFrame(data, columns=cols) #14 attributes with 3 attributes as a group
        cliques = [set(cl) for cl in self.cliques] #14 attributes with 3 attributes as a group

        def synthetic_col(counts, total):
            "Generate a col of synthetic tabular data from the distribution"
            if method == 'sample':
                probas = counts / counts.sum()
                return np.random.choice(counts.size, total, True, probas)
            counts *= total / counts.sum()
            frac, integ = np.modf(counts)#divide float as frac and integ
            integ = integ.astype(int)
            extra = total - integ.sum()
            if extra > 0:#allocate extra individual into a group
                idx = np.random.choice(counts.size, extra, False, frac / frac.sum())
                integ[idx] += 1
            vals = np.repeat(np.arange(counts.size), integ)
            np.random.shuffle(vals)
            return vals

        order = self.elimination_order[::-1] #14 attributes
        col = order[0] #a attribute

        marg = self.project([col]).datavector(flatten=False) #m每个属性对应的每个值的个数
        df.loc[:, col] = synthetic_col(marg, total) # get the first col in synthetic tabular data
        used = {col}#a attribute

        for col in order[1:]:#residual all attributes
            relevant = [cl for cl in cliques if col in cl] #marginal including a attribute called col
            relevant = used.intersection(set.union(*relevant))
            proj = tuple(relevant)
            used.add(col)
            marg = self.project(proj + (col,)).datavector(flatten=False)


            def foo(group):
                idx = group.name
                vals = synthetic_col(marg[idx], group.shape[0])
                group[col] = vals
                return group

            if len(proj) >= 1:
                df = df.groupby(list(proj), group_keys=False).apply(foo)
            else:
                df[col] = synthetic_col(marg, df.shape[0])

        return Dataset(df, self.domain)


def variable_elimination_logspace(potentials, elim, total):
    """ run variable elimination on a list of **logspace** factors """
    k = len(potentials)
    psi = dict(zip(range(k), potentials))
    for z in elim:
        psi2 = [psi.pop(i) for i in list(psi.keys()) if z in psi[i].domain]
        phi = reduce(lambda x, y: x + y, psi2, 0)
        tau = phi.logsumexp([z])
        psi[k] = tau
        k += 1
    ans = reduce(lambda x, y: x + y, psi.values(), 0)
    return (ans - ans.logsumexp() + np.log(total)).exp()


def variable_elimination(factors, elim):
    """ run variable elimination on a list of (non-logspace) factors """
    k = len(factors)
    psi = dict(zip(range(k), factors))
    for z in elim:
        psi2 = [psi.pop(i) for i in list(psi.keys()) if z in psi[i].domain]
        phi = reduce(lambda x, y: x * y, psi2, 1)
        tau = phi.sum([z])
        psi[k] = tau
        k += 1
    return reduce(lambda x, y: x * y, psi.values(), 1)


def greedy_order(domain, cliques, elim):
    order = []
    unmarked = set(elim)
    cliques = set(cliques)
    total_cost = 0
    for k in range(len(elim)):
        cost = {}
        for a in unmarked:
            # all cliques that have a
            neighbors = list(filter(lambda cl: a in cl, cliques))
            # variables in this "super-clique"
            variables = tuple(set.union(set(), *map(set, neighbors)))
            # domain for the resulting factor
            newdom = domain.project(variables)
            # cost of removing a
            cost[a] = newdom.size()

        # find the best variable to eliminate
        a = min(cost, key=lambda a: cost[a])

        # do some cleanup
        order.append(a)
        unmarked.remove(a)
        neighbors = list(filter(lambda cl: a in cl, cliques))
        variables = tuple(set.union(set(), *map(set, neighbors)) - {a})
        cliques -= set(neighbors)
        cliques.add(variables)
        total_cost += cost[a]

    return order


class FactoredInference:
    def __init__(self, domain, backend='numpy', structural_zeros={}, metric='L2', log=False, iters=1000,
                 warm_start=False, elim_order=None):
        """
        Class for learning a GraphicalModel from  noisy measurements on a data distribution

        :param domain: The domain information (A Domain object)
        :param backend: numpy or torch backend
        :param structural_zeros: An encoding of the known (structural) zeros in the distribution.
            Specified as a dictionary where
                - each key is a subset of attributes of size r
                - each value is a list of r-tuples corresponding to impossible attribute settings
        :param metric: The optimization metric.  May be L1, L2 or a custom callable function
            - custom callable function must consume the marginals and produce the loss and gradient
            - see FactoredInference._marginal_loss for more information
        :param log: flag to log iterations of optimization
        :param iters: number of iterations to optimize for
        :param warm_start: initialize new model or reuse last model when calling infer multiple times
        :param elim_order: an elimination order for the JunctionTree algorithm
            - Elimination order will impact the efficiency by not correctness.
              By default, a greedy elimination order is used
        """
        self.domain = domain
        self.backend = backend
        self.metric = metric
        self.log = log
        self.iters = iters
        self.warm_start = warm_start
        self.history = []
        self.elim_order = elim_order
        self.Factor = Factor
        # if backend == 'torch':
        #
        #     self.Factor = Factor
        # else:
        #
        #     self.Factor = Factor_t

        self.structural_zeros = CliqueVector({})
        for cl in structural_zeros:
            dom = self.domain.project(cl)
            fact = structural_zeros[cl]
            self.structural_zeros[cl] = self.Factor.active(dom, fact)

    def estimate(self, measurements, total=None, engine='MD', callback=None, options={}):
        """
        Estimate a GraphicalModel from the given measurements

        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param engine: the optimization algorithm to use, options include:
            MD - Mirror Descent with armijo line search
            RDA - Regularized Dual Averaging
            IG - Interior Gradient
        :param callback: a function to be called after each iteration of optimization
        :param options: solver specific options passed as a dictionary
            { param_name : param_value }

        :return model: A GraphicalModel that best matches the measurements taken
        """
        measurements = self.fix_measurements(measurements)
        options['callback'] = callback
        if callback is None and self.log:
            options['callback'] = Logger(self)
        if engine == 'MD':
            self.mirror_descent(measurements, total, **options)
        elif engine == 'RDA':
            self.dual_averaging(measurements, total, **options)
        elif engine == 'IG':
            self.interior_gradient(measurements, total, **options)
        return self.model

    def fix_measurements(self, measurements):
        assert type(measurements) is list, 'measurements must be a list, given ' + measurements
        assert all(len(m) == 4 for m in measurements), \
            'each measurement must be a 4-tuple (Q, y, noise,proj)'
        ans = []
        for Q, y, noise, proj in measurements:
            assert Q is None or Q.shape[0] == y.size, 'shapes of Q and y are not compatible'
            if type(proj) is list:
                proj = tuple(proj)
            if type(proj) is not tuple:
                proj = (proj,)
            if Q is None:
                Q = sparse.eye(self.domain.size(proj))
            assert np.isscalar(noise), 'noise must be a real value, given ' + str(noise)
            assert all(a in self.domain for a in proj), str(proj) + ' not contained in domain'
            assert Q.shape[1] == self.domain.size(proj), 'shapes of Q and proj are not compatible'
            ans.append((Q, y, noise, proj))
        return ans

    def interior_gradient(self, measurements, total, lipschitz=None, c=1, sigma=1, callback=None):
        """ Use the interior gradient algorithm to estimate the GraphicalModel
            See https://epubs.siam.org/doi/pdf/10.1137/S1052623403427823 for more information

        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param lipschitz: the Lipchitz constant of grad L(mu)
            - automatically calculated for metric=L2
            - doesn't exist for metric=L1
            - must be supplied for custom callable metrics
        :param c, sigma: parameters of the algorithm
        :param callback: a function to be called after each iteration of optimization
        """
        assert self.metric != 'L1', 'dual_averaging cannot be used with metric=L1'
        assert not callable(self.metric) or lipschitz is not None, 'lipschitz constant must be supplied'
        self._setup(measurements, total)
        # what are c and sigma?  For now using 1
        model = self.model
        domain, cliques, total = model.domain, model.cliques, model.total
        L = self._lipschitz(measurements) if lipschitz is None else lipschitz
        if self.log:
            print('Lipchitz constant:', L)

        theta = model.potentials
        x = y = z = model.belief_propagation(theta)
        c0 = c
        l = sigma / L
        for k in range(1, self.iters + 1):
            a = (np.sqrt((c * l) ** 2 + 4 * c * l) - l * c) / 2
            y = (1 - a) * x + a * z
            c *= (1 - a)
            _, g = self._marginal_loss(y)
            theta = theta - a / c / total * g
            z = model.belief_propagation(theta)
            x = (1 - a) * x + a * z
            if callback is not None:
                callback(x)

        model.marginals = x
        model.potentials = model.mle(x)

    def dual_averaging(self, measurements, total=None, lipschitz=None, callback=None):
        """ Use the regularized dual averaging algorithm to estimate the GraphicalModel
            See https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/xiao10JMLR.pdf

        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param lipschitz: the Lipchitz constant of grad L(mu)
            - automatically calculated for metric=L2
            - doesn't exist for metric=L1
            - must be supplied for custom callable metrics
        :param callback: a function to be called after each iteration of optimization
        """
        assert self.metric != 'L1', 'dual_averaging cannot be used with metric=L1'
        assert not callable(self.metric) or lipschitz is not None, 'lipschitz constant must be supplied'
        self._setup(measurements, total)
        model = self.model
        domain, cliques, total = model.domain, model.cliques, model.total
        L = self._lipschitz(measurements) if lipschitz is None else lipschitz
        print('Lipchitz constant:', L)
        if L == 0: return

        theta = model.potentials
        gbar = CliqueVector({cl: self.Factor.zeros(domain.project(cl)) for cl in cliques})
        w = v = model.belief_propagation(theta)
        beta = 0

        for t in range(1, self.iters + 1):
            c = 2.0 / (t + 1)
            u = (1 - c) * w + c * v
            _, g = self._marginal_loss(u)  # not interested in loss of this query point
            gbar = (1 - c) * gbar + c * g
            theta = -t * (t + 1) / (4 * L + beta) / self.model.total * gbar
            v = model.belief_propagation(theta)
            w = (1 - c) * w + c * v

            if callback is not None:
                callback(w)

        model.marginals = w
        model.potentials = model.mle(w)

    def mirror_descent(self, measurements, total=None, stepsize=None, callback=None):
        """ Use the mirror descent algorithm to estimate the GraphicalModel
            See https://web.iem.technion.ac.il/images/user-files/becka/papers/3.pdf

        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param stepsize: The step size function for the optimization (None or scalar or function)
            if None, will perform line search at each iteration (requires smooth objective)
            if scalar, will use constant step size
            if function, will be called with the iteration number
        :param total: The total number of records (if known)
        :param callback: a function to be called after each iteration of optimization
        """
        assert not (self.metric == 'L1' and stepsize is None), \
            'loss function not smooth, cannot use line search (specify stepsize)'

        self._setup(measurements, total)
        model = self.model
        cliques, theta = model.cliques, model.potentials
        # print(cliques) #各个属性名字
        mu = model.belief_propagation(theta)
        ans = self._marginal_loss(mu)

        if ans[0] == 0:
            return ans[0]

        nols = stepsize is not None
        if np.isscalar(stepsize):
            alpha = float(stepsize)
            stepsize = lambda t: alpha
        if stepsize is None:
            alpha = 1.0 / self.model.total ** 2
            stepsize = lambda t: 2.0 * alpha

        for t in range(1, self.iters + 1):
            if callback is not None:
                callback(mu)
            omega, nu = theta, mu
            curr_loss, dL = ans
            # print('Gradient Norm', np.sqrt(dL.dot(dL)))
            alpha = stepsize(t)
            for i in range(25):
                theta = omega - alpha * dL
                mu = model.belief_propagation(theta)
                ans = self._marginal_loss(mu)
                if nols or curr_loss - ans[0] >= 0.5 * alpha * dL.dot(nu - mu):
                    break
                alpha *= 0.5

        model.potentials = theta
        model.marginals = mu

        return ans[0]

    def _marginal_loss(self, marginals, metric=None):
        """ Compute the loss and gradient for a given dictionary of marginals

        :param marginals: A dictionary with keys as projections and values as Factors
        :return loss: the loss value
        :return grad: A dictionary with gradient for each marginal
        """
        if metric is None:
            metric = self.metric

        if callable(metric):
            return metric(marginals)

        loss = 0.0
        gradient = {}

        for cl in marginals:
            mu = marginals[cl]
            gradient[cl] = self.Factor.zeros(mu.domain)
            for Q, y, noise, proj in self.groups[cl]:
                c = 1.0 / noise
                mu2 = mu.project(proj)
                x = mu2.datavector()
                diff = c * (Q @ x - y)
                if metric == 'L1':
                    loss += abs(diff).sum()
                    sign = diff.sign() if hasattr(diff, 'sign') else np.sign(diff)
                    grad = c * (Q.T @ sign)
                else:
                    loss += 0.5 * (diff @ diff)
                    grad = c * (Q.T @ diff)
                gradient[cl] += self.Factor(mu2.domain, grad)
        return float(loss), CliqueVector(gradient)

    def _setup(self, measurements, total):
        """ Perform necessary setup for running estimation algorithms

        1. If total is None, find the minimum variance unbiased estimate for total and use that
        2. Construct the GraphicalModel
            * If there are structural_zeros in the distribution, initialize factors appropriately
        3. Pre-process measurements into groups so that _marginal_loss may be evaluated efficiently
        """
        if total is None:
            # find the minimum variance estimate of the total given the measurements
            variances = np.array([])
            estimates = np.array([])
            for Q, y, noise, proj in measurements:
                o = np.ones(Q.shape[1])
                v = lsmr(Q.T, o, atol=0, btol=0)[0]#最优解为o
                if np.allclose(Q.T.dot(v), o):#判断最小二乘得到的结果质量
                    variances = np.append(variances, noise ** 2 * np.dot(v, v))
                    estimates = np.append(estimates, np.dot(v, y))
            if estimates.size == 0:
                total = 1
            else:
                variance = 1.0 / np.sum(1.0 / variances)
                estimate = variance * np.sum(estimates / variances)
                total = max(1, estimate)

        # if not self.warm_start or not hasattr(self, 'model'):
        # initialize the model and parameters
        cliques = [m[3] for m in measurements]#get all ax/proj
        if self.structural_zeros is not None:
            cliques += list(self.structural_zeros.keys())

        model = GraphicalModel(self.domain, cliques, total, elimination_order=self.elim_order)

        model.potentials = CliqueVector.zeros(self.domain, model.cliques)
        model.potentials.combine(self.structural_zeros)
        if self.warm_start and hasattr(self, 'model'):
            model.potentials.combine(self.model.potentials)
        self.model = model

        # group the measurements into model cliques
        cliques = self.model.cliques
        # self.groups = { cl : [] for cl in cliques }
        self.groups = defaultdict(lambda: [])
        for Q, y, noise, proj in measurements:
            if self.backend == 'torch':
                import torch
                device = self.Factor.device
                y = torch.tensor(y, dtype=torch.float32, device=device)
                if isinstance(Q, np.ndarray):
                    Q = torch.tensor(Q, dtype=torch.float32, device=device)
                elif sparse.issparse(Q):
                    Q = Q.tocoo()
                    idx = torch.LongTensor([Q.row, Q.col])
                    vals = torch.FloatTensor(Q.data)
                    Q = torch.sparse.FloatTensor(idx, vals).to(device)

                # else Q is a Linear Operator, must be compatible with torch
            m = (Q, y, noise, proj)
            for cl in sorted(cliques, key=model.domain.size):
                # (Q, y, noise, proj) tuple
                if set(proj) <= set(cl):
                    self.groups[cl].append(m)
                    break

    def _lipschitz(self, measurements):
        """ compute lipschitz constant for L2 loss

            Note: must be called after _setup
        """
        eigs = {cl: 0.0 for cl in self.model.cliques}
        for Q, _, noise, proj in measurements:
            for cl in self.model.cliques:
                if set(proj) <= set(cl):
                    n = self.domain.size(cl)
                    p = self.domain.size(proj)
                    Q = aslinearoperator(Q)
                    Q.dtype = np.dtype(Q.dtype)
                    eig = eigsh(Q.H * Q, 1)[0][0]
                    eigs[cl] += eig * n / p / noise ** 2
                    break
        return max(eigs.values())

    def infer(self, measurements, total=None, engine='MD', callback=None, options={}):
        import warnings
        message = "Function infer is deprecated.  Please use estimate instead."
        warnings.warn(message, DeprecationWarning)
        return self.estimate(measurements, total, engine, callback, options)

class JunctionTree:
    """ A JunctionTree is a transformation of a GraphicalModel into a tree structure.  It is used
        to find the maximal cliques in the graphical model, and for specifying the message passing
        order for belief propagation.  The JunctionTree is characterized by an elimination_order,
        which is chosen greedily by default, but may be passed in if desired.
    """

    def __init__(self, domain, cliques, elimination_order=None):
        self.cliques = [tuple(cl) for cl in cliques]
        self.domain = domain
        self.graph = self._make_graph()
        self.tree, self.order = self._make_tree(elimination_order)

    def maximal_cliques(self):
        """ return the list of maximal cliques in the model """
        # return list(self.tree.nodes())
        return list(nx.dfs_preorder_nodes(self.tree))

    def mp_order(self):
        """ return a valid message passing order """
        edges = set()
        messages = [(a, b) for a, b in self.tree.edges()] + [(b, a) for a, b in self.tree.edges()]
        for m1 in messages:
            for m2 in messages:
                if m1[1] == m2[0] and m1[0] != m2[1]:
                    edges.add((m1, m2))
        G = nx.DiGraph()
        G.add_nodes_from(messages)
        G.add_edges_from(edges)
        return list(nx.topological_sort(G))

    def separator_axes(self):
        return {(i, j): tuple(set(i) & set(j)) for i, j in self.mp_order()}

    def neighbors(self):
        return {i: set(self.tree.neighbors(i)) for i in self.maximal_cliques()}

    def _make_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.domain.attrs)
        for cl in self.cliques:
            G.add_edges_from(itertools.combinations(cl, 2))
        return G

    def _triangulated(self, order):
        edges = set()
        G = nx.Graph(self.graph)
        for node in order:
            tmp = set(itertools.combinations(G.neighbors(node), 2))
            edges |= tmp
            G.add_edges_from(tmp)
            G.remove_node(node)
        tri = nx.Graph(self.graph)
        tri.add_edges_from(edges)
        cliques = [tuple(c) for c in nx.find_cliques(tri)]
        cost = sum(self.domain.project(cl).size() for cl in cliques)
        return tri, cost

    def _greedy_order(self, stochastic=True):
        order = []
        domain, cliques = self.domain, self.cliques
        unmarked = list(domain.attrs)
        cliques = set(cliques)
        total_cost = 0
        for k in range(len(domain)):
            cost = OrderedDict()
            for a in unmarked:
                # all cliques that have a
                neighbors = list(filter(lambda cl: a in cl, cliques))
                # variables in this "super-clique"
                variables = tuple(set.union(set(), *map(set, neighbors)))
                # domain for the resulting factor
                newdom = domain.project(variables)
                # cost of removing a
                cost[a] = newdom.size()

            # find the best variable to eliminate
            if stochastic:
                choices = list(unmarked)
                costs = np.array([cost[a] for a in choices], dtype=float)
                probas = np.max(costs) - costs + 1
                probas /= probas.sum()
                i = np.random.choice(probas.size, p=probas)
                a = choices[i]
                # print(choices, probas)
            else:
                a = min(cost, key=lambda a: cost[a])

            # do some cleanup
            order.append(a)
            unmarked.remove(a)
            neighbors = list(filter(lambda cl: a in cl, cliques))
            variables = tuple(set.union(set(), *map(set, neighbors)) - {a})
            cliques -= set(neighbors)
            cliques.add(variables)
            total_cost += cost[a]

        return order, total_cost

    def _make_tree(self, order=None):
        if order is None:
            # orders = [self._greedy_order(stochastic=True) for _ in range(1000)]
            # orders.append(self._greedy_order(stochastic=False))
            # order = min(orders, key=lambda x: x[1])[0]
            order = self._greedy_order(stochastic=False)[0]
        elif type(order) is int:
            orders = [self._greedy_order(stochastic=False)] + [self._greedy_order(stochastic=True) for _ in
                                                               range(order)]
            order = min(orders, key=lambda x: x[1])[0]
        self.elimination_order = order
        tri, cost = self._triangulated(order)
        # cliques = [tuple(c) for c in nx.find_cliques(tri)]
        cliques = sorted([self.domain.canonical(c) for c in nx.find_cliques(tri)])
        complete = nx.Graph()
        complete.add_nodes_from(cliques)
        for c1, c2 in itertools.combinations(cliques, 2):
            wgt = len(set(c1) & set(c2))
            complete.add_edge(c1, c2, weight=-wgt)
        spanning = nx.minimum_spanning_tree(complete)
        return spanning, order


class LocalInference:
    def __init__(self, domain, backend='numpy', structural_zeros={}, metric='L2', log=False, iters=1000,
                 warm_start=False, marginal_oracle='convex', inner_iters=1):
        """
        Class for learning a GraphicalModel from  noisy measurements on a data distribution

        :param domain: The domain information (A Domain object)
        :param backend: numpy or torch backend
        :param structural_zeros: An encoding of the known (structural) zeros in the distribution.
            Specified as a dictionary where
                - each key is a subset of attributes of size r
                - each value is a list of r-tuples corresponding to impossible attribute settings
        :param metric: The optimization metric.  May be L1, L2 or a custom callable function
            - custom callable function must consume the marginals and produce the loss and gradient
            - see FactoredInference._marginal_loss for more information
        :param log: flag to log iterations of optimization
        :param iters: number of iterations to optimize for
        :param warm_start: initialize new model or reuse last model when calling infer multiple times
        :param marginal_oracle: One of
            - convex (Region graph, convex Kikuchi entropy)
            - approx (Region graph, Kikuchi entropy)
            - pairwise-convex (Factor graph, convex Bethe entropy)
            - pairwise (Factor graph, Bethe entropy)
            - Can also pass any and FactorGraph or RegionGraph object
        """
        self.domain = domain
        self.backend = backend
        self.metric = metric
        self.log = log
        self.iters = iters
        self.warm_start = warm_start
        self.history = []
        self.marginal_oracle = marginal_oracle
        self.inner_iters = inner_iters
        if backend == 'torch':
            import Factor
            self.Factor = Factor
        else:
            import Factor_t
            self.Factor = Factor_t

        self.structural_zeros = CliqueVector({})
        for cl in structural_zeros:
            dom = self.domain.project(cl)
            fact = structural_zeros[cl]
            self.structural_zeros[cl] = self.Factor.active(dom, fact)

    def estimate(self, measurements, total=None, callback=None, options={}):
        """
        Estimate a GraphicalModel from the given measurements

        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param callback: a function to be called after each iteration of optimization
        :param options: solver specific options passed as a dictionary
            { param_name : param_value }

        :return model: A GraphicalModel that best matches the measurements taken
        """
        options['callback'] = callback
        if callback is None and self.log:
            options['callback'] = Logger(self)
        self.mirror_descent(measurements, total, **options)
        return self.model

    def mirror_descent_auto(self, alpha, iters, callback=None):
        model = self.model
        theta0 = model.potentials
        messages0 = deepcopy(model.messages)
        theta = theta0
        mu = model.belief_propagation(theta)
        l0, _ = self._marginal_loss(mu)

        prev_l = np.inf
        for t in range(iters):
            if callback is not None:
                callback(mu)
            l, dL = self._marginal_loss(mu)
            theta = theta - alpha * dL
            # print(np.sqrt(dL.dot(dL)), np.sqrt(theta.dot(theta)))
            mu = model.belief_propagation(theta)
            if l > prev_l:
                if t <= 50:
                    if self.log: print('Reducing learning rate and restarting', alpha / 2)
                    model.potentials = theta0
                    model.messages = messages0
                    return self.mirror_descent_auto(alpha / 2, iters, callback)
                else:
                    # print('Reducing learning rate and continuing', alpha/2)
                    model.damping = (0.9 + model.damping) / 2.0
                    if self.log: print('Increasing damping and continuing', model.damping)
                    alpha *= 0.5
            prev_l = l

        # run some extra iterations with no gradient update to make sure things are primal feasible
        for _ in range(1000):
            if model.primal_feasibility(mu) < 1.0:
                break
            mu = model.belief_propagation(theta)
            if callback is not None:
                callback(mu)
        return l, theta, mu

    def mirror_descent(self, measurements, total=None, initial_alpha=10.0, callback=None):
        """ Use the mirror descent algorithm to estimate the GraphicalModel
            See https://web.iem.technion.ac.il/images/user-files/becka/papers/3.pdf

        :param measurements: a list of (Q, y, noise, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the noisy answers to the measurement queries
            noise is the standard deviation of the noise added to y
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param stepsize: the learning rate function
        :param callback: a function to be called after each iteration of optimization
        """
        self._setup(measurements, total)
        l, theta, mu = self.mirror_descent_auto(alpha=initial_alpha, iters=self.iters, callback=callback)

        self.model.potentials = theta
        self.model.marginals = mu

        return l

    def _marginal_loss(self, marginals, metric=None):
        """ Compute the loss and gradient for a given dictionary of marginals

        :param marginals: A dictionary with keys as projections and values as Factors
        :return loss: the loss value
        :return grad: A dictionary with gradient for each marginal
        """
        if metric is None:
            metric = self.metric

        if callable(metric):
            return metric(marginals)

        loss = 0.0
        gradient = {}

        for cl in marginals:
            mu = marginals[cl]
            gradient[cl] = self.Factor.zeros(mu.domain)
            for Q, y, noise, proj in self.groups[cl]:
                c = 1.0 / noise
                mu2 = mu.project(proj)
                x = mu2.datavector()
                diff = c * (Q @ x - y)
                if metric == 'L1':
                    loss += abs(diff).sum()
                    sign = diff.sign() if hasattr(diff, 'sign') else np.sign(diff)
                    grad = c * (Q.T @ sign)
                else:
                    loss += 0.5 * (diff @ diff)
                    grad = c * (Q.T @ diff)
                gradient[cl] += self.Factor(mu2.domain, grad)
        return float(loss), CliqueVector(gradient)

    def _setup(self, measurements, total):
        """ Perform necessary setup for running estimation algorithms

        1. If total is None, find the minimum variance unbiased estimate for total and use that
        2. Construct the GraphicalModel
            * If there are structural_zeros in the distribution, initialize factors appropriately
        3. Pre-process measurements into groups so that _marginal_loss may be evaluated efficiently
        """
        if total is None:
            # find the minimum variance estimate of the total given the measurements
            variances = np.array([])
            estimates = np.array([])
            for Q, y, noise, proj in measurements:
                o = np.ones(Q.shape[1])
                v = lsmr(Q.T, o, atol=0, btol=0)[0]
                if np.allclose(Q.T.dot(v), o):
                    variances = np.append(variances, noise ** 2 * np.dot(v, v))
                    estimates = np.append(estimates, np.dot(v, y))
            if estimates.size == 0:
                total = 1
            else:
                variance = 1.0 / np.sum(1.0 / variances)
                estimate = variance * np.sum(estimates / variances)
                total = max(1, estimate)

        # if not self.warm_start or not hasattr(self, 'model'):
        # initialize the model and parameters
        cliques = [m[3] for m in measurements]
        if self.structural_zeros is not None:
            cliques += list(self.structural_zeros.keys())
        if self.marginal_oracle == 'approx':
            model = RegionGraph(self.domain, cliques, total, convex=False, iters=self.inner_iters)
        elif self.marginal_oracle == 'convex':
            model = RegionGraph(self.domain, cliques, total, convex=True, iters=self.inner_iters)
        elif self.marginal_oracle == 'pairwise':
            model = FactorGraph(self.domain, cliques, total, convex=False, iters=self.inner_iters)
        elif self.marginal_oracle == 'pairwise-convex':
            model = FactorGraph(self.domain, cliques, total, convex=True, iters=self.inner_iters)
        else:
            model = self.marginal_oracle
            model.total = total

        if type(self.marginal_oracle) is str:
            model.potentials = CliqueVector.zeros(self.domain, model.cliques)
            model.potentials.combine(self.structural_zeros)
            if self.warm_start and hasattr(self, 'model'):
                model.potentials.combine(self.model.potentials)
        self.model = model

        # group the measurements into model cliques
        cliques = self.model.cliques
        # self.groups = { cl : [] for cl in cliques }
        self.groups = defaultdict(lambda: [])
        for Q, y, noise, proj in measurements:
            if self.backend == 'torch':
                import torch
                device = self.Factor.device
                y = torch.tensor(y, dtype=torch.float32, device=device)
                if isinstance(Q, np.ndarray):
                    Q = torch.tensor(Q, dtype=torch.float32, device=device)
                elif sparse.issparse(Q):
                    Q = Q.tocoo()
                    idx = torch.LongTensor([Q.row, Q.col])
                    vals = torch.FloatTensor(Q.data)
                    Q = torch.sparse.FloatTensor(idx, vals).to(device)

                # else Q is a Linear Operator, must be compatible with torch
            m = (Q, y, noise, proj)
            for cl in sorted(cliques, key=model.domain.size):
                # (Q, y, noise, proj) tuple
                if set(proj) <= set(cl):
                    self.groups[cl].append(m)
                    break


def run(dataset, measurements, eps=1.0, delta=0.0, bounded=True, engine='MD',
        options={}, iters=10000, seed=None, metric='L2', elim_order=None, frequency=1, workload=None, oracle='exact'):
    """
    Run a mechanism that measures the given measurements and runs inference.
    This is a convenience method for running end-to-end experiments.
    """

    domain = dataset.domain
    total = None

    state = np.random.RandomState(seed)

    if len(measurements) >= 1 and type(measurements[0][0]) is str:
        matrix = lambda proj: sparse.eye(domain.project(proj).size())
        measurements = [(proj, matrix(proj)) for proj in measurements]

    l1 = 0
    l2 = 0
    for _, Q in measurements:
        l1 += np.abs(Q).sum(axis=0).max()
        try:
            l2 += Q.power(2).sum(axis=0).max()  # for spares matrices
        except:
            l2 += np.square(Q).sum(axis=0).max()  # for dense matrices

    if bounded:
        total = dataset.df.shape[0]
        l1 *= 2
        l2 *= 2

    if delta > 0:
        noise = norm(loc=0, scale=np.sqrt(l2 * 2 * np.log(2 / delta)) / eps)
    else:
        noise = laplace(loc=0, scale=l1 / eps)

    if workload is None:
        workload = measurements

    truth = []
    for proj, W, in workload:
        x = dataset.project(proj).datavector()
        y = W.dot(x)
        truth.append((W, y, proj))

    answers = []
    for proj, Q in measurements:
        x = dataset.project(proj).datavector()
        z = noise.rvs(size=Q.shape[0], random_state=state)
        y = Q.dot(x)
        answers.append((Q, y + z, 1.0, proj))

    if oracle == 'exact':
        estimator = FactoredInference(domain, metric=metric, iters=iters, warm_start=False, elim_order=elim_order)
    else:
        estimator = LocalInference(domain, metric=metric, iters=iters, warm_start=False, marginal_oracle=oracle)
    logger = Logger(estimator, true_answers=truth, frequency=frequency)
    model = estimator.estimate(answers, total, engine=engine, callback=logger, options=options)

    return model, logger, answers

class Factor_t:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, domain, values):
        """ Initialize a factor over the given domain

        :param domain: the domain of the factor
        :param values: the ndarray or tensor of factor values (for each element of the domain)

        Note: values may be a flattened 1d array or a ndarray with same shape as domain
        """
        if type(values) == np.ndarray:
            values = torch.tensor(values, dtype=torch.float32, device=Factor_t.device)
        assert domain.size() == values.nelement(), 'domain size does not match values size'
        assert len(values.shape) == 1 or values.shape == domain.shape, 'invalid shape for values array'
        self.domain = domain
        self.values = values.reshape(domain.shape).to(Factor_t.device)

    @staticmethod
    def zeros(domain):
        return Factor_t(domain, torch.zeros(domain.shape, device=Factor_t.device))

    @staticmethod
    def ones(domain):
        return Factor_t(domain, torch.ones(domain.shape, device=Factor_t.device))

    @staticmethod
    def random(domain):
        return Factor_t(domain, torch.rand(domain.shape, device=Factor_t.device))

    @staticmethod
    def uniform(domain):
        return Factor_t.ones(domain) / domain.size()

    @staticmethod
    def active(domain, structural_zeros):
        """ create a factor that is 0 everywhere except in positions present in
            'structural_zeros', where it is -infinity

        :param: domain: the domain of this factor
        :param: structural_zeros: a list of values that are not possible
        """
        idx = tuple(np.array(structural_zeros).T)
        vals = torch.zeros(domain.shape, device=Factor_t.device)
        vals[idx] = -np.inf
        return Factor_t(domain, vals)

    def expand(self, domain):
        assert domain.contains(self.domain), 'expanded domain must contain current domain'
        dims = len(domain) - len(self.domain)
        values = self.values.view(self.values.size() + tuple([1] * dims))
        ax = domain.axes(self.domain.attrs)
        # need to find replacement for moveaxis
        ax = ax + tuple(i for i in range(len(domain)) if not i in ax)
        ax = tuple(np.argsort(ax))
        values = values.permute(ax)
        values = values.expand(domain.shape)
        return Factor_t(domain, values)

    def transpose(self, attrs):
        assert set(attrs) == set(self.domain.attrs), 'attrs must be same as domain attributes'
        newdom = self.domain.project(attrs)
        ax = newdom.axes(self.domain.attrs)
        ax = tuple(np.argsort(ax))
        values = self.values.permute(ax)
        return Factor_t(newdom, values)

    def project(self, attrs, agg='sum'):
        """
        project the factor onto a list of attributes (in order)
        using either sum or logsumexp to aggregate along other attributes
        """
        assert agg in ['sum', 'logsumexp'], 'agg must be sum or logsumexp'
        marginalized = self.domain.marginalize(attrs)
        if agg == 'sum':
            ans = self.sum(marginalized.attrs)
        elif agg == 'logsumexp':
            ans = self.logsumexp(marginalized.attrs)
        return ans.transpose(attrs)

    def sum(self, attrs=None):
        if attrs is None:
            return float(self.values.sum())
        elif attrs == tuple():
            return self
        axes = self.domain.axes(attrs)
        values = self.values.sum(dim=axes)
        newdom = self.domain.marginalize(attrs)
        return Factor_t(newdom, values)

    def logsumexp(self, attrs=None):
        if attrs is None:
            return float(self.values.logsumexp(dim=tuple(range(len(self.values.shape)))))
        elif attrs == tuple():
            return self
        axes = self.domain.axes(attrs)
        values = self.values.logsumexp(dim=axes)
        newdom = self.domain.marginalize(attrs)
        return Factor_t(newdom, values)

    def logaddexp(self, other):
        return NotImplementedError

    def max(self, attrs=None):
        if attrs is None:
            return float(self.values.max())
        return NotImplementedError  # torch.max does not behave like numpy

    def condition(self, evidence):
        """ evidence is a dictionary where
                keys are attributes, and
                values are elements of the domain for that attribute """
        slices = [evidence[a] if a in evidence else slice(None) for a in self.domain]
        newdom = self.domain.marginalize(evidence.keys())
        values = self.values[tuple(slices)]
        return Factor_t(newdom, values)

    def copy(self, out=None):
        if out is None:
            return Factor_t(self.domain, self.values.clone())
        np.copyto(out.values, self.values)
        return out

    def __mul__(self, other):
        if np.isscalar(other):
            return Factor_t(self.domain, other * self.values)
        # print(self.values.max(), other.values.max(), self.domain, other.domain)
        newdom = self.domain.merge(other.domain)
        factor1 = self.expand(newdom)
        factor2 = other.expand(newdom)
        return Factor_t(newdom, factor1.values * factor2.values)

    def __add__(self, other):
        if np.isscalar(other):
            return Factor_t(self.domain, other + self.values)
        newdom = self.domain.merge(other.domain)
        factor1 = self.expand(newdom)
        factor2 = other.expand(newdom)
        return Factor_t(newdom, factor1.values + factor2.values)

    def __iadd__(self, other):
        if np.isscalar(other):
            self.values += other
            return self
        factor2 = other.expand(self.domain)
        self.values += factor2.values
        return self

    def __imul__(self, other):
        if np.isscalar(other):
            self.values *= other
            return self
        factor2 = other.expand(self.domain)
        self.values *= factor2.values
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if np.isscalar(other):
            return Factor_t(self.domain, self.values - other)
        zero = torch.tensor(0.0, device=Factor.device)
        inf = torch.tensor(np.inf, device=Factor.device)
        values = torch.where(other.values == -inf, zero, -other.values)
        other = Factor_t(other.domain, values)
        return self + other

    def __truediv__(self, other):
        # assert np.isscalar(other), 'divisor must be a scalar'
        if np.isscalar(other):
            return self * (1.0 / other)
        tmp = other.expand(self.domain)
        vals = torch.div(self.values, tmp.values)
        vals[tmp.values <= 0] = 0.0
        return Factor_t(self.domain, vals)

    def exp(self, out=None):
        if out is None:
            return Factor_t(self.domain, self.values.exp())
        torch.exp(self.values, out=out.values)
        return out

    def log(self, out=None):
        if out is None:
            return Factor_t(self.domain, torch.log(self.values + 1e-100))
        torch.log(self.values, out=out.values)
        return out

    def datavector(self, flatten=True):
        """ Materialize the data vector as a numpy array """
        ans = self.values.to("cpu").numpy()
        return ans.flatten() if flatten else ans


class RegionGraph():
    def __init__(self, domain, cliques, total=1.0, minimal=True, convex=True, iters=25, convergence=1e-3, damping=0.5):
        self.domain = domain
        self.cliques = cliques
        if not convex:
            self.cliques = []
            for r in cliques:
                if not any(set(r) < set(s) for s in cliques):
                    self.cliques.append(r)
        self.total = total
        self.minimal = minimal
        self.convex = convex
        self.iters = iters
        self.convergence = convergence
        self.damping = damping
        if convex:
            self.belief_propagation = self.hazan_peng_shashua
        else:
            self.belief_propagation = self.generalized_belief_propagation
        self.build_graph()
        self.cliques = sorted(self.regions, key=len)
        self.potentials = CliqueVector.zeros(domain, self.cliques)
        self.marginals = CliqueVector.uniform(domain, self.cliques) * total

    def show(self):
        import matplotlib.pyplot as plt
        labels = {r: ''.join(r) for r in self.regions}

        pos = {}
        xloc = defaultdict(lambda: 0)
        for r in sorted(self.regions):
            y = len(r)
            pos[r] = (xloc[y] + 0.5 * (y % 2), y)
            xloc[y] += 1

        colormap = {r: 'red' if r in self.cliques else 'blue' for r in self.regions}

        nx.draw(self.G, pos=pos, node_color='orange', node_size=1000)
        nx.draw(self.G, pos=pos, nodelist=self.cliques, node_color='green', node_size=1000)
        nx.draw_networkx_labels(self.G, pos=pos, labels=labels)
        plt.show()

    def project(self, attrs, maxiter=100, alpha=None):
        if type(attrs) is list:
            attrs = tuple(attrs)

        for cl in self.cliques:
            if set(attrs) <= set(cl):
                return self.marginals[cl].project(attrs)

        # Use multiplicative weights/entropic mirror descent to solve projection problem
        intersections = [set(cl) & set(attrs) for cl in self.cliques]
        target_cliques = [tuple(t) for t in intersections if not any(t < s for s in intersections)]
        target_cliques = list(set(target_cliques))
        target_mu = CliqueVector.from_data(self, target_cliques)

        if len(target_cliques) == 0:
            return Factor.uniform(self.domain.project(attrs)) * self.total
        # P = Factor.uniform(self.domain.project(attrs))*self.total
        # Use a smart initialization
        P = estimate_kikuchi_marginal(self.domain.project(attrs), self.total, target_mu)
        if alpha is None:
            # start with a safe step size
            alpha = 1.0 / (self.total * len(target_cliques))

        curr_mu = CliqueVector.from_data(P, target_cliques)
        diff = curr_mu - target_mu
        curr_loss, dL = diff.dot(diff), sum(diff.values()).expand(P.domain)
        begun = False

        for _ in range(maxiter):
            if curr_loss <= 1e-8:
                return P  # stop early if marginals are almost exactly realized
            Q = P * (-alpha * dL).exp()
            Q *= self.total / Q.sum()
            curr_mu = CliqueVector.from_data(Q, target_cliques)
            diff = curr_mu - target_mu
            loss = diff.dot(diff)
            # print(alpha, diff.dot(diff))

            if curr_loss - loss >= 0.5 * alpha * dL.dot(P - Q):
                P = Q
                curr_loss = loss
                dL = sum(diff.values()).expand(P.domain)
                # increase step size if we haven't already decreased it at least once
                if not begun: alpha *= 2
            else:
                alpha *= 0.5
                begun = True

        return P

    def primal_feasibility(self, mu):
        ans = 0
        count = 0
        for r in self.cliques:
            for s in self.children[r]:
                x = mu[r].project(s).datavector()
                y = mu[s].datavector()
                err = np.linalg.norm(x - y, 1)
                ans += err
                count += 1
        return 0 if count == 0 else ans / count

    def is_converged(self, mu):
        return self.primal_feasibility(mu) <= self.convergence

    def build_graph(self):
        # Alg 11.3 of Koller & Friedman
        regions = set(self.cliques)
        size = 0
        while len(regions) > size:
            size = len(regions)
            for r1, r2 in itertools.combinations(regions, 2):
                z = tuple(sorted(set(r1) & set(r2)))
                if len(z) > 0 and not z in regions:
                    regions.update({z})

        G = nx.DiGraph()
        G.add_nodes_from(regions)
        for r1 in regions:
            for r2 in regions:
                if set(r2) < set(r1) and not \
                        any(set(r2) < set(r3) and set(r3) < set(r1) for r3 in regions):
                    G.add_edge(r1, r2)

        H = G.reverse()
        G1, H1 = nx.transitive_closure(G), nx.transitive_closure(H)

        self.children = {r: list(G.neighbors(r)) for r in regions}
        self.parents = {r: list(H.neighbors(r)) for r in regions}
        self.descendants = {r: list(G1.neighbors(r)) for r in regions}
        self.ancestors = {r: list(H1.neighbors(r)) for r in regions}
        self.forebears = {r: set([r] + self.ancestors[r]) for r in regions}
        self.downp = {r: set([r] + self.descendants[r]) for r in regions}

        if self.minimal:
            min_edges = []
            for r in regions:
                ds = DisjointSet()
                for u in self.parents[r]: ds.find(u)
                for u, v in itertools.combinations(self.parents[r], 2):
                    uv = set(self.ancestors[u]) & set(self.ancestors[v])
                    if len(uv) > 0: ds.union(u, v)
                canonical = set()
                for u in self.parents[r]:
                    canonical.update({ds.find(u)})
                # if len(canonical) > 1:# or r in self.cliques:
                min_edges.extend([(u, r) for u in canonical])
            # G = nx.DiGraph(min_edges)
            # regions = list(G.nodes)
            G = nx.DiGraph()
            G.add_nodes_from(regions)
            G.add_edges_from(min_edges)

            H = G.reverse()
            G1, H1 = nx.transitive_closure(G), nx.transitive_closure(H)

            self.children = {r: list(G.neighbors(r)) for r in regions}
            self.parents = {r: list(H.neighbors(r)) for r in regions}
            # self.descendants = { r : list(G1.neighbors(r)) for r in regions }
            # self.ancestors = { r : list(H1.neighbors(r)) for r in regions }
            # self.forebears = { r : set([r] + self.ancestors[r]) for r in regions }
            # self.downp = { r : set([r] + self.descendants[r]) for r in regions }

        self.G = G
        self.regions = regions

        if self.convex:
            self.counting_numbers = {r: 1.0 for r in regions}
        else:
            moebius = {}

            def get_counting_number(r):
                if not r in moebius:
                    moebius[r] = 1 - sum(get_counting_number(s) for s in self.ancestors[r])
                return moebius[r]

            for r in regions: get_counting_number(r)
            self.counting_numbers = moebius

            if self.minimal:
                # https://people.eecs.berkeley.edu/~ananth/2002+/Payam/submittedkikuchi.pdf
                # Eq. 30 and 31
                N, D, B = {}, {}, {}
                for r in regions:
                    B[r] = set()
                    for p in self.parents[r]:
                        B[r].add((p, r))
                    for d in self.descendants[r]:
                        for p in set(self.parents[d]) - {r} - set(self.descendants[r]):
                            B[r].add((p, d))

                for p in self.regions:
                    for r in self.children[p]:
                        N[p, r], D[p, r] = set(), set()
                        for s in self.parents[p]:
                            N[p, r].add((s, p))
                        for d in self.descendants[p]:
                            for s in set(self.parents[d]) - {p} - set(self.descendants[p]):
                                N[p, r].add((s, d))
                        for s in set(self.parents[r]) - {p}:
                            D[p, r].add((s, r))
                        for d in self.descendants[r]:
                            for p1 in set(self.parents[d]) - {r} - set(self.descendants[r]):
                                D[p, r].add((p1, d))
                        cancel = N[p, r] & D[p, r]
                        N[p, r] = N[p, r] - cancel
                        D[p, r] = D[p, r] - cancel

                self.N, self.D, self.B = N, D, B

            else:
                # From Yedida et al. for fully saturated region graphs
                # for sending messages ru --> rd and computing beliefs B_r
                N, D, B = {}, {}, {}
                for r in regions:
                    B[r] = [(ru, r) for ru in self.parents[r]]
                    for rd in self.descendants[r]:
                        for ru in set(self.parents[rd]) - self.downp[r]:
                            B[r].append((ru, rd))

                for ru in regions:
                    for rd in self.children[ru]:
                        fu, fd = self.downp[ru], self.downp[rd]
                        cond = lambda r: not r[0] in fu and r[1] in (fu - fd)
                        N[ru, rd] = [e for e in G.edges if cond(e)]
                        cond = lambda r: r[0] in (fu - fd) and r[1] in fd and r != (ru, rd)
                        D[ru, rd] = [e for e in G.edges if cond(e)]

                self.N, self.D, self.B = N, D, B

        self.messages = {}
        self.message_order = []
        for ru in sorted(regions, key=len):  # nx.topological_sort(H): # should be G or H?
            for rd in self.children[ru]:
                self.message_order.append((ru, rd))
                self.messages[ru, rd] = Factor.zeros(self.domain.project(rd))
                self.messages[rd, ru] = Factor.zeros(self.domain.project(rd))  # only for hazan et al

    def generalized_belief_propagation(self, potentials, callback=None):
        # https://users.cs.duke.edu/~brd/Teaching/Bio/asmb/current/4paper/4-2.pdf
        pot = {}
        for r in self.regions:
            if r in self.cliques:
                pot[r] = potentials[r]
            else:
                pot[r] = Factor.zeros(self.domain.project(r))

        for _ in range(self.iters):
            new = {}
            for ru, rd in self.message_order:
                # Yedida et al. strongly recommend using updated messages for LHS (denom in our case)
                # num = sum(pot[c] for c in self.downp[ru] if c != rd)
                num = pot[ru]
                num = num + sum(self.messages[r1, r2] for r1, r2 in self.N[ru, rd])
                denom = sum(new[r1, r2] for r1, r2 in self.D[ru, rd])
                diff = tuple(set(ru) - set(rd))
                new[ru, rd] = num.logsumexp(diff) - denom
                new[ru, rd] -= new[ru, rd].logsumexp()

            # self.messages = new
            for ru, rd in self.message_order:
                self.messages[ru, rd] = 0.5 * self.messages[ru, rd] + 0.5 * new[ru, rd]
            # print(self.messages[ru,rd].datavector())
            # ru, rd = self.message_order[0]
            # print(ru, rd, self.messages[ru,rd].values)

        marginals = {}
        for r in self.cliques:
            # belief = sum(potentials[c] for c in self.downp[r]) + sum(self.messages[r1,r2] for r1,r2 in self.B[r])
            belief = potentials[r] + sum(self.messages[r1, r2] for r1, r2 in self.B[r])
            belief += np.log(self.total) - belief.logsumexp()
            marginals[r] = belief.exp()
        # print(marginals[('A','B')].datavector())

        return CliqueVector(marginals)

    def hazan_peng_shashua(self, potentials, callback=None):
        # https://arxiv.org/pdf/1210.4881.pdf
        c0 = self.counting_numbers
        pot = {}
        for r in self.regions:
            if r in self.cliques:
                pot[r] = potentials[r]
            else:
                pot[r] = Factor.zeros(self.domain.project(r))

        messages = self.messages
        # for p in sorted(self.regions, key=len): #nx.topological_sort(H): # should be G or H?
        #    for r in self.children[p]:
        #        messages[p,r] = Factor.zeros(self.domain.project(r))
        #        messages[r,p] = Factor.zeros(self.domain.project(r))

        cc = {}
        for r in self.regions:
            for p in self.parents[r]:
                cc[p, r] = c0[p] / (c0[r] + sum(c0[p1] for p1 in self.parents[r]))

        for _ in range(self.iters):
            new = {}
            for r in self.regions:
                for p in self.parents[r]:
                    new[p, r] = (pot[p] + sum(messages[c, p] for c in self.children[p] if c != r) - sum(
                        messages[p, p1] for p1 in self.parents[p])) / c0[p]
                    new[p, r] = c0[p] * new[p, r].logsumexp(tuple(set(p) - set(r)))
                    new[p, r] -= new[p, r].logsumexp()

            for r in self.regions:
                for p in self.parents[r]:
                    new[r, p] = cc[p, r] * (pot[r] + sum(messages[c, r] for c in self.children[r]) + sum(
                        messages[p1, r] for p1 in self.parents[r])) - messages[p, r]
                    # new[r,p] = cc[p,r]*(pot[r] + sum(messages[c,r] for c in self.children[r]) + sum(new[p1,r] for p1 in self.parents[r])) - new[p,r]
                    new[r, p] -= new[r, p].logsumexp()

            # messages = new
            # Damping is not described in paper, but is needed to get convergence for dense graphs
            rho = self.damping
            for p in self.regions:
                for r in self.children[p]:
                    messages[p, r] = rho * messages[p, r] + (1.0 - rho) * new[p, r]
                    messages[r, p] = rho * messages[r, p] + (1.0 - rho) * new[r, p]
            mu = {}
            for r in self.regions:
                belief = (pot[r] + sum(messages[c, r] for c in self.children[r]) - sum(
                    messages[r, p] for p in self.parents[r])) / c0[r]
                belief += np.log(self.total) - belief.logsumexp()
                mu[r] = belief.exp()

            if callback is not None:
                callback(mu)

            if self.is_converged(mu):
                self.messages = messages
                return CliqueVector(mu)

        self.messages = messages
        return CliqueVector(mu)

    def wiegerinck(self, potentials, callback=None):
        c = self.counting_numbers
        m = {}
        for delta in self.regions:
            m[delta] = 0
            for alpha in self.ancestors[delta]:
                m[delta] += c[alpha]

        Q = {}
        for r in self.regions:
            if r in self.cliques:
                Q[r] = potentials[r] / c[r]
            else:
                Q[r] = Factor.zeros(self.domain.project(r))

        inner = [r for r in self.regions if len(self.parents[r]) > 0]
        diff = lambda r, s: tuple(set(r) - set(s))
        for _ in range(self.iters):
            for r in inner:
                A = c[r] / (m[r] + c[r])
                B = m[r] / (m[r] + c[r])
                Qbar = sum(c[s] * Q[s].logsumexp(diff(s, r)) for s in self.ancestors[r]) / m[r]
                Q[r] = Q[r] * A + Qbar * B
                Q[r] -= Q[r].logsumexp()
                for s in self.ancestors[r]:
                    Q[s] = Q[s] + Q[r] - Q[s].logsumexp(diff(s, r))
                    Q[s] -= Q[s].logsumexp()

            marginals = {}
            for r in self.regions:
                marginals[r] = (Q[r] + np.log(self.total) - Q[r].logsumexp()).exp()
            if callback is not None:
                callback(marginals)

        return CliqueVector(marginals)

    def loh_wibisono(self, potentials, callback=None):
        # https://papers.nips.cc/paper/2014/file/39027dfad5138c9ca0c474d71db915c3-Paper.pdf
        pot = {}
        for r in self.regions:
            if r in self.cliques:
                pot[r] = potentials[r]
            else:
                pot[r] = Factor.zeros(self.domain.project(r))

        rho = self.counting_numbers

        for _ in range(self.iters):
            new = {}
            for s, r in self.message_order:
                diff = tuple(set(s) - set(r))
                num = pot[s] / rho[s]
                for v in self.parents[s]:
                    num += self.messages[v, s] * rho[v] / rho[s]
                for w in self.children[s]:
                    if w != r:
                        num -= self.messages[s, w]
                num = num.logsumexp(diff)
                denom = pot[r] / rho[r]
                for u in self.parents[r]:
                    if u != s:
                        denom += self.messages[u, r] * rho[u] / rho[r]
                for t in self.children[r]:
                    denom -= self.messages[r, t]

                new[s, r] = rho[r] / (rho[r] + rho[s]) * (num - denom)
                new[s, r] -= new[s, r].logsumexp()

            for ru, rd in self.message_order:
                self.messages[ru, rd] = 0.5 * self.messages[ru, rd] + 0.5 * new[ru, rd]

            # ru, rd = self.message_order[0]
            # print(ru, rd, self.messages[ru,rd].values)

            marginals = {}
            for r in self.regions:
                belief = pot[r] / rho[r]
                for s in self.parents[r]:
                    belief += self.messages[s, r] * rho[s] / rho[r]
                for t in self.children[r]:
                    belief -= self.messages[r, t]
                belief += np.log(self.total) - belief.logsumexp()
                marginals[r] = belief.exp()
            # print(marginals[('A','B')].datavector())
            if callback is not None:
                callback(marginals)

        return CliqueVector(marginals)

    def kikuchi_entropy(self, marginals):
        """
        Return the Bethe Entropy and the gradient with respect to the marginals

        """
        weights = self.counting_numbers
        entropy = 0
        dmarginals = {}
        for cl in self.regions:
            mu = marginals[cl] / self.total
            entropy += weights[cl] * (mu * mu.log()).sum()
            dmarginals[cl] = weights[cl] * (1 + mu.log()) / self.total
        return -entropy, -1 * CliqueVector(dmarginals)

    def mle(self, mu):
        return -1 * self.kikuchi_entropy(mu)[1]


def estimate_kikuchi_marginal(domain, total, marginals):
    marginals = dict(marginals)
    regions = set(marginals.keys())
    size = 0
    while len(regions) > size:
        size = len(regions)
        for r1, r2 in itertools.combinations(regions, 2):
            z = tuple(sorted(set(r1) & set(r2)))
            if len(z) > 0 and not z in regions:
                marginals[z] = marginals[r1].project(z)
                regions.update({z})

    G = nx.DiGraph()
    G.add_nodes_from(regions)
    for r1 in regions:
        for r2 in regions:
            if set(r2) < set(r1) and not \
                    any(set(r2) < set(r3) and set(r3) < set(r1) for r3 in regions):
                G.add_edge(r1, r2)

    H1 = nx.transitive_closure(G.reverse())
    ancestors = {r: list(H1.neighbors(r)) for r in regions}
    moebius = {}

    def get_counting_number(r):
        if not r in moebius:
            moebius[r] = 1 - sum(get_counting_number(s) for s in ancestors[r])
        return moebius[r]

    logP = Factor.zeros(domain)
    for r in regions:
        kr = get_counting_number(r)
        logP += kr * marginals[r].log()
    logP += np.log(total) - logP.logsumexp()
    return logP.exp()


def entropic_mirror_descent(loss_and_grad, x0, total, iters=250):
    logP = np.log(x0 + np.nextafter(0, 1)) + np.log(total) - np.log(x0.sum())
    P = np.exp(logP)
    P = x0 * total / x0.sum()
    loss, dL = loss_and_grad(P)
    alpha = 1.0
    begun = False

    for _ in range(iters):
        logQ = logP - alpha * dL
        logQ += np.log(total) - logsumexp(logQ)
        Q = np.exp(logQ)
        # Q = P * np.exp(-alpha*dL)
        # Q *= total / Q.sum()
        new_loss, new_dL = loss_and_grad(Q)

        if loss - new_loss >= 0.5 * alpha * dL.dot(P - Q):
            # print(alpha, loss)
            logP = logQ
            loss, dL = new_loss, new_dL
            # increase step size if we haven't already decreased it at least once
            if not begun: alpha *= 2
        else:
            alpha *= 0.5
            begun = True

    return np.exp(logP)


def estimate_total(measurements):
    # find the minimum variance estimate of the total given the measurements
    variances = np.array([])
    estimates = np.array([])
    for Q, y, noise, proj in measurements:
        o = np.ones(Q.shape[1])
        v = lsmr(Q.T, o, atol=0, btol=0)[0]
        if np.allclose(Q.T.dot(v), o):
            variances = np.append(variances, noise ** 2 * np.dot(v, v))
            estimates = np.append(estimates, np.dot(v, y))
    if estimates.size == 0:
        return 1
    else:
        variance = 1.0 / np.sum(1.0 / variances)
        estimate = variance * np.sum(estimates / variances)
        return max(1, estimate)


class PublicInference:
    def __init__(self, public_data, metric='L2'):
        self.public_data = public_data
        self.metric = metric
        self.weights = np.ones(self.public_data.records)

    def estimate(self, measurements, total=None):
        if total is None:
            total = estimate_total(measurements)
        self.measurements = measurements
        cliques = [M[-1] for M in measurements]

        def loss_and_grad(weights):
            est = Dataset(self.public_data.df, self.public_data.domain, weights)
            mu = CliqueVector.from_data(est, cliques)
            loss, dL = self._marginal_loss(mu)
            dweights = np.zeros(weights.size)
            for cl in dL:
                idx = est.project(cl).df.values
                dweights += dL[cl].values[tuple(idx.T)]
            return loss, dweights

            # bounds = [(0,None) for _ in self.weights]

        # res = minimize(loss_and_grad, x0=self.weights, method='L-BFGS-B', jac=True, bounds=bounds)
        # self.weights = res.x

        self.weights = entropic_mirror_descent(loss_and_grad, self.weights, total)
        return Dataset(self.public_data.df, self.public_data.domain, self.weights)

    def _marginal_loss(self, marginals, metric=None):
        """ Compute the loss and gradient for a given dictionary of marginals

        :param marginals: A dictionary with keys as projections and values as Factors
        :return loss: the loss value
        :return grad: A dictionary with gradient for each marginal
        """
        if metric is None:
            metric = self.metric

        if callable(metric):
            return metric(marginals)

        loss = 0.0
        gradient = {cl: Factor.zeros(marginals[cl].domain) for cl in marginals}

        for Q, y, noise, cl in self.measurements:
            mu = marginals[cl]
            c = 1.0 / noise
            x = mu.datavector()
            diff = c * (Q @ x - y)
            if metric == 'L1':
                loss += abs(diff).sum()
                sign = diff.sign() if hasattr(diff, 'sign') else np.sign(diff)
                grad = c * (Q.T @ sign)
            else:
                loss += 0.5 * (diff @ diff)
                grad = c * (Q.T @ diff)
            gradient[cl] += Factor(mu.domain, grad)
        return float(loss), CliqueVector(gradient)


def estimate_total(measurements):
    # find the minimum variance estimate of the total given the measurements
    variances = np.array([])
    estimates = np.array([])
    for Q, y, noise, proj in measurements:
        o = np.ones(Q.shape[1])
        v = lsmr(Q.T, o, atol=0, btol=0)[0]
        if np.allclose(Q.T.dot(v), o):
            variances = np.append(variances, noise ** 2 * np.dot(v, v))
            estimates = np.append(estimates, np.dot(v, y))
    if estimates.size == 0:
        return 1
    else:
        variance = 1.0 / np.sum(1.0 / variances)
        estimate = variance * np.sum(estimates / variances)
        return max(1, estimate)


def adam(loss_and_grad, x0, iters=250):
    a = 1.0
    b1, b2 = 0.9, 0.999
    eps = 10e-8

    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for t in range(1, iters + 1):
        l, g = loss_and_grad(x)
        # print(l)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        x = x - a * mhat / (np.sqrt(vhat) + eps)
    #        print np.linalg.norm(A.dot(x) - y, ord=2)
    return x


def synthetic_col(counts, total):
    counts *= total / counts.sum()
    frac, integ = np.modf(counts)
    integ = integ.astype(int)
    extra = total - integ.sum()
    if extra > 0:
        idx = np.random.choice(counts.size, extra, False, frac / frac.sum())
        integ[idx] += 1
    vals = np.repeat(np.arange(counts.size), integ)
    np.random.shuffle(vals)
    return vals


class MixtureOfProducts:
    def __init__(self, products, domain, total):
        self.products = products
        self.domain = domain
        self.total = total
        self.num_components = next(iter(products.values())).shape[0]

    def project(self, cols):
        products = {col: self.products[col] for col in cols}
        domain = self.domain.project(cols)
        return MixtureOfProducts(products, domain, self.total)

    def datavector(self, flatten=True):
        letters = 'bcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(self.domain)]
        formula = ','.join(['a%s' % l for l in letters]) + '->' + ''.join(letters)
        components = [self.products[col] for col in self.domain]
        ans = np.einsum(formula, *components) * self.total / self.num_components
        return ans.flatten() if flatten else ans

    def synthetic_data(self, rows=None):
        total = rows or int(self.total)
        subtotal = total // self.num_components + 1

        dfs = []
        for i in range(self.num_components):
            df = pd.DataFrame()
            for col in self.products:
                counts = self.products[col][i]
                df[col] = synthetic_col(counts, subtotal)
            dfs.append(df)

        df = pd.concat(dfs).sample(frac=1).reset_index(drop=True)[:total]
        return Dataset(df, self.domain)


class MixtureInference:
    def __init__(self, domain, components=10, metric='L2', iters=2500, warm_start=False):
        """
        :param domain: A Domain object
        :param components: The number of mixture components
        :metric: The metric to use for the loss function (can be callable)
        """
        self.domain = domain
        self.components = components
        self.metric = metric
        self.iters = iters
        self.warm_start = warm_start
        self.params = np.random.normal(loc=0, scale=0.25, size=sum(domain.shape) * components)

    def estimate(self, measurements, total=None, alpha=0.1):
        if total == None:
            total = estimate_total(measurements)
        self.measurements = measurements
        cliques = [M[-1] for M in measurements]
        letters = 'bcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

        def get_products(params):
            products = {}
            idx = 0
            for col in self.domain:
                n = self.domain[col]
                k = self.components
                products[col] = jax_softmax(params[idx:idx + k * n].reshape(k, n), axis=1)
                idx += k * n
            return products

        def marginals_from_params(params):
            products = get_products(params)
            mu = {}
            for cl in cliques:
                let = letters[:len(cl)]
                formula = ','.join(['a%s' % l for l in let]) + '->' + ''.join(let)
                components = [products[col] for col in cl]
                ans = jnp.einsum(formula, *components) * total / self.components
                mu[cl] = ans.flatten()
            return mu

        def loss_and_grad(params):
            # For computing dL / dmu we will use ordinary numpy so as to support scipy sparse and linear operator inputs
            # For computing dL / dparams we will use jax to avoid manually deriving gradients
            params = jnp.array(params)
            mu, backprop = vjp(marginals_from_params, params)
            mu = {cl: np.array(mu[cl]) for cl in cliques}
            loss, dL = self._marginal_loss(mu)
            dL = {cl: jnp.array(dL[cl]) for cl in cliques}
            dparams = backprop(dL)
            return loss, np.array(dparams[0])

        if not self.warm_start:
            self.params = np.random.normal(loc=0, scale=0.25, size=sum(self.domain.shape) * self.components)
        self.params = adam(loss_and_grad, self.params, iters=self.iters)
        products = get_products(self.params)
        return MixtureOfProducts(products, self.domain, total)

    def _marginal_loss(self, marginals, metric=None):
        """ Compute the loss and gradient for a given dictionary of marginals

        :param marginals: A dictionary with keys as projections and values as Factors
        :return loss: the loss value
        :return grad: A dictionary with gradient for each marginal
        """
        if metric is None:
            metric = self.metric

        loss = 0.0
        gradient = {cl: np.zeros_like(marginals[cl]) for cl in marginals}

        for Q, y, noise, cl in self.measurements:
            x = marginals[cl]
            c = 1.0 / noise
            diff = c * (Q @ x - y)
            if metric == 'L1':
                loss += abs(diff).sum()
                sign = diff.sign() if hasattr(diff, 'sign') else np.sign(diff)
                grad = c * (Q.T @ sign)
            else:
                loss += 0.5 * (diff @ diff)
                grad = c * (Q.T @ diff)
            gradient[cl] += grad

        return float(loss), gradient