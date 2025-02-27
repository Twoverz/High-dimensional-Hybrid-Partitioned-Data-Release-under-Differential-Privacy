import math

import numpy as np
import itertools
from src.mbi import Dataset, Domain, estimation, junction_tree, LinearMeasurement, LinearMeasurement
from mechanism import Mechanism
from collections import defaultdict
from scipy.optimize import bisect
import pandas as pd
from src.mbi import Factor
from scipy.linalg import hadamard
import argparse
import random
import ast
import torch
import os
import json


def powerset(iterable):
  "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
  s = list(iterable)
  return itertools.chain.from_iterable(
    itertools.combinations(s, r) for r in range(1, len(s) + 1)
  )


def downward_closure(Ws):
  ans = set()
  for proj in Ws:
    ans.update(powerset(proj))
  return list(sorted(ans, key=len))


def hypothetical_model_size(domain, cliques):
  jtree, _ = junction_tree.make_junction_tree(domain, cliques)
  maximal_cliques = junction_tree.maximal_cliques(jtree)
  cells = sum(domain.size(cl) for cl in maximal_cliques)
  size_mb = cells * 8 / 2**20
  return size_mb


def compile_workload(workload):
  weights = {cl: wt for (cl, wt) in workload}
  workload_cliques = weights.keys()

  def score(cl):
    return sum(
      weights[workload_cl] * len(set(cl) & set(workload_cl))
      for workload_cl in workload_cliques
    )

  return {cl: score(cl) for cl in downward_closure(workload_cliques)}


def filter_candidates(candidates, model, size_limit):
  ans = {}
  free_cliques = downward_closure(model.cliques)
  for cl in candidates:
    cond1 = (
      hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
    )
    cond2 = cl in free_cliques
    if cond1 or cond2:
      ans[cl] = candidates[cl]
  return ans

"gausse noisy to random response mechan"


def matrix_mechanism(data, cl):
    clsize = [data.domain.size(cl[i]) for i in range(len(cl))]
    W = construct_matrix(clsize)
    # 计算 W^T W
    WW = np.dot(W.T, W)
    # 进行特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(WW)
    # 构建 Q 和 D
    D = np.diag(eigenvalues)  # 特征值构成的对角矩阵
    Q = eigenvectors  # 特征向量构成的正交矩阵

    lambdas = np.diag(D)  # 直接提取 D 的对角元素

    # Step 3: 构造 A' = ΛQ
    Lambda = np.diag(lambdas)  # 构建 Λ 对角矩阵
    A_prime = np.dot(Lambda, Q)  # 矩阵乘法

    # print("Matrix A' (ΛQ):")
    # print(A_prime)

    # Step 4: 计算每一列的 L2 范数 m_{ii}
    m = np.linalg.norm(A_prime, axis=0)  # 计算每一列的 L2 范数
    # print(m)

    # 构建 D' 的对角元素
    D_prime_diag = []
    for u in m:
        a = max([np.sqrt(abs(m_i ** 2 - u ** 2)) for m_i in m])
        D_prime_diag.append(a)

    # D_prime_diag = [max(np.sqrt(m_i**2 - m[0]**2)) for m_i in m]
    D_prime = np.diag(D_prime_diag)

    # print("Matrix D' (diag elements):")
    # print(D_prime)

    A = np.vstack((A_prime, D_prime))
    A_add = np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)
    noise = np.dot(W, A_add)

    return 0
def modify_vector(binary_vector, x):
    # 获取当前二值向量中 0 和 1 的索引
    binary_vector = np.array(binary_vector)
    zero_indices = np.where(binary_vector == 0)[0]  # 0的位置索引
    one_indices = np.where(binary_vector == 1)[0]  # 1的位置索引

    # 如果 x > 0，选择 x 个 0 转为 1
    if x > 0:
        if len(one_indices) + x >= len(binary_vector):
            binary_vector[zero_indices] = 1
        else:
            # 随机选择 x 个索引位置，将其 0 改为 1
            if len(zero_indices) == 0:
                indices_to_change = np.random.choice(zero_indices, size=0, replace=False)
                binary_vector = binary_vector
            else:
                indices_to_change = np.random.choice(zero_indices, size=int(x), replace=False)
                binary_vector[indices_to_change] = 1

    # 如果 x < 0，选择 |x| 个 1 转为 0
    elif x < 0:
        if len(zero_indices) + abs(x) >= len(binary_vector):
            binary_vector[one_indices] = 0
        else:
            # 随机选择 |x| 个索引位置，将其 1 改为 0
            if len(one_indices) == 0:
                indices_to_change = np.random.choice(one_indices, size=0, replace=False)
                binary_vector = binary_vector
            else:
                indices_to_change = np.random.choice(one_indices, size=int(-x), replace=False)
                binary_vector[indices_to_change] = 0

    return binary_vector.tolist()

def construct_matrix(clsize):
    """
    根据任意长度的 cl 列表动态构造矩阵。

    参数：
    - cl: 包含多个整数的列表，每个整数表示每列的范围。

    返回：
    - 矩阵：构造后的矩阵，行数为所有可能组合的数量，列数为 len(cl)。
    """
    # 使用 itertools.product 生成所有组合
    combinations = list(itertools.product(*[range(c) for c in clsize]))

    # 转换为矩阵
    matrix = np.array(combinations, dtype=int)
    return matrix

class AIM_vmulti(Mechanism):
  def __init__(
    self,
    epsilon,
    delta,
    prng=None,
    rounds=224,
    max_model_size=80,
    max_iters=1000,
    structural_zeros={},
  ):
    super(AIM_vmulti, self).__init__(epsilon, delta, prng)
    self.rounds = rounds
    self.max_iters = max_iters
    self.max_model_size = max_model_size
    self.structural_zeros = structural_zeros

  def worst_approximated(self, candidates, answers, model, eps, sigma, weight_sum):
    errors = {}
    sensitivity = {}
    for cl in candidates:
      wgt = candidates[cl]
      x = answers[cl]
      bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)

      rho1 = (len(cl) ** (2 / 3)) / weight_sum * (0.2 * self.rho)
      sigma_r = np.sqrt(1 / (2 * rho1))
      bias_r = np.sqrt(2 / np.pi) * sigma_r * model.domain.size(cl)

      xest = model.project(cl).datavector()
      errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias - bias_r)#
      sensitivity[cl] = abs(wgt)

    max_sensitivity = max(
      sensitivity.values()
    )  # if all weights are 0, could be a problem
    return self.exponential_mechanism(errors, eps, max_sensitivity)

  def noise_ans_ve(self, multi_ve_data, cl, weight_sum):
            "slove ve paritational data"
            print('current cl', cl)
            rho = (len(cl) ** (2 / 3)) / weight_sum * (0.2 * self.rho)
            # rho = weight_sum * (0.2 * self.rho)
            sigma = np.sqrt(1 / (2 * rho))

            answlist = []
            size_of_attr = []
            for sdate in multi_ve_data:
                print('current data domain', sdate.domain.attrs)
                common_tuples = [tuple for tuple in cl if tuple in sdate.domain.attrs]
                print('common_tuples', common_tuples)

                if len(common_tuples) == 0:
                    continue
                elif len(common_tuples) == len(cl):
                    print('current data domain including all cl')
                    res = sdate.project(common_tuples).datavector()
                    noise = [self.gaussian_noise(sigma, res.size) for _ in range(8)]
                    average_noise = np.mean(noise, axis=0)
                    res = res + average_noise
                    # res = res + self.gaussian_noise(sigma, res.size)
                    break
                elif len(common_tuples) < len(cl) and len(common_tuples) != 0:
                    onehot = sdate.project(common_tuples).dataonehot()
                    answlist.append(onehot)
                    size_of_attr.append(len(common_tuples))

            print('size of anslist', len(answlist), len(size_of_attr))
            # anslist = answlist

            "add noisy for binary vectors"
            anslist = []
            for mudata, c in zip(answlist, size_of_attr):
                # rho = (c ** (2 / 3)) / weight_sum * (0.2 * self.rho)  # 0.2 *
                # sigma = np.sqrt(1 / (2 * rho))
                noise = [self.gaussian_noise(sigma, len(mudata)) for _ in range(8)]
                average_noise = np.mean(noise, axis=0)
                # noisy = self.gaussian_noise(sigma, len(mudata))
                noisy = average_noise.astype(int)
                mumarg = []
                for marginal_cell, x in zip(mudata, noisy):
                    modified_vector = modify_vector(marginal_cell, x)
                    mumarg.append(modified_vector)
                anslist.append(mumarg)


            if len(anslist) == 0:
                ans = res

            elif len(anslist) == 2:
                answ = []
                for vec1 in anslist[0]:
                    for vec2 in anslist[1]:
                        multiplied_vector = [x * y for x, y in zip(vec1, vec2)]
                        inner_product = sum(multiplied_vector)
                        answ.append(np.float64(inner_product))
                res = np.array(answ)
                ans = res

            elif len(anslist) == 3:
                answ = []
                for vec1 in anslist[0]:
                    for vec2 in anslist[1]:
                        for vec3 in anslist[2]:
                            multiplied_vector = [x * y * z for x, y, z in zip(vec1, vec2, vec3)]
                            inner_product = sum(multiplied_vector)
                            answ.append(np.float64(inner_product))
                res = np.array(answ)
                ans = res

            else:
                print('error from marginal')
            # print('result of cl in mu', len(ans[cl]))
            # a = data.project(cl).datavector()
            # print('result of cl in da', len(a))

            return ans

  def noise_ans_vepa(self, multi_pa_data, cl, weight_sum):
      "slove pa paritational data, after slove ve datasets"
      counts = 0
      for sub_data in multi_pa_data:
          ans = self.noise_ans_ve(sub_data, cl, weight_sum)
          counts = counts + ans
      return counts


  def measurement_ve(self, multi_ve_data, cl, sigma):
            print('current cl', cl)

            answlist = []
            size_of_attr = []
            for sdate in multi_ve_data:
                print('current data domain', sdate.domain.attrs)
                common_tuples = [tuple for tuple in cl if tuple in sdate.domain.attrs]
                print('common_tuples', common_tuples)

                if len(common_tuples) == 0:
                    continue
                elif len(common_tuples) == len(cl):
                    print('current data domain including all cl')
                    res = sdate.project(common_tuples).datavector()
                    noise = [self.gaussian_noise(sigma, res.size) for _ in range(8)]
                    average_noise = np.mean(noise, axis=0)
                    res = res + average_noise
                    # res = res + self.gaussian_noise(sigma, res.size)
                    break
                elif len(common_tuples) < len(cl) and len(common_tuples) != 0:
                    onehot = sdate.project(common_tuples).dataonehot()
                    answlist.append(onehot)
                    size_of_attr.append(len(common_tuples))

            print('size of anslist', len(answlist), len(size_of_attr))
            # anslist = answlist

            "add noisy for binary vectors"
            anslist = []
            for mudata, c in zip(answlist, size_of_attr):
                noise = [self.gaussian_noise(sigma, len(mudata)) for _ in range(8)]
                average_noise = np.mean(noise, axis=0)
                # noisy = self.gaussian_noise(sigma, len(mudata))
                noisy = average_noise.astype(int)
                mumarg = []
                for marginal_cell, x in zip(mudata, noisy):
                    modified_vector = modify_vector(marginal_cell, x)
                    mumarg.append(modified_vector)
                anslist.append(mumarg)


            if len(anslist) == 0:
                ans = res

            elif len(anslist) == 2:
                answ = []
                for vec1 in anslist[0]:
                    for vec2 in anslist[1]:
                        multiplied_vector = [x * y for x, y in zip(vec1, vec2)]
                        inner_product = sum(multiplied_vector)
                        answ.append(np.float64(inner_product))
                res = np.array(answ)
                ans = res

            elif len(anslist) == 3:
                answ = []
                for vec1 in anslist[0]:
                    for vec2 in anslist[1]:
                        for vec3 in anslist[2]:
                            multiplied_vector = [x * y * z for x, y, z in zip(vec1, vec2, vec3)]
                            inner_product = sum(multiplied_vector)
                            answ.append(np.float64(inner_product))
                res = np.array(answ)
                ans = res

            else:
                print('error from marginal')
            # print('result of cl in mu', len(ans[cl]))
            # a = data.project(cl).datavector()
            # print('result of cl in da', len(a))

            return ans

  def measurement_repa(self, multi_pa_data, cl, sigma):
      "solve pa paritational data, after sloving ve datasets "
      counts = 0
      for sub_data in multi_pa_data:
          ans = self.measurement_ve(sub_data, cl, sigma)
          counts = counts + ans
      return counts


  def run(self, mudata, data, workload, num_synth_rows=None, initial_cliques=None):
    rounds = self.rounds or 16 * sum(len(item.domain) for item in mudata)#224
    candidates = compile_workload(workload)#469

    # weight_sum = 1 / len(candidates)
    weight_sum = 0
    for cl in candidates:
        weight_sum = weight_sum + len(cl) ** (2 / 3)

    answers = {cl: self.noise_ans_vepa(mudata, cl, weight_sum) for cl in candidates}
    # answers = {cl: data.project(cl).datavector() for cl in candidates}

    # "save answers"
    # folder_path = "../answers/"
    # file_name = "adult52_r_33d_0.8e_3way.json"
    # file_path = os.path.join(folder_path, file_name)
    # if os.path.exists(file_path):
    #     # 读取 JSON 文件
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         answer = json.load(f)
    #         print(111111111111111111111111111111111111)
    # else:
    #     answer = {cl: self.noise_ans_vepa(mudata, cl, weight_sum) for cl in candidates}
    #     answer = {str(key): value.tolist() for key, value in answer.items()}
    #     # 如果文件夹不存在，创建文件夹
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #         print(f"文件夹 {folder_path} 已创建")
    #
    #     # 保存为 JSON 文件
    #     with open(file_path, 'w', encoding='utf-8') as f:
    #         json.dump(answer, f, ensure_ascii=False, indent=4)
    #     print(f"已保存 a.json 文件到路径: {file_path}")
    #
    # answers = {ast.literal_eval(key): np.array(value) for key, value in answer.items()}

    if not initial_cliques:
      initial_cliques = [
        cl for cl in candidates if len(cl) == 1
      ]  # use one-way marginals

    oneway = [cl for cl in candidates if len(cl) == 1]

    measurements = []
    sigma = np.sqrt(rounds / (2 * 0.7 * self.rho))############################
    epsilon = np.sqrt(8 * 0.1 * self.rho / rounds)
    print("Initial Sigma", sigma)

    rho_used = len(oneway) * 0.5 / sigma ** 2
    for cl in initial_cliques:
      x = data.project(cl).datavector()
      y = x + self.gaussian_noise(sigma, x.size)
      # y = x#####################nopriv
      measurements.append(LinearMeasurement(y, cl, stddev=sigma))

    zeros = self.structural_zeros
    # WARNING: this repostiory recently underwent a refactoring
    # do not compare against this version of AIM until this comment is resolved.
    # TODO: add warm start and structural zeros back in
    model = estimation.mirror_descent(data.domain, measurements, iters=self.max_iters)

    t = 0
    terminate = False
    while not terminate:
      t += 1
      if self.rho - rho_used < 2 * (0.5 / sigma ** 2 + 1.0 / 8 * epsilon ** 2):
        # Just use up whatever remaining budget there is for one last round
        remaining = self.rho - rho_used
        sigma = np.sqrt(1 / (2 * 0.9 * remaining))
        epsilon = np.sqrt(8 * 0.1 * remaining)
        terminate = True

      rho_used += 1.0 / 8 * epsilon ** 2 + 0.5 / sigma ** 2
      size_limit = self.max_model_size * rho_used / self.rho

      small_candidates = filter_candidates(candidates, model, size_limit)
      cl = self.worst_approximated(small_candidates, answers, model, epsilon, sigma, weight_sum)
      print("exp selects attributes called", cl)

      n = data.domain.size(cl)
      # y = data.project(cl).datavector()########nopriv
      y = self.measurement_repa(mudata, cl, sigma)
      measurements.append(LinearMeasurement(y, cl, stddev=sigma))
      z = model.project(cl).datavector()

      # incorporate warm start here
      model = estimation.mirror_descent(data.domain, measurements, iters=self.max_iters)
      w = model.project(cl).datavector()
      # print('Selected',cl,'Size',n,'Budget Used',rho_used/self.rho)
      print("norm is ", np.linalg.norm(w - z, 1), n)
      print("relorant is ", sigma * np.sqrt(2 / np.pi) * n) #281W(1636.0456375254853 0.7978845608028654 2160) 12578.9135952108(1576.5330240947842 0.7978845608028654 12)
      if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
        print("(!!!!!!!!!!!!!!!!!!!!!!) Reducing sigma", sigma / 2)
        sigma /= 2
        epsilon *= 2

    print("Generating Data...")
    model = estimation.mirror_descent(data.domain, measurements, iters=self.max_iters)
    synth = model.synthetic_data(rows=num_synth_rows)

    return model, synth


def default_params():
  """
  Return default parameters to run this program

  :returns: a dictionary of default parameter settings for each command line argument
  """
  params = {}
  params["dataset"] = "../data/big15.csv"
  params["domain"] = "../data/big15-domain.json"
  params["epsilon"] = 1.0
  params["delta"] = 1e-5
  params["noise"] = "laplace"
  params["max_model_size"] = 80
  params["max_iters"] = 100
  params["degree"] = 2
  params["num_marginals"] = 128##########################################################128
  params["max_cells"] = 10000
  params["save"] = "../result_ablation/big15_OAr_22d_1.0e_2way_DM11.csv"

  return params


if __name__ == "__main__":

  description = ""
  formatter = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
  parser.add_argument("--dataset", help="dataset to use")
  parser.add_argument("--domain", help="domain to use")
  parser.add_argument("--epsilon", type=float, help="privacy parameter")
  parser.add_argument("--delta", type=float, help="privacy parameter")
  parser.add_argument(
    "--max_model_size", type=float, help="maximum size (in megabytes) of model"
  )
  parser.add_argument("--max_iters", type=int, help="maximum number of iterations")
  parser.add_argument("--degree", type=int, help="degree of marginals in workload")
  parser.add_argument(
    "--num_marginals", type=int, help="number of marginals in workload"
  )
  parser.add_argument(
    "--max_cells",
    type=int,
    help="maximum number of cells for marginals in workload",
  )
  parser.add_argument("--save", type=str, help="path to save synthetic data")

  parser.set_defaults(**default_params())
  args = parser.parse_args()

  data = Dataset.load(args.dataset, args.domain)###############################################
  multidata = Dataset.divide_random(args.dataset, args.domain, 2, vn=2)

  workload = list(itertools.combinations(data.domain, args.degree))

  workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]

  if args.num_marginals is not None:
    workload = [
      workload[i]
      for i in np.random.choice(len(workload), args.num_marginals, replace=False)
    ]

  workload = [(cl, 1.0) for cl in workload]
  mech = AIM_vmulti(
    args.epsilon,
    args.delta,
    max_model_size=args.max_model_size,
    max_iters=args.max_iters,
  )
  model, synth = mech.run(multidata, data, workload)

  if args.save is not None:
    synth.df.to_csv(args.save, index=False)

  errors = []
  for proj, wgt in workload:
    X = data.project(proj).datavector()
    Y = synth.project(proj).datavector()
    e = 0.5 * wgt * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
    errors.append(e)
  print("Average Error: ", np.mean(errors))
