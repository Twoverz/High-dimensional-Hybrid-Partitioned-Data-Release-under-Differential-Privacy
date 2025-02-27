import numpy as np
import itertools
from src.mbi import Dataset


data = Dataset.load("../data/adult52.csv", "../data/adult52-domain.json")
# data = Dataset.load("../result_ve_bayes/big15_1.csv", "../result_ve_bayes/big15_1-domain.json")

# 计算中间索引
# n = len(data.df)
# mid_index = len(data.domain.attrs) // 2
# part1 = data.domain.attrs[:mid_index]
# part2 = data.domain.attrs[mid_index:]
# workload1 = list(itertools.combinations(part1, 2))
# workload2 = list(itertools.combinations(part2, 2))
# workload = workload1+workload2

workload = list(itertools.combinations(data.domain, 2))
# workload = [
#       workload[i]
#       for i in np.random.choice(len(workload), 128, replace=False)
#     ]
workload = [(cl, 1.0) for cl in workload]




data_syn = Dataset.load("../result_pa_csv/big15_indbayes_epsilon_0.1.csv", "../data/adult52-domain.json")

KL = []
for cl, _ in workload:
        arr_ori = np.array(data.project(cl).datavector())
        arr_syn = np.array(data_syn.project(cl).datavector())

        nom_ori = arr_ori / np.sum(arr_ori)
        nom_ori = np.where(nom_ori > 0, nom_ori, 1e-5)
        nom_syn = arr_syn / np.sum(arr_syn)
        nom_syn = np.where(nom_syn > 0, nom_syn, 1e-5)

        KL_divergence = np.sum(nom_ori * np.log(nom_ori / nom_syn))

        KL.append(KL_divergence)

totol_KL = np.mean(KL)
print(totol_KL)

# errors = []
# for proj, wgt in workload:
#     X = data.project(proj).datavector()
#     Y = data_syn.project(proj).datavector()
#     e = 0.5 * wgt * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
#     errors.append(e)
# print("Average Error: ", np.mean(errors))