import numpy as np


def generate_configuration(n, seed, degree_of_sparsity=None):
    attr = {}
    np.random.seed(seed)
    attr['seed'] = seed
    attr['n'] = n
    num_movable_nodes = []
    attr['degree_of_sparsity'] = degree_of_sparsity
    D = np.random.rand(n, n)  # all the distances are less than 1
    degree_of_sparsity = n - 1 if degree_of_sparsity is None else degree_of_sparsity
    for i in range(n):
        indices = np.random.choice(n, degree_of_sparsity, replace=False)
        D[i][indices] = 2
    np.fill_diagonal(D, 0)
    attr['D'] = D

    for i in range(n):
        a = np.where(D[i] < 1)
        num_movable_nodes.append(np.size(a))
    attr["num_movable_nodes"] = num_movable_nodes
    #unnormalized_P = np.random.rand(n)
    unnormalized_P = np.random.normal(0.5, 0.1, n)
    p = unnormalized_P / sum(unnormalized_P)
    attr["p"] = p
    utility = (np.random.rand(n) - 0.5) * 2
    #utility = np.random.uniform(0.0,1,n)
    utility = np.sort(utility)
    utility = utility[::-1]
    attr["utility"] = utility
    pi = np.zeros(n)
    attr["pi"] = pi
    return attr
# print("D = " + str(D))
# print("movable_nodes = " +str(num_movable_nodes))
# print("Utility = " + str(utility))
