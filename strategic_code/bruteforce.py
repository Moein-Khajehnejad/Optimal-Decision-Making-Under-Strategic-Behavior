import scipy.optimize
import numpy as np
from strategic_code import configuration as Configuration
import scipy as sp
#import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import pickle
import click
import json as js


def dump_data(attr, result, time):
    out = {"n": attr["n"],
           "bruteforce": result,
           "time": time,
           "seed": attr['seed'],
           "sparsity": attr['degree_of_sparsity']}
    return out


def backtrack(matched, pi, p, utility, D, n, best_utility_till_now):
    if len(matched) == n:
        return LP(matched, pi, p, utility, D, best_utility_till_now)

    best_utility = -1
    best_pi = None
    temp_matched = [a for a in matched]
    ii = len(matched)
    candidate_values = np.where(D[ii, :] <= np.min(D[ii, :]) + 1)[0]
    for j in candidate_values:
        matched.append(j)
        #best_result = max(best_result, backtrack(matched,pi, p, utility, D, n))
        [temp_utility, temp_pi] = backtrack(matched, pi, p, utility, D, n, best_utility_till_now)
        if best_utility < temp_utility:
            best_utility = temp_utility
            best_pi = temp_pi
        if best_utility_till_now < temp_utility:
            best_utility_till_now = temp_utility
        #matched = matched[:-1]
        matched = [a for a in temp_matched]
    return [best_utility, best_pi]


def LP(matched, pi, p, utility, D, best_utility_till_now):
    #print(matched)

    fancy_pi = np.ones([len(pi)])
    fancy_pi[utility < 0] = 0
    if np.sum(p * fancy_pi[matched] * utility[matched]) < best_utility_till_now:
        #print('*******************************')
        return [-np.Inf, None]

    n = pi.size

    c = np.zeros([n])
    for i in range(n):
        j_star = matched[i]
        c[j_star] -= p[i] * utility[j_star]

    M = n * (n - 1)

    head = 0
    A_ub = np.zeros([M, n])
    b_ub = np.zeros([M])
    for i in range(n):
        j_star = matched[i]
        for k in range(n):
            if k != j_star:
                A_ub[head, k] = 1
                A_ub[head, j_star] = -1
                b_ub[head] = D[i, k] - D[i, j_star]
                head += 1

    result = scipy.optimize.linprog(
        c=c, A_ub=A_ub, b_ub=b_ub, A_eq=None, b_eq=None, bounds=[(0, 1)])
    if result.status != 0:
        if result.status != 2:
            print('not working!!! ', result.status)
        return [-np.Inf, None]
    return [-result.fun, result.x]


@click.command()
@click.option('--max_iter', default=50, )
@click.option('--output', required=True, help="output directory")
@click.option('--n', default=4, help="Number of states")
@click.option('--seed', default=2, help="random number for seed.")
@click.option('--sparsity', default=2, help="sparsity of the graph")
@click.option('--parallel', is_flag=True, help="if you use --parallel, the values of pi's will be updated in parallel,"
                                               " otherwise they will be updated sequentially.")
def experiment(output, n, seed, sparsity, parallel, max_iter):
    attr = Configuration.generate_configuration(
        n, seed, degree_of_sparsity=sparsity)

    matched = []
    start = time.time()
    [best_utility, best_pi] = backtrack(
        matched, attr["pi"], attr["p"], attr["utility"], attr["D"], attr["n"], -np.Inf)

    print("global total utility = " + str(best_utility))
    print("global pi = " + str(best_pi))
    end = time.time()
    with open(output + '_bruteforce_config.json', "w") as fi:
        fi.write(js.dumps(dump_data(attr, best_utility, end - start)))
    print("Bruteforce RunTime = " + str(end - start))


if __name__ == '__main__':
    experiment()
