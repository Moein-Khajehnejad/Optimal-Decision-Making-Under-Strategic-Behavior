import numpy as np
import scipy as sp
from strategic_code import configuration as Configuration
import pandas as pd
import random
import time
import pickle
import pandas as pd
import json as js
import click


def dump_data(utility_list, iter, best_utility):

    out = {"iteratoin": iter, "utility": best_utility}
    return out


def compute_utility(pi_c, p, D, utility):
    n = pi_c.size
    u = 0
    for i in range(n):
        z = pi_c - D[i]
        mx_z = np.max(z)
        epsilon = 1e-9
        ind = np.where(np.abs(z - mx_z) < epsilon)[0]
        mx_val = np.max(pi_c[ind] * utility[ind])
        u += p[i] * mx_val
    return u


def update(k, pi, p, D, utility):
    n = pi.size
    pi_c = pi.copy()
    pi_c[k] = -np.Inf
    candidate_values = [0, 1]
    for i in range(n):
        candidate_values.append(np.max(pi_c - D[i, :]) + D[i, k])
    candidate_values = np.unique(candidate_values)
    candidate_values = candidate_values[(
        candidate_values >= 0) & (candidate_values <= 1)]
    best_possible_utility = -np.Inf
    best_value = None
    for v in candidate_values:
        pi_c[k] = v
        u = compute_utility(pi_c, p, D, utility)
        if u > best_possible_utility:
            best_possible_utility = u
            best_value = v
    return [best_value, best_possible_utility]


def transition_check(pi, D, n):
    transition_matrix = np.zeros((n, n))
    for i in range(n):
        transition_matrix[i][np.argmax(pi - D[i])] = 1
    return(transition_matrix)


@click.command()
@click.option('--output', required=True, help="output directory")
@click.option('--n', default=4, help="Number of states")
@click.option('--seed', default=2, help="random number for seed.")
@click.option('--sparsity', default=2, help="sparsity of the graph")
def experiment(output, n, seed, sparsity):
    attr = Configuration.generate_configuration(
        n, seed, degree_of_sparsity=sparsity)
    best_utility = -1.5
    u_greedy = []
    transition_list = []
    iter = 1
    start = time.time()
    result_data = []
    while True:
        any_update = False
        for k in range(attr["n"]):
            [best_value, best_possible_utility] = update(
                k, attr["pi"], attr["p"], attr["D"], attr["utility"])
            if best_possible_utility > best_utility:
                attr["pi"][k] = best_value
                best_utility = best_possible_utility
                any_update = True
        print("step = " + str(iter) + ":")
        print("greedy total utility  = " + str(best_utility))
       # print("pi = " + str(Configuration.pi))
        u_greedy.append(best_utility)
        iter += 1
        transition_matrix = transition_check(attr["pi"], attr["D"], attr["n"])
        transition_list.append(transition_matrix)
        #print("matrix = " + str(transition_matrix))
        if not any_update:
            # if not np.any(transition_matrix[-1]-transition_matrix[-2]):
            end = time.time()
            run_time = end - start
            print("Greedy RunTime = " + str(run_time))
            break
        result_data.append(dump_data(attr, iter-1, best_utility))

    non_strategic_u = attr["utility"].copy()
    non_strategic_u[non_strategic_u < 0] = 0
    pi_non_strategic = non_strategic_u.copy()
    pi_non_strategic[pi_non_strategic > 0] = 1

    non_strategic_utility = compute_utility(
        pi_non_strategic, attr["p"], attr["D"], non_strategic_u)

    print()
    print('----------')
    print(result_data)
    print('----------')
    print()
    with open(output + '_config.json', "w") as fi:
        fi.write(js.dumps(result_data))


if __name__ == '__main__':
    experiment()

    #
    #
    # outfile_path = output + '_greedy_results.csv'
    # df = pd.read_csv(outfile_path)
    #
    # # df = pd.DataFrame(columns=['n', 'seed', 'degree_of_sparsity', 'utilities', 'mass', 'D', 'total_utility_greedy', 'run_time', 'difference_strategic_with_nonstrategic'])
    # df.loc[len(df)] = [Configuration.n, Configuration.random_seed, Configuration.degree_of_sparsity,
    #                    Configuration.utility, Configuration.p, Configuration.D, u_greedy, run_time, difference_stra_nostra]
    # df.to_csv(outfile_path, index=False)
