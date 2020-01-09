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


def dump_data(attr, non_strategic, strategic, strategic_deter, time, new_p, new_p_non_strat, new_p_strat_deter):
    out = {"n": attr["n"],
           "strategic": strategic,
           "non_strategic": non_strategic,
           "strategic_deter": strategic_deter,
           "seed": attr['seed'],
           "time": time,
           "sparsity": attr['degree_of_sparsity'],
           "parallel": attr['parallel'],
           "new_p": new_p,
           "new_p_non_strat": new_p_non_strat,
           "new_p_strat_deter": new_p_strat_deter}
    return out


def dump_iterations(attr, strategic, num_of_changed_steps):
    out_list = []
    for iter, val in enumerate(strategic):
        out = {"n": attr["n"],
               "strategic": val,
               "number of changed steps": num_of_changed_steps[iter],
               "seed": attr['seed'],
               "sparsity": attr['degree_of_sparsity'],
               "iteration": iter}
        out_list.append(out)
    return out_list

def dump_pi(attr, pi):
    out_list = []
    for iter, val in enumerate(pi):
        out = {"n": attr["n"],
               "pi": val,
               "seed": attr['seed'],
               "sparsity": attr['degree_of_sparsity'],
               "pos": iter}
        out_list.append(out)
    return out_list

def dump_p(attr, p):
    out_list = []
    for iter, val in enumerate(p):
        out = {"n": attr["n"],
               "p": val,
               "seed": attr['seed'],
               "sparsity": attr['degree_of_sparsity'],
               "pos": iter}
        out_list.append(out)
    return out_list

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
@click.option('--max_iter', default=20, )
@click.option('--njobs', default=40, )
@click.option('--output', required=True, help="output directory")
@click.option('--n', default=4, help="Number of states")
@click.option('--seed', default=2, help="random number for seed.")
@click.option('--sparsity', default=2, help="sparsity of the graph")
@click.option('--parallel', is_flag=True, help="if you use --parallel, the values of pi's will be updated in parallel,"
                                               " otherwise they will be updated sequentially.")
def experiment(output, n, seed, sparsity, parallel, max_iter, njobs):
    attr = Configuration.generate_configuration(
        n, seed, degree_of_sparsity=sparsity)
    best_utility = -1.5
    u_greedy = []
    count_changed_list = []
    iter = 1
    start = time.time()
    sorted_index = np.arange(attr["n"])[np.argsort(attr["utility"])[::-1]]
    attr['parallel'] = parallel
    transition_matrix = transition_check(attr["pi"], attr["D"], attr["n"])
    while True:
        any_update = False
        if parallel:
            from joblib import Parallel, delayed
            previous_pi = attr["pi"].copy()
            results = Parallel(n_jobs=njobs)(delayed(lambda x: update(
                x, previous_pi, attr["p"], attr["D"], attr["utility"]))(i) for i in sorted_index)
            for (pi_k, best_util_k), k in zip(results, sorted_index):
                if best_util_k > best_utility:
                    attr["pi"][k] = pi_k
                    any_update = True
            best_utility = compute_utility(
                attr["pi"], attr["p"], attr["D"], attr["utility"])

        else:
            for k in sorted_index:
                print("state ", k)
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
        previous_transition_matrix = transition_matrix
        transition_matrix = transition_check(attr["pi"], attr["D"], attr["n"])
        count_changed_list.append(
            np.sum(np.abs(transition_matrix - previous_transition_matrix)) / 2)

        #print("matrix = " + str(transition_matrix))
        if not any_update or iter > max_iter:
            # if not np.any(transition_matrix[-1]-transition_matrix[-2]):
            end = time.time()
            run_time = end - start
            print("Greedy RunTime = " + str(run_time))
            break
    
    new_p = np.abs(transition_matrix.T.dot(attr['p'])-attr['p']).sum()/2
    non_strategic_u = attr["utility"].copy()
    non_strategic_u[non_strategic_u < 0] = 0
    pi_non_strategic = non_strategic_u.copy()
    pi_non_strategic[pi_non_strategic > 0] = 1
  
    non_strategic_utility = compute_utility(
        pi_non_strategic, attr["p"], attr["D"], non_strategic_u)
    transition_matrix_non_strat = transition_check(pi_non_strategic, attr["D"], attr["n"])
    new_p_non_strat = np.abs(transition_matrix_non_strat.T.dot(attr['p'])-attr["p"]).sum()/2

    non_strategic_u[non_strategic_u < 0] = 0
    pi_strategic_deter = attr["pi"].copy()
    pi_strategic_deter[pi_strategic_deter > 0.5] = 1
    pi_strategic_deter[pi_strategic_deter <= 0.5] = 0

    strategic_deterministic_utility = compute_utility(
        pi_strategic_deter, attr["p"], attr["D"], non_strategic_u)
    transition_matrix_strat_det = transition_check(pi_strategic_deter, attr["D"], attr["n"])
    new_p_strat_deter = np.abs(transition_matrix_strat_det.T.dot(attr['p'])-attr["p"]).sum()/2

    attr['p']
    with open(output + '_config.json', "w") as fi:
        fi.write(js.dumps(dump_data(attr, non_strategic_utility,
                                    best_utility, strategic_deterministic_utility, run_time,
                                    new_p, new_p_non_strat, new_p_strat_deter)))

    with open(output + '_iteration.json', "w") as fi:
        fi.write(js.dumps(dump_iterations(attr, u_greedy, count_changed_list)))


def compute(output, D, U, Px, seed, max_iter=100, verbose=False, parallel=False, njobs=1):
    attr = Configuration.generate_configuration_state(
        U, D, Px, seed)
    best_utility = -1.5
    u_greedy = []
    count_changed_list = []
    iter = 1
    start = time.time()
    sorted_index = np.arange(attr["n"])[np.argsort(attr["utility"])[::-1]]
    attr['parallel'] = parallel
    transition_matrix = transition_check(attr["pi"], attr["D"], attr["n"])
    while True:
        print("step = " + str(iter) + ":")
        any_update = False
        if parallel:
            from joblib import Parallel, delayed
            previous_pi = attr["pi"].copy()
            results = Parallel(n_jobs=njobs)(delayed(lambda x: update(
                x, previous_pi, attr["p"], attr["D"], attr["utility"]))(i) for i in sorted_index)
            for (pi_k, best_util_k), k in zip(results, sorted_index):
                if best_util_k > best_utility:
                    attr["pi"][k] = pi_k
                    any_update = True
            best_utility = compute_utility(
                attr["pi"], attr["p"], attr["D"], attr["utility"])

        else:
            for k in sorted_index:
                if verbose:
                    print("state ", k)
                [best_value, best_possible_utility] = update(
                    k, attr["pi"], attr["p"], attr["D"], attr["utility"])
                if best_possible_utility > best_utility:
                    attr["pi"][k] = best_value
                    best_utility = best_possible_utility
                    any_update = True

        if verbose:
            print("greedy total utility  = " + str(best_utility))
       # print("pi = " + str(Configuration.pi))
        u_greedy.append(best_utility)
        iter += 1
        previous_transition_matrix = transition_matrix
        transition_matrix = transition_check(attr["pi"], attr["D"], attr["n"])
        count_changed_list.append(
            np.sum(np.abs(transition_matrix - previous_transition_matrix)) / 2)

        #print("matrix = " + str(transition_matrix))
        if not any_update or iter > max_iter:
            # if not np.any(transition_matrix[-1]-transition_matrix[-2]):
            end = time.time()
            run_time = end - start
            print("Greedy RunTime = " + str(run_time))
            break
    new_p = np.abs(transition_matrix.T.dot(attr['p'])-attr['p']).sum()/2
    non_strategic_u = attr["utility"].copy()
    non_strategic_u[non_strategic_u < 0] = 0
    pi_non_strategic = non_strategic_u.copy()
    pi_non_strategic[pi_non_strategic > 0] = 1
  
    non_strategic_utility = compute_utility(
        pi_non_strategic, attr["p"], attr["D"], non_strategic_u)
    transition_matrix_non_strat = transition_check(pi_non_strategic, attr["D"], attr["n"])
    new_p_non_strat = np.abs(transition_matrix_non_strat.T.dot(attr['p'])-attr["p"]).sum()/2

    non_strategic_u[non_strategic_u < 0] = 0
    pi_strategic_deter = attr["pi"].copy()
    pi_strategic_deter[pi_strategic_deter > 0.5] = 1
    pi_strategic_deter[pi_strategic_deter <= 0.5] = 0

    strategic_deterministic_utility = compute_utility(
        pi_strategic_deter, attr["p"], attr["D"], non_strategic_u)
    transition_matrix_strat_det = transition_check(pi_strategic_deter, attr["D"], attr["n"])
    new_p_strat_deter = np.abs(transition_matrix_strat_det.T.dot(attr['p'])-attr["p"]).sum()/2

    with open(output + '_config.json', "w") as fi:
        fi.write(js.dumps(dump_data(attr, non_strategic_utility,
                                    best_utility, strategic_deterministic_utility, run_time,
                                    new_p, new_p_non_strat, new_p_strat_deter)))
    with open(output + '_iteration.json', "w") as fi:
        fi.write(js.dumps(dump_iterations(attr, u_greedy, count_changed_list)))

    with open(output + '_pi.json', "w") as fi:
        fi.write(js.dumps(dump_pi(attr,attr["pi"])))
    
    with open(output + '_p.json', "w") as fi:
        fi.write(js.dumps(dump_p(attr,transition_matrix.T.dot(attr['p'])))) 

    with open(output + '_p_orig.json', "w") as fi:
        fi.write(js.dumps(dump_p(attr, attr['p'])))


    return transition_matrix, attr


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
