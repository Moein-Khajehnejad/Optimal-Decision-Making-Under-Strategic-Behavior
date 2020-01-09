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
from strategic_code.greedy import *


@click.command()
@click.option('--max_iter', default=20, )
@click.option('--njobs', default=40, )
@click.option('--u', required=True)
@click.option('--px', required=True)
@click.option('--alpha',type=float ,required=True)
@click.option('--d', required=True)
@click.option('--c', type=float, required=True)
@click.option('--seed', default=2, help="random number for seed.")
@click.option('--output', required=True, help="output")
@click.option('--parallel', is_flag=True, help="if you use --parallel, the values of pi's will be updated in parallel,"
                                               " otherwise they will be updated sequentially.")
def experiment(output, seed, parallel, max_iter, njobs, u, px, d, c, alpha):
    distances = pd.read_csv(
        d, index_col=0).values*4.0/alpha
    px = pd.read_csv(px, index_col=0).values.flatten()
    u = pd.read_csv(u,
                    index_col=0, names=["index", "u"]).values.flatten() - c
    print(u.shape,px.shape, distances.shape)
    np.arange(u.shape[0])[np.argsort(u)[::-1]]
    compute(output, distances, u, px, seed, max_iter=max_iter,
            parallel=parallel, verbose=False, njobs=njobs)


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
