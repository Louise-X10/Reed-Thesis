import os
import numpy as np
import pandas as pd
from sys import argv, exit


# @param directory: a results directory from default.py
# Compute 1 epsilon value for each trial (consisting of 4 prediction arrays and 2 epsilon arrays)
# @return: Save all epsilon values to compiled_epsilons.csv file

def compile_epsilons(directory):
    # placeholder roww: i, reg, eps for delta0, delta 10^-9
    epsilons = np.array([0, True, 100, 100]).reshape((1, 4)) 

    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        # only consider the epsilons.csv and epsilons_reg.csv files
        if "sample" in filename and 'epsilons' in filename:
            # isolate index i from filename
            specs = filename[0:-4].split('_')
            sample = specs[0]
            i = int(sample[6:])
            data = np.loadtxt(file)  
            # Append i, reg to data
            if "reg" in filename:
                data = np.concatenate([np.asarray([True]), data])
            else:
                data = np.concatenate([np.asarray([False]), data])
            data = np.concatenate([np.asarray([i]), data])
            # compile results into epsilons array
            epsilons = np.concatenate([epsilons, data.reshape((1, 4))])

    epsilons = epsilons[1:, :]  # remove placeholder 1st row
    np.savetxt(f'./{directory}/compiled-epsilons.csv', epsilons)

def print_epsilons(directory):
    header = np.array(['i', 'reg', 'd0', 'd1e-9'])
    epsilons = np.loadtxt(f'./{directory}/compiled-epsilons.csv')
    df = pd.DataFrame(epsilons, columns=header)
    df = df.sort_values(['i', 'reg'])
    print("="*20, directory, ": epsilons ", "="*20)
    print(df.pivot(index='i', columns='reg'))
    pivot_dfs = {}
    for col in ['d0', 'd1e-9']:
        pivot_df = df.pivot(index='i', columns='reg', values=col)
        pivot_dfs[col] = pivot_df[1] - pivot_df[0]
    diff_df = pd.DataFrame(pivot_dfs)
    # positive diff means epsilon increased = privacy worsened
    print("="*20, directory, ": epsilon diffs ", "="*20)
    print(diff_df)


# Can compile epsilons for multiple result directories at once
if __name__ == "__main__":
    if len(argv) < 2:
        print("Please provide one or more directory name to proceed.")
        exit(0)
    directories = argv[1:]
    for directory in directories:
        compile_epsilons(directory)
        print_epsilons(directory)
