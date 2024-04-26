
import os
import numpy as np
from post_process import *
from sys import argv, exit

# @param meta_directory: a directory that contains multiple results directories with same number of trials
#   Combine the results from these directories into one big result, then compute epsilons
# @return: Save new prediction arrays, save epsilon values to compiled_epsilons.csv file
def compile_target_epsilons(meta_directory):
    # For each trial, combine the prediction arrays across all directories
    for i in range(10):
        predict_i = np.array([0., 0.])
        predict_noi = np.array([0., 0.])
        predict_i_reg = np.array([0., 0.])
        predict_noi_reg = np.array([0., 0.])
        for dir in os.listdir(meta_directory):
            if 'results' in dir:
                predict_i += np.loadtxt(f'{dir}/sample13689_trial{i}_predict_i.csv')
                predict_noi += np.loadtxt(
                    f'{dir}/sample13689_trial{i}_predict_noi.csv')
                predict_i_reg += np.loadtxt(
                    f'{dir}/sample13689_trial{i}_predict_i_reg.csv')
                predict_noi_reg += np.loadtxt(f'{dir}/sample13689_trial{i}_predict_noi_reg.csv')
        # Save combined prediction arrays
        np.savetxt(
            f"./{meta_directory}/sample13689_trial{i}_predict_i.csv", predict_i, delimiter=",")
        np.savetxt(
            f"./{meta_directory}/sample13689_trial{i}_predict_noi.csv", predict_noi, delimiter=",")
        np.savetxt(
            f"./{meta_directory}/sample13689_trial{i}_predict_i_reg.csv", predict_i_reg, delimiter=",")
        np.savetxt(
            f"./{meta_directory}/sample13689_trial{i}_predict_noi_reg.csv", predict_noi_reg, delimiter=",")
        # Compute epsilons using the new prediction arrays
        epsilons = get_epsilon_delta(
            predict_i, predict_noi, reps=1000, deltas=[0, pow(10, -9), 0.01])
        epsilons_reg = get_epsilon_delta(
            predict_i_reg, predict_noi_reg, reps=1000, deltas=[0, pow(10, -9), 0.01])
        np.savetxt(
            f"./{meta_directory}/sample13689_trial{i}_epsilons.csv", epsilons, delimiter=",")
        np.savetxt(
            f"./{meta_directory}/sample13689_trial{i}_epsilons_reg.csv", epsilons_reg, delimiter=",")

# Same function as print_epsilons, except use trial instead of 1 as 1st column name
def print_target_epsilons(directory):
    header = np.array(['trial', 'reg', 'd0', 'd1e-9'])
    epsilons = np.loadtxt(f'./{directory}/compiled-epsilons.csv')
    df = pd.DataFrame(epsilons, columns=header)
    df = df.sort_values(['trial', 'reg'])
    print("="*20, directory, ": epsilons ", "="*20)
    print(df.pivot(index='trial', columns='reg'))
    pivot_dfs = {}
    for col in ['d0', 'd1e-9']:
        pivot_df = df.pivot(index='trial', columns='reg', values=col)
        pivot_dfs[col] = pivot_df[1] - pivot_df[0]
    diff_df = pd.DataFrame(pivot_dfs)
    # positive diff means epsilon increased = privacy worsened
    print("="*20, directory, ": epsilon diffs ", "="*20)
    print(diff_df)

if __name__ == "__main__":
    if len(argv) != 2:
        print("Please provide a meta directory (containing results directories to be combined) to proceed.")
        exit(0)
    meta_directory = argv[1:][0]
    compile_target_epsilons(meta_directory)
    print_target_epsilons(meta_directory)
