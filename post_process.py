import numpy as np
import math
import os

# Compute epsilon for a given delta and given setting (aka whether reg is used)
# @param predict_i: the array [number of 0 predictions, number of 1 predictions] when row i is included
# @param predict_noi: the array [number of 0 predictions, number of 1 predictions] when row i is excluded
# @param reps: number of repetitions, also equal to sum of each array, i.e. reps = number of 0 predictions + number of 1 predictions
# @param delta: some delta value between 0 and 1

def get_epsilon_from_delta(predict_i, predict_noi, reps, delta):
    prob_i = np.divide(predict_i, reps)
    prob_i0, prob_i1 = prob_i
    prob_noi = np.divide(predict_noi, reps)
    prob_noi0, prob_noi1 = prob_noi

    def compute_single_epsilon(prob_D, prob_Dp):
        # Compute epsilon s.t. prob(with i) <= e^eps prob(without i) + delta
        if prob_D == prob_Dp:
            # if prob D == prob D', then have complete privacy, i.e. eps = 0
            epsilon = 0
        elif prob_D <= delta:
            # if prob_D <= delta, then DP is trivially satisfied, i.e. eps = 0
            epsilon = 0
        elif prob_Dp == 0:
            # If prob D = 0 and prob D' - delta > 0, then have no privacy, i.e. eps = infinity
            epsilon = np.infty
        else:
            ratio = (prob_D - delta) / prob_Dp
            # Take the maximum ratio (between outcome 0 and outcome 1), figure out min epsilon st ln(prob ratio) <= eps
            epsilon = math.log(ratio)
        return epsilon
    epsilon = max(
        compute_single_epsilon(prob_i0, prob_noi0),
        compute_single_epsilon(prob_i1, prob_noi1),
        compute_single_epsilon(prob_noi0, prob_i0),
        compute_single_epsilon(prob_noi1, prob_i1))
    return epsilon

# Compute epsilon from given list of delta values

def get_epsilon_delta(predict_i, predict_noi, reps, deltas):
    epsilons = list(map(lambda delta: get_epsilon_from_delta(
        predict_i, predict_noi, reps, delta), deltas))
    return epsilons

def print_prediction_arrays_from_dir(directory):
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        specs = filename.split('_')
        # Find epsilon files, load prediction arrays, then print them
        if "sample" in filename and 'epsilons' in filename:
            # isolate index i from filename
            specs = filename[0:-4].split('_')
            sample = specs[0]
            i = int(sample[6:])
            predict_i = np.loadtxt(os.path.join(
                directory, "_".join([sample, 'predict', 'i'])+'.csv'))
            predict_noi = np.loadtxt(os.path.join(
                directory, "_".join([sample, 'predict', 'noi'])+'.csv'))
            predict_i_reg = np.loadtxt(os.path.join(
                directory, "_".join([sample, 'predict', 'i', 'reg'])+'.csv'))
            predict_noi_reg = np.loadtxt(os.path.join(
                directory, "_".join([sample, 'predict', 'noi', 'reg'])+'.csv'))
            print(i, ": ", predict_i, predict_noi,
                    predict_i_reg, predict_noi_reg)
    return predict_i, predict_noi, predict_i_reg, predict_noi_reg
