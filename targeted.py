from post_process import *
from functions import *
from scipy.special import softmax
import os

results_dir = 'results-exp2-targeted13689-500reps'
checkpoints_dir = 'checkpoints-exp2-targeted13689-500reps'

if os.path.exists(results_dir) == False:
    os.mkdir(results_dir)
if os.path.exists(checkpoints_dir) == False:
    os.mkdir(checkpoints_dir)

# Same as function in default.py, restated here because checkpoints_dir is different

def train_and_predict(i: int, rep: int, ith_sample_X, sample_X, sample_Y, train_model_func, predict, trial):
    model = train_model_func(
        sample_X, sample_Y, X_test, y_test, epochs, batch_size)
    # Save model weights
    if sample_X.shape[0] == n:
        scenario = 1
    else:
        scenario = 0
    if train_model_func == train_model:
        reg = 1
    elif train_model_func == train_model_reg:
        reg = 0
    model.save_weights(
        f'./{checkpoints_dir}/checkpoint_sample{i}_rep{rep}_i={scenario == 1}_reg={reg == 1}.weights.h5')

    # Update model prediction array
    logit = model.predict(ith_sample_X)
    output = np.argmax(softmax(logit, axis=1))
    if output == 0:
        predict[0] += 1
    else:
        predict[1] += 1

# Same function as in default.py, but on target i value instead of random i value

def prob_of_model_predict_on_ith(n: int, reps: int, trial):
    # number of 0 and 1 predictions for scenarios (with i, no i, with reg, no reg)
    predict_i = np.array([0, 0])
    predict_noi = np.array([0, 0])
    predict_i_reg = np.array([0, 0])
    predict_noi_reg = np.array([0, 0])

    start_time_reps = time.time()
    # Pick the ith index
    i = target
    print("Picked the ", i, "th datapoint")
    # Update prediction arrays after training reps many models
    for rep in range(reps):
        print(
            f"Start prediction procedure for trial {trial} = sample {i}, repetition {rep}")
        # Create samples with and without i
        sample_idx_noi = sample_dataset(i, n-1)
        sample_X_noi = X_train[sample_idx_noi]
        sample_y_noi = y_train[sample_idx_noi]
        # pass [i] instead of i to get correct shape of (1,86)
        ith_sample_X = X_train[[i]]
        ith_sample_y = y_train[[i]]
        sample_X = np.concatenate([ith_sample_X, sample_X_noi])
        sample_Y = np.concatenate([ith_sample_y, sample_y_noi])
        # Update prediction arrays after training model in each scenario
        train_and_predict(i, rep, ith_sample_X,
                          sample_X, sample_Y, train_model, predict_i, trial)
        train_and_predict(i, rep, ith_sample_X,
                          sample_X_noi, sample_y_noi, train_model, predict_noi, trial)
        train_and_predict(i, rep, ith_sample_X,
                          sample_X, sample_Y, train_model_reg, predict_i_reg, trial)
        train_and_predict(i, rep, ith_sample_X,
                          sample_X_noi, sample_y_noi, train_model_reg, predict_noi_reg, trial)
    end_time_reps = time.time()
    time_elapsed_reps = (end_time_reps - start_time_reps)
    print(
        f"Time elapsed for single datapoint, total {reps} reps: ", time_elapsed_reps)
    print("="*40)
    return i, predict_i, predict_noi, predict_i_reg, predict_noi_reg

epochs = 50
batch_size = 48
n = 1000
reps = 500
target = 13689
trials = 1
deltas = [0, pow(10, -9)] 

if __name__ == "__main__":
    print("Start process for targeted ...")
    overall_start_time = time.time()
    i_indices = []
    for trial in range(trials):
        i, predict_i, predict_noi, predict_i_reg, predict_noi_reg = prob_of_model_predict_on_ith(
            n, reps, trial)
        i_indices += [i]
        # Save model predictions
        np.savetxt(
            f"./{results_dir}/sample{i}_predict_i.csv", predict_i, delimiter=",")
        np.savetxt(
            f"./{results_dir}/sample{i}_predict_noi.csv", predict_noi, delimiter=",")
        np.savetxt(
            f"./{results_dir}/sample{i}_predict_i_reg.csv", predict_i_reg, delimiter=",")
        np.savetxt(
            f"./{results_dir}/sample{i}_predict_noi_reg.csv", predict_noi_reg, delimiter=",")
        # Save epsilon values
        epsilons = get_epsilon_delta(predict_i, predict_noi, reps, deltas)
        epsilons_reg = get_epsilon_delta(
            predict_i_reg, predict_noi_reg, reps, deltas)
        np.savetxt(
            f"./{results_dir}/sample{i}_epsilons.csv", epsilons, delimiter=",")
        np.savetxt(
            f"./{results_dir}/sample{i}_epsilons_reg.csv", epsilons_reg, delimiter=",")

    # Save indices of i
    np.savetxt(f"./{results_dir}/all_i_indices.csv", i_indices, delimiter=",")

    overall_end_time = time.time()
    overall_time_elapsed = (overall_end_time - overall_start_time)
    print(
        f"Overall time elapsed for {trials} different trials: ", overall_time_elapsed)
