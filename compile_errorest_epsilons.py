
import os
import numpy as np
from sys import argv
from compile_epsilons import print_epsilons

# Same function as compile_epsilons, but record trial number instead of i value (because i values are the same)
def compile_errorest_epsilons(directory):
    # placeholder roww: i, reg, eps for delta0, delta 10^-9
    epsilons = np.array([0, True, 100, 100]).reshape((1, 4)) 

    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        # only consider the epsilons.csv and epsilons_reg.csv files
        if "sample" in filename and 'epsilons' in filename:
            # isolate index i from filename
            specs = filename[0:-4].split('_')
            trial = specs[1]
            trial = int(trial[5:])
            data = np.loadtxt(file)  
            # Append i, reg to data
            if "reg" in filename:
                data = np.concatenate([np.asarray([True]), data])
            else:
                data = np.concatenate([np.asarray([False]), data])
            data = np.concatenate([np.asarray([trial]), data])
            # compile results into epsilons array
            epsilons = np.concatenate([epsilons, data.reshape((1, 4))])

    epsilons = epsilons[1:, :]  # remove placeholder 1st row
    np.savetxt(f'./{directory}/compiled-epsilons.csv', epsilons)


# Can compile epsilons for multiple result directories at once
if __name__ == "__main__":
    if len(argv) < 2:
        print("Please provide one or more directory name to proceed.")
        exit(0)
    directories = argv[1:]
    for directory in directories:
        compile_errorest_epsilons(directory)
        print_epsilons(directory)