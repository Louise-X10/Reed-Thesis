# Reed-Thesis

This is the code for my undergraduate thesis "Privacy Paradox: Regularized Neural Networks and Distributional Differential Privacy".
 
Here are the files for running the experiments
- default.py: code for running Experiment 1 Breadth exploration
- targeted.py: code for running Experiment 2 Depth exploration
- errorest.py: code for running Experiment 3 Error estimation
- functions.py: code for constructing experiments

Here are the files for analyzing experimental results
- post_process.py: code for computing epsilon values from experimental results
- compile_epsilons.py: code to compute epsilons for results from Experiment 1
- compile_target_epsilons.py: code to compute epsilons for results from Experiment 2
- compile_errorest_epsilons.py: code to compute epsilons for results from Experiment 3
- hypothesis_test.py: run Wilcoxon Signed Rank Test for results from Experiment 3

Here are the files for making plots for experimental results
- plot_results.py: plot results for Experiments 1 & 3
- plot_normal_laplace.py: plot Laplace vs Gaussian distribution