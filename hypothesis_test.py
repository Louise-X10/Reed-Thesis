from scipy.stats import wilcoxon
import numpy as np
import pandas as pd

epsilons = np.loadtxt("compile-errorbar13689-500reps-20pts/compiled-epsilons.csv")
header = np.array(['i', 'reg', 'd0', 'd1e-9'])
df = pd.DataFrame(epsilons, columns=header)
df = df.sort_values(['i', 'reg'])
df1, df2 = [df.pivot(index='i', columns='reg', values=col)
                     for col in ['d0', 'd1e-9']]
x = df1[1] # regularized epsilons
y = df1[0] # non-regularized epsilons

result = wilcoxon(x=x, y=y, alternative="greater", zero_method="zsplit") 
print(result.pvalue)