import matplotlib.pyplot as plt
from scipy.stats import laplace, norm
import numpy as np

x = np.linspace(-5, 5, 1000)  # Range of x values

# Parameters for the distributions
loc_laplace = 0  # Location parameter for Laplace distribution
scale_laplace = 1  # Scale parameter for Laplace distribution
loc_normal = 0  # Mean for Normal distribution
scale_normal = 1  # Standard deviation for Normal distribution

# Calculate probability density functions (PDFs) for the distributions
pdf_laplace = laplace.pdf(x, loc=loc_laplace, scale=scale_laplace)
pdf_normal = norm.pdf(x, loc=loc_normal, scale=scale_normal)

# Plot the Laplace distribution
plt.plot(x, pdf_laplace, label='Laplace', color='blue')

# Plot the Normal distribution
plt.plot(x, pdf_normal, label='Gaussian', color='green', ls='--')

# Add labels and legend
plt.ylabel('Probability Density')
plt.legend()

# Show the plot
plt.grid(False)
plt.show()
