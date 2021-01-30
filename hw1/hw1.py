import numpy as np
import random as rand
import matplotlib.pyplot as ppl
# I pledge my honor that I have abided by the Stevens honor system.

# Experiments
d = 1000
# All of our k values
k_array = [10, 30, 50, 80, 100, 150, 200, 300, 400, 500, 600, 800, 1000]

# Create an empty array so we can store our L2 norm calculations
Ax_l2 = np.zeros(len(k_array))

# Create a uniform random vector of dimesion d
x = np.random.default_rng().uniform(-100, 100, d)

# Generate our A matrix and calculate the L2 of [Ax]
for i, k in enumerate(k_array):
    A = np.random.default_rng().normal(0, (1/np.sqrt(k)), (k, d))
    Ax_l2[i] = np.linalg.norm(A @ x)

# Calculate the ratios
ratio = Ax_l2 / np.linalg.norm(x)

# Plot data
ppl.scatter(k_array, ratio, s=np.pi*3, c=(0, 0, 0), alpha=0.5)
ppl.plot(k_array, ratio)
ppl.title('Plot of Questoin 1')
ppl.xlabel('k value')
ppl.ylabel('Ratio of L2 Norms')
ppl.show()