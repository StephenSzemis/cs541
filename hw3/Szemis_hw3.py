import numpy as np
import random as rand
import matplotlib.pyplot as plt

# Author: Stephen Szemis
# Date: November 8, 2020
# I pledge my honor that I have abided by the Steven's Honor system.

# A simple helper for sorting our eignvectors before returning
def get_eigen(S):
    eigenValues, eigenVectors = np.linalg.eig(S)
    idx = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues, eigenVectors)

n = 100
d = 200

X = np.random.randn(n, d)
y = np.transpose(np.random.randn(n))
w_star = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), y))

eigenValues, eigenVectors = get_eigen(np.dot(np.transpose(X), X))

mu0 = 1 / eigenValues[0]

print('mu:', mu0)

def gradient(X, Y, w):
    return np.dot(np.transpose(X), np.dot(X, w)) - np.dot(np.transpose(X), y)

diff = lambda w: np.linalg.norm(w - w_star)

diff_2 = lambda w: 0.5 * (np.linalg.norm(y - np.dot(X, w)) ** 2)

it = 100

def gd(X, Y, w, rate):
    t = 1
    data = [diff(w)]
    while (t < it):
        z = (rate * gradient(X, Y, w))
        w = w - z
        data.append(diff_2(w))
        t += 1
    return data

for i in [0.01, 0.1, 1, 2, 20, 100]:
    data = gd(X, y, np.zeros((d)), i * mu0)

    # Graph data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(range(it), data, s=10, c='b', marker='s')

    # Produce Graph
    ax.set_title("Plot hw3 " + str(i))
    ax.set_xlabel('t')
    ax.set_ylabel('F(wt)')
    plt.show()

# Note that for part 4 versus part 5, we change 
# diff function to diff_2 function.
# We also change labels on graphs, and changed n and d values. 
# Otherwise the code is exactly the same.