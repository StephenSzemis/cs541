import numpy as np
import random as rand
import matplotlib.pyplot as plt
import csv
from numpy.linalg import matrix_rank

# Author: Stephen Szemis
# Date: December 5, 2020
# I pledge my honor that I have abided by the Steven's Honor system.
np.seterr(all='raise')
n = 0
p = 0
mov_index = []
dim = 5

# Part 1
with open('ml-latest-small/ratings.csv', 'r') as temp:
    ratings_csv = csv.DictReader(temp)
    ratings = list(ratings_csv)
    for row in ratings:
        if int(row['userId']) > n:
            n = int(row['userId'])
        if int(row['movieId']) not in mov_index:
            mov_index.append(int(row['movieId']))
    p = len(mov_index)

    # Create M matrix
    M = np.zeros((n,p))
    # Create test
    M_test = np.zeros((n,p))

    for row in ratings:
        if rand.random() < 0.9:
            t = mov_index.index(int(row['movieId']))
            M[int(row['userId']) - 1][t] = float(row['rating'])
        else:
            t = mov_index.index(int(row['movieId']))
            M_test[int(row['userId']) - 1][t] = float(row['rating'])

print('Number of users:', n)
print('Number of movies:', p)

def objective(lbda, U, V):
    omega = np.transpose(np.nonzero(M))
    summation = 0
    for i, j in omega:
        summation += (M[i][j] - np.dot(U[i], np.transpose(V[j]))) ** 2
    summation /= 2
    return summation + ((lbda / 2) * ((np.linalg.norm(U) ** 2) + (np.linalg.norm(V) ** 2)))

def train(lr, lbda, num):
    U = np.random.rand(n, dim)
    V = np.random.rand(p, dim)

    obj = []
    users, items = M.nonzero()
    for epoch in range(num):
        obj.append(objective(lbda, U, V))
        print('Iteration: ' + str(epoch) + ' Objective: ' + str(obj[-1]))
        for i, j in zip(users, items):
            error = M[i, j] - np.dot(U[i], V[j].T)
            U[i] = U[i] + (lr * ((error * V[j]) - (lbda * U[i])))
            V[j] = V[j]  + (lr * ((error * U[i]) - (lbda * V[j])))
    return obj, U, V

def rmse(X):
    omega = np.transpose(np.nonzero(M_test))
    error = 0
    for i, j in omega:
        error += (M_test[i][j] - X[i][j]) ** 2
    error = np.sqrt((1 / len(omega)) * error)
    return error

def evaluate():
    lbda = 1

    # Part 2-3
    objective, U, V = train(0.1, 1, 20)

    # Graph data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(range(len(objective)), objective)#, s=10, c='b', marker='s')

    # Produce Graph
    ax.set_title("Plot hw4 Objective VS Iterations")
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Objective')
    fig.savefig('hw4_part2.png')

    # Part 3-1
    X = np.dot(U, V.T)
    rmse1 = rmse(X)
    print('RMSE of lambda 1:', rmse1)

    lambdas = [0.001, 0.01, 0.1, 0.5, 2, 5, 10, 20, 50, 100, 500, 1000]
    R = []
    for l in lambdas:
        if l < 1:
            _, U, V = train(0.1 * l, l, 20)
        else:
            _, U, V = train(0.2/l, l, 20)
        X = np.dot(U, V.T)
        r = rmse(X)
        print('RMSE of lambda ' + str(l) + ' : ' + str(r))
        R.append(r)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(lambdas, R)#, s=10, c='b', marker='s')

    # Produce Graph
    ax.set_title("Plot hw4 Lambdas VS RMSE")
    ax.set_xlabel('Lambdas')
    ax.set_ylabel('RMSE')
    fig.savefig('hw4_part3.png')

    # print('RMSE of lambda 1:', rmse1)

evaluate()