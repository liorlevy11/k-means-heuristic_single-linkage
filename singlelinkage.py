
import numpy as np
import scipy.io as sio
from kmeans import one_c
from kmeans import gensmallm

def singlelinkage(X, k):
    """
    param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    m, d = X.shape
    return_C = np.zeros((m, 1))
    C = [[[X[i], [i]]] for i in range(len(X))]  # singletons
    while len(C) > k:
        set_A_Index = -1
        set_B_Index = -1
        min_dist = 999999999999999
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                A = C[i]
                B = C[j]
                for a in A:
                    a = a[0]
                    for b in B:
                        b = b[0]
                        currentDist = np.linalg.norm(a - b)
                        if currentDist < min_dist:
                            min_dist = currentDist
                            set_A_Index = i
                            set_B_Index = j
        for i in range(len(C[set_B_Index])):
            C[set_A_Index].append(C[set_B_Index][i])
        C.pop(set_B_Index)
    for indexSet in range(k):
        for c in C[indexSet]:
            x_index = c[1][0]
            return_C[x_index] = indexSet
    # let me in
    return return_C


#  [x1,1], [x2,2] ->
def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    # selected_digits = [data[f'train{i}'][np.random.choice(data[f'train{i}'].shape[0], 20, replace=False)]
    #                    for i in range(10)]
    #
    # # concatenate the selected digits
    # X = np.concatenate(selected_digits)
    # m, d = X.shape
    # # run single-linkage
    # c1 = singlelinkage(X, k=10)
    # selected_digits = [data[f'train{i}'][np.random.choice(data[f'train{i}'].shape[0], 20, replace=False)]
    #                    for i in range(10)]
    #
    # # concatenate the selected digits
    # X = np.concatenate(selected_digits)
    # c2 = singlelinkage(X, k=10)
    # print(c1)
    # print("-----------")
    # print(c2)
    # assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    # assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"
    m = 300
    for _ in range(20):
        X, Y_test = gensmallm([data[f'train{i}'] for i in range(10)], [i for i in range(10)], m)
        # run K-means
        c = singlelinkage(X, k=6)
        c1 = [int(cluster[0]) for cluster in c]
        one_c(c1, 6, Y_test, m)


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2
