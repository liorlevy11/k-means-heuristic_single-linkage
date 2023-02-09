import numpy as np
import random


def kmeans(X, k, t):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :param t: the number of iterations to run
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    min_value = np.min(X)
    max_value = np.max(X)
    m, d = X.shape
    C_set = [[] for _ in range(k)]
    C = np.zeros((m, 1))

    centroids = np.array([[random.randrange(min_value, max_value) for _ in range(d)] for _ in range(k)])
    for _ in range(t):
        # for i in range(k):
        for index in range(m):
            minIndex = 0
            minDist = 999999999999999999999999999999999999999
            currentX = X[index]
            for j in range(k):
                currentDistance = np.linalg.norm(currentX - centroids[j])
                if currentDistance < minDist:
                    minDist = currentDistance
                    minIndex = j
            x_index = [currentX, [index]]
            C_set[minIndex].append(x_index)

        for indexSet in range(k):
            c_i = C_set[indexSet]
            if len(c_i) == 0:
                continue
            sum_ = np.zeros((d,))
            for x1, _ in c_i:
                sum_ += x1
            centroids[indexSet] = sum_ / len(c_i)

    for indexSet in range(k):
        for _, x_index in C_set[indexSet]:
            C[x_index[0]] = indexSet

    return C


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m alongside its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    m = 1000
    for _ in range(20):
        X, Y_test = gensmallm([data[f'train{i}'] for i in range(10)], [i for i in range(10)], m)
        # run K-means

        c = kmeans(X, k=6, t=60)
        c1 = [int(cluster[0]) for cluster in c]
        one_c(c1, 10, Y_test, m)
        assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
        assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


def one_c(c, k, Y_test, m):
    map_x_to_indexes = create_data_structure(c)
    # list_of_sizes = getClustersSize(map_x_to_indexes, k)
    information_per_cluster = most_common_label_in_cluster(map_x_to_indexes, Y_test, k)
    classificationErrorOnSample = getClassificationErrorOnSample(information_per_cluster, m)

    for index, information in enumerate(information_per_cluster):
        if information == 0:
            continue
        size, real_digit, percentage, _ = information
        print(f"Cluster: {index+1}\tSize: {size}\tMost common label: {real_digit}\tpercentage: {percentage}")
    print(f"The classification error on the sample is: {classificationErrorOnSample}")


def getClassificationErrorOnSample(information_per_cluster, m):
    _sum = 0
    for information in information_per_cluster:
        if information == 0:
            continue
        _sum += information[3]
    return _sum / m


def getClustersSize(map_x_to_indexes, k):
    list_of_sizes = [0 for i in range(k)]
    for digit, indexes in map_x_to_indexes.items():
        list_of_sizes[digit] = len(indexes)

    return list_of_sizes


def most_common_label_in_cluster(map_x_to_indexes: {}, Y_test, k):
    list_of_sizes = [0 for _ in range(k)]

    for key, indexes in map_x_to_indexes.items():
        # key=7:indexes=[1,2,3], key=5:indexes=[4, 5]  -> real_digit=3:count=2, 4:1, real_digit=4:count=1, 2:1
        hash_count = list_to_hash_map(indexes, Y_test)
        # real_digit=3:count=2, 4:1
        real_digit = max(hash_count, key=hash_count.get)
        count = hash_count[real_digit]
        size = len(indexes)
        percentage = count / size
        badClassification = size - count
        common_Label_In_Cluster = (size, real_digit, percentage, badClassification)
        list_of_sizes[key] = common_Label_In_Cluster

    return list_of_sizes


def create_data_structure(c):
    map = {}
    for index, number in enumerate(c):
        if number in map:
            map[number].append(index)
        else:
            map[number] = [index]
    return map


def list_to_hash_map(list_of_indexes, Y_test):
    map = {}
    for i in list_of_indexes:
        real_digit = Y_test[i]
        if real_digit in map:
            map[real_digit] += 1
        else:
            map[real_digit] = 1
    return map


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2

# def assig1c
