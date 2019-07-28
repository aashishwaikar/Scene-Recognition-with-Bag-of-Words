import numpy as np
import math
from distance import pdist

def dist(vec1, vec2):
    sum_sq=0
    for i in range(len(vec1)):
        sum_sq+=(vec1[i]-vec2[i])**2
    return math.sqrt(sum_sq)


def kmeans_clustering(all_features, vocab_size, epsilon, max_iter):
    """
    The function kmeans implements a k-means algorithm that finds the centers of vocab_size clusters
    and groups the all_features around the clusters. As an output, centroids contains a
    center of the each cluster.

    :param all_features: an N x d matrix, where d is the dimensionality of the feature representation.
    :param vocab_size: number of clusters.
    :param epsilon: When the maximum distance between previous and current centroid is less than epsilon,
        stop the iteration.
    :param max_iter: maximum iteration of the k-means algorithm.

    :return: an vocab_size x d array, where each entry is a center of the cluster.
    """

    # Your code here. You should also change the return value.
    #choosing random centroids
    #print(all_features.shape)

    total_pts=len(all_features[:,0])
    rand_arr=np.random.choice(total_pts, vocab_size, replace=False)
    rand_arr.sort()
    vocab_matrix=np.zeros((vocab_size, all_features.shape[1]))
    class_arr=np.zeros(total_pts)

    # #initialised vocab matrix
    ###########
    for i in range(vocab_size):
        vocab_matrix[i]=all_features[rand_arr[i]]

    #print(vocab_matrix)
    
    for r in range(max_iter):
        #assign each pt to a cluster
        dist_mat = pdist(all_features, vocab_matrix)
        c1=0
        for vec in dist_mat:
            index = np.argmin(vec)
            class_arr[c1]=index
            c1+=1

        #finding new centroids
        class_sum=np.zeros((vocab_size,all_features.shape[1]))
        class_count=np.zeros(vocab_size)
        new_vocab=np.zeros((vocab_size,all_features.shape[1]))

        for i in range(total_pts):
            cur_class=class_arr[i]
            #print(int(cur_class))
            class_sum[int(cur_class)]+=all_features[i]
            class_count[int(cur_class)]+=1

        for i in range(vocab_size):
            if class_count[i]==0:
                new_vocab[i]=vocab_matrix[i]
            else:
                new_vocab[i]=class_sum[i]/class_count[i]
        #print(new_vocab)
        dist_mat1 = pdist(vocab_matrix,new_vocab)

        for c3 in range(vocab_size):
            if dist_mat1[c3][c3]>epsilon :
                break

        vocab_matrix=new_vocab
        #print(vocab_matrix)
        if c3==vocab_size-1:
            break

    return vocab_matrix


