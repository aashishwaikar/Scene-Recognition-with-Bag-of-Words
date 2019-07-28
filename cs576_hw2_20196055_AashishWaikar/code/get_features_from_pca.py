import numpy as np


def get_features_from_pca(feat_num, feature):
    """
    This function loads 'vocab_sift.npy' or 'vocab_hog.npg' file and
    returns dimension-reduced vocab into 2D or 3D.

    :param feat_num: 2 when we want 2D plot, 3 when we want 3D plot
    :param feature: 'Hog' or 'SIFT'

    :return: an N x feat_num matrix
    """

    if feature == 'HoG':
        vocab = np.load('vocab_hog.npy')
    elif feature == 'SIFT':
        vocab = np.load('vocab_sift.npy')
        #print(vocab.shape)

    #print(vocab.shape)
    # Your code here. You should also change the return value.

    m=np.mean(vocab.T,axis=1)
    cent_m=vocab-m
    covar=np.cov(cent_m.T)
    val, vec=np.linalg.eig(covar)
    l=val.tolist()
    tup=[l.index(x) for x in sorted(l, reverse=True)[:feat_num]]
    red_vec=np.zeros((vocab.shape[1],feat_num))
    for i in range(feat_num):
        red_vec[:,i]=vec[:,tup[i]]

    P=red_vec.T.dot(cent_m.T)
    return P.T
    #return np.zeros((vocab.shape[0],2))


