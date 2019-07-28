import cv2
import numpy as np
from numpy import linalg
import math
from distance import pdist
from feature_extraction import feature_extraction


def get_spatial_pyramid_feats(image_paths, max_level, feature):
    """
    This function assumes that 'vocab_hog.npy' (for HoG) or 'vocab_sift.npy' (for SIFT)
    exists and contains an N x feature vector length matrix 'vocab' where each row
    is a kmeans centroid or visual word. This matrix is saved to disk rather than passed
    in a parameter to avoid recomputing the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path,
    :param max_level: level of pyramid,
    :param feature: name of image feature representation.

    :return: an N x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each           
        image's histogram ('vocab_size'), multiplies with
        (1 / 3) * (4 ^ (max_level + 1) - 1).
    """
    if feature == 'HoG':
        vocab = np.load('vocab_hog.npy')
    elif feature == 'SIFT':
        vocab = np.load('vocab_sift.npy')

    vocab_size = vocab.shape[0]

    # Your code here. You should also change the return value.
    ft_size = vocab.shape[1]

    output_mat=np.zeros((1,vocab_size))

    # # Your code here. You should also change the return value.
    i=0
    for path in image_paths:
        #dealing with one image
        img = cv2.imread(path)[:, :, ::-1]
        hor = img.shape[1]
        ver = img.shape[0]
        
        f_image_mat=np.zeros((1,1))
        for l in range(max_level+1):
            hstep = math.floor(hor/(2**l))
            vstep = math.floor(ver/(2**l))
            x, y = 0, 0
            for c1 in range(1,2**l + 1):
                x = 0
                for c2 in range(1, 2**l + 1):                
                    features = feature_extraction(img[y:y+vstep, x:x+hstep], feature)                
                    #print("type:",desc is None, "x:",x,"y:",y, "desc_size:",desc is None)
                    distance_mat = pdist(features,vocab)
                    row = np.zeros(vocab_size)
                    for vec in distance_mat:
                        index = np.argmin(vec)
                        row[index]+=1
                    weight = 2**(l-max_level)
                    f_row_mat = np.zeros((1,vocab_size))
                    f_row_mat[0] = weight*row
                    f_image_mat = np.append(f_image_mat, f_row_mat, axis=1)

                    x = x + hstep
                y = y + vstep
        if i==0:
            output_mat = np.append(output_mat, f_image_mat[:,1:], axis=1)
            output_mat = output_mat[:,vocab_size:]
        else:
            output_mat = np.append(output_mat, f_image_mat[:,1:], axis=0)
        i+=1
    #output_mat = output_mat[1:,:]
    output_mat = (1 / 3) * (4 ^ (max_level + 1) - 1) * output_mat

    #print(output_mat)
    return output_mat

    #return np.zeros((1500, 36))


    