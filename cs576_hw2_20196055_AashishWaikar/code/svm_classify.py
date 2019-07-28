import numpy as np
from sklearn import svm


def svm_classify(train_image_feats, train_labels, test_image_feats, kernel_type):
    """
    This function should train a linear SVM for every category (i.e., one vs all)
    and then use the learned linear classifiers to predict the category of every
    test image. Every test feature will be evaluated with all 15 SVMs and the
    most confident SVM will 'win'.

    :param train_image_feats: an N x d matrix, where d is the dimensionality of the feature representation.
    :param train_labels: an N array, where each entry is a string indicating the ground truth category
        for each training image.
    :param test_image_feats: an M x d matrix, where d is the dimensionality of the feature representation.
        You can assume M = N unless you've modified the starter code.
    :param kernel_type: SVM kernel type. 'linear' or 'RBF'

    :return:
        an M array, where each entry is a string indicating the predicted
        category for each test image.
    """

    categories = np.unique(train_labels)
    #pred_labels = np.zeros(test_image_feats.shape[0])
    

    # Your code here. You should also change the return value.

    pred_labels = train_labels
    cfd_matrix = np.zeros((test_image_feats.shape[0], len(categories)))
    i=0

    for categ in categories:
        new_train_labels1 = np.zeros(len(train_labels))
        new_train_labels1 = np.where(train_labels==categ,1,new_train_labels1)
        classifier = svm.SVC(random_state=0,C=0.025,kernel=kernel_type)
        classifier.fit(train_image_feats, new_train_labels1)
        cfd_arr = classifier.decision_function(test_image_feats)
        cfd_matrix[:,i] = cfd_arr
        i+=1

    j=0
    for vec in cfd_matrix:
        index = np.argmax(vec)
        pred_labels[j] = categories[index]
        j+=1


    return pred_labels


    #return np.array([categories[0]] * 1500)