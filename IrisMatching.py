from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from ImageEnhancement import *
from IrisNormalization import *
from IrisLocalization import *
from FeatureExtraction import *
from sklearn.decomposition import PCA



offsets = [-9,-6,-3,0,3,6,9]

def fisherLinearDiscriminant(X_train,y_train,X_test,n_components=107):
    clf = LinearDiscriminantAnalysis(n_components=n_components,solver="svd").fit(X_train,y_train)
    X_train_lda = clf.transform(X_train)
    X_test_lda = clf.transform(X_test)
    return X_train_lda, X_test_lda

def nearestCentroid(X_train,y_train,X_test,y_test,metric):
    clf = NearestCentroid(
        metric=metric,shrink_threshold=None
        ).fit(X_train,y_train)
    yhat = clf.predict(X_test)
    crr = CRR(y_test,yhat)
    return crr, clf.centroids_
    # return yhat, clf.centroids_
    
def CRR(y_test,yhat):
    return (y_test==yhat).sum()/len(y_test)


def ROC(results,scores,threshold):
    true_accept = ((results == True) & (scores <= threshold)).sum()
    false_accept = ((results == False) & (scores <= threshold)).sum()
    false_reject = ((results == True) & (scores > threshold)).sum()
    true_reject = ((results == False) & (scores > threshold)).sum()
    
    false_match_rate = false_accept/(false_accept+true_reject)
    false_nonmatch_rate = false_reject/(false_reject+true_accept)
    
    return false_match_rate,false_nonmatch_rate


def preprocess(img,rotate=False,offsets=offsets):
    """rotate the img into 7 different angles

    Args:
        img ([type]): img processed after enhancement
        
    """
    feature_vectors = []
    inner_circle, outer_circle = irisLocalization(img)
    
    if rotate:
        for offset in offsets:
            img_norm = irisNormalization(img,inner_circle,outer_circle,offset)
            img_enhance = imageEnhancement(img_norm)
            feature_vector = featureExtraction(img_enhance)
            feature_vectors.append(feature_vector)
    else:
        img_norm = irisNormalization(img,inner_circle,outer_circle)
        img_enhance = imageEnhancement(img_norm)
        feature_vector = featureExtraction(img_enhance)
        feature_vectors.append(feature_vector)
    
    return feature_vectors

def generateLabels(N=108,rotate=False,offsets=offsets):
    if rotate:
        y_train = np.repeat(range(N),3*len(offsets))
    else:
        y_train = np.repeat(range(N),3)
    
    y_test = np.repeat(range(N),4)
    
    return y_train, y_test
    


def irisMatching(train, test, n_components = 107,rotate=False, dimReduce = False):
    # X_train = []
    # X_test = []
    
    # for i,img in enumerate(train):
    #     X_train_vect = preprocess(img,rotate,offsets)
    #     # print(len(X_train_vect))
    #     X_train.append(X_train_vect)
    #     print(f"process train image {int((i)/3)}: {int((i))}th")
    
    # np.save("X_train",X_train)
    
    # for i,img in enumerate(test):
    #     X_test_vect = preprocess(img,rotate=False,offsets=0)
    #     X_test.append(X_test_vect)
    #     print(f"process test image {int((i)/4)}: {int((i))}th")
    
    # np.save("X_test",X_test)
    
    

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    if rotate:
        # X_train = []
        # X_test = []
        # for i,img in enumerate(train):
        #     X_train_vect = preprocess(img,rotate,offsets)
        #     # print(len(X_train_vect)) this is a seven-vector 
        #     X_train.append(X_train_vect)
        #     print(f"process train image {int((i)/21)}: {int((i))}th")
        
        # np.save("X_train_rotated",X_train)
        
        # for i,img in enumerate(test):
        #     X_test_vect = preprocess(img,rotate=False,offsets=0)
        #     X_test.append(X_test_vect)
        #     print(f"process test image {int((i)/4)}: {int((i))}th")
        
        # np.save("X_test_rotated",X_test)
        
        
        X_train = np.load("X_train_rotate.npy")
        # do not rotate test dataset
        X_test = np.load("X_test.npy")
        
    else:
        X_train = np.load("X_train.npy")
        X_test = np.load("X_test.npy")
    
    
    y_train, y_test = generateLabels(N=n_components+1,rotate=rotate,offsets=offsets)


    
    # dimension reduction
    if dimReduce:
        X_train_pca, X_test_pca = \
            principalComponentsAnalysis(X_train,X_test,y_train=y_train)
        np.save("X_train_pca",X_train_pca)
        np.save("X_test_pca",X_test_pca)
        
        return X_train_pca,y_train, X_test_pca, y_test
    

    # L1 = []
    # L2 = []
    # cosine = []
        
    return X_train,y_train, X_test, y_test


def principalComponentsAnalysis(X_train,X_test,y_train=None):
    nsamples, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples,nx*ny))
    
    nsamples, nx, ny = X_test.shape
    X_test = X_test.reshape((nsamples,nx*ny))
    
    pca = PCA().fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca,X_test_pca



if __name__ == "__main__":
    pass