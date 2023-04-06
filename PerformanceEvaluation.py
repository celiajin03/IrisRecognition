from cv2 import threshold
from IrisMatching import *
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
    
def identification(X_train, y_train, X_test, y_test):
    # In identification mode, the algorithm is
    # measured by Correct Recognition Rate (CRR), the ratio of
    # the number of samples being correctly classified to the total
    # number of test samples. 
    if len(X_train.shape) != 2:
        nsamples, nx, ny = X_train.shape
        X_train = X_train.reshape((nsamples,nx*ny))
    if len(X_test.shape) != 2:
        nsamples, nx, ny = X_test.shape
        X_test = X_test.reshape((nsamples,nx*ny))
    
    # orginal feature vectors 
    l1,_ = nearestCentroid(X_train, y_train, X_test, y_test, metric="l1")
    l2,_ = nearestCentroid(X_train, y_train, X_test, y_test, metric="l2")
    cosine,_ = nearestCentroid(X_train, y_train, X_test, y_test, metric="cosine")
    print(l1,l2,cosine)
    
    # dimension reduction
    X_train_lda, X_test_lda = \
            fisherLinearDiscriminant(X_train,y_train,X_test,n_components=107)
            
    l1_lda,_ = nearestCentroid(X_train_lda, y_train, X_test_lda, y_test, metric="l1")
    l2_lda,_ = nearestCentroid(X_train_lda, y_train, X_test_lda, y_test, metric="l2")
    cosine_lda,_ = nearestCentroid(X_train_lda, y_train, X_test_lda, y_test, metric="cosine")
    print(l1_lda,l2_lda,cosine_lda)
    
    crr = []
    dims = np.arange(1,108,10)
   
    for dim in dims:
        X_train_lda, X_test_lda = \
                    fisherLinearDiscriminant(X_train,y_train,X_test,n_components=dim)
        
        cosine,_ = nearestCentroid(X_train_lda, y_train, X_test_lda, y_test, metric="cosine")
        crr.append(cosine)
        
    plt.plot(dims,crr,"-*")    
    plt.xlabel("Dimensionality of the feature vector")
    plt.ylabel("Correct recognition rate")
    plt.title("Recognition results using features of different dimensionality")
    plt.show()

    return X_train_lda, X_test_lda




def verification(X_train, y_train, X_test, y_test):
    # In verification mode, the Receiver
    # Operating Characteristic (ROC) curve is used to report the
    # performance of the proposed method. 
    _,centroids= nearestCentroid(X_train, y_train, X_test, y_test,metric="cosine")
    thresholds = np.linspace(0.1,0.8,20)
    n_bootstrap = 1000
    scores = pairwise_distances(X_test,centroids,metric="cosine")
    results = np.zeros_like(scores,dtype=np.bool8)
    results[range(len(results)),y_test] = True
    
    scores = scores.reshape((-1, 4, 108))
    results = results.reshape((-1, 4, 108))
    
    fmrs_org = np.zeros(20)
    fnmrs_org = np.zeros(20)
    for j,threshold in enumerate(thresholds):
        fmrs_org[j], fnmrs_org[j] = map(lambda x:x*100, ROC(results,scores,threshold))

    # bootstrap
    # create masks to randomly select labels and 
    labels = np.random.choice(108, size=108 * n_bootstrap, replace=True)
    samples = np.random.randint(4, size=108 * n_bootstrap)
    # select 108 results 1000 times with replacement
    scores_bootstrap = scores[labels, samples].reshape((n_bootstrap, -1, 108))
    results_bootstrap = results[labels, samples].reshape((n_bootstrap, -1, 108))

    fmrs = np.zeros((n_bootstrap,20))
    fnmrs = np.zeros((n_bootstrap,20))
    
    # calculate fmr and fnmr for each bootstrap sample with the threshold interval
    for i,(result, score) in enumerate(zip(results_bootstrap,scores_bootstrap)):
        for j,threshold in enumerate(thresholds):
            fmrs[i,j], fnmrs[i,j] = ROC(result,score,threshold)

        
    mean_fmrs = np.mean(fmrs,axis=0)*100
    mean_fnmrs = np.mean(fnmrs,axis=0)*100
    
    
    ci_fmrs = np.percentile(fmrs,[2.5,97.5],axis=0).T*100
    ci_fnmrs = np.percentile(fnmrs,[2.5,97.5],axis=0).T*100
    
    _, axs = plt.subplots(1,2)
    
    for ax in axs:
        ax.set_xticks([1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2])
        ax.set_xscale("log")
        ax.set_xlabel('False match rate (%)')
        ax.set_ylabel('False non-match rate (%)')
        ax.plot(fmrs_org, fnmrs_org, linestyle='--')
    axs[0].plot(ci_fmrs, mean_fnmrs)
    axs[0].set_title('FMR CI')
    axs[1].plot(mean_fmrs, ci_fnmrs)
    axs[1].set_title('FNMR CI')
    plt.show()




def performanceEvaluation(X_train, y_train, X_test, y_test):
    X_train_lda, X_test_lda = identification(X_train, y_train, X_test, y_test)
    verification(X_train_lda, y_train, X_test_lda, y_test)
    


