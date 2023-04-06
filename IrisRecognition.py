from IrisLocalization import *
from IrisNormalization import *
from IrisMatching import *
from ImageEnhancement import *
from FeatureExtraction import *
from PerformanceEvaluation import *
from utils import *

import random


def test():
    subject = random.randint(0,108*3-1)
    print(subject)
    # train,test= readDataset(train=True,test=True)   
    # np.save("train",train)
    # np.save("test",test)
    
    
    train = np.load("train_denoise.npy")
    # train = np.load("train.npy")
    # test = np.load("test.npy")
    
    # img = train[subject]
    img = train[282]
    # 81 55 10 94 71 55*3-1
    # print(img.shape)
    inner_circle, outer_circle = irisLocalization(img)
    img_norm = irisNormalization(img,inner_circle,outer_circle)
    img_enhance = imageEnhancement(img_norm)
    plt.imshow(img_enhance,cmap="gray")
    plt.show()
    
    feature_vect = featureExtraction(img_enhance)
    # print((feature_vect))
    
    


if __name__=="__main__":
    # test()
    import warnings
    warnings.filterwarnings("ignore")
    # train = np.load("train.npy")
    # test = np.load("test.npy")
    # X_train,y_train, X_test, y_test= \
    #     irisMatching(train, test, n_components = 107,rotate=False, dimReduce = False)
    
    # X_train = np.load("X_train.npy")
    # X_test = np.load("X_test.npy")
    
    # X_train,y_train, X_test, y_test = irisMatching(train=train, test=None,rotate=True)
    # performanceEvaluation(X_train, y_train, X_test, y_test)
    # 0.7592592592592593 0.7152777777777778 0.7013888888888888

    X_train,y_train, X_test, y_test = irisMatching(train=None, test=None,rotate=False)
    performanceEvaluation(X_train, y_train, X_test, y_test)
    # 0.8125 0.7708333333333334 0.7662037037037037
    # 0.7870370370370371 0.8009259259259259 0.8541666666666666

    # X_train,y_train, X_test, y_test = irisMatching(train=None, test=None,rotate=False,dimReduce=True)
    # performanceEvaluation(X_train, y_train, X_test, y_test)
    # 0.8587962962962963 0.7708333333333334 0.7986111111111112
    # 0.006944444444444444 0.011574074074074073 0.004629629629629629
    
    
    
    
    
    



