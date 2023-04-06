import glob
import cv2
import numpy as np
from IrisLocalization import *
from IrisNormalization import *
from ImageEnhancement import *
from FeatureExtraction import *


directory = './data/'

def readSignleImg(file_path):
    img = cv2.imread(filename=file_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def readDataset(train=True,test=False):
    # train_img = [[],[],[]]
    # test_img = [[],[],[],[]]
    train_img = []
    test_img = []
    if train:
        paths = [file for file in glob.glob("./data/*/1/*.bmp")]
        # for indx in range(3):
        #     for path in paths[indx::3]:
        #         train_img[indx].append(readSignleImg(path))
        for path in paths:
            train_img.append(readSignleImg(path))
    
    if test:
        paths = [file for file in glob.glob("./data/*/2/*.bmp")]
        # for indx in range(4):
        #     for path in paths[indx::3]:
        #         test_img[indx].append(readSignleImg(path))
        for path in paths:
            test_img.append(readSignleImg(path))

    return (train_img, test_img) if (train and test) else (train_img if train else test_img)


def rotateImg(img,offset):
    pixels = abs(int(512*offset/360))
    if offset > 0:
        return np.hstack([img[:,pixels:],img[:,:pixels]] )
    else:
        return np.hstack([img[:,(512 - pixels):],img[:,:(512 - pixels)]])



def saveNormalizedImage():
    train = np.load("train.npy")
    norm = []
    for i,img in enumerate(train):
        inner_circle, outer_circle = irisLocalization(img)
        # img = denoising(img)
        img_norm = irisNormalization(img,inner_circle,outer_circle)
        img_enhance = imageEnhancement(img_norm)
        norm.append(img_enhance)
        print(f"process class {int(i/3)}: {i}/{len(train)}")
    np.save("train_norm",norm)


def rotateAll():
    norm = np.load("train_norm.npy")
    offsets = [-9,-6,-3,0,3,6,9]
    res = []
    for i, img in enumerate(norm):
        for offset in offsets:
            p = rotateImg(img,offset)
            res.append(p)
        print(f"process class {int(i/3)}:{i}/{len(norm)}")
    np.save("train_norm_rotate.npy",res)
    
    
def extractFeatureFromRotatedImg():
    imgs = np.load("train_norm_rotate.npy")
    V = []
    for i,img in enumerate(imgs):
        feature_vector = featureExtraction(img)
        V.append([feature_vector])
        print(f"process class {int(i/21)}:{i}/{len(imgs)}")
    np.save("X_train_rotate",V)


# denoising decreased the accuracy 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# def denoisingAll():
#     imgs = np.load("train.npy")
#     denoise = []
#     for i,img in enumerate(imgs):
#         (_, B) = cv2.threshold(img ,180 ,255 ,cv2.THRESH_BINARY)
#         (_, C) = cv2.threshold(img ,100 ,255 ,cv2.THRESH_BINARY)
#         img = img & ~B & C
#         denoise.append(img)
#         print(f"process class {int(i/21)}:{i}/{len(imgs)}")
#     np.save("train_denoise",denoise)
    
# def denoising(img):
#     (_, B) = cv2.threshold(img ,180 ,255 ,cv2.THRESH_BINARY)
#     (_, C) = cv2.threshold(img ,100 ,255 ,cv2.THRESH_BINARY)
#     img = img & ~B & C
#     return img

if __name__=="__main__":
    # train,test = readDataset(True,True)
    # print(len(train),len(test))
    
    saveNormalizedImage()
    rotateAll()
    extractFeatureFromRotatedImg()
    
    # imgs = np.load("X_train_rotate.npy")
    # print(imgs.shape)
    # denoising()
    pass
